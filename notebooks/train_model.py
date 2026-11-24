# Databricks notebook source
# MAGIC %md
# MAGIC # Model Development
# MAGIC **Project:** Credit card default prediction
# MAGIC
# MAGIC **What this notebook does**
# MAGIC 1. Loads `gold_features` and builds train / validation / test sets (test is never touched until the end)
# MAGIC 2. Trains a **baseline logistic regression** (tracked with MLflow autologging)
# MAGIC 3. Tunes a **LightGBM** model with a random search, one nested MLflow run per trial
# MAGIC 4. Picks a **decision threshold from a business cost matrix** (a missed defaulter costs more than a false alarm)
# MAGIC 5. Evaluates on the test set: AUC, PR-AUC, KS, Brier score, confusion matrix
# MAGIC 6. Explains the model with **SHAP**
# MAGIC 7. Logs the model with a signature and registers it in the **Unity Catalog Model Registry**
# MAGIC
# MAGIC **Fairness note:** `sex` and `age_band` are excluded from the model inputs. They are used only for the audit in `monitoring_drift_fairness`.

# COMMAND ----------

# MAGIC %pip install lightgbm shap --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

import json
import tempfile
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.pyfunc
import lightgbm as lgb
import sklearn
import shap
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve,
                             brier_score_loss, confusion_matrix,
                             precision_score, recall_score, ConfusionMatrixDisplay)

CATALOG = "bank_mlops"
SCHEMA = "credit"
GOLD = f"{CATALOG}.{SCHEMA}.gold_features"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.default_model"

TARGET = "default_next_month"
SEED = 42

# Business cost matrix (relative units): missing a defaulter is 5x worse than a false alarm
COST_FN = 5.0
COST_FP = 1.0

N_TRIALS = 10

# Columns that must NOT be model inputs
EXCLUDE = ["client_id", "split", TARGET, "sex", "age_band"]

user = spark.sql("SELECT current_user()").collect()[0][0]
EXPERIMENT = f"/Users/{user}/credit_default"

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT)
print(f"Experiment: {EXPERIMENT}\nModel: {MODEL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load data and split
# MAGIC The `split` column was created in `data_prep` (stratified, seed 42). We carve a validation set out of the training rows for early stopping and threshold selection.

# COMMAND ----------

pdf = spark.table(GOLD).toPandas()

FEATURES = [c for c in pdf.columns if c not in EXCLUDE]
print(f"{len(FEATURES)} features: {FEATURES}")

train_all = pdf[pdf["split"] == "train"]
test = pdf[pdf["split"] == "test"]

X_trainval = train_all[FEATURES].astype("float64")
y_trainval = train_all[TARGET].astype(int)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.2, stratify=y_trainval, random_state=SEED)

X_test = test[FEATURES].astype("float64")
y_test = test[TARGET].astype(int)

print(f"train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}")
print(f"default rate  train={y_train.mean():.3f}  val={y_val.mean():.3f}  test={y_test.mean():.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Helper functions

# COMMAND ----------

def ks_statistic(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))

def expected_cost(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return float(fn * COST_FN + fp * COST_FP)

def best_threshold(y_true, y_score):
    grid = np.arange(0.05, 0.95, 0.01)
    costs = [expected_cost(y_true, (y_score >= t).astype(int)) for t in grid]
    return float(grid[int(np.argmin(costs))])

def score_metrics(y_true, y_score, prefix):
    return {
        f"{prefix}_auc": float(roc_auc_score(y_true, y_score)),
        f"{prefix}_pr_auc": float(average_precision_score(y_true, y_score)),
        f"{prefix}_ks": ks_statistic(y_true, y_score),
        f"{prefix}_brier": float(brier_score_loss(y_true, y_score)),
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## Baseline: logistic regression
# MAGIC A simple, interpretable benchmark. If the boosted model cannot beat this clearly, it is not worth the complexity.

# COMMAND ----------

mlflow.sklearn.autolog(log_models=False, silent=True)

with mlflow.start_run(run_name="logreg_baseline") as base_run:
    baseline = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
    ])
    baseline.fit(X_train, y_train)

    val_scores = baseline.predict_proba(X_val)[:, 1]
    test_scores = baseline.predict_proba(X_test)[:, 1]

    mlflow.set_tag("model_type", "logistic_regression")
    mlflow.log_metrics(score_metrics(y_val, val_scores, "val"))
    baseline_test = score_metrics(y_test, test_scores, "test")
    mlflow.log_metrics(baseline_test)

mlflow.autolog(disable=True)
print({k: round(v, 4) for k, v in baseline_test.items()})

# COMMAND ----------

# MAGIC %md
# MAGIC ## LightGBM: random hyperparameter search
# MAGIC Each trial is a **nested MLflow run** under one parent run. Trials are scored on the validation set only. Early stopping picks the number of trees.

# COMMAND ----------

rng = np.random.default_rng(SEED)

def sample_params():
    return {
        "num_leaves": int(rng.choice([15, 31, 63])),
        "learning_rate": float(np.exp(rng.uniform(np.log(0.02), np.log(0.15)))),
        "min_child_samples": int(rng.choice([20, 50, 100])),
        "subsample": float(rng.uniform(0.6, 1.0)),
        "colsample_bytree": float(rng.uniform(0.6, 1.0)),
        "reg_lambda": float(rng.uniform(0.0, 5.0)),
    }

def fit_lgbm(params):
    model = lgb.LGBMClassifier(
        n_estimators=1000, subsample_freq=1, random_state=SEED, n_jobs=-1, verbose=-1, **params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)], eval_metric="auc",
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    return model

with mlflow.start_run(run_name="lgbm_tuning") as parent_run:
    mlflow.set_tag("model_type", "lightgbm")
    trials = []
    for i in range(N_TRIALS):
        params = sample_params()
        with mlflow.start_run(run_name=f"trial_{i}", nested=True):
            m = fit_lgbm(params)
            val_auc = roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])
            mlflow.log_params(params)
            mlflow.log_metric("val_auc", val_auc)
            mlflow.log_metric("best_iteration", int(m.best_iteration_))
        trials.append((val_auc, params))
        print(f"trial {i}: val_auc={val_auc:.4f}  {params}")

    best_val_auc, best_params = max(trials, key=lambda t: t[0])
    mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
    mlflow.log_metric("best_val_auc", best_val_auc)

print(f"\nBest validation AUC: {best_val_auc:.4f}")
print(best_params)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Final model, business threshold and test evaluation
# MAGIC The threshold is chosen on the **validation** set by minimising `5 x missed defaulters + 1 x false alarms`. The test set is used once, to report honest performance.

# COMMAND ----------

with mlflow.start_run(run_name="lgbm_final") as run:
    mlflow.set_tag("model_type", "lightgbm")
    mlflow.set_tag("stage", "candidate")

    model = fit_lgbm(best_params)
    val_scores = model.predict_proba(X_val)[:, 1]
    test_scores = model.predict_proba(X_test)[:, 1]

    threshold = best_threshold(y_val, val_scores)
    test_pred = (test_scores >= threshold).astype(int)

    metrics = score_metrics(y_test, test_scores, "test")
    metrics.update(score_metrics(y_val, val_scores, "val"))
    metrics.update({
        "threshold": threshold,
        "test_precision": float(precision_score(y_test, test_pred)),
        "test_recall": float(recall_score(y_test, test_pred)),
        "test_expected_cost": expected_cost(y_test, test_pred),
        "test_cost_at_default_0_5": expected_cost(y_test, (test_scores >= 0.5).astype(int)),
        "baseline_test_auc": baseline_test["test_auc"],
    })

    mlflow.log_params({**best_params, "n_estimators_used": int(model.best_iteration_),
                       "cost_fn": COST_FN, "cost_fp": COST_FP,
                       "features": json.dumps(FEATURES)})
    mlflow.log_metrics(metrics)

    # Confusion matrix at the business threshold
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(confusion_matrix(y_test, test_pred),
                           display_labels=["No default", "Default"]).plot(ax=ax, colorbar=False)
    ax.set_title(f"Test confusion matrix (threshold={threshold:.2f})")
    mlflow.log_figure(fig, "confusion_matrix.png")
    plt.show()

    # SHAP explainability on a test sample
    sample = X_test.sample(n=min(2000, len(X_test)), random_state=SEED)
    sv = shap.TreeExplainer(model).shap_values(sample)
    if isinstance(sv, list):
        sv = sv[1]
    elif getattr(sv, "ndim", 2) == 3:
        sv = sv[:, :, 1]
    plt.figure()
    shap.summary_plot(sv, sample, show=False)
    mlflow.log_figure(plt.gcf(), "shap_summary.png")
    plt.show()

    run_id = run.info.run_id

print({k: round(v, 4) for k, v in metrics.items()})

# COMMAND ----------

# MAGIC %md
# MAGIC ### Value of the business threshold
# MAGIC Compare the expected cost at the default 0.5 cutoff with the cost-optimised cutoff.

# COMMAND ----------

pd.DataFrame({
    "threshold": [0.5, threshold],
    "expected_cost": [metrics["test_cost_at_default_0_5"], metrics["test_expected_cost"]],
    "recall": [recall_score(y_test, (test_scores >= 0.5).astype(int)), metrics["test_recall"]],
    "precision": [precision_score(y_test, (test_scores >= 0.5).astype(int)), metrics["test_precision"]],
}, index=["default 0.5", "cost-optimised"]).round(4)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log the model and register it in Unity Catalog
# MAGIC The model is wrapped as a small `pyfunc` that returns **probability of default**. This makes batch scoring (`spark_udf`) and Model Serving return the same simple output.

# COMMAND ----------

class DefaultProbModel(mlflow.pyfunc.PythonModel):
    """Returns the probability of default for each row."""
    def load_context(self, context):
        import joblib
        self.model = joblib.load(context.artifacts["model_path"])

    def predict(self, context, model_input, params=None):
        return self.model.predict_proba(model_input)[:, 1]

model_file = f"{tempfile.mkdtemp()}/lgbm_model.joblib"
joblib.dump(model, model_file)

signature = infer_signature(X_train.head(100), model.predict_proba(X_train.head(100))[:, 1])

with mlflow.start_run(run_id=run_id):
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=DefaultProbModel(),
        artifacts={"model_path": model_file},
        signature=signature,
        input_example=X_train.head(5),
        pip_requirements=[f"lightgbm=={lgb.__version__}",
                          f"scikit-learn=={sklearn.__version__}",
                          f"joblib=={joblib.__version__}",
                          "pandas", "numpy"],
        registered_model_name=MODEL_NAME,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set aliases
# MAGIC * `challenger` always points to the newest version
# MAGIC * `champion` is set automatically only for the very first version. After that, promotion is decided by the validation gate in `validate_and_promote`.

# COMMAND ----------

client = MlflowClient()
versions = [v for v in client.search_model_versions(f"name='{MODEL_NAME}'") if v.run_id == run_id]
new_version = max(versions, key=lambda v: int(v.version)).version

client.set_registered_model_alias(MODEL_NAME, "challenger", new_version)
client.set_model_version_tag(MODEL_NAME, new_version, "threshold", str(threshold))
client.set_model_version_tag(MODEL_NAME, new_version, "test_auc", f"{metrics['test_auc']:.4f}")

try:
    client.get_model_version_by_alias(MODEL_NAME, "champion")
    print(f"Champion already exists. Version {new_version} is the challenger.")
except Exception:
    client.set_registered_model_alias(MODEL_NAME, "champion", new_version)
    print(f"No champion yet. Version {new_version} is now the champion.")

print(f"Registered {MODEL_NAME} version {new_version}  |  test AUC {metrics['test_auc']:.4f}  |  threshold {threshold:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results summary

# COMMAND ----------

pd.DataFrame({
    "model": ["Logistic regression (baseline)", "LightGBM (tuned)"],
    "test_auc": [baseline_test["test_auc"], metrics["test_auc"]],
    "test_pr_auc": [baseline_test["test_pr_auc"], metrics["test_pr_auc"]],
    "test_ks": [baseline_test["test_ks"], metrics["test_ks"]],
}).round(4)

# COMMAND ----------

# MAGIC %md
# MAGIC **Next:** `batch_and_serving`, which scores all clients in batch with `spark_udf` and deploys a real-time endpoint (or the local fallback on Free Edition).
