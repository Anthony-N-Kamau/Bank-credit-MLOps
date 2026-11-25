# Databricks notebook source
# MAGIC %md
# MAGIC # Advanced MLOps: Drift Monitoring, Retraining Trigger and Fairness Audit
# MAGIC **Project:** Credit card default prediction
# MAGIC
# MAGIC **What this notebook does**
# MAGIC 1. **Simulates production drift** (stable, moderate and severe economic stress) by changing raw customer behaviour and recomputing the engineered features
# MAGIC 2. Computes **PSI** and **KS** drift metrics per feature and for the model score (these work without labels)
# MAGIC 3. Tracks **model performance** per scenario against the champion's baseline
# MAGIC 4. Applies a **retraining trigger** rule and can optionally start the pipeline Job
# MAGIC 5. Audits **fairness** by sex and age band
# MAGIC 6. Shows **lineage** from raw data to model, and optionally creates a Lakehouse Monitor
# MAGIC
# MAGIC **Output tables** (`bank_mlops.credit`): `drift_metrics`, `monitoring_metrics`, `retraining_triggers`, `fairness_audit`
# MAGIC
# MAGIC **PSI guide:** below 0.10 stable, 0.10 to 0.25 moderate shift, above 0.25 significant shift.

# COMMAND ----------

# MAGIC %pip install lightgbm --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

import numpy as np
import pandas as pd
import mlflow
import mlflow.pyfunc
from mlflow import MlflowClient
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score, average_precision_score
from pyspark.sql import functions as F

dbutils.widgets.dropdown("trigger_retrain", "false", ["false", "true"])
dbutils.widgets.dropdown("create_lakehouse_monitor", "false", ["false", "true"])
TRIGGER_RETRAIN = dbutils.widgets.get("trigger_retrain") == "true"
CREATE_MONITOR = dbutils.widgets.get("create_lakehouse_monitor") == "true"

CATALOG = "bank_mlops"
SCHEMA = "credit"
GOLD = f"{CATALOG}.{SCHEMA}.gold_features"
PREDICTIONS = f"{CATALOG}.{SCHEMA}.batch_predictions"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.default_model"
MODEL_URI = f"models:/{MODEL_NAME}@champion"
TARGET = "default_next_month"
SEED = 42

# Retraining rule thresholds
PSI_SIGNIFICANT = 0.25
MIN_FEATURES_DRIFTED = 3
MAX_AUC_DROP = 0.03

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()
user = spark.sql("SELECT current_user()").collect()[0][0]

champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
THRESHOLD = float(champion.tags.get("threshold", 0.5))
BASELINE_AUC = float(client.get_run(champion.run_id).data.metrics["test_auc"])

loaded = mlflow.pyfunc.load_model(MODEL_URI)
FEATURES = loaded.metadata.get_input_schema().input_names()

print(f"Champion version {champion.version} | baseline test AUC {BASELINE_AUC:.4f} | threshold {THRESHOLD:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reference and production data
# MAGIC * **Reference:** the training split, which is what the model learned from
# MAGIC * **"Production":** the test split, which we then distort to simulate drift

# COMMAND ----------

gold_pdf = spark.table(GOLD).toPandas()
reference = gold_pdf[gold_pdf["split"] == "train"].reset_index(drop=True)
production_base = gold_pdf[gold_pdf["split"] == "test"].reset_index(drop=True)

def score(df):
    return np.asarray(loaded.predict(df[FEATURES].astype("float64")), dtype=float)

reference_scores = score(reference)
print(f"reference={len(reference):,} rows  production base={len(production_base):,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Recompute engineered features
# MAGIC When raw behaviour changes, the engineered features must change consistently. This pandas function mirrors the logic in `data_prep`, and the assertion proves it reproduces the stored values.

# COMMAND ----------

PAY = [f"pay_{i}" for i in range(1, 7)]
BILL = [f"bill_amt{i}" for i in range(1, 7)]
PAYAMT = [f"pay_amt{i}" for i in range(1, 7)]

def engineer(df):
    out = df.copy()
    limit = out["limit_bal"].clip(lower=1.0)
    out["avg_bill"] = out[BILL].mean(axis=1)
    out["avg_pay_amt"] = out[PAYAMT].mean(axis=1)
    out["util_ratio"] = out["bill_amt1"] / limit
    out["avg_util_ratio"] = out["avg_bill"] / limit
    out["pay_to_bill_ratio"] = out[PAYAMT].sum(axis=1) / out[BILL].clip(lower=0).sum(axis=1).clip(lower=1.0)
    out["avg_delay"] = out[PAY].mean(axis=1)
    out["max_delay"] = out[PAY].max(axis=1)
    out["months_delayed"] = (out[PAY] > 0).sum(axis=1)
    out["bill_trend"] = (out["bill_amt1"] - out["bill_amt6"]) / limit
    return out

engineered_cols = ["avg_bill", "avg_pay_amt", "util_ratio", "avg_util_ratio", "pay_to_bill_ratio",
                   "avg_delay", "max_delay", "months_delayed", "bill_trend"]
recomputed = engineer(production_base)
for c in engineered_cols:
    assert np.allclose(recomputed[c].astype(float), production_base[c].astype(float), atol=1e-6), f"Mismatch in {c}"
print("Feature recomputation matches the stored gold features")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Simulate drift scenarios
# MAGIC | Scenario | What changes |
# MAGIC |---|---|
# MAGIC | `stable` | Nothing. A control, to confirm the monitors stay quiet |
# MAGIC | `moderate` | 20% of clients slip one month later on recent payments; bills +10%; payments -10%; limits -5% |
# MAGIC | `severe` | 40% slip one to two months later; bills +30%; payments -30%; limits -15% |
# MAGIC
# MAGIC Labels are left unchanged, so performance metrics only show the effect of **input** drift. Drift metrics like PSI do not need labels, which is exactly why they are used in production where labels arrive weeks later.

# COMMAND ----------

def apply_drift(df, share_late, extra_delay, bill_mult, pay_mult, limit_mult, seed):
    rng = np.random.default_rng(seed)
    out = df.copy()
    late = rng.random(len(out)) < share_late
    for i, col in enumerate(PAY):
        bump = extra_delay if i < 3 else max(extra_delay - 1, 0)   # recent months hit hardest
        out[col] = np.clip(out[col].astype(float) + late * bump, -2, 8)
    out[BILL] = out[BILL].astype(float) * bill_mult
    out[PAYAMT] = out[PAYAMT].astype(float) * pay_mult
    out["limit_bal"] = out["limit_bal"].astype(float) * limit_mult
    return engineer(out)

scenarios = {
    "stable": production_base.copy(),
    "moderate": apply_drift(production_base, 0.20, 1, 1.10, 0.90, 0.95, SEED),
    "severe": apply_drift(production_base, 0.40, 2, 1.30, 0.70, 0.85, SEED + 1),
}
scenario_scores = {name: score(df) for name, df in scenarios.items()}
print({name: round(float(s.mean()), 4) for name, s in scenario_scores.items()}, "<- mean predicted default probability")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Drift metrics: PSI and KS per feature and for the model score

# COMMAND ----------

def psi(reference_values, current_values, bins=10):
    ref = np.asarray(reference_values, dtype=float)
    cur = np.asarray(current_values, dtype=float)
    if len(np.unique(ref)) <= bins:                      # discrete feature: one bin per value
        cats = np.union1d(np.unique(ref), np.unique(cur))
        r = np.array([(ref == c).mean() for c in cats])
        c_ = np.array([(cur == c).mean() for c in cats])
    else:                                                # continuous: reference deciles
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        r = np.histogram(ref, bins=edges)[0] / len(ref)
        c_ = np.histogram(cur, bins=edges)[0] / len(cur)
    r, c_ = np.clip(r, 1e-4, None), np.clip(c_, 1e-4, None)
    return float(np.sum((c_ - r) * np.log(c_ / r)))

def psi_status(value):
    return "significant" if value > PSI_SIGNIFICANT else ("moderate" if value > 0.10 else "stable")

rows = []
for name, df in scenarios.items():
    for feat in FEATURES:
        p = psi(reference[feat], df[feat])
        ks = ks_2samp(reference[feat].astype(float), df[feat].astype(float))
        rows.append((name, feat, p, float(ks.statistic), float(ks.pvalue), psi_status(p)))
    p = psi(reference_scores, scenario_scores[name])
    ks = ks_2samp(reference_scores, scenario_scores[name])
    rows.append((name, "model_score", p, float(ks.statistic), float(ks.pvalue), psi_status(p)))

drift_pdf = pd.DataFrame(rows, columns=["scenario", "feature", "psi", "ks_statistic", "ks_pvalue", "status"])

(spark.createDataFrame(drift_pdf)
 .withColumn("computed_at", F.current_timestamp())
 .write.mode("overwrite").option("overwriteSchema", "true")
 .saveAsTable(f"{CATALOG}.{SCHEMA}.drift_metrics"))

summary = (drift_pdf.groupby("scenario")
           .agg(features_significant=("status", lambda s: int((s == "significant").sum())),
                features_moderate=("status", lambda s: int((s == "moderate").sum())),
                max_psi=("psi", "max")).round(3))
display(summary.reset_index())

# COMMAND ----------

# Top drifting features in the severe scenario
display(drift_pdf[drift_pdf["scenario"] == "severe"].sort_values("psi", ascending=False).head(10).round(4))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model performance per scenario
# MAGIC The champion's test AUC is the baseline. Compare how the share of clients flagged for review moves, because that changes business outcomes even when ranking quality (AUC) holds up.

# COMMAND ----------

perf_rows = []
for name, df in scenarios.items():
    s = scenario_scores[name]
    y = df[TARGET].astype(int).values
    perf_rows.append((name, len(df), float(roc_auc_score(y, s)), float(average_precision_score(y, s)),
                      float(s.mean()), float((s >= THRESHOLD).mean()), float(y.mean()),
                      BASELINE_AUC, BASELINE_AUC - float(roc_auc_score(y, s))))

perf_pdf = pd.DataFrame(perf_rows, columns=["scenario", "rows", "auc", "pr_auc", "mean_score",
                                            "flagged_rate", "actual_default_rate", "baseline_auc", "auc_drop"])

(spark.createDataFrame(perf_pdf)
 .withColumn("computed_at", F.current_timestamp())
 .write.mode("overwrite").option("overwriteSchema", "true")
 .saveAsTable(f"{CATALOG}.{SCHEMA}.monitoring_metrics"))

display(perf_pdf.round(4))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Retraining trigger
# MAGIC Retraining is recommended when **any** of these holds:
# MAGIC * at least `MIN_FEATURES_DRIFTED` features have PSI above 0.25
# MAGIC * the model score itself has PSI above 0.25
# MAGIC * AUC has dropped by more than `MAX_AUC_DROP` from the baseline

# COMMAND ----------

trigger_rows = []
for name in scenarios:
    d = drift_pdf[drift_pdf["scenario"] == name]
    n_high = int(((d["feature"] != "model_score") & (d["psi"] > PSI_SIGNIFICANT)).sum())
    score_psi = float(d.loc[d["feature"] == "model_score", "psi"].iloc[0])
    auc_drop = float(perf_pdf.loc[perf_pdf["scenario"] == name, "auc_drop"].iloc[0])

    reasons = []
    if n_high >= MIN_FEATURES_DRIFTED:
        reasons.append(f"{n_high} features with PSI > {PSI_SIGNIFICANT}")
    if score_psi > PSI_SIGNIFICANT:
        reasons.append(f"score PSI {score_psi:.2f}")
    if auc_drop > MAX_AUC_DROP:
        reasons.append(f"AUC drop {auc_drop:.3f}")

    trigger_rows.append((name, n_high, score_psi, auc_drop, bool(reasons), "; ".join(reasons) or "no trigger"))

trig_pdf = pd.DataFrame(trigger_rows, columns=["scenario", "features_psi_high", "score_psi",
                                               "auc_drop", "retrain_recommended", "reasons"])

(spark.createDataFrame(trig_pdf)
 .withColumn("computed_at", F.current_timestamp())
 .write.mode("append").option("mergeSchema", "true")
 .saveAsTable(f"{CATALOG}.{SCHEMA}.retraining_triggers"))

display(trig_pdf.round(4))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Optionally start the pipeline Job
# MAGIC Set the `trigger_retrain` widget to `true` to run the pipeline automatically when the **severe** scenario triggers. The pipeline retrains, the validation gate decides on promotion, and batch scoring refreshes. It is off by default so it never starts a run by surprise.

# COMMAND ----------

severe_flag = bool(trig_pdf.loc[trig_pdf["scenario"] == "severe", "retrain_recommended"].iloc[0])

if severe_flag and TRIGGER_RETRAIN:
    try:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        jobs = [j for j in w.jobs.list() if j.settings and "credit-default-pipeline" in (j.settings.name or "")]
        if jobs:
            run = w.jobs.run_now(job_id=jobs[0].job_id)
            print(f"Started pipeline job '{jobs[0].settings.name}' (run id {run.run_id})")
        else:
            print("Retraining recommended, but the Job was not found. Run `databricks bundle deploy` first.")
    except Exception as e:
        print(f"Could not start the Job: {str(e)[:300]}")
elif severe_flag:
    print("Retraining recommended for the severe scenario. Set trigger_retrain=true to start the Job automatically.")
else:
    print("No retraining needed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Alert on the trigger table (set up once in the UI)
# MAGIC In the **SQL editor**, save this query, then go to **Alerts**, create an alert on it, set the condition to "row count above 0" and add yourself as the recipient:
# MAGIC ```sql
# MAGIC SELECT scenario, reasons, computed_at
# MAGIC FROM bank_mlops.credit.retraining_triggers
# MAGIC WHERE retrain_recommended = true
# MAGIC   AND computed_at > current_timestamp() - INTERVAL 1 DAY
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fairness audit by sex and age band
# MAGIC `sex` and `age_band` were **not** model inputs, but a model can still treat groups differently through correlated features, and `age` is a model input. We score the **test split** with the champion and compare outcomes per group at the business threshold.
# MAGIC
# MAGIC * **Flag rate:** share of clients predicted to default (these would be declined or reviewed)
# MAGIC * **Approval rate:** 1 minus the flag rate
# MAGIC * **TPR (recall):** share of real defaulters caught. **FPR:** share of good clients wrongly flagged
# MAGIC * **Disparate impact ratio:** a group's approval rate divided by the highest group's. Below 0.80 is the common "four-fifths" warning level

# COMMAND ----------

test_pdf = gold_pdf[gold_pdf["split"] == "test"].reset_index(drop=True)
test_pdf["score"] = score(test_pdf)
test_pdf["flagged"] = (test_pdf["score"] >= THRESHOLD).astype(int)
test_pdf["sex_label"] = test_pdf["sex"].map({1: "male", 2: "female"})

def group_metrics(g):
    y, f, s = g[TARGET].astype(int), g["flagged"], g["score"]
    tp, fp = int(((f == 1) & (y == 1)).sum()), int(((f == 1) & (y == 0)).sum())
    fn, tn = int(((f == 0) & (y == 1)).sum()), int(((f == 0) & (y == 0)).sum())
    return pd.Series({
        "n": len(g),
        "actual_default_rate": y.mean(),
        "flag_rate": f.mean(),
        "approval_rate": 1 - f.mean(),
        "tpr": tp / max(tp + fn, 1),
        "fpr": fp / max(fp + tn, 1),
        "precision": tp / max(tp + fp, 1),
        "auc": roc_auc_score(y, s) if y.nunique() > 1 else np.nan,
    })

fair_frames = []
for attr in ["sex_label", "age_band"]:
    m = test_pdf.groupby(attr).apply(group_metrics).reset_index().rename(columns={attr: "group"})
    m.insert(0, "attribute", "sex" if attr == "sex_label" else "age_band")
    m["disparate_impact_ratio"] = m["approval_rate"] / m["approval_rate"].max()
    m["tpr_gap_vs_max"] = m["tpr"].max() - m["tpr"]
    m["fpr_gap_vs_min"] = m["fpr"] - m["fpr"].min()
    m["review_flag"] = (m["disparate_impact_ratio"] < 0.80) | (m["tpr_gap_vs_max"] > 0.10)
    fair_frames.append(m)

fair_pdf = pd.concat(fair_frames, ignore_index=True)
fair_pdf["n"] = fair_pdf["n"].astype(int)

(spark.createDataFrame(fair_pdf)
 .withColumn("computed_at", F.current_timestamp())
 .write.mode("overwrite").option("overwriteSchema", "true")
 .saveAsTable(f"{CATALOG}.{SCHEMA}.fairness_audit"))

display(fair_pdf.round(3))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Reading the fairness results (for your README)
# MAGIC * Flag-rate gaps partly reflect **real differences in default rates** between groups in this dataset. They are a prompt to investigate, not proof of bias.
# MAGIC * `review_flag = true` means a group is below the 0.80 disparate impact level or has a TPR gap above 0.10.
# MAGIC * **Limitations:** one historical Taiwanese dataset from 2005, only two protected attributes, small groups for the oldest ages, and no causal analysis. A real lender would also need legal review, since fair-lending rules differ by country.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lineage and governance
# MAGIC Unity Catalog records lineage automatically. For the visual graph, open **Catalog → bank_mlops → credit → gold_features → Lineage** and click **See lineage graph**. Take a screenshot for your README. The query below lists the same relationships as data, if system tables are available in your workspace.

# COMMAND ----------

try:
    display(spark.sql("""
        SELECT DISTINCT source_table_full_name AS source, target_table_full_name AS target
        FROM system.access.table_lineage
        WHERE target_table_full_name LIKE 'bank_mlops.credit.%'
        ORDER BY target
    """))
except Exception as e:
    print(f"System lineage tables are not available here: {str(e)[:200]}")
    print("Use the Lineage tab in Catalog Explorer instead.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Optional: Lakehouse Monitoring
# MAGIC Databricks' built-in monitor profiles a table and writes drift and quality dashboards. It may not be available on Free Edition, so this is optional and wrapped in a try block. Set the `create_lakehouse_monitor` widget to `true` to attempt it.

# COMMAND ----------

if CREATE_MONITOR:
    try:
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.service.catalog import MonitorSnapshot
        w = WorkspaceClient()
        w.quality_monitors.create(
            table_name=PREDICTIONS,
            assets_dir=f"/Users/{user}/lakehouse_monitoring/credit",
            output_schema_name=f"{CATALOG}.{SCHEMA}",
            snapshot=MonitorSnapshot())
        print(f"Monitor created on {PREDICTIONS}")
    except Exception as e:
        print("Lakehouse Monitoring is not available in this workspace.")
        print(f"Reason: {str(e)[:300]}")
else:
    print("Skipped (create_lakehouse_monitor = false). The PSI/KS monitoring above is the fallback.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print("Drift:     bank_mlops.credit.drift_metrics (PSI and KS per feature and score)")
print("Quality:   bank_mlops.credit.monitoring_metrics (AUC and flag rate per scenario)")
print("Trigger:   bank_mlops.credit.retraining_triggers")
print("Fairness:  bank_mlops.credit.fairness_audit")
display(trig_pdf.round(3))

# COMMAND ----------

# MAGIC %md
# MAGIC **Project notebooks complete.** Remaining: deploy the Job with the Asset Bundle, then write the README with the architecture diagram, results tables, screenshots and glossary.
