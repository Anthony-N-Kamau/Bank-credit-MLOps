# Databricks notebook source
# MAGIC %md
# MAGIC # Machine Learning at Scale
# MAGIC **Project:** Credit card default prediction
# MAGIC
# MAGIC The real dataset has only 30,000 rows. To show how the platform behaves at scale, this notebook **replicates it to millions of rows** (with random noise so rows are not exact copies) and benchmarks single-node against distributed approaches.
# MAGIC
# MAGIC | Experiment | Single-node / baseline | Distributed / optimised |
# MAGIC |---|---|---|
# MAGIC | Storage layout | Unclustered Delta table | **Liquid clustering + OPTIMIZE** |
# MAGIC | Batch scoring | pandas on the driver | **pandas UDF** across Spark |
# MAGIC | Model training | scikit-learn | **Spark ML** |
# MAGIC | Hyperparameter search | Sequential loop | **Parallel trials with `mapInPandas`** |
# MAGIC | Many models | | **One model per segment with `applyInPandas`** |
# MAGIC
# MAGIC **Important:** the scaled data is synthetic and exists only for benchmarking. It says nothing about model quality.
# MAGIC
# MAGIC **Free Edition tip:** start with `scale_factor = 100` (3 million rows). Raise it only if everything runs smoothly. Every timing is saved to a Delta table and to MLflow.

# COMMAND ----------

# MAGIC %pip install lightgbm --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

import re
import time
import numpy as np
import pandas as pd
import mlflow
import mlflow.pyfunc
import lightgbm as lgb
from pyspark.sql import functions as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

dbutils.widgets.text("scale_factor", "100")
SCALE_FACTOR = int(dbutils.widgets.get("scale_factor"))   # 30,000 rows x scale_factor

CATALOG = "bank_mlops"
SCHEMA = "credit"
GOLD = f"{CATALOG}.{SCHEMA}.gold_features"
SCALED_RAW = f"{CATALOG}.{SCHEMA}.gold_features_scaled_raw"
SCALED = f"{CATALOG}.{SCHEMA}.gold_features_scaled"
BENCH_TABLE = f"{CATALOG}.{SCHEMA}.scale_benchmarks"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.default_model"
MODEL_URI = f"models:/{MODEL_NAME}@champion"
TARGET = "default_next_month"
SEED = 42

mlflow.set_registry_uri("databricks-uc")
user = spark.sql("SELECT current_user()").collect()[0][0]
mlflow.set_experiment(f"/Users/{user}/credit_default")

results = []

def timed(fn):
    start = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - start

def record(experiment, approach, rows, seconds, note=""):
    results.append({"experiment": experiment, "approach": approach, "rows": int(rows),
                    "seconds": round(float(seconds), 2), "note": note})
    print(f"[{experiment}] {approach}: {seconds:.1f}s  ({rows:,} rows) {note}")

print(f"Scale factor {SCALE_FACTOR} -> about {30000 * SCALE_FACTOR:,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the champion model and its feature list

# COMMAND ----------

loaded = mlflow.pyfunc.load_model(MODEL_URI)
FEATURES = loaded.metadata.get_input_schema().input_names()
lgbm_model = loaded.unwrap_python_model().model   # the underlying LightGBM model
print(f"{len(FEATURES)} features; model type: {type(lgbm_model).__name__}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create the scaled dataset
# MAGIC Cross-join the 30,000 rows with `scale_factor` replicas, give each row a new unique `client_id`, and add 5% multiplicative noise to continuous columns (the first replica keeps the original values). The data is then **shuffled** before writing so the table behaves like randomly arriving data.

# COMMAND ----------

CONTINUOUS = [c for c in FEATURES if c.startswith(("bill_amt", "pay_amt")) or c in
              ["limit_bal", "avg_bill", "avg_pay_amt", "util_ratio", "avg_util_ratio",
               "pay_to_bill_ratio", "bill_trend"]]

replicas = F.broadcast(spark.range(SCALE_FACTOR).withColumnRenamed("id", "replica"))
scaled = (spark.table(GOLD).crossJoin(replicas)
          .withColumn("client_id", F.col("client_id").cast("long") + F.col("replica") * 1_000_000))

for i, c in enumerate(CONTINUOUS):
    scaled = scaled.withColumn(
        c, F.when(F.col("replica") == 0, F.col(c))
            .otherwise(F.col(c) * (1 + 0.05 * F.randn(100 + i))))

scaled = (scaled
          .select("client_id", "split", TARGET, *[F.col(c).cast("double").alias(c) for c in FEATURES])
          .orderBy(F.rand(1)))

_, secs = timed(lambda: (scaled.write.mode("overwrite")
                         .option("overwriteSchema", "true").saveAsTable(SCALED_RAW)))
N_ROWS = spark.table(SCALED_RAW).count()
record("data generation", "spark write", N_ROWS, secs, "replicate, add noise, shuffle, write Delta")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Experiment 1: storage layout (liquid clustering + OPTIMIZE)
# MAGIC The same selective query (one slice of `client_id` values) is run on a randomly ordered table and on a **liquid clustered** copy. Clustering lets Delta skip files that cannot contain the requested ids.

# COMMAND ----------

_, secs = timed(lambda: (
    spark.sql(f"CREATE OR REPLACE TABLE {SCALED} CLUSTER BY (client_id) AS SELECT * FROM {SCALED_RAW}"),
    spark.sql(f"OPTIMIZE {SCALED}")))
record("table build", "liquid clustering + OPTIMIZE", N_ROWS, secs, "one-off cost")

lo, hi = 5_000_000, 5_029_999

def selective_query(table):
    return spark.sql(f"SELECT COUNT(*) AS c, AVG(limit_bal) AS a FROM {table} "
                     f"WHERE client_id BETWEEN {lo} AND {hi}").collect()

def best_of(table, runs=3):
    times = []
    for _ in range(runs):
        _, s = timed(lambda: selective_query(table))
        times.append(s)
    return min(times)

record("selective query", "unclustered", N_ROWS, best_of(SCALED_RAW), "best of 3")
record("selective query", "liquid clustering", N_ROWS, best_of(SCALED), "best of 3")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Experiment 2: batch scoring
# MAGIC * **Single-node:** pull everything to the driver as pandas and call `predict_proba`
# MAGIC * **Distributed:** a pandas UDF scores batches in parallel on the Spark workers; results are written with the `noop` sink so only compute time is measured

# COMMAND ----------

def score_single_node():
    pdf = spark.table(SCALED).select(*FEATURES).toPandas()
    return lgbm_model.predict_proba(pdf[FEATURES])[:, 1]

try:
    _, secs = timed(score_single_node)
    record("batch scoring", "single-node", N_ROWS, secs, "toPandas + predict on driver")
except Exception as e:
    record("batch scoring", "single-node", N_ROWS, float("nan"), f"failed: {type(e).__name__}")

@F.pandas_udf("double")
def score_udf(batch: pd.DataFrame) -> pd.Series:
    return pd.Series(lgbm_model.predict_proba(batch[FEATURES])[:, 1])

def score_distributed():
    (spark.table(SCALED)
     .select("client_id", score_udf(F.struct(*FEATURES)).alias("p"))
     .write.format("noop").mode("overwrite").save())

_, secs = timed(score_distributed)
record("batch scoring", "distributed pandas UDF", N_ROWS, secs, "parallel across Spark tasks")

# COMMAND ----------

# Correctness: both paths must produce identical probabilities
sample = spark.table(SCALED).orderBy("client_id").limit(1000)
sample_pdf = sample.toPandas()
single = lgbm_model.predict_proba(sample_pdf[FEATURES])[:, 1]
dist = (sample.select("client_id", score_udf(F.struct(*FEATURES)).alias("p"))
        .toPandas().set_index("client_id").loc[sample_pdf["client_id"], "p"].values)
assert np.allclose(single, dist, atol=1e-9), "Distributed scoring differs from single-node!"
print("Distributed and single-node predictions match")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Experiment 3: model training (logistic regression)
# MAGIC The same algorithm trained with scikit-learn on one node and with Spark ML across the cluster. If Spark ML is not available on your compute, the failure is recorded instead of crashing the notebook.

# COMMAND ----------

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

train_sdf = spark.table(SCALED).filter("split = 'train'")
N_TRAIN = train_sdf.count()

try:
    train_pdf, load_secs = timed(lambda: train_sdf.select(*FEATURES, TARGET).toPandas())
    pipe = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=100))])
    _, secs = timed(lambda: pipe.fit(train_pdf[FEATURES], train_pdf[TARGET]))
    record("logistic regression training", "single-node scikit-learn", N_TRAIN, secs,
           f"fit only; loading to pandas took {load_secs:.1f}s")
except Exception as e:
    record("logistic regression training", "single-node scikit-learn", N_TRAIN, float("nan"),
           f"failed: {type(e).__name__}")

try:
    from pyspark.ml.feature import VectorAssembler, StandardScaler as SparkScaler
    from pyspark.ml.classification import LogisticRegression as SparkLR
    from pyspark.ml import Pipeline as SparkPipeline

    spark_pipe = SparkPipeline(stages=[
        VectorAssembler(inputCols=FEATURES, outputCol="raw_features"),
        SparkScaler(inputCol="raw_features", outputCol="features", withMean=True, withStd=True),
        SparkLR(featuresCol="features", labelCol=TARGET, maxIter=100),
    ])
    _, secs = timed(lambda: spark_pipe.fit(train_sdf))
    record("logistic regression training", "distributed Spark ML", N_TRAIN, secs, "full pipeline fit")
except Exception as e:
    record("logistic regression training", "distributed Spark ML", N_TRAIN, float("nan"),
           f"not supported on this compute: {type(e).__name__}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Experiment 4: parallel hyperparameter search
# MAGIC Sixteen LightGBM configurations are trained with equal work each (300 trees, no early stopping). The sequential loop uses every core of one machine; the parallel version gives each trial to a Spark task using one core, so the speedup depends on how many workers serverless provides.

# COMMAND ----------

base_pdf = spark.table(GOLD).filter("split = 'train'").toPandas()
Xb = base_pdf[FEATURES].astype("float64").values
yb = base_pdf[TARGET].astype(int).values
X_tr, X_va, y_tr, y_va = train_test_split(Xb, yb, test_size=0.2, stratify=yb, random_state=SEED)

rng = np.random.default_rng(SEED)
N_TRIALS = 16
configs = pd.DataFrame({
    "trial": range(N_TRIALS),
    "num_leaves": rng.choice([15, 31, 63, 127], N_TRIALS),
    "learning_rate": np.exp(rng.uniform(np.log(0.02), np.log(0.15), N_TRIALS)),
    "min_child_samples": rng.choice([20, 50, 100], N_TRIALS),
    "subsample": rng.uniform(0.6, 1.0, N_TRIALS),
    "colsample_bytree": rng.uniform(0.6, 1.0, N_TRIALS),
})

def run_trial(row, n_jobs):
    m = lgb.LGBMClassifier(
        n_estimators=300, num_leaves=int(row["num_leaves"]), learning_rate=float(row["learning_rate"]),
        min_child_samples=int(row["min_child_samples"]), subsample=float(row["subsample"]),
        subsample_freq=1, colsample_bytree=float(row["colsample_bytree"]),
        n_jobs=n_jobs, verbose=-1, random_state=SEED)
    m.fit(X_tr, y_tr)
    return float(roc_auc_score(y_va, m.predict_proba(X_va)[:, 1]))

def sequential_search():
    return [(int(r["trial"]), run_trial(r, -1)) for _, r in configs.iterrows()]

seq_out, secs = timed(sequential_search)
record("hyperparameter search", "sequential", N_TRIALS, secs, f"{N_TRIALS} trials; rows = trials")

def tune_batch(iterator):
    for pdf in iterator:
        rows = [(int(r["trial"]), run_trial(r, 1)) for _, r in pdf.iterrows()]
        yield pd.DataFrame(rows, columns=["trial", "val_auc"])

def parallel_search():
    sdf = spark.createDataFrame(configs).repartition(N_TRIALS)
    return sdf.mapInPandas(tune_batch, "trial long, val_auc double").toPandas()

par_out, secs = timed(parallel_search)
record("hyperparameter search", "parallel mapInPandas", N_TRIALS, secs, f"{N_TRIALS} trials; rows = trials")

print(f"Best val AUC  sequential: {max(a for _, a in seq_out):.4f}  parallel: {par_out['val_auc'].max():.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Experiment 5: one model per segment with `applyInPandas`
# MAGIC The "many models" pattern: split the data by `education` and `marriage` and train an independent LightGBM model per group in parallel. Useful when each customer segment deserves its own model.

# COMMAND ----------

def train_segment(pdf: pd.DataFrame) -> pd.DataFrame:
    import time as _t
    import lightgbm as _lgb
    from sklearn.metrics import roc_auc_score as _auc

    start = _t.perf_counter()
    tr, te = pdf[pdf["split"] == "train"], pdf[pdf["split"] == "test"]
    auc = float("nan")
    if len(te) > 0 and tr[TARGET].nunique() > 1 and te[TARGET].nunique() > 1:
        m = _lgb.LGBMClassifier(n_estimators=100, n_jobs=1, verbose=-1)
        m.fit(tr[FEATURES], tr[TARGET])
        auc = float(_auc(te[TARGET], m.predict_proba(te[FEATURES])[:, 1]))
    return pd.DataFrame({
        "education": [float(pdf["education"].iloc[0])],
        "marriage": [float(pdf["marriage"].iloc[0])],
        "n_rows": [len(pdf)],
        "test_auc": [auc],
        "train_seconds": [_t.perf_counter() - start],
    })

seg_schema = "education double, marriage double, n_rows long, test_auc double, train_seconds double"

def many_models():
    return (spark.table(SCALED).select("split", TARGET, *FEATURES)
            .groupBy("education", "marriage")
            .applyInPandas(train_segment, seg_schema).toPandas())

seg_pdf, secs = timed(many_models)
record("many models", "distributed applyInPandas", N_ROWS, secs,
       f"{len(seg_pdf)} segment models; sum of per-model time {seg_pdf['train_seconds'].sum():.0f}s")
display(seg_pdf.sort_values(["education", "marriage"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results
# MAGIC `speedup` compares each approach with the first approach listed for the same experiment (the baseline).

# COMMAND ----------

res = pd.DataFrame(results)
res["baseline_seconds"] = res.groupby("experiment")["seconds"].transform("first")
res["speedup"] = (res["baseline_seconds"] / res["seconds"]).round(2)
res = res.drop(columns="baseline_seconds")
display(res)

# COMMAND ----------

# Save the benchmarks to Delta and MLflow
(spark.createDataFrame(res.fillna({"seconds": -1.0}))
 .withColumn("scale_factor", F.lit(SCALE_FACTOR))
 .withColumn("recorded_at", F.current_timestamp())
 .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(BENCH_TABLE))

with mlflow.start_run(run_name="scale_experiments"):
    mlflow.set_tag("stage", "benchmark")
    mlflow.log_param("scale_factor", SCALE_FACTOR)
    mlflow.log_param("rows", N_ROWS)
    for _, r in res.dropna(subset=["seconds"]).iterrows():
        name = re.sub(r"[^0-9a-zA-Z_\-\. /]", "_", f"{r['experiment']} | {r['approach']}")
        mlflow.log_metric(name + " seconds", float(r["seconds"]))

print(f"Saved benchmarks to {BENCH_TABLE} and MLflow")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clean up (optional)
# MAGIC The scaled tables are large. Uncomment to remove them once you have your screenshots. Keep `scale_benchmarks`, which is tiny.

# COMMAND ----------

# spark.sql(f"DROP TABLE IF EXISTS {SCALED_RAW}")
# spark.sql(f"DROP TABLE IF EXISTS {SCALED}")

# COMMAND ----------

# MAGIC %md
# MAGIC **Next:** `monitoring_drift_fairness`, which simulates data drift, computes PSI and KS drift metrics, audits fairness by sex and age band, and sets up the retraining trigger.
