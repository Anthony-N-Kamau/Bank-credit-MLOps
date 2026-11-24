# Databricks notebook source
# MAGIC %md
# MAGIC # Model Deployment: Batch Scoring and Real-Time Serving
# MAGIC **Project:** Credit card default prediction
# MAGIC
# MAGIC **What this notebook does**
# MAGIC 1. Loads the **`@champion`** model from the Unity Catalog Model Registry (loading by alias, never by hard-coded version)
# MAGIC 2. **Batch scoring:** scores all clients with `mlflow.pyfunc.spark_udf` and writes a `batch_predictions` Delta table
# MAGIC 3. Sanity-checks the batch output against the known labels
# MAGIC 4. **Real-time serving:** creates a Model Serving endpoint with an inference table and sends a test request
# MAGIC
# MAGIC **Free Edition note:** custom Model Serving endpoints may not be available. If endpoint creation fails, the notebook falls back to local `pyfunc` scoring of the same request payload, so everything still runs.

# COMMAND ----------

# MAGIC %pip install lightgbm --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

import json
import pandas as pd
import mlflow
import mlflow.pyfunc
from mlflow import MlflowClient
from pyspark.sql import functions as F
from sklearn.metrics import roc_auc_score

CATALOG = "bank_mlops"
SCHEMA = "credit"
GOLD = f"{CATALOG}.{SCHEMA}.gold_features"
PREDICTIONS = f"{CATALOG}.{SCHEMA}.batch_predictions"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.default_model"
MODEL_URI = f"models:/{MODEL_NAME}@champion"

ENDPOINT_NAME = "credit-default-endpoint"
DEPLOY_ENDPOINT = True   # set to False to skip the Model Serving step

TARGET = "default_next_month"

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
CHAMPION_VERSION = champion.version
THRESHOLD = float(champion.tags.get("threshold", 0.5))

print(f"Champion: {MODEL_NAME} version {CHAMPION_VERSION}")
print(f"Decision threshold (from model version tag): {THRESHOLD:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read the model's expected inputs from its signature
# MAGIC The signature tells us exactly which columns the model needs, so the scoring code never drifts from the training code.

# COMMAND ----------

loaded = mlflow.pyfunc.load_model(MODEL_URI)
FEATURES = loaded.metadata.get_input_schema().input_names()
print(f"{len(FEATURES)} model inputs: {FEATURES}")

gold = spark.table(GOLD)
features_df = gold.select("client_id", "split", TARGET,
                          *[F.col(c).cast("double").alias(c) for c in FEATURES])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Batch scoring
# MAGIC `spark_udf` wraps the registered model as a Spark function so scoring runs in parallel over a DataFrame. If `spark_udf` is not supported in your compute environment, the fallback scores locally with pandas (30,000 rows is tiny).

# COMMAND ----------

scored = None
scoring_method = None

try:
    predict_udf = mlflow.pyfunc.spark_udf(spark, MODEL_URI, result_type="double", env_manager="local")
    scored = features_df.withColumn("default_probability", predict_udf(F.struct(*FEATURES)))
    scored.limit(1).collect()   # force evaluation to surface any runtime error now
    scoring_method = "spark_udf"
except Exception as e:
    print(f"spark_udf not available here ({str(e)[:200]}). Falling back to local pandas scoring.")
    scored = None

if scored is None:
    pdf = features_df.toPandas()
    pdf["default_probability"] = loaded.predict(pdf[FEATURES])
    scored = spark.createDataFrame(pdf)
    scoring_method = "local_pandas"

result = (scored
          .withColumn("predicted_default", (F.col("default_probability") >= F.lit(THRESHOLD)).cast("int"))
          .withColumn("model_name", F.lit(MODEL_NAME))
          .withColumn("model_version", F.lit(str(CHAMPION_VERSION)))
          .withColumn("threshold", F.lit(THRESHOLD))
          .withColumn("scored_at", F.current_timestamp())
          .select("client_id", "split", TARGET, "default_probability", "predicted_default",
                  "model_name", "model_version", "threshold", "scored_at"))

(result.write.mode("overwrite")
 .option("overwriteSchema", "true")
 .saveAsTable(PREDICTIONS))

spark.sql(f"COMMENT ON TABLE {PREDICTIONS} IS 'Batch default probabilities from the champion model'")
print(f"Scoring method: {scoring_method}")
print(f"Wrote {spark.table(PREDICTIONS).count():,} rows to {PREDICTIONS}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sanity checks on the batch output

# COMMAND ----------

preds = spark.table(PREDICTIONS)

display(preds.groupBy("split").agg(
    F.count("*").alias("rows"),
    F.round(F.avg("default_probability"), 4).alias("avg_probability"),
    F.round(F.avg("predicted_default"), 4).alias("flagged_rate"),
    F.round(F.avg(TARGET), 4).alias("actual_default_rate"),
).orderBy("split"))

# COMMAND ----------

# The AUC on the test split must match the AUC logged during training
test_pdf = preds.filter("split = 'test'").select(TARGET, "default_probability").toPandas()
batch_auc = roc_auc_score(test_pdf[TARGET], test_pdf["default_probability"])

run_auc = float(client.get_run(champion.run_id).data.metrics["test_auc"])
print(f"Batch test AUC: {batch_auc:.4f}  |  AUC logged in training: {run_auc:.4f}")
assert abs(batch_auc - run_auc) < 1e-3, "Batch scoring does not reproduce the training metric!"
print("Batch scoring reproduces the training AUC")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Real-time serving: build a test request
# MAGIC Model Serving accepts JSON in the `dataframe_records` format. We build one from real rows.

# COMMAND ----------

sample = (spark.table(GOLD).select(*FEATURES).limit(3).toPandas().astype("float64"))
records = sample.to_dict(orient="records")
payload = {"dataframe_records": records}

print(json.dumps(payload, indent=2)[:1200])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create the Model Serving endpoint
# MAGIC * Serves the registered model version with **scale to zero** (no cost when idle)
# MAGIC * Enables an **inference table**, which logs every request and response to Delta. This feeds the monitoring work in `monitoring_drift_fairness`.
# MAGIC * Endpoints serve a **version number**, not an alias, so the promotion job later updates the endpoint when the champion changes.

# COMMAND ----------

endpoint_created = False

if DEPLOY_ENDPOINT:
    try:
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.service.serving import (
            EndpointCoreConfigInput, ServedEntityInput, AutoCaptureConfigInput)

        w = WorkspaceClient()
        config = EndpointCoreConfigInput(
            name=ENDPOINT_NAME,
            served_entities=[ServedEntityInput(
                entity_name=MODEL_NAME,
                entity_version=str(CHAMPION_VERSION),
                workload_size="Small",
                scale_to_zero_enabled=True)],
            auto_capture_config=AutoCaptureConfigInput(
                catalog_name=CATALOG,
                schema_name=SCHEMA,
                table_name_prefix="credit_endpoint",
                enabled=True),
        )

        existing = [e.name for e in w.serving_endpoints.list()]
        if ENDPOINT_NAME in existing:
            w.serving_endpoints.update_config_and_wait(
                name=ENDPOINT_NAME,
                served_entities=config.served_entities,
                auto_capture_config=config.auto_capture_config)
            print(f"Updated endpoint {ENDPOINT_NAME} to version {CHAMPION_VERSION}")
        else:
            w.serving_endpoints.create_and_wait(name=ENDPOINT_NAME, config=config)
            print(f"Created endpoint {ENDPOINT_NAME}")
        endpoint_created = True

    except Exception as e:
        print("Model Serving endpoint could not be created in this workspace.")
        print(f"Reason: {str(e)[:400]}")
        print("Continuing with the local fallback below.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Send a test request

# COMMAND ----------

if endpoint_created:
    response = w.serving_endpoints.query(name=ENDPOINT_NAME, dataframe_records=records)
    probs = list(response.predictions)
    print(f"Source: Model Serving endpoint '{ENDPOINT_NAME}'")
else:
    probs = list(loaded.predict(sample))
    print("Source: local pyfunc fallback (same model, same payload)")

out = pd.DataFrame({
    "default_probability": [round(float(p), 4) for p in probs],
    "decision": ["flag for review" if p >= THRESHOLD else "approve" for p in probs],
})
display(out)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Example call from outside Databricks
# MAGIC Replace the host and token with your own (create a personal access token under **Settings → Developer**). Never commit a token.
# MAGIC
# MAGIC ```bash
# MAGIC curl -X POST https://YOUR-WORKSPACE-URL/serving-endpoints/credit-default-endpoint/invocations \
# MAGIC   -H "Authorization: Bearer $DATABRICKS_TOKEN" \
# MAGIC   -H "Content-Type: application/json" \
# MAGIC   -d @payload.json
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print(f"Model:             {MODEL_NAME} (champion = version {CHAMPION_VERSION})")
print(f"Batch table:       {PREDICTIONS}  ({scoring_method})")
print(f"Real-time serving: {'endpoint ' + ENDPOINT_NAME if endpoint_created else 'local fallback (endpoint not available)'}")

# COMMAND ----------

# MAGIC %md
# MAGIC **Next:** `validate_and_promote`, the automated gate that only promotes a new model to `@champion` if it beats the current one, then the Job and Asset Bundle that run the whole pipeline.
