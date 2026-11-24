# Databricks notebook source
# MAGIC %md
# MAGIC # Validation Gate and Model Promotion
# MAGIC **Project:** Credit card default prediction
# MAGIC
# MAGIC This notebook is the automated quality gate in the pipeline. It decides whether the newest model (`@challenger`) may replace the current production model (`@champion`).
# MAGIC
# MAGIC **Checks the challenger must pass**
# MAGIC 1. Has a model signature
# MAGIC 2. Has a decision threshold recorded
# MAGIC 3. Test AUC above a minimum absolute level
# MAGIC 4. Smoke test: loads from the registry and returns valid probabilities
# MAGIC 5. **Beats the current champion's test AUC** (by at least `min_auc_improvement`)
# MAGIC
# MAGIC **If it passes:** `@champion` moves to the new version, the old champion keeps an `@previous_champion` alias (instant rollback), and the serving endpoint is updated.
# MAGIC **Every decision** is appended to the `promotion_log` Delta table for auditing.

# COMMAND ----------

# MAGIC %pip install lightgbm --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration
# MAGIC Widgets let the Job override these values without editing code.

# COMMAND ----------

import numpy as np
import pandas as pd
import mlflow
import mlflow.pyfunc
from mlflow import MlflowClient
from pyspark.sql import functions as F

dbutils.widgets.text("min_auc_improvement", "0.0")
dbutils.widgets.text("min_absolute_auc", "0.70")
dbutils.widgets.dropdown("fail_if_not_promoted", "false", ["true", "false"])
dbutils.widgets.text("endpoint_name", "credit-default-endpoint")

MIN_IMPROVEMENT = float(dbutils.widgets.get("min_auc_improvement"))
MIN_ABSOLUTE_AUC = float(dbutils.widgets.get("min_absolute_auc"))
FAIL_IF_NOT_PROMOTED = dbutils.widgets.get("fail_if_not_promoted") == "true"
ENDPOINT_NAME = dbutils.widgets.get("endpoint_name")

CATALOG = "bank_mlops"
SCHEMA = "credit"
GOLD = f"{CATALOG}.{SCHEMA}.gold_features"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.default_model"
LOG_TABLE = f"{CATALOG}.{SCHEMA}.promotion_log"

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load challenger and champion

# COMMAND ----------

challenger = client.get_model_version_by_alias(MODEL_NAME, "challenger")
try:
    champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
except Exception:
    champion = None

def test_auc_of(version):
    return float(client.get_run(version.run_id).data.metrics["test_auc"])

cand_auc = test_auc_of(challenger)
champ_auc = test_auc_of(champion) if champion else None

print(f"Challenger: version {challenger.version}  test AUC {cand_auc:.4f}")
print(f"Champion:   " + (f"version {champion.version}  test AUC {champ_auc:.4f}" if champion else "none yet"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run the validation checks

# COMMAND ----------

checks = {}

# 1. Signature present
info = mlflow.models.get_model_info(f"models:/{MODEL_NAME}/{challenger.version}")
checks["has_signature"] = info.signature is not None

# 2. Threshold recorded
checks["has_threshold_tag"] = "threshold" in challenger.tags

# 3. Minimum absolute quality
checks[f"test_auc_ge_{MIN_ABSOLUTE_AUC}"] = cand_auc >= MIN_ABSOLUTE_AUC

# 4. Smoke test through the registry
try:
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{challenger.version}")
    cols = model.metadata.get_input_schema().input_names()
    sample = (spark.table(GOLD).select(*[F.col(c).cast("double").alias(c) for c in cols])
              .limit(200).toPandas())
    probs = np.asarray(model.predict(sample), dtype=float)
    checks["smoke_test_valid_probabilities"] = bool(
        len(probs) == len(sample) and not np.isnan(probs).any()
        and probs.min() >= 0 and probs.max() <= 1)
except Exception as e:
    print(f"Smoke test error: {str(e)[:300]}")
    checks["smoke_test_valid_probabilities"] = False

display(pd.DataFrame({"check": list(checks.keys()), "passed": list(checks.values())}))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Decision

# COMMAND ----------

all_checks_pass = all(checks.values())

if champion is not None and champion.version == challenger.version:
    decision, reason = "skipped", "Challenger is already the champion"
elif not all_checks_pass:
    failed = [k for k, v in checks.items() if not v]
    decision, reason = "rejected", f"Failed checks: {failed}"
elif champion is None:
    decision, reason = "promoted", "No champion exists yet"
elif cand_auc > champ_auc + MIN_IMPROVEMENT:
    decision, reason = "promoted", f"AUC {cand_auc:.4f} beats champion {champ_auc:.4f} by more than {MIN_IMPROVEMENT}"
else:
    decision, reason = "rejected", f"AUC {cand_auc:.4f} does not beat champion {champ_auc:.4f} (required improvement {MIN_IMPROVEMENT})"

print(f"Decision: {decision.upper()}  |  {reason}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Act on the decision

# COMMAND ----------

if decision == "promoted":
    if champion is not None:
        client.set_registered_model_alias(MODEL_NAME, "previous_champion", champion.version)
    client.set_registered_model_alias(MODEL_NAME, "champion", challenger.version)
    client.set_model_version_tag(MODEL_NAME, challenger.version, "validation_status", "approved")
    print(f"Version {challenger.version} is now @champion"
          + (f" (version {champion.version} kept as @previous_champion)" if champion else ""))

    # Point the serving endpoint at the new champion (endpoints serve versions, not aliases)
    try:
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.service.serving import ServedEntityInput

        w = WorkspaceClient()
        if ENDPOINT_NAME in [e.name for e in w.serving_endpoints.list()]:
            w.serving_endpoints.update_config_and_wait(
                name=ENDPOINT_NAME,
                served_entities=[ServedEntityInput(
                    entity_name=MODEL_NAME,
                    entity_version=str(challenger.version),
                    workload_size="Small",
                    scale_to_zero_enabled=True)])
            print(f"Endpoint {ENDPOINT_NAME} updated to version {challenger.version}")
        else:
            print(f"Endpoint {ENDPOINT_NAME} not found, skipping endpoint update")
    except Exception as e:
        print(f"Endpoint update skipped: {str(e)[:200]}")

elif decision == "rejected":
    client.set_model_version_tag(MODEL_NAME, challenger.version, "validation_status", "rejected")
    print(f"Version {challenger.version} rejected. Champion unchanged.")
else:
    print("Nothing to do.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write the audit log

# COMMAND ----------

log_row = [(
    str(challenger.version),
    str(champion.version) if champion else None,
    cand_auc,
    champ_auc,
    decision,
    reason,
)]

(spark.createDataFrame(
    log_row,
    "challenger_version STRING, champion_version STRING, challenger_auc DOUBLE, "
    "champion_auc DOUBLE, decision STRING, reason STRING")
 .withColumn("decided_at", F.current_timestamp())
 .write.mode("append").saveAsTable(LOG_TABLE))

display(spark.table(LOG_TABLE).orderBy(F.col("decided_at").desc()))

# COMMAND ----------

# Pass the result to downstream Job tasks
try:
    dbutils.jobs.taskValues.set(key="decision", value=decision)
    dbutils.jobs.taskValues.set(key="champion_version",
                                value=str(challenger.version if decision == "promoted"
                                          else (champion.version if champion else challenger.version)))
except Exception as e:
    print(f"Task values not set (only available inside a Job): {str(e)[:100]}")

if FAIL_IF_NOT_PROMOTED and decision == "rejected":
    raise Exception(f"Validation gate failed: {reason}")

# COMMAND ----------

# MAGIC %md
# MAGIC **Next:** the Job definition (`resources/credit_pipeline.yml`) chains `data_prep`, `train_model`, `validate_and_promote` and `batch_and_serving`, and `databricks.yml` deploys it as an Asset Bundle.
