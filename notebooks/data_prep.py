# Databricks notebook source
# MAGIC %md
# MAGIC # Data Preparation for Machine Learning
# MAGIC **Project:** Credit card default prediction (UCI Taiwan dataset)
# MAGIC
# MAGIC **What this notebook does**
# MAGIC 1. **Bronze**: ingest the raw CSV from a Unity Catalog Volume into Delta, untouched apart from column names and audit columns
# MAGIC 2. **Silver**: clean undocumented codes, enforce types, de-duplicate, run data quality checks
# MAGIC 3. **Gold**: engineer features and create a reproducible, stratified train/test split
# MAGIC 4. Demonstrate Delta time travel and register the gold table as a feature table (primary key)
# MAGIC
# MAGIC **Output tables** (`bank_mlops.credit`): `bronze_clients`, `silver_clients`, `dq_results`, `gold_features`

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Configuration

# COMMAND ----------

import re
import builtins
from functools import reduce
from pyspark.sql import functions as F
from pyspark.sql.window import Window

CATALOG = "bank_mlops"
SCHEMA = "credit"
RAW_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/raw/credit_default.csv"

BRONZE = f"{CATALOG}.{SCHEMA}.bronze_clients"
SILVER = f"{CATALOG}.{SCHEMA}.silver_clients"
GOLD = f"{CATALOG}.{SCHEMA}.gold_features"
DQ_TABLE = f"{CATALOG}.{SCHEMA}.dq_results"

TARGET = "default_next_month"
SPLIT_SEED = 42
TRAIN_FRACTION = 0.8

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")
print(f"Using {CATALOG}.{SCHEMA}; reading {RAW_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Bronze: raw ingestion
# MAGIC Delta columns should not contain spaces, so we standardise names to `snake_case`. No values are changed in Bronze.

# COMMAND ----------

def to_snake(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", name.strip()).strip("_").lower()

raw = (spark.read
       .option("header", True)
       .option("inferSchema", True)
       .csv(RAW_PATH))

renamed = raw.toDF(*[to_snake(c) for c in raw.columns])

# Friendlier names. PAY_0 in the UCI file is really "repayment status of the most recent month".
rename_map = {"id": "client_id", "pay_0": "pay_1", "default_payment_next_month": TARGET}
for old, new in rename_map.items():
    if old in renamed.columns:
        renamed = renamed.withColumnRenamed(old, new)

bronze_df = (renamed
             .withColumn("_ingested_at", F.current_timestamp())
             .withColumn("_source_file", F.col("_metadata.file_path")))

(bronze_df.write
 .mode("overwrite")
 .option("overwriteSchema", "true")
 .saveAsTable(BRONZE))

print(f"Bronze rows: {spark.table(BRONZE).count():,}")
display(spark.table(BRONZE).limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Silver: cleaning and typing
# MAGIC Known issues in this dataset (documented in the UCI description vs actual values):
# MAGIC * `education` should be 1-4 but contains 0, 5 and 6 -> merge into 4 ("others")
# MAGIC * `marriage` should be 1-3 but contains 0 -> merge into 3 ("others")
# MAGIC * `pay_*` columns range from -2 to 8 (-2 = no consumption, -1 = paid in full, 0 = revolving credit, 1+ = months overdue)

# COMMAND ----------

bronze = spark.table(BRONZE)

pay_cols = [f"pay_{i}" for i in range(1, 7)]
bill_cols = [f"bill_amt{i}" for i in range(1, 7)]
pay_amt_cols = [f"pay_amt{i}" for i in range(1, 7)]

int_cols = ["client_id", "sex", "education", "marriage", "age", TARGET] + pay_cols
dbl_cols = ["limit_bal"] + bill_cols + pay_amt_cols

silver_df = bronze
for c in int_cols:
    silver_df = silver_df.withColumn(c, F.col(c).cast("int"))
for c in dbl_cols:
    silver_df = silver_df.withColumn(c, F.col(c).cast("double"))

silver_df = (silver_df
             .withColumn("education", F.when(F.col("education").isin(0, 5, 6), 4).otherwise(F.col("education")))
             .withColumn("marriage", F.when(F.col("marriage") == 0, 3).otherwise(F.col("marriage")))
             .dropDuplicates(["client_id"])
             .select("client_id", "limit_bal", "sex", "education", "marriage", "age",
                     *pay_cols, *bill_cols, *pay_amt_cols, TARGET)
             .withColumn("_processed_at", F.current_timestamp()))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data quality checks
# MAGIC Each check counts the rows that violate a rule. The notebook **fails** if any rule is broken, which is what makes this safe to run in an automated job later.

# COMMAND ----------

def violations(cond):
    return F.sum(F.when(cond, 1).otherwise(0))

all_feature_cols = ["client_id", "limit_bal", "sex", "education", "marriage", "age",
                    *pay_cols, *bill_cols, *pay_amt_cols, TARGET]

null_expr = reduce(lambda a, b: a | b, [F.col(c).isNull() for c in all_feature_cols])

checks = silver_df.agg(
    F.count("*").alias("total_rows"),
    F.countDistinct("client_id").alias("distinct_ids"),
    violations(null_expr).alias("rows_with_nulls"),
    violations(~F.col("sex").isin(1, 2)).alias("bad_sex"),
    violations(~F.col("education").isin(1, 2, 3, 4)).alias("bad_education"),
    violations(~F.col("marriage").isin(1, 2, 3)).alias("bad_marriage"),
    violations((F.col("age") < 18) | (F.col("age") > 100)).alias("bad_age"),
    violations(F.col("limit_bal") <= 0).alias("bad_limit"),
    violations(~F.col(TARGET).isin(0, 1)).alias("bad_target"),
    violations(reduce(lambda a, b: a | b, [(F.col(c) < -2) | (F.col(c) > 9) for c in pay_cols])).alias("bad_pay_status"),
    violations(reduce(lambda a, b: a | b, [F.col(c) < 0 for c in pay_amt_cols])).alias("negative_payments"),
).collect()[0].asDict()

dq_rules = [
    ("row_count_is_30000",       checks["total_rows"] == 30000),
    ("client_id_unique",         checks["distinct_ids"] == checks["total_rows"]),
    ("no_nulls",                 checks["rows_with_nulls"] == 0),
    ("sex_in_1_2",               checks["bad_sex"] == 0),
    ("education_in_1_4",         checks["bad_education"] == 0),
    ("marriage_in_1_3",          checks["bad_marriage"] == 0),
    ("age_between_18_100",       checks["bad_age"] == 0),
    ("limit_bal_positive",       checks["bad_limit"] == 0),
    ("target_binary",            checks["bad_target"] == 0),
    ("pay_status_in_range",      checks["bad_pay_status"] == 0),
    ("payment_amounts_nonneg",   checks["negative_payments"] == 0),
]

dq_df = spark.createDataFrame(
    [(name, bool(ok), str(checks)) for name, ok in dq_rules],
    "rule STRING, passed BOOLEAN, raw_metrics STRING",
).withColumn("checked_at", F.current_timestamp())

dq_df.write.mode("overwrite").saveAsTable(DQ_TABLE)
display(dq_df.select("rule", "passed", "checked_at"))

failed = [name for name, ok in dq_rules if not ok]
assert not failed, f"Data quality checks failed: {failed}"
print("All data quality checks passed")

# COMMAND ----------

(silver_df.write
 .mode("overwrite")
 .option("overwriteSchema", "true")
 .saveAsTable(SILVER))

print(f"Silver rows: {spark.table(SILVER).count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Gold: feature engineering and stratified split
# MAGIC Engineered features:
# MAGIC | Feature | Meaning |
# MAGIC |---|---|
# MAGIC | `util_ratio` | latest bill / credit limit (credit utilisation) |
# MAGIC | `avg_util_ratio` | average bill over 6 months / limit |
# MAGIC | `avg_bill`, `avg_pay_amt` | 6-month averages |
# MAGIC | `pay_to_bill_ratio` | total paid / total positive billed (repayment behaviour) |
# MAGIC | `avg_delay`, `max_delay` | average and worst repayment status |
# MAGIC | `months_delayed` | number of months with an overdue status (> 0) |
# MAGIC | `bill_trend` | (latest bill - oldest bill) / limit, is debt growing? |
# MAGIC | `age_band` | used for the fairness analysis in notebook 06 |
# MAGIC
# MAGIC **Note on `sex` and `age_band`:** they are kept in the gold table so we can audit fairness later, but we will **exclude them from the model's inputs** in notebook 02.

# COMMAND ----------

silver = spark.table(SILVER)

total_bill = builtins.sum([F.greatest(F.col(c), F.lit(0.0)) for c in bill_cols])
total_paid = builtins.sum([F.col(c) for c in pay_amt_cols])
safe_limit = F.greatest(F.col("limit_bal"), F.lit(1.0))

features = (silver
    .withColumn("avg_bill", builtins.sum([F.col(c) for c in bill_cols]) / 6)
    .withColumn("avg_pay_amt", total_paid / 6)
    .withColumn("util_ratio", F.col("bill_amt1") / safe_limit)
    .withColumn("avg_util_ratio", F.col("avg_bill") / safe_limit)
    .withColumn("pay_to_bill_ratio", total_paid / F.greatest(total_bill, F.lit(1.0)))
    .withColumn("avg_delay", builtins.sum([F.col(c) for c in pay_cols]) / 6)
    .withColumn("max_delay", F.greatest(*[F.col(c) for c in pay_cols]))
    .withColumn("months_delayed", builtins.sum([F.when(F.col(c) > 0, 1).otherwise(0) for c in pay_cols]))
    .withColumn("bill_trend", (F.col("bill_amt1") - F.col("bill_amt6")) / safe_limit)
    .withColumn("age_band",
                F.when(F.col("age") < 30, "21-29")
                 .when(F.col("age") < 40, "30-39")
                 .when(F.col("age") < 50, "40-49")
                 .otherwise("50+"))
    .drop("_processed_at"))

# Stratified, reproducible split: rank rows randomly within each class, first 80% -> train
w = Window.partitionBy(TARGET).orderBy(F.rand(SPLIT_SEED))
gold_df = (features
           .withColumn("_pct", F.percent_rank().over(w))
           .withColumn("split", F.when(F.col("_pct") < TRAIN_FRACTION, "train").otherwise("test"))
           .drop("_pct"))

(gold_df.write
 .mode("overwrite")
 .option("overwriteSchema", "true")
 .saveAsTable(GOLD))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Verify the split is stratified (default rate should be about equal in train and test)

# COMMAND ----------

gold = spark.table(GOLD)
display(gold.groupBy("split")
        .agg(F.count("*").alias("rows"),
             F.round(F.avg(TARGET), 4).alias("default_rate"))
        .orderBy("split"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Make the gold table a feature table
# MAGIC In Unity Catalog, any Delta table with a **primary key** can be used as a feature table (for point-in-time lookups and for Model Serving feature lookup). We add the key and descriptions here.

# COMMAND ----------

spark.sql(f"ALTER TABLE {GOLD} ALTER COLUMN client_id SET NOT NULL")
try:
    spark.sql(f"ALTER TABLE {GOLD} ADD CONSTRAINT gold_features_pk PRIMARY KEY (client_id)")
except Exception as e:
    print(f"Primary key already exists or could not be added: {str(e)[:200]}")

spark.sql(f"COMMENT ON TABLE {BRONZE} IS 'Raw UCI credit card default data, column names standardised only'")
spark.sql(f"COMMENT ON TABLE {SILVER} IS 'Cleaned and typed client records, passed data quality checks'")
spark.sql(f"COMMENT ON TABLE {GOLD} IS 'Model-ready features with stratified train/test split (seed 42). Primary key: client_id'")
spark.sql(f"COMMENT ON COLUMN {GOLD}.split IS 'train or test, stratified on default_next_month'")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Delta time travel
# MAGIC Every write creates a new table version. We can audit changes and re-read old data, which is useful for reproducing a model's training data.

# COMMAND ----------

display(spark.sql(f"DESCRIBE HISTORY {GOLD}").select("version", "timestamp", "operation", "operationMetrics"))

# COMMAND ----------

# Read version 0 of the gold table (reproducibility demo)
v0 = spark.read.option("versionAsOf", 0).table(GOLD)
print(f"Rows in gold_features at version 0: {v0.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Summary

# COMMAND ----------

for t in [BRONZE, SILVER, GOLD]:
    print(f"{t:45s} rows={spark.table(t).count():>7,}  cols={len(spark.table(t).columns)}")

# COMMAND ----------

# MAGIC %md
# MAGIC **Next:** `train_model.py`, which trains a baseline logistic regression and a LightGBM model, tracks everything in MLflow, and registers the winner to Unity Catalog.
