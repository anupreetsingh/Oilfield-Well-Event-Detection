# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Train NPT cause taggers
# MAGIC
# MAGIC Parses the Volve daily drilling reports, trains the TF-IDF baseline and the DistilBERT
# MAGIC fine-tune on held-out wells, and saves the better model to the volume for `04_tag_npt`.
# MAGIC Put the "Daily Drilling Report - XML Version" folder at `<volume_root>/volve/ddr` and, once
# MAGIC reviewed, the hand labels at `<volume_root>/labels/npt_gold.csv`.

# COMMAND ----------

import os

dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "oilfield")
dbutils.widgets.text("volume_root", "/Volumes/main/oilfield/files")
dbutils.widgets.text("experiment_root", "/Shared/oilfield")

os.environ["OILFIELD_ROOT"] = os.path.dirname(os.getcwd())
v = dbutils.widgets.get("volume_root")

# COMMAND ----------

import pandas as pd

from oilfield.config import load_config
from oilfield.npt.train import load_activities, run

cfg = load_config("configs/npt.yaml", {
    "raw_dir": f"{v}/volve/ddr",
    "out_dir": f"{v}/npt",
    "gold_labels": f"{v}/labels/npt_gold.csv",
    "review_sample": f"{v}/labels/npt_review.csv",
    "model_dir": f"{v}/models/npt-cause",
    "reports_dir": f"{v}/reports",
    "experiment": f"{dbutils.widgets.get('experiment_root')}/npt-tagging",
})
results = run(cfg, ["tfidf", "distilbert"], refresh=True, export_review=True)

# COMMAND ----------

# Publish the parsed reports as Delta tables for SQL and Power BI.
catalog, schema = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
activities, reports = load_activities(cfg)
for name, df in [("gold_ddr_activities", activities), ("gold_ddr_daily_reports", reports)]:
    spark.createDataFrame(df).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{catalog}.{schema}.{name}")
display(pd.DataFrame({m: {"macro_f1": r["macro_f1"], "accuracy": r["accuracy"], "n": r["n"]} for m, r in results.items()}).T)
