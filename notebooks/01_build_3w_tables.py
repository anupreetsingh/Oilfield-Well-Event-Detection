# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Build the 3W Delta tables
# MAGIC
# MAGIC Raw 3W runs in a UC volume → `silver_3w_quality`, `silver_3w_instances`, `silver_3w_readings`
# MAGIC → `gold_3w_windows`. Upload the dataset first: `scripts/download_3w.sh` locally, then
# MAGIC `databricks fs cp -r data/raw/3w/dataset dbfs:<volume_root>/3w/dataset`.

# COMMAND ----------

import os

dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "oilfield")
dbutils.widgets.text("volume_root", "/Volumes/main/oilfield/files")

catalog, schema = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
volume_root = dbutils.widgets.get("volume_root")
os.environ["OILFIELD_ROOT"] = os.path.dirname(os.getcwd())  # bundle files: configs/ sits next to notebooks/

# COMMAND ----------

from oilfield.config import load_config
from oilfield.spark import TableStore
from oilfield.threew.pipeline import build_gold, build_silver, summarize

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
cfg = load_config("configs/threew.yaml", {
    "raw_dir": f"{volume_root}/3w/dataset",
    "tables.catalog": catalog,
    "tables.schema": schema,
})
store = TableStore.from_config(spark, cfg["tables"])

build_silver(spark, store, cfg)
display(summarize(store))

# COMMAND ----------

build_gold(spark, store, cfg)
spark.sql(f"OPTIMIZE {store.location('gold_3w_windows')} ZORDER BY (instance_id, window_end)")
display(spark.sql(f"SELECT label, count(*) AS windows FROM {store.location('gold_3w_windows')} GROUP BY label ORDER BY label"))
