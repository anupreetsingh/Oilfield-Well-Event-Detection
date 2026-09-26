# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Tag NPT causes
# MAGIC
# MAGIC Applies the saved cause model to every NPT activity and publishes `gold_npt_events`, which
# MAGIC both the copilot and the Power BI report read.

# COMMAND ----------

import os

dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "oilfield")
dbutils.widgets.text("volume_root", "/Volumes/main/oilfield/files")

os.environ["OILFIELD_ROOT"] = os.path.dirname(os.getcwd())
v = dbutils.widgets.get("volume_root")
catalog, schema = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")

# COMMAND ----------

from pathlib import Path

import pandas as pd

from oilfield.npt.labeling import load_gold
from oilfield.npt.tag import tag_npt

activities = pd.read_parquet(f"{v}/npt/activities.parquet")
events = tag_npt(activities, Path(f"{v}/models/npt-cause"), load_gold(f"{v}/labels/npt_gold.csv"))
events.to_parquet(f"{v}/npt/npt_events.parquet", index=False)
spark.createDataFrame(events).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{catalog}.{schema}.gold_npt_events")

display(events.groupby("cause")["duration_h"].sum().sort_values(ascending=False).reset_index())
