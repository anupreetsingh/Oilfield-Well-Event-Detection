# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Train well-event classifiers
# MAGIC
# MAGIC Cross-validation over held-out wells for the random-forest baseline, gradient boosting, the
# MAGIC 1D-CNN and their ensemble, tracked in MLflow under `<experiment_root>/well-events`.

# COMMAND ----------

import os

dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "oilfield")
dbutils.widgets.text("volume_root", "/Volumes/main/oilfield/files")
dbutils.widgets.text("experiment_root", "/Shared/oilfield")
dbutils.widgets.text("models", "rf hgb cnn ensemble")

os.environ["OILFIELD_ROOT"] = os.path.dirname(os.getcwd())
volume_root = dbutils.widgets.get("volume_root")

# COMMAND ----------

from oilfield.config import load_config
from oilfield.events.train import run

cfg = load_config("configs/events.yaml", {
    "experiment": f"{dbutils.widgets.get('experiment_root')}/well-events",
    "reports_dir": f"{volume_root}/reports",
    "predictions_path": f"{volume_root}/events/oof_predictions.parquet",
    "pipeline_overrides": {"tables.catalog": dbutils.widgets.get("catalog"), "tables.schema": dbutils.widgets.get("schema")},
})
results = run(cfg, dbutils.widgets.get("models").split(), spark=spark)

# COMMAND ----------

import pandas as pd

display(pd.DataFrame({m: {"macro_f1": r["macro_f1"], **{k: v for k, v in r["alerts"].items() if not isinstance(v, dict)}}
                      for m, r in results.items()}).T.reset_index(names="model"))
