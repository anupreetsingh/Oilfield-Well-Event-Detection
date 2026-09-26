# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · Draft daily NPT summaries
# MAGIC
# MAGIC For each wellbore that reported on `report_date`, the copilot agent queries the activity
# MAGIC tables through its SQL tools, cites the remarks it used, and the drafts land in
# MAGIC `gold_npt_daily_summaries`. The tables are small, so the agent works on a read-only SQLite
# MAGIC copy on the driver; ANTHROPIC_API_KEY comes from the `oilfield` secret scope.

# COMMAND ----------

import os

dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "oilfield")
dbutils.widgets.text("volume_root", "/Volumes/main/oilfield/files")
dbutils.widgets.text("report_date", "")

os.environ["OILFIELD_ROOT"] = os.path.dirname(os.getcwd())
v = dbutils.widgets.get("volume_root")
catalog, schema = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")

# COMMAND ----------

from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import text

from oilfield.config import load_config
from oilfield.copilot.cli import build, make_context, summarize

cfg = load_config("configs/copilot.yaml", {
    "npt_dir": f"{v}/npt",
    "db_url": "sqlite:////local_disk0/tmp/oilfield/npt.db",
    "summaries_dir": f"{v}/copilot/summaries",
})
build(cfg)
ctx = make_context(cfg)

report_date = dbutils.widgets.get("report_date")
with ctx.engine.connect() as con:
    if not report_date:
        report_date = con.execute(text("SELECT MAX(report_date) FROM daily_reports")).scalar()
    wellbores = [r[0] for r in con.execute(
        text("SELECT DISTINCT wellbore FROM daily_reports WHERE report_date = :d ORDER BY wellbore"), {"d": report_date})]

drafts = summarize(cfg, ctx, wellbores, report_date, offline=not os.environ.get("ANTHROPIC_API_KEY"))

# COMMAND ----------

rows = pd.DataFrame([
    {**d.__dict__, "cited": ",".join(d.cited), "unknown_citations": ",".join(d.unknown_citations),
     "created_at": datetime.now(timezone.utc)}
    for d in drafts
])
(spark.createDataFrame(rows).write.mode("overwrite")
    .option("replaceWhere", f"report_date = '{report_date}'")
    .saveAsTable(f"{catalog}.{schema}.gold_npt_daily_summaries"))
for d in drafts:
    displayHTML(f"<h3>{d.wellbore}</h3><pre>{d.markdown}</pre>")
