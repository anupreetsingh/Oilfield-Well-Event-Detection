"""The activity tables the copilot queries, and read-only connections to them.

Three tables, keyed by ``wellbore`` (slug such as ``F-12``) and ``report_date`` (YYYY-MM-DD):

- ``activities``     every reported activity, productive or not, with its remark;
- ``npt_events``     the NPT activities with the tagged cause and model confidence;
- ``daily_reports``  the 24 h summary and forecast from each report.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine, create_engine

TABLES = ("activities", "npt_events", "daily_reports")

ACTIVITY_COLS = ["activity_id", "wellbore", "well", "report_date", "start_time", "end_time", "duration_h", "md_m",
                 "phase", "code", "state", "is_npt", "comments"]
NPT_COLS = ["activity_id", "wellbore", "well", "report_date", "start_time", "end_time", "duration_h", "md_m", "code",
            "cause", "confidence", "comments"]
REPORT_COLS = ["wellbore", "well", "report_date", "md_m", "summary_24h", "forecast_24h"]


def _as_text_times(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "report_date" in df:
        df["report_date"] = pd.to_datetime(df["report_date"]).dt.strftime("%Y-%m-%d")
    for col in ("start_time", "end_time"):
        if col in df:
            df[col] = pd.to_datetime(df[col]).dt.strftime("%Y-%m-%d %H:%M")
    return df


def build_sqlite(activities: pd.DataFrame, npt_events: pd.DataFrame, reports: pd.DataFrame, db_path: str | Path) -> Path:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.unlink(missing_ok=True)
    with sqlite3.connect(db_path) as con:
        _as_text_times(activities[ACTIVITY_COLS]).assign(is_npt=lambda d: d["is_npt"].astype(int)).to_sql("activities", con, index=False)
        _as_text_times(npt_events[NPT_COLS]).to_sql("npt_events", con, index=False)
        _as_text_times(reports[REPORT_COLS]).to_sql("daily_reports", con, index=False)
        for table in TABLES:
            con.execute(f"CREATE INDEX idx_{table}_day ON {table} (wellbore, report_date)")
    return db_path


def connect(db_url: str) -> Engine:
    """SQLAlchemy engine; SQLite files are opened read-only so no tool call can modify them."""
    if db_url.startswith("sqlite:///"):
        path = Path(db_url.removeprefix("sqlite:///")).resolve()
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; build it with `python -m oilfield.copilot.cli build`")
        return create_engine("sqlite://", creator=lambda: sqlite3.connect(f"file:{path}?mode=ro", uri=True))
    return create_engine(db_url)
