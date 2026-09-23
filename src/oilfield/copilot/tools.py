"""Tools the copilot can call, and the plain query functions behind them.

Every row a tool returns that carries an ``activity_id`` is recorded in the context's
``seen_ids``; the drafted summary may only cite IDs from that set (see ``citations``).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import pandas as pd
from langchain_core.tools import tool
from sqlalchemy import Engine, text

from oilfield.copilot.retrieval import RemarkIndex
from oilfield.copilot.store import TABLES
from oilfield.npt.witsml import wellbore_slug


@dataclass
class CopilotContext:
    engine: Engine
    index: RemarkIndex
    max_rows: int = 200
    search_k: int = 6
    seen_ids: set[str] = field(default_factory=set)

    def query(self, sql: str, **params) -> list[dict]:
        with self.engine.connect() as con:
            rows = pd.read_sql(text(sql), con, params=params).head(self.max_rows).to_dict("records")
        return self.remember(rows)

    def remember(self, rows: list[dict]) -> list[dict]:
        self.seen_ids.update(r["activity_id"] for r in rows if r.get("activity_id"))
        return rows


# -- query functions (shared by the tools and the offline summary) -----------------------------

def npt_events(ctx: CopilotContext, wellbore: str, report_date: str) -> list[dict]:
    return ctx.query(
        "SELECT activity_id, start_time, end_time, ROUND(duration_h, 2) AS hours, md_m, code, cause, confidence, comments "
        "FROM npt_events WHERE wellbore = :wb AND report_date = :d ORDER BY start_time",
        wb=wellbore_slug(wellbore), d=report_date,
    )


def hours_by_cause(ctx: CopilotContext, wellbore: str, start_date: str, end_date: str) -> dict:
    wb = wellbore_slug(wellbore)
    causes = ctx.query(
        "SELECT cause, ROUND(SUM(duration_h), 2) AS npt_hours, COUNT(*) AS events FROM npt_events "
        "WHERE wellbore = :wb AND report_date BETWEEN :a AND :b GROUP BY cause ORDER BY npt_hours DESC",
        wb=wb, a=start_date, b=end_date,
    )
    total = ctx.query(
        "SELECT ROUND(SUM(duration_h), 2) AS reported_hours FROM activities "
        "WHERE wellbore = :wb AND report_date BETWEEN :a AND :b", wb=wb, a=start_date, b=end_date,
    )
    reported = total[0]["reported_hours"] or 0.0
    npt = round(sum(c["npt_hours"] for c in causes), 2)
    return {"wellbore": wb, "start_date": start_date, "end_date": end_date, "reported_hours": reported,
            "npt_hours": npt, "npt_share": round(npt / reported, 3) if reported else None, "by_cause": causes}


def daily_report(ctx: CopilotContext, wellbore: str, report_date: str) -> dict | None:
    rows = ctx.query(
        "SELECT wellbore, report_date, md_m, summary_24h, forecast_24h FROM daily_reports "
        "WHERE wellbore = :wb AND report_date = :d", wb=wellbore_slug(wellbore), d=report_date,
    )
    return rows[0] if rows else None


# -- read-only SQL -------------------------------------------------------------------------------

class UnsafeQuery(ValueError):
    pass


_COMMENTS = re.compile(r"--[^\n]*|/\*.*?\*/", re.S)
_WRITES = re.compile(r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|pragma|vacuum|grant|revoke|"
                     r"truncate|merge|copy|reindex|analyze)\b", re.I)
_TABLE_REFS = re.compile(r"\b(?:from|join)\s+([A-Za-z_][\w.]*)", re.I)
_CTE_NAMES = re.compile(r"(?:\bwith|,)\s*([A-Za-z_]\w*)\s+as\s*\(", re.I)


def check_select(sql: str, allowed: tuple[str, ...] = TABLES) -> str:
    """Accept a single SELECT (or WITH ... SELECT) over the activity tables, or raise UnsafeQuery."""
    s = _COMMENTS.sub(" ", sql).strip().rstrip(";").strip()
    if ";" in s:
        raise UnsafeQuery("only one statement is allowed")
    if not re.match(r"(select|with)\b", s, re.I):
        raise UnsafeQuery("only SELECT queries are allowed")
    if _WRITES.search(s):
        raise UnsafeQuery("the query contains a write or admin keyword")
    ctes = {n.lower() for n in _CTE_NAMES.findall(s)}
    unknown = {t for t in _TABLE_REFS.findall(s) if t.lower() not in ctes and t.lower() not in allowed}
    if unknown:
        raise UnsafeQuery(f"unknown tables {sorted(unknown)}; available: {', '.join(allowed)}")
    return s


def run_select(ctx: CopilotContext, sql: str) -> list[dict]:
    safe = check_select(sql)
    return ctx.query(f"SELECT * FROM ({safe}) AS q LIMIT {int(ctx.max_rows)}")


# -- LangChain tools -----------------------------------------------------------------------------

def _json(obj) -> str:
    return json.dumps(obj, default=str, ensure_ascii=False)


def make_tools(ctx: CopilotContext) -> list:
    @tool
    def get_npt_events(wellbore: str, report_date: str) -> str:
        """List the NPT (non-productive time) activities logged for one wellbore on one report date.

        Returns each activity's ID, start/end time, hours, measured depth, activity code, the
        cause predicted from its remark with the model's confidence (0-1), and the remark text.
        wellbore: e.g. "F-12" or "NO 15/9-F-12". report_date: YYYY-MM-DD.
        """
        return _json(npt_events(ctx, wellbore, report_date))

    @tool
    def get_npt_hours_by_cause(wellbore: str, start_date: str, end_date: str) -> str:
        """Total NPT hours and event counts per cause for a wellbore over an inclusive date range,
        plus the total reported hours and the NPT share of them. Dates are YYYY-MM-DD."""
        return _json(hours_by_cause(ctx, wellbore, start_date, end_date))

    @tool
    def get_daily_report(wellbore: str, report_date: str) -> str:
        """The rig's own 24-hour summary, the forecast for the next 24 hours, and the depth reached,
        for one wellbore and report date (YYYY-MM-DD)."""
        return _json(daily_report(ctx, wellbore, report_date))

    @tool
    def search_remarks(query: str, wellbore: str | None = None, start_date: str | None = None,
                       end_date: str | None = None, npt_only: bool = False) -> str:
        """Keyword search over activity remarks, e.g. to find earlier occurrences of the same
        problem or what was done about it. Optionally restrict to a wellbore, a date range
        (YYYY-MM-DD, inclusive) and NPT activities only. Returns the best matches with IDs."""
        hits = ctx.index.search(query, wellbore_slug(wellbore) if wellbore else None, start_date, end_date,
                                npt_only, ctx.search_k)
        return _json(ctx.remember(hits))

    @tool
    def query_activity_tables(sql: str) -> str:
        """Run one read-only SQL SELECT against the tables activities, npt_events and daily_reports
        (SQLite dialect; dates are 'YYYY-MM-DD' text, times 'YYYY-MM-DD HH:MM'). Use it for
        questions the other tools don't cover. Results are capped at a few hundred rows.

        activities(activity_id, wellbore, well, report_date, start_time, end_time, duration_h, md_m,
                   phase, code, state, is_npt, comments)
        npt_events(activity_id, wellbore, well, report_date, start_time, end_time, duration_h, md_m,
                   code, cause, confidence, comments)
        daily_reports(wellbore, well, report_date, md_m, summary_24h, forecast_24h)
        """
        try:
            return _json(run_select(ctx, sql))
        except UnsafeQuery as e:
            return _json({"error": str(e)})

    return [get_npt_events, get_npt_hours_by_cause, get_daily_report, search_remarks, query_activity_tables]
