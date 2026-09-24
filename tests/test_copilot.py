import json

import pandas as pd
import pytest

pytest.importorskip("langchain")

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402

from oilfield.copilot.agent import build_agent, draft_daily_summary, offline_summary  # noqa: E402
from oilfield.copilot.citations import check_citations, cited_ids  # noqa: E402
from oilfield.copilot.retrieval import RemarkIndex  # noqa: E402
from oilfield.copilot.store import build_sqlite, connect  # noqa: E402
from oilfield.copilot.tools import CopilotContext, UnsafeQuery, check_select, hours_by_cause, make_tools  # noqa: E402

DAY = "2014-03-02"


@pytest.fixture()
def ctx(tmp_path):
    t = lambda h: pd.Timestamp(f"{DAY} {h}")  # noqa: E731
    activities = pd.DataFrame([
        {"activity_id": "S-1:20140302:01", "wellbore": "S-1", "well": "NO 15/9-S-1", "report_date": DAY, "start_time": t("00:00"),
         "end_time": t("10:00"), "duration_h": 10.0, "md_m": 2400, "phase": "drilling", "code": "drilling -- drill",
         "state": "ok", "is_npt": False, "comments": "Drilled 8 1/2\" hole from 2300 m to 2400 m."},
        {"activity_id": "S-1:20140302:02", "wellbore": "S-1", "well": "NO 15/9-S-1", "report_date": DAY, "start_time": t("10:00"),
         "end_time": t("13:00"), "duration_h": 3.0, "md_m": 2400, "phase": "drilling", "code": "interruption -- repair",
         "state": "fail", "is_npt": True, "comments": "Repaired top drive, hydraulic leak on IBOP actuator."},
        {"activity_id": "S-1:20140302:03", "wellbore": "S-1", "well": "NO 15/9-S-1", "report_date": DAY, "start_time": t("13:00"),
         "end_time": pd.Timestamp("2014-03-03 00:00"), "duration_h": 11.0, "md_m": 2410, "phase": "drilling",
         "code": "interruption -- other", "state": "fail", "is_npt": True, "comments": "Pipe stuck at 2410 m, jarred free."},
    ])
    npt = activities[activities.is_npt].assign(cause=["rig_repair", "stuck_pipe"], confidence=[0.93, 0.41])
    reports = pd.DataFrame([{"wellbore": "S-1", "well": "NO 15/9-S-1", "report_date": DAY, "md_m": 2410,
                             "summary_24h": "Drilled to 2410 m.", "forecast_24h": "POOH to change BHA."}])
    path = build_sqlite(activities, npt, reports, tmp_path / "npt.db")
    engine = connect(f"sqlite:///{path}")
    return CopilotContext(engine, RemarkIndex.from_engine(engine))


def test_hours_by_cause_and_share(ctx):
    totals = hours_by_cause(ctx, "NO 15/9-S-1", DAY, DAY)
    assert totals["npt_hours"] == 14.0 and totals["reported_hours"] == 24.0
    assert totals["npt_share"] == pytest.approx(0.583, abs=1e-3)
    assert [c["cause"] for c in totals["by_cause"]] == ["stuck_pipe", "rig_repair"]


@pytest.mark.parametrize("sql", [
    "DROP TABLE activities",
    "SELECT * FROM activities; DELETE FROM activities",
    "SELECT * FROM sqlite_master",
    "WITH x AS (SELECT 1) INSERT INTO activities SELECT * FROM x",
    "PRAGMA table_info(activities)",
])
def test_unsafe_sql_is_rejected(sql):
    with pytest.raises(UnsafeQuery):
        check_select(sql)


def test_sql_tool_runs_selects_and_ctes_and_reports_errors(ctx):
    tools = {t.name: t for t in make_tools(ctx)}
    rows = json.loads(tools["query_activity_tables"].invoke(
        {"sql": "WITH n AS (SELECT * FROM npt_events) SELECT activity_id, cause FROM n ORDER BY activity_id -- comment"}))
    assert [r["cause"] for r in rows] == ["rig_repair", "stuck_pipe"]
    assert "S-1:20140302:03" in ctx.seen_ids
    err = json.loads(tools["query_activity_tables"].invoke({"sql": "DELETE FROM npt_events"}))
    assert "error" in err


def test_database_is_opened_read_only(ctx):
    from sqlalchemy import text
    with pytest.raises(Exception, match="readonly"):
        with ctx.engine.begin() as con:
            con.execute(text("DELETE FROM npt_events"))


def test_remark_search_filters_and_registers_ids(ctx):
    tools = {t.name: t for t in make_tools(ctx)}
    hits = json.loads(tools["search_remarks"].invoke({"query": "top drive hydraulic leak", "wellbore": "S-1", "npt_only": True}))
    assert hits[0]["activity_id"] == "S-1:20140302:02"
    assert json.loads(tools["search_remarks"].invoke({"query": "top drive", "wellbore": "F-12"})) == []


def test_citation_check():
    md = "Top drive repair [S-1:20140302:02]; stuck pipe [S-1:20140302:03, S-1:20140302:09]. [note]"
    assert cited_ids(md) == ["S-1:20140302:02", "S-1:20140302:03", "S-1:20140302:09"]
    check = check_citations(md, {"S-1:20140302:02", "S-1:20140302:03"})
    assert check.unknown == ["S-1:20140302:09"] and not check.ok


def test_offline_summary_cites_every_event_and_flags_low_confidence(ctx):
    draft = offline_summary(ctx, "S-1", DAY)
    assert "14.0 h" in draft.markdown and "58%" in draft.markdown
    assert set(draft.cited) == {"S-1:20140302:02", "S-1:20140302:03"} and not draft.unknown_citations
    assert "Stuck pipe (uncertain)" in draft.markdown
    assert "POOH to change BHA." in draft.markdown


class ToolCallingFake(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def test_agent_loop_calls_tools_and_repairs_bad_citations(ctx):
    script = iter([
        AIMessage(content="", tool_calls=[{"name": "get_npt_events", "args": {"wellbore": "S-1", "report_date": DAY}, "id": "c1"}]),
        AIMessage(content="14 h NPT: top drive [S-1:20140302:02], stuck pipe [S-1:20140302:07]."),
        AIMessage(content="14 h NPT: top drive [S-1:20140302:02], stuck pipe [S-1:20140302:03]."),
    ])
    agent = build_agent(ctx, ToolCallingFake(messages=script))
    draft = draft_daily_summary(agent, ctx, "NO 15/9-S-1", DAY)
    assert draft.tool_calls == 1
    assert draft.cited == ["S-1:20140302:02", "S-1:20140302:03"] and draft.unknown_citations == []
