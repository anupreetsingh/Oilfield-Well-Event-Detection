"""NPT copilot command line.

    python -m oilfield.copilot.cli build                                # SQLite store from data/npt/*.parquet
    python -m oilfield.copilot.cli summary --wellbore F-12 --date 2008-02-14
    python -m oilfield.copilot.cli daily --date 2008-02-14              # every wellbore reporting that day
    add --offline to draft from templates without calling Claude
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from oilfield.config import load_config, parse_overrides, resolve_path
from oilfield.copilot.retrieval import RemarkIndex
from oilfield.copilot.store import build_sqlite, connect
from oilfield.copilot.tools import CopilotContext

log = logging.getLogger(__name__)


def _db_url(cfg: dict) -> str:
    url = cfg["db_url"]
    if url.startswith("sqlite:///") and not Path(url.removeprefix("sqlite:///")).is_absolute():
        url = f"sqlite:///{resolve_path(url.removeprefix('sqlite:///'))}"
    return url


def build(cfg: dict) -> Path:
    out = resolve_path(cfg["npt_dir"])
    path = build_sqlite(
        pd.read_parquet(out / "activities.parquet"),
        pd.read_parquet(out / "npt_events.parquet"),
        pd.read_parquet(out / "daily_reports.parquet"),
        resolve_path(_db_url(cfg).removeprefix("sqlite:///")),
    )
    log.info("built %s", path)
    return path


def make_context(cfg: dict) -> CopilotContext:
    engine = connect(_db_url(cfg))
    return CopilotContext(engine, RemarkIndex.from_engine(engine), cfg["tools"]["max_rows"], cfg["tools"]["search_k"])


def summarize(cfg: dict, ctx: CopilotContext, wellbores: list[str], report_date: str, offline: bool) -> list:
    from oilfield.copilot.agent import build_agent, draft_daily_summary, make_llm, offline_summary

    agent = None if offline else build_agent(ctx, make_llm(cfg["llm"]))
    out_dir = resolve_path(cfg["summaries_dir"]) / report_date
    out_dir.mkdir(parents=True, exist_ok=True)
    drafts = []
    for wb in wellbores:
        draft = offline_summary(ctx, wb, report_date) if offline else draft_daily_summary(agent, ctx, wb, report_date)
        (out_dir / f"{draft.wellbore}.md").write_text(draft.markdown + "\n")
        (out_dir / f"{draft.wellbore}.json").write_text(json.dumps(draft.__dict__, indent=2))
        if draft.unknown_citations:
            log.warning("%s: unverified citations %s", draft.wellbore, draft.unknown_citations)
        drafts.append(draft)
    return drafts


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["build", "summary", "daily"])
    parser.add_argument("--config", default="configs/copilot.yaml")
    parser.add_argument("--wellbore")
    parser.add_argument("--date", help="report date, YYYY-MM-DD")
    parser.add_argument("--offline", action="store_true", help="template summary, no LLM call")
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config, parse_overrides(args.set))

    if args.command == "build":
        build(cfg)
        return
    if not args.date:
        parser.error("--date is required")
    ctx = make_context(cfg)
    if args.command == "summary":
        if not args.wellbore:
            parser.error("--wellbore is required")
        wellbores = [args.wellbore]
    else:
        with ctx.engine.connect() as con:
            wellbores = [r[0] for r in con.execute(
                text("SELECT DISTINCT wellbore FROM daily_reports WHERE report_date = :d ORDER BY wellbore"), {"d": args.date})]
    for draft in summarize(cfg, ctx, wellbores, args.date, args.offline):
        print(f"\n{draft.markdown}\n\n[{draft.mode}; {draft.tool_calls} tool calls; cited {len(draft.cited)}; "
              f"unverified {draft.unknown_citations or 'none'}]")


if __name__ == "__main__":
    main()
