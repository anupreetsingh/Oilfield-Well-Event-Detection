"""Draft a wellbore's daily NPT summary with a LangChain agent on Claude.

Flow for one (wellbore, report date):

    prompt -> agent loop: Claude calls the SQL and remark-search tools as it needs them
           -> draft in Markdown, each fact citing the activity ID it came from, e.g. [F-12:20080214:03]
           -> citation check against the IDs the tools returned this run
           -> if any citation is unknown, one repair turn asking Claude to fix or drop it
           -> DraftSummary (Markdown + citation report)

``offline_summary`` builds the same summary from the same query functions with no LLM, for
CI, cron fallbacks and environments without an API key.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage

from oilfield.copilot.citations import check_citations
from oilfield.copilot.tools import CopilotContext, daily_report, hours_by_cause, make_tools, npt_events
from oilfield.npt.labeling import CAUSE_LABELS
from oilfield.npt.witsml import wellbore_slug

SYSTEM_PROMPT = """\
You draft the daily non-productive time (NPT) summary that drilling engineers read before the \
morning call. You are given a wellbore and a report date.

Use the tools to find that day's NPT events and hours by cause, the rig's own daily summary and \
forecast, and, when it helps explain an event, earlier remarks about the same problem on this \
wellbore. Then write the summary:

- one line with total NPT hours and their share of the reported day;
- NPT hours by cause;
- for each significant event: what happened, what was done, and whether it is resolved;
- anything likely to carry into the next day, based on the forecast and unresolved events.

Ground every statement in tool results and cite the activity remark behind it with its ID in \
square brackets, for example [F-12:20080214:03]. Only cite IDs a tool returned. Causes come from \
a text classifier; if an event's confidence is below 0.5, say its cause is uncertain rather than \
asserting it. If there was no NPT that day, say so in one line. Write Markdown, at most about \
250 words, with no preamble."""


class CopilotRefusal(RuntimeError):
    pass


@dataclass
class DraftSummary:
    wellbore: str
    report_date: str
    markdown: str
    cited: list[str] = field(default_factory=list)
    unknown_citations: list[str] = field(default_factory=list)
    tool_calls: int = 0
    mode: str = "agent"


def make_llm(llm_cfg: dict) -> ChatAnthropic:
    return ChatAnthropic(
        model=llm_cfg.get("model", "claude-opus-5-5"),
        max_tokens=llm_cfg.get("max_tokens", 16000),
        output_config={"effort": llm_cfg.get("effort", "medium")},
        # If a request is declined by a safety classifier, retry it server-side on Anthropic's
        # recommended fallback model instead of returning the refusal.
        betas=["server-side-fallback-2026-07-01"],
        model_kwargs={"fallbacks": "default"},
    )


def build_agent(ctx: CopilotContext, llm):
    return create_agent(llm, tools=make_tools(ctx), system_prompt=SYSTEM_PROMPT)


def _final_text(messages: list) -> str:
    final = messages[-1]
    if not isinstance(final, AIMessage):
        raise RuntimeError("agent finished without a reply")
    if final.response_metadata.get("stop_reason") == "refusal":
        raise CopilotRefusal(f"request declined: {final.response_metadata.get('stop_details')}")
    return final.text.strip()


def _count_tool_calls(messages: list) -> int:
    return sum(len(getattr(m, "tool_calls", None) or []) for m in messages)


def draft_daily_summary(agent, ctx: CopilotContext, wellbore: str, report_date: str) -> DraftSummary:
    wb = wellbore_slug(wellbore)
    ctx.seen_ids.clear()
    state = agent.invoke({"messages": [HumanMessage(f"Draft the daily NPT summary for wellbore {wb} on {report_date}.")]})
    text = _final_text(state["messages"])
    check = check_citations(text, ctx.seen_ids)
    if not check.ok:
        fix = HumanMessage(
            "These citations were not returned by any tool in this conversation: "
            + ", ".join(check.unknown)
            + ". Look them up or remove them, and return the full corrected summary."
        )
        state = agent.invoke({"messages": [*state["messages"], fix]})
        text = _final_text(state["messages"])
        check = check_citations(text, ctx.seen_ids)
    return DraftSummary(wb, report_date, text, check.cited, check.unknown, _count_tool_calls(state["messages"]))


def offline_summary(ctx: CopilotContext, wellbore: str, report_date: str) -> DraftSummary:
    """Template summary from the same queries, without an LLM."""
    wb = wellbore_slug(wellbore)
    ctx.seen_ids.clear()
    events = npt_events(ctx, wb, report_date)
    totals = hours_by_cause(ctx, wb, report_date, report_date)
    report = daily_report(ctx, wb, report_date) or {}

    lines = [f"**{wb} - {report_date}**", ""]
    if not events:
        lines.append(f"No NPT reported ({totals['reported_hours']:.1f} h reported).")
    else:
        share = f" ({totals['npt_share']:.0%} of {totals['reported_hours']:.1f} h reported)" if totals["npt_share"] else ""
        lines.append(f"NPT: **{totals['npt_hours']:.1f} h**{share}.")
        lines += ["", "| cause | hours | events |", "|---|---|---|"]
        lines += [f"| {CAUSE_LABELS.get(c['cause'], c['cause'])} | {c['npt_hours']:.1f} | {c['events']} |" for c in totals["by_cause"]]
        lines += ["", "Events:"]
        for e in events:
            cause = CAUSE_LABELS.get(e["cause"], e["cause"])
            if e["confidence"] is not None and e["confidence"] < 0.5:
                cause += " (uncertain)"
            lines.append(f"- {e['start_time'][-5:]}-{e['end_time'][-5:]}, {e['hours']:.1f} h, {cause}: "
                         f"{e['comments']} [{e['activity_id']}]")
    if report.get("forecast_24h"):
        lines += ["", f"Next 24 h: {report['forecast_24h']}"]
    text = "\n".join(lines)
    check = check_citations(text, ctx.seen_ids)
    return DraftSummary(wb, report_date, text, check.cited, check.unknown, 0, mode="offline")
