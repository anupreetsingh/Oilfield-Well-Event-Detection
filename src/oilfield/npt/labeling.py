"""NPT cause taxonomy and the rules that give each NPT activity a starting ("silver") label.

Volve's activity codes say *that* time was lost (``interruption -- ...``) and sometimes roughly
why (``repair``, ``wait``, ``fish``, ``well control``), but not the causes engineers track:
stuck pipe hides under ``other``, and ``wait`` mixes weather with late equipment. The rules
below combine the code with the remark text, first match wins. They are a bootstrap for
training, not ground truth: hand-reviewed labels in ``npt_gold.csv`` override them and are
what evaluation should use (see ``export_review_sample``).
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

CAUSES = [
    "stuck_pipe",
    "fishing",
    "well_control",
    "downhole_tool_failure",
    "rig_repair",
    "wait_on_weather",
    "wait_on_equipment",
    "other",
]
CAUSE_LABELS = {
    "stuck_pipe": "Stuck pipe",
    "fishing": "Fishing",
    "well_control": "Well control / losses",
    "downhole_tool_failure": "Downhole tool failure",
    "rig_repair": "Rig repair",
    "wait_on_weather": "Waiting on weather",
    "wait_on_equipment": "Waiting on equipment / materials",
    "other": "Other",
}


def _rx(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.I)


FISHING = _rx(r"\bfish(ing|ed)?\b|\bjunk\b|overshot|\bspear\b|junk basket|magnet run|retriev\w* (the )?(lost|dropped)")
STUCK = _rx(r"\bstuck\b|pack(ed|ing)?[- ]?off|overpull|\bjar(red|ring|s)?\b|differential(ly)? stuck|work(ed|ing)? (the )?(pipe|string) free"
            r"|free ?point|back(ed)?[- ]?off|unable to (move|rotate|pull) (the )?(pipe|string)")
WELL_CONTROL = _rx(r"\bkick\b|influx|well control|shut[- ]in (the )?well|\bgain(s|ed)? in (the )?pit|lost circulation|\blosses\b"
                   r"|\bloss(es)? (of|to) (the )?formation|bullhead|kill mud|circulat\w* out (the )?(kick|gas)")
DOWNHOLE_TOOL = _rx(r"\b(mwd|lwd|rss|motor|mud motor|powerdrive|autotrak|pulser|telemetry|downhole tool|bha tool)\b.{0,60}"
                    r"\b(fail\w*|lost signal|no signal|malfunction\w*|not working|stopped working|no pulses?)\b"
                    r"|\b(fail\w*|malfunction\w*|lost signal|no signal)\b.{0,40}\b(mwd|lwd|rss|motor|pulser|telemetry)\b")
# Weather words are anchored to sea-state phrasing: "heave compensator" and "swell packer" are rig kit.
WEATHER = _rx(r"\bw\.?o\.?w\b|weather|high winds?|\bwinds? (speed|gust\w*|above|exceed\w*|of|up to)|heave (above|exceed\w*|of|up to)"
              r"|wave heights?|\bwaves? (above|exceed\w*|of|up to)|\bswell\b(?! packer)|\bstorm\b|sea state")
WAITING = _rx(r"\bwait(ed|ing)? (on|for)\b|\bw\.?o\.?(e|m|o)\b|awaiting")
REPAIR = _rx(r"\brepair\w*|top ?drive|draw ?works|iron roughneck|mud pump|\bbop\b|hydraulic (leak|hose)|troubleshoot\w*"
             r"|change(d)? out (the )?(saver sub|swivel|wash ?pipe)|maintenance|electrical fault|\bfault\w* on\b")

RULES: list[tuple[str, str, object]] = [
    # (cause, rule name, predicate(code_minor, text))
    ("fishing", "code:fish", lambda code, text: code == "fish"),
    ("fishing", "text:fishing", lambda code, text: bool(FISHING.search(text))),
    ("stuck_pipe", "text:stuck", lambda code, text: bool(STUCK.search(text))),
    ("well_control", "code:well control", lambda code, text: code == "well control"),
    ("well_control", "text:well control", lambda code, text: bool(WELL_CONTROL.search(text))),
    ("downhole_tool_failure", "text:downhole tool", lambda code, text: bool(DOWNHOLE_TOOL.search(text))),
    ("wait_on_weather", "text:weather", lambda code, text: bool(WEATHER.search(text))),
    ("wait_on_equipment", "code:wait", lambda code, text: code == "wait"),
    ("wait_on_equipment", "text:waiting", lambda code, text: bool(WAITING.search(text))),
    ("rig_repair", "code:repair", lambda code, text: code in ("repair", "maintain")),
    ("rig_repair", "text:repair", lambda code, text: bool(REPAIR.search(text))),
]


def label_activity(code_minor: str | None, comments: str | None) -> tuple[str, str]:
    code, text = (code_minor or "").strip().lower(), comments or ""
    for cause, rule, predicate in RULES:
        if predicate(code, text):
            return cause, rule
    return "other", "default"


def apply_silver_labels(npt: pd.DataFrame) -> pd.DataFrame:
    labelled = [label_activity(c, t) for c, t in zip(npt["code_minor"], npt["comments"])]
    return npt.assign(silver_cause=[c for c, _ in labelled], silver_rule=[r for _, r in labelled])


def load_gold(path: str | Path) -> pd.DataFrame | None:
    path = Path(path)
    if not path.exists():
        return None
    gold = pd.read_csv(path, dtype=str).dropna(subset=["activity_id", "cause"])
    unknown = set(gold["cause"]) - set(CAUSES)
    if unknown:
        raise ValueError(f"unknown causes in {path}: {sorted(unknown)}")
    return gold[["activity_id", "cause"]]


def export_review_sample(npt: pd.DataFrame, path: str | Path, per_cause: int = 60, seed: int = 0) -> pd.DataFrame:
    """Write a stratified sample for hand review; fill the ``cause`` column and save as npt_gold.csv."""
    sample = (
        npt.groupby("silver_cause", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), per_cause), random_state=seed))
        [["activity_id", "wellbore", "report_date", "code", "duration_h", "comments", "silver_cause", "silver_rule"]]
        .assign(cause="")
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(path, index=False)
    return sample
