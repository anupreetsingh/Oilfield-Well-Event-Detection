"""Parse WITSML 1.4 ``drillReport`` XML (the Norwegian DDR format Volve's reports use).

Each file holds one 24-hour report for one wellbore: a status block (``sum24Hr`` summary,
``forecast24Hr``) and a list of ``activity`` elements, each with start/end times, measured
depth, phase, a ``proprietaryCode`` such as ``"interruption -- repair"``, and the free-text
``comments`` the rig wrote. Those comments are the "remarks" the NPT models read.

Times keep the rig's local wall clock (the offset in the file is dropped after conversion), so
a report date is the calendar day the crew worked.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

NPT_MAJOR_CODE = "interruption"


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child(el: ET.Element, name: str) -> ET.Element | None:
    for c in el:
        if _local(c.tag) == name:
            return c
    return None


def _text(el: ET.Element | None, *path: str) -> str | None:
    for name in path:
        if el is None:
            return None
        el = _child(el, name)
    if el is None or el.text is None:
        return None
    return " ".join(el.text.split()) or None


def _time(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    ts = pd.Timestamp(value)
    return ts.tz_localize(None) if ts.tzinfo else ts


def _float(value: str | None) -> float | None:
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def wellbore_slug(name: str | None) -> str:
    """'NO 15/9-F-12' -> 'F-12', 'NO 15/9-F-1 C' -> 'F-1C', '15/9-19 BT2' -> '19BT2'."""
    if not name:
        return "UNKNOWN"
    s = re.sub(r"^NO\s+", "", name.strip(), flags=re.I)
    s = re.sub(r"^\d+/\d+-", "", s)
    return re.sub(r"\s+", "", s).upper()


def parse_ddr_file(path: str | Path) -> tuple[list[dict], list[dict]]:
    root = ET.parse(path).getroot()
    reports = [root] if _local(root.tag) == "drillReport" else [el for el in root if _local(el.tag) == "drillReport"]
    activities, summaries = [], []
    for rep in reports:
        well, wellbore = _text(rep, "nameWell"), _text(rep, "nameWellbore")
        start = _time(_text(rep, "dTimStart"))
        report_date = start.date() if start is not None else None
        slug = wellbore_slug(wellbore)
        summaries.append({
            "wellbore": slug,
            "well": well,
            "wellbore_name": wellbore,
            "report_date": report_date,
            "md_m": _float(_text(rep, "statusInfo", "md")),
            "summary_24h": _text(rep, "statusInfo", "sum24Hr"),
            "forecast_24h": _text(rep, "statusInfo", "forecast24Hr"),
            "source_file": Path(path).name,
        })
        seq = 0
        for act in rep:
            if _local(act.tag) != "activity":
                continue
            seq += 1
            t0, t1 = _time(_text(act, "dTimStart")), _time(_text(act, "dTimEnd"))
            code = (_text(act, "proprietaryCode") or "").lower()
            major, _, minor = (part.strip() for part in code.partition("--"))
            activities.append({
                "activity_id": f"{slug}:{report_date:%Y%m%d}:{seq:02d}" if report_date else f"{slug}:?:{seq:02d}",
                "wellbore": slug,
                "well": well,
                "wellbore_name": wellbore,
                "report_date": report_date,
                "start_time": t0,
                "end_time": t1,
                "duration_h": (t1 - t0).total_seconds() / 3600 if t0 is not None and t1 is not None else None,
                "md_m": _float(_text(act, "md")),
                "phase": _text(act, "phase"),
                "code": code or None,
                "code_major": major or None,
                "code_minor": minor or None,
                "state": _text(act, "state"),
                "state_detail": _text(act, "stateDetailActivity"),
                "comments": _text(act, "comments"),
                "is_npt": major == NPT_MAJOR_CODE,
                "source_file": Path(path).name,
            })
    return activities, summaries


def parse_ddr_dir(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    files = sorted(Path(path).rglob("*.xml"))
    if not files:
        raise FileNotFoundError(f"no DDR XML files under {path}")
    acts, reps = [], []
    for f in files:
        a, r = parse_ddr_file(f)
        acts += a
        reps += r
    activities = pd.DataFrame(acts)
    reports = pd.DataFrame(reps)
    # A revised report for the same wellbore and day replaces the earlier file wholesale.
    reports = reports.drop_duplicates(["wellbore", "report_date"], keep="last")
    kept = set(zip(reports["wellbore"], reports["report_date"], reports["source_file"]))
    mask = [k in kept for k in zip(activities["wellbore"], activities["report_date"], activities["source_file"])]
    return activities[mask].reset_index(drop=True), reports.reset_index(drop=True)
