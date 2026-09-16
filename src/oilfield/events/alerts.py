"""Turn per-window class probabilities into debounced alerts and score how early they come.

A run's windows arrive once a minute. A fault class *votes* in a window when its probability
reaches ``threshold``; an alert for that class fires at the first window where it has voted in
``k`` of the last ``n`` windows. For runs whose event has a labelled transient, the lead time is

    lead = steady_onset - alert_time

so a positive lead means the engineer was warned before the fault reached steady state.
Alerts are only credited from the start of the labelled event onward; anything earlier is
counted as a false alert on normal operation instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from oilfield.threew.schema import CLASS_NAMES, EVENTS_WITH_TRANSIENT, model_label


@dataclass(frozen=True)
class AlertRule:
    threshold: float = 0.6
    k: int = 3
    n: int = 5


def alert_flags(proba: np.ndarray, rule: AlertRule) -> np.ndarray:
    """(T, C) probabilities for one run, time-ordered -> (T, C) booleans: alert active."""
    votes = (proba >= rule.threshold).astype(np.int32)
    csum = np.cumsum(votes, axis=0)
    shifted = np.vstack([np.zeros((rule.n, votes.shape[1]), dtype=np.int32), csum])[: len(votes)]
    return (csum - shifted) >= rule.k


def score_alerts(windows: pd.DataFrame, proba: np.ndarray, instances: pd.DataFrame, rule: AlertRule) -> pd.DataFrame:
    """One row per run: first alert time for its own event class, lead time, false alerts.

    ``windows`` needs instance_id, window_end and label, sorted by run then time, aligned with
    ``proba``.
    """
    meta = instances.set_index("instance_id")
    rows = []
    bounds = np.flatnonzero(np.r_[True, windows["instance_id"].to_numpy()[1:] != windows["instance_id"].to_numpy()[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        run = windows.iloc[a:b]
        iid = run["instance_id"].iloc[0]
        m = meta.loc[iid]
        flags = alert_flags(proba[a:b], rule)
        times = run["window_end"].to_numpy()
        normal = run["label"].to_numpy() == 0

        # A fault alert on a normal window is a false alert (any fault class).
        fault_flags = flags[:, 1:].any(axis=1)
        false_alert_windows = int((fault_flags & normal).sum())
        false_alert_onsets = int((np.diff(np.r_[0, (fault_flags & normal).astype(int)]) == 1).sum())

        event = model_label(m["event_type"])
        row = {
            "instance_id": iid,
            "well_id": m["well_id"],
            "source": m["source"],
            "event": event,
            "normal_minutes": int(normal.sum()),
            "false_alert_windows": false_alert_windows,
            "false_alerts": false_alert_onsets,
            "alert_time": pd.NaT,
            "lead_minutes": np.nan,
            "delay_minutes": np.nan,
        }
        if event and not pd.isna(m["event_start"]):
            credited = flags[:, event] & (times >= np.datetime64(m["event_start"]))
            if credited.any():
                t_alert = times[np.argmax(credited)]
                row["alert_time"] = t_alert
                row["delay_minutes"] = (t_alert - np.datetime64(m["event_start"])) / np.timedelta64(1, "m")
                if m["event_type"] in EVENTS_WITH_TRANSIENT and not pd.isna(m["steady_onset"]):
                    row["lead_minutes"] = (np.datetime64(m["steady_onset"]) - t_alert) / np.timedelta64(1, "m")
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_alerts(scored: pd.DataFrame) -> dict:
    """Run-level alert metrics, overall and per fault class.

    Transient lengths differ by orders of magnitude between events (a spurious DHSV closure
    settles in minutes, choke scaling over many hours), so the per-class lead times are the
    meaningful ones; the pooled median mostly reflects which classes dominate the test wells.
    """
    faults = scored[scored["event"] > 0]
    transient = faults["event"].isin(_merged_transient_classes())
    lead = faults.loc[transient, "lead_minutes"].dropna()
    normal_hours = scored["normal_minutes"].sum() / 60

    def _median(s: pd.Series) -> float:
        s = s.dropna()
        return float(s.median()) if len(s) else float("nan")

    return {
        "fault_runs": int(len(faults)),
        "detection_rate": float(faults["alert_time"].notna().mean()) if len(faults) else float("nan"),
        "median_lead_minutes": _median(lead),
        "share_alerted_before_steady_state": float((lead > 0).sum() / max(int(transient.sum()), 1)),
        "median_delay_minutes": _median(faults["delay_minutes"]),
        "false_alerts_per_normal_day": float(scored["false_alerts"].sum() / max(normal_hours, 1e-9) * 24),
        "per_class": {
            CLASS_NAMES[int(c)]: {
                "runs": int(len(g)),
                "detection_rate": float(g["alert_time"].notna().mean()),
                "median_lead_minutes": _median(g["lead_minutes"]),
                "median_delay_minutes": _median(g["delay_minutes"]),
            }
            for c, g in faults.groupby("event")
        },
    }


def _merged_transient_classes() -> set[int]:
    return {model_label(e) for e in EVENTS_WITH_TRANSIENT}

