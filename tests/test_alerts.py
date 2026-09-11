import numpy as np
import pandas as pd
import pytest

from oilfield.events.alerts import AlertRule, alert_flags, score_alerts, summarize_alerts
from oilfield.threew.schema import N_CLASSES

T0 = pd.Timestamp("2018-03-01 00:00:00")


def test_alert_needs_k_of_n_votes():
    p = np.zeros((10, 2))
    p[[1, 3, 4, 5], 1] = 0.9  # votes at t=1,3,4,5
    flags = alert_flags(p, AlertRule(threshold=0.6, k=3, n=5))
    assert flags[:, 1].tolist() == [False] * 4 + [True] * 4 + [False] * 2


def run_windows(iid, labels, fault_from=None, fault_class=5, conf=0.9):
    n = len(labels)
    windows = pd.DataFrame({
        "instance_id": iid,
        "window_end": T0 + pd.to_timedelta(np.arange(n), unit="min"),
        "label": labels,
    })
    proba = np.zeros((n, N_CLASSES), dtype=np.float32)
    proba[:, 0] = 1.0
    if fault_from is not None:
        proba[fault_from:, 0] = 1 - conf
        proba[fault_from:, fault_class] = conf
    return windows, proba


def test_lead_time_is_measured_against_steady_state_onset():
    # normal for 30 min, transient from minute 30, steady state from minute 60
    labels = [0] * 30 + [5] * 60
    windows, proba = run_windows("5-WELL-00015_a", labels, fault_from=40)
    instances = pd.DataFrame([{
        "instance_id": "5-WELL-00015_a", "well_id": "WELL-00015", "source": "WELL", "event_type": 5,
        "event_start": T0 + pd.Timedelta(minutes=30), "steady_onset": T0 + pd.Timedelta(minutes=60),
    }])
    scored = score_alerts(windows, proba, instances, AlertRule(0.6, 3, 5))
    row = scored.iloc[0]
    # votes start at minute 40, third vote lands at minute 42
    assert row.alert_time == T0 + pd.Timedelta(minutes=42)
    assert row.lead_minutes == pytest.approx(18.0)
    assert row.delay_minutes == pytest.approx(12.0)
    assert row.false_alerts == 0


def test_alert_before_event_start_is_a_false_alert():
    labels = [0] * 30 + [5] * 30
    windows, proba = run_windows("5-WELL-00016_b", labels, fault_from=10)
    instances = pd.DataFrame([{
        "instance_id": "5-WELL-00016_b", "well_id": "WELL-00016", "source": "WELL", "event_type": 5,
        "event_start": T0 + pd.Timedelta(minutes=30), "steady_onset": T0 + pd.Timedelta(minutes=45),
    }])
    scored = score_alerts(windows, proba, instances, AlertRule(0.6, 3, 5))
    row = scored.iloc[0]
    assert row.false_alerts == 1 and row.false_alert_windows == 18
    assert row.alert_time == T0 + pd.Timedelta(minutes=30)  # first credited window
    summary = summarize_alerts(scored)
    assert summary["detection_rate"] == 1.0
    assert summary["median_lead_minutes"] == pytest.approx(15.0)
