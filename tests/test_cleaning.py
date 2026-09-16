import numpy as np
import pandas as pd
import pytest

from oilfield.threew.cleaning import (
    CleaningConfig,
    clean_instance,
    frozen_mask,
    out_of_range_mask,
    resample_instance,
)
from oilfield.threew.schema import is_transient, model_label

SENSORS = {
    "P-TPT": {"column": "p_tpt", "kind": "pressure", "min": 1.0, "max": 1000.0},
    "T-TPT": {"column": "t_tpt", "kind": "temperature", "min": -5.0, "max": 200.0},
    "QGL": {"column": "qgl", "kind": "flow", "min": 0.0, "max": 20.0},
}


def make_run(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.Series(pd.date_range("2017-01-01", periods=n, freq="1s", tz="UTC"))
    return pd.DataFrame({
        "ts": ts,
        "p_tpt": 150 + rng.normal(0, 0.05, n),
        "t_tpt": 110 + rng.normal(0, 0.01, n),
        "qgl": np.zeros(n),
        "raw_class": np.r_[np.zeros(n // 2), np.full(n - n // 2, 105.0)],
    })


def test_out_of_range_flags_sentinels_and_dead_gauges():
    s = pd.Series([-99_990.0, 0.0, 150.0, 2.9e4, np.nan])
    assert out_of_range_mask(s, 1.0, 1000.0).tolist() == [True, True, False, True, False]


def test_frozen_mask_only_flags_long_runs():
    ts = pd.Series(pd.date_range("2017-01-01", periods=100, freq="1s", tz="UTC"))
    values = pd.Series(np.r_[np.full(40, 7.0), np.arange(20.0), np.full(40, 3.0)])
    values.iloc[70] = np.nan  # a gap splits the second plateau into two short runs
    mask = frozen_mask(values, ts, min_seconds=30)
    assert mask.iloc[:40].all()
    assert not mask.iloc[40:].any()


def test_frozen_mask_ignores_timestamp_resolution():
    ts = pd.Series(pd.date_range("2017-01-01", periods=40, freq="1s")).astype("datetime64[us]")
    assert frozen_mask(pd.Series(np.full(40, 7.0)), ts, min_seconds=30).all()


def test_clean_instance_masks_and_drops_sparse_sensors():
    run = make_run()
    run.loc[:99, "p_tpt"] = -9_999_000 / 1e5      # historian sentinel
    run.loc[200:1100, "t_tpt"] = 110.0              # stuck temperature gauge, 15 minutes
    cfg = CleaningConfig(frozen_min_seconds=600, min_sensor_coverage=0.5)
    clean, quality = clean_instance(run, SENSORS, cfg)
    q = {row["sensor"]: row for row in quality}

    assert q["p_tpt"]["n_out_of_range"] == 100 and q["p_tpt"]["kept"]
    assert clean["p_tpt"].iloc[:100].isna().all()
    assert q["t_tpt"]["n_frozen"] == 901 and not q["t_tpt"]["kept"]
    assert clean["t_tpt"].isna().all()
    # gas-lift rate at zero is a valid state, not a frozen gauge
    assert q["qgl"]["n_frozen"] == 0 and q["qgl"]["kept"]


def test_resample_keeps_label_at_bucket_end():
    run = make_run(n=60)
    run.loc[25, "raw_class"] = np.nan
    run.loc[29, "raw_class"] = 0.0
    run.loc[30:, "raw_class"] = 105.0
    out = resample_instance(run, ["p_tpt", "t_tpt", "qgl"], seconds=10)
    assert len(out) == 6
    assert out["n_obs"].tolist() == [10] * 6
    assert out["p_tpt"].iloc[0] == pytest.approx(run["p_tpt"].iloc[:10].mean())
    assert out["raw_class"].tolist() == [0, 0, 0, 105, 105, 105]


def test_model_labels_merge_hydrates_and_strip_transients():
    assert model_label(0) == 0
    assert model_label(105) == 5
    assert model_label(9) == 8 and model_label(109) == 8
    assert model_label(float("nan")) is None
    assert is_transient(107) and not is_transient(7)
