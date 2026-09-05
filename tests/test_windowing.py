import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.spark

STEP, LENGTH, BASELINE, RES = 60, 600, 600, 10


@pytest.fixture(scope="module")
def run_frames():
    n = 240  # 40 minutes of 10 s buckets
    t = np.arange(n) * RES
    rng = np.random.default_rng(3)
    ts = pd.Timestamp("2017-05-01", tz="UTC") + pd.to_timedelta(t, unit="s")
    readings = pd.DataFrame({
        "instance_id": "5-WELL-00015_x",
        "ts": ts,
        "p_tpt": 100.0 + 0.01 * t,                     # ramp: 0.6 bar/min
        "t_tpt": 80.0 + rng.normal(0, 0.2, n),
        "label": np.where(t < 1800, 0, 5),
        "is_transient": t >= 1800,
    })
    readings.loc[readings.index[-35:], "t_tpt"] = np.nan  # t_tpt drops out near the end
    instances = pd.DataFrame({
        "instance_id": ["5-WELL-00015_x"],
        "well_id": ["WELL-00015"],
        "source": ["WELL"],
        "event_type": [5],
        "start_ts": [ts[0]],
    })
    return readings, instances


@pytest.fixture(scope="module")
def gold(spark, run_frames):
    from oilfield.threew import windowing

    readings, instances = run_frames
    r = spark.createDataFrame(readings)
    i = spark.createDataFrame(instances)
    buckets = windowing.bucket_stats(r, i, ["p_tpt", "t_tpt"], STEP)
    out = windowing.window_features(
        buckets, i, ["p_tpt", "t_tpt"], length_seconds=LENGTH, step_seconds=STEP,
        baseline_seconds=BASELINE, resample_seconds=RES, min_coverage=0.5, min_valid_sensors=1,
    )
    return out.toPandas().sort_values("window_end").reset_index(drop=True)


def test_one_window_per_step_after_first_full_window(gold):
    # 40 one-minute buckets; the first full 10-minute window ends at minute 10
    assert len(gold) == 31
    assert gold["window_end"].iloc[0] == pd.Timestamp("2017-05-01 00:10:00")
    assert gold["window_end"].diff().dropna().eq(pd.Timedelta(minutes=1)).all()


def test_window_statistics_match_numpy(gold, run_frames):
    readings, _ = run_frames
    end = pd.Timestamp("2017-05-01 00:25:00", tz="UTC")
    inside = readings[(readings.ts >= end - pd.Timedelta(seconds=LENGTH)) & (readings.ts < end)]
    row = gold[gold.window_end == end.tz_localize(None)].iloc[0]

    assert row.p_tpt_mean == pytest.approx(inside.p_tpt.mean(), rel=1e-6)
    assert row.t_tpt_std == pytest.approx(inside.t_tpt.std(ddof=1), rel=1e-4)
    assert row.t_tpt_min == pytest.approx(inside.t_tpt.min(), rel=1e-6)
    assert row.p_tpt_slope == pytest.approx(0.6, rel=1e-4)
    assert row.p_tpt_delta == pytest.approx(inside.p_tpt.iloc[-1] - inside.p_tpt.iloc[0], rel=1e-5)

    before = readings[(readings.ts >= end - pd.Timedelta(seconds=LENGTH + BASELINE))
                      & (readings.ts < end - pd.Timedelta(seconds=LENGTH))]
    assert row.p_tpt_dev == pytest.approx(inside.p_tpt.mean() - before.p_tpt.mean(), rel=1e-5)


def test_label_is_state_at_window_end(gold):
    labels = gold.set_index("window_end")["label"]
    assert labels[pd.Timestamp("2017-05-01 00:30:00")] == 0
    assert labels[pd.Timestamp("2017-05-01 00:31:00")] == 5


def test_sparse_sensor_features_are_null(gold):
    last = gold.iloc[-1]
    assert np.isnan(last.t_tpt_mean) and not np.isnan(last.p_tpt_mean)
    first = gold.iloc[0]
    assert np.isnan(first.p_tpt_dev)  # no baseline history before the first window
