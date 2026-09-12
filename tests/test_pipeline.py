import pandas as pd
import pytest

pytestmark = pytest.mark.spark


def test_run_timing_finds_event_transient_and_steady_state(spark):
    from oilfield.threew.pipeline import run_timing

    ts = pd.date_range("2018-01-01", periods=7, freq="10s")
    readings = pd.DataFrame({"instance_id": "5-x", "ts": ts, "raw_class": [None, 0, 0, 105, 105, 5, 5]})
    row = run_timing(spark.createDataFrame(readings, "instance_id string, ts timestamp, raw_class int")).toPandas().iloc[0]
    assert row.start_ts == ts[0] and row.end_ts == ts[6]
    assert row.n_buckets == 7 and row.n_labelled == 6
    assert row.event_start == ts[3] and row.transient_start == ts[3]
    assert row.steady_onset == ts[5]


def test_model_label_expr_merges_and_strips_transients(spark):
    from pyspark.sql import functions as F

    from oilfield.threew.pipeline import model_label_expr

    df = spark.createDataFrame(pd.DataFrame({"raw_class": [0, 3, 105, 9, 109, None]}), "raw_class int")
    out = df.select(model_label_expr(F.col("raw_class")).alias("label")).toPandas()["label"].tolist()
    assert out[:5] == [0, 3, 5, 8, 8] and pd.isna(out[5])
