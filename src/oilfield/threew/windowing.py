"""Sliding-window features over cleaned 10 s readings, in native Spark.

Windows are built in two passes so no row is ever duplicated:

1. ``bucket_stats`` collapses 10 s readings into ``step_seconds`` buckets of additive
   statistics (count, sum, sum of squares, min, max, first, last, and the sums a least-squares
   slope needs);
2. ``window_features`` combines the buckets inside each ``length_seconds`` window with range
   window functions, giving exact means, standard deviations, slopes and drift from the
   trailing baseline, one row per (run, window end).

Each window also gets long-context features over the hour that ends with it (``l*``) and drift
against a 3-hour trailing baseline: slugging cycles and choke scaling unfold over tens of
minutes to hours, far longer than one 10-minute window.

A window is labelled with the well's state at its end, which is the moment an online detector
would raise an alert.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

FEATURE_STATS = ("mean", "std", "min", "max", "delta", "slope", "dev", "lstd", "lslope", "lrange", "ldev")


def feature_names(sensor_columns: list[str], stats=FEATURE_STATS) -> list[str]:
    return [f"{c}_{s}" for c in sensor_columns for s in stats]


def bucket_stats(readings: DataFrame, instances: DataFrame, sensor_columns: list[str], step_seconds: int) -> DataFrame:
    # Time relative to the run start keeps the slope sums well inside double precision.
    start = instances.select("instance_id", F.col("start_ts").cast("long").alias("t0"))
    t = F.col("ts").cast("long") - F.col("t0")
    df = readings.join(F.broadcast(start), "instance_id").withColumn("t", t)
    bucket = (F.col("t") - F.col("t") % step_seconds).alias("bucket")

    labelled_t = F.when(F.col("label").isNotNull(), F.col("t"))
    aggs = [
        F.max_by("label", labelled_t).alias("label"),
        F.max_by("is_transient", labelled_t).alias("is_transient"),
    ]
    for c in sensor_columns:
        x = F.col(c).cast("double")
        tx = F.when(x.isNotNull(), F.col("t").cast("double"))
        aggs += [
            F.count(x).alias(f"{c}__n"),
            F.sum(x).alias(f"{c}__s1"),
            F.sum(x * x).alias(f"{c}__s2"),
            F.min(x).alias(f"{c}__min"),
            F.max(x).alias(f"{c}__max"),
            F.min_by(x, tx).alias(f"{c}__first"),
            F.max_by(x, tx).alias(f"{c}__last"),
            F.sum(tx).alias(f"{c}__st"),
            F.sum(tx * tx).alias(f"{c}__stt"),
            F.sum(tx * x).alias(f"{c}__stx"),
        ]
    return df.groupBy("instance_id", bucket).agg(*aggs)


def _frame_stats(c: str, frame) -> dict:
    n = F.sum(f"{c}__n").over(frame)
    s1, s2 = F.sum(f"{c}__s1").over(frame), F.sum(f"{c}__s2").over(frame)
    st, stt, stx = F.sum(f"{c}__st").over(frame), F.sum(f"{c}__stt").over(frame), F.sum(f"{c}__stx").over(frame)
    mean = s1 / n
    slope_den = n * stt - st * st
    return {
        "n": n,
        "mean": mean,
        "std": F.sqrt(F.greatest((s2 - n * mean * mean) / (n - 1), F.lit(0.0))),
        "min": F.min(f"{c}__min").over(frame),
        "max": F.max(f"{c}__max").over(frame),
        "first": F.first(f"{c}__first", ignorenulls=True).over(frame),
        "last": F.last(f"{c}__last", ignorenulls=True).over(frame),
        "slope": F.when(slope_den > 0, (n * stx - st * s1) / slope_den * 60.0),  # per minute
    }


def window_features(
    buckets: DataFrame,
    instances: DataFrame,
    sensor_columns: list[str],
    *,
    length_seconds: int,
    step_seconds: int,
    baseline_seconds: int,
    resample_seconds: int,
    min_coverage: float,
    min_valid_sensors: int,
    long_seconds: int = 3600,
    long_baseline_seconds: int = 10800,
) -> DataFrame:
    if length_seconds % step_seconds or long_seconds % step_seconds:
        raise ValueError("window lengths must be multiples of the step")
    span = length_seconds - step_seconds
    order = Window.partitionBy("instance_id").orderBy("bucket")
    win = order.rangeBetween(-span, 0)
    longwin = order.rangeBetween(-(long_seconds - step_seconds), 0)
    base = order.rangeBetween(-span - baseline_seconds, -length_seconds)
    longbase = order.rangeBetween(-span - long_baseline_seconds, -length_seconds)

    def base_mean(c, frame):
        return F.sum(f"{c}__s1").over(frame) / F.sum(f"{c}__n").over(frame)

    cols, coverage_ok = [], []
    for c in sensor_columns:
        short, long = _frame_stats(c, win), _frame_stats(c, longwin)
        ok = (short["n"] / F.lit(length_seconds / resample_seconds)) >= min_coverage
        long_ok = ok & ((long["n"] / F.lit(long_seconds / resample_seconds)) >= min_coverage)
        coverage_ok.append(F.when(ok, 1).otherwise(0))
        stats = {
            "mean": (ok, short["mean"]),
            "std": (ok, short["std"]),
            "min": (ok, short["min"]),
            "max": (ok, short["max"]),
            "delta": (ok, short["last"] - short["first"]),
            "slope": (ok, short["slope"]),
            "dev": (ok, short["mean"] - base_mean(c, base)),
            "lstd": (long_ok, long["std"]),
            "lslope": (long_ok, long["slope"]),
            "lrange": (long_ok, long["max"] - long["min"]),
            "ldev": (ok, short["mean"] - base_mean(c, longbase)),
        }
        cols += [F.when(valid, expr).cast("float").alias(f"{c}_{name}") for name, (valid, expr) in stats.items()]

    meta = instances.select("instance_id", "well_id", "source", "event_type", F.col("start_ts").cast("long").alias("t0"))
    windows = buckets.select(
        "instance_id",
        "bucket",
        "label",
        "is_transient",
        *cols,
        sum(coverage_ok).alias("n_valid_sensors"),
    )
    return (
        windows.where(F.col("bucket") >= span)  # only full-length windows
        .where(F.col("label").isNotNull())
        .where(F.col("n_valid_sensors") >= min_valid_sensors)
        .join(F.broadcast(meta), "instance_id")
        .withColumn("window_end", F.timestamp_seconds(F.col("t0") + F.col("bucket") + step_seconds))
        .drop("t0", "bucket")
    )
