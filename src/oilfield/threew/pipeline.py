"""3W pipeline: raw Parquet runs -> silver Delta tables -> gold window features.

Flow, end to end:

    dataset/<event>/<run>.parquet
      -> mapInPandas: read each run, mask out-of-range / frozen / sparse sensors, resample to 10 s
      -> silver_3w_quality    one row per (run, sensor): how much each mask removed
      -> silver_3w_instances  one row per run: source, well, event timing, keep/drop decision
      -> silver_3w_readings   10 s readings of the kept runs, with merged model labels
      -> gold_3w_windows      one row per (run, window end): features + label

Each run is a complete, time-ordered file, so cleaning happens inside the task that reads it
and no 1 Hz row is ever shuffled; Spark-native aggregation and window functions take over
once the data is down to 10 s buckets.

Run locally:  python -m oilfield.threew.pipeline --stage all
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterator
from functools import partial

import pandas as pd
from pyspark import StorageLevel
from pyspark.sql import functions as F
from pyspark.sql import types as T

from oilfield.config import load_config, parse_overrides, resolve_path
from oilfield.spark import TableStore, get_spark
from oilfield.threew import windowing
from oilfield.threew.cleaning import CleaningConfig, clean_instance, resample_instance
from oilfield.threew.ingest import discover_instances, read_instance
from oilfield.threew.schema import LABEL_MERGE, TRANSIENT_OFFSET

log = logging.getLogger(__name__)

QUALITY_FIELDS = [
    ("sensor", T.StringType()),
    ("n_rows", T.LongType()),
    ("n_missing", T.LongType()),
    ("n_out_of_range", T.LongType()),
    ("n_frozen", T.LongType()),
    ("valid_fraction", T.DoubleType()),
    ("kept", T.BooleanType()),
]


def sensor_columns(cfg: dict) -> list[str]:
    return [spec["column"] for spec in cfg["sensors"].values()]


def processed_schema(columns: list[str]) -> T.StructType:
    """Readings and quality rows travel through one mapInPandas pass, told apart by ``kind``."""
    fields = [
        T.StructField("kind", T.StringType()),
        T.StructField("instance_id", T.StringType()),
        T.StructField("ts", T.TimestampType()),
        *[T.StructField(c, T.FloatType()) for c in columns],
        T.StructField("raw_class", T.IntegerType()),
        T.StructField("n_obs", T.IntegerType()),
    ]
    fields += [T.StructField(name, dtype) for name, dtype in QUALITY_FIELDS]
    return T.StructType(fields)


def _process_runs(batches: Iterator[pd.DataFrame], sensors: dict, cleaning: CleaningConfig) -> Iterator[pd.DataFrame]:
    columns = [spec["column"] for spec in sensors.values()]
    for batch in batches:
        for run in batch.itertuples(index=False):
            raw = read_instance(run.path, sensors)
            clean, quality = clean_instance(raw, sensors, cleaning)
            readings = resample_instance(clean, columns, cleaning.resample_seconds)

            readings = readings.assign(kind="reading", instance_id=run.instance_id)
            readings[columns] = readings[columns].astype("float32")
            readings["raw_class"] = readings["raw_class"].round().astype("Int32")
            readings["n_obs"] = readings["n_obs"].astype("Int32")

            q = pd.DataFrame(quality).assign(kind="quality", instance_id=run.instance_id)
            out = pd.concat([readings, q], ignore_index=True)
            for name, dtype in QUALITY_FIELDS:
                if isinstance(dtype, T.LongType):
                    out[name] = out[name].astype("Int64")
                elif isinstance(dtype, T.BooleanType):
                    out[name] = out[name].astype("boolean")
            yield out[["kind", "instance_id", "ts", *columns, "raw_class", "n_obs", *[n for n, _ in QUALITY_FIELDS]]]


def model_label_expr(raw):
    event = raw % TRANSIENT_OFFSET
    for src, dst in LABEL_MERGE.items():
        event = F.when(event == src, F.lit(dst)).otherwise(event)
    return F.when(raw.isNotNull(), event)


def build_silver(spark, store: TableStore, cfg: dict, event_types: list[int] | None = None) -> None:
    sensors = cfg["sensors"]
    columns = sensor_columns(cfg)
    cleaning = CleaningConfig.from_dict(cfg["cleaning"])
    runs = discover_instances(resolve_path(cfg["raw_dir"]), event_types)

    # One batch per event-type folder bounds memory and lets a re-run replace just that folder.
    for event_type, runs_k in runs.groupby("event_type"):
        log.info("event type %s: %d runs", event_type, len(runs_k))
        meta = spark.createDataFrame(
            runs_k[["instance_id", "path", "event_type", "source", "well_id"]],
            "instance_id string, path string, event_type int, source string, well_id string",
        )
        n_parts = max(1, min(cfg.get("spark", {}).get("partitions", 64), len(runs_k)))
        processed = (
            meta.select("instance_id", "path")
            .repartition(n_parts)
            .mapInPandas(partial(_process_runs, sensors=sensors, cleaning=cleaning), processed_schema(columns))
            .persist(StorageLevel.MEMORY_AND_DISK)
        )
        meta = meta.drop("path")

        quality = processed.where("kind = 'quality'").select("instance_id", *[n for n, _ in QUALITY_FIELDS]).join(meta, "instance_id")
        readings = processed.where("kind = 'reading'").select("instance_id", "ts", *columns, "raw_class", "n_obs")

        raw = F.col("raw_class")
        per_run = readings.groupBy("instance_id").agg(
            F.min("ts").alias("start_ts"),
            F.max("ts").alias("end_ts"),
            F.count("*").alias("n_buckets"),
            F.count(raw).alias("n_labelled"),
            F.min(F.when(raw > 0, "ts")).alias("event_start"),
            F.min(F.when(raw >= TRANSIENT_OFFSET, "ts")).alias("transient_start"),
            F.min(F.when((raw > 0) & (raw < TRANSIENT_OFFSET), "ts")).alias("steady_onset"),
        )
        sensors_kept = quality.groupBy("instance_id").agg(F.sum(F.col("kept").cast("int")).alias("n_valid_sensors"))
        min_sensors = cleaning.min_valid_sensors
        instances = (
            meta.join(per_run, "instance_id", "left")
            .join(sensors_kept, "instance_id", "left")
            .withColumn(
                "drop_reason",
                F.when(F.coalesce("n_labelled", F.lit(0)) == 0, "no labelled readings")
                .when(F.col("n_valid_sensors") < min_sensors, f"fewer than {min_sensors} usable sensors"),
            )
            .withColumn("kept", F.col("drop_reason").isNull())
        )

        kept = instances.where("kept").select("instance_id", "event_type", "source", "well_id")
        silver = (
            readings.join(F.broadcast(kept), "instance_id")
            .withColumn("label", model_label_expr(raw))
            .withColumn("is_transient", raw >= TRANSIENT_OFFSET)
        )

        where = f"event_type = {int(event_type)}"
        for df, name, parts in [
            (quality.coalesce(1), "silver_3w_quality", None),
            (instances.coalesce(1), "silver_3w_instances", None),
            (silver, "silver_3w_readings", ["event_type"]),
        ]:
            exists = store.exists(name)
            store.write(df, name, mode="overwrite", partition_by=parts if not exists else None,
                        replace_where=where if exists else None)
        processed.unpersist()


def build_gold(spark, store: TableStore, cfg: dict) -> None:
    w = cfg["windows"]
    columns = sensor_columns(cfg)
    instances = store.read("silver_3w_instances").where("kept")
    readings = store.read("silver_3w_readings")
    buckets = windowing.bucket_stats(readings, instances, columns, w["step_seconds"])
    gold = windowing.window_features(
        buckets,
        instances,
        columns,
        length_seconds=w["length_seconds"],
        step_seconds=w["step_seconds"],
        baseline_seconds=w["baseline_seconds"],
        resample_seconds=cfg["cleaning"]["resample_seconds"],
        min_coverage=w["min_coverage"],
        min_valid_sensors=w["min_valid_sensors"],
    )
    store.write(gold, "gold_3w_windows", mode="overwrite")


def summarize(store: TableStore) -> pd.DataFrame:
    inst = store.read("silver_3w_instances").toPandas()
    summary = inst.groupby(["source", "kept"]).size().unstack(fill_value=0)
    log.info("runs by source / kept:\n%s", summary)
    log.info("drop reasons:\n%s", inst["drop_reason"].value_counts())
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/threew.yaml")
    parser.add_argument("--stage", choices=["silver", "gold", "all"], default="all")
    parser.add_argument("--event-types", type=int, nargs="*", help="limit silver to these 3W folders")
    parser.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. tables.catalog=main")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config, parse_overrides(args.set))
    spark = get_spark("oilfield-3w", shuffle_partitions=cfg.get("spark", {}).get("partitions", 64))
    store = TableStore.from_config(spark, cfg["tables"])
    if args.stage in ("silver", "all"):
        build_silver(spark, store, cfg, args.event_types)
        summarize(store)
    if args.stage in ("gold", "all"):
        build_gold(spark, store, cfg)
        n = store.read("gold_3w_windows").count()
        log.info("gold_3w_windows: %d windows", n)


if __name__ == "__main__":
    main()

