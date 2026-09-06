"""SparkSession and Delta table access that work the same locally and on Databricks."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from oilfield.config import resolve_path


def on_databricks() -> bool:
    return "DATABRICKS_RUNTIME_VERSION" in os.environ


def get_spark(app_name: str = "oilfield", shuffle_partitions: int = 64):
    from pyspark.sql import SparkSession

    if on_databricks():
        return SparkSession.builder.getOrCreate()

    from delta import configure_spark_with_delta_pip

    # Workers must run the same interpreter as the driver.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    builder = (
        SparkSession.builder.master(os.environ.get("SPARK_MASTER", "local[*]"))
        .appName(app_name)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.parquet.compression.codec", "zstd")
        .config("spark.driver.memory", os.environ.get("SPARK_DRIVER_MEMORY", "8g"))
        .config("spark.driver.maxResultSize", "4g")
        .config("spark.ui.showConsoleProgress", "false")
    )
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


class TableStore:
    """Names Delta tables once: Unity Catalog ``catalog.schema.name`` or ``<path_root>/<name>``."""

    def __init__(self, spark, path_root: str | None = None, catalog: str | None = None, schema: str | None = None):
        self.spark = spark
        self.catalog = catalog
        self.schema = schema
        self.path_root = Path(resolve_path(path_root)) if path_root else None
        if not (catalog and schema) and self.path_root is None:
            raise ValueError("TableStore needs either catalog+schema or path_root")

    @classmethod
    def from_config(cls, spark, tables_cfg: dict) -> TableStore:
        return cls(spark, tables_cfg.get("path_root"), tables_cfg.get("catalog"), tables_cfg.get("schema"))

    @property
    def uses_catalog(self) -> bool:
        return bool(self.catalog and self.schema)

    def location(self, name: str) -> str:
        return f"{self.catalog}.{self.schema}.{name}" if self.uses_catalog else str(self.path_root / name)

    def read(self, name: str):
        if self.uses_catalog:
            return self.spark.table(self.location(name))
        return self.spark.read.format("delta").load(self.location(name))

    def write(self, df, name: str, mode: str = "overwrite", partition_by: list[str] | None = None,
              replace_where: str | None = None) -> None:
        writer = df.write.format("delta").mode(mode)
        if partition_by:
            writer = writer.partitionBy(*partition_by)
        if replace_where:
            writer = writer.option("replaceWhere", replace_where)
        elif mode == "overwrite":
            writer = writer.option("overwriteSchema", "true")
        if self.uses_catalog:
            writer.saveAsTable(self.location(name))
        else:
            writer.save(self.location(name))

    def exists(self, name: str) -> bool:
        if self.uses_catalog:
            return self.spark.catalog.tableExists(self.location(name))
        return (self.path_root / name / "_delta_log").exists()
