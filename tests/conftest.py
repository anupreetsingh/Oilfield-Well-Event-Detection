import os
import sys

import pytest


@pytest.fixture(scope="session")
def spark():
    sql = pytest.importorskip("pyspark.sql")

    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    session = (
        sql.SparkSession.builder.master("local[2]")
        .appName("oilfield-tests")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.ui.enabled", "false")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()
