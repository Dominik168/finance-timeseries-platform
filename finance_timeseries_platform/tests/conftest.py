"""
conftest.py

Minimal pytest configuration for local unit tests.
Uses a local SparkSession (no Databricks Connect required).
"""

import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark():
    """Local SparkSession for unit tests - no cluster connection needed."""
    session = (
        SparkSession.builder
        .master("local[1]")
        .appName("finance_ts_unit_tests")
        .config("spark.sql.shuffle.partitions", "1")
        .getOrCreate()
    )
    yield session
    session.stop()