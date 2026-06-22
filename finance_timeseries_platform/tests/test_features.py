"""
test_features.py

Unit tests for features.py transformations.
Run with: pytest tests/test_features.py -v

Tests use a small synthetic DataFrame (not the real Bronze table) to keep
tests fast, isolated, and reproducible - no dependency on Databricks/Unity
Catalog connection needed for the logic tests.
"""

import math
import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import isnan
from pyspark.sql.types import (
    BooleanType,
    DateType,
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
)
from datetime import date

import sys
import os

# Add src to path so we can import features module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from features import (
    FeaturesConfig,
    add_illiquid_flag,
    add_log_returns,
    clean_nan,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def spark():
    """Create a local SparkSession for testing."""
    return (
        SparkSession.builder
        .master("local[1]")
        .appName("features_unit_tests")
        .config("spark.sql.shuffle.partitions", "1")
        .getOrCreate()
    )


@pytest.fixture
def base_config():
    """Minimal FeaturesConfig for testing."""
    return FeaturesConfig(
        volume_exempt_tickers=["^VIX"],
        illiquid_volume_threshold=0,
        returns_method="log",
        returns_price_column="adj_close",
        volatility_windows=[20, 60],
        ma_windows=[20, 50, 200],
        ma_price_column="adj_close",
        rsi_window=14,
        rsi_price_column="adj_close",
        vix_ticker="^VIX",
        vix_correlation_window=60,
    )


OHLCV_SCHEMA = StructType([
    StructField("ticker", StringType(), False),
    StructField("trade_date", DateType(), False),
    StructField("adj_close", DoubleType(), True),
    StructField("volume", LongType(), True),
])


# ---------------------------------------------------------------------------
# Tests: add_illiquid_flag
# ---------------------------------------------------------------------------

class TestIlliquidFlag:

    def test_normal_ticker_zero_volume_flagged(self, spark, base_config):
        """Regular ticker with volume=0 should be flagged as illiquid."""
        df = spark.createDataFrame([
            ("HYFT", date(2024, 1, 2), 5.0, 0),
        ], schema=OHLCV_SCHEMA)

        result = add_illiquid_flag(df, base_config)
        row = result.collect()[0]

        assert row["is_illiquid"] is True

    def test_normal_ticker_nonzero_volume_not_flagged(self, spark, base_config):
        """Regular ticker with volume > 0 should not be flagged."""
        df = spark.createDataFrame([
            ("HYFT", date(2024, 1, 2), 5.0, 10000),
        ], schema=OHLCV_SCHEMA)

        result = add_illiquid_flag(df, base_config)
        row = result.collect()[0]

        assert row["is_illiquid"] is False

    def test_vix_zero_volume_not_flagged(self, spark, base_config):
        """^VIX with volume=0 should NOT be flagged (it's volume-exempt)."""
        df = spark.createDataFrame([
            ("^VIX", date(2024, 1, 2), 15.0, 0),
        ], schema=OHLCV_SCHEMA)

        result = add_illiquid_flag(df, base_config)
        row = result.collect()[0]

        assert row["is_illiquid"] is False

    def test_mixed_tickers(self, spark, base_config):
        """Mix of tickers and volumes - each should be flagged correctly."""
        df = spark.createDataFrame([
            ("HYFT", date(2024, 1, 2), 5.0, 0),       # illiquid
            ("HYFT", date(2024, 1, 3), 5.1, 50000),    # not illiquid
            ("^VIX", date(2024, 1, 2), 15.0, 0),       # exempt, not illiquid
            ("SPY", date(2024, 1, 2), 480.0, 1000000), # not illiquid
        ], schema=OHLCV_SCHEMA)

        result = add_illiquid_flag(df, base_config)
        rows = {(r["ticker"], str(r["trade_date"])): r["is_illiquid"]
                for r in result.collect()}

        assert rows[("HYFT", "2024-01-02")] is True
        assert rows[("HYFT", "2024-01-03")] is False
        assert rows[("^VIX", "2024-01-02")] is False
        assert rows[("SPY", "2024-01-02")] is False


# ---------------------------------------------------------------------------
# Tests: add_log_returns
# ---------------------------------------------------------------------------

LOG_RETURN_SCHEMA = StructType([
    StructField("ticker", StringType(), False),
    StructField("trade_date", DateType(), False),
    StructField("adj_close", DoubleType(), True),
    StructField("volume", LongType(), True),
    StructField("is_illiquid", BooleanType(), True),
])


class TestLogReturns:

    def test_first_row_is_null(self, spark, base_config):
        """First row per ticker has no previous price, so log_return must be NULL."""
        df = spark.createDataFrame([
            ("HYFT", date(2024, 1, 2), 100.0, 10000, False),
            ("HYFT", date(2024, 1, 3), 105.0, 10000, False),
        ], schema=LOG_RETURN_SCHEMA)

        result = add_log_returns(df, base_config)
        rows = result.orderBy("trade_date").collect()

        assert rows[0]["log_return"] is None

    def test_log_return_value_correct(self, spark, base_config):
        """Log return for day 2 should be ln(105/100) ≈ 0.04879."""
        df = spark.createDataFrame([
            ("HYFT", date(2024, 1, 2), 100.0, 10000, False),
            ("HYFT", date(2024, 1, 3), 105.0, 10000, False),
        ], schema=LOG_RETURN_SCHEMA)

        result = add_log_returns(df, base_config)
        rows = result.orderBy("trade_date").collect()

        expected = math.log(105.0 / 100.0)
        assert abs(rows[1]["log_return"] - expected) < 1e-9

    def test_partitioned_by_ticker(self, spark, base_config):
        """Each ticker should compute log returns independently."""
        df = spark.createDataFrame([
            ("HYFT", date(2024, 1, 2), 100.0, 10000, False),
            ("HYFT", date(2024, 1, 3), 110.0, 10000, False),
            ("SPY",  date(2024, 1, 2), 480.0, 500000, False),
            ("SPY",  date(2024, 1, 3), 490.0, 500000, False),
        ], schema=LOG_RETURN_SCHEMA)

        result = add_log_returns(df, base_config)
        rows = {(r["ticker"], str(r["trade_date"])): r["log_return"]
                for r in result.collect()}

        # First row of each ticker must be NULL
        assert rows[("HYFT", "2024-01-02")] is None
        assert rows[("SPY",  "2024-01-02")] is None

        # SPY return should be ln(490/480), not influenced by HYFT
        expected_spy = math.log(490.0 / 480.0)
        assert abs(rows[("SPY", "2024-01-03")] - expected_spy) < 1e-9

    def test_negative_return(self, spark, base_config):
        """Price decrease should produce negative log return."""
        df = spark.createDataFrame([
            ("HYFT", date(2024, 1, 2), 100.0, 10000, False),
            ("HYFT", date(2024, 1, 3),  90.0, 10000, False),
        ], schema=LOG_RETURN_SCHEMA)

        result = add_log_returns(df, base_config)
        rows = result.orderBy("trade_date").collect()

        assert rows[1]["log_return"] < 0


# ---------------------------------------------------------------------------
# Tests: clean_nan
# ---------------------------------------------------------------------------

class TestCleanNan:

    def test_nan_replaced_with_null(self, spark, base_config):
        """NaN values in numeric columns should be replaced with NULL."""
        schema = StructType([
            StructField("ticker", StringType(), False),
            StructField("trade_date", DateType(), False),
            StructField("log_return", DoubleType(), True),
            StructField("rsi", DoubleType(), True),
        ])

        df = spark.createDataFrame([
            ("HYFT", date(2024, 1, 2), float("nan"), float("nan")),
            ("HYFT", date(2024, 1, 3), 0.05, 65.0),
        ], schema=schema)

        result = clean_nan(df)
        rows = result.orderBy("trade_date").collect()

        # NaN row should now be NULL
        assert rows[0]["log_return"] is None
        assert rows[0]["rsi"] is None

        # Normal row should be unchanged
        assert abs(rows[1]["log_return"] - 0.05) < 1e-9
        assert abs(rows[1]["rsi"] - 65.0) < 1e-9

    def test_no_nan_unchanged(self, spark, base_config):
        """DataFrame without NaN should be unchanged."""
        schema = StructType([
            StructField("ticker", StringType(), False),
            StructField("trade_date", DateType(), False),
            StructField("log_return", DoubleType(), True),
            StructField("rsi", DoubleType(), True),
        ])

        df = spark.createDataFrame([
            ("HYFT", date(2024, 1, 2), 0.03, 55.0),
            ("HYFT", date(2024, 1, 3), -0.01, 48.0),
        ], schema=schema)

        result = clean_nan(df)
        rows = result.orderBy("trade_date").collect()

        assert abs(rows[0]["log_return"] - 0.03) < 1e-9
        assert abs(rows[1]["rsi"] - 48.0) < 1e-9
