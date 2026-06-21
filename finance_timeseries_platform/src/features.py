"""
features.py

Generic, config-driven feature transformation module for the Silver layer.
Reads transformation parameters from features.yaml and applies them to the
Bronze OHLCV table, producing the Silver Delta table.

Usage (from a Databricks notebook or job):

    from features import run_silver_transform
    run_silver_transform(spark, config_path="config/features.yaml")

Design principle: adding a new feature means editing features.yaml and
adding one function here, not rewriting the pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

logger = logging.getLogger("features")
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

@dataclass
class FeaturesConfig:
    volume_exempt_tickers: list[str]
    illiquid_volume_threshold: int
    returns_method: str
    returns_price_column: str


def load_config(config_path: str) -> FeaturesConfig:
    """Read features.yaml."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    cleaning_cfg = raw.get("cleaning", {})
    returns_cfg = raw.get("returns", {})

    return FeaturesConfig(
        volume_exempt_tickers=cleaning_cfg.get("volume_exempt_tickers", []),
        illiquid_volume_threshold=cleaning_cfg.get("illiquid_volume_threshold", 0),
        returns_method=returns_cfg.get("method", "log"),
        returns_price_column=returns_cfg.get("price_column", "adj_close"),
    )


# ---------------------------------------------------------------------------
# Cleaning: illiquid day flag
# ---------------------------------------------------------------------------

def add_illiquid_flag(df: DataFrame, cfg: FeaturesConfig) -> DataFrame:
    """
    Add an `is_illiquid` boolean column.

    A row is flagged illiquid if volume <= threshold AND the ticker is not
    in the volume-exempt list (e.g. ^VIX, which has no real volume concept
    because it's an index, not a traded security).
    """
    is_exempt = F.col("ticker").isin(cfg.volume_exempt_tickers)
    is_low_volume = F.col("volume") <= cfg.illiquid_volume_threshold

    return df.withColumn(
        "is_illiquid",
        F.when(is_exempt, F.lit(False)).otherwise(is_low_volume),
    )


# ---------------------------------------------------------------------------
# Log returns
# ---------------------------------------------------------------------------

def add_log_returns(df: DataFrame, cfg: FeaturesConfig) -> DataFrame:
    """
    Add a `log_return` column: ln(price_today / price_yesterday), computed
    per ticker using a window function ordered by trade_date.

    The first row for each ticker has no previous price, so log_return
    is null there - this is expected and correct (nothing to compute).
    """
    price_col = cfg.returns_price_column

    window_spec = Window.partitionBy("ticker").orderBy("trade_date")

    df_with_prev = df.withColumn(
        "_prev_price", F.lag(F.col(price_col)).over(window_spec)
    )

    df_with_returns = df_with_prev.withColumn(
        "log_return",
        F.when(
            (F.col("_prev_price").isNotNull()) & (F.col("_prev_price") > 0),
            F.log(F.col(price_col) / F.col("_prev_price")),
        ).otherwise(F.lit(None)),
    )

    return df_with_returns.drop("_prev_price")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_silver_transform(
    spark: SparkSession,
    config_path: str = "config/features.yaml",
    source_table: str = "finance_dev.bronze.ohlcv_daily",
    target_table: str = "finance_dev.silver.ohlcv_features",
    mode: str = "overwrite",
) -> DataFrame:
    """
    Run the Bronze -> Silver transform: read Bronze, apply cleaning and
    log returns, write Silver Delta table.

    Returns the resulting Spark DataFrame for inspection/testing.
    """
    cfg = load_config(config_path)
    logger.info(f"Loaded features config: returns method={cfg.returns_method}")

    df = spark.table(source_table)
    logger.info(f"Read {df.count()} rows from {source_table}")

    df = add_illiquid_flag(df, cfg)
    df = add_log_returns(df, cfg)

    logger.info(f"Writing to {target_table} (mode={mode})")
    (
        df.write.format("delta")
        .mode(mode)
        .option("mergeSchema", "true")
        .saveAsTable(target_table)
    )

    logger.info("Silver transform complete.")
    return df
