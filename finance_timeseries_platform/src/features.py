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
    volatility_windows: list[int]
    ma_windows: list[int]
    ma_price_column: str
    rsi_window: int
    rsi_price_column: str
    vix_ticker: str
    vix_correlation_window: int


def load_config(config_path: str) -> FeaturesConfig:
    """Read features.yaml."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    cleaning_cfg = raw.get("cleaning", {})
    returns_cfg = raw.get("returns", {})
    volatility_cfg = raw.get("volatility", {})
    ma_cfg = raw.get("moving_averages", {})
    rsi_cfg = raw.get("rsi", {})
    cross_asset_cfg = raw.get("cross_asset", {})

    return FeaturesConfig(
        volume_exempt_tickers=cleaning_cfg.get("volume_exempt_tickers", []),
        illiquid_volume_threshold=cleaning_cfg.get("illiquid_volume_threshold", 0),
        returns_method=returns_cfg.get("method", "log"),
        returns_price_column=returns_cfg.get("price_column", "adj_close"),
        volatility_windows=volatility_cfg.get("windows", [20, 60]),
        ma_windows=ma_cfg.get("windows", [20, 50, 200]),
        ma_price_column=ma_cfg.get("price_column", "adj_close"),
        rsi_window=rsi_cfg.get("window", 14),
        rsi_price_column=rsi_cfg.get("price_column", "adj_close"),
        vix_ticker=cross_asset_cfg.get("vix_ticker", "^VIX"),
        vix_correlation_window=cross_asset_cfg.get("correlation_window", 60),
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
# Rolling volatility
# ---------------------------------------------------------------------------

def add_rolling_volatility(df: DataFrame, cfg: FeaturesConfig) -> DataFrame:
    """
    Add volatility_{N}d columns: rolling stddev of log_return over a
    backward-looking window of N trading days, per ticker.

    Backward-looking is essential here - rowsBetween(-(N-1), 0) means
    "the current row plus the N-1 rows before it", never future rows.
    Using future data here would be a look-ahead bias bug.
    """
    for window_size in cfg.volatility_windows:
        window_spec = (
            Window.partitionBy("ticker")
            .orderBy("trade_date")
            .rowsBetween(-(window_size - 1), 0)
        )
        col_name = f"volatility_{window_size}d"
        df = df.withColumn(col_name, F.stddev(F.col("log_return")).over(window_spec))

    return df


# ---------------------------------------------------------------------------
# Moving averages
# ---------------------------------------------------------------------------

def add_moving_averages(df: DataFrame, cfg: FeaturesConfig) -> DataFrame:
    """
    Add ma_{N} columns: rolling average of the price column over a
    backward-looking window of N trading days, per ticker.
    """
    price_col = cfg.ma_price_column

    for window_size in cfg.ma_windows:
        window_spec = (
            Window.partitionBy("ticker")
            .orderBy("trade_date")
            .rowsBetween(-(window_size - 1), 0)
        )
        col_name = f"ma_{window_size}"
        df = df.withColumn(col_name, F.avg(F.col(price_col)).over(window_spec))

    return df


# ---------------------------------------------------------------------------
# RSI (Relative Strength Index)
# ---------------------------------------------------------------------------

def add_rsi(df: DataFrame, cfg: FeaturesConfig) -> DataFrame:
    """
    Add an `rsi` column: the standard 14-period Relative Strength Index.

    RSI = 100 - (100 / (1 + RS)), where RS = avg_gain / avg_loss over the
    lookback window. Gains and losses are split from the daily price change,
    then averaged separately using the same backward-looking window pattern
    as volatility and moving averages.
    """
    price_col = cfg.rsi_price_column
    window_size = cfg.rsi_window

    window_order = Window.partitionBy("ticker").orderBy("trade_date")
    window_roll = (
        Window.partitionBy("ticker")
        .orderBy("trade_date")
        .rowsBetween(-(window_size - 1), 0)
    )

    df_change = df.withColumn(
        "_price_change", F.col(price_col) - F.lag(F.col(price_col)).over(window_order)
    )

    df_gain_loss = df_change.withColumn(
        "_gain", F.when(F.col("_price_change") > 0, F.col("_price_change")).otherwise(F.lit(0.0))
    ).withColumn(
        "_loss", F.when(F.col("_price_change") < 0, -F.col("_price_change")).otherwise(F.lit(0.0))
    )

    df_avg = df_gain_loss.withColumn(
        "_avg_gain", F.avg(F.col("_gain")).over(window_roll)
    ).withColumn(
        "_avg_loss", F.avg(F.col("_loss")).over(window_roll)
    )

    df_rs = df_avg.withColumn(
        "_rs",
        F.when(F.col("_avg_loss") == 0, F.lit(None)).otherwise(
            F.col("_avg_gain") / F.col("_avg_loss")
        ),
    )

    df_rsi = df_rs.withColumn(
        "rsi",
        F.when(F.col("_rs").isNotNull(), F.lit(100) - (F.lit(100) / (F.lit(1) + F.col("_rs"))))
        .otherwise(F.lit(100.0)),  # avg_loss == 0 means pure gains -> RSI saturates at 100
    )

    return df_rsi.drop("_price_change", "_gain", "_loss", "_avg_gain", "_avg_loss", "_rs")


# ---------------------------------------------------------------------------
# Cross-asset: VIX correlation
# ---------------------------------------------------------------------------

def add_vix_correlation(df: DataFrame, cfg: FeaturesConfig) -> DataFrame:
    """
    Add corr_vix_60d: rolling Pearson correlation between each ticker's
    log_return and VIX log_return over a backward-looking window.

    Requires a self-join on trade_date to align VIX returns with each
    ticker's rows before computing the rolling correlation.

    Note: VIX itself will have corr_vix_60d = 1.0 (perfect self-correlation),
    which is expected and correct — it can be filtered out downstream if needed.
    """
    df_vix = (
        df.filter(F.col("ticker") == cfg.vix_ticker)
        .select(
            "trade_date",
            F.col("log_return").alias("vix_log_return"),
        )
    )

    df_joined = df.join(df_vix, on="trade_date", how="left")

    window_spec = (
        Window.partitionBy("ticker")
        .orderBy("trade_date")
        .rowsBetween(-(cfg.vix_correlation_window - 1), 0)
    )

    df_corr = df_joined.withColumn(
        "corr_vix_60d",
        F.corr(F.col("log_return"), F.col("vix_log_return")).over(window_spec),
    )

    return df_corr.drop("vix_log_return")


# ---------------------------------------------------------------------------
# Post-processing: NaN → NULL cleanup
# ---------------------------------------------------------------------------

def clean_nan(df: DataFrame) -> DataFrame:
    """
    Replace NaN values with NULL in all numeric columns.

    NaN (Not a Number) is different from NULL in Spark - it's a special
    floating point value that can appear when computing stddev/corr/division
    over incomplete or edge-case data (e.g. last ingestion day with partial
    market data from yfinance). NaN propagates silently through calculations
    and is not caught by isNull() checks, so we explicitly replace it with
    NULL here as a final cleanup step.
    """
    from pyspark.sql.functions import isnan

    numeric_cols = [
        "log_return",
        "volatility_20d", "volatility_60d",
        "ma_20", "ma_50", "ma_200",
        "rsi",
        "corr_vix_60d",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df = df.withColumn(
                col,
                F.when(isnan(F.col(col)), F.lit(None)).otherwise(F.col(col)),
            )

    return df


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
    df = add_rolling_volatility(df, cfg)
    df = add_moving_averages(df, cfg)
    df = add_rsi(df, cfg)
    df = add_vix_correlation(df, cfg)
    df = clean_nan(df)

    logger.info(f"Writing to {target_table} (mode={mode})")
    (
        df.write.format("delta")
        .mode(mode)
        .option("mergeSchema", "true")
        .saveAsTable(target_table)
    )

    logger.info("Silver transform complete.")
    return df