"""
ingestion.py

Generic, config-driven ingestion module for the Bronze layer.
Reads ticker universe and parameters from sources.yaml, fetches daily
OHLCV data via yfinance, and writes the result as a single Delta table
(finance_dev.bronze.ohlcv_daily) with `ticker` as a column.

Usage (from a Databricks notebook or job):

    from ingestion import run_ingestion
    run_ingestion(spark, config_path="config/sources.yaml")

Design principle: adding a new ticker means editing sources.yaml,
not touching this file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import yaml
import yfinance as yf
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
    DateType,
)

logger = logging.getLogger("ingestion")
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

@dataclass
class IngestionConfig:
    tickers: list[str]
    start_date: str
    end_date: str | None
    interval: str
    retry_attempts: int
    retry_delay_seconds: int
    fail_on_missing_ticker: bool


def load_config(config_path: str) -> IngestionConfig:
    """Read sources.yaml and flatten the ticker universe into a single list."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    universe = raw["universe"]
    tickers: list[str] = []
    for group_name, group_tickers in universe.items():
        tickers.extend(group_tickers)

    ingestion_cfg = raw.get("ingestion", {})
    date_cfg = raw.get("date_range", {})

    return IngestionConfig(
        tickers=tickers,
        start_date=date_cfg.get("start", "2018-01-01"),
        end_date=date_cfg.get("end"),
        interval=ingestion_cfg.get("interval", "1d"),
        retry_attempts=ingestion_cfg.get("retry_attempts", 3),
        retry_delay_seconds=ingestion_cfg.get("retry_delay_seconds", 5),
        fail_on_missing_ticker=ingestion_cfg.get("fail_on_missing_ticker", False),
    )


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

BRONZE_SCHEMA = StructType(
    [
        StructField("ticker", StringType(), nullable=False),
        StructField("trade_date", DateType(), nullable=False),
        StructField("open", DoubleType(), nullable=True),
        StructField("high", DoubleType(), nullable=True),
        StructField("low", DoubleType(), nullable=True),
        StructField("close", DoubleType(), nullable=True),
        StructField("adj_close", DoubleType(), nullable=True),
        StructField("volume", LongType(), nullable=True),
        StructField("ingestion_timestamp", StringType(), nullable=False),
    ]
)


# ---------------------------------------------------------------------------
# Fetch logic
# ---------------------------------------------------------------------------

def fetch_ticker_history(
    ticker: str,
    start_date: str,
    end_date: str | None,
    interval: str,
    retry_attempts: int,
    retry_delay_seconds: int,
) -> "pd.DataFrame | None":
    """Fetch OHLCV history for a single ticker via yfinance, with retries."""
    import time
    import pandas as pd  # noqa: F401  (imported for type hint clarity)

    for attempt in range(1, retry_attempts + 1):
        try:
            hist = yf.Ticker(ticker).history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=False,
            )
            if hist is None or hist.empty:
                logger.warning(f"[{ticker}] No data returned (attempt {attempt})")
                return None
            return hist
        except Exception as exc:
            logger.warning(f"[{ticker}] Fetch failed (attempt {attempt}/{retry_attempts}): {exc}")
            if attempt < retry_attempts:
                time.sleep(retry_delay_seconds)
            else:
                logger.error(f"[{ticker}] Giving up after {retry_attempts} attempts")
                return None


def to_bronze_rows(ticker: str, hist) -> list[tuple]:
    """Convert a yfinance history DataFrame into rows matching BRONZE_SCHEMA."""
    now_iso = datetime.utcnow().isoformat()
    rows = []
    for idx, row in hist.iterrows():
        rows.append(
            (
                ticker,
                idx.date(),
                float(row.get("Open")) if row.get("Open") is not None else None,
                float(row.get("High")) if row.get("High") is not None else None,
                float(row.get("Low")) if row.get("Low") is not None else None,
                float(row.get("Close")) if row.get("Close") is not None else None,
                float(row.get("Adj Close")) if "Adj Close" in row and row.get("Adj Close") is not None else None,
                int(row.get("Volume")) if row.get("Volume") is not None else None,
                now_iso,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_ingestion(
    spark: SparkSession,
    config_path: str = "config/sources.yaml",
    target_table: str = "finance_dev.bronze.ohlcv_daily",
    mode: str = "overwrite",
) -> DataFrame:
    """
    Run full ingestion: read config, fetch all tickers, write Bronze Delta table.

    Returns the resulting Spark DataFrame for inspection/testing.
    """
    cfg = load_config(config_path)
    logger.info(f"Loaded config: {len(cfg.tickers)} tickers, range {cfg.start_date} to {cfg.end_date or 'today'}")

    all_rows: list[tuple] = []
    failed_tickers: list[str] = []

    for ticker in cfg.tickers:
        logger.info(f"Fetching {ticker}...")
        hist = fetch_ticker_history(
            ticker=ticker,
            start_date=cfg.start_date,
            end_date=cfg.end_date,
            interval=cfg.interval,
            retry_attempts=cfg.retry_attempts,
            retry_delay_seconds=cfg.retry_delay_seconds,
        )
        if hist is None:
            failed_tickers.append(ticker)
            if not cfg.fail_on_missing_ticker:
                continue
            else:
                raise RuntimeError(f"Failed to fetch required ticker: {ticker}")

        all_rows.extend(to_bronze_rows(ticker, hist))

    if failed_tickers:
        logger.warning(f"Tickers with no data: {failed_tickers}")

    if not all_rows:
        raise RuntimeError("No data fetched for any ticker - aborting write.")

    df = spark.createDataFrame(all_rows, schema=BRONZE_SCHEMA)

    logger.info(f"Writing {df.count()} rows to {target_table} (mode={mode})")
    (
        df.write.format("delta")
        .mode(mode)
        .option("mergeSchema", "true")
        .saveAsTable(target_table)
    )

    logger.info("Ingestion complete.")
    return df
