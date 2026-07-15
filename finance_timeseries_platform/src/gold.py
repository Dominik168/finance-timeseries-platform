"""
gold.py

Gold layer: star schema built on top of the Silver OHLCV features table.

Tables produced:
    - dim_date    : calendar dimension, pre-generated for a fixed date range
    - dim_ticker  : SCD Type 2 dimension tracking when each ticker entered
                    our tracked universe (supports inferred members for
                    late-arriving facts)
    - fact_ohlcv  : insert-only fact table at ticker x day grain, referencing
                    dim_date and dim_ticker via surrogate keys

Usage:
    from gold import build_dim_date, build_dim_ticker, build_fact_ohlcv

    build_dim_date(spark)
    build_dim_ticker(spark)
    build_fact_ohlcv(spark)
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import (
    BooleanType,
    DateType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

logger = logging.getLogger("gold")
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# dim_date
# ---------------------------------------------------------------------------

DIM_DATE_SCHEMA = StructType(
    [
        StructField("date_sk", IntegerType(), False),   # YYYYMMDD as int
        StructField("full_date", DateType(), False),
        StructField("day_of_week", StringType(), False),
        StructField("day_of_month", IntegerType(), False),
        StructField("week_of_year", IntegerType(), False),
        StructField("month", IntegerType(), False),
        StructField("month_name", StringType(), False),
        StructField("quarter", IntegerType(), False),
        StructField("year", IntegerType(), False),
        StructField("is_weekend", BooleanType(), False),
    ]
)


def _generate_date_range(start: date, end: date) -> list[date]:
    """Generate every calendar date between start and end (inclusive)."""
    days = (end - start).days
    return [start + timedelta(days=i) for i in range(days + 1)]


def build_dim_date(
    spark: SparkSession,
    start_date: str = "2015-01-01",
    end_date: str = "2030-12-31",
    target_table: str = "finance_dev.gold.dim_date",
) -> DataFrame:
    """
    Build the date dimension for a fixed calendar range.

    This is a conformed dimension - it's built once and is intended to be
    shared by every fact table in the Gold layer (fact_ohlcv today, and any
    future fact tables like fact_sentiment or fact_trades), so that "month"
    or "quarter" always means exactly the same thing across the warehouse.

    Idempotent: always overwrites, since the calendar itself never changes.
    """
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)

    rows = []
    for d in _generate_date_range(start, end):
        rows.append(
            (
                int(d.strftime("%Y%m%d")),
                d,
                d.strftime("%A"),
                d.day,
                int(d.strftime("%W")),
                d.month,
                d.strftime("%B"),
                (d.month - 1) // 3 + 1,
                d.year,
                d.weekday() >= 5,
            )
        )

    df = spark.createDataFrame(rows, schema=DIM_DATE_SCHEMA)

    logger.info(f"Writing {df.count()} rows to {target_table}")
    df.write.format("delta").mode("overwrite").saveAsTable(target_table)

    return df


# ---------------------------------------------------------------------------
# dim_ticker (SCD Type 2)
# ---------------------------------------------------------------------------

DIM_TICKER_SCHEMA = StructType(
    [
        StructField("ticker_sk", IntegerType(), False),
        StructField("ticker", StringType(), False),
        StructField("company_name", StringType(), True),
        StructField("sector", StringType(), True),
        StructField("market_cap_category", StringType(), True),
        StructField("effective_from", DateType(), False),
        StructField("effective_to", DateType(), True),
        StructField("is_current", BooleanType(), False),
        StructField("is_inferred", BooleanType(), False),  # True = skeleton row
    ]
)


def _classify_market_cap(price_col: str = "adj_close") -> "pyspark.sql.Column":
    """
    Simplified market cap category proxy based on price alone.

    NOTE: This is a deliberate simplification for demonstrating SCD2 update
    mechanics, not a real market cap calculation (which requires shares
    outstanding, not available from yfinance's .history() endpoint).
    Real implementation would use price * shares_outstanding against
    standard thresholds (microcap < $300M, smallcap < $2B, etc).
    """
    return (
        F.when(F.col(price_col) < 5, F.lit("microcap"))
        .when(F.col(price_col) < 20, F.lit("smallcap"))
        .otherwise(F.lit("midcap_plus"))
    )


def build_dim_ticker(
    spark: SparkSession,
    source_table: str = "finance_dev.silver.ohlcv_features",
    target_table: str = "finance_dev.gold.dim_ticker",
) -> DataFrame:
    """
    Build the ticker dimension as SCD Type 2.

    First run: creates one current row per distinct ticker found in Silver,
    with effective_from = the ticker's earliest trade_date, and
    market_cap_category derived from its latest known price.

    market_cap_category is the tracked SCD2 attribute here - see
    update_dim_ticker_scd2() for the logic that detects changes on
    subsequent runs and creates new dimension versions.
    """
    df_silver = spark.table(source_table)

    window_latest = Window.partitionBy("ticker").orderBy(F.desc("trade_date"))

    df_latest_price = (
        df_silver
        .filter(F.col("adj_close").isNotNull())
        .filter(~F.isnan(F.col("adj_close")))
        .withColumn("_rn", F.row_number().over(window_latest))
        .filter(F.col("_rn") == 1)
        .select("ticker", "adj_close")
    )

    df_first_seen = (
        df_silver.groupBy("ticker")
        .agg(F.min("trade_date").alias("effective_from"))
    )

    df_tickers = df_first_seen.join(df_latest_price, on="ticker", how="inner")

    window_sk = Window.orderBy("ticker")
    df_with_sk = df_tickers.withColumn("ticker_sk", F.row_number().over(window_sk))

    df_dim = (
        df_with_sk.withColumn("company_name", F.lit(None).cast("string"))
        .withColumn("sector", F.lit(None).cast("string"))
        .withColumn("market_cap_category", _classify_market_cap("adj_close"))
        .withColumn("effective_to", F.lit(None).cast("date"))
        .withColumn("is_current", F.lit(True))
        .withColumn("is_inferred", F.lit(False))
        .select(
            "ticker_sk", "ticker", "company_name", "sector",
            "market_cap_category", "effective_from", "effective_to",
            "is_current", "is_inferred",
        )
    )

    logger.info(f"Writing {df_dim.count()} rows to {target_table}")
    df_dim.write.format("delta").mode("overwrite").saveAsTable(target_table)

    return df_dim


def update_dim_ticker_scd2(
    spark: SparkSession,
    source_table: str = "finance_dev.silver.ohlcv_features",
    dim_table: str = "finance_dev.gold.dim_ticker",
) -> dict:
    """
    Apply SCD Type 2 update logic: for each ticker, compare its current
    market_cap_category (derived from latest price) against the value
    stored in dim_ticker. If it changed, close the old row and insert a
    new current version - this is the real SCD2 mechanism, as opposed to
    build_dim_ticker() which only does the initial load.

    Run this on subsequent pipeline runs (not on first load).

    Returns a dict summarizing how many tickers changed category.
    """
    df_silver = spark.table(source_table)
    df_dim = spark.table(dim_table)

    # Compute current category from latest VALID known price per ticker
    # (filter out NULL/NaN to avoid using incomplete intraday data)
    window_latest = Window.partitionBy("ticker").orderBy(F.desc("trade_date"))
    df_current_state = (
        df_silver
        .filter(F.col("adj_close").isNotNull())
        .filter(~F.isnan(F.col("adj_close")))
        .withColumn("_rn", F.row_number().over(window_latest))
        .filter(F.col("_rn") == 1)
        .select(
            "ticker",
            _classify_market_cap("adj_close").alias("new_category"),
        )
    )

    df_dim_current = df_dim.filter(F.col("is_current") == True)

    # Compare new vs. stored category
    df_compare = df_dim_current.join(
        df_current_state, on="ticker", how="inner"
    ).select(
        "ticker_sk", "ticker", "market_cap_category", "new_category"
    )

    df_changed = df_compare.filter(
        F.col("market_cap_category") != F.col("new_category")
    )

    changed_tickers = [row["ticker"] for row in df_changed.collect()]

    if not changed_tickers:
        logger.info("No market_cap_category changes detected - dim_ticker unchanged.")
        return {"changed_count": 0, "changed_tickers": []}

    logger.info(f"Detected category change for {len(changed_tickers)} ticker(s): {changed_tickers}")

    today = date.today()

    # Step 1: close out old rows for changed tickers (set effective_to, is_current=False)
    df_to_close = (
        df_dim.filter(F.col("is_current") == True)
        .filter(F.col("ticker").isin(changed_tickers))
        .withColumn("effective_to", F.lit(today))
        .withColumn("is_current", F.lit(False))
    )

    df_unchanged = df_dim.filter(
        ~((F.col("is_current") == True) & (F.col("ticker").isin(changed_tickers)))
    )

    # Step 2: build new current rows with the new category
    max_sk = df_dim.agg(F.max("ticker_sk")).collect()[0][0] or 0

    new_rows = []
    for i, row in enumerate(df_changed.collect(), start=1):
        new_rows.append(
            (
                max_sk + i,
                row["ticker"],
                None,
                None,
                row["new_category"],
                today,
                None,
                True,
                False,
            )
        )

    df_new_versions = spark.createDataFrame(new_rows, schema=DIM_TICKER_SCHEMA)

    # Combine: unchanged rows + closed-out old rows + new current rows
    df_result = df_unchanged.union(df_to_close).union(df_new_versions)

    logger.info(f"Writing updated dim_ticker ({df_result.count()} total rows) to {dim_table}")
    df_result.write.format("delta").mode("overwrite").saveAsTable(dim_table)

    return {"changed_count": len(changed_tickers), "changed_tickers": changed_tickers}


def add_inferred_members(
    spark: SparkSession,
    fact_tickers: list[str],
    dim_table: str = "finance_dev.gold.dim_ticker",
) -> None:
    """
    Late-arriving dimension handling: if a ticker appears in fact data but
    has no row in dim_ticker yet, insert a minimal "inferred member" row
    for it instead of letting the fact row become an orphan.

    This is the Kimball "inferred member" pattern - the skeleton row has
    only ticker_sk and ticker populated; everything else is NULL/Unknown
    and is_inferred=True flags it for later enrichment.
    """
    df_dim = spark.table(dim_table)
    existing_tickers = {row["ticker"] for row in df_dim.select("ticker").distinct().collect()}

    missing = [t for t in fact_tickers if t not in existing_tickers]

    if not missing:
        logger.info("No inferred members needed - all fact tickers exist in dim_ticker.")
        return

    logger.warning(f"Found {len(missing)} ticker(s) missing from dim_ticker: {missing}")

    max_sk = df_dim.agg(F.max("ticker_sk")).collect()[0][0] or 0

    inferred_rows = []
    for i, ticker in enumerate(missing, start=1):
        inferred_rows.append(
            (
                max_sk + i,
                ticker,
                None,  # company_name unknown
                None,  # sector unknown
                None,  # market_cap_category unknown
                date.today(),  # best guess: today
                None,
                True,
                True,  # is_inferred
            )
        )

    df_inferred = spark.createDataFrame(inferred_rows, schema=DIM_TICKER_SCHEMA)

    logger.info(f"Appending {df_inferred.count()} inferred member row(s) to {dim_table}")
    df_inferred.write.format("delta").mode("append").saveAsTable(dim_table)


# ---------------------------------------------------------------------------
# fact_ohlcv
# ---------------------------------------------------------------------------

def build_fact_ohlcv(
    spark: SparkSession,
    source_table: str = "finance_dev.silver.ohlcv_features",
    dim_ticker_table: str = "finance_dev.gold.dim_ticker",
    dim_date_table: str = "finance_dev.gold.dim_date",
    target_table: str = "finance_dev.gold.fact_ohlcv",
) -> DataFrame:
    """
    Build the OHLCV fact table at ticker x day grain.

    Insert-only (Type 0): a trading day's data, once recorded, never
    changes - there is no SCD logic on a fact table, only on dimensions.

    Before writing, handles late-arriving dimension members (any ticker
    in Silver not yet in dim_ticker gets an inferred row), so the fact
    table never ends up with orphan foreign keys.
    """
    df_silver = spark.table(source_table)

    # Late-arriving dimension handling
    fact_tickers = [row["ticker"] for row in df_silver.select("ticker").distinct().collect()]
    add_inferred_members(spark, fact_tickers, dim_table=dim_ticker_table)

    df_dim_ticker = spark.table(dim_ticker_table).filter(F.col("is_current") == True)
    df_dim_date = spark.table(dim_date_table)

    df_fact = (
        df_silver
        .join(
            df_dim_ticker.select("ticker", "ticker_sk"),
            on="ticker",
            how="inner",  # inner is correct here: every fact row must have a dimension match
        )
        .join(
            df_dim_date.select(F.col("full_date").alias("trade_date"), "date_sk"),
            on="trade_date",
            how="inner",
        )
        .select(
            "ticker_sk",
            "date_sk",
            "ticker",
            "trade_date",
            "open", "high", "low", "close", "adj_close", "volume",
            "is_illiquid", "log_return",
            "volatility_20d", "volatility_60d",
            "ma_20", "ma_50", "ma_200",
            "rsi",
            "corr_vix_60d",
        )
    )

    logger.info(f"Writing {df_fact.count()} rows to {target_table}")
    df_fact.write.format("delta").mode("overwrite").saveAsTable(target_table)

    return df_fact


# ---------------------------------------------------------------------------
# Referential integrity check
# ---------------------------------------------------------------------------

def check_referential_integrity(
    spark: SparkSession,
    fact_table: str = "finance_dev.gold.fact_ohlcv",
    dim_ticker_table: str = "finance_dev.gold.dim_ticker",
    dim_date_table: str = "finance_dev.gold.dim_date",
) -> dict:
    """
    Verify that every foreign key in the fact table has a matching row in
    its dimension table. Returns a dict with orphan counts - both should
    be 0 in a healthy pipeline.
    """
    df_fact = spark.table(fact_table)
    df_dim_ticker = spark.table(dim_ticker_table)
    df_dim_date = spark.table(dim_date_table)

    orphan_tickers = (
        df_fact.select("ticker_sk")
        .distinct()
        .join(df_dim_ticker.select("ticker_sk"), on="ticker_sk", how="left_anti")
        .count()
    )

    orphan_dates = (
        df_fact.select("date_sk")
        .distinct()
        .join(df_dim_date.select("date_sk"), on="date_sk", how="left_anti")
        .count()
    )

    result = {
        "orphan_ticker_keys": orphan_tickers,
        "orphan_date_keys": orphan_dates,
        "is_healthy": orphan_tickers == 0 and orphan_dates == 0,
    }

    if result["is_healthy"]:
        logger.info("Referential integrity check passed - no orphan foreign keys.")
    else:
        logger.error(f"Referential integrity check FAILED: {result}")

    return result