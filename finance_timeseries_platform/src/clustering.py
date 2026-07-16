"""
clustering.py

Market regime detection via K-means clustering on Silver OHLCV features.
Uses MLflow for experiment tracking - each run logs parameters, metrics,
and the fitted model so we can compare different K values and feature sets.

Usage:
    from clustering import run_clustering
    run_clustering(spark, n_clusters=4)
"""

from __future__ import annotations

import logging

import mlflow
import mlflow.spark
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

logger = logging.getLogger("clustering")
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Feature config
# ---------------------------------------------------------------------------

# Features selected for clustering:
# - log_return: direction of price movement
# - volatility_20d: short-term nervousness
# - rsi: momentum signal
# - corr_vix_60d: relationship with market-wide fear
#
# Excluded:
# - ma_20/50/200: absolute price levels, not comparable across tickers
# - volume: different scales across tickers
# - volatility_60d: correlated with volatility_20d, adds noise not signal
CLUSTER_FEATURES = ["log_return", "volatility_20d", "rsi", "corr_vix_60d"]

# Regime label mapping based on cluster centroid analysis (Day 13)
# Cluster 0: low volatility, near-zero return, neutral RSI → calm market
# Cluster 1: high return, high volatility, strong VIX correlation → momentum
# Cluster 2: large negative return, highest volatility, idiosyncratic → selloff
REGIME_LABELS = {
    0: "calm_low_vol",
    1: "high_vol_momentum",
    2: "high_vol_selloff",
}


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def prepare_features(
    df: DataFrame,
    feature_cols: list[str] = CLUSTER_FEATURES,
) -> tuple[DataFrame, VectorAssembler, StandardScaler]:
    """
    Assemble and standardize features for K-means.

    Steps:
    1. Drop rows with NULL in any feature column (K-means can't handle NULLs)
    2. VectorAssembler: combine feature columns into a single vector column
    3. StandardScaler: zero mean, unit variance per feature

    Returns the transformed DataFrame plus the fitted assembler and scaler
    (needed later to transform new data for prediction).
    """
    # Drop rows with any NULL in feature columns
    df_clean = df.dropna(subset=feature_cols)

    null_dropped = df.count() - df_clean.count()
    if null_dropped > 0:
        logger.warning(f"Dropped {null_dropped} rows with NULL in feature columns")

    # VectorAssembler: [log_return, volatility_20d, rsi, corr_vix_60d] → features vector
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features_raw",
    )
    df_assembled = assembler.transform(df_clean)

    # StandardScaler: zero mean, unit stddev → features_scaled vector
    scaler = StandardScaler(
        inputCol="features_raw",
        outputCol="features_scaled",
        withMean=True,
        withStd=True,
    )
    scaler_model = scaler.fit(df_assembled)
    df_scaled = scaler_model.transform(df_assembled)

    return df_scaled, assembler, scaler_model


# ---------------------------------------------------------------------------
# Elbow method helper
# ---------------------------------------------------------------------------

def find_optimal_k(
    spark: SparkSession,
    source_table: str = "finance_dev.silver.ohlcv_features",
    k_range: list[int] = [2, 3, 4, 5, 6, 7, 8],
    mlflow_experiment: str = "/finance_ts/clustering_elbow",
) -> dict:
    """
    Run K-means for multiple K values and log Silhouette scores to MLflow.

    The Silhouette score measures how well each point fits its assigned
    cluster vs. other clusters (range -1 to 1, higher = better separation).

    Use the resulting scores to pick the optimal K before running the
    full clustering with run_clustering().
    """
    df = spark.table(source_table)
    df_scaled, _, _ = prepare_features(df)

    mlflow.set_experiment(mlflow_experiment)

    results = {}

    for k in k_range:
        with mlflow.start_run(run_name=f"elbow_k{k}"):
            kmeans = KMeans(
                featuresCol="features_scaled",
                predictionCol="cluster",
                k=k,
                seed=42,
            )
            model = kmeans.fit(df_scaled)

            df_pred = model.transform(df_scaled)

            evaluator = ClusteringEvaluator(
                featuresCol="features_scaled",
                predictionCol="cluster",
                metricName="silhouette",
            )
            silhouette = evaluator.evaluate(df_pred)

            mlflow.log_param("k", k)
            mlflow.log_param("features", CLUSTER_FEATURES)
            mlflow.log_metric("silhouette", silhouette)

            results[k] = silhouette
            logger.info(f"K={k}: silhouette={silhouette:.4f}")

    logger.info(f"Elbow results: {results}")
    return results


# ---------------------------------------------------------------------------
# Main clustering run
# ---------------------------------------------------------------------------

def label_regimes(
    spark: SparkSession,
    target_table: str = "finance_dev.gold.fact_regimes",
    regime_labels: dict = REGIME_LABELS,
) -> DataFrame:
    """
    Assign human-readable regime labels to the fact_regimes table based
    on the REGIME_LABELS mapping. Updates the regime_label column in place.

    Run this after run_clustering() once you've inspected the centroids
    and decided on names (Day 13).
    """
    df = spark.table(target_table)

    # Build a mapping expression from cluster int → label string
    label_expr = F.lit("unknown")
    for cluster_id, label in regime_labels.items():
        label_expr = F.when(
            F.col("cluster") == cluster_id, F.lit(label)
        ).otherwise(label_expr)

    df_labeled = df.withColumn("regime_label", label_expr)

    logger.info(f"Writing labeled regimes to {target_table}")
    df_labeled.write.format("delta").mode("overwrite").saveAsTable(target_table)

    return df_labeled


def run_clustering(
    spark: SparkSession,
    n_clusters: int = 4,
    source_table: str = "finance_dev.silver.ohlcv_features",
    target_table: str = "finance_dev.gold.fact_regimes",
    mlflow_experiment: str = "/finance_ts/clustering",
) -> DataFrame:
    """
    Run K-means clustering with the chosen K, log to MLflow, write results
    to Gold layer as fact_regimes table.

    fact_regimes adds two columns to the Silver feature set:
    - cluster: integer cluster id (0 to K-1)
    - regime_label: human-readable label (populated after Day 13 naming)
    """
    df = spark.table(source_table)
    df_scaled, assembler, scaler_model = prepare_features(df)

    mlflow.set_experiment(mlflow_experiment)

    with mlflow.start_run(run_name=f"kmeans_k{n_clusters}"):

        mlflow.log_param("k", n_clusters)
        mlflow.log_param("features", CLUSTER_FEATURES)
        mlflow.log_param("source_table", source_table)

        kmeans = KMeans(
            featuresCol="features_scaled",
            predictionCol="cluster",
            k=n_clusters,
            seed=42,
        )

        logger.info(f"Fitting K-means with K={n_clusters}...")
        model = kmeans.fit(df_scaled)

        df_pred = model.transform(df_scaled)

        evaluator = ClusteringEvaluator(
            featuresCol="features_scaled",
            predictionCol="cluster",
            metricName="silhouette",
        )
        silhouette = evaluator.evaluate(df_pred)

        mlflow.log_metric("silhouette", silhouette)
        mlflow.spark.log_model(model, "kmeans_model")

        logger.info(f"Silhouette score: {silhouette:.4f}")

        # Add regime_label from REGIME_LABELS mapping
        label_expr = F.lit("unknown")
        for cluster_id, label in REGIME_LABELS.items():
            label_expr = F.when(
                F.col("cluster") == cluster_id, F.lit(label)
            ).otherwise(label_expr)

        df_result = df_pred.withColumn(
            "regime_label", label_expr
        ).drop("features_raw", "features_scaled")

        logger.info(f"Writing {df_result.count()} rows to {target_table}")
        df_result.write.format("delta").mode("overwrite").saveAsTable(target_table)

        logger.info("Clustering complete.")
        return df_result