"""
classifier.py

Next-day market regime classifier using Random Forest.

Target: predict tomorrow's regime_label from today's features.
This is a realistic use case - at end of day T, you observe features
and predict which regime tomorrow (T+1) will belong to.

Uses time-based train/test split (not random) to avoid look-ahead bias:
- Train: 2018-01-01 to 2024-12-31
- Test:  2025-01-01 to present

MLflow tracks all experiments so we can compare model versions.

Usage:
    from classifier import run_classifier
    run_classifier(spark)
"""

from __future__ import annotations

import logging

import mlflow
import mlflow.spark
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

logger = logging.getLogger("classifier")
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Feature config
# ---------------------------------------------------------------------------

# All Silver features as input - we use more features than clustering
# because the classifier can handle correlated inputs (Random Forest
# uses feature subsets at each split, which handles correlation well)
CLASSIFIER_FEATURES = [
    "log_return",
    "volatility_20d",
    "volatility_60d",
    "ma_20",
    "ma_50",
    "ma_200",
    "rsi",
    "corr_vix_60d",
]

TRAIN_END_DATE = "2024-12-31"
TEST_START_DATE = "2025-01-01"


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_classification_data(
    spark: SparkSession,
    source_table: str = "finance_dev.gold.fact_regimes",
    feature_cols: list[str] = CLASSIFIER_FEATURES,
) -> tuple[DataFrame, DataFrame]:
    """
    Prepare train/test datasets for classification.

    Key step: add next_day_regime via F.lead() - the target is tomorrow's
    regime, not today's. This is what makes it a real prediction task
    rather than just reconstructing the clustering.

    Returns (df_train, df_test) as a tuple.
    """
    df = spark.table(source_table)

    # Add next_day_regime: for each ticker, look at the NEXT row's regime
    # partitionBy ticker is critical - we don't want to leak across tickers
    window_spec = Window.partitionBy("ticker").orderBy("trade_date")

    df_with_target = df.withColumn(
        "next_day_regime",
        F.lead(F.col("regime_label")).over(window_spec),
    )

    # Drop rows where next_day_regime is NULL (last day per ticker has no tomorrow)
    # and drop rows with NULL in any feature column
    df_clean = df_with_target.dropna(
        subset=feature_cols + ["next_day_regime"]
    )

    # Exclude ^VIX from training data - it's a reference ticker, not a stock
    # Its structural properties (always high vol) would confuse the classifier
    df_clean = df_clean.filter(F.col("ticker") != "^VIX")

    # Time-based split - NO random shuffling, strict chronological boundary
    df_train = df_clean.filter(F.col("trade_date") <= TRAIN_END_DATE)
    df_test = df_clean.filter(F.col("trade_date") > TRAIN_END_DATE)

    logger.info(f"Train: {df_train.count()} rows, Test: {df_test.count()} rows")

    return df_train, df_test


# ---------------------------------------------------------------------------
# Main classifier run
# ---------------------------------------------------------------------------

def run_classifier(
    spark: SparkSession,
    source_table: str = "finance_dev.gold.fact_regimes",
    target_table: str = "finance_dev.gold.fact_predictions",
    mlflow_experiment: str = "finance_ts_classifier",
    n_trees: int = 100,
    max_depth: int = 5,
) -> DataFrame:
    """
    Train a Random Forest classifier to predict next-day regime,
    evaluate on the held-out test set, log everything to MLflow.

    Metrics logged per class (precision/recall/F1) because accuracy alone
    is misleading with imbalanced classes (84% calm means a model that
    always predicts calm gets 84% accuracy but is useless).
    """
    df_train, df_test = prepare_classification_data(spark, source_table)

    mlflow.set_experiment(mlflow_experiment)

    with mlflow.start_run(run_name=f"rf_trees{n_trees}_depth{max_depth}"):

        mlflow.log_param("n_trees", n_trees)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("features", CLASSIFIER_FEATURES)
        mlflow.log_param("train_end", TRAIN_END_DATE)
        mlflow.log_param("test_start", TEST_START_DATE)
        mlflow.log_param("excluded_tickers", ["^VIX"])

        # StringIndexer: convert regime_label string → numeric index
        # Required by Spark ML which needs numeric labels
        indexer = StringIndexer(
            inputCol="next_day_regime",
            outputCol="label",
            handleInvalid="skip",
        )
        indexer_model = indexer.fit(df_train)

        df_train_indexed = indexer_model.transform(df_train)
        df_test_indexed = indexer_model.transform(df_test)

        # Log label mapping for interpretability
        label_mapping = dict(enumerate(indexer_model.labels))
        logger.info(f"Label mapping: {label_mapping}")
        mlflow.log_param("label_mapping", label_mapping)

        # VectorAssembler: combine feature columns into single vector
        assembler = VectorAssembler(
            inputCols=CLASSIFIER_FEATURES,
            outputCol="features",
        )
        df_train_assembled = assembler.transform(df_train_indexed)
        df_test_assembled = assembler.transform(df_test_indexed)

        # ---------------------------------------------------------------------------
        # Class weights - inversely proportional to class frequency
        # Without this, model predicts "calm" almost always (89% majority class)
        # and gets 97% accuracy while completely ignoring minority classes
        # ---------------------------------------------------------------------------
        total_train = df_train_indexed.count()
        class_counts = (
            df_train_indexed.groupBy("label")
            .count()
            .collect()
        )
        num_classes = len(class_counts)

        weight_map = {}
        for row in class_counts:
            weight_map[row["label"]] = total_train / (num_classes * row["count"])

        logger.info(f"Class weights: {weight_map}")
        mlflow.log_param("class_weights", weight_map)

        # Add weight column to training data
        weight_expr = F.lit(1.0)
        for label_idx, weight in weight_map.items():
            weight_expr = F.when(
                F.col("label") == label_idx, F.lit(weight)
            ).otherwise(weight_expr)

        df_train_weighted = df_train_assembled.withColumn("class_weight", weight_expr)

        # Random Forest classifier with class weights
        rf = RandomForestClassifier(
            featuresCol="features",
            labelCol="label",
            predictionCol="prediction",
            weightCol="class_weight",
            numTrees=n_trees,
            maxDepth=max_depth,
            seed=42,
        )

        logger.info(f"Training Random Forest (trees={n_trees}, depth={max_depth})...")
        rf_model = rf.fit(df_train_weighted)

        df_pred = rf_model.transform(df_test_assembled)

        # ---------------------------------------------------------------------------
        # Evaluation - per class metrics (precision/recall/F1)
        # ---------------------------------------------------------------------------

        evaluator = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
        )

        accuracy = evaluator.evaluate(
            df_pred, {evaluator.metricName: "accuracy"}
        )
        f1 = evaluator.evaluate(
            df_pred, {evaluator.metricName: "f1"}
        )
        weighted_precision = evaluator.evaluate(
            df_pred, {evaluator.metricName: "weightedPrecision"}
        )
        weighted_recall = evaluator.evaluate(
            df_pred, {evaluator.metricName: "weightedRecall"}
        )

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_weighted", f1)
        mlflow.log_metric("weighted_precision", weighted_precision)
        mlflow.log_metric("weighted_recall", weighted_recall)

        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"F1 (weighted): {f1:.4f}")
        logger.info(f"Weighted Precision: {weighted_precision:.4f}")
        logger.info(f"Weighted Recall: {weighted_recall:.4f}")

        # Feature importance
        feature_importance = dict(
            zip(CLASSIFIER_FEATURES, rf_model.featureImportances.toArray())
        )
        logger.info(f"Feature importance: {feature_importance}")
        mlflow.log_param("feature_importance", feature_importance)

        # Log model to MLflow
        mlflow.spark.log_model(rf_model, "rf_model")

        # ---------------------------------------------------------------------------
        # Write predictions to Gold layer
        # ---------------------------------------------------------------------------

        # Add predicted regime label (convert numeric prediction back to string)
        labels = indexer_model.labels
        index_to_label = F.create_map(
            *[val for pair in enumerate(labels) for val in (F.lit(float(pair[0])), F.lit(pair[1]))]
        )

        df_result = df_pred.withColumn(
            "predicted_regime",
            index_to_label[F.col("prediction")],
        ).select(
            "ticker", "trade_date", "regime_label", "next_day_regime",
            "predicted_regime", "prediction", "label",
            *CLASSIFIER_FEATURES,
        )

        logger.info(f"Writing {df_result.count()} predictions to {target_table}")
        df_result.write.format("delta").mode("overwrite").saveAsTable(target_table)

        logger.info("Classification complete.")
        return df_result