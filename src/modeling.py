"""PySpark ML modeling for credit default risk classification."""

from __future__ import annotations

from pathlib import Path

from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier, LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


FEATURE_COLUMNS = [
    "Age",
    "Sex",
    "Job",
    "Housing",
    "Saving accounts",
    "Checking account",
    "Credit amount",
    "Duration",
    "Purpose",
]


def _prepare_label(df: DataFrame) -> DataFrame:
    return df.withColumn("label", F.when(F.col("Risk") == "bad", F.lit(1.0)).otherwise(F.lit(0.0)))


def train_and_evaluate(df: DataFrame, output_dir: Path) -> DataFrame:
    """Train benchmark classifiers and export ROC-AUC comparison."""
    modeled_df = _prepare_label(df)

    categorical = ["Sex", "Housing", "Saving accounts", "Checking account", "Purpose"]
    numeric = ["Age", "Job", "Credit amount", "Duration"]

    indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in categorical]
    encoders = [OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_ohe") for c in categorical]

    assembler_inputs = numeric + [f"{c}_ohe" for c in categorical]
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

    classifiers = {
        "logistic_regression": LogisticRegression(featuresCol="features", labelCol="label", maxIter=60),
        "random_forest": RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=120, maxDepth=7),
        "gradient_boosted_trees": GBTClassifier(featuresCol="features", labelCol="label", maxIter=80, maxDepth=5),
    }

    train_df, test_df = modeled_df.randomSplit([0.75, 0.25], seed=42)
    evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")

    rows = []
    for name, classifier in classifiers.items():
        pipeline = Pipeline(stages=[*indexers, *encoders, assembler, classifier])
        model = pipeline.fit(train_df)
        predictions = model.transform(test_df)
        auc = evaluator.evaluate(predictions)

        rows.append((name, float(round(auc, 4))))

    result = df.sparkSession.createDataFrame(rows, ["model", "roc_auc"]).orderBy(F.desc("roc_auc"))

    output_dir.mkdir(parents=True, exist_ok=True)
    result.coalesce(1).write.mode("overwrite").option("header", True).csv(str(output_dir / "model_performance"))
    return result
