"""Data ingestion utilities for a real-world credit risk dataset."""

from __future__ import annotations

from pathlib import Path

from pyspark.sql import DataFrame, SparkSession


def create_spark_session(app_name: str = "credit-risk-analysis") -> SparkSession:
    """Create a local Spark session configured for notebook/script execution."""
    return (
        SparkSession.builder.master("local[*]")
        .appName(app_name)
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )


def load_german_credit_data(spark: SparkSession, csv_path: str | Path) -> DataFrame:
    """Load Kaggle German Credit Data from CSV.

    Expected dataset:
    - Kaggle: German Credit Data with `Risk` target column (good/bad).
    """
    path = str(csv_path)
    return (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .csv(path)
    )
