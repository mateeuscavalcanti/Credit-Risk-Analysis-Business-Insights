"""PySpark-based data validation and business analytics."""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


REQUIRED_COLUMNS = {
    "Age",
    "Sex",
    "Job",
    "Housing",
    "Saving accounts",
    "Checking account",
    "Credit amount",
    "Duration",
    "Purpose",
    "Risk",
}


def validate_dataset(df: DataFrame) -> None:
    """Validate required columns and target integrity."""
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Dataset is missing required columns: {missing_list}")

    invalid_target_count = df.filter(~F.col("Risk").isin("good", "bad")).count()
    if invalid_target_count > 0:
        raise ValueError("Risk column must contain only 'good' or 'bad'.")


def missing_values_summary(df: DataFrame) -> DataFrame:
    """Return count of null values by column."""
    exprs = [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df.columns]
    raw = df.agg(*exprs)

    stack_expr = ", ".join([f"'{c}', `{c}`" for c in df.columns])
    return raw.selectExpr(f"stack({len(df.columns)}, {stack_expr}) as (column_name, null_count)")


def portfolio_kpis(df: DataFrame) -> DataFrame:
    """Compute portfolio-level indicators from German credit data."""
    base = (
        df.withColumn("is_bad", F.when(F.col("Risk") == "bad", F.lit(1)).otherwise(F.lit(0)))
        .withColumn("is_good", F.when(F.col("Risk") == "good", F.lit(1)).otherwise(F.lit(0)))
    )

    return base.agg(
        F.count("*").alias("applications"),
        F.avg("is_bad").alias("bad_rate"),
        F.avg("Credit amount").alias("avg_credit_amount"),
        F.avg("Duration").alias("avg_duration_months"),
    )


def segment_risk_by_purpose(df: DataFrame) -> DataFrame:
    """Risk view by purpose and housing profile."""
    return (
        df.withColumn("is_bad", F.when(F.col("Risk") == "bad", F.lit(1)).otherwise(F.lit(0)))
        .groupBy("Purpose", "Housing")
        .agg(
            F.count("*").alias("applications"),
            F.avg("is_bad").alias("bad_rate"),
            F.avg("Credit amount").alias("avg_credit_amount"),
            F.avg("Duration").alias("avg_duration"),
        )
        .orderBy(F.desc("bad_rate"), F.desc("applications"))
    )
