"""End-to-end credit risk workflow using Kaggle German Credit data and PySpark."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from src.analysis import (
    missing_values_summary,
    portfolio_kpis,
    segment_risk_by_purpose,
    validate_dataset,
)
from src.business_insights import write_business_report
from src.data_simulation import create_spark_session, load_german_credit_data
from src.modeling import train_and_evaluate


DATA_PATH = Path("data/german_credit_data.csv")
OUTPUT_DIR = Path("outputs")


def save_visuals(segments_spark_df, output_dir: Path) -> None:
    """Generate visuals using pandas only at plotting time."""
    pdf = segments_spark_df.limit(12).toPandas()
    if pdf.empty:
        return

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 5))
    sns.barplot(data=pdf, x="Purpose", y="bad_rate", hue="Housing")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Bad Rate")
    plt.title("Bad Rate by Purpose and Housing")
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "risk_by_purpose_housing.png", dpi=180)
    plt.close()


def main() -> None:
    spark = create_spark_session()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Dataset not found. Download 'German Credit Data' from Kaggle and place it at "
            "data/german_credit_data.csv"
        )

    df = load_german_credit_data(spark, DATA_PATH)
    validate_dataset(df)

    nulls = missing_values_summary(df)
    nulls.coalesce(1).write.mode("overwrite").option("header", True).csv(str(OUTPUT_DIR / "null_summary"))

    kpis = portfolio_kpis(df)
    kpis.coalesce(1).write.mode("overwrite").option("header", True).csv(str(OUTPUT_DIR / "portfolio_kpis"))

    segments = segment_risk_by_purpose(df)
    segments.coalesce(1).write.mode("overwrite").option("header", True).csv(str(OUTPUT_DIR / "risk_segments"))

    model_perf = train_and_evaluate(df, OUTPUT_DIR)

    write_business_report(OUTPUT_DIR / "business_insights_report.md", kpis, segments, model_perf)
    save_visuals(segments, OUTPUT_DIR)

    spark.stop()


if __name__ == "__main__":
    main()
