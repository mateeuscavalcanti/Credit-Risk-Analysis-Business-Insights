"""Generate a concise strategy report from Spark outputs."""

from __future__ import annotations

from pathlib import Path

from pyspark.sql import DataFrame


REPORT_TEMPLATE = """# Credit Risk Analysis - Business Insights

## Executive Metrics
- Applications analyzed: **{applications}**
- Bad rate: **{bad_rate:.2f}%**
- Average credit amount: **${avg_credit_amount:,.2f}**
- Average duration: **{avg_duration:.2f} months**

## Highest-Risk Segments
{segments}

## Modeling Snapshot
Best benchmark model: **{best_model}** (ROC-AUC: **{best_auc:.4f}**).

## Recommendations
1. Apply tighter limits for high-risk purpose/housing combinations.
2. Prioritize manual review for segments with sustained bad-rate concentration.
3. Use model scores as a ranking layer, combined with policy rules for final approval decisions.
4. Track monthly drift on bad-rate by purpose and account profile.
"""


def write_business_report(output_path: Path, kpis: DataFrame, segments: DataFrame, model_perf: DataFrame) -> None:
    """Write business report markdown file."""
    k = kpis.first().asDict()
    top_segments = segments.limit(3).collect()
    best_model = model_perf.first().asDict()

    segment_lines = []
    for row in top_segments:
        segment_lines.append(
            f"- **{row['Purpose']} | {row['Housing']}**: bad rate {row['bad_rate'] * 100:.2f}%, "
            f"applications {row['applications']}, avg credit ${row['avg_credit_amount']:,.2f}"
        )

    content = REPORT_TEMPLATE.format(
        applications=int(k["applications"]),
        bad_rate=float(k["bad_rate"] * 100),
        avg_credit_amount=float(k["avg_credit_amount"]),
        avg_duration=float(k["avg_duration_months"]),
        segments="\n".join(segment_lines),
        best_model=best_model["model"],
        best_auc=float(best_model["roc_auc"]),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
