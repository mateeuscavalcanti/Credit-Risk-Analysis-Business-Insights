# Credit Risk Analysis & Business Insights

This repository contains an end-to-end credit risk workflow for a fintech use case, built around a **real Kaggle dataset** and implemented primarily with **PySpark**.

## Dataset
This project uses the Kaggle dataset commonly published as **German Credit Data**.

- Source: Kaggle (search: `German Credit Data`)
- Expected local file: `data/german_credit_data.csv`
- Target variable: `Risk` (`good` / `bad`)

> The dataset file is not committed to this repository. Download it from Kaggle and place it in the `data/` folder.

## Project Structure

```text
.
├── main.py
├── requirements.txt
├── src
│   ├── analysis.py
│   ├── business_insights.py
│   ├── data_simulation.py
│   └── modeling.py
├── tests
│   └── test_pipeline_shapes.py
└── outputs/                  # generated artifacts
```

## Workflow

1. **Load real data with Spark**
   - Creates a local Spark session.
   - Loads Kaggle CSV with inferred schema.

2. **Validate and understand data**
   - Checks required columns and target integrity.
   - Generates null-value summary by column.
   - Produces portfolio KPI table (application count, bad rate, avg amount, avg duration).

3. **Segment portfolio risk**
   - Creates risk view by `Purpose` and `Housing`.
   - Exports segment table sorted by bad rate.

4. **Train benchmark models (PySpark ML)**
   - Logistic Regression
   - Random Forest
   - Gradient Boosted Trees
   - Exports ROC-AUC comparison.

5. **Generate business report and visual**
   - Markdown report with key metrics, high-risk segments, and actions.
   - Bar chart for bad-rate distribution by purpose/housing.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Outputs

`outputs/` includes:
- `null_summary/` (Spark CSV folder)
- `portfolio_kpis/` (Spark CSV folder)
- `risk_segments/` (Spark CSV folder)
- `model_performance/` (Spark CSV folder)
- `business_insights_report.md`
- `risk_by_purpose_housing.png`

## Notes
- PySpark is the primary processing engine.
- Pandas is only used when converting Spark data to plotting input.
