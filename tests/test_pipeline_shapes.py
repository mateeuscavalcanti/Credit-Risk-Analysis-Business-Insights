from pathlib import Path

import pytest

pyspark = pytest.importorskip("pyspark")

from src.analysis import missing_values_summary, portfolio_kpis, segment_risk_by_purpose, validate_dataset
from src.data_simulation import create_spark_session
from src.modeling import train_and_evaluate


@pytest.fixture(scope="module")
def spark():
    session = create_spark_session("test-credit-risk")
    yield session
    session.stop()


@pytest.fixture()
def sample_df(spark):
    data = [
        (35, "male", 2, "own", "little", "moderate", 5000, 24, "car", "good"),
        (28, "female", 1, "rent", "little", "little", 2600, 18, "education", "bad"),
        (44, "male", 3, "free", "rich", "moderate", 9000, 36, "business", "good"),
    ]
    cols = ["Age", "Sex", "Job", "Housing", "Saving accounts", "Checking account", "Credit amount", "Duration", "Purpose", "Risk"]
    return spark.createDataFrame(data, cols)


def test_validate_dataset(sample_df):
    validate_dataset(sample_df)


def test_missing_values_summary(sample_df):
    out = missing_values_summary(sample_df)
    assert out.filter(out.column_name == "Risk").count() == 1


def test_portfolio_and_segment(sample_df):
    k = portfolio_kpis(sample_df).first().asDict()
    assert k["applications"] == 3

    s = segment_risk_by_purpose(sample_df)
    assert s.count() >= 1


def test_train_and_evaluate(sample_df, tmp_path: Path):
    perf = train_and_evaluate(sample_df, tmp_path)
    assert perf.count() == 3
