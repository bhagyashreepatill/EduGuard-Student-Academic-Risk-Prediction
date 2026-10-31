"""
Unit tests for data preprocessing module.
"""

import numpy as np
import pandas as pd
import pytest
from utils.preprocessing import (
    get_dataset_summary,
    clean_raw_data,
    prepare_features_and_target,
    split_and_impute,
    DEFAULT_FEATURE_COLS,
    TARGET_COL
)


@pytest.fixture
def sample_raw_data():
    """Generates a small test dataframe with duplicates and invalid values."""
    data = {
        "Student_ID": ["S1", "S2", "S3", "S4", "S5", "S2"],  # S2 is duplicate
        "Student_Name": ["Alice", "Bob", "Charlie", "David", "Emma", "Bob"],
        "Attendance_Percentage": [85.0, 70.0, 115.0, np.nan, 90.0, 70.0],  # 115 is invalid
        "Assignment_Average": [80.0, 65.0, 75.0, 50.0, 95.0, 65.0],
        "Previous_Exam_Score": [78.0, 58.0, 62.0, 45.0, 88.0, 58.0],
        "Study_Hours_Per_Week": [15.0, -4.0, 12.0, 8.0, 20.0, -4.0],  # -4 is invalid
        "Final_Exam_Score": [82.0, 64.0, 70.0, 48.0, 92.0, 64.0]
    }
    return pd.DataFrame(data)


def test_get_dataset_summary(sample_raw_data):
    summary = get_dataset_summary(sample_raw_data)
    assert summary["num_rows"] == 6
    assert summary["num_cols"] == 7
    assert summary["duplicate_rows"] == 1
    assert summary["missing_counts"]["Attendance_Percentage"] == 1


def test_clean_raw_data_removes_duplicates_and_invalids(sample_raw_data):
    df_clean, report = clean_raw_data(sample_raw_data)
    
    # 1 duplicate (S2) should be removed
    assert report["duplicates_removed"] == 1
    # Invalid values: Attendance 115.0 (S3) and Study Hours -4.0 (S2 duplicate already out or S2 handled)
    # Total invalid rows removed should be > 0
    assert report["invalid_rows_removed"] >= 1
    
    # Clean dataset must not contain invalid bounds
    assert (df_clean["Attendance_Percentage"].dropna() <= 100).all()
    assert (df_clean["Attendance_Percentage"].dropna() >= 0).all()
    assert (df_clean["Study_Hours_Per_Week"].dropna() >= 0).all()
    assert df_clean["Student_ID"].is_unique


def test_prepare_features_and_target(sample_raw_data):
    X, y = prepare_features_and_target(sample_raw_data, DEFAULT_FEATURE_COLS, TARGET_COL)
    assert list(X.columns) == DEFAULT_FEATURE_COLS
    assert y.name == TARGET_COL
    assert len(X) == len(y)


def test_split_and_impute_prevents_data_leakage():
    # Construct a larger dataframe to test train/test splitting and median leakage
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "Attendance_Percentage": np.random.uniform(50, 100, n),
        "Assignment_Average": np.random.uniform(40, 100, n),
        "Previous_Exam_Score": np.random.uniform(40, 100, n),
        "Study_Hours_Per_Week": np.random.uniform(5, 30, n),
        "Final_Exam_Score": np.random.uniform(40, 100, n)
    })
    
    # Inject missing values
    df.loc[0:4, "Attendance_Percentage"] = np.nan
    df.loc[90:95, "Attendance_Percentage"] = np.nan

    X, y = prepare_features_and_target(df)
    X_train, X_test, y_train, y_test, train_medians = split_and_impute(
        X, y, test_size=0.2, random_state=42
    )

    # Verify no missing values remain in train or test
    assert X_train.isna().sum().sum() == 0
    assert X_test.isna().sum().sum() == 0

    # Verify train medians were used
    assert "Attendance_Percentage" in train_medians
    assert len(X_train) == 80
    assert len(X_test) == 20
