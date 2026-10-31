"""
Data Preprocessing Module for Student Academic Risk Early-Warning System.
Handles data loading, exploratory statistics, duplicate detection, invalid value cleansing,
and leakage-free train/test splitting with imputation.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_FEATURE_COLS = [
    "Attendance_Percentage",
    "Assignment_Average",
    "Previous_Exam_Score",
    "Study_Hours_Per_Week"
]

TARGET_COL = "Final_Exam_Score"


def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads student academic dataset from a CSV file.
    
    Args:
        filepath: Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    df = pd.read_csv(filepath)
    return df


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculates summary statistics, missing counts, duplicates, and column types.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        Dict containing dataset metrics and statistics.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    return {
        "shape": df.shape,
        "num_rows": len(df),
        "num_cols": df.shape[1],
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_counts": df.isna().sum().to_dict(),
        "total_missing_cells": int(df.isna().sum().sum()),
        "rows_with_missing": int(df.isna().any(axis=1).sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "describe": df.describe().to_dict() if len(numeric_cols) > 0 else {}
    }


def clean_raw_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Performs data cleaning:
    1. Removes duplicate records based on all columns or Student_ID.
    2. Identifies and filters invalid out-of-bounds values:
       - Attendance_Percentage: must be between 0 and 100
       - Assignment_Average: must be between 0 and 100
       - Previous_Exam_Score: must be between 0 and 100
       - Study_Hours_Per_Week: must be >= 0 and <= 80
       - Final_Exam_Score: must be between 0 and 100
       
    Args:
        df: Raw DataFrame.
        
    Returns:
        Tuple[pd.DataFrame, Dict]: Cleaned DataFrame and audit report dictionary.
    """
    initial_count = len(df)
    cleaning_report = {
        "initial_rows": initial_count,
        "duplicates_removed": 0,
        "invalid_rows_removed": 0,
        "missing_rows_remaining": 0,
        "final_clean_rows": 0,
        "invalid_details": []
    }

    # 1. Remove duplicate records
    # Check duplicates based on either all columns or Student_ID
    dup_mask = df.duplicated(keep="first") | df.duplicated(subset=["Student_ID"], keep="first")
    num_duplicates = int(dup_mask.sum())
    df_dedup = df[~dup_mask].copy()
    cleaning_report["duplicates_removed"] = num_duplicates

    # 2. Check for invalid values
    invalid_mask = pd.Series(False, index=df_dedup.index)

    # Attendance bounds [0, 100]
    att_invalid = (df_dedup["Attendance_Percentage"] < 0) | (df_dedup["Attendance_Percentage"] > 100)
    if att_invalid.any():
        cleaning_report["invalid_details"].append(
            f"Attendance out of [0, 100]%: {int(att_invalid.sum())} records"
        )
        invalid_mask = invalid_mask | att_invalid

    # Study hours bounds [0, 80]
    hours_invalid = (df_dedup["Study_Hours_Per_Week"] < 0) | (df_dedup["Study_Hours_Per_Week"] > 80)
    if hours_invalid.any():
        cleaning_report["invalid_details"].append(
            f"Study hours out of [0, 80] hrs/wk: {int(hours_invalid.sum())} records"
        )
        invalid_mask = invalid_mask | hours_invalid

    # Assignment bounds [0, 100]
    assign_invalid = (df_dedup["Assignment_Average"] < 0) | (df_dedup["Assignment_Average"] > 100)
    if assign_invalid.any():
        cleaning_report["invalid_details"].append(
            f"Assignment score out of [0, 100]: {int(assign_invalid.sum())} records"
        )
        invalid_mask = invalid_mask | assign_invalid

    # Previous exam score bounds [0, 100]
    prev_invalid = (df_dedup["Previous_Exam_Score"] < 0) | (df_dedup["Previous_Exam_Score"] > 100)
    if prev_invalid.any():
        cleaning_report["invalid_details"].append(
            f"Previous exam score out of [0, 100]: {int(prev_invalid.sum())} records"
        )
        invalid_mask = invalid_mask | prev_invalid

    # Final exam score bounds [0, 100] if present
    if "Final_Exam_Score" in df_dedup.columns:
        final_invalid = (df_dedup["Final_Exam_Score"] < 0) | (df_dedup["Final_Exam_Score"] > 100)
        if final_invalid.any():
            cleaning_report["invalid_details"].append(
                f"Final exam score out of [0, 100]: {int(final_invalid.sum())} records"
            )
            invalid_mask = invalid_mask | final_invalid

    # Filter out invalid rows (excluding rows that have NaN, since NaN is handled separately)
    # Note: comparisons with NaN yield False in Pandas, so only non-NaN invalid values are caught here
    num_invalid = int(invalid_mask.sum())
    df_valid = df_dedup[~invalid_mask].copy()
    cleaning_report["invalid_rows_removed"] = num_invalid

    # Record remaining missing counts before train/test split
    cleaning_report["missing_rows_remaining"] = int(df_valid.isna().any(axis=1).sum())
    cleaning_report["final_clean_rows"] = len(df_valid)

    return df_valid.reset_index(drop=True), cleaning_report


def prepare_features_and_target(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    target_col: str = TARGET_COL
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separates input features and target variable.
    
    Args:
        df: Input DataFrame.
        feature_cols: List of feature column names. Defaults to DEFAULT_FEATURE_COLS.
        target_col: Name of target column. Defaults to 'Final_Exam_Score'.
        
    Returns:
        Tuple[pd.DataFrame, pd.Series]: Feature matrix X and target vector y.
    """
    if feature_cols is None:
        feature_cols = DEFAULT_FEATURE_COLS

    X = df[feature_cols].copy()
    y = df[target_col].copy()
    return X, y


def split_and_impute(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, float]]:
    """
    Splits data into train and test sets, then imputes missing values using
    MEDIANS COMPUTED EXCLUSIVELY FROM THE TRAINING SET to strictly prevent data leakage.
    
    Args:
        X: Feature DataFrame.
        y: Target Series.
        test_size: Fraction of dataset for testing (default: 0.2).
        random_state: Random state for reproducibility.
        
    Returns:
        Tuple: (X_train, X_test, y_train, y_test, train_medians)
    """
    # If target y has missing values, drop those rows before splitting
    valid_target_mask = ~y.isna()
    X = X[valid_target_mask]
    y = y[valid_target_mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Compute training medians to prevent data leakage
    train_medians = X_train.median().to_dict()

    # Impute training set with train medians
    X_train_imputed = X_train.fillna(train_medians)

    # Impute test set with the SAME train medians (NO data leakage)
    X_test_imputed = X_test.fillna(train_medians)

    return X_train_imputed, X_test_imputed, y_train, y_test, train_medians
