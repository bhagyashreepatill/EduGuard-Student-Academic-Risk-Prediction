"""
Model Training and Evaluation Pipeline for Student Academic Risk Early-Warning System.
Trains a Multiple Linear Regression model with Scikit-learn, computes MAE, MSE, RMSE, R2,
and exports the trained model bundle using Joblib.
"""

import os
from datetime import datetime
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils.preprocessing import (
    load_data,
    clean_raw_data,
    prepare_features_and_target,
    split_and_impute,
    DEFAULT_FEATURE_COLS,
    TARGET_COL
)


def train_and_save_model(
    data_path: str = "student_academic_data.csv",
    model_output_path: str = "models/student_performance_model.pkl",
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Executes the end-to-end training workflow:
    1. Loads dataset
    2. Cleans duplicates and removes invalid values
    3. Splits into train and test sets
    4. Imputes missing values strictly using training set medians (no data leakage)
    5. Fits Multiple Linear Regression
    6. Evaluates test set metrics (MAE, MSE, RMSE, R2)
    7. Saves model bundle to disk via Joblib
    
    Returns:
        Dict: Complete training report including metrics, equation, and data stats.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    # 1. Load data
    df_raw = load_data(data_path)
    initial_rows = len(df_raw)

    # 2. Clean raw data (deduplicate and filter invalid values)
    df_clean, clean_report = clean_raw_data(df_raw)

    # 3. Separate features and target
    X, y = prepare_features_and_target(df_clean, DEFAULT_FEATURE_COLS, TARGET_COL)

    # 4. Leakage-free train/test split with training-derived median imputation
    X_train, X_test, y_train, y_test, train_medians = split_and_impute(
        X, y, test_size=test_size, random_state=random_state
    )

    # 5. Fit Multiple Linear Regression
    model = LinearRegression(fit_intercept=True)
    model.fit(X_train, y_train)

    # 6. Evaluate on Test Set
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    # Ensure predictions for evaluation are bounded [0, 100]
    y_pred_test_clipped = np.clip(y_pred_test, 0.0, 100.0)

    mae = float(mean_absolute_error(y_test, y_pred_test_clipped))
    mse = float(mean_squared_error(y_test, y_pred_test_clipped))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test, y_pred_test_clipped))

    # Also compute training metrics for comparison
    r2_train = float(r2_score(y_train, np.clip(y_pred_train, 0.0, 100.0)))

    # 7. Extract coefficients and format equation
    intercept = float(model.intercept_)
    coefficients = {col: float(coef) for col, coef in zip(DEFAULT_FEATURE_COLS, model.coef_)}

    equation_terms = [f"{intercept:+.3f}"]
    for col, coef in zip(DEFAULT_FEATURE_COLS, model.coef_):
        equation_terms.append(f"{coef:+.3f} * ({col})")
    equation_str = "Final_Exam_Score = " + " ".join(equation_terms)

    # 8. Create and save model bundle
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    
    bundle = {
        "model": model,
        "feature_names": DEFAULT_FEATURE_COLS,
        "target_name": TARGET_COL,
        "train_medians": train_medians,
        "intercept": intercept,
        "coefficients": coefficients,
        "metrics": {
            "mae": round(mae, 3),
            "mse": round(mse, 3),
            "rmse": round(rmse, 3),
            "r2": round(r2, 4),
            "r2_train": round(r2_train, 4)
        },
        "dataset_stats": {
            "initial_rows": initial_rows,
            "duplicates_removed": clean_report["duplicates_removed"],
            "invalid_rows_removed": clean_report["invalid_rows_removed"],
            "clean_rows": len(df_clean),
            "train_rows": len(X_train),
            "test_rows": len(X_test)
        },
        "equation_str": equation_str,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    joblib.dump(bundle, model_output_path)

    print("=" * 70)
    print("STUDENT ACADEMIC RISK EARLY-WARNING SYSTEM: MODEL TRAINING REPORT")
    print("=" * 70)
    print(f"Data File:               {data_path} (Initial records: {initial_rows})")
    print(f"Data Cleansing:          Duplicates removed: {clean_report['duplicates_removed']} | Invalids removed: {clean_report['invalid_rows_removed']}")
    print(f"Clean Records:           {len(df_clean)} (Train: {len(X_train)}, Test: {len(X_test)})")
    print(f"Target Variable:         {TARGET_COL}")
    print("-" * 70)
    print("Learned Regression Equation:")
    print(f"  {equation_str}")
    print("-" * 70)
    print("Evaluation Metrics (Test Set):")
    print(f"  - R2 Score (Variance Explained): {r2:.4f} (Train R2: {r2_train:.4f})")
    print(f"  - Mean Absolute Error (MAE):     {mae:.3f} points")
    print(f"  - Mean Squared Error (MSE):      {mse:.3f}")
    print(f"  - Root Mean Squared Error (RMSE):{rmse:.3f} points")
    print("-" * 70)
    print(f"Model successfully serialized to: {os.path.abspath(model_output_path)}")
    print("=" * 70)

    return bundle


if __name__ == "__main__":
    train_and_save_model()
