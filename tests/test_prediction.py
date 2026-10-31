"""
Unit tests for prediction, risk classification, and recommendation module.
"""

import pytest
import numpy as np
from sklearn.linear_model import LinearRegression
from utils.prediction import (
    validate_student_inputs,
    predict_score,
    classify_risk,
    calculate_contributions,
    generate_recommendations
)


@pytest.fixture
def dummy_model():
    """Returns a simple fitted linear model with known coefficients."""
    # Equation: Score = 5 + 0.3*Attendance + 0.3*Assignment + 0.3*PrevExam + 0.5*Hours
    X = np.array([
        [50, 50, 50, 10],
        [80, 80, 80, 20],
        [30, 30, 30, 5],
        [90, 95, 85, 25]
    ])
    y = np.array([55.0, 82.0, 34.5, 95.0])
    model = LinearRegression().fit(X, y)
    return model


def test_validate_student_inputs():
    # Valid inputs
    errors = validate_student_inputs(85.0, 75.0, 68.0, 12.0)
    assert len(errors) == 0

    # Invalid inputs (out of bounds)
    errors = validate_student_inputs(110.0, -5.0, 70.0, 95.0)
    assert len(errors) == 3
    assert any("Attendance" in e for e in errors)
    assert any("Assignment" in e for e in errors)
    assert any("Study Hours" in e for e in errors)


def test_predict_score_clipping(dummy_model):
    features = {
        "Attendance_Percentage": 99.0,
        "Assignment_Average": 99.0,
        "Previous_Exam_Score": 99.0,
        "Study_Hours_Per_Week": 40.0
    }
    score = predict_score(dummy_model, features)
    assert isinstance(score, float)
    assert 0.0 <= score <= 100.0

    # Test extreme negative inputs to verify clipping at 0.0
    extreme_low = {
        "Attendance_Percentage": 0.0,
        "Assignment_Average": 0.0,
        "Previous_Exam_Score": 0.0,
        "Study_Hours_Per_Week": 0.0
    }
    low_score = predict_score(dummy_model, extreme_low)
    assert low_score >= 0.0


def test_classify_risk_thresholds():
    # High Risk (< 40)
    high_risk = classify_risk(35.5, low_threshold=60.0, med_threshold=40.0)
    assert high_risk["category"] == "High Risk"
    assert "High Academic Risk" in high_risk["label"]

    # Medium Risk (40 <= score < 60)
    med_risk = classify_risk(52.0, low_threshold=60.0, med_threshold=40.0)
    assert med_risk["category"] == "Medium Risk"
    assert "Medium Academic Risk" in med_risk["label"]

    # Low Risk (>= 60)
    low_risk = classify_risk(78.0, low_threshold=60.0, med_threshold=40.0)
    assert low_risk["category"] == "Low Risk"
    assert "Low Academic Risk" in low_risk["label"]


def test_calculate_contributions(dummy_model):
    feature_names = [
        "Attendance_Percentage",
        "Assignment_Average",
        "Previous_Exam_Score",
        "Study_Hours_Per_Week"
    ]
    inputs = {
        "Attendance_Percentage": 80.0,
        "Assignment_Average": 70.0,
        "Previous_Exam_Score": 65.0,
        "Study_Hours_Per_Week": 15.0
    }
    contrib_df = calculate_contributions(dummy_model, feature_names, inputs)
    assert len(contrib_df) == 5  # Intercept + 4 features
    assert "Base Intercept (β₀)" in contrib_df["Feature"].values
    assert "Attendance Percentage" in contrib_df["Feature"].values


def test_generate_recommendations_actionable():
    # Student with low attendance and low assignment
    recs = generate_recommendations(
        attendance=55.0,
        assignment=45.0,
        prev_exam=70.0,
        study_hours=6.0,
        predicted_score=38.0,
        risk_category="High Risk"
    )
    assert len(recs) >= 3
    domains = [r["domain"] for r in recs]
    assert "Attendance & Engagement" in domains
    assert "Continuous Assessment" in domains
    assert "Study Habits & Time Management" in domains
    assert "Comprehensive Academic Care" in domains
