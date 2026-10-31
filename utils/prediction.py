"""
Prediction and Risk Assessment Module for Student Academic Risk Early-Warning System.
Handles model inference, academic risk tier classification, feature contribution analysis,
and supportive, non-punitive intervention recommendations.
"""

from typing import Dict, List, Any, Union, Optional
import numpy as np
import pandas as pd


def validate_student_inputs(
    attendance: float,
    assignment: float,
    prev_exam: float,
    study_hours: float
) -> List[str]:
    """
    Validates input ranges for student features.
    
    Returns:
        List of validation error messages (empty if all valid).
    """
    errors = []
    if attendance is None or not (0.0 <= attendance <= 100.0):
        errors.append("Attendance Percentage must be between 0.0% and 100.0%.")
    if assignment is None or not (0.0 <= assignment <= 100.0):
        errors.append("Assignment Average must be between 0.0 and 100.0.")
    if prev_exam is None or not (0.0 <= prev_exam <= 100.0):
        errors.append("Previous Exam Score must be between 0.0 and 100.0.")
    if study_hours is None or not (0.0 <= study_hours <= 80.0):
        errors.append("Study Hours Per Week must be between 0.0 and 80.0 hours.")
    return errors


def predict_score(
    model: Any,
    feature_values: Union[Dict[str, float], pd.DataFrame, List[float]],
    feature_names: Optional[List[str]] = None
) -> float:
    """
    Predicts the expected final exam score using the trained Multiple Linear Regression model.
    The raw prediction is clipped to the realistic score range [0.0, 100.0].
    
    Args:
        model: Trained scikit-learn LinearRegression model.
        feature_values: Input dictionary, DataFrame, or list of values.
        feature_names: Order of feature names expected by the model.
        
    Returns:
        float: Bounded predicted final exam score rounded to 1 decimal place.
    """
    if feature_names is None:
        feature_names = [
            "Attendance_Percentage",
            "Assignment_Average",
            "Previous_Exam_Score",
            "Study_Hours_Per_Week"
        ]

    if isinstance(feature_values, dict):
        input_df = pd.DataFrame([{col: feature_values[col] for col in feature_names}])
    elif isinstance(feature_values, list):
        input_df = pd.DataFrame([feature_values], columns=feature_names)
    elif isinstance(feature_values, pd.DataFrame):
        input_df = feature_values[feature_names]
    else:
        raise ValueError("Unsupported input format for feature_values.")

    if hasattr(model, "feature_names_in_"):
        raw_prediction = float(model.predict(input_df)[0])
    else:
        raw_prediction = float(model.predict(input_df.values)[0])
    clipped_prediction = max(0.0, min(100.0, raw_prediction))
    return round(clipped_prediction, 1)


def classify_risk(
    score: float,
    low_threshold: float = 60.0,
    med_threshold: float = 40.0
) -> Dict[str, Any]:
    """
    Maps a predicted final exam score to an academic risk category.
    
    Default Illustrative Thresholds:
    - Low Risk: Score >= 60.0
    - Medium Risk: 40.0 <= Score < 60.0
    - High Risk: Score < 40.0
    
    Args:
        score: Predicted final exam score.
        low_threshold: Cut-off for Low Risk (default: 60.0).
        med_threshold: Cut-off between High and Medium Risk (default: 40.0).
        
    Returns:
        Dict containing category name, hex color, icon, and description.
    """
    if score >= low_threshold:
        return {
            "category": "Low Risk",
            "label": "Low Academic Risk",
            "color": "#10b981",          # Emerald Green
            "bg_color": "rgba(16, 185, 129, 0.15)",
            "border_color": "#059669",
            "icon": "🟢",
            "status_badge": "LOW RISK",
            "description": "Student demonstrates steady mastery of learning objectives. Expected to perform well on the final exam."
        }
    elif score >= med_threshold:
        return {
            "category": "Medium Risk",
            "label": "Medium Academic Risk",
            "color": "#f59e0b",          # Amber
            "bg_color": "rgba(245, 158, 11, 0.15)",
            "border_color": "#d97706",
            "icon": "🟡",
            "status_badge": "MEDIUM RISK",
            "description": "Student is on the borderline. Targeted support and structured revision can prevent academic decline."
        }
    else:
        return {
            "category": "High Risk",
            "label": "High Academic Risk",
            "color": "#ef4444",          # Rose / Red
            "bg_color": "rgba(239, 68, 68, 0.15)",
            "border_color": "#dc2626",
            "icon": "🔴",
            "status_badge": "HIGH RISK",
            "description": "Student is vulnerable to failing the course. Immediate supportive academic intervention is strongly advised."
        }


def calculate_contributions(
    model: Any,
    feature_names: List[str],
    input_values: Dict[str, float]
) -> pd.DataFrame:
    """
    Calculates the exact linear contribution of each feature to the predicted score:
    Contribution_i = beta_i * x_i.
    
    Args:
        model: Trained LinearRegression model.
        feature_names: List of feature names.
        input_values: Dictionary of feature name -> input value.
        
    Returns:
        pd.DataFrame: Table with feature, input value, coefficient, and absolute contribution.
    """
    intercept = float(model.intercept_)
    coefficients = model.coef_

    contributions = []
    # Base intercept
    contributions.append({
        "Feature": "Base Intercept (β₀)",
        "Input_Value": 1.0,
        "Coefficient": round(intercept, 3),
        "Contribution": round(intercept, 2),
        "Type": "Baseline"
    })

    for name, coef in zip(feature_names, coefficients):
        val = float(input_values[name])
        contrib = float(coef * val)
        contributions.append({
            "Feature": name.replace("_", " "),
            "Input_Value": val,
            "Coefficient": round(float(coef), 3),
            "Contribution": round(contrib, 2),
            "Type": "Feature"
        })

    return pd.DataFrame(contributions)


def generate_recommendations(
    attendance: float,
    assignment: float,
    prev_exam: float,
    study_hours: float,
    predicted_score: float,
    risk_category: str
) -> List[Dict[str, str]]:
    """
    Generates tailored, supportive, non-punitive academic intervention recommendations.
    
    Args:
        attendance: Student's attendance percentage.
        assignment: Student's assignment score average.
        prev_exam: Student's previous examination score.
        study_hours: Student's weekly study hours.
        predicted_score: Model predicted score.
        risk_category: 'High Risk', 'Medium Risk', or 'Low Risk'.
        
    Returns:
        List of dicts containing recommendation domain, urgency, and specific actionable guidance.
    """
    recs = []

    # 1. Attendance Domain
    if attendance < 65.0:
        recs.append({
            "domain": "Attendance & Engagement",
            "tag": "Critical Attendance Deficit",
            "priority": "Urgent",
            "action": (
                "Attendance is critically low at {:.1f}%. Initiate a supportive check-in to understand "
                "underlying hurdles (health, transport, or course schedule conflicts). Pair with a student "
                "buddy to share lecture synopses and class recordings."
            ).format(attendance)
        })
    elif attendance < 75.0:
        recs.append({
            "domain": "Attendance & Engagement",
            "tag": "Attendance Warning",
            "priority": "Moderate",
            "action": (
                "Attendance is at {:.1f}%, below the institutional 75% target. Send a friendly reminder "
                "highlighting the strong correlation between lecture participation and exam success."
            ).format(attendance)
        })

    # 2. Assignment Domain
    if assignment < 50.0:
        recs.append({
            "domain": "Continuous Assessment",
            "tag": "Assignment Skill Gap",
            "priority": "Urgent",
            "action": (
                "Assignment average is {:.1f}/100. Recommend visiting faculty or TA office hours for one-on-one "
                "rubric deconstructions. Provide opportunities for low-stakes practice quizzes and peer review."
            ).format(assignment)
        })
    elif assignment < 65.0:
        recs.append({
            "domain": "Continuous Assessment",
            "tag": "Assignment Reinforcement",
            "priority": "Moderate",
            "action": (
                "Assignment average is {:.1f}/100. Encourage submitting draft assignments early for formative "
                "feedback and utilizing learning center writing/coding labs."
            ).format(assignment)
        })

    # 3. Previous Examination Domain
    if prev_exam < 45.0:
        recs.append({
            "domain": "Foundational Mastery",
            "tag": "Prerequisite Concept Gap",
            "priority": "Urgent",
            "action": (
                "Previous exam score was {:.1f}/100. Foundational gaps exist in core syllabus units. "
                "Provide targeted remedial problem sets and recommend attending weekly review workshops."
            ).format(prev_exam)
        })
    elif prev_exam < 60.0:
        recs.append({
            "domain": "Foundational Mastery",
            "tag": "Exam Technique Coaching",
            "priority": "Moderate",
            "action": (
                "Previous exam score was {:.1f}/100. Offer past examination paper walkthroughs to refine "
                "time-management, structured problem-solving, and answer formatting."
            ).format(prev_exam)
        })

    # 4. Study Habits Domain
    if study_hours < 8.0:
        recs.append({
            "domain": "Study Habits & Time Management",
            "tag": "Low Weekly Study Allocation",
            "priority": "Moderate",
            "action": (
                "Weekly study time is {:.1f} hours/week. Support the student in creating a weekly calendar "
                "allocating at least 12-15 hours for self-directed study and coursework using the Pomodoro technique."
            ).format(study_hours)
        })

    # 5. Overarching Tier Guidance
    if risk_category == "High Risk":
        recs.append({
            "domain": "Comprehensive Academic Care",
            "tag": "Individualized Academic Success Plan (IASP)",
            "priority": "Urgent",
            "action": (
                "Convene a collaborative advising conference with the student and academic mentor. "
                "Establish a supportive Individualized Academic Success Plan (IASP) with bi-weekly check-ins. "
                "Ensure this communication is supportive, empowering, and focused on growth."
            )
        })
    elif risk_category == "Medium Risk":
        recs.append({
            "domain": "Academic Guidance",
            "tag": "Mid-Term Progress Check",
            "priority": "Standard",
            "action": (
                "Schedule a 15-minute mid-term check-in to monitor progress in upcoming module deliverables "
                "and ensure study questions are clarified before the final exam period."
            )
        })
    else:
        recs.append({
            "domain": "Enrichment & Leadership",
            "tag": "Advanced Learning & Peer Mentorship",
            "priority": "Low",
            "action": (
                "Student is on a strong trajectory (predicted score: {:.1f}/100). Invite the student to participate "
                "in research enrichment projects, honors modules, or serve as a peer learning facilitator."
            ).format(predicted_score)
        })

    return recs
