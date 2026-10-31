"""
Student Academic Risk Early-Warning System
A professional Machine Learning Web Application built with Streamlit, Multiple Linear Regression,
Pandas, Scikit-learn, and Plotly.
"""

import os
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from utils.preprocessing import (
    load_data,
    clean_raw_data,
    prepare_features_and_target,
    split_and_impute,
    DEFAULT_FEATURE_COLS,
    TARGET_COL
)
from utils.prediction import (
    validate_student_inputs,
    predict_score,
    classify_risk,
    calculate_contributions,
    generate_recommendations
)
from train_model import train_and_save_model

# ==============================================================================
# 1. STREAMLIT CONFIGURATION & THEMING
# ==============================================================================
st.set_page_config(
    page_title="EduGuard | Student Academic Risk Early-Warning System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Academic UI Styling (CSS)
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Top Banner / Hero Card */
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #312e81 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 28px 32px;
        color: #ffffff;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0;
        background: linear-gradient(90deg, #ffffff 0%, #c7d2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-top: 8px;
        max-width: 820px;
        line-height: 1.5;
    }

    /* Metric Cards */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 20px 22px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.95rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.1;
    }
    .metric-subtext {
        font-size: 0.82rem;
        color: #94a3b8;
        margin-top: 6px;
    }

    /* Risk Badges */
    .badge-high-risk {
        background-color: #fef2f2;
        color: #dc2626;
        border: 1px solid #fecaca;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.8rem;
        display: inline-block;
    }
    .badge-medium-risk {
        background-color: #fffbeb;
        color: #d97706;
        border: 1px solid #fde68a;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.8rem;
        display: inline-block;
    }
    .badge-low-risk {
        background-color: #ecfdf5;
        color: #059669;
        border: 1px solid #a7f3d0;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.8rem;
        display: inline-block;
    }

    /* Info Callout */
    .academic-callout {
        background-color: #f8fafc;
        border-left: 4px solid #4f46e5;
        border-radius: 0 10px 10px 0;
        padding: 16px 20px;
        margin: 16px 0;
        color: #334155;
    }

    /* Code & Formula Box */
    .formula-box {
        background: #0f172a;
        color: #e2e8f0;
        border-radius: 12px;
        padding: 18px 24px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.95rem;
        line-height: 1.6;
        border: 1px solid #334155;
    }

    /* Recommendation Item */
    .rec-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    }
    .rec-urgent {
        border-left: 5px solid #ef4444;
    }
    .rec-moderate {
        border-left: 5px solid #f59e0b;
    }
    .rec-growth {
        border-left: 5px solid #10b981;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==============================================================================
# 2. DATA & MODEL LOADING FUNCTIONS (CACHED)
# ==============================================================================
DATA_FILE = "student_academic_data.csv"
MODEL_FILE = "models/student_performance_model.pkl"


@st.cache_data
def load_and_prepare_dataset(csv_path: str = DATA_FILE):
    """Loads and cleans dataset with cached performance."""
    if not os.path.exists(csv_path):
        from data_generator import save_dataset
        save_dataset(csv_path)
    
    df_raw = load_data(csv_path)
    df_clean, clean_report = clean_raw_data(df_raw)
    return df_raw, df_clean, clean_report


@st.cache_resource
def load_trained_model(model_path: str = MODEL_FILE):
    """Loads Joblib model bundle with cached memory."""
    if not os.path.exists(model_path):
        train_and_save_model(data_path=DATA_FILE, model_output_path=model_path)
    return joblib.load(model_path)


# Ensure dataset and model exist
try:
    raw_df, clean_df, clean_report = load_and_prepare_dataset(DATA_FILE)
    model_bundle = load_trained_model(MODEL_FILE)
except Exception as e:
    st.error(f"Error loading system assets: {e}")
    st.stop()

model = model_bundle["model"]
metrics = model_bundle["metrics"]
equation_str = model_bundle["equation_str"]
feature_names = model_bundle["feature_names"]


# Compute predictions for clean dataset to enable enriched overview and search table
@st.cache_data
def compute_dataset_predictions(df: pd.DataFrame, low_t: float, med_t: float):
    """Appends predicted scores and risk categories to the clean dataset."""
    df_copy = df.copy()
    X = df_copy[feature_names].copy()
    
    # Handle any remaining NaNs using median imputation to prevent errors
    train_medians = model_bundle.get("train_medians", X.median().to_dict())
    X_imputed = X.fillna(train_medians)
    
    raw_preds = model.predict(X_imputed)
    clipped_preds = np.clip(raw_preds, 0.0, 100.0).round(1)
    df_copy["Predicted_Final_Score"] = clipped_preds
    
    # Calculate Residual if Final_Exam_Score exists
    if "Final_Exam_Score" in df_copy.columns:
        df_copy["Residual_Error"] = (df_copy["Final_Exam_Score"] - df_copy["Predicted_Final_Score"]).round(1)
        
    risk_labels = []
    risk_colors = []
    for s in clipped_preds:
        res = classify_risk(s, low_threshold=low_t, med_threshold=med_t)
        risk_labels.append(res["category"])
        risk_colors.append(res["color"])
        
    df_copy["Risk_Category"] = risk_labels
    df_copy["Risk_Color"] = risk_colors
    return df_copy


# ==============================================================================
# 3. SIDEBAR NAVIGATION & CONFIGURATION
# ==============================================================================
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 12px 0 20px 0;">
        <span style="font-size: 2.8rem;">🎓</span>
        <h2 style="margin: 6px 0 2px 0; font-weight: 800; color: #1e293b; font-size: 1.35rem;">EduGuard System</h2>
        <span style="background: #e0e7ff; color: #4338ca; padding: 3px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: 600;">ML Early-Warning v1.0</span>
    </div>
    """, unsafe_allow_html=True)

    menu_option = st.radio(
        "Navigation Menu",
        options=[
            "📊 Dashboard Overview",
            "🔮 Student Risk Prediction",
            "📋 Student Records",
            "📈 Data Visualizations",
            "⚙️ Model Performance",
            "💡 Early-Warning Recommendations",
            "🛡️ Explainability & Fairness"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("### ⚙️ Risk Thresholds")
    st.caption("Configurable illustrative thresholds for risk tiering.")

    low_risk_threshold = st.slider(
        "Low Risk Threshold (≥)",
        min_value=50.0,
        max_value=75.0,
        value=60.0,
        step=1.0,
        help="Scores at or above this threshold indicate low academic risk."
    )

    med_risk_threshold = st.slider(
        "Medium Risk Threshold (≥)",
        min_value=30.0,
        max_value=55.0,
        value=40.0,
        step=1.0,
        help="Scores between this threshold and Low Risk indicate medium academic risk."
    )

    if med_risk_threshold >= low_risk_threshold:
        st.warning("Medium threshold must be strictly lower than Low threshold.")
        med_risk_threshold = low_risk_threshold - 10.0

    st.markdown("---")
    st.markdown("### 🔄 Model Management")
    if st.button("Retrain Model with Data", use_container_width=True):
        with st.spinner("Retraining Multiple Linear Regression model..."):
            train_and_save_model(data_path=DATA_FILE, model_output_path=MODEL_FILE)
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Model successfully retrained and cached!")
            st.rerun()

    st.markdown("---")
    st.markdown(f"""
    <div style="font-size: 0.78rem; color: #64748b; line-height: 1.4;">
        <b>Model:</b> Multiple Linear Regression<br>
        <b>Validation R²:</b> {metrics['r2']:.4f}<br>
        <b>Clean Cohort:</b> {clean_report['final_clean_rows']} records<br>
        <b>Last Updated:</b> {model_bundle.get('trained_at', 'Active')}
    </div>
    """, unsafe_allow_html=True)


# Generate scored dataset with active configurable thresholds
scored_df = compute_dataset_predictions(clean_df, low_risk_threshold, med_risk_threshold)


# ==============================================================================
# SECTION A: DASHBOARD OVERVIEW
# ==============================================================================
if menu_option == "📊 Dashboard Overview":
    st.markdown("""
    <div class="hero-banner">
        <h1 class="hero-title">Student Academic Risk Early-Warning System</h1>
        <p class="hero-subtitle">
            An interpretable Machine Learning intelligence platform designed for educators to identify academic 
            vulnerabilities early, forecast final performance, and coordinate timely, supportive interventions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Risk counts
    total_students = len(scored_df)
    high_risk_count = (scored_df["Risk_Category"] == "High Risk").sum()
    med_risk_count = (scored_df["Risk_Category"] == "Medium Risk").sum()
    low_risk_count = (scored_df["Risk_Category"] == "Low Risk").sum()

    high_risk_pct = (high_risk_count / total_students) * 100
    med_risk_pct = (med_risk_count / total_students) * 100
    low_risk_pct = (low_risk_count / total_students) * 100

    avg_attendance = scored_df["Attendance_Percentage"].mean()
    avg_assignment = scored_df["Assignment_Average"].mean()
    avg_pred_score = scored_df["Predicted_Final_Score"].mean()

    # KPI Metric Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Cohort Size</div>
            <div class="metric-value">{total_students}</div>
            <div class="metric-subtext">Active Enrolled Students</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="border-top: 4px solid #ef4444;">
            <div class="metric-label">High Risk (Tier 1)</div>
            <div class="metric-value" style="color: #dc2626;">{high_risk_count}</div>
            <div class="metric-subtext"><span class="badge-high-risk">{high_risk_pct:.1f}% of cohort</span></div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="border-top: 4px solid #f59e0b;">
            <div class="metric-label">Medium Risk (Tier 2)</div>
            <div class="metric-value" style="color: #d97706;">{med_risk_count}</div>
            <div class="metric-subtext"><span class="badge-medium-risk">{med_risk_pct:.1f}% of cohort</span></div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card" style="border-top: 4px solid #10b981;">
            <div class="metric-label">Low Risk (Tier 3)</div>
            <div class="metric-value" style="color: #059669;">{low_risk_count}</div>
            <div class="metric-subtext"><span class="badge-low-risk">{low_risk_pct:.1f}% on track</span></div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Forecast Avg</div>
            <div class="metric-value" style="color: #4f46e5;">{avg_pred_score:.1f}</div>
            <div class="metric-subtext">Expected Final Score</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Secondary row: Visualizations
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("🎯 Academic Risk Category Distribution")
        risk_fig = go.Figure(data=[go.Pie(
            labels=["Low Risk", "Medium Risk", "High Risk"],
            values=[low_risk_count, med_risk_count, high_risk_count],
            hole=0.55,
            marker=dict(colors=["#10b981", "#f59e0b", "#ef4444"], line=dict(color="#ffffff", width=2)),
            textinfo="label+percent",
            hoverinfo="label+value+percent"
        )])
        risk_fig.update_layout(
            showlegend=True,
            margin=dict(t=20, b=20, l=20, r=20),
            height=340,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Plus Jakarta Sans", size=13)
        )
        st.plotly_chart(risk_fig, use_container_width=True)

    with col_right:
        st.subheader("📊 Key Academic Indicators (Cohort Averages)")
        summary_bar = go.Figure(data=[
            go.Bar(
                name="Averages",
                x=["Attendance %", "Assignment Avg", "Previous Exam", "Study Hrs (x2.5)", "Pred. Final"],
                y=[
                    round(avg_attendance, 1),
                    round(avg_assignment, 1),
                    round(scored_df["Previous_Exam_Score"].mean(), 1),
                    round(scored_df["Study_Hours_Per_Week"].mean() * 2.5, 1),  # Normalized for visualization
                    round(avg_pred_score, 1)
                ],
                marker_color=["#3b82f6", "#6366f1", "#8b5cf6", "#ec4899", "#10b981"],
                text=[
                    f"{avg_attendance:.1f}%",
                    f"{avg_assignment:.1f}",
                    f"{scored_df['Previous_Exam_Score'].mean():.1f}",
                    f"{scored_df['Study_Hours_Per_Week'].mean():.1f}h",
                    f"{avg_pred_score:.1f}"
                ],
                textposition="auto"
            )
        ])
        summary_bar.update_layout(
            yaxis=dict(range=[0, 105], title="Score / Percentage"),
            margin=dict(t=20, b=20, l=20, r=20),
            height=340,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Plus Jakarta Sans", size=13)
        )
        st.plotly_chart(summary_bar, use_container_width=True)

    # Actionable Executive Insights Box
    st.markdown("""
    <div class="academic-callout">
        <b>💡 Executive Advisory:</b>
        Currently, <b>{:.1f}% of students</b> are flagged as High Academic Risk and may fail the final examination 
        without structured academic intervention. The single highest leverage indicator in this cohort is 
        <b>Previous Exam Scores</b> followed by <b>Continuous Assignment Consistency</b>. 
        Teachers are advised to prioritize one-on-one advising for Tier 1 students before Week 10.
    </div>
    """.format(high_risk_pct), unsafe_allow_html=True)


# ==============================================================================
# SECTION B: STUDENT RISK PREDICTION (INTERACTIVE CALCULATOR)
# ==============================================================================
elif menu_option == "🔮 Student Risk Prediction":
    st.markdown("""
    <div class="hero-banner">
        <h1 class="hero-title">Individual Student Risk Predictor</h1>
        <p class="hero-subtitle">
            Input a student's engagement and assessment indicators to estimate their expected final examination score, 
            understand the mathematical factors driving the prediction, and receive personalized academic support guidance.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Preset Quick Load Buttons
    st.markdown("##### ⚡ Quick Demonstration Presets")
    preset_cols = st.columns(4)
    preset_choice = None
    with preset_cols[0]:
        if st.button("🚩 Load High-Risk Student", use_container_width=True):
            st.session_state["p_att"] = 48.0
            st.session_state["p_assign"] = 38.0
            st.session_state["p_prev"] = 34.0
            st.session_state["p_hours"] = 4.0
            preset_choice = "high"
    with preset_cols[1]:
        if st.button("⚠️ Load Borderline Student", use_container_width=True):
            st.session_state["p_att"] = 72.0
            st.session_state["p_assign"] = 52.0
            st.session_state["p_prev"] = 48.0
            st.session_state["p_hours"] = 9.0
            preset_choice = "med"
    with preset_cols[2]:
        if st.button("🌟 Load Top Performer", use_container_width=True):
            st.session_state["p_att"] = 94.0
            st.session_state["p_assign"] = 91.0
            st.session_state["p_prev"] = 89.0
            st.session_state["p_hours"] = 22.0
            preset_choice = "low"
    with preset_cols[3]:
        if st.button("🔄 Reset to Averages", use_container_width=True):
            st.session_state["p_att"] = 78.0
            st.session_state["p_assign"] = 68.0
            st.session_state["p_prev"] = 64.0
            st.session_state["p_hours"] = 12.0
            preset_choice = "avg"

    # Default session state initialization
    if "p_att" not in st.session_state:
        st.session_state["p_att"] = 75.0
    if "p_assign" not in st.session_state:
        st.session_state["p_assign"] = 65.0
    if "p_prev" not in st.session_state:
        st.session_state["p_prev"] = 60.0
    if "p_hours" not in st.session_state:
        st.session_state["p_hours"] = 12.0

    st.markdown("<br>", unsafe_allow_html=True)
    input_left, input_right = st.columns([1, 1])

    with input_left:
        st.subheader("📝 Student Academic Parameters")
        student_name_input = st.text_input("Student Identifier / Name (Optional)", value="Demonstration Student")
        
        in_attendance = st.slider(
            "Attendance Percentage (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(st.session_state["p_att"]),
            step=0.5,
            help="Student's overall lecture and lab attendance rate."
        )

        in_assignment = st.slider(
            "Assignment Average (out of 100)",
            min_value=0.0,
            max_value=100.0,
            value=float(st.session_state["p_assign"]),
            step=0.5,
            help="Average score across homework assignments and continuous assessments."
        )

        in_prev_exam = st.slider(
            "Previous Examination Score (out of 100)",
            min_value=0.0,
            max_value=100.0,
            value=float(st.session_state["p_prev"]),
            step=0.5,
            help="Score achieved in the previous mid-term or prerequisite exam."
        )

        in_study_hours = st.slider(
            "Self-Study Hours Per Week",
            min_value=0.0,
            max_value=40.0,
            value=float(st.session_state["p_hours"]),
            step=0.5,
            help="Reported hours dedicated to weekly independent revision and coursework."
        )

        predict_btn = st.button("🔮 Predict Academic Risk", type="primary", use_container_width=True)

    with input_right:
        st.subheader("🎯 Prediction & Early-Warning Output")

        # Perform prediction
        validation_errors = validate_student_inputs(in_attendance, in_assignment, in_prev_exam, in_study_hours)
        if validation_errors:
            for err in validation_errors:
                st.error(err)
        else:
            inputs_dict = {
                "Attendance_Percentage": in_attendance,
                "Assignment_Average": in_assignment,
                "Previous_Exam_Score": in_prev_exam,
                "Study_Hours_Per_Week": in_study_hours
            }
            predicted_score = predict_score(model, inputs_dict, feature_names)
            risk_info = classify_risk(predicted_score, low_risk_threshold, med_risk_threshold)

            # Visual Indicator: Gauge Chart
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=predicted_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"Expected Final Score: <b>{predicted_score:.1f}</b> / 100", 'font': {'size': 18}},
                delta={'reference': low_risk_threshold, 'increasing': {'color': "#10b981"}, 'decreasing': {'color': "#ef4444"}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
                    'bar': {'color': risk_info["color"], 'thickness': 0.3},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "#cbd5e1",
                    'steps': [
                        {'range': [0, med_risk_threshold], 'color': "rgba(239, 68, 68, 0.25)"},
                        {'range': [med_risk_threshold, low_risk_threshold], 'color': "rgba(245, 158, 11, 0.25)"},
                        {'range': [low_risk_threshold, 100], 'color': "rgba(16, 185, 129, 0.25)"}
                    ],
                    'threshold': {
                        'line': {'color': "#1e293b", 'width': 3},
                        'thickness': 0.85,
                        'value': predicted_score
                    }
                }
            ))
            gauge_fig.update_layout(
                height=260,
                margin=dict(t=30, b=10, l=30, r=30),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Plus Jakarta Sans")
            )
            st.plotly_chart(gauge_fig, use_container_width=True)

            # Risk Category Callout
            st.markdown(f"""
            <div style="background-color: {risk_info['bg_color']}; border-left: 6px solid {risk_info['color']}; padding: 16px 20px; border-radius: 10px; margin-top: 4px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 800; font-size: 1.15rem; color: {risk_info['color']};">
                        {risk_info['icon']} {risk_info['label'].upper()}
                    </span>
                    <span style="font-size: 0.85rem; font-weight: 600; color: #475569;">
                        Passing Cutoff: {low_risk_threshold} pts
                    </span>
                </div>
                <p style="margin: 6px 0 0 0; color: #334155; font-size: 0.92rem;">
                    {risk_info['description']}
                </p>
            </div>
            """, unsafe_allow_html=True)

    # Detailed Feature Breakdown (Equation Contribution)
    st.markdown("---")
    st.subheader("🔬 Mathematical Factor Contribution Breakdown")
    st.caption("How each indicator contributes to the final score according to the Multiple Linear Regression model.")

    contrib_df = calculate_contributions(model, feature_names, inputs_dict)
    
    col_breakdown1, col_breakdown2 = st.columns([1, 1])
    with col_breakdown1:
        st.dataframe(
            contrib_df,
            column_config={
                "Feature": "Parameter",
                "Input_Value": st.column_config.NumberColumn("Input Value", format="%.1f"),
                "Coefficient": st.column_config.NumberColumn("Weight (β)", format="%.3f"),
                "Contribution": st.column_config.NumberColumn("Score Points (+/-)", format="%.2f")
            },
            hide_index=True,
            use_container_width=True
        )

    with col_breakdown2:
        # Waterfall or horizontal bar of contributions
        waterfall_fig = go.Figure(go.Bar(
            x=contrib_df["Contribution"],
            y=contrib_df["Feature"],
            orientation='h',
            marker=dict(
                color=["#64748b" if t == "Baseline" else "#4f46e5" for t in contrib_df["Type"]]
            ),
            text=[f"{v:+.1f} pts" for v in contrib_df["Contribution"]],
            textposition="auto"
        ))
        waterfall_fig.update_layout(
            title="Points Added to Final Score",
            xaxis_title="Points Contributed",
            margin=dict(t=35, b=20, l=120, r=20),
            height=250,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Plus Jakarta Sans", size=12)
        )
        st.plotly_chart(waterfall_fig, use_container_width=True)

    # Tailored Supportive Academic Improvement Steps
    st.markdown("---")
    st.subheader("📋 Tailored Academic Improvement Plan")
    recs = generate_recommendations(
        attendance=in_attendance,
        assignment=in_assignment,
        prev_exam=in_prev_exam,
        study_hours=in_study_hours,
        predicted_score=predicted_score,
        risk_category=risk_info["category"]
    )

    for r in recs:
        priority_class = "rec-urgent" if r["priority"] == "Urgent" else ("rec-moderate" if r["priority"] == "Moderate" else "rec-growth")
        badge_color = "#ef4444" if r["priority"] == "Urgent" else ("#f59e0b" if r["priority"] == "Moderate" else "#10b981")
        st.markdown(f"""
        <div class="rec-card {priority_class}">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span style="font-weight: 700; color: #1e293b; font-size: 0.95rem;">{r['domain']}</span>
                <span style="background: {badge_color}18; color: {badge_color}; border: 1px solid {badge_color}40; padding: 2px 8px; border-radius: 9999px; font-size: 0.75rem; font-weight: 700;">
                    {r['priority'].upper()}
                </span>
            </div>
            <p style="margin: 0; color: #475569; font-size: 0.88rem; line-height: 1.5;">{r['action']}</p>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# SECTION C: STUDENT RECORDS (SEARCHABLE & FILTERABLE TABLE)
# ==============================================================================
elif menu_option == "📋 Student Records":
    st.markdown("""
    <div class="hero-banner">
        <h1 class="hero-title">Student Cohort Roster & Performance Records</h1>
        <p class="hero-subtitle">
            Search, filter, and inspect student records. Identify cohorts requiring outreach and export clean reports for faculty meetings.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Search & Filter Controls
    col_search, col_risk, col_score = st.columns([1.5, 1, 1.5])
    
    with col_search:
        search_query = st.text_input("🔍 Search by Student Name or ID", placeholder="e.g. STU1024 or Jordan")
    
    with col_risk:
        risk_filter = st.selectbox(
            "Filter by Risk Category",
            options=["All Categories", "High Risk", "Medium Risk", "Low Risk"],
            index=0
        )
        
    with col_score:
        score_range = st.slider(
            "Predicted Score Range",
            min_value=0.0,
            max_value=100.0,
            value=(0.0, 100.0),
            step=1.0
        )

    # Filter Logic
    filtered_df = scored_df.copy()
    if search_query:
        q = search_query.lower()
        filtered_df = filtered_df[
            filtered_df["Student_Name"].str.lower().str.contains(q) |
            filtered_df["Student_ID"].str.lower().str.contains(q)
        ]
    if risk_filter != "All Categories":
        filtered_df = filtered_df[filtered_df["Risk_Category"] == risk_filter]

    filtered_df = filtered_df[
        (filtered_df["Predicted_Final_Score"] >= score_range[0]) &
        (filtered_df["Predicted_Final_Score"] <= score_range[1])
    ]

    st.markdown(f"**Showing {len(filtered_df)} of {len(scored_df)} student records**")

    # Display Table with Formatting
    columns_to_show = [
        "Student_ID", "Student_Name", "Attendance_Percentage", "Assignment_Average",
        "Previous_Exam_Score", "Study_Hours_Per_Week", "Final_Exam_Score",
        "Predicted_Final_Score", "Residual_Error", "Risk_Category"
    ]
    
    st.dataframe(
        filtered_df[columns_to_show],
        column_config={
            "Student_ID": "Student ID",
            "Student_Name": "Full Name",
            "Attendance_Percentage": st.column_config.ProgressColumn("Attendance %", format="%.1f%%", min_value=0, max_value=100),
            "Assignment_Average": st.column_config.NumberColumn("Assignments", format="%.1f"),
            "Previous_Exam_Score": st.column_config.NumberColumn("Prev Exam", format="%.1f"),
            "Study_Hours_Per_Week": st.column_config.NumberColumn("Study Hrs/Wk", format="%.1f"),
            "Final_Exam_Score": st.column_config.NumberColumn("Actual Final", format="%.1f"),
            "Predicted_Final_Score": st.column_config.NumberColumn("Predicted Final", format="%.1f"),
            "Residual_Error": st.column_config.NumberColumn("Residual (Act - Pred)", format="%+.1f"),
            "Risk_Category": "Risk Tier"
        },
        hide_index=True,
        use_container_width=True
    )

    # CSV Download Button
    csv_buffer = io.StringIO()
    filtered_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📥 Download Filtered Records as CSV",
        data=csv_buffer.getvalue(),
        file_name="student_academic_risk_filtered_cohort.csv",
        mime="text/csv",
        use_container_width=False
    )

    # Individual Student Drill-Down Expander
    st.markdown("---")
    st.subheader("🔍 Individual Student Academic Profile Viewer")
    selected_id = st.selectbox(
        "Select a student from current cohort to inspect:",
        options=filtered_df["Student_ID"].tolist() if not filtered_df.empty else []
    )
    if selected_id:
        student_row = filtered_df[filtered_df["Student_ID"] == selected_id].iloc[0]
        
        drill_col1, drill_col2, drill_col3 = st.columns([1, 1, 1.2])
        with drill_col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Student Profile</div>
                <div class="metric-value" style="font-size: 1.3rem;">{student_row['Student_Name']}</div>
                <div class="metric-subtext">ID: <b>{student_row['Student_ID']}</b></div>
            </div>
            """, unsafe_allow_html=True)
        with drill_col2:
            badge_color = "#dc2626" if student_row['Risk_Category'] == "High Risk" else ("#d97706" if student_row['Risk_Category'] == "Medium Risk" else "#059669")
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Predicted Final Score</div>
                <div class="metric-value" style="color: {badge_color};">{student_row['Predicted_Final_Score']:.1f}</div>
                <div class="metric-subtext">Actual: <b>{student_row.get('Final_Exam_Score', 'N/A')}</b> | Residual: <b>{student_row.get('Residual_Error', 'N/A')}</b></div>
            </div>
            """, unsafe_allow_html=True)
        with drill_col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Input Indicators</div>
                <div style="font-size: 0.88rem; color: #334155; line-height: 1.6;">
                    • Attendance: <b>{student_row['Attendance_Percentage']:.1f}%</b><br>
                    • Assignment Avg: <b>{student_row['Assignment_Average']:.1f}</b><br>
                    • Prev Exam: <b>{student_row['Previous_Exam_Score']:.1f}</b><br>
                    • Study Time: <b>{student_row['Study_Hours_Per_Week']:.1f} hrs/wk</b>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ==============================================================================
# SECTION D: DATA VISUALIZATIONS & ANALYTICS
# ==============================================================================
elif menu_option == "📈 Data Visualizations":
    st.markdown("""
    <div class="hero-banner">
        <h1 class="hero-title">Academic Data Analytics & Feature Exploration</h1>
        <p class="hero-subtitle">
            Explore empirical correlations and distributions across student attendance, assignments, previous exams, and study hours.
        </p>
    </div>
    """, unsafe_allow_html=True)

    chart_tabs = st.tabs([
        "📊 Bivariate Relationships",
        "📦 Distributions & Risk Tiers",
        "🔥 Correlation Heatmap"
    ])

    # TAB 1: Bivariate Scatter Plots with Trendlines
    with chart_tabs[0]:
        st.subheader("Scatter Plots vs. Final Examination Score")
        scatter_col1, scatter_col2 = st.columns(2)

        with scatter_col1:
            # 1. Attendance vs Final Exam
            fig_att = px.scatter(
                scored_df,
                x="Attendance_Percentage",
                y="Final_Exam_Score",
                color="Risk_Category",
                color_discrete_map={"High Risk": "#ef4444", "Medium Risk": "#f59e0b", "Low Risk": "#10b981"},
                trendline="ols",
                title="Attendance Percentage vs. Final Exam Score",
                labels={"Attendance_Percentage": "Attendance (%)", "Final_Exam_Score": "Final Exam Score"}
            )
            fig_att.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Plus Jakarta Sans"))
            st.plotly_chart(fig_att, use_container_width=True)

            # 2. Previous Exam vs Final Exam
            fig_prev = px.scatter(
                scored_df,
                x="Previous_Exam_Score",
                y="Final_Exam_Score",
                color="Risk_Category",
                color_discrete_map={"High Risk": "#ef4444", "Medium Risk": "#f59e0b", "Low Risk": "#10b981"},
                trendline="ols",
                title="Previous Exam Score vs. Final Exam Score",
                labels={"Previous_Exam_Score": "Previous Exam Score", "Final_Exam_Score": "Final Exam Score"}
            )
            fig_prev.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Plus Jakarta Sans"))
            st.plotly_chart(fig_prev, use_container_width=True)

        with scatter_col2:
            # 3. Assignment vs Final Exam
            fig_assign = px.scatter(
                scored_df,
                x="Assignment_Average",
                y="Final_Exam_Score",
                color="Risk_Category",
                color_discrete_map={"High Risk": "#ef4444", "Medium Risk": "#f59e0b", "Low Risk": "#10b981"},
                trendline="ols",
                title="Assignment Average vs. Final Exam Score",
                labels={"Assignment_Average": "Assignment Average", "Final_Exam_Score": "Final Exam Score"}
            )
            fig_assign.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Plus Jakarta Sans"))
            st.plotly_chart(fig_assign, use_container_width=True)

            # 4. Study Hours vs Final Exam
            fig_hours = px.scatter(
                scored_df,
                x="Study_Hours_Per_Week",
                y="Final_Exam_Score",
                color="Risk_Category",
                color_discrete_map={"High Risk": "#ef4444", "Medium Risk": "#f59e0b", "Low Risk": "#10b981"},
                trendline="ols",
                title="Study Hours Per Week vs. Final Exam Score",
                labels={"Study_Hours_Per_Week": "Study Hours / Week", "Final_Exam_Score": "Final Exam Score"}
            )
            fig_hours.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Plus Jakarta Sans"))
            st.plotly_chart(fig_hours, use_container_width=True)

    # TAB 2: Distributions & Risk Categories
    with chart_tabs[1]:
        dist_col1, dist_col2 = st.columns(2)
        with dist_col1:
            st.subheader("Distribution of Predicted Final Exam Scores")
            fig_hist = px.histogram(
                scored_df,
                x="Predicted_Final_Score",
                nbins=35,
                color="Risk_Category",
                color_discrete_map={"High Risk": "#ef4444", "Medium Risk": "#f59e0b", "Low Risk": "#10b981"},
                marginal="box",
                title="Predicted Final Score Histogram & Box Plot",
                labels={"Predicted_Final_Score": "Predicted Score"}
            )
            fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Plus Jakarta Sans"))
            st.plotly_chart(fig_hist, use_container_width=True)

        with dist_col2:
            st.subheader("Academic Risk Tier Breakdown")
            risk_counts = scored_df["Risk_Category"].value_counts().reset_index()
            risk_counts.columns = ["Risk_Category", "Count"]
            fig_bar = px.bar(
                risk_counts,
                x="Risk_Category",
                y="Count",
                color="Risk_Category",
                color_discrete_map={"High Risk": "#ef4444", "Medium Risk": "#f59e0b", "Low Risk": "#10b981"},
                text="Count",
                title="Count of Students by Academic Risk Category"
            )
            fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Plus Jakarta Sans"))
            st.plotly_chart(fig_bar, use_container_width=True)

    # TAB 3: Correlation Heatmap
    with chart_tabs[2]:
        st.subheader("Feature Correlation Matrix (Pearson r)")
        st.caption("Quantifies the strength and direction of linear associations between all academic variables.")
        
        numeric_cols = [
            "Attendance_Percentage", "Assignment_Average", "Previous_Exam_Score",
            "Study_Hours_Per_Week", "Final_Exam_Score"
        ]
        corr_matrix = scored_df[numeric_cols].corr().round(3)

        fig_heat = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Pearson Correlation Heatmap"
        )
        fig_heat.update_layout(
            height=450,
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Plus Jakarta Sans", size=13)
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown("""
        <div class="academic-callout">
            <b>Key Correlation Insights:</b>
            <ul>
                <li><b>Previous Exam Score:</b> Exhibits the strongest positive correlation with Final Exam Score, highlighting the predictive value of baseline mastery.</li>
                <li><b>Continuous Assignments & Attendance:</b> Strongly correlated with each other and moderately to strongly correlated with the final grade.</li>
                <li><b>Study Hours:</b> Positively reinforces performance, acting as an important moderating factor for borderline students.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# SECTION E: MODEL PERFORMANCE & EVALUATION
# ==============================================================================
elif menu_option == "⚙️ Model Performance":
    st.markdown("""
    <div class="hero-banner">
        <h1 class="hero-title">Model Performance & Algorithmic Evaluation</h1>
        <p class="hero-subtitle">
            Comprehensive statistical evaluation of the Multiple Linear Regression model on unseen holdout test data.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Metric Cards
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.markdown(f"""
        <div class="metric-card" style="border-top: 4px solid #4f46e5;">
            <div class="metric-label">R² Score (Variance Explained)</div>
            <div class="metric-value" style="color: #4f46e5;">{metrics['r2']:.4f}</div>
            <div class="metric-subtext">Training R²: <b>{metrics.get('r2_train', 'N/A')}</b></div>
        </div>
        """, unsafe_allow_html=True)
    with m_col2:
        st.markdown(f"""
        <div class="metric-card" style="border-top: 4px solid #06b6d4;">
            <div class="metric-label">Mean Absolute Error (MAE)</div>
            <div class="metric-value" style="color: #0891b2;">{metrics['mae']:.2f} pts</div>
            <div class="metric-subtext">Average magnitude of error</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col3:
        st.markdown(f"""
        <div class="metric-card" style="border-top: 4px solid #8b5cf6;">
            <div class="metric-label">Root Mean Squared Error (RMSE)</div>
            <div class="metric-value" style="color: #7c3aed;">{metrics['rmse']:.2f} pts</div>
            <div class="metric-subtext">Penalizes larger outliers</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col4:
        st.markdown(f"""
        <div class="metric-card" style="border-top: 4px solid #f59e0b;">
            <div class="metric-label">Mean Squared Error (MSE)</div>
            <div class="metric-value" style="color: #d97706;">{metrics['mse']:.2f}</div>
            <div class="metric-subtext">Variance of residuals</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Plain-Language Explanation of Metrics
    st.subheader("📖 Simple Language Guide to Evaluation Metrics")
    guide_cols = st.columns(2)
    with guide_cols[0]:
        st.markdown("""
        - **R² Score (Coefficient of Determination):**
          Measures what percentage of variation in final exam scores is explained by the model's inputs. A score of **{:.2f}%** means over three-quarters of score differences among students are captured by attendance, assignments, previous exams, and study hours.
        - **Mean Absolute Error (MAE):**
          The average point difference between the model's forecast and the student's actual score. On average, predictions are off by only **{:.2f} points** out of 100.
        """.format(metrics['r2'] * 100, metrics['mae']))
    with guide_cols[1]:
        st.markdown("""
        - **Root Mean Squared Error (RMSE):**
          Similar to MAE, but gives extra mathematical weight to larger prediction errors. An RMSE of **{:.2f} points** confirms that severe outliers are rare.
        - **Mean Squared Error (MSE):**
          The raw average of squared errors ({:.2f}). Primarily used by optimizers during training to penalize large mistakes.
        """.format(metrics['rmse'], metrics['mse']))

    # Regression Equation Display
    st.markdown("---")
    st.subheader("📐 Learned Multiple Linear Regression Equation")
    st.markdown(f"""
    <div class="formula-box">
        {equation_str}
    </div>
    """, unsafe_allow_html=True)

    # Actual vs Predicted & Residuals
    st.markdown("<br>", unsafe_allow_html=True)
    plot_col1, plot_col2 = st.columns(2)

    with plot_col1:
        st.subheader("Actual vs. Predicted Final Exam Scores")
        # Scatter of actual vs predicted with ideal 45 deg line
        fig_actual_pred = go.Figure()
        fig_actual_pred.add_trace(go.Scatter(
            x=scored_df["Final_Exam_Score"],
            y=scored_df["Predicted_Final_Score"],
            mode="markers",
            marker=dict(color="#4f46e5", opacity=0.6, size=7),
            name="Student Records"
        ))
        # 45-degree perfect prediction line
        fig_actual_pred.add_trace(go.Scatter(
            x=[0, 100],
            y=[0, 100],
            mode="lines",
            line=dict(color="#ef4444", dash="dash", width=2),
            name="Perfect Forecast (Ideal)"
        ))
        fig_actual_pred.update_layout(
            xaxis_title="Actual Final Exam Score",
            yaxis_title="Predicted Final Exam Score",
            margin=dict(t=30, b=20, l=20, r=20),
            height=380,
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Plus Jakarta Sans")
        )
        st.plotly_chart(fig_actual_pred, use_container_width=True)

    with plot_col2:
        st.subheader("Residual Error Distribution (Actual - Predicted)")
        residuals = scored_df["Residual_Error"]
        fig_resid = px.histogram(
            residuals,
            nbins=35,
            title="Distribution of Model Residuals (Bell-Curve Diagnostic)",
            color_discrete_sequence=["#06b6d4"]
        )
        fig_resid.add_vline(x=0, line_dash="dash", line_color="#ef4444")
        fig_resid.update_layout(
            xaxis_title="Residual Error (Points)",
            yaxis_title="Count of Students",
            margin=dict(t=30, b=20, l=20, r=20),
            height=380,
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Plus Jakarta Sans")
        )
        st.plotly_chart(fig_resid, use_container_width=True)


# ==============================================================================
# SECTION F: EARLY-WARNING RECOMMENDATIONS & INTERVENTION PROTOCOL
# ==============================================================================
elif menu_option == "💡 Early-Warning Recommendations":
    st.markdown("""
    <div class="hero-banner">
        <h1 class="hero-title">Early-Warning Intervention Protocol</h1>
        <p class="hero-subtitle">
            A comprehensive, non-punitive multi-tiered framework for faculty, academic advisors, and mentors 
            to translate analytical alerts into compassionate student success actions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### 🏛️ The Three-Tier Academic Support Matrix
    Our early-warning philosophy is grounded in positive psychology, student empowerment, and proactive remediation.
    """)

    tier_col1, tier_col2, tier_col3 = st.columns(3)
    with tier_col1:
        st.markdown("""
        <div class="metric-card" style="border-top: 5px solid #ef4444; height: 100%;">
            <span class="badge-high-risk">TIER 1 • HIGH RISK</span>
            <h4 style="margin: 10px 0 6px 0; color: #dc2626;">Intensive Multi-Tier Care</h4>
            <p style="font-size: 0.85rem; color: #475569; line-height: 1.5;">
                For students predicted to score below the 40-point threshold.
            </p>
            <hr style="border: none; border-top: 1px solid #f1f5f9; margin: 10px 0;">
            <ul style="font-size: 0.82rem; color: #334155; padding-left: 18px; margin: 0;">
                <li>Convene supportive advising conference within 5 days.</li>
                <li>Establish an Individualized Academic Success Plan (IASP).</li>
                <li>Mandatory assignment clinic and peer tutoring assignment.</li>
                <li>Address external wellness / transportation factors.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with tier_col2:
        st.markdown("""
        <div class="metric-card" style="border-top: 5px solid #f59e0b; height: 100%;">
            <span class="badge-medium-risk">TIER 2 • MEDIUM RISK</span>
            <h4 style="margin: 10px 0 6px 0; color: #d97706;">Targeted Remediation</h4>
            <p style="font-size: 0.85rem; color: #475569; line-height: 1.5;">
                For borderline students predicted between 40 and 60 points.
            </p>
            <hr style="border: none; border-top: 1px solid #f1f5f9; margin: 10px 0;">
            <ul style="font-size: 0.82rem; color: #334155; padding-left: 18px; margin: 0;">
                <li>Mid-term check-in with subject matter faculty.</li>
                <li>Targeted prerequisite topic review worksheets.</li>
                <li>Encourage study group formation and structured time-blocking.</li>
                <li>Low-stakes weekly concept check quizzes.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with tier_col3:
        st.markdown("""
        <div class="metric-card" style="border-top: 5px solid #10b981; height: 100%;">
            <span class="badge-low-risk">TIER 3 • LOW RISK</span>
            <h4 style="margin: 10px 0 6px 0; color: #059669;">Enrichment & Leadership</h4>
            <p style="font-size: 0.85rem; color: #475569; line-height: 1.5;">
                For students predicted at 60 points or above.
            </p>
            <hr style="border: none; border-top: 1px solid #f1f5f9; margin: 10px 0;">
            <ul style="font-size: 0.82rem; color: #334155; padding-left: 18px; margin: 0;">
                <li>Encourage enrollment in advanced honors or elective research.</li>
                <li>Invite student to serve as peer study group facilitator.</li>
                <li>Support competitive internship or academic conference applications.</li>
                <li>Acknowledge consistent effort and maintain momentum.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📋 Domain-Specific Faculty Intervention Playbook")

    with st.expander("📌 1. Attendance & Class Engagement Interventions", expanded=True):
        st.markdown("""
        - **Proactive Empathy Check-In:** Avoid accusatory language ("Why are you skipping?"). Instead: *"I noticed you were unable to attend the last few lectures. We missed your presence—is everything alright with your schedule, transit, or wellness?"*
        - **Lecture Catch-Up Bundles:** Provide access to annotated slide decks, audio summaries, and peer study notes so missing a class doesn't cause cumulative knowledge debt.
        - **Attendance Buddy System:** Pair students with active peers who can remind each other about class sessions.
        """)

    with st.expander("📌 2. Continuous Assignment & Coursework Support"):
        st.markdown("""
        - **Milestone Submissions:** For extensive term papers or coding projects, break deliverables into 3 manageable checkpoints (Outline, Draft, Final) with partial credit.
        - **Rubric Demystification:** Host an informal 15-minute walkthrough of grading rubrics prior to submission deadlines.
        - **Writing & Problem-Solving Labs:** Partner with institutional learning resource centers to provide targeted feedback on drafts.
        """)

    with st.expander("📌 3. Foundational Exam Prep & Remediation"):
        st.markdown("""
        - **Diagnostic Prerequisite Review:** Identify specific syllabus units where the student lost points in the previous examination.
        - **Worked Example Coaching:** Provide step-by-step problem sets illustrating common misconceptions and effective answer structuring.
        - **Mock Timed Practices:** Offer low-stakes mock exams under realistic conditions to build test-taking confidence and reduce examination anxiety.
        """)


# ==============================================================================
# SECTION G: EXPLAINABILITY, ETHICS & FAIRNESS
# ==============================================================================
elif menu_option == "🛡️ Explainability & Fairness":
    st.markdown("""
    <div class="hero-banner">
        <h1 class="hero-title">Model Explainability, Ethics & Fairness</h1>
        <p class="hero-subtitle">
            Our commitment to glass-box transparency, algorithmic fairness, ethical machine learning, 
            and the responsible application of predictive analytics in higher education.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_eth1, col_eth2 = st.columns(2)

    with col_eth1:
        st.subheader("🔍 Why Multiple Linear Regression?")
        st.markdown("""
        In high-stakes academic environments, **interpretability is non-negotiable**:
        - **Complete Glass-Box Transparency:** Unlike black-box neural networks or complex ensembles, every single point in the prediction can be audited using simple arithmetic.
        - **Direct Pedagogical Meaning:** Each coefficient directly informs educators of the expected score increase associated with a 1-unit improvement in study hours, attendance, or assignments.
        - **Actionable Feedback:** Because the relationship is transparent, teachers can show students exactly how raising their attendance by 10% or studying 4 more hours per week changes their trajectory.
        """)

        st.subheader("⚖️ Strict Exclusion of Sensitive Attributes")
        st.markdown("""
        To prevent algorithmic bias and discriminatory harm:
        - **No Demographics:** The model strictly excludes gender, race, caste, religion, socioeconomic proxies, disability status, or geographic origin.
        - **Behavioral & Academic Focus Only:** Predictions are strictly derived from direct, modifiable academic behaviors (attendance, assignments, exams, study time).
        """)

    with col_eth2:
        st.subheader("⚠️ Correlation Does Not Equal Causation")
        st.markdown("""
        <div class="academic-callout" style="border-left-color: #f59e0b;">
            <b>Important Statistical Advisory:</b><br>
            A high statistical correlation between study hours and final exam scores does not guarantee that merely logging hours 
            automatically produces higher grades without active, deliberate learning. Teachers and students must focus on the 
            <b>quality of study</b> rather than treating the regression equation as a mechanical guarantee.
        </div>
        """, unsafe_allow_html=True)

        st.subheader("🤝 Human-in-the-Loop Principle")
        st.markdown("""
        - **Advisory Support, Never Automated Decisions:** 
          The EduGuard system is explicitly designed as a supportive decision-aid for teachers. It **never** automates grading, course failures, admissions, or administrative penalties.
        - **Holistic Context:** 
          A model cannot observe personal hardships, bereavement, illness, or creative breakthroughs. Teachers must always exercise professional human judgment.
        - **Non-Punitive Language:** 
          Risk indicators are designed to allocate supportive resources and care, never to stigmatize or label a student negatively.
        """)

    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #94a3b8; font-size: 0.85rem; padding: 16px 0;">
        EduGuard Academic Risk Early-Warning System • Built for compassionate educational excellence with Python, Scikit-learn, and Streamlit.
    </div>
    """, unsafe_allow_html=True)
