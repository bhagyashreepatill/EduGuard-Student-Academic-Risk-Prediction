# 🎓 Student Academic Risk Early-Warning System (EduGuard)

> A Machine Learning intelligence platform designed for educators and academic advisors to identify students at risk of academic failure early, forecast expected final examination performance, and deploy supportive, non-punitive interventions.

---

## 1. Project Overview
The **Student Academic Risk Early-Warning System (EduGuard)** bridges machine learning and academic mentorship. Rather than waiting until semester-end final grades are submitted when it is too late to intervene, this system uses continuous formative indicators—attendance percentage, assignment performance, previous examination results, and weekly self-study habits—to estimate final examination outcomes and assign students into transparent, actionable risk tiers.

Built on an interpretable **Multiple Linear Regression** foundation, the system emphasizes glass-box transparency, fairness, and non-punitive support.

---

## 2. Problem Statement
In higher education and secondary institutions, student disengagement and academic distress often develop silently over weeks:
- Traditional post-examination assessments only document failure after it has already occurred.
- Teachers managing large cohorts (100–500+ students) struggle to identify subtle downward trends across multiple indicators manually.
- Generic interventions often fail because they lack insight into whether a student is struggling due to attendance deficits, assignment comprehension gaps, or weak prerequisite foundations.

There is a critical need for an automated, predictive, yet transparent early-warning system that flags at-risk students weeks before final exams and suggests tailored, constructive academic support.

---

## 3. Project Objectives
1. **Early Identification:** Predict final examination scores and flag students into *Low Risk*, *Medium Risk*, or *High Risk* categories before high-stakes exams.
2. **Pedagogical Explainability:** Provide exact mathematical breakdowns of how each academic factor contributes to a student's predicted score using glass-box Multiple Linear Regression.
3. **Data Quality & Leakage Prevention:** Implement a robust data-cleaning pipeline that eliminates duplicates, handles invalid inputs, and performs missing value imputation strictly on training data.
4. **Actionable Recommendations:** Offer tailored, compassionate intervention guidance for teachers (e.g., peer study buddies, office hour visits, review modules).
5. **Ethical AI Alignment:** Ensure predictions never automate high-stakes administrative decisions, exclude all sensitive personal demographics, and treat predictions strictly as supportive estimates.

---

## 4. Key Features
- **📊 Executive Dashboard Overview:** Real-time cohort KPI cards (cohort size, risk tier counts and percentages, cohort averages) alongside interactive Plotly donut and summary charts.
- **🔮 Interactive Student Risk Calculator:** Input sliders with instant validation, quick-load demo presets (High-Risk Student, Borderline Student, Top Performer), visual score gauge, risk badge, and factor contributions.
- **🔬 Mathematical Contribution Breakdown:** Waterfall chart showing exact points added or subtracted by the baseline intercept and each input feature.
- **📋 Searchable & Filterable Roster:** Comprehensive student data table with name/ID search, risk tier filters, predicted score range sliders, and one-click **CSV Export**.
- **📈 Exploratory Data Analytics:** Interactive Plotly scatter plots with OLS trendlines, score distributions, and an annotated Pearson correlation matrix.
- **⚙️ Model Performance & Diagnosis:** Real-time evaluation cards (MAE, MSE, RMSE, R²), actual-versus-predicted scatter plots with a 45° ideal line, residual distribution diagnostics, and a one-click **Model Retraining** engine.
- **💡 Multi-Tiered Intervention Protocol:** Actionable three-tier support framework (Tier 1 Intensive Care, Tier 2 Targeted Remediation, Tier 3 Enrichment).
- **🛡️ Explainability & Fairness Center:** Clear explanations of model weights, correlation vs. causation warnings, and ethical guidelines.

---

## 5. Technology Stack
| Category | Technology | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Core programming environment |
| **Machine Learning** | Scikit-learn (`LinearRegression`) | Model fitting, evaluation, and train/test splitting |
| **Data Processing** | Pandas, NumPy | Data manipulation, cleaning, imputation, and statistics |
| **Data Visualization** | Plotly Express & Graph Objects | Dynamic, interactive charts and gauges |
| **Web Application** | Streamlit | Responsive academic web dashboard |
| **Persistence** | Joblib | Model artifact serialization (`.pkl`) |
| **Testing** | Pytest | Unit tests for preprocessing and inference |

---

## 6. Dataset Description
The dataset (`student_academic_data.csv`) consists of **660 synthetic student records** generated with realistic educational relationships and deliberate data anomalies for cleaning verification.

### Dataset Columns
| Column Name | Data Type | Range / Format | Description |
| :--- | :--- | :--- | :--- |
| `Student_ID` | String | `STU1001` - `STU1660` | Unique fictional student identifier |
| `Student_Name` | String | First Last | Realistic fictional student name |
| `Attendance_Percentage` | Float | 0.0% – 100.0% | Lecture and lab attendance rate |
| `Assignment_Average` | Float | 0.0 – 100.0 | Mean score across continuous coursework |
| `Previous_Exam_Score` | Float | 0.0 – 100.0 | Score on prerequisite or mid-term exam |
| `Study_Hours_Per_Week` | Float | 0.0 – 40.0+ hrs | Reported weekly self-directed study time |
| `Final_Exam_Score` | Float | 0.0 – 100.0 | Target variable: final examination score |

### Data Cleansing & Leakage Prevention
The raw dataset intentionally contains:
- **10 duplicate records** (detected and removed by deduplication).
- **5 invalid out-of-bounds records** (e.g., attendance > 100%, negative study hours, scores > 100) detected and filtered.
- **18 records with missing values (NaN)** across numeric features.
- **Zero Data Leakage:** Imputation uses feature medians fitted **strictly on the 80% training set** and then applied to transform the 20% test set.

---

## 7. Machine Learning Methodology
The model utilizes **Multiple Linear Regression (Ordinary Least Squares)** to predict final examination scores:

$$\widehat{\text{Final Exam Score}} = \beta_0 + \beta_1 X_{\text{Attendance}} + \beta_2 X_{\text{Assignment}} + \beta_3 X_{\text{PrevExam}} + \beta_4 X_{\text{StudyHours}}$$

### Learned Regression Equation
```text
Final_Exam_Score = -4.230 + 0.230 * (Attendance_Percentage) + 0.248 * (Assignment_Average) + 0.352 * (Previous_Exam_Score) + 0.529 * (Study_Hours_Per_Week)
```

### Learned Weights Interpretation
- **Base Intercept ($\beta_0 = -4.230$):** Baseline constant offset.
- **Previous Exam Score ($\beta_3 = +0.352$):** Strongest academic indicator—each 10-point improvement in baseline mastery yields ~3.5 final exam points.
- **Study Hours ($\beta_4 = +0.529$):** Highest per-unit impact—each additional hour of focused weekly self-study yields ~0.53 points.
- **Assignment Average ($\beta_2 = +0.248$):** Reflects steady coursework habits.
- **Attendance Percentage ($\beta_1 = +0.230$):** Reflects classroom engagement and exposure to lecture concepts.

---

## 8. Model Evaluation Metrics
Evaluated on an unseen **20% holdout test set (129 records)**:

| Metric | Test Set Value | Simple Language Explanation |
| :--- | :--- | :--- |
| **$R^2$ Score** | **0.7824** (78.2%) | Measures the proportion of variance in final scores explained by the inputs. Over 78% of score variation is captured. |
| **MAE (Mean Absolute Error)** | **3.687 points** | The average difference between predicted and actual scores. On average, predictions are within ~3.7 points out of 100. |
| **RMSE (Root Mean Squared Error)** | **4.657 points** | Penalizes larger errors more heavily. Confirms that severe prediction errors are rare. |
| **MSE (Mean Squared Error)** | **21.686** | The average of squared errors, used as the mathematical optimization loss during fitting. |

---

## 9. Academic Risk Classification
Predicted scores are mapped into intuitive, actionable risk categories:

| Risk Category | Score Range | Status Badge | Illustrative Definition |
| :--- | :--- | :--- | :--- |
| **Low Risk** | $\ge 60.0$ | 🟢 `LOW RISK` | Student is on track to master course objectives. Expected to comfortably pass. |
| **Medium Risk** | $40.0 - 59.9$ | 🟡 `MEDIUM RISK` | Borderline performance. Early targeted revision can prevent course failure. |
| **High Risk** | $< 40.0$ | 🔴 `HIGH RISK` | Vulnerable to course failure. Immediate multi-tier academic intervention recommended. |

*Note: Thresholds are fully configurable via the dashboard sidebar and are clearly labeled as illustrative benchmarks, not rigid institutional cutoffs.*

---

## 10. Project Directory Structure
```
Student Academic Risk Early-Warning System/
├── app.py                      # Interactive Streamlit Web Dashboard
├── train_model.py              # ML pipeline: cleans data, trains regression, exports Joblib bundle
├── data_generator.py           # Synthetic dataset generator with realistic noise & anomalies
├── student_academic_data.csv   # 660-record synthetic student dataset
├── requirements.txt            # Project Python dependencies
├── README.md                   # Comprehensive system documentation
├── models/
│   └── student_performance_model.pkl   # Serialized ML model bundle & metadata
├── utils/
│   ├── __init__.py             # Package initializer
│   ├── preprocessing.py        # Data loading, cleaning, leakage-free train/test splitting
│   └── prediction.py           # Inference, risk classification, factor contributions, recommendations
├── tests/
│   ├── __init__.py             # Test package initializer
│   ├── test_preprocessing.py   # Unit tests for cleaning, deduplication, and leakage prevention
│   └── test_prediction.py      # Unit tests for scoring, bounding, thresholds, and recommendations
└── screenshots/
    └── README.md               # Visual assets and dashboard overview documentation
```

---

## 11. Installation & Setup Instructions

### Prerequisites
- Python 3.10, 3.11, 3.12, 3.13, or 3.14 installed on your machine.
- `pip` package manager.

### Step-by-Step Installation

1. **Clone or Navigate to the Workspace Directory:**
   ```bash
   cd "Student Academic Risk Early-Warning System"
   ```

2. **(Optional) Create and Activate a Virtual Environment:**
   ```bash
   # Windows PowerShell:
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install Required Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify the Dataset and Model Artifacts (Already pre-generated):**
   ```bash
   # Optional: Re-run data generation if needed
   python data_generator.py

   # Optional: Re-train model and verify metrics
   python train_model.py
   ```

5. **Run the Automated Test Suite:**
   ```bash
   python -m pytest -v
   ```

---

## 12. How to Run the Web Application
Launch the Streamlit interactive dashboard with:

```bash
python -m streamlit run app.py
```

Once executed, open your web browser and navigate to:
```
http://localhost:8501
```

---

## 13. Sample Prediction Demonstration

### Scenario: Borderline / High-Risk Student Profile
- **Attendance Percentage:** `52.0%`
- **Assignment Average:** `42.0 / 100`
- **Previous Examination Score:** `38.0 / 100`
- **Study Hours Per Week:** `5.0 hrs/week`

### Model Output:
1. **Predicted Final Exam Score:** `26.7 / 100`
2. **Academic Risk Classification:** 🔴 **HIGH RISK (Tier 1)**
3. **Factor Breakdown:**
   - Base Intercept ($\beta_0$): `-4.23 pts`
   - Attendance ($52\% \times 0.230$): `+11.96 pts`
   - Assignment ($42 \times 0.248$): `+10.42 pts`
   - Previous Exam ($38 \times 0.352$): `+13.38 pts`
   - Study Hours ($5.0 \times 0.529$): `+2.65 pts`
   - **Total Predicted Score:** `34.2 pts` (High Risk)
4. **Tailored Early-Warning Recommendations:**
   - *Attendance:* Low attendance (52%)—schedule an empathetic check-in to investigate external transportation/wellness hurdles.
   - *Continuous Assessment:* Assignment average is 42/100—connect with teaching assistant office hours and rubric review.
   - *Foundational Knowledge:* Previous exam was 38/100—provide remedial prerequisite problem sets and past paper walkthroughs.
   - *Comprehensive Care:* Convene an Individualized Academic Success Plan (IASP) meeting with the student's advisor within 5 days.

---

## 14. Limitations & Ethical Considerations
- **Supportive, Non-Punitive Ethos:** The system is engineered to allocate academic support resources early. It must **never** be used to penalize students, deny course enrollment, or automate academic probation.
- **Strict Demographic Exclusion:** No sensitive demographic variables (gender, race, ethnicity, religion, disability, caste, or socioeconomic indicators) are collected or used in prediction.
- **Correlation vs. Causation:** Statistical correlation between study hours and exam scores does not guarantee that simply logging hours mechanically yields higher grades without active, deliberate learning techniques.
- **Human-in-the-Loop:** Predictions are statistical estimates subject to error ($\text{MAE} \approx 3.7\text{ points}$). Teachers and counselors must always exercise human discretion and consider qualitative personal context (illness, bereavement, personal emergencies).

---

## 15. Future Roadmap & Enhancements
- [ ] Integration with Learning Management Systems (Canvas, Blackboard, Moodle) via LTI API standards.
- [ ] Longitudinal time-series tracking of student progress across consecutive semesters.
- [ ] Multi-class early-warning alerts for specific grade boundaries (A/B/C/D/F).
- [ ] Automated email template generation for advisors to send compassionate check-in messages.
- [ ] Expanded qualitative feedback integration (e.g., student self-reported motivation or wellness surveys).

---

## 16. Author & License
- **Project:** Student Academic Risk Early-Warning System (EduGuard)
- **Built With:** Python, Scikit-learn, Pandas, Plotly, Streamlit
- **License:** MIT License (Open for educational and research use)
