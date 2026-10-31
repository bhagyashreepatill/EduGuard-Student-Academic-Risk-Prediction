"""
Data Generator for Student Academic Risk Early-Warning System.
Generates realistic synthetic student academic records with realistic statistical
correlations, realistic noise, and deliberate data-quality anomalies (missing values,
duplicates, invalid values) for data-cleaning demonstrations.
"""

import numpy as np
import pandas as pd
import random
import os

FIRST_NAMES = [
    "Alex", "Jordan", "Taylor", "Morgan", "Sam", "Chris", "Pat", "Avery",
    "Casey", "Riley", "Jamie", "Dakota", "Reese", "Quinn", "Skylar", "Cameron",
    "Priya", "Arjun", "Rohan", "Ananya", "Dev", "Meera", "Vikram", "Siddharth",
    "Fatima", "Zaid", "Amina", "Tariq", "Hassan", "Layla", "Omar", "Noor",
    "Elena", "Mateo", "Sofia", "Lucas", "Isabella", "Diego", "Camila", "Carlos",
    "Liam", "Emma", "Noah", "Olivia", "Ethan", "Ava", "Mason", "Mia",
    "Chen", "Wei", "Mei", "Jun", "Yuki", "Kenji", "Hana", "Ren"
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis", "Wilson",
    "Anderson", "Taylor", "Thomas", "Moore", "Jackson", "Martin", "Lee", "Perez",
    "Sharma", "Patel", "Verma", "Rao", "Reddy", "Gupta", "Nair", "Iyer",
    "Khan", "Ali", "Ahmed", "Malik", "Syed", "Hussain", "Rahman", "Farooq",
    "Garcia", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Perez",
    "Wang", "Li", "Zhang", "Liu", "Chen", "Yang", "Huang", "Wu",
    "Tanaka", "Sato", "Suzuki", "Takahashi", "Watanabe", "Ito", "Yamamoto"
]


def generate_student_data(n_samples: int = 650, seed: int = 42) -> pd.DataFrame:
    """
    Generates a realistic student academic dataset.
    
    Args:
        n_samples: Number of records to generate (default: 650)
        seed: Random seed for reproducibility
        
    Returns:
        pd.DataFrame: Synthetic student records with realistic relationships and quality anomalies.
    """
    np.random.seed(seed)
    random.seed(seed)

    # 1. Generate core features with realistic student distributions
    # Attendance: Beta distribution shifted to [40, 100], peaked around 75-85%
    raw_attendance = np.random.beta(a=5, b=2, size=n_samples)
    attendance = 40 + raw_attendance * 60  # Range: 40% - 100%

    # Previous Exam Score: Bimodal or bell curve around 62 with spread
    raw_prev_exam = np.random.normal(loc=63, scale=16, size=n_samples)
    previous_exam = np.clip(raw_prev_exam, 20, 98)

    # Assignment Average: Correlated with attendance and previous exam
    assignment_noise = np.random.normal(loc=0, scale=8, size=n_samples)
    assignment = (
        0.35 * attendance + 
        0.45 * previous_exam + 
        assignment_noise + 
        15
    )
    assignment = np.clip(assignment, 25, 99)

    # Study Hours Per Week: Gamma distribution (most study 8-20 hrs, few study 30+)
    raw_hours = np.random.gamma(shape=3.5, scale=4.0, size=n_samples)
    study_hours = np.clip(raw_hours, 2.0, 36.0)

    # 2. Realistic Dependent Variable (Final Exam Score)
    # Final score is strongly related to previous exam, assignments, attendance, and study hours
    # Coefficients reflect realistic academic weightage:
    # Final = 0.35*PrevExam + 0.28*Assignment + 0.22*Attendance + 0.50*StudyHours - 5 + noise
    noise = np.random.normal(loc=0, scale=4.5, size=n_samples)
    final_score = (
        0.35 * previous_exam +
        0.28 * assignment +
        0.22 * attendance +
        0.50 * study_hours -
        5.0 +
        noise
    )
    final_score = np.clip(final_score, 12.0, 99.5)

    # 3. Names and IDs
    student_ids = [f"STU{1001 + i}" for i in range(n_samples)]
    names = [
        f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
        for _ in range(n_samples)
    ]

    df = pd.DataFrame({
        "Student_ID": student_ids,
        "Student_Name": names,
        "Attendance_Percentage": np.round(attendance, 1),
        "Assignment_Average": np.round(assignment, 1),
        "Previous_Exam_Score": np.round(previous_exam, 1),
        "Study_Hours_Per_Week": np.round(study_hours, 1),
        "Final_Exam_Score": np.round(final_score, 1)
    })

    # 4. Inject intentional data anomalies for data-cleaning demonstrations
    # A) Missing values (~18 rows)
    missing_indices = np.random.choice(df.index, size=18, replace=False)
    cols_for_na = [
        "Attendance_Percentage", 
        "Assignment_Average", 
        "Previous_Exam_Score", 
        "Study_Hours_Per_Week"
    ]
    for idx in missing_indices:
        col = random.choice(cols_for_na)
        df.loc[idx, col] = np.nan

    # B) Invalid out-of-range values (~5 rows)
    invalid_indices = np.random.choice(df.index, size=5, replace=False)
    df.loc[invalid_indices[0], "Attendance_Percentage"] = 112.5  # Impossible >100%
    df.loc[invalid_indices[1], "Study_Hours_Per_Week"] = -3.5     # Negative study hours
    df.loc[invalid_indices[2], "Assignment_Average"] = 108.0      # Impossible >100
    df.loc[invalid_indices[3], "Previous_Exam_Score"] = 115.0     # Impossible >100
    df.loc[invalid_indices[4], "Attendance_Percentage"] = -5.0    # Impossible negative

    # C) Duplicate records (~10 rows)
    duplicate_indices = np.random.choice(df.index[:200], size=10, replace=False)
    duplicates = df.loc[duplicate_indices].copy()
    df = pd.concat([df, duplicates], ignore_index=True)

    # Shuffle rows to distribute duplicates and anomalies
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return df


def save_dataset(filepath: str = "student_academic_data.csv", n_samples: int = 650) -> str:
    """Generates and writes student data to CSV file."""
    df = generate_student_data(n_samples=n_samples)
    df.to_csv(filepath, index=False)
    print(f"Dataset successfully created at: {os.path.abspath(filepath)}")
    print(f"Total records (including anomalies & duplicates): {len(df)}")
    print(f"Missing values count: {df.isna().sum().sum()}")
    print(f"Duplicate records count: {df.duplicated().sum()}")
    return filepath


if __name__ == "__main__":
    save_dataset()
