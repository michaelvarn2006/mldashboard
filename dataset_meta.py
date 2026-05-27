from __future__ import annotations

TARGET_COL = "Target"
TOP_K = 15
RANDOM_STATE = 42

RAW_DATA_CANDIDATES = [
    "data/data_clf.csv",
    "data/data.csv",
]

MODEL_DISPLAY_NAMES = {
    "LogisticRegression": "Логистическая регрессия",
    "AdaBoostClassifier": "AdaBoost",
    "BaggingClassifier": "Bagging",
    "StackingClassifier": "Stacking",
    "MLPClassifier": "MLP",
    "XGBClassifier": "XGBoost",
    "GradientBoostingClassifier": "Gradient Boosting",
}

FIELD_LABELS = {
    "Marital Status": "Marital status",
    "Application mode": "Application mode",
    "Application order": "Application order",
    "Course": "Course",
    "Daytime/evening attendance": "Attendance",
    "Previous qualification": "Previous qualification",
    "Previous qualification (grade)": "Previous qualification grade",
    "Nacionality": "Nationality",
    "Mother's qualification": "Mother's qualification",
    "Father's qualification": "Father's qualification",
    "Mother's occupation": "Mother's occupation",
    "Father's occupation": "Father's occupation",
    "Admission grade": "Admission grade",
    "Displaced": "Displaced",
    "Educational special needs": "Special educational needs",
    "Debtor": "Debtor",
    "Tuition fees up to date": "Tuition fees up to date",
    "Gender": "Gender",
    "Scholarship holder": "Scholarship holder",
    "Age at enrollment": "Age at enrollment",
    "International": "International",
    "Curricular units 1st sem (credited)": "1st sem credited units",
    "Curricular units 1st sem (enrolled)": "1st sem enrolled units",
    "Curricular units 1st sem (evaluations)": "1st sem evaluations",
    "Curricular units 1st sem (approved)": "1st sem approved units",
    "Curricular units 1st sem (grade)": "1st sem grade",
    "Curricular units 1st sem (without evaluations)": "1st sem no evaluations",
    "Curricular units 2nd sem (credited)": "2nd sem credited units",
    "Curricular units 2nd sem (enrolled)": "2nd sem enrolled units",
    "Curricular units 2nd sem (evaluations)": "2nd sem evaluations",
    "Curricular units 2nd sem (approved)": "2nd sem approved units",
    "Curricular units 2nd sem (grade)": "2nd sem grade",
    "Curricular units 2nd sem (without evaluations)": "2nd sem no evaluations",
    "Unemployment rate": "Unemployment rate",
    "Inflation rate": "Inflation rate",
    "GDP": "GDP",
}

FIELD_UNITS = {
    "Age at enrollment": "years",
    "Application order": "rank",
    "Previous qualification (grade)": "points",
    "Admission grade": "points",
    "Curricular units 1st sem (credited)": "units",
    "Curricular units 1st sem (enrolled)": "units",
    "Curricular units 1st sem (evaluations)": "count",
    "Curricular units 1st sem (approved)": "units",
    "Curricular units 1st sem (grade)": "points",
    "Curricular units 1st sem (without evaluations)": "count",
    "Curricular units 2nd sem (credited)": "units",
    "Curricular units 2nd sem (enrolled)": "units",
    "Curricular units 2nd sem (evaluations)": "count",
    "Curricular units 2nd sem (approved)": "units",
    "Curricular units 2nd sem (grade)": "points",
    "Curricular units 2nd sem (without evaluations)": "count",
    "Unemployment rate": "%",
    "Inflation rate": "%",
    "GDP": "index",
}

YES_NO_LABELS = {0: "No", 1: "Yes"}

FIELD_VALUE_LABELS = {
    "Marital Status": {
        1: "Single",
        2: "Married",
        3: "Widower",
        4: "Divorced",
        5: "Facto union",
        6: "Legally separated",
    },
    "Daytime/evening attendance": {
        0: "Evening",
        1: "Daytime",
    },
    "Displaced": YES_NO_LABELS,
    "Educational special needs": YES_NO_LABELS,
    "Debtor": YES_NO_LABELS,
    "Tuition fees up to date": YES_NO_LABELS,
    "Gender": {
        0: "Female",
        1: "Male",
    },
    "Scholarship holder": YES_NO_LABELS,
    "International": YES_NO_LABELS,
}

BINARY_VARS = {
    "Daytime/evening attendance",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
}

CATEGORICAL_VARS = {
    "Marital Status",
    "Application mode",
    "Course",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
} | BINARY_VARS

NUMERIC_VARS = {
    "Application order",
    "Previous qualification (grade)",
    "Admission grade",
    "Age at enrollment",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
}