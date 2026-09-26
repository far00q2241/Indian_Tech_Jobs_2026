import streamlit as st
import pandas as pd
import joblib

# ----------------------------
# Load Model Files
# ----------------------------
model = joblib.load("work_mode_xgboost_model.pkl")
label_encoder = joblib.load("work_mode_label_encoder.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.set_page_config(page_title="Work Mode Prediction", page_icon="💼")

st.title("💼 Work Mode Prediction")
st.write("Predict whether a job is **On-site, Hybrid, or Remote**.")

# ----------------------------
# User Inputs
# ----------------------------
st.header("Enter Job Details")

experience_raw = st.number_input(
    "Minimum Experience (Years)", 0, 20, 2
)

salary_raw = st.number_input(
    "Minimum Salary (LPA)", 0.0, 50.0, 6.0
)

company_rating = st.slider(
    "Company Rating", 0.0, 5.0, 3.5, 0.1
)

skills_count = st.number_input(
    "Number of Skills Required", 0, 20, 5
)

salary_disclosed = st.selectbox(
    "Salary Disclosed?", ["Yes", "No"]
)

is_fresher_friendly = st.selectbox(
    "Fresher Friendly?", ["Yes", "No"]
)

is_senior = st.selectbox(
    "Senior Level Job?", ["Yes", "No"]
)

days_since_posted = st.number_input(
    "Days Since Posted", 0, 30, 3
)

company_size_bucket = st.selectbox(
    "Company Size",
    ["Small/Startup (<100)", "Mid (100-999)", "Large (1000+)"]
)

experience_tier = st.selectbox(
    "Experience Tier",
    [
        "Fresher",
        "Junior (0-2 Yrs)",
        "Mid (3-5 Yrs)",
        "Senior (6-8 Yrs)",
        "Lead/Architect (9+ Yrs)"
    ]
)

salary_tier = st.selectbox(
    "Salary Tier",
    [
        "Undisclosed",
        "Entry (<3 LPA)",
        "Junior (3-6 LPA)",
        "Mid (6-10 LPA)",
        "Senior (10-20 LPA)",
        "Leadership (20+ LPA)"
    ]
)

skill_domain = st.selectbox(
    "Skill Domain",
    [
        "Business Intelligence",
        "AI/ML/DL",
        "Cloud & DevOps",
        "Data Engineering",
        "Data Science"
    ]
)

# ----------------------------
# Encoding
# ----------------------------

salary_disclosed = 1 if salary_disclosed == "Yes" else 0
is_fresher_friendly = 1 if is_fresher_friendly == "Yes" else 0
is_senior = 1 if is_senior == "Yes" else 0

company_size_map = {
    "Small/Startup (<100)": 0,
    "Mid (100-999)": 1,
    "Large (1000+)": 2
}

experience_map = {
    "Fresher": 0,
    "Junior (0-2 Yrs)": 1,
    "Mid (3-5 Yrs)": 2,
    "Senior (6-8 Yrs)": 3,
    "Lead/Architect (9+ Yrs)": 4
}

salary_map = {
    "Undisclosed": 0,
    "Entry (<3 LPA)": 1,
    "Junior (3-6 LPA)": 2,
    "Mid (6-10 LPA)": 3,
    "Senior (10-20 LPA)": 4,
    "Leadership (20+ LPA)": 5
}

# ----------------------------
# Create Input DataFrame
# ----------------------------

# Create all columns used during training
input_df = pd.DataFrame(0, index=[0], columns=feature_columns)

# Fill numeric features
input_df["experience_raw"] = experience_raw
input_df["salary_raw"] = salary_raw
input_df["company_rating"] = company_rating
input_df["skills_count"] = skills_count
input_df["salary_disclosed"] = salary_disclosed
input_df["is_fresher_friendly"] = is_fresher_friendly
input_df["is_senior"] = is_senior
input_df["days_since_posted"] = days_since_posted
input_df["company_size_bucket"] = company_size_map[company_size_bucket]
input_df["experience_tier"] = experience_map[experience_tier]
input_df["salary_tier"] = salary_map[salary_tier]

# One-hot encode Skill Domain
skill_col = f"skill_domain_{skill_domain}"
if skill_col in input_df.columns:
    input_df[skill_col] = 1

# ----------------------------
# Prediction
# ----------------------------

if st.button("Predict Work Mode"):

    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)

    result = label_encoder.inverse_transform(prediction)[0]

    st.success(f"### Predicted Work Mode: {result}")

    st.subheader("Prediction Probability")

    prob_df = pd.DataFrame({
        "Work Mode": label_encoder.classes_,
        "Probability": probability[0]
    })

    st.dataframe(prob_df, use_container_width=True)
    st.bar_chart(prob_df.set_index("Work Mode"))
