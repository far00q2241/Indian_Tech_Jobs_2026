import streamlit as st
import pandas as pd
import joblib

# -------------------------------------------------
# Load Model, Label Encoder and Feature Columns
# -------------------------------------------------
model = joblib.load("work_mode_xgboost_model.pkl")
label_encoder = joblib.load("work_mode_label_encoder.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# -------------------------------------------------
# Page Settings
# -------------------------------------------------
st.set_page_config(page_title="Work Mode Prediction", page_icon="💼")

st.title("💼 Work Mode Prediction")
st.write("Predict whether a job is **On-site, Hybrid, or Remote**.")

st.header("Enter Job Details")

# -------------------------------------------------
# User Inputs
# -------------------------------------------------

experience_raw = st.number_input(
    "Minimum Experience (Years)", 0, 20, 2
)

salary_raw = st.number_input(
    "Minimum Salary (LPA)", 0.0, 50.0, 6.0
)

company_rating = st.number_input(
    "Company Rating",
    min_value=0.0,
    max_value=5.0,
    value=3.5,
    step=0.1
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
    [
        "Small/Startup (<100)",
        "Mid (100-999)",
        "Large (1000+)"
    ]
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

scraped_city = st.selectbox(
    "City",
    [
        "Bangalore",
        "Chennai",
        "Delhi",
        "Gurgaon",
        "Hyderabad",
        "Kolkata",
        "Mumbai",
        "Noida",
        "Pune",
        "Remote"
    ]
)

role_category = st.selectbox(
    "Job Role",
    [
        "Data Analyst",
        "Data Engineer",
        "Data Scientist",
        "Machine Learning Engineer",
        "Python Developer"
    ]
)

skill_domain = st.selectbox(
    "Skill Domain",
    [
        "AI/ML/DL",
        "Business Intelligence",
        "Cloud & DevOps",
        "Data Engineering",
        "Data Science"
    ]
)

# -------------------------------------------------
# Encoding Maps
# -------------------------------------------------

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

# -------------------------------------------------
# Create Input DataFrame (40 Features)
# -------------------------------------------------

input_df = pd.DataFrame(0, index=[0], columns=feature_columns)

# Numeric Features
input_df["company_rating"] = company_rating
input_df["experience_raw"] = experience_raw
input_df["experience_min_yrs"] = experience_raw
input_df["experience_max_yrs"] = experience_raw

input_df["salary_raw"] = salary_raw
input_df["salary_min_lpa"] = salary_raw
input_df["salary_max_lpa"] = salary_raw
input_df["salary_midpoint_lpa"] = salary_raw

input_df["salary_disclosed"] = salary_disclosed
input_df["skills_count"] = skills_count
input_df["company_size_bucket"] = company_size_map[company_size_bucket]
input_df["salary_tier"] = salary_map[salary_tier]
input_df["experience_tier"] = experience_map[experience_tier]
input_df["is_senior"] = is_senior
input_df["days_since_posted"] = days_since_posted
input_df["is_fresher_friendly"] = is_fresher_friendly

# Constant column from training
input_df["salary_negotiable"] = 0

# Frequency-encoded columns (unknown for new jobs, keep 0)
input_df["job_title_freq"] = 0
input_df["company_name_freq"] = 0
input_df["location_freq"] = 0
input_df["primary_city_freq"] = 0

# -------------------------------------------------
# One-Hot Encoded Columns
# -------------------------------------------------

city_col = f"scraped_city_{scraped_city}"
if city_col in input_df.columns:
    input_df[city_col] = 1

role_col = f"role_category_{role_category}"
if role_col in input_df.columns:
    input_df[role_col] = 1

skill_col = f"skill_domain_{skill_domain}"
if skill_col in input_df.columns:
    input_df[skill_col] = 1

# -------------------------------------------------
# Prediction
# -------------------------------------------------

if st.button("Predict Work Mode"):

    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)

    result = label_encoder.inverse_transform(prediction)[0]

    st.success(f"🎯 Predicted Work Mode: **{result}**")

    st.subheader("Prediction Probability")

    prob_df = pd.DataFrame({
        "Work Mode": label_encoder.classes_,
        "Probability (%)": (probability[0] * 100).round(2)
    })

    st.dataframe(prob_df, use_container_width=True)

    chart_data = prob_df.set_index("Work Mode")[["Probability (%)"]]
    st.bar_chart(chart_data)
