import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Load Model and Label Encoder
# -------------------------------
model = joblib.load("work_mode_xgboost_model.pkl")
label_encoder = joblib.load("work_mode_label_encoder.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.set_page_config(page_title="Work Mode Prediction", page_icon="💼")

st.title("💼 Work Mode Prediction")
st.write("Predict whether a job is **On-site, Hybrid, or Remote**.")

st.header("Enter Job Details")

# -------------------------------
# User Inputs
# -------------------------------

experience_raw = st.number_input(
    "Minimum Experience (Years)", min_value=0, max_value=20, value=2
)

salary_raw = st.number_input(
    "Minimum Salary (LPA)", min_value=0.0, max_value=50.0, value=6.0
)

company_rating = st.slider(
    "Company Rating", 0.0, 5.0, 3.5, step=0.1
)

skills_count = st.number_input(
    "Number of Skills Required", min_value=0, max_value=20, value=5
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
    "Days Since Posted", min_value=0, max_value=30, value=3
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

# -------------------------------
# Convert Categorical Inputs
# -------------------------------

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

# -------------------------------
# Create Input DataFrame
# -------------------------------

input_df = pd.DataFrame([{
    "experience_raw": experience_raw,
    "salary_raw": salary_raw,
    "company_rating": company_rating,
    "skills_count": skills_count,
    "salary_disclosed": salary_disclosed,
    "is_fresher_friendly": is_fresher_friendly,
    "is_senior": is_senior,
    "days_since_posted": days_since_posted,
    "company_size_bucket": company_size_map[company_size_bucket],
    "experience_tier": experience_map[experience_tier],
    "salary_tier": salary_map[salary_tier]
}])

# -------------------------------
# Prediction
# -------------------------------

if st.button("Predict Work Mode"):

    prediction = model.predict(input_df)
    prediction_prob = model.predict_proba(input_df)

    work_mode = label_encoder.inverse_transform(prediction)[0]

    st.success(f"Predicted Work Mode: **{work_mode}**")

    st.subheader("Prediction Probability")

    probability_df = pd.DataFrame({
        "Work Mode": label_encoder.classes_,
        "Probability": prediction_prob[0]
    })

    st.dataframe(probability_df)
    st.bar_chart(probability_df.set_index("Work Mode"))
