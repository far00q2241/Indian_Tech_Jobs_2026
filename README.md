# 💼 Work Mode Prediction

A simple Machine Learning project that predicts whether a job is **On-site**, **Hybrid**, or **Remote** using an XGBoost model.

## Project Overview

This project predicts the work mode of a job using features such as experience, salary, company rating, company size, posting date, and skills count.

**Target Variable:** `work_mode`

* On-site
* Hybrid
* Remote

## Model Used

* **XGBoost Classifier** (Final Model)

## Model Accuracy

* **Accuracy:** 95.88%

## Files in This Project

* `app.py` – Streamlit application.
* `work_mode_xgboost_model.pkl` – Saved XGBoost model.
* `work_mode_label_encoder.pkl` – Saved Label Encoder.
* `requirements.txt` – Required Python libraries.

## Run the Project

Install the libraries:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

## Author

**Farooq Khan**

GitHub: https://github.com/far00q2241
