# export_model.py
# Run this script ONCE after training to persist the model and threshold.
# Place this file alongside your notebook, or copy-paste the last cell into the notebook.

import joblib

# --- Option A: Single model (XGBoost pipeline) ---
# Assumes `final_model` and `best_threshold` are already in scope (run from notebook).

# joblib.dump(final_model, "attrition_model.joblib")
# joblib.dump(best_threshold, "threshold.joblib")
# print("Saved: attrition_model.joblib, threshold.joblib")


# --- Option B: Ensemble (LR + XGBoost) ---
# Uncomment below if you're using the ensemble version instead.

# joblib.dump(final_lr_ensemble,  "lr_model.joblib")
# joblib.dump(final_xgb_ensemble, "xgb_model.joblib")
# joblib.dump(
#     {
#         "lr_weight":  best_lr_weight,
#         "xgb_weight": best_xgb_weight,
#         "threshold":  best_ensemble_threshold,
#     },
#     "ensemble_config.joblib",
# )
# print("Saved: lr_model.joblib, xgb_model.joblib, ensemble_config.joblib")


# ---------------------------------------------------------------------------
# Standalone usage: load an already-saved model and verify it works.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import pandas as pd
    from feature_engineering import add_hr_features

    model = joblib.load("attrition_model.joblib")
    threshold = joblib.load("threshold.joblib")

    # Quick smoke-test with one dummy row
    sample = {
        "Age": 35,
        "BusinessTravel": "Travel_Rarely",
        "DailyRate": 800,
        "Department": "Research & Development",
        "DistanceFromHome": 5,
        "Education": 3,
        "EducationField": "Life Sciences",
        "EnvironmentSatisfaction": 3,
        "Gender": "Male",
        "HourlyRate": 65,
        "JobInvolvement": 3,
        "JobLevel": 2,
        "JobRole": "Research Scientist",
        "JobSatisfaction": 3,
        "MaritalStatus": "Single",
        "MonthlyIncome": 5000,
        "MonthlyRate": 14000,
        "NumCompaniesWorked": 2,
        "OverTime": "No",
        "PercentSalaryHike": 13,
        "PerformanceRating": 3,
        "RelationshipSatisfaction": 3,
        "StockOptionLevel": 1,
        "TotalWorkingYears": 10,
        "TrainingTimesLastYear": 3,
        "WorkLifeBalance": 3,
        "YearsAtCompany": 5,
        "YearsInCurrentRole": 3,
        "YearsSinceLastPromotion": 1,
        "YearsWithCurrManager": 3,
    }

    df = pd.DataFrame([sample])
    df = add_hr_features(df)
    prob = model.predict_proba(df)[0, 1]
    pred = int(prob >= threshold)
    print(f"Smoke test → probability={prob:.4f}, prediction={'Yes' if pred else 'No'}, threshold={threshold}")
