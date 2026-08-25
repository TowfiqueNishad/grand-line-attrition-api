# main.py — FastAPI HR Attrition Prediction Service
# Aligned with Group09_HR_attrition_FINAL_Submission_File.ipynb
# 11-model imbalance-aware competition pipeline

import os
import joblib
import pandas as pd
from typing import Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from feature_engineering import add_hr_features

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="HR Attrition Predictor",
    description=(
        "Predicts employee attrition probability using a rigorous 11-model "
        "imbalance-aware competition pipeline (notebook: Group09_HR_attrition_FINAL_Submission_File). "
        "The final model is selected via nested outer-CV on Left F1 → PR-AUC → Balanced Accuracy → Recall → Precision. "
        "Submit raw IBM HR dataset fields and receive a probability, binary prediction, risk tier, "
        "and the selected model name."
    ),
    version="2.0.0",
)

# ---------------------------------------------------------------------------
# CORS — allow frontend (Vercel) to call this API
# Set CORS_ORIGINS env var to a comma-separated list of allowed origins.
# Defaults to allow all Vercel domains + localhost.
# ---------------------------------------------------------------------------
_raw_origins = os.getenv("CORS_ORIGINS", "")
allowed_origins = (
    [o.strip() for o in _raw_origins.split(",") if o.strip()]
    if _raw_origins
    else ["http://localhost:5173", "http://localhost:3000"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    missing_fields = []
    invalid_fields = []
    for err in errors:
        loc = err.get("loc", [])
        field_name = loc[-1] if loc else "Field"
        if err.get("type") == "missing":
            missing_fields.append(str(field_name))
        else:
            invalid_fields.append(str(field_name))

    msgs = []
    if missing_fields:
        msgs.append(f"Missing required fields: {', '.join(missing_fields)}")
    if invalid_fields:
        msgs.append(f"Invalid or empty values in fields: {', '.join(invalid_fields)}")

    return JSONResponse(
        status_code=422,
        content={"detail": "Please review your input: " + ". ".join(msgs)}
    )


# ---------------------------------------------------------------------------
# Lazy model loading — loaded on first request, not at import time.
# Also loads optional metadata saved alongside the model.
# ---------------------------------------------------------------------------
_model = None
_threshold: float | None = None
_model_meta: dict = {}

MODEL_PATH     = os.getenv("MODEL_PATH",     "attrition_model.joblib")
THRESHOLD_PATH = os.getenv("THRESHOLD_PATH", "threshold.joblib")
META_PATH      = os.getenv("META_PATH",      "model_meta.joblib")


def get_model() -> tuple[Any, float]:
    global _model, _threshold, _model_meta
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Model file not found: '{MODEL_PATH}'. "
                    "Re-run train_model.py to regenerate it:\n"
                    "  python train_model.py"
                ),
            )
        if not os.path.exists(THRESHOLD_PATH):
            raise HTTPException(
                status_code=503,
                detail=f"Threshold file not found: '{THRESHOLD_PATH}'.",
            )
        _model     = joblib.load(MODEL_PATH)
        _threshold = float(joblib.load(THRESHOLD_PATH))
        # Load optional metadata (model name, strategy, test metrics)
        if os.path.exists(META_PATH):
            _model_meta = joblib.load(META_PATH)
    assert _threshold is not None
    return _model, _threshold


# ---------------------------------------------------------------------------
# Request schema — mirrors IBM HR dataset columns used in training
# (constant/ID-like columns dropped: EmployeeCount, EmployeeNumber, Over18, StandardHours)
# ---------------------------------------------------------------------------
class Employee(BaseModel):
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
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
            ]
        }
    }

    Age:                      int = Field(..., ge=18, le=65)
    BusinessTravel:           str
    DailyRate:                int = Field(..., ge=0)
    Department:               str
    DistanceFromHome:         int = Field(..., ge=0)
    Education:                int = Field(..., ge=1, le=5)
    EducationField:           str
    EnvironmentSatisfaction:  int = Field(..., ge=1, le=4)
    Gender:                   str
    HourlyRate:               int = Field(..., ge=0)
    JobInvolvement:           int = Field(..., ge=1, le=4)
    JobLevel:                 int = Field(..., ge=1, le=5)
    JobRole:                  str
    JobSatisfaction:          int = Field(..., ge=1, le=4)
    MaritalStatus:            str
    MonthlyIncome:            int = Field(..., ge=0)
    MonthlyRate:              int = Field(..., ge=0)
    NumCompaniesWorked:       int = Field(..., ge=0)
    OverTime:                 str
    PercentSalaryHike:        int = Field(..., ge=0)
    PerformanceRating:        int = Field(..., ge=1, le=4)
    RelationshipSatisfaction: int = Field(..., ge=1, le=4)
    StockOptionLevel:         int = Field(..., ge=0, le=3)
    TotalWorkingYears:        int = Field(..., ge=0)
    TrainingTimesLastYear:    int = Field(..., ge=0)
    WorkLifeBalance:          int = Field(..., ge=1, le=4)
    YearsAtCompany:           int = Field(..., ge=0)
    YearsInCurrentRole:       int = Field(..., ge=0)
    YearsSinceLastPromotion:  int = Field(..., ge=0)
    YearsWithCurrManager:     int = Field(..., ge=0)


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------
class PredictionResponse(BaseModel):
    attrition_probability: float
    prediction: str          # "Yes" | "No"
    risk_level: str          # "High" | "Medium" | "Low"
    threshold_used: float
    model_name: str | None = None    # e.g. "Random Forest", "XGBoost"
    model_strategy: str | None = None  # e.g. "ClassWeight balanced"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", tags=["Health"])
def health_check():
    """Liveness probe — returns 200 OK when the service is ready."""
    model_ready = os.path.exists(MODEL_PATH) and os.path.exists(THRESHOLD_PATH)
    meta = {}
    if os.path.exists(META_PATH):
        try:
            meta = joblib.load(META_PATH)
        except Exception:
            pass
    return {
        "status": "ok",
        "model_ready": model_ready,
        "model_path": MODEL_PATH,
        "threshold_path": THRESHOLD_PATH,
        "model_name": meta.get("model_name"),
        "model_strategy": meta.get("model_strategy"),
        "version": "2.0.0",
    }


@app.get("/model-info", tags=["Health"])
def model_info():
    """Return metadata about the loaded model (name, strategy, feature count, threshold, test metrics)."""
    model, threshold = get_model()

    # Determine the model name from the pipeline's final estimator
    if hasattr(model, "steps"):
        final_step_name, final_estimator = model.steps[-1]
        estimator_class = type(final_estimator).__name__
    else:
        estimator_class = type(model).__name__

    # Build a sample to count engineered features
    _example: dict = {
        "Age": 35, "BusinessTravel": "Travel_Rarely", "DailyRate": 800,
        "Department": "Research & Development", "DistanceFromHome": 5,
        "Education": 3, "EducationField": "Life Sciences",
        "EnvironmentSatisfaction": 3, "Gender": "Male", "HourlyRate": 65,
        "JobInvolvement": 3, "JobLevel": 2, "JobRole": "Research Scientist",
        "JobSatisfaction": 3, "MaritalStatus": "Single", "MonthlyIncome": 5000,
        "MonthlyRate": 14000, "NumCompaniesWorked": 2, "OverTime": "No",
        "PercentSalaryHike": 13, "PerformanceRating": 3,
        "RelationshipSatisfaction": 3, "StockOptionLevel": 1,
        "TotalWorkingYears": 10, "TrainingTimesLastYear": 3,
        "WorkLifeBalance": 3, "YearsAtCompany": 5, "YearsInCurrentRole": 3,
        "YearsSinceLastPromotion": 1, "YearsWithCurrManager": 3,
    }
    sample = pd.DataFrame([_example])
    engineered = add_hr_features(sample)
    feature_count = engineered.shape[1]

    response = {
        "estimator_class": estimator_class,
        "feature_count": feature_count,
        "threshold": round(threshold, 4),
    }

    # Merge in saved metadata (model_name, strategy, test metrics) if available
    response.update({
        k: v for k, v in _model_meta.items()
        if k not in response
    })

    return response


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(employee: Employee):
    """
    Predict attrition for a single employee.

    - Converts raw fields to a DataFrame.
    - Applies add_hr_features() — identical to training pipeline.
    - Returns probability, binary label, risk tier, and winning model info.
    """
    model, threshold = get_model()
    try:
        raw_df = pd.DataFrame([employee.model_dump()])
        engineered_df = add_hr_features(raw_df)
        probability = float(model.predict_proba(engineered_df)[0, 1])
        prediction = int(probability >= threshold)

        # Risk tiers: High = clearly above threshold and high absolute prob
        if probability >= 0.65:
            risk_level = "High"
        elif probability >= threshold:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        return PredictionResponse(
            attrition_probability=round(probability, 4),
            prediction="Yes" if prediction == 1 else "No",
            risk_level=risk_level,
            threshold_used=round(threshold, 4),
            model_name=_model_meta.get("model_name"),
            model_strategy=_model_meta.get("model_strategy"),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(employees: list[Employee]):
    """
    Predict attrition for a list of employees in one call.
    Returns a list of prediction objects in the same order as the input.
    Max batch size: 500.
    """
    if not employees:
        raise HTTPException(status_code=400, detail="Employee list must not be empty.")
    if len(employees) > 500:
        raise HTTPException(status_code=400, detail="Batch size limit is 500 employees.")

    model, threshold = get_model()
    try:
        raw_df = pd.DataFrame([e.model_dump() for e in employees])
        engineered_df = add_hr_features(raw_df)
        probabilities = model.predict_proba(engineered_df)[:, 1]

        results = []
        for prob in probabilities:
            prob = float(prob)
            pred = int(prob >= threshold)
            if prob >= 0.65:
                risk = "High"
            elif prob >= threshold:
                risk = "Medium"
            else:
                risk = "Low"
            results.append(
                {
                    "attrition_probability": round(prob, 4),
                    "prediction": "Yes" if pred == 1 else "No",
                    "risk_level": risk,
                    "threshold_used": round(threshold, 4),
                    "model_name": _model_meta.get("model_name"),
                    "model_strategy": _model_meta.get("model_strategy"),
                }
            )
        return results

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
