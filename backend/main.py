from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import shap
import os

# ── App ───────────────────────────────────────────────────────
app = FastAPI(
    title="Student Performance Predictor",
    description="Predicts student final grade (A/B/C/D/F) from behavioural and demographic features.",
    version="1.0.0"
)

# ── Load artifacts ────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

preprocessor  = joblib.load(os.path.join(MODEL_DIR, "preprocessor.pkl"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
model         = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))

scaler       = preprocessor["scaler"]
num_cols     = preprocessor["num_cols"]
feature_cols = preprocessor["feature_cols"]

# Initialise SHAP explainer once at startup — not on every request
explainer = shap.TreeExplainer(model)

# ── Input schema ──────────────────────────────────────────────
class StudentInput(BaseModel):
    studytime : int   # 1–4
    absences  : int   # 0–75
    failures  : int   # 0–3
    higher    : int   # 1=yes, 0=no
    internet  : int   # 1=yes, 0=no
    famrel    : int   # 1–5
    goout     : int   # 1–5
    health    : int   # 1–5

# ── Advice logic ──────────────────────────────────────────────
def get_advice(grade: str) -> str:
    advice = {
        "A": "Excellent performance! Keep up the great work.",
        "B": "Good performance! Focus on reducing absences.",
        "C": "Average performance. Try increasing your study time.",
        "D": "Below average. Consider seeking extra support.",
        "F": "At risk of failing. Please seek help immediately."
    }
    return advice.get(grade, "No advice available.")

# ── Preprocessing ─────────────────────────────────────────────
def preprocess_input(data: StudentInput) -> pd.DataFrame:
    # Build a raw row matching the columns the model was trained on
    # All columns not asked in the chat get sensible defaults
    raw = {
        # Asked in chat
        "studytime" : data.studytime,
        "absences"  : data.absences,
        "failures"  : data.failures,
        "higher"    : data.higher,
        "internet"  : data.internet,
        "famrel"    : data.famrel,
        "goout"     : data.goout,
        "health"    : data.health,

        # Defaults for columns not collected in chat
        "age"       : 17,
        "Medu"      : 2,
        "Fedu"      : 2,
        "traveltime": 1,
        "freetime"  : 3,
        "Dalc"      : 1,
        "Walc"      : 1,
        "sex"       : 1,       # M
        "address"   : 1,       # Urban
        "famsize"   : 1,       # GT3
        "Pstatus"   : 1,       # Together
        "schoolsup" : 0,
        "famsup"    : 1,
        "paid"      : 0,
        "activities": 0,
        "nursery"   : 1,
        "romantic"  : 0,

        # One-hot encoded columns — all set to 0 (drop_first baseline)
        "school_MS"         : 0,
        "Mjob_health"       : 0,
        "Mjob_other"        : 0,
        "Mjob_services"     : 0,
        "Mjob_teacher"      : 0,
        "Fjob_health"       : 0,
        "Fjob_other"        : 0,
        "Fjob_services"     : 0,
        "Fjob_teacher"      : 0,
        "reason_home"       : 0,
        "reason_other"      : 0,
        "reason_reputation" : 0,
        "guardian_mother"   : 0,
        "guardian_other"    : 0,
    }

    df = pd.DataFrame([raw])

    # Reindex to match exact training columns — fills any missing with 0
    df = df.reindex(columns=feature_cols, fill_value=0)

    # Scale numerical columns
    df[num_cols] = scaler.transform(df[num_cols])

    return df

# ── Routes ────────────────────────────────────────────────────
@app.get("/")
def health_check():
    return {"status": "ok", "message": "Student Performance Predictor API is running."}


@app.post("/predict")
def predict(data: StudentInput):
    try:
        # Preprocess
        X = preprocess_input(data)

        # Predict
        pred_encoded = model.predict(X)[0]
        pred_grade   = label_encoder.inverse_transform([pred_encoded])[0]
        pred_proba   = model.predict_proba(X)[0]
        confidence   = round(float(pred_proba[pred_encoded]), 4)

        # SHAP — shape (n_samples, n_features, n_classes)
        shap_values      = explainer.shap_values(X)
        shap_for_class   = shap_values[0, :, pred_encoded]  # shape: (n_features,)
        feature_names    = list(X.columns)

        # Top 5 features by absolute SHAP value
        shap_series = pd.Series(shap_for_class, index=feature_names)
        top_shap    = (
            shap_series
            .reindex(["studytime", "absences", "failures", "higher", "internet",
                      "famrel", "goout", "health"])
            .sort_values(key=abs, ascending=False)
            .head(5)
            .round(4)
            .to_dict()
        )

        return {
            "predicted_grade" : pred_grade,
            "confidence"      : confidence,
            "shap_values"     : top_shap,
            "advice"          : get_advice(pred_grade)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))