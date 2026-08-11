import sys
import os

# Add backend directory to path so we can import main.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

# ── Sample valid input ────────────────────────────────────────
VALID_INPUT = {
    "studytime" : 2,
    "absences"  : 4,
    "failures"  : 0,
    "higher"    : 1,
    "internet"  : 1,
    "famrel"    : 4,
    "goout"     : 2,
    "health"    : 3
}

# ── Test 1 — Health check ─────────────────────────────────────
def test_health():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

# ── Test 2 — Valid prediction returns 200 ─────────────────────
def test_predict_valid():
    response = client.post("/predict", json=VALID_INPUT)
    assert response.status_code == 200

# ── Test 3 — Response has all required fields ─────────────────
def test_predict_output_format():
    response = client.post("/predict", json=VALID_INPUT)
    data = response.json()

    assert "predicted_grade" in data
    assert "confidence"      in data
    assert "shap_values"     in data
    assert "advice"          in data

# ── Test 4 — Predicted grade is a valid letter ────────────────
def test_predict_grade_is_valid():
    response = client.post("/predict", json=VALID_INPUT)
    data = response.json()

    assert data["predicted_grade"] in ["A", "B", "C", "D", "F"]

# ── Test 5 — Confidence is between 0 and 1 ───────────────────
def test_predict_confidence_range():
    response = client.post("/predict", json=VALID_INPUT)
    data = response.json()

    assert 0.0 <= data["confidence"] <= 1.0

# ── Test 6 — SHAP values is a non-empty dict ─────────────────
def test_predict_shap_is_dict():
    response = client.post("/predict", json=VALID_INPUT)
    data = response.json()

    assert isinstance(data["shap_values"], dict)
    assert len(data["shap_values"]) > 0

# ── Test 7 — Advice is a non-empty string ────────────────────
def test_predict_advice_is_string():
    response = client.post("/predict", json=VALID_INPUT)
    data = response.json()

    assert isinstance(data["advice"], str)
    assert len(data["advice"]) > 0

# ── Test 8 — Missing field returns 422 ───────────────────────
def test_predict_missing_field():
    incomplete = {
        "studytime" : 2,
        "absences"  : 4
        # rest of fields missing
    }
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422

# ── Test 9 — Wrong data type returns 422 ─────────────────────
def test_predict_wrong_type():
    bad_input = VALID_INPUT.copy()
    bad_input["studytime"] = "a lot"   # string instead of int
    response = client.post("/predict", json=bad_input)
    assert response.status_code == 422

# ── Test 10 — Different valid inputs all return valid grades ──
def test_predict_multiple_inputs():
    test_cases = [
        {"studytime": 4, "absences": 0,  "failures": 0, "higher": 1, "internet": 1, "famrel": 5, "goout": 1, "health": 5},
        {"studytime": 1, "absences": 20, "failures": 3, "higher": 0, "internet": 0, "famrel": 1, "goout": 5, "health": 1},
        {"studytime": 2, "absences": 8,  "failures": 1, "higher": 1, "internet": 1, "famrel": 3, "goout": 3, "health": 3},
    ]
    for case in test_cases:
        response = client.post("/predict", json=case)
        assert response.status_code == 200
        assert response.json()["predicted_grade"] in ["A", "B", "C", "D", "F"]