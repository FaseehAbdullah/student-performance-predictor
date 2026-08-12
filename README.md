[![CI](https://github.com/FaseehAbdullah/student-performance-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/FaseehAbdullah/student-performance-predictor/actions/workflows/ci.yml)

# 🎓 Student Performance Predictor

## 🔗 Live Demo

| | Link |
|--|------|
| 💬 Chat UI | [student-performance-predictor.streamlit.app](https://student-performance-predictor-dd8772djbyemqgdbpkqwqh.streamlit.app/) |
| ⚙️ API Docs | [student-performance-predictor.railway.app/docs](https://student-performance-predictor-production-7884.up.railway.app/docs) |

---

## 📌 Project Overview

Most student performance models cheat — they include mid-year grades (G1, G2) which directly reveal the final grade (G3). This project deliberately drops both to build something that works in the real world, where you only have student background and behavioural data.

A conversational Streamlit chat UI collects 8 inputs from the student, sends them to a FastAPI backend, and returns a predicted grade with a SHAP explanation showing exactly which factors drove the prediction.

---

## 🏗️ Architecture

```
User
 │
 ▼
Streamlit Chat UI          (Streamlit Cloud)
 │  8 questions → answers
 │  POST /predict
 ▼
FastAPI Backend            (Railway.app)
 │
 ├── preprocessor.pkl      (StandardScaler + column alignment)
 ├── label_encoder.pkl     (A/B/C/D/F ↔ 0/1/2/3/4)
 └── best_model.pkl        (Tuned Random Forest)
      │
      ├── predict()        → predicted grade
      ├── predict_proba()  → confidence score
      └── SHAP values      → feature contributions
 │
 ▼
Response: grade + confidence + shap_values + advice
 │
 ▼
Streamlit displays result + SHAP bar chart in chat
```

---

## 📊 Model Comparison

All four models were trained and evaluated in Google Colab. The winner was chosen by weighted F1 — more honest than accuracy for imbalanced multi-class problems.

| Model | Accuracy | Weighted F1 | Precision | Recall |
|-------|----------|-------------|-----------|--------|
| Logistic Regression | 0.5000 | 0.4728 | 0.4666 | 0.5000 |
| Random Forest | 0.4923 | 0.4353 | 0.4263 | 0.4923 |
| K-Nearest Neighbors | 0.4462 | 0.4283 | 0.4359 | 0.4462 |
| Decision Tree | 0.3385 | 0.3359 | 0.3352 | 0.3385 |
| **Random Forest (Tuned)** ✅ | **0.5077** | **0.4530** | **0.4567** | **0.5077** |

> Scores are moderate by design — G1 and G2 were deliberately dropped to avoid data leakage. A realistic model predicting final grades from background data alone on 649 students will not score 90%, and that is the point.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| ML & Explainability | Scikit-learn, SHAP |
| Experiment Tracking | MLflow |
| Backend API | FastAPI, Uvicorn |
| Frontend | Streamlit |
| Testing | pytest, httpx |
| CI/CD | GitHub Actions |
| Containerization | Docker, Docker Compose |
| Deployment | Railway.app (backend), Streamlit Cloud (frontend) |

---

## 📁 Project Structure

```
student-performance-predictor/
│
├── data/                        
├── notebooks/
│   └── EDA.ipynb                # Full pipeline — EDA, preprocessing, training
├── src/
│   ├── preprocess.py            
│   └── train.py                 
├── models/                      
│   ├── best_model.pkl
│   ├── preprocessor.pkl
│   └── label_encoder.pkl
├── backend/
│   ├── main.py                  # FastAPI app
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── app.py                   # Streamlit chat UI
│   ├── requirements.txt
│   └── Dockerfile
├── tests/
│   └── test_api.py              # pytest test suite
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions pipeline
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## ⚙️ Run Locally

### Prerequisites
- Python 3.10+
- Git

### Steps

```bash
# Clone the repo
git clone https://github.com/FaseehAbdullah/student-performance-predictor.git
cd student-performance-predictor

# Terminal 1 — Start backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload

# Terminal 2 — Start frontend
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501` for the chat UI.  
Open `http://localhost:8000/docs` for the API docs.

---

## 🐳 Run with Docker

```bash
# Build and start both services
docker-compose up --build

# Run in background
docker-compose up --build -d

# Stop
docker-compose down
```

Open `http://localhost:8501` — the full stack runs inside Docker.

---

## 🔌 API Reference

### `GET /`
Health check.

**Response:**
```json
{
  "status": "ok",
  "message": "Student Performance Predictor API is running."
}
```

### `POST /predict`
Predict a student's final grade.

**Request body:**
```json
{
  "studytime" : 2,
  "absences"  : 4,
  "failures"  : 0,
  "higher"    : 1,
  "internet"  : 1,
  "famrel"    : 4,
  "goout"     : 2,
  "health"    : 3
}
```

**Field reference:**

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| studytime | int | 1–4 | Weekly study hours (1=<2hrs, 4=>10hrs) |
| absences | int | 0–75 | Number of school absences |
| failures | int | 0–3 | Number of past class failures |
| higher | int | 0–1 | Wants higher education (1=yes, 0=no) |
| internet | int | 0–1 | Internet at home (1=yes, 0=no) |
| famrel | int | 1–5 | Family relationship quality |
| goout | int | 1–5 | Going out with friends frequency |
| health | int | 1–5 | Current health status |

**Response:**
```json
{
  "predicted_grade" : "B",
  "confidence"      : 0.81,
  "shap_values"     : {
    "failures"  : -0.03,
    "studytime" :  0.02,
    "higher"    :  0.01,
    "absences"  : -0.01,
    "famrel"    :  0.005
  },
  "advice": "Good performance! Focus on reducing absences."
}
```

---

## 📈 MLflow Experiments

5 runs were tracked during training in Google Colab:

| Run | Model | Weighted F1 |
|-----|-------|-------------|
| 1 | logistic_regression | 0.4728 |
| 2 | decision_tree | 0.3359 |
| 3 | random_forest | 0.4353 |
| 4 | knn | 0.4283 |
| 5 | random_forest_tuned ✅ | 0.4530 |

---

## 🧪 Tests

10 tests covering health check, prediction validity, response format, confidence range, SHAP output, advice format, missing fields, wrong types, and multiple input scenarios.

```bash
pytest tests/ -v
```

```
test_health                   PASSED
test_predict_valid            PASSED
test_predict_output_format    PASSED
test_predict_grade_is_valid   PASSED
test_predict_confidence_range PASSED
test_predict_shap_is_dict     PASSED
test_predict_advice_is_string PASSED
test_predict_missing_field    PASSED
test_predict_wrong_type       PASSED
test_predict_multiple_inputs  PASSED
```

---

## 🔄 CI/CD

Every push to `main` triggers the GitHub Actions pipeline automatically:

```
Push to main → Checkout → Python 3.10 → Install deps → Create dummy artifacts → Run pytest → Pass ✅ or Fail ❌
```

---

## 📉 Limitations

- **Small dataset** — 649 students from two Portuguese schools. Results may not generalise to other schools or countries.
- **No G1/G2** — deliberately excluded to avoid data leakage. This makes prediction harder but realistic.
- **Default values** — the API collects 8 features. Remaining features use dataset averages as defaults.
- **Railway free tier** — deployment credits are limited. The live demo may go offline after the free allowance is used.
- **Static model** — the model does not retrain on new data. A changing student population would need periodic retraining.

---

## 👤 Author

**Faseeh Abdullah**  
[GitHub](https://github.com/FaseehAbdullah)