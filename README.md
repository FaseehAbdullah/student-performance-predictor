[![CI](https://github.com/FaseehAbdullah/student-performance-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/FaseehAbdullah/student-performance-predictor/actions/workflows/ci.yml)

# 🎓 Student Performance Predictor

## 🔗 Live Demo

| | Link |
|--|------|
| 💬 Chat UI | [student-predictor.streamlit.app](https://student-performance-predictor-dd8772djbyemqgdbpkqwqh.streamlit.app/) |
| ⚙️ API Docs | [your-api.up.railway.app/docs](https://student-performance-predictor-production-7884.up.railway.app/docs) |

## 📊 Model Comparison

All four models were tracked with MLflow. Winner chosen by weighted F1 — more honest than accuracy for imbalanced multi-class problems.

| Model | Accuracy | Weighted F1 | Precision | Recall |
|-------|----------|-------------|-----------|--------|
| Logistic Regression | 0.5000 | 0.4728 | 0.4666 | 0.5000 |
| Random Forest | 0.4923 | 0.4353 | 0.4263 | 0.4923 |
| K-Nearest Neighbors | 0.4462 | 0.4283 | 0.4359 | 0.4462 |
| Decision Tree | 0.3385 | 0.3359 | 0.3352 | 0.3385 |
| **Random Forest (Tuned)** ✅ | **0.5077** | **0.4530** | **0.4567** | **0.5077** |