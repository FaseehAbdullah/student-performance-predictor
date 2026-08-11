import joblib
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

os.makedirs("models", exist_ok=True)

num_cols = [
    "age", "Medu", "Fedu", "traveltime", "studytime",
    "failures", "famrel", "freetime", "goout",
    "Dalc", "Walc", "health", "absences"
]
binary_cols = [
    "sex", "address", "famsize", "Pstatus",
    "schoolsup", "famsup", "paid", "activities",
    "nursery", "higher", "internet", "romantic"
]
ohe_cols = [
    "school_MS",
    "Mjob_health", "Mjob_other", "Mjob_services", "Mjob_teacher",
    "Fjob_health", "Fjob_other", "Fjob_services", "Fjob_teacher",
    "reason_home", "reason_other", "reason_reputation",
    "guardian_mother", "guardian_other"
]

feature_cols = num_cols + binary_cols + ohe_cols
n_features   = len(feature_cols)
print(f"Total features: {n_features}")

# Label encoder — fitted on grade strings
le = LabelEncoder()
le.fit(["A", "B", "C", "D", "F"])
joblib.dump(le, "models/label_encoder.pkl")
print(f"Label encoder classes: {le.classes_}")

# Training data — y must be encoded integers, exactly like real training
X_dummy   = np.random.rand(25, n_features)
y_strings = ["A", "B", "C", "D", "F"] * 5
y_encoded = le.transform(y_strings)

clf = RandomForestClassifier(n_estimators=5, random_state=42)
clf.fit(X_dummy, y_encoded)
joblib.dump(clf, "models/best_model.pkl")
print(f"Model classes: {clf.classes_}")

# Preprocessor
scaler = StandardScaler()
scaler.fit(X_dummy[:, :len(num_cols)])

preprocessor = {
    "scaler"       : scaler,
    "num_cols"     : num_cols,
    "feature_cols" : feature_cols
}
joblib.dump(preprocessor, "models/preprocessor.pkl")

print("All artifacts created successfully.")