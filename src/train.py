from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("data/processed/credit_default.csv")
MODEL_PATH = Path("models/loan_default_model.joblib")
TARGET = "default payment next month"

# Load data.
df = pd.read_csv(DATA_PATH)

# ID is only a row identifier.
X = df.drop(columns=[TARGET, "ID"])
y = df[TARGET]

# Keep the test set untouched until final evaluation.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Validation data is used for model and threshold selection.
X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train,
)

models = {
    "Logistic Regression": Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
}

best_model = None
best_name = ""
best_auc = 0.0

# Compare models using validation ROC-AUC.
for name, model in models.items():
    model.fit(X_train, y_train)

    val_probabilities = model.predict_proba(X_val)[:, 1]
    val_predictions = model.predict(X_val)
    val_auc = roc_auc_score(y_val, val_probabilities)

    print(f"\n{'=' * 50}")
    print(name)
    print("=" * 50)

    print("\nValidation confusion matrix:")
    print(confusion_matrix(y_val, val_predictions))

    print("\nValidation classification report:")
    print(classification_report(y_val, val_predictions))

    print("Validation ROC-AUC:", round(val_auc, 4))

    if val_auc > best_auc:
        best_auc = val_auc
        best_model = model
        best_name = name

# Find the threshold that gives the best validation F1-score.
val_probabilities = best_model.predict_proba(X_val)[:, 1]

precision, recall, thresholds = precision_recall_curve(
    y_val,
    val_probabilities,
)

f1_scores = (
    2 * precision[:-1] * recall[:-1]
    / (precision[:-1] + recall[:-1] + 1e-10)
)

best_index = f1_scores.argmax()
best_threshold = float(thresholds[best_index])

print(f"\nBest model: {best_name}")
print(f"Best validation ROC-AUC: {best_auc:.4f}")
print(f"Selected threshold: {best_threshold:.3f}")
print(f"Validation F1-score: {f1_scores[best_index]:.3f}")

# Final evaluation on the untouched test set.
test_probabilities = best_model.predict_proba(X_test)[:, 1]
test_predictions = (test_probabilities >= best_threshold).astype(int)

print(f"\n{'=' * 50}")
print("Final test results")
print("=" * 50)

print("\nTest confusion matrix:")
print(confusion_matrix(y_test, test_predictions))

print("\nTest classification report:")
print(classification_report(y_test, test_predictions))

print(
    "Test ROC-AUC:",
    round(roc_auc_score(y_test, test_probabilities), 4),
)

# Save both the model and its selected threshold.
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

model_data = {
    "model": best_model,
    "threshold": best_threshold,
    "feature_names": X.columns.tolist(),
}

joblib.dump(model_data, MODEL_PATH)

print(f"\nModel and threshold saved to: {MODEL_PATH}")