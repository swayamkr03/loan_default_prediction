from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path("models/loan_default_model.joblib")


def load_model():
    saved_data = joblib.load(MODEL_PATH)

    return (
        saved_data["model"],
        saved_data["threshold"],
        saved_data["feature_names"],
    )


def predict_default(customer_data):
    model, threshold, feature_names = load_model()

    customer = pd.DataFrame([customer_data])

    # Ensure columns have the same order used during training.
    customer = customer[feature_names]

    probability = model.predict_proba(customer)[0, 1]
    prediction = int(probability >= threshold)

    return prediction, probability, threshold


if __name__ == "__main__":
    # Example customer values.
    customer = {
        "LIMIT_BAL": 20000,
        "SEX": 2,
        "EDUCATION": 2,
        "MARRIAGE": 1,
        "AGE": 35,
        "PAY_0": 2,
        "PAY_2": 2,
        "PAY_3": 0,
        "PAY_4": 0,
        "PAY_5": 0,
        "PAY_6": 0,
        "BILL_AMT1": 18000,
        "BILL_AMT2": 17500,
        "BILL_AMT3": 16800,
        "BILL_AMT4": 15000,
        "BILL_AMT5": 14000,
        "BILL_AMT6": 13000,
        "PAY_AMT1": 1000,
        "PAY_AMT2": 1200,
        "PAY_AMT3": 1500,
        "PAY_AMT4": 1000,
        "PAY_AMT5": 1000,
        "PAY_AMT6": 800,
    }

    prediction, probability, threshold = predict_default(customer)

    print(f"Default probability: {probability:.2%}")
    print(f"Decision threshold: {threshold:.2%}")

    if prediction == 1:
        print("Prediction: High risk of default")
    else:
        print("Prediction: Low risk of default")