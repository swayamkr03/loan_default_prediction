from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "loan_default_model.joblib"

app = Flask(__name__)


def load_saved_model():
    saved = joblib.load(MODEL_PATH)
    return saved["model"], saved["threshold"], saved["feature_names"]


model, threshold, feature_names = load_saved_model()


def risk_label(probability):
    if probability >= threshold:
        return "High risk"
    if probability >= threshold * 0.7:
        return "Moderate risk"
    return "Low risk"


def build_warnings(customer_data):
    warnings = []
    statuses = [
        customer_data["PAY_0"],
        customer_data["PAY_2"],
        customer_data["PAY_3"],
        customer_data["PAY_4"],
        customer_data["PAY_5"],
        customer_data["PAY_6"],
    ]

    latest_bill = customer_data["BILL_AMT1"]
    latest_payment = customer_data["PAY_AMT1"]
    credit_limit = max(customer_data["LIMIT_BAL"], 1)

    if statuses[0] > 0:
        warnings.append("Latest repayment is delayed.")
    if max(statuses) >= 2:
        warnings.append("Repayment history includes a delay of two or more months.")
    if latest_bill > credit_limit * 0.8:
        warnings.append("Latest bill uses more than 80% of the credit limit.")
    if latest_payment < latest_bill * 0.1:
        warnings.append("Latest payment is less than 10% of the latest bill.")

    if not warnings:
        warnings.append("No simple warning rules were triggered.")

    return warnings


@app.route("/")
def index():
    return render_template("index.html", threshold=round(threshold * 100, 1))


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json()

    try:
        customer_data = {name: float(payload[name]) for name in feature_names}
    except KeyError as error:
        return jsonify({"error": f"Missing field: {error.args[0]}"}), 400
    except (TypeError, ValueError):
        return jsonify({"error": "All fields must contain numeric values."}), 400

    customer = pd.DataFrame([customer_data])[feature_names]
    probability = float(model.predict_proba(customer)[0, 1])
    prediction = int(probability >= threshold)

    return jsonify(
        {
            "probability": probability,
            "probability_percent": round(probability * 100, 2),
            "threshold": threshold,
            "threshold_percent": round(threshold * 100, 2),
            "prediction": prediction,
            "label": risk_label(probability),
            "credit_usage_percent": round(
                customer_data["BILL_AMT1"] / max(customer_data["LIMIT_BAL"], 1) * 100,
                2,
            ),
            "warnings": build_warnings(customer_data),
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
