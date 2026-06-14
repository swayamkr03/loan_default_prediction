from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

MODEL_PATH = Path("models/loan_default_model.joblib")

st.set_page_config(
    page_title="Loan Default Predictor",
    page_icon="💳",
    layout="wide",
)


@st.cache_resource
def load_saved_model():
    saved = joblib.load(MODEL_PATH)
    return saved["model"], saved["threshold"], saved["feature_names"]


def risk_gauge(probability, threshold):
    color = (
        "#22c55e" if probability < threshold * 0.7
        else "#f59e0b" if probability < threshold
        else "#ef4444"
    )

    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 40}},
            title={"text": "Estimated Default Risk"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {
                        "range": [0, threshold * 70],
                        "color": "#dcfce7",
                    },
                    {
                        "range": [threshold * 70, threshold * 100],
                        "color": "#fef3c7",
                    },
                    {
                        "range": [threshold * 100, 100],
                        "color": "#fee2e2",
                    },
                ],
                "threshold": {
                    "line": {"color": "#111827", "width": 4},
                    "value": threshold * 100,
                },
            },
        )
    )

    figure.update_layout(height=330, margin=dict(l=30, r=30, t=60, b=20))
    return figure


st.title("Loan Default Risk Predictor")
st.caption("Estimate whether a credit-card customer may default next month.")

try:
    model, threshold, feature_names = load_saved_model()
except FileNotFoundError:
    st.error("Model not found. Run `python src/train.py` first.")
    st.stop()

preset = st.radio(
    "Load an example profile",
    ["Custom", "Low-risk example", "High-risk example"],
    horizontal=True,
)

if preset == "Low-risk example":
    default_limit = 200000
    default_age = 40
    default_statuses = [-1, -1, -1, -1, -1, -1]
    default_bills = [30000, 28000, 25000, 22000, 20000, 18000]
    default_payments = [15000, 12000, 10000, 9000, 8000, 7000]
elif preset == "High-risk example":
    default_limit = 20000
    default_age = 30
    default_statuses = [3, 2, 2, 2, 1, 1]
    default_bills = [19500, 19000, 18500, 18000, 17500, 17000]
    default_payments = [0, 500, 0, 500, 0, 500]
else:
    default_limit = 50000
    default_age = 35
    default_statuses = [0, 0, 0, 0, 0, 0]
    default_bills = [18000, 17500, 16800, 15000, 14000, 13000]
    default_payments = [1000, 1200, 1500, 1000, 1000, 800]

with st.form("customer_form"):
    customer_tab, history_tab = st.tabs(
        ["Customer details", "Six-month financial history"]
    )

    with customer_tab:
        col1, col2 = st.columns(2)

        with col1:
            limit_balance = st.number_input(
                "Credit limit",
                min_value=0,
                value=default_limit,
                step=1000,
                help="Maximum credit available to the customer.",
            )
            age = st.slider("Age", 18, 100, default_age)

        with col2:
            sex = st.selectbox("Sex", [1, 2], format_func=lambda x: {
                1: "Male", 2: "Female"
            }[x])

            education = st.selectbox(
                "Education",
                [1, 2, 3, 4],
                format_func=lambda x: {
                    1: "Graduate school",
                    2: "University",
                    3: "High school",
                    4: "Other",
                }[x],
            )

            marriage = st.selectbox(
                "Marital status",
                [1, 2, 3],
                format_func=lambda x: {
                    1: "Married",
                    2: "Single",
                    3: "Other",
                }[x],
            )

    with history_tab:
        st.info(
            "Repayment status: -2 = no balance, -1 = paid in full, "
            "0 = no delay, 1–8 = months late."
        )

        statuses = []
        bills = []
        payments = []

        for index in range(6):
            st.markdown(f"#### Month {index + 1}")
            col1, col2, col3 = st.columns(3)

            with col1:
                statuses.append(
                    st.select_slider(
                        f"Repayment status {index + 1}",
                        options=list(range(-2, 9)),
                        value=default_statuses[index],
                    )
                )

            with col2:
                bills.append(
                    st.number_input(
                        f"Bill amount {index + 1}",
                        value=default_bills[index],
                        step=500,
                    )
                )

            with col3:
                payments.append(
                    st.number_input(
                        f"Payment amount {index + 1}",
                        min_value=0,
                        value=default_payments[index],
                        step=500,
                    )
                )

    submitted = st.form_submit_button(
        "Calculate default risk",
        type="primary",
        use_container_width=True,
    )

if submitted:
    customer_data = {
        "LIMIT_BAL": limit_balance,
        "SEX": sex,
        "EDUCATION": education,
        "MARRIAGE": marriage,
        "AGE": age,
        "PAY_0": statuses[0],
        "PAY_2": statuses[1],
        "PAY_3": statuses[2],
        "PAY_4": statuses[3],
        "PAY_5": statuses[4],
        "PAY_6": statuses[5],
        **{f"BILL_AMT{i + 1}": bills[i] for i in range(6)},
        **{f"PAY_AMT{i + 1}": payments[i] for i in range(6)},
    }

    customer = pd.DataFrame([customer_data])[feature_names]
    probability = float(model.predict_proba(customer)[0, 1])
    prediction = int(probability >= threshold)

    st.divider()
    st.subheader("Risk assessment")

    left, right = st.columns([1.4, 1])

    with left:
        st.plotly_chart(
            risk_gauge(probability, threshold),
            use_container_width=True,
        )

    with right:
        st.metric("Default probability", f"{probability:.2%}")
        st.metric("Decision threshold", f"{threshold:.2%}")
        st.metric("Credit usage", f"{bills[0] / max(limit_balance, 1):.1%}")

        if prediction:
            st.error("High risk: estimated probability exceeds the threshold.")
        else:
            st.success("Low risk: estimated probability is below the threshold.")

    with st.expander("Why might this profile be risky?"):
        warnings = []

        if statuses[0] > 0:
            warnings.append("The latest repayment is delayed.")
        if max(statuses) >= 2:
            warnings.append("There is a delay of at least two months.")
        if bills[0] > limit_balance * 0.8:
            warnings.append("The latest bill uses over 80% of the credit limit.")
        if payments[0] < bills[0] * 0.1:
            warnings.append("The latest payment is below 10% of the latest bill.")

        if warnings:
            for warning in warnings:
                st.write(f"- {warning}")
        else:
            st.write("No simple warning rules were triggered.")

    st.caption(
        "This is an educational ML estimate and should not be used as the "
        "sole basis for a lending decision."
    )