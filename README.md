# Loan Default Prediction

**Live Demo:** [loandefaultyprediction.streamlit.app](https://loandefaultyprediction.streamlit.app/)

## Overview

This project predicts whether a credit-card customer is likely to default on their payment next month. It compares Logistic Regression and Random Forest models, tunes the classification threshold, and provides a polished Flask dashboard for interactive predictions.

The Flask frontend includes example customer profiles, a custom risk gauge, default-probability output, credit-usage metrics, and simple risk-signal explanations.

## Dataset

The project uses the [UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) dataset, containing 30,000 customer records. The features include credit limits, demographics, repayment history, bill amounts, and previous payment amounts.

The target column is `default payment next month`:

| Value | Meaning |
|-------|---------|
| `0` | No default |
| `1` | Default |

## Project Structure

```text
loan-default-prediction/
|-- data/
|   |-- raw/
|   `-- processed/
|-- models/
|   `-- loan_default_model.joblib
|-- reports/
|   `-- figures/
|-- src/
|   |-- train.py
|   `-- predict.py
|-- static/
|   |-- script.js
|   `-- styles.css
|-- templates/
|   `-- index.html
|-- flask_app.py
|-- app.py
|-- convert_to_csv.py
|-- explore_data.py
|-- requirements.txt
|-- .gitignore
`-- README.md
```

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Convert the Excel dataset to CSV:

```bash
python convert_to_csv.py
```

Explore the dataset:

```bash
python explore_data.py
```

Train and evaluate the models:

```bash
python src/train.py
```

Run a sample prediction:

```bash
python src/predict.py
```

Launch the Flask dashboard:

```bash
python flask_app.py
```

Then open:

```text
http://127.0.0.1:5000
```

Optionally launch the older Streamlit interface:

```bash
streamlit run app.py
```

## Model Results

The selected Random Forest model achieved approximately:

| Metric | Score |
|--------|------:|
| Test ROC-AUC | 0.759 |
| Default precision | 0.47 |
| Default recall | 0.58 |
| Default F1-score | 0.52 |
| Accuracy | 0.76 |

Threshold tuning improved the model's ability to identify customers who default, although it also increased false-positive predictions.

## Technologies

- Python
- Pandas
- scikit-learn
- Joblib
- Flask
- HTML
- CSS
- JavaScript
- Streamlit
- Plotly
- Matplotlib
- Seaborn

## Future Improvements

- Hyperparameter tuning
- Cross-validation
- SHAP model explanations
- Probability calibration
- Fairness and bias analysis
- Flask deployment on Render or Railway

## Disclaimer

This project is intended for educational and portfolio purposes only. It should not be used as the sole basis for real financial or lending decisions.
