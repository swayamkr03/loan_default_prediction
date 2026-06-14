# Loan Default Prediction

## Overview
This project predicts whether a credit-card customer is likely to default on their payment next month. It compares Logistic Regression and Random Forest models, tunes the decision threshold, and provides an interactive Streamlit interface for predictions.

## Dataset
The project uses the [UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) dataset containing 30,000 customer records.

The features include credit limits, demographics, repayment history, bill amounts, and previous payment amounts.

The target column is `default payment next month`:

- `0`: No default
- `1`: Default

## Project Structure
```text
loan-default-prediction/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── reports/
│   └── figures/
├── src/
│   ├── train.py
│   └── predict.py
├── app.py
├── convert_to_csv.py
├── explore_data.py
├── requirements.txt
├── .gitignore
└── README.md

###Installation
Create and activate a virtual environment:
python -m venv .venv
.venv\Scripts\activate
Install the dependencies:
pip install -r requirements.txt
Usage
Convert the Excel dataset to CSV:
python convert_to_csv.py
Explore the dataset:
python explore_data.py
Train and evaluate the models:
python src/train.py
Run a sample prediction:
python src/predict.py
Launch the Streamlit interface:
streamlit run app.py
Model Results
The selected Random Forest model achieved approximately:
Metric	Score
Test ROC-AUC	0.759
Default precision	0.47
Default recall	0.58
Default F1-score	0.52
Accuracy	0.76

Threshold tuning improved the model's ability to identify customers who default, although it also increased false-positive predictions.
###Technologies
Python
Pandas
scikit-learn
Joblib
Streamlit
Plotly
Matplotlib
Seaborn
###Future Improvements
Hyperparameter tuning
Cross-validation
SHAP model explanations
Probability calibration
Fairness and bias analysis
Cloud deployment