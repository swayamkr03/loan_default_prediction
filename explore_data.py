import pandas as pd

df=pd.read_csv("data/processed/credit_default.csv")

print(df.shape)
print(df.head())

print(df.columns.tolist())

print(df.isnull().sum())

print(df.duplicated().sum())

target="default payment next month"
print(df[target].value_counts())

print(df[target].value_counts(normalize=True)*100)