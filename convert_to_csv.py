import pandas as pd

input_path="data/raw/credit_default.xls"
output_path="data/processed/credit_default.csv"

df=pd.read_excel(input_path,header=1)
df.to_csv(output_path,index=False)
