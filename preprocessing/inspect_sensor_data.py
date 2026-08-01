import pandas as pd

df = pd.read_csv("data/Crop_recommendationV2.csv")

print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())