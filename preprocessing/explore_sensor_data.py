import pandas as pd

df = pd.read_csv("data/Crop_recommendationV2.csv")

print(df["label"].value_counts())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistics:")
print(df.describe())