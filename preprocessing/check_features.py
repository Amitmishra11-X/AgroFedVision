import pandas as pd

df = pd.read_csv("uav_features.csv")

print(df.shape)
print(df.head())
print(df.describe())