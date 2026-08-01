import pandas as pd

df = pd.read_csv("data/leaf_dataset.csv")

print(df["label"].value_counts())