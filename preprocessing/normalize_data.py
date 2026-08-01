import pandas as pd
import os

from sklearn.preprocessing import StandardScaler

base_dir = os.path.dirname(os.path.dirname(__file__))

csv_path = os.path.join(
    base_dir,
    "data",
    "encoded_dataset.csv"
)

df = pd.read_csv(csv_path)

X = df.drop("label",axis=1)

y = df["label"]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_scaled = pd.DataFrame(
    X_scaled,
    columns=X.columns
)

X_scaled["label"] = y

save_path = os.path.join(
    base_dir,
    "data",
    "normalized_dataset.csv"
)

X_scaled.to_csv(
    save_path,
    index=False
)

print("Normalization Complete")
print(X_scaled.head())