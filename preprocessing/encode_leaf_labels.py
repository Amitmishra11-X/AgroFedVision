import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data/leaf_dataset_clean.csv")

encoder = LabelEncoder()

df["label_encoded"] = encoder.fit_transform(df["label"])

df.to_csv(
    "data/leaf_dataset_encoded.csv",
    index=False
)

print("Classes:")
for i, cls in enumerate(encoder.classes_):
    print(i, cls)