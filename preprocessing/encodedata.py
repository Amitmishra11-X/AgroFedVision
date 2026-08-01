# preprocessing/encode_data.py

import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

base_dir = os.path.dirname(os.path.dirname(__file__))
csv_path = os.path.join(base_dir, "data", "Crop_recommendationV2.csv")

df = pd.read_csv(csv_path)

encoder = LabelEncoder()

df["label"] = encoder.fit_transform(df["label"])

print(df["label"].head())

print("\nClasses:")
print(encoder.classes_)

save_path = os.path.join(base_dir, "data", "encoded_dataset.csv")

df.to_csv(save_path,index=False)

print("\nSaved Successfully")