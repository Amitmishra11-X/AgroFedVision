import pandas as pd
import os

base_dir = os.path.dirname(os.path.dirname(__file__))
csv_path = os.path.join(base_dir, "data", "Crop_recommendationV2.csv")

df = pd.read_csv(csv_path)

print("="*50)
print("DATASET SHAPE")
print("="*50)
print(df.shape)

print("\n")

print("="*50)
print("COLUMN NAMES")
print("="*50)
print(df.columns)

print("\n")

print("="*50)
print("MISSING VALUES")
print("="*50)
print(df.isnull().sum())

print("\n")

print("="*50)
print("DUPLICATES")
print("="*50)
print(df.duplicated().sum())

print("\n")

print("="*50)
print("DATA TYPES")
print("="*50)
print(df.dtypes)

print("\n")

print("="*50)
print("STATISTICS")
print("="*50)
print(df.describe())

print("\n")

print("="*50)
print("CROP DISTRIBUTION")
print("="*50)
print(df['label'].value_counts())