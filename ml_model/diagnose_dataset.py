"""
diagnose_dataset.py — Checks for the most common causes of "stuck at majority class"
Run: python diagnose_dataset.py
"""

import pandas as pd
import numpy as np
import os
from collections import Counter

df = pd.read_csv("data/leaf_dataset_clean.csv")

print("="*60)
print("CHECK 1: CSV structure")
print("="*60)
print(df.head(10))
print("\nColumns:", df.columns.tolist())
print("Total rows:", len(df))

print("\n" + "="*60)
print("CHECK 2: Do image_path values actually exist on disk?")
print("="*60)
missing = 0
for i, row in df.iterrows():
    if not os.path.exists(row["image_path"]):
        missing += 1
        if missing <= 10:
            print("MISSING:", row["image_path"])
print(f"\nTotal missing files: {missing} / {len(df)}")

print("\n" + "="*60)
print("CHECK 3: Are paths actually unique, or duplicated?")
print("="*60)
path_counts = Counter(df["image_path"])
dupes = {p: c for p, c in path_counts.items() if c > 1}
print(f"Duplicate image_path entries: {len(dupes)}")
if dupes:
    for p, c in list(dupes.items())[:10]:
        print(f"  {p} appears {c} times")

print("\n" + "="*60)
print("CHECK 4: Sample 5 random rows — manually verify these")
print("="*60)
sample = df.sample(min(5, len(df)), random_state=1)
for _, row in sample.iterrows():
    print(f"  label='{row['label']}'  ->  path='{row['image_path']}'")
print("\n>>> Open these 5 image files manually and check if the")
print(">>> image content actually matches the claimed label.")

print("\n" + "="*60)
print("CHECK 5: Class distribution")
print("="*60)
print(df["label"].value_counts())

print("\n" + "="*60)
print("CHECK 6: Folder structure sanity check")
print("="*60)
# Show what folder each image_path is actually sitting in,
# compared against its CSV label
df["actual_folder"] = df["image_path"].apply(lambda p: os.path.basename(os.path.dirname(p)))
mismatch_check = df.groupby("label")["actual_folder"].unique()
print("Label  ->  Set of folders its images actually come from:")
for label, folders in mismatch_check.items():
    flag = "  <<< MULTIPLE FOLDERS — SUSPICIOUS" if len(folders) > 1 else ""
    print(f"  {label:30s} -> {list(folders)}{flag}")
