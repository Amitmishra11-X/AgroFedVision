"""
build_phase1_datasets.py

AgroFedVision Phase 1 -- Guava + Maize dataset construction

Builds two independent, crop-specific CSVs:
    data/guava_dataset.csv
    data/maize_dataset.csv

IMPORTANT: Only ONE copy of the guava data is used (CNN256), because
CNN256, CNN640, and the five guyava_yolovN folders are almost certainly
the SAME underlying Roboflow dataset exported 6 different times (same
9 class names, same order, in every export). Loading more than one
copy would duplicate images across train/validation splits and
silently inflate accuracy through data leakage. Do not add the other
folders without first checking for exact duplicate images.
"""

import os
import glob
import pandas as pd

# ==========================================================
# CONFIG -- update these two paths to match your machine
# ==========================================================

GUAVA_ROOT = r"D:\Download1\archive (5)\CNN256\CNN256"   # contains train/ test/ valid/
MAIZE_ROOT = r"D:\Download1\archive (1)\Corn Disease detection"  # contains Healthy corn/ Infected/

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# GUAVA: 9 original classes -> 3-class severity mapping
# ==========================================================
# This mapping is a reasonable default, not an agronomist-verified
# ground truth -- adjust if you have domain expertise to refine it.

GUAVA_HEALTH_MAP = {
    "healthy_leaf":        "Healthy",
    "fruit_healthy":       "Healthy",

    "algal_leaf_spot":     "Moderate_Risk",
    "insect_bite":         "Moderate_Risk",
    "scorch":              "Moderate_Risk",

    "black_mold":          "High_Risk",
    "red_rust":             "High_Risk",
    "scab":                "High_Risk",
    "yellow_leaf_disease": "High_Risk",
}

# ==========================================================
# MAIZE: 2 original classes -> 3-class severity mapping
# ==========================================================
# Only 2 source classes exist (Healthy / Infected), so Maize
# starts as an effectively 2-class problem folded into the same
# 3-label schema. "Infected" is mapped to High_Risk rather than
# Moderate_Risk since the dataset gives no severity gradation --
# treat this as a known simplification, not a hidden assumption.

MAIZE_HEALTH_MAP = {
    "Healthy corn": "Healthy",
    "Infected":     "High_Risk",
}


def build_guava_dataset():
    print("\n" + "=" * 50)
    print("Building Guava Dataset (CNN256 only)")
    print("=" * 50)

    records = []
    skipped_classes = []

    for split in ["train", "test", "valid"]:
        split_dir = os.path.join(GUAVA_ROOT, split)
        if not os.path.isdir(split_dir):
            print(f"  [skip] {split_dir} not found")
            continue

        for class_name in os.listdir(split_dir):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            if class_name not in GUAVA_HEALTH_MAP:
                skipped_classes.append(class_name)
                continue

            mapped_label = GUAVA_HEALTH_MAP[class_name]

            image_paths = (
                glob.glob(os.path.join(class_dir, "*.jpg")) +
                glob.glob(os.path.join(class_dir, "*.jpeg")) +
                glob.glob(os.path.join(class_dir, "*.png"))
            )

            for path in image_paths:
                records.append({
                    "image_path": path,
                    "original_label": class_name,
                    "crop_health": mapped_label,
                    "crop": "guava",
                    "source_split": split,   # kept for reference only;
                                              # we re-split ourselves later
                })

    if skipped_classes:
        print(f"  [warning] Unmapped class folders found and skipped: "
              f"{sorted(set(skipped_classes))}")

    df = pd.DataFrame(records)

    if len(df) == 0:
        print("  [ERROR] No guava images found. Check GUAVA_ROOT path.")
        return None

    print(f"\nTotal guava images: {len(df)}")
    print("\nOriginal class distribution:")
    print(df["original_label"].value_counts())
    print("\nMapped crop_health distribution:")
    print(df["crop_health"].value_counts())

    # ---- Duplicate filename check (cheap sanity check for the
    #      "same dataset exported twice" risk, based on basename) ----
    df["basename"] = df["image_path"].apply(os.path.basename)
    dup_count = df["basename"].duplicated().sum()
    if dup_count > 0:
        print(f"\n  [warning] {dup_count} duplicate filenames detected "
              f"within CNN256 itself (likely train/test/valid overlap "
              f"or repeated images). Review before training.")
    df = df.drop(columns=["basename"])

    out_path = os.path.join(OUTPUT_DIR, "guava_dataset.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    return df


def build_maize_dataset():
    print("\n" + "=" * 50)
    print("Building Maize Dataset")
    print("=" * 50)

    records = []

    for class_name, mapped_label in MAIZE_HEALTH_MAP.items():
        class_dir = os.path.join(MAIZE_ROOT, class_name)
        if not os.path.isdir(class_dir):
            print(f"  [warning] {class_dir} not found, skipping")
            continue

        image_paths = (
            glob.glob(os.path.join(class_dir, "*.jpg")) +
            glob.glob(os.path.join(class_dir, "*.jpeg")) +
            glob.glob(os.path.join(class_dir, "*.png"))
        )

        for path in image_paths:
            records.append({
                "image_path": path,
                "original_label": class_name,
                "crop_health": mapped_label,
                "crop": "maize",
            })

    df = pd.DataFrame(records)

    if len(df) == 0:
        print("  [ERROR] No maize images found. Check MAIZE_ROOT path.")
        return None

    print(f"\nTotal maize images: {len(df)}")
    print("\nOriginal class distribution:")
    print(df["original_label"].value_counts())
    print("\nMapped crop_health distribution:")
    print(df["crop_health"].value_counts())

    out_path = os.path.join(OUTPUT_DIR, "maize_dataset.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    return df


if __name__ == "__main__":
    guava_df = build_guava_dataset()
    maize_df = build_maize_dataset()

    print("\n" + "=" * 50)
    print("PHASE 1 DATASET SUMMARY")
    print("=" * 50)
    if guava_df is not None:
        print(f"Guava : {len(guava_df)} images, "
              f"{guava_df['crop_health'].nunique()} classes")
    if maize_df is not None:
        print(f"Maize : {len(maize_df)} images, "
              f"{maize_df['crop_health'].nunique()} classes")
    print("\nNote: Maize currently only has 2 source classes "
          "(Healthy/Infected), so Moderate_Risk will be empty for maize "
          "until a more granular labeled dataset is found.")
    print("\nDone.")