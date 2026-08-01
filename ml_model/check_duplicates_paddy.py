"""
==============================================================
AgroFedVision - Paddy Dataset Duplicate Checker
==============================================================

Purpose:
    Detect duplicate images ONLY in the official training set
    defined by train.csv.

Features
--------
✓ Reads train.csv
✓ Checks only official training images
✓ Detects exact duplicates using MD5 hash
✓ Detects same-class duplicates
✓ Detects cross-class duplicates
✓ Generates duplicate_report.csv
✓ Generates duplicate_summary.txt

Author : AgroFedVision
==============================================================
"""

import os
import hashlib
import pandas as pd

# ==========================================================
# CONFIG
# ==========================================================

DATASET_PATH = r"D:\Download1\archive\paddy-disease-classification"

TRAIN_CSV = os.path.join(DATASET_PATH, "train.csv")
TRAIN_FOLDER = os.path.join(DATASET_PATH, "train_images")

OUTPUT_DIR = "duplicate_report"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================================
# HASH FUNCTION
# ==========================================================

def calculate_md5(filepath):
    """Return MD5 hash of a file."""

    md5 = hashlib.md5()

    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)

    return md5.hexdigest()


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 65)
    print("AgroFedVision - Paddy Duplicate Checker")
    print("=" * 65)

    df = pd.read_csv(TRAIN_CSV)

    print(f"\nImages in train.csv : {len(df)}")

    hash_database = {}

    duplicate_records = []

    missing_images = []

    total_images = 0

    # ------------------------------------------------------

    for _, row in df.iterrows():

        image_name = str(row["image_id"]).strip()
        label = str(row["label"]).strip()

        image_path = os.path.join(TRAIN_FOLDER, label, image_name)

        if not os.path.exists(image_path):

            missing_images.append(
                {
                    "image": image_name,
                    "label": label,
                }
            )

            continue

        total_images += 1

        image_hash = calculate_md5(image_path)

        if image_hash not in hash_database:

            hash_database[image_hash] = {
                "image": image_name,
                "label": label,
                "path": image_path,
            }

        else:

            original = hash_database[image_hash]

            duplicate_records.append(
                {
                    "Original Image": original["image"],
                    "Original Label": original["label"],
                    "Duplicate Image": image_name,
                    "Duplicate Label": label,
                    "Cross Class": original["label"] != label,
                    "MD5": image_hash,
                }
            )

    # ======================================================
    # SAVE REPORTS
    # ======================================================

    duplicate_df = pd.DataFrame(duplicate_records)

    duplicate_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "duplicate_report.csv",
        ),
        index=False,
    )

    pd.DataFrame(missing_images).to_csv(
        os.path.join(
            OUTPUT_DIR,
            "missing_images.csv",
        ),
        index=False,
    )

    # ======================================================
    # SUMMARY
    # ======================================================

    same_class = 0
    cross_class = 0

    if not duplicate_df.empty:

        same_class = (~duplicate_df["Cross Class"]).sum()
        cross_class = duplicate_df["Cross Class"].sum()

    unique_images = total_images - len(duplicate_df)

    summary_lines = [
        "=" * 60,
        "SUMMARY",
        "=" * 60,
        f"Images in CSV          : {len(df)}",
        f"Images Checked         : {total_images}",
        f"Missing Images         : {len(missing_images)}",
        f"Unique Images          : {unique_images}",
        f"Duplicate Images       : {len(duplicate_df)}",
        f"Same Class Duplicates  : {same_class}",
        f"Cross Class Duplicates : {cross_class}",
        "",
    ]

    if len(duplicate_df) == 0:
        summary_lines.append("Dataset is CLEAN.")
    else:
        summary_lines.append("Duplicate images detected.")

    with open(
        os.path.join(
            OUTPUT_DIR,
            "duplicate_summary.txt",
        ),
        "w",
    ) as f:

        for line in summary_lines:
            f.write(line + "\n")

    # ======================================================
    # CONSOLE OUTPUT
    # ======================================================

    print("\n")

    for line in summary_lines:
        print(line)

    print("\nReports saved to:")

    print(os.path.abspath(OUTPUT_DIR))


# ==========================================================
# ENTRY
# ==========================================================

if __name__ == "__main__":
    main()