"""
==========================================================
AgroFedVision Dataset Validator
==========================================================

Author : AgroFedVision
Purpose:
    Validate image datasets before training.

Features
--------
✓ Validates every image listed in train.csv
✓ Detects missing images
✓ Detects corrupted images
✓ Reports class distribution
✓ Reports image resolutions
✓ Saves validation reports
✓ Supports:
      - Flat image folders
      - Class-wise image folders

Outputs
-------
dataset_report/
│
├── dataset_statistics.csv
├── missing_images.csv
├── corrupted_images.csv
├── image_resolution_summary.csv
└── validation_report.txt
==========================================================
"""

import os
from collections import Counter

import pandas as pd
from PIL import Image

# ======================================================
# CONFIG
# ======================================================

DATASET_PATH = r"D:\Download1\archive\paddy-disease-classification"

TRAIN_CSV = os.path.join(DATASET_PATH, "train.csv")
TRAIN_FOLDER = os.path.join(DATASET_PATH, "train_images")

OUTPUT_DIR = "dataset_report"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ======================================================
# IMAGE VALIDATION
# ======================================================

def validate_image(image_path):
    """
    Returns:
        valid, width, height
    """

    try:

        with Image.open(image_path) as img:
            img.verify()

        with Image.open(image_path) as img:
            width, height = img.size

        return True, width, height

    except Exception:

        return False, None, None


# ======================================================
# FIND IMAGE
# ======================================================

def get_image_path(image_name, label):
    """
    Supports both:

    train_images/image.jpg

    and

    train_images/class_name/image.jpg
    """

    flat = os.path.join(TRAIN_FOLDER, image_name)

    if os.path.exists(flat):
        return flat

    class_path = os.path.join(TRAIN_FOLDER, label, image_name)

    if os.path.exists(class_path):
        return class_path

    return None


# ======================================================
# MAIN
# ======================================================

def main():

    print("=" * 65)
    print("AgroFedVision Dataset Validator")
    print("=" * 65)

    df = pd.read_csv(TRAIN_CSV)

    print(f"\nDataset Entries : {len(df)}")

    class_distribution = df["label"].value_counts().sort_index()

    print("\nClass Distribution")
    print(class_distribution)

    valid_records = []
    missing_images = []
    corrupted_images = []

    resolution_counter = Counter()

    # ---------------------------------------------

    for _, row in df.iterrows():

        image_name = str(row["image_id"]).strip()
        label = str(row["label"]).strip()

        image_path = get_image_path(image_name, label)

        if image_path is None:

            missing_images.append(
                {
                    "image": image_name,
                    "label": label,
                }
            )

            continue

        valid, width, height = validate_image(image_path)

        if not valid:

            corrupted_images.append(
                {
                    "image": image_name,
                    "label": label,
                }
            )

            continue

        resolution = f"{width}x{height}"

        resolution_counter[resolution] += 1

        valid_records.append(
            {
                "filepath": image_path,
                "image": image_name,
                "label": label,
                "width": width,
                "height": height,
            }
        )

    # ======================================================
    # SAVE CSV FILES
    # ======================================================

    pd.DataFrame(valid_records).to_csv(
        os.path.join(OUTPUT_DIR, "dataset_statistics.csv"),
        index=False,
    )

    pd.DataFrame(missing_images).to_csv(
        os.path.join(OUTPUT_DIR, "missing_images.csv"),
        index=False,
    )

    pd.DataFrame(corrupted_images).to_csv(
        os.path.join(OUTPUT_DIR, "corrupted_images.csv"),
        index=False,
    )

    resolution_df = pd.DataFrame(
        {
            "resolution": list(resolution_counter.keys()),
            "count": list(resolution_counter.values()),
        }
    ).sort_values("count", ascending=False)

    resolution_df.to_csv(
        os.path.join(OUTPUT_DIR, "image_resolution_summary.csv"),
        index=False,
    )

    # ======================================================
    # REPORT
    # ======================================================

    report_path = os.path.join(
        OUTPUT_DIR,
        "validation_report.txt",
    )

    with open(report_path, "w") as f:

        f.write("=" * 60 + "\n")
        f.write("AGROFEDVISION DATASET VALIDATION REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Total CSV Entries : {len(df)}\n")
        f.write(f"Valid Images      : {len(valid_records)}\n")
        f.write(f"Missing Images    : {len(missing_images)}\n")
        f.write(f"Corrupted Images  : {len(corrupted_images)}\n\n")

        f.write("CLASS DISTRIBUTION\n")
        f.write("-" * 60 + "\n")

        for cls, count in class_distribution.items():
            f.write(f"{cls:30s} {count}\n")

        f.write("\nIMAGE RESOLUTIONS\n")
        f.write("-" * 60 + "\n")

        for _, row in resolution_df.iterrows():
            f.write(f"{row['resolution']:15s} {row['count']}\n")

    # ======================================================
    # CONSOLE OUTPUT
    # ======================================================

    print("\n" + "=" * 65)
    print("VALIDATION SUMMARY")
    print("=" * 65)

    print(f"Valid Images     : {len(valid_records)}")
    print(f"Missing Images   : {len(missing_images)}")
    print(f"Corrupted Images : {len(corrupted_images)}")

    print("\nTop Image Resolutions")

    print(resolution_df.head(10))

    print("\nReports Saved")

    print(f"  {OUTPUT_DIR}/dataset_statistics.csv")
    print(f"  {OUTPUT_DIR}/missing_images.csv")
    print(f"  {OUTPUT_DIR}/corrupted_images.csv")
    print(f"  {OUTPUT_DIR}/image_resolution_summary.csv")
    print(f"  {OUTPUT_DIR}/validation_report.txt")

    print("\nDataset validation completed successfully.")


# ======================================================
# ENTRY POINT
# ======================================================

if __name__ == "__main__":
    main()