import hashlib
import os
from pathlib import Path
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = r"D:\Download1\archive (1)\Corn Disease detection"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# ============================================================
# DUPLICATE DETECTION
# ============================================================

hash_database = {}

duplicates = []

total_images = 0

print("=" * 60)
print("AgroFedVision - Duplicate Image Checker")
print("=" * 60)

for class_folder in sorted(os.listdir(DATASET_DIR)):

    class_path = os.path.join(DATASET_DIR, class_folder)

    if not os.path.isdir(class_path):
        continue

    print(f"\nScanning : {class_folder}")

    for root, _, files in os.walk(class_path):

        for file in files:

            if not file.lower().endswith(IMAGE_EXTENSIONS):
                continue

            total_images += 1

            filepath = os.path.join(root, file)

            try:

                with open(filepath, "rb") as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()

                if file_hash not in hash_database:

                    hash_database[file_hash] = {
                        "path": filepath,
                        "class": class_folder
                    }

                else:

                    original = hash_database[file_hash]

                    duplicates.append({

                        "Original Image":
                            original["path"],

                        "Original Class":
                            original["class"],

                        "Duplicate Image":
                            filepath,

                        "Duplicate Class":
                            class_folder,

                        "Cross Class Duplicate":
                            original["class"] != class_folder

                    })

            except Exception as e:

                print(f"Error reading {filepath}")

                print(e)

# ============================================================
# SAVE REPORT
# ============================================================

df = pd.DataFrame(duplicates)

report_file = "duplicate_report.csv"

df.to_csv(report_file, index=False)

same_class = 0
cross_class = 0

for row in duplicates:

    if row["Cross Class Duplicate"]:
        cross_class += 1
    else:
        same_class += 1

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"Total Images           : {total_images}")
print(f"Unique Images          : {len(hash_database)}")
print(f"Duplicate Images       : {len(duplicates)}")
print(f"Same Class Duplicates  : {same_class}")
print(f"Cross Class Duplicates : {cross_class}")

print(f"\nCSV Report Saved : {report_file}")

if len(duplicates) == 0:
    print("\n✅ Dataset is CLEAN.")
else:
    print("\n⚠ Duplicate images found. Review duplicate_report.csv")

print("=" * 60)