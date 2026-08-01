"""
dedup_guava_dataset.py

Uses the duplicate report from check_duplicates.py to build a clean,
deduplicated guava dataset -- keeping exactly ONE image per duplicate
group (the Roboflow export padded 1,250-per-class by literally
copying existing images under new "_dupNNN" filenames; training on
those copies with a random split is what caused the false 99-100%
validation accuracy).

Input : data/guava_duplicate_report.txt  (from check_duplicates.py)
        data/guava_dataset.csv           (original, with labels)
Output: data/guava_dataset_dedup.csv     (clean, ~4,439 unique images)
"""

import os
import pandas as pd

REPORT_PATH = "data/guava_duplicate_report.txt"
ORIGINAL_CSV = "data/guava_dataset.csv"
OUTPUT_CSV = "data/guava_dataset_dedup.csv"


def parse_duplicate_groups(report_path):
    groups = []
    current_group = []

    with open(report_path, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("\nGroup") or line.startswith("Group"):
                if current_group:
                    groups.append(current_group)
                current_group = []
            elif line.strip().startswith("D:") or line.strip().startswith("/") or ":\\" in line:
                current_group.append(line.strip())
        if current_group:
            groups.append(current_group)

    return groups


def main():
    print("Parsing duplicate report...")
    groups = parse_duplicate_groups(REPORT_PATH)
    print(f"Found {len(groups)} duplicate groups in report")

    # Keep exactly one representative image per group (the first one listed)
    paths_to_drop = set()
    for group in groups:
        # keep group[0], drop the rest
        for path in group[1:]:
            paths_to_drop.add(path)

    print(f"Images to drop (redundant copies): {len(paths_to_drop)}")

    print("\nLoading original dataset...")
    df = pd.read_csv(ORIGINAL_CSV)
    print(f"Original size: {len(df)}")

    df_clean = df[~df["image_path"].isin(paths_to_drop)].reset_index(drop=True)
    print(f"Deduplicated size: {len(df_clean)}")

    print("\nClass distribution after dedup:")
    print(df_clean["crop_health"].value_counts())

    print("\nOriginal-label distribution after dedup:")
    print(df_clean["original_label"].value_counts())

    df_clean.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}")
    print("\nUpdate train_phase1_crop.py's DATASET_MAP to point 'guava' at "
          f"'{OUTPUT_CSV}' before retraining.")


if __name__ == "__main__":
    main()