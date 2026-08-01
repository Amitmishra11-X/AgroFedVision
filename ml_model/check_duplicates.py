"""
check_duplicates.py

Checks the guava dataset for near-duplicate images (same photo,
slightly rotated/cropped/brightness-shifted) using perceptual
hashing. This is the likely explanation for the suspicious
99-100% validation accuracy seen in the first training run --
if near-duplicates exist across train/validation folds, the
model can "cheat" by recognizing an image it already memorized
a near-twin of, rather than learning real disease features.

Requires: pip install imagehash pillow --break-system-packages
"""

import os
import pandas as pd
from PIL import Image
import imagehash
from collections import defaultdict

DATASET_CSV = "data/guava_dataset.csv"
HASH_SIZE = 8            # standard perceptual hash size
SIMILARITY_THRESHOLD = 5  # hamming distance <= this = "near duplicate"
                           # 0 = identical, 5 = very similar, 10+ = different

def main():
    df = pd.read_csv(DATASET_CSV)
    print(f"Checking {len(df)} images for near-duplicates...")
    print("(this will take a few minutes for 11,250 images)\n")

    hashes = {}
    errors = 0

    for i, row in df.iterrows():
        try:
            img = Image.open(row["image_path"])
            h = imagehash.phash(img, hash_size=HASH_SIZE)
            hashes[row["image_path"]] = h
        except Exception as e:
            errors += 1
        if i % 1000 == 0:
            print(f"  processed {i}/{len(df)}...")

    print(f"\nHashed {len(hashes)} images ({errors} failed to open)")

    # Group near-identical hashes together
    paths = list(hashes.keys())
    hash_values = list(hashes.values())

    print("\nComparing hashes for near-duplicates (may take a while)...")
    duplicate_groups = []
    checked = set()

    for i in range(len(paths)):
        if paths[i] in checked:
            continue
        group = [paths[i]]
        for j in range(i + 1, len(paths)):
            if paths[j] in checked:
                continue
            if hash_values[i] - hash_values[j] <= SIMILARITY_THRESHOLD:
                group.append(paths[j])
                checked.add(paths[j])
        if len(group) > 1:
            duplicate_groups.append(group)
            checked.add(paths[i])

    total_duplicate_images = sum(len(g) for g in duplicate_groups)

    print(f"\n{'='*50}")
    print("DUPLICATE CHECK RESULTS")
    print(f"{'='*50}")
    print(f"Total images checked      : {len(hashes)}")
    print(f"Near-duplicate groups     : {len(duplicate_groups)}")
    print(f"Images involved in groups : {total_duplicate_images}")
    print(f"Percentage of dataset     : {100 * total_duplicate_images / len(hashes):.1f}%")

    if duplicate_groups:
        print(f"\nExample duplicate group (first 3 shown):")
        for path in duplicate_groups[0][:3]:
            print(f"  {path}")

        # Save full report
        with open("data/guava_duplicate_report.txt", "w") as f:
            for gi, group in enumerate(duplicate_groups):
                f.write(f"\nGroup {gi + 1} ({len(group)} images):\n")
                for path in group:
                    f.write(f"  {path}\n")
        print(f"\nFull report saved to data/guava_duplicate_report.txt")

        print("\n" + "!" * 50)
        print("RECOMMENDATION: This dataset has significant near-duplicate")
        print("content. Do NOT train with a random stratified split --")
        print("duplicates will leak across train/validation and inflate")
        print("accuracy artificially. Use group-aware splitting instead")
        print("(keep every image in a duplicate group on the SAME side")
        print("of the train/val split).")
        print("!" * 50)
    else:
        print("\nNo significant near-duplicates found. The 99-100% accuracy")
        print("may be a genuinely easy dataset (Roboflow-curated, studio-")
        print("quality images) rather than a leakage artifact -- but this")
        print("is unusual enough that I'd still treat it cautiously.")


if __name__ == "__main__":
    main()