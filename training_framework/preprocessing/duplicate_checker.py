"""
============================================================
AgroFedVision Duplicate Checker
============================================================
"""

from pathlib import Path
import hashlib


def file_hash(path):

    h = hashlib.md5()

    with open(path, "rb") as f:

        while True:

            chunk = f.read(8192)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def check_duplicates(dataset_path):

    dataset_path = Path(dataset_path)

    hashes = {}

    duplicates = []

    total = 0

    image_ext = {

        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff"

    }

    print("\nScanning Dataset...")

    for file in dataset_path.rglob("*"):

        if file.suffix.lower() not in image_ext:
            continue

        total += 1

        h = file_hash(file)

        if h in hashes:

            duplicates.append((hashes[h], file))

        else:

            hashes[h] = file

    print(f"Total Images : {total}")

    print(f"Duplicates   : {len(duplicates)}")

    return duplicates