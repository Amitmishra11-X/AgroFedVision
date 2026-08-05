"""
============================================================
AgroFedVision Research Dataset Splitter
============================================================

Features

✓ Original dataset never modified
✓ Uses COPY instead of MOVE
✓ Split only once
✓ Resume support
✓ split_info.json
✓ Class statistics
✓ Reproducible
============================================================
"""

import json
import random
import shutil

from pathlib import Path


class DatasetSplitter:

    def __init__(

        self,

        train_dir,

        valid_dir=None,

        test_dir=None,

        valid_ratio=0.10,

        test_ratio=0.10,

        seed=42,

        overwrite=False

    ):

        self.train_dir = Path(train_dir)

        self.valid_dir = (
            Path(valid_dir)
            if valid_dir
            else self.train_dir.parent / "valid"
        )

        self.test_dir = (
            Path(test_dir)
            if test_dir
            else self.train_dir.parent / "test"
        )

        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.overwrite = overwrite

        random.seed(seed)

        self.extensions = {

            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tif",
            ".tiff",
            ".webp"

        }

        self.info_file = (
            self.train_dir.parent /
            "split_info.json"
        )
            # ======================================================
    # Check Existing Split
    # ======================================================

    def already_split(self):

        if not self.info_file.exists():
            return False

        try:

            with open(self.info_file, "r") as f:

                info = json.load(f)

            if (

                self.valid_dir.exists()

                and

                self.test_dir.exists()

                and

                info.get("completed", False)

            ):

                return True

        except Exception:

            return False

        return False


    # ======================================================
    # Save Split Information
    # ======================================================

    def save_info(

        self,

        train_images,

        valid_images,

        test_images

    ):

        info = {

            "completed": True,

            "seed": self.seed,

            "train_images": train_images,

            "valid_images": valid_images,

            "test_images": test_images,

            "valid_ratio": self.valid_ratio,

            "test_ratio": self.test_ratio

        }

        with open(

            self.info_file,

            "w"

        ) as f:

            json.dump(

                info,

                f,

                indent=4

            )


    # ======================================================
    # Split Dataset
    # ======================================================

    def split(self):

        # ----------------------------------------------
        # Skip if already split
        # ----------------------------------------------

        if self.already_split() and not self.overwrite:

            print()

            print("=" * 60)

            print("Dataset already split.")

            print("Skipping dataset splitter.")

            print("=" * 60)

            return


        # ----------------------------------------------
        # Overwrite existing split
        # ----------------------------------------------

        if self.overwrite:

            if self.valid_dir.exists():

                shutil.rmtree(self.valid_dir)

            if self.test_dir.exists():

                shutil.rmtree(self.test_dir)


        self.valid_dir.mkdir(

            parents=True,

            exist_ok=True

        )

        self.test_dir.mkdir(

            parents=True,

            exist_ok=True

        )

        print()

        print("=" * 60)

        print("Creating Validation/Test Dataset")

        print("=" * 60)
                # ----------------------------------------------
        # Statistics
        # ----------------------------------------------

        total_train = 0
        total_valid = 0
        total_test = 0

        # ----------------------------------------------
        # Split Every Class
        # ----------------------------------------------

        for class_folder in self.train_dir.iterdir():

            if not class_folder.is_dir():
                continue

            images = [

                img

                for img in class_folder.iterdir()

                if img.suffix.lower() in self.extensions

            ]

            random.shuffle(images)

            total = len(images)

            valid_count = int(total * self.valid_ratio)

            test_count = int(total * self.test_ratio)

            train_count = total - valid_count - test_count

            train_images = images[:train_count]

            valid_images = images[
                train_count:
                train_count + valid_count
            ]

            test_images = images[
                train_count + valid_count:
            ]

            # Create folders

            train_out = self.train_dir / class_folder.name

            valid_out = self.valid_dir / class_folder.name

            test_out = self.test_dir / class_folder.name

            valid_out.mkdir(
                parents=True,
                exist_ok=True
            )

            test_out.mkdir(
                parents=True,
                exist_ok=True
            )

            # ------------------------------------------
            # Copy Validation Images
            # ------------------------------------------

            for img in valid_images:

                shutil.copy2(

                    img,

                    valid_out / img.name

                )

            # ------------------------------------------
            # Copy Test Images
            # ------------------------------------------

            for img in test_images:

                shutil.copy2(

                    img,

                    test_out / img.name

                )

            # ------------------------------------------
            # Statistics
            # ------------------------------------------

            total_train += train_count
            total_valid += valid_count
            total_test += test_count

            print(

                f"{class_folder.name:25}"

                f" Train:{train_count:5}"

                f" Valid:{valid_count:5}"

                f" Test:{test_count:5}"

            )

        # ----------------------------------------------
        # Save Metadata
        # ----------------------------------------------

        self.save_info(

            total_train,

            total_valid,

            total_test

        )

        print()

        print("=" * 60)
        print("Dataset Split Completed")
        print("=" * 60)

        print(f"Train Images : {total_train}")
        print(f"Valid Images : {total_valid}")
        print(f"Test Images  : {total_test}")

        print()

        print("Train Folder :", self.train_dir)
        print("Valid Folder :", self.valid_dir)
        print("Test Folder  :", self.test_dir)

        print("=" * 60)


def split_dataset(

    train_dir,

    valid_dir=None,

    test_dir=None,

    valid_ratio=0.10,

    test_ratio=0.10,

    overwrite=False

):

    splitter = DatasetSplitter(

        train_dir=train_dir,

        valid_dir=valid_dir,

        test_dir=test_dir,

        valid_ratio=valid_ratio,

        test_ratio=test_ratio,

        overwrite=overwrite

    )

    splitter.split()       