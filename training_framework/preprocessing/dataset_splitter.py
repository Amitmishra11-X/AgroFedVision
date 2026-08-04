"""
============================================================
AgroFedVision Universal Dataset Splitter
============================================================
"""

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

        seed=42

    ):

        self.train_dir = Path(train_dir)

        self.valid_dir = Path(valid_dir) if valid_dir else self.train_dir.parent / "valid"

        self.test_dir = Path(test_dir) if test_dir else self.train_dir.parent / "test"

        self.valid_ratio = valid_ratio

        self.test_ratio = test_ratio

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

    # --------------------------------------------------

    def split(self):

        if self.valid_dir.exists():

            shutil.rmtree(self.valid_dir)

        if self.test_dir.exists():

            shutil.rmtree(self.test_dir)

        self.valid_dir.mkdir(parents=True, exist_ok=True)

        self.test_dir.mkdir(parents=True, exist_ok=True)

        print()

        print("=" * 60)

        print("Creating Validation/Test Dataset")

        print("=" * 60)

        self.valid_dir.mkdir(parents=True, exist_ok=True)

        self.test_dir.mkdir(parents=True, exist_ok=True)

        for class_folder in self.train_dir.iterdir():

            if not class_folder.is_dir():

                continue

            images = [ 

                x for x in class_folder.iterdir()

                if x.suffix.lower() in self.extensions

            ]

            random.shuffle(images)

            total = len(images)

            valid_count = int(total * self.valid_ratio)

            test_count = int(total * self.test_ratio)

            valid_images = images[:valid_count]

            test_images = images[valid_count:valid_count + test_count]

            (self.valid_dir / class_folder.name).mkdir(

                parents=True,

                exist_ok=True

            )

            (self.test_dir / class_folder.name).mkdir(

                parents=True,

                exist_ok=True

            )

            for img in valid_images:

                shutil.move(

                    str(img),

                    str(self.valid_dir / class_folder.name / img.name)

                )

            for img in test_images:

                shutil.move(

                    str(img),

                    str(self.test_dir / class_folder.name / img.name)

                )

        print()

        print("=" * 60)

        print("Dataset Split Completed")

        print("=" * 60)

        print("Train :", self.train_dir)

        print("Valid :", self.valid_dir)

        print("Test  :", self.test_dir)

        print("=" * 60)


def split_dataset(

    train_dir,

    valid_dir=None,

    test_dir=None,

    valid_ratio=0.10,

    test_ratio=0.10

):

    DatasetSplitter(

        train_dir,

        valid_dir,

        test_dir,

        valid_ratio,

        test_ratio

    ).split()