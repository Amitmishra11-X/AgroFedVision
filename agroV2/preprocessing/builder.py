"""
Leaf Dataset Builder

Purpose
-------
Scans the leaf image dataset and creates a CSV
containing image paths and class labels.

Author:
AgroFedVision v1.5
"""

from pathlib import Path
import logging
import pandas as pd


class LeafDatasetBuilder:
    """
    Build a dataset CSV from a directory
    containing crop disease images.
    """

    SUPPORTED_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
    }

    def __init__(self, dataset_root, output_csv):

        self.dataset_root = Path(dataset_root)

        self.output_csv = Path(output_csv)

        self.logger = logging.getLogger(__name__)

        self.records = []

    # --------------------------------------------------

    def scan(self):

        """
        Scan every class folder.
        """

        self.logger.info("Scanning dataset...")

        if not self.dataset_root.exists():

            raise FileNotFoundError(

                f"{self.dataset_root} does not exist."

            )

        self.records.clear()

        for folder in sorted(self.dataset_root.iterdir()):

            if not folder.is_dir():

                continue

            label = folder.name

            for image in folder.rglob("*"):

                if image.suffix.lower() not in self.SUPPORTED_EXTENSIONS:

                    continue

                self.records.append(

                    {

                        "image_path": str(image.resolve()),

                        "label": label,

                    }

                )

        self.logger.info(

            "Found %d images",

            len(self.records)

        )

    # --------------------------------------------------

    def create_dataframe(self):

        """
        Convert records into DataFrame.
        """

        return pd.DataFrame(

            self.records,

            columns=[

                "image_path",

                "label",

            ],

        )

    # --------------------------------------------------

    def validate(self, dataframe):

        """
        Basic validation.
        """

        if dataframe.empty:

            raise ValueError(

                "Dataset is empty."

            )

        if dataframe["image_path"].duplicated().any():

            self.logger.warning(

                "Duplicate image paths detected."

            )

        return dataframe

    # --------------------------------------------------

    def save(self, dataframe):

        """
        Save CSV.
        """

        self.output_csv.parent.mkdir(

            parents=True,

            exist_ok=True,

        )

        dataframe.to_csv(

            self.output_csv,

            index=False,

        )

        self.logger.info(

            "Saved dataset to %s",

            self.output_csv,

        )

    # --------------------------------------------------

    def summary(self, dataframe):

        """
        Print summary.
        """

        self.logger.info("")

        self.logger.info("Dataset Summary")

        self.logger.info("-------------------------")

        self.logger.info(

            "Images : %d",

            len(dataframe),

        )

        self.logger.info(

            "Classes : %d",

            dataframe["label"].nunique(),

        )

        self.logger.info("")

        self.logger.info(

            dataframe["label"].value_counts()

        )

    # --------------------------------------------------

    def run(self):

        self.scan()

        df = self.create_dataframe()

        df = self.validate(df)

        self.save(df)

        self.summary(df)

        return df