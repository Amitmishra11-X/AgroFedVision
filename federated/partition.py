"""
AgroFedVision Federated Dataset Partitioner

Supports

✓ IID
✓ Stratified IID
✓ Dirichlet Non-IID
✓ Quantity Skew

Works for

✓ Paddy
✓ Maize
✓ Guava
✓ Image
✓ UAV
✓ Sensor
"""

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold


class FederatedPartitioner:

    def __init__(self,
                 dataframe,
                 label_column,
                 output_dir):

        self.df = dataframe.copy()

        self.label_column = label_column

        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    ########################################################
    # IID
    ########################################################

    def iid_partition(
            self,
            num_clients=4,
            seed=42):

        np.random.seed(seed)

        shuffled = self.df.sample(
            frac=1,
            random_state=seed
        ).reset_index(drop=True)

        splits = np.array_split(
            shuffled,
            num_clients
        )

        self._save_clients(splits)

        return splits

    ########################################################
    # Stratified
    ########################################################

    def stratified_partition(
            self,
            num_clients=4,
            seed=42):

        skf = StratifiedKFold(
            n_splits=num_clients,
            shuffle=True,
            random_state=seed
        )

        splits = []

        X = np.zeros(len(self.df))

        y = self.df[self.label_column]

        for _, test_idx in skf.split(X, y):

            splits.append(
                self.df.iloc[test_idx]
            )

        self._save_clients(splits)

        return splits

    ########################################################
    # Quantity Skew
    ########################################################

    def quantity_skew(
            self,
            proportions):

        assert abs(sum(proportions)-1) < 1e-5

        shuffled = self.df.sample(
            frac=1
        )

        n = len(shuffled)

        splits = []

        start = 0

        for p in proportions:

            end = start + int(n*p)

            splits.append(
                shuffled.iloc[start:end]
            )

            start = end

        self._save_clients(splits)

        return splits

    ########################################################
    # Dirichlet
    ########################################################

    def dirichlet_partition(
            self,
            alpha=0.5,
            num_clients=4,
            seed=42):

        np.random.seed(seed)

        classes = self.df[
            self.label_column
        ].unique()

        client_indices = [
            [] for _ in range(num_clients)
        ]

        for cls in classes:

            cls_idx = self.df[
                self.df[self.label_column] == cls
            ].index.values

            np.random.shuffle(cls_idx)

            proportions = np.random.dirichlet(
                alpha*np.ones(num_clients)
            )

            proportions = (
                np.cumsum(proportions)
                * len(cls_idx)
            ).astype(int)[:-1]

            split = np.split(
                cls_idx,
                proportions
            )

            for client, idx in zip(
                    client_indices,
                    split):

                client.extend(idx)

        splits = []

        for idx in client_indices:

            splits.append(
                self.df.loc[idx]
            )

        self._save_clients(splits)

        return splits

    ########################################################
    # Save
    ########################################################

    def _save_clients(
            self,
            splits):

        for i, part in enumerate(splits):

            folder = self.output_dir / f"farm_{i+1}"

            folder.mkdir(
                exist_ok=True
            )

            part.to_csv(
                folder/"metadata.csv",
                index=False
            )

            print(
                f"Farm {i+1}: {len(part)} samples"
            )