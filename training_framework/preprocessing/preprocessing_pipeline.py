from config import *

from preprocessing.verification import verify_dataset
from preprocessing.duplicate_remover import remove_duplicates
from preprocessing.dataset_splitter import split_dataset


def run_preprocessing(dataset):

    print()

    print("=" * 60)
    print("Preprocessing Pipeline")
    print("=" * 60)

    verify_dataset(dataset)

    clean_dataset = remove_duplicates(dataset)
    split_dataset(
        clean_dataset,
        valid_ratio=VALIDATION_SPLIT,
        test_ratio=TEST_SPLIT
)

    print()

    print("=" * 60)
    print("Using Clean Dataset")
    print("=" * 60)
    print(clean_dataset)
    print("=" * 60)

    return clean_dataset