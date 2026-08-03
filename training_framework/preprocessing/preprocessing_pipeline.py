from preprocessing.verification import verify_dataset
from preprocessing.duplicate_checker import check_duplicates


def run_preprocessing(dataset):

    print()

    print("="*60)

    print("Preprocessing Pipeline")

    print("="*60)

    verify_dataset(dataset)

    duplicates = check_duplicates(dataset)

    print()

    print("="*60)

    print("Summary")

    print("="*60)

    print("Duplicates :", len(duplicates))

    print("="*60)

    return duplicates