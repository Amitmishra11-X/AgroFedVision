from pathlib import Path


def verify_dataset(dataset_path):

    dataset_path = Path(dataset_path)

    if not dataset_path.exists():

        raise FileNotFoundError(dataset_path)

    image_count = 0

    class_count = 0

    image_ext = {

        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff"

    }

    for folder in dataset_path.iterdir():

        if folder.is_dir():

            class_count += 1

            image_count += len(

                [

                    x for x in folder.iterdir()

                    if x.suffix.lower() in image_ext

                ]

            )

    print()

    print("="*60)

    print("Dataset Verification")

    print("="*60)

    print("Classes :", class_count)

    print("Images  :", image_count)

    print("="*60)

    return {

        "classes": class_count,

        "images": image_count

    }