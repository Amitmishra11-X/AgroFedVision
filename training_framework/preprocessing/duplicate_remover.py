import hashlib
import shutil
from pathlib import Path


class DuplicateRemover:

    def __init__(self, dataset_path):

        self.dataset = Path(dataset_path)

        self.clean_dataset = self.dataset.parent / (self.dataset.name + "_clean")

        self.hashes = {}

        self.unique = []

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

    def is_image(self, file):

        return file.suffix.lower() in self.extensions

    # --------------------------------------------------

    def file_hash(self, file):

        h = hashlib.sha256()

        with open(file, "rb") as f:

            while True:

                chunk = f.read(8192)

                if not chunk:

                    break

                h.update(chunk)

        return h.hexdigest()

    # --------------------------------------------------

    def scan(self):

        print()

        print("=" * 60)

        print("Scanning Dataset")

        print("=" * 60)

        duplicates = 0

        images = list(self.dataset.rglob("*"))

        for file in images:

            if not file.is_file():

                continue

            if not self.is_image(file):

                continue

            h = self.file_hash(file)

            if h in self.hashes:

                duplicates += 1

            else:

                self.hashes[h] = file

                self.unique.append(file)

        print("Total Images :", len(self.unique) + duplicates)

        print("Duplicates   :", duplicates)

        print("Unique       :", len(self.unique))

    # --------------------------------------------------

    def create_clean_dataset(self):

        if self.clean_dataset.exists():

            shutil.rmtree(self.clean_dataset)

        self.clean_dataset.mkdir(parents=True)

        print()

        print("Creating Clean Dataset...")

        for file in self.unique:

            relative = file.relative_to(self.dataset)

            dst = self.clean_dataset / relative

            dst.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy2(file, dst)

        print("Saved :", self.clean_dataset)

        return str(self.clean_dataset)

    # --------------------------------------------------

    def run(self):

        if self.clean_dataset.exists():

            print()

            print("=" * 60)

            print("Clean Dataset Already Exists")

            print(self.clean_dataset)

            return str(self.clean_dataset)

        self.scan()

        return self.create_clean_dataset()

# ======================================================

def remove_duplicates(dataset_path):

    return DuplicateRemover(dataset_path).run()