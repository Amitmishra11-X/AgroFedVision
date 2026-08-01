from agrofedvision.preprocessing.leaf.builder import LeafDatasetBuilder

builder = LeafDatasetBuilder(
    dataset_root="data/raw/leaf_images",
    output_csv="data/processed/leaf_dataset.csv",
)

df = builder.run()