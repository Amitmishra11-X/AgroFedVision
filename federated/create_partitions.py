import pandas as pd

from federated.partition import FederatedPartitioner

# Load your dataset
df = pd.read_csv("data/paddy_dataset.csv")

# Create partitioner
partitioner = FederatedPartitioner(
    dataframe=df,
    label_column="crop_health",
    output_dir="data/clients/paddy"
)

# Create 4 non-IID clients
partitioner.dirichlet_partition(
    alpha=0.3,
    num_clients=4
)

print("Paddy partitions created successfully!")