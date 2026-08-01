import pandas as pd
import os

base_dir = os.path.dirname(os.path.dirname(__file__))

csv_path = os.path.join(
    base_dir,
    "data",
    "normalized_dataset.csv"
)

df = pd.read_csv(csv_path)

# Farm A
farm_a = df[df["label"].isin([0,1,2,3,4])]

# Farm B
farm_b = df[df["label"].isin([5,6,7,8,9])]

# Farm C
farm_c = df[df["label"].isin([10,11,12,13,14])]

# Farm D
farm_d = df[df["label"].isin([15,16,17,18,19,20,21])]

client_dir = os.path.join(
    base_dir,
    "data",
    "clients"
)

os.makedirs(client_dir,exist_ok=True)

farm_a.to_csv(
    os.path.join(client_dir,"farm_a.csv"),
    index=False
)

farm_b.to_csv(
    os.path.join(client_dir,"farm_b.csv"),
    index=False
)

farm_c.to_csv(
    os.path.join(client_dir,"farm_c.csv"),
    index=False
)

farm_d.to_csv(
    os.path.join(client_dir,"farm_d.csv"),
    index=False
)

print("Client datasets created")

print("Farm A:",farm_a.shape)
print("Farm B:",farm_b.shape)
print("Farm C:",farm_c.shape)
print("Farm D:",farm_d.shape)