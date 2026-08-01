import pandas as pd
import os

DATASET = "data/multimodal_dataset.csv"

df = pd.read_csv(DATASET)

print("Dataset Shape:", df.shape)
print(df["crop_health"].value_counts())

os.makedirs("data/clients", exist_ok=True)

healthy = df[df["crop_health"] == "Healthy"]
moderate = df[df["crop_health"] == "Moderate_Risk"]
high = df[df["crop_health"] == "High_Risk"]

# Farm A → mostly healthy
farm_a = pd.concat([
    healthy.sample(min(150, len(healthy)), random_state=42),
    moderate.sample(min(30, len(moderate)), random_state=42)
])

# Farm B → mostly moderate
farm_b = pd.concat([
    moderate.sample(min(150, len(moderate)), random_state=1),
    healthy.sample(min(30, len(healthy)), random_state=1)
])

# Farm C → mostly high risk
farm_c = pd.concat([
    high.sample(min(200, len(high)), random_state=10),
    moderate.sample(min(30, len(moderate)), random_state=10)
])

# Farm D → mixed
remaining = df.drop(
    farm_a.index.union(
        farm_b.index.union(
            farm_c.index
        )
    ),
    errors="ignore"
)

farm_d = remaining

farm_a.to_csv("data/clients/farm_a.csv", index=False)
farm_b.to_csv("data/clients/farm_b.csv", index=False)
farm_c.to_csv("data/clients/farm_c.csv", index=False)
farm_d.to_csv("data/clients/farm_d.csv", index=False)

print("\nSaved:")
print("farm_a.csv", farm_a.shape)
print("farm_b.csv", farm_b.shape)
print("farm_c.csv", farm_c.shape)
print("farm_d.csv", farm_d.shape)
