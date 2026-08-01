import pandas as pd

# Load dataset
df = pd.read_csv("data/leaf_dataset.csv")

# Label mapping
label_map = {

    # Healthy
    "Healthy_brinjal": "Healthy",
    "Healthy_Guava": "Healthy",
    "Healthy_castor": "Healthy",
    "Healthy_Cumin": "Healthy",
    "Healthy_papaya": "Healthy",

    # Unhealthy
    "Unhealthy_brinjal": "Unhealthy",
    "Unhealthy_castor": "Unhealthy",
    "Unhealthy_Guava": "Unhealthy",
    "Unhealthy_papaya": "Unhealthy",

    # Nitrogen
    "Nitrogen": "Nitrogen_Deficiency",

    # Potassium
    "Pottasium": "Potassium_Deficiency",
    "Pottassium": "Potassium_Deficiency",
    "pottasium": "Potassium_Deficiency",

    # Magnesium
    "Magnessium": "Magnesium_Deficiency",
    "Magnesium": "Magnesium_Deficiency",
    "magnesiuam": "Magnesium_Deficiency",

    # Phosphorus
    "phosphorus": "Phosphorus_Deficiency",

    # Pest Damage
    "pest_damage": "Pest_Damage",
    "Peast_disease": "Pest_Damage",
    "insect_infection": "Pest_Damage",

    # Alternaria
    "Alternia Leaf Blight": "Alternaria_Blight",
    "Alternatia Blight": "Alternaria_Blight",

    # Anthracnose
    "anthrecnose": "Anthracnose",
    "Anthracnose _ Cercospora": "Anthracnose",

    # Leaf Curl
    "Leaf_curl": "Leaf_Curl",
    "Leaf_curl_viral_diseases": "Leaf_Curl",

    # Bacterial Disease
    "Bacterial Leaf Blight": "Bacterial_Disease",
    "Bacterial Leaf Spot": "Bacterial_Disease"
}

# Apply mapping
df["label"] = df["label"].replace(label_map)

# Remove very small classes (<10 samples)
counts = df["label"].value_counts()

valid_classes = counts[counts >= 10].index

df = df[df["label"].isin(valid_classes)]

# Save cleaned dataset
df.to_csv(
    "data/leaf_dataset_clean.csv",
    index=False
)

print("Saved: data/leaf_dataset_clean.csv")
print("\nClass Distribution:\n")
print(df["label"].value_counts())
print("\nTotal Images:", len(df))