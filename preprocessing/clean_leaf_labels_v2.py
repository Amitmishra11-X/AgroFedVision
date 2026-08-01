

import pandas as pd

df = pd.read_csv("data/leaf_dataset_clean.csv")

# Coarse grouping — biological/agronomic category, not exact disease name
coarse_map = {
    "Healthy": "Healthy",

    # Nutrient deficiencies — all grouped
    "Nitrogen_Deficiency":   "Nutrient_Deficiency",
    "Phosphorus_Deficiency": "Nutrient_Deficiency",
    "Potassium_Deficiency":  "Nutrient_Deficiency",
    "Magnesium_Deficiency":  "Nutrient_Deficiency",

    # Fungal diseases — all grouped
    "Alternaria_Blight":     "Fungal_Disease",
    "Anthracnose":           "Fungal_Disease",
    "Cercospera Leaf Spot":  "Fungal_Disease",
    "Cercospora":            "Fungal_Disease",
    "Early_Leaf_spot":       "Fungal_Disease",
    "Fungal infection":      "Fungal_Disease",
    "Powdery_Mildew":        "Fungal_Disease",
    "Rust_fungal":           "Fungal_Disease",
    "algal_leaf_spot":       "Fungal_Disease",
    "blackspot":             "Fungal_Disease",

    # Bacterial/viral — all grouped
    "Bacterial_Disease":     "Bacterial_Viral_Disease",
    "Leaf_Curl":             "Bacterial_Viral_Disease",
    "Mosaic":                "Bacterial_Viral_Disease",
    "Little_leaf":           "Bacterial_Viral_Disease",
    "Wilt":                  "Bacterial_Viral_Disease",

    # Pest damage stays separate (distinct visual signature)
    "Pest_Damage":           "Pest_Damage",
}

df["label"] = df["label"].map(coarse_map)

# Drop anything that didn't map (shouldn't happen, but safe)
df = df.dropna(subset=["label"])

df.to_csv("data/leaf_dataset_coarse.csv", index=False)

print("Saved: data/leaf_dataset_coarse.csv\n")
print("New Class Distribution:\n")
print(df["label"].value_counts())
print("\nTotal Images:", len(df))
print("Total Classes:", df["label"].nunique())
