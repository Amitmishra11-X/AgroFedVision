
"""""
Aligns three separate datasets into a single row-per-sample fusion dataset.
Each row contains: leaf image path + sensor features + UAV features + health label.

Since the three datasets are from different sources (not field-synchronized),
we use a principled synthetic alignment strategy:
  - The single ground truth label is derived from LEAF disease class
    (most granular signal we have)
  - Sensor values are randomly sampled from the sensor CSV
  - UAV features are randomly sampled from the UAV feature CSV
  - The crop_health label is mapped from leaf disease class

This is scientifically honest for a prototype/demo:
  - It is clearly declared as synthetic alignment in the paper
  - The model architecture and federated pipeline are the real contributions
  - Real synchronized field data would replace this CSV, the model stays the same

"""

import pandas as pd
import numpy as np
import os

SENSOR_CSV    = "data/Crop_recommendationV2.csv"
UAV_CSV       = "data/fused_dataset.csv"
LEAF_CSV      = "data/leaf_dataset_clean.csv"
OUTPUT_CSV    = "data/multimodal_dataset.csv"
RANDOM_SEED   = 42

# ── Crop health label mapping ──────────────────────────────────────────
# Maps leaf disease class → single unified crop health label
# This is the "one output" your professor asked for
HEALTH_MAP = {
    "Healthy":                  "Healthy",
    "Potassium_Deficiency":     "Moderate_Risk",
    "Magnesium_Deficiency":     "Moderate_Risk",
    "Nitrogen_Deficiency":      "Moderate_Risk",
    "Phosphorus_Deficiency":    "Moderate_Risk",
    "Alternaria_Blight":        "High_Risk",
    "Anthracnose":              "High_Risk",
    "Bacterial_Disease":        "High_Risk",
    "Cercospera Leaf Spot":     "High_Risk",
    "Cercospora":               "High_Risk",
    "Early_Leaf_spot":          "High_Risk",
    "Fungal infection":         "High_Risk",
    "Powdery_Mildew":           "High_Risk",
    "Rust_fungal":              "High_Risk",
    "algal_leaf_spot":          "High_Risk",
    "blackspot":                "High_Risk",
    "Mosaic":                   "High_Risk",
    "Little_leaf":              "High_Risk",
    "Leaf_Curl":                "High_Risk",
    "Wilt":                     "High_Risk",
    "Pest_Damage":              "Moderate_Risk",
}


def load_datasets():
    print("[Load] Reading datasets...")
    sensor = pd.read_csv(SENSOR_CSV)
    leaf   = pd.read_csv(LEAF_CSV)

    # UAV CSV — check what columns are actually present
    if os.path.exists(UAV_CSV):
        uav = pd.read_csv(UAV_CSV)
        print(f"[Load] UAV CSV columns: {list(uav.columns)}")
    else:
        print(f"[Warning] UAV CSV not found at {UAV_CSV}. Generating synthetic UAV features.")
        uav = None

    print(f"[Load] Sensor: {len(sensor)} rows | Leaf: {len(leaf)} rows")
    return sensor, leaf, uav


def make_fusion_dataset(sensor, leaf, uav):
    rng = np.random.default_rng(RANDOM_SEED)
    n   = len(leaf)

    # ── Sensor features ────────────────────────────────────────────────
    sensor_cols = [c for c in sensor.columns if c.lower() != "label"]
    sensor_sample = sensor.sample(n=n, replace=True, random_state=RANDOM_SEED) \
                          .reset_index(drop=True)

    # ── UAV features ───────────────────────────────────────────────────
    uav_cols = ["NDVI_Mean", "NDVI_Std", "NDRE_Mean", "NDRE_Std"]

    if uav is not None:
        # Keep only the 4 key columns (handle case differences)
        available_uav_cols = {c.lower(): c for c in uav.columns}
        actual_uav_cols = []
        for expected in uav_cols:
            match = available_uav_cols.get(expected.lower())
            if match:
                actual_uav_cols.append(match)

        if len(actual_uav_cols) >= 2:
            uav_sample = uav[actual_uav_cols].sample(n=n, replace=True, random_state=RANDOM_SEED) \
                                             .reset_index(drop=True)
            uav_sample.columns = [c for c in uav_cols[:len(actual_uav_cols)]]
        else:
            print(f"[Warning] Could not match UAV columns. Generating synthetic.")
            uav_sample = _synthetic_uav(n, rng)
    else:
        uav_sample = _synthetic_uav(n, rng)

    # ── Health label from leaf disease ─────────────────────────────────
    leaf_reset = leaf.reset_index(drop=True)

    def map_health(raw_label):
        # leaf_dataset_clean.csv may already have merged labels
        # Try direct map first, then try prefix matching
        if raw_label in HEALTH_MAP:
            return HEALTH_MAP[raw_label]
        for key in HEALTH_MAP:
            if key.lower() in raw_label.lower():
                return HEALTH_MAP[key]
        # Default: anything with disease keywords → High_Risk
        disease_kws = ["disease", "blight", "spot", "rust", "wilt",
                       "mildew", "fungal", "mosaic", "curl", "pest"]
        if any(k in raw_label.lower() for k in disease_kws):
            return "High_Risk"
        deficiency_kws = ["deficiency", "nitrogen", "potassium", "magnesium", "phosphorus"]
        if any(k in raw_label.lower() for k in deficiency_kws):
            return "Moderate_Risk"
        return "Healthy"

    health_labels = leaf_reset["label"].apply(map_health)

    # ── Assemble fusion dataframe ──────────────────────────────────────
    fusion = pd.DataFrame()
    fusion["image_path"] = leaf_reset["image_path"]
    fusion["leaf_label"] = leaf_reset["label"]        # keep original for reference

    for col in sensor_cols:
        fusion[col] = sensor_sample[col].values

    for col in uav_sample.columns:
        fusion[col] = uav_sample[col].values

    fusion["crop_health"] = health_labels             # THE single output label

    fusion.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[Done] Saved {len(fusion)} rows to {OUTPUT_CSV}")
    print(f"Columns ({len(fusion.columns)}): {list(fusion.columns)}")
    print(f"\nLabel distribution:")
    print(fusion["crop_health"].value_counts())
    return fusion


def _synthetic_uav(n, rng):
    """Generate plausible synthetic NDVI/NDRE values when UAV CSV is absent."""
    return pd.DataFrame({
        "NDVI_Mean": rng.uniform(0.10, 0.45, n).round(4),
        "NDVI_Std":  rng.uniform(0.13, 0.28, n).round(4),
        "NDRE_Mean": rng.uniform(0.05, 0.37, n).round(4),
        "NDRE_Std":  rng.uniform(0.12, 0.24, n).round(4),
    })


if __name__ == "__main__":
    sensor, leaf, uav = load_datasets()
    fusion = make_fusion_dataset(sensor, leaf, uav)
    print("\nFirst 3 rows:")
    print(fusion.head(3).to_string())
