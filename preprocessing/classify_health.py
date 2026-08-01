import pandas as pd

df = pd.read_csv("uav_features.csv")

def health_status(ndvi):
    if ndvi > 0.5:
        return "Healthy"
    elif ndvi > 0.2:
        return "Moderate"
    else:
        return "Stress"

df["Health"] = df["NDVI_Mean"].apply(health_status)

print(df["Health"].value_counts())

df.to_csv("uav_health_labels.csv", index=False)