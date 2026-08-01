import pandas as pd

sensor = pd.read_csv("data/Crop_recommendationV2.csv")
uav = pd.read_csv("uav_features.csv")

uav_small = uav[["NDVI_Mean","NDVI_Std","NDRE_Mean","NDRE_Std"]]

sensor = sensor.iloc[:302].reset_index(drop=True)

fused = pd.concat([sensor, uav_small], axis=1)

fused.to_csv("data/fused_dataset.csv", index=False)

print(fused.shape)
print(fused.head())