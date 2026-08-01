import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("uav_features.csv")

plt.hist(df["NDVI_Mean"], bins=30)
plt.title("NDVI Distribution")
plt.show()

plt.hist(df["NDRE_Mean"], bins=30)
plt.title("NDRE Distribution")
plt.show()