import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data/Crop_recommendationV2.csv")

X = df.drop("label", axis=1)
y = df["label"]

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X, y)

importance = model.feature_importances_

feat = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

feat = feat.sort_values(
    by="Importance",
    ascending=False
)

print(feat)

plt.figure(figsize=(10,6))
plt.barh(feat["Feature"], feat["Importance"])
plt.title("Feature Importance")
plt.tight_layout()
plt.show()