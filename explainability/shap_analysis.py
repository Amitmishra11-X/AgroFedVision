import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data/Crop_recommendationV2.csv")

X = df.drop("label", axis=1)
y = df["label"]

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X, y)

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X[:200])

shap.summary_plot(
    shap_values,
    X[:200]
)