import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data/fused_dataset.csv")

X = df.drop("label", axis=1)
y = df["label"]
#model = LogisticRegression(max_iter=1000, random_state=42) ration
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X, y)

explainer = shap.TreeExplainer(model)

sample = X.sample(
    min(200, len(X)),
    random_state=42
)

shap_values = explainer.shap_values(sample)

shap.summary_plot(
    shap_values,
    sample,
    plot_type="bar"
)