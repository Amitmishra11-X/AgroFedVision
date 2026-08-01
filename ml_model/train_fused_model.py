from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

df = pd.read_csv("data/fused_dataset.csv")

X = df.drop("label", axis=1)
y = df["label"]

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

scores = cross_val_score(
    model,
    X,
    y,
    cv=5
)

print(scores)
print("Mean:", scores.mean())