import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def train_client(path):
    df = pd.read_csv(path)

    X = df.drop("label", axis=1)
    y = df["label"]

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X, y)

    return model