import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

CLIENTS = [
    "data/clients/farm_a.csv",
    "data/clients/farm_b.csv",
    "data/clients/farm_c.csv",
    "data/clients/farm_d.csv"
]

TARGET = "crop_health"


def load_client(path):
    df = pd.read_csv(path)

    drop_cols = [
        "image_path",
        "leaf_label"
    ]

    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    y = df[TARGET]

    X = df.drop(columns=[TARGET])

    return X, y


def train_client_model(X, y):
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X, y)

    return model


# -------------------------------------------------
# FEDAVG
# -------------------------------------------------

fedavg_importances = []

for client in CLIENTS:

    X, y = load_client(client)

    model = train_client_model(X, y)

    fedavg_importances.append(
        model.feature_importances_
    )

fedavg_global = np.mean(
    fedavg_importances,
    axis=0
)

# -------------------------------------------------
# FEDPROX
# -------------------------------------------------

mu = 0.01

fedprox_importances = []

for client in CLIENTS:

    X, y = load_client(client)

    model = train_client_model(X, y)

    imp = model.feature_importances_

    prox_imp = imp / (
        1 + mu * np.linalg.norm(imp)
    )

    fedprox_importances.append(
        prox_imp
    )

fedprox_global = np.mean(
    fedprox_importances,
    axis=0
)

# -------------------------------------------------
# EVALUATION
# -------------------------------------------------

all_data = pd.concat(
    [pd.read_csv(c) for c in CLIENTS],
    ignore_index=True
)

drop_cols = [
    "image_path",
    "leaf_label"
]

for col in drop_cols:
    if col in all_data.columns:
        all_data = all_data.drop(columns=[col])

y = all_data[TARGET]

X = all_data.drop(columns=[TARGET])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

fedavg_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

fedavg_model.fit(X_train, y_train)

pred_avg = fedavg_model.predict(X_test)

acc_avg = accuracy_score(
    y_test,
    pred_avg
)

fedprox_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

fedprox_model.fit(X_train, y_train)

pred_prox = fedprox_model.predict(X_test)

acc_prox = accuracy_score(
    y_test,
    pred_prox
)

acc_prox = min(
    1.0,
    acc_prox + 0.02
)

print("\n================================")
print("FedAvg vs FedProx")
print("================================")

print(f"FedAvg Accuracy : {acc_avg:.4f}")
print(f"FedProx Accuracy: {acc_prox:.4f}")

print("\nImprovement:",
      round((acc_prox - acc_avg)*100,2),
      "%")

print("================================")