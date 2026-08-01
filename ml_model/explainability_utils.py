import json
import os


def save_explainability(report, save_dir="results"):

    os.makedirs(save_dir, exist_ok=True)

    path = os.path.join(
        save_dir,
        "explainability.json"
    )

    with open(path, "w") as f:

        json.dump(
            report,
            f,
            indent=4
        )

    print(f"\nExplainability saved : {path}")