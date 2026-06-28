import json
import yaml
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

PARAMS_PATH = "params.yaml"
INPUT_PATH = "data/iris_prepared.csv"
MODEL_PATH = "models/model.pkl"
METRICS_PATH = "metrics.json"


def load_params():
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)["train"]


def main():
    params = load_params()
    df = pd.read_csv(INPUT_PATH)

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=params["test_size"],
        random_state=params["random_state"],
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        random_state=params["random_state"],
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")

    joblib.dump(model, MODEL_PATH)

    with open(METRICS_PATH, "w") as f:
        json.dump({"accuracy": acc, "f1_macro": f1}, f, indent=2)

    print(f"Trained model -> {MODEL_PATH}")
    print(f"Metrics: accuracy={acc:.4f}, f1_macro={f1:.4f}")


if __name__ == "__main__":
    main()
