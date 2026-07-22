# train.py
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import argparse
import os
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def train(max_depth: int, min_samples_split: int, experiment_name: str) -> None:
    """Train a DecisionTree model and log everything to MLflow.

    Args:
        max_depth: Max depth of the decision tree.
        min_samples_split: Min samples required to split a node.
        experiment_name: MLflow experiment name.
    """
    # load prepared data
    data_dir = PROJECT_ROOT / "data" / "processed"
    try:
        X = np.load(data_dir / "prepared_data.npy")
        y = np.load(data_dir / "prepared_target.npy")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Prepared data not found: {e}") from e

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        # log hyperparams
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)

        # train
        model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
        model.fit(X_train, y_train)

        # evaluate
        preds = model.predict(X_test)
        accuracy  = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, average="weighted")
        f1        = f1_score(y_test, preds, average="weighted")

        # log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("f1_score", f1)

        # save model locally with joblib
        artifacts_dir = PROJECT_ROOT / "artifacts"
        os.makedirs(artifacts_dir, exist_ok=True)
        model_path = artifacts_dir / "model.joblib"
        joblib.dump(model, model_path)

        # log model to MLflow registry
        mlflow.sklearn.log_model(
            sk_model=model,
            name="decision_tree",
            registered_model_name="iris_decision_tree"
        )

        print(f"accuracy={accuracy:.4f}  precision={precision:.4f}  f1={f1:.4f}")
        print("Run logged to MLflow.")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train DecisionTree on IRIS data with MLflow tracking.")
    parser.add_argument("--max-depth",          type=int, default=3,            help="Max tree depth")
    parser.add_argument("--min-samples-split",  type=int, default=2,            help="Min samples to split a node")
    parser.add_argument("--experiment-name",    type=str, default="iris_experiment", help="MLflow experiment name")
    args = parser.parse_args()

    train(
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        experiment_name=args.experiment_name,
    )


if __name__ == "__main__":
    main()
