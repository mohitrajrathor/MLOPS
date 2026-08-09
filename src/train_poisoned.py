import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Set random seed for reproducibility
SEED = 42


def train_and_log(csv_path: Path, poison_level: int) -> dict:
    """Train IRIS classifier on given dataset and log metrics to MLflow."""
    df = pd.read_csv(csv_path)
    df = df.dropna()

    # Preprocessing
    feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    X_raw = df[feature_cols].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    le = LabelEncoder()
    y = le.fit_transform(df['species'])

    # Train-test split with fixed seed for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    with mlflow.start_run(run_name=f"poison_level_{poison_level}"):
        # Log parameters
        mlflow.log_param("poison_level", poison_level)
        mlflow.log_param("max_depth", 3)
        mlflow.log_param("min_samples_split", 2)

        # Same model architecture as src/train.py
        model = DecisionTreeClassifier(max_depth=3, min_samples_split=2, random_state=SEED)
        model.fit(X_train, y_train)

        # Evaluation metrics
        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, average="weighted", zero_division=0)
        recall = recall_score(y_test, preds, average="weighted", zero_division=0)
        f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

        # Log metrics to MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

    return {
        "poison_level": poison_level,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }


def main() -> None:
    # Set MLflow experiment
    experiment_name = "week_8_poision_exp"
    mlflow.set_experiment(experiment_name)

    datasets = [
        (0, RAW_DATA_DIR / "iris.csv"),
        (5, RAW_DATA_DIR / "iris_poisoned_5.csv"),
        (10, RAW_DATA_DIR / "iris_poisoned_10.csv"),
        (50, RAW_DATA_DIR / "iris_poisoned_50.csv"),
    ]

    results = []
    for poison_level, csv_path in datasets:
        if not csv_path.exists():
            print(f"Error: Dataset file not found at {csv_path}. Please run src/poison.py first.")
            return
        
        print(f"Training model on dataset with poison_level={poison_level}%...")
        res = train_and_log(csv_path, poison_level)
        results.append(res)

    # Print summary table of all 4 runs
    print("\n" + "=" * 65)
    print(f"SUMMARY TABLE (MLflow Experiment: '{experiment_name}')")
    print("=" * 65)
    print(f"{'Poison Level (%)':<18} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10}")
    print("-" * 65)
    for r in results:
        print(f"{r['poison_level']:<18} | {r['accuracy']:<10.4f} | {r['precision']:<10.4f} | {r['recall']:<10.4f} | {r['f1_score']:<10.4f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
