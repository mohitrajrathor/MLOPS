import pandas as pd
import numpy as np
from pathlib import Path

# Set project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
CLEAN_DATA_PATH = RAW_DATA_DIR / "iris.csv"


def poison_dataset(df: pd.DataFrame, corruption_percent: int) -> pd.DataFrame:
    """Poison a given percentage of dataset samples with random features and class labels."""
    df_poisoned = df.copy()
    num_samples = len(df_poisoned)
    num_corrupt = int(num_samples * (corruption_percent / 100.0))

    # Truly random selection of row indices to simulate attacker behavior
    corrupt_indices = np.random.choice(num_samples, size=num_corrupt, replace=False)

    feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    species_labels = list(df['species'].unique())

    # Range of features from baseline dataset
    min_vals = df[feature_cols].min()
    max_vals = df[feature_cols].max()

    for idx in corrupt_indices:
        # Replace all 4 features with random float values within valid feature ranges
        for col in feature_cols:
            random_val = np.random.uniform(min_vals[col], max_vals[col])
            df_poisoned.loc[idx, col] = round(float(random_val), 1)

        # Assign a random class label (0, 1, or 2)
        random_label_idx = np.random.choice([0, 1, 2])
        df_poisoned.loc[idx, 'species'] = species_labels[random_label_idx]

    return df_poisoned


def main() -> None:
    if not CLEAN_DATA_PATH.exists():
        print(f"Error: Baseline dataset not found at {CLEAN_DATA_PATH}")
        return

    # 1. Load data/raw/iris.csv as clean baseline
    df_clean = pd.read_csv(CLEAN_DATA_PATH)
    print(f"Loaded clean dataset from {CLEAN_DATA_PATH} ({len(df_clean)} samples)")

    # 2 & 4. Create 3 poisoned variants and save to data/raw/
    corruption_levels = [5, 10, 50]
    for level in corruption_levels:
        df_poisoned = poison_dataset(df_clean, level)
        output_path = RAW_DATA_DIR / f"iris_poisoned_{level}.csv"
        df_poisoned.to_csv(output_path, index=False)
        print(f"Saved {level}% poisoned dataset to {output_path}")


if __name__ == "__main__":
    main()
