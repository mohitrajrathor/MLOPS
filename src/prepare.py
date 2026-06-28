import yaml
import numpy as np
import pandas as pd

PARAMS_PATH = "params.yaml"
INPUT_PATH = "data/iris.csv"
OUTPUT_PATH = "data/iris_prepared.csv"


def load_params():
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)["prepare"]


def augment(df, augment_factor, noise_std, random_state):
    rng = np.random.RandomState(random_state)
    feature_cols = [c for c in df.columns if c != "target"]

    augmented_frames = [df]
    for i in range(augment_factor):
        noisy = df.copy()
        noise = rng.normal(loc=0.0, scale=noise_std, size=noisy[feature_cols].shape)
        noisy[feature_cols] = noisy[feature_cols].values + noise
        augmented_frames.append(noisy)

    result = pd.concat(augmented_frames, ignore_index=True)
    return result


def main():
    params = load_params()
    df = pd.read_csv(INPUT_PATH)

    augmented = augment(
        df,
        augment_factor=params["augment_factor"],
        noise_std=params["noise_std"],
        random_state=params["random_state"],
    )

    augmented.to_csv(OUTPUT_PATH, index=False)
    print(f"Prepared data shape: {augmented.shape} -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
