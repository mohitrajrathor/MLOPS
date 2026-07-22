# 22f3003109_MLOPS_WEEKLY_ASSIGNMENT - Week 5

## Overview
This branch integrates **MLflow** into the IRIS classification pipeline for experiment tracking and model registry. Models are now versioned through MLflow.

---

## Project Structure

```
├── src/
│   ├── prepare.py          # Data cleaning, scaling, saves .npy files
│   └── train.py            # Model training with MLflow tracking & evaluation
├── data/
│   ├── raw/
│   │   └── iris.csv        # Raw dataset (tracked by DVC)
│   ├── processed/          # Prepared .npy files (tracked by DVC)
│   ├── v1/
│   │   └── data.csv        # Dataset version 1
│   └── v2/
│       └── data.csv        # Dataset version 2
├── artifacts/              # Saved model (joblib)
├── mlruns/                 # MLflow local tracking data
├── .github/
│   └── workflows/
│       └── ci.yaml         # CI sanity check via MLflow registry
├── notebook.ipynb          # Experimentation notebook
├── requirements.txt
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Prepare Data
```bash
python src/prepare.py -d data/raw/iris.csv -s data/processed/
```

### 2. Train Model
```bash
# single run
python src/train.py --max-depth 3 --min-samples-split 2

# multiple runs to compare experiments
python src/train.py --max-depth 3 --min-samples-split 2
python src/train.py --max-depth 5 --min-samples-split 4
python src/train.py --max-depth 8 --min-samples-split 10
```

### 3. View MLflow UI
```bash
mlflow ui --port 8100
# open http://localhost:8100
```

---

## MLflow Tracking

Each training run logs:

| Type | Values |
|------|--------|
| Params | `max_depth`, `min_samples_split` |
| Metrics | `accuracy`, `precision`, `f1_score` |
| Model | Registered as `iris_decision_tree` |

---

## Tasks Completed

- ✅ Task 1 — Hyperparameter tuning (`max_depth`, `min_samples_split`)
- ✅ Task 2 — MLflow experiment tracking (params, metrics, model artifact)
- ✅ Task 3 — Compare experiments via MLflow UI
- ✅ Task 4 — Model removed from DVC; DVC tracks data only
- ✅ Task 5 — Model fetched from MLflow Registry in evaluation
- ✅ Task 6 — CI fetches latest model from MLflow and runs sanity check

---

## Requirements

```
pandas
numpy
scikit-learn
matplotlib
pytest
mlflow
```