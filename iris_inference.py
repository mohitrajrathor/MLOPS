from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.prepare import remove_outliers


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "artifacts" / "model.joblib"
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "iris.csv"
TARGET_COLUMN = "species"
FEATURE_COLUMNS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

app = FastAPI(title="Iris Classifier API")


class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


def build_inference_artifacts() -> tuple[ColumnTransformer, LabelEncoder]:
    """Recreate the training-time preprocessing and label mapping."""
    df = pd.read_csv(DATA_PATH).dropna()
    df = remove_outliers(df)

    preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                FEATURE_COLUMNS,
            )
        ]
    )
    preprocessor.fit(df[FEATURE_COLUMNS])

    label_encoder = LabelEncoder()
    label_encoder.fit(df[TARGET_COLUMN])

    return preprocessor, label_encoder


model = joblib.load(MODEL_PATH)
preprocessor, label_encoder = build_inference_artifacts()


@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris Classifier API!"}


@app.post("/predict/")
def predict_species(data: IrisInput):
    input_df = pd.DataFrame([data.model_dump()], columns=FEATURE_COLUMNS)
    transformed_input = preprocessor.transform(input_df)
    predicted_class = int(model.predict(transformed_input)[0])
    predicted_species = label_encoder.inverse_transform([predicted_class])[0]

    return {
        "predicted_class": predicted_class,
        "predicted_species": predicted_species,
    }
