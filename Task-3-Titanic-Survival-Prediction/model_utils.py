from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "Titanic-Dataset.csv"
MODEL_PATH = PROJECT_DIR / "models" / "titanic_model.pkl"
FEATURES = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
NUMERIC_FEATURES = ["pclass", "age", "sibsp", "parch", "fare"]
CATEGORICAL_FEATURES = ["sex", "embarked"]


def load_raw_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Titanic dataset not found: {DATA_PATH}")

    dataset = pd.read_csv(DATA_PATH)
    dataset.columns = dataset.columns.str.lower()
    return dataset


def load_dataset():
    dataset = load_raw_dataset()
    required_columns = FEATURES + ["survived"]
    missing_columns = [
        column for column in required_columns if column not in dataset.columns
    ]
    if missing_columns:
        raise ValueError(
            "The Titanic dataset is missing required columns: "
            + ", ".join(missing_columns)
        )

    features = dataset[FEATURES].copy()
    if dataset["survived"].isna().any():
        raise ValueError("The Survived column cannot contain missing labels.")
    target = pd.to_numeric(dataset["survived"], errors="raise")
    if set(target.unique()) != {0, 1}:
        raise ValueError("The Survived column must contain only 0 and 1 labels.")
    return features, target.astype(int)


def build_pipeline(classifier):
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", classifier)]
    )


def load_model(path=MODEL_PATH):
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. Run `python train.py` first."
        )

    artifact = joblib.load(model_path)
    required_keys = {
        "model",
        "model_name",
        "features",
        "model_comparison",
        "validation_metrics",
        "validation_confusion_matrices",
        "validation",
    }
    missing_keys = required_keys.difference(artifact) if isinstance(artifact, dict) else required_keys
    if missing_keys:
        raise ValueError(
            "The saved model artifact is incomplete. Missing: "
            + ", ".join(sorted(missing_keys))
        )
    if not callable(getattr(artifact["model"], "predict", None)):
        raise ValueError("The saved model artifact does not contain a usable model.")
    return artifact


def validate_passenger(passenger):
    missing = [feature for feature in FEATURES if feature not in passenger]
    if missing:
        raise ValueError("Missing passenger fields: " + ", ".join(missing))

    if passenger["pclass"] not in (1, 2, 3):
        raise ValueError("Passenger class must be 1, 2, or 3.")
    if passenger["sex"] not in ("female", "male"):
        raise ValueError("Gender must be female or male.")
    if passenger["embarked"] not in ("C", "Q", "S"):
        raise ValueError("Embarkation port must be C, Q, or S.")

    limits = {"age": (0, 100), "sibsp": (0, 10), "parch": (0, 10), "fare": (0, 1000)}
    for feature, (minimum, maximum) in limits.items():
        try:
            value = float(passenger[feature])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{feature.title()} must be a number.") from error
        if not np.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(
                f"{feature.title()} must be between {minimum} and {maximum}."
            )
        if feature in ("sibsp", "parch") and not value.is_integer():
            raise ValueError(f"{feature.title()} must be a whole number.")


def predict_passenger(model, passenger):
    validate_passenger(passenger)
    input_frame = pd.DataFrame(
        [{feature: passenger[feature] for feature in FEATURES}],
        columns=FEATURES,
    )
    prediction = int(model.predict(input_frame)[0])
    predict_probability = getattr(model, "predict_proba", None)
    probability = (
        float(predict_probability(input_frame)[0, 1])
        if callable(predict_probability)
        else None
    )
    return prediction, probability