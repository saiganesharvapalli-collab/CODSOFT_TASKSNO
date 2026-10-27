from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "data"
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "model_comparison.csv"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REQUIRED_COLUMNS = [
    "Name",
    "Year",
    "Duration",
    "Genre",
    "Rating",
    "Votes",
    "Director",
    "Actor 1",
    "Actor 2",
    "Actor 3",
]
FEATURE_COLUMNS = [
    "Year",
    "Duration",
    "Genre",
    "Director",
    "Actor 1",
    "Actor 2",
    "Actor 3",
]
RANDOM_SEED = 42


def find_dataset_path(base_dir: Optional[Path] = None) -> Path:
    root = Path(base_dir) if base_dir is not None else PROJECT_ROOT
    expected = root / "data" / "IMDb Movies India.csv"
    if expected.exists():
        return expected
    matches = list(root.glob("**/IMDb Movies India.csv"))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        "Dataset not found. Expected 'data/IMDb Movies India.csv' in the project folder. "
        "Please place the CSV in the data directory and rerun the command."
    )


class GenreTransformer(BaseEstimator, TransformerMixin):
    """Transforms a comma-separated genre string into a dense multi-label matrix."""

    def __init__(self):
        self.mlb_ = None
        self.is_fitted_ = False

    def fit(self, X: Iterable[Any], y: Optional[Any] = None):
        values = [self._parse_genres(value) for value in X]
        self.mlb_ = MultiLabelBinarizer(sparse_output=False)
        self.mlb_.fit(values)
        self.is_fitted_ = True
        return self

    def transform(self, X: Iterable[Any]):
        if not self.is_fitted_ or self.mlb_ is None:
            raise ValueError("GenreTransformer has not been fitted yet.")
        values = [self._parse_genres(value) for value in X]
        return self.mlb_.transform(values)

    def get_feature_names_out(self, input_features=None):
        if not self.is_fitted_ or self.mlb_ is None:
            raise ValueError("GenreTransformer has not been fitted yet.")
        return [f"genre_{label}" for label in self.mlb_.classes_]

    @staticmethod
    def _parse_genres(value: Any) -> List[str]:
        if pd.isna(value):
            return []
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            return []
        return [part.strip() for part in text.split(",") if part.strip()]


class MultiLabelBinarizer:
    """Compatibility wrapper for a multi-label binarizer that works well with text-like genre values."""

    def __init__(self, sparse_output: bool = False):
        self.sparse_output = sparse_output
        self.classes_: List[str] = []

    def fit(self, values: Iterable[List[str]]):
        labels = sorted({label for item in values for label in item})
        self.classes_ = labels
        return self

    def transform(self, values: Iterable[List[str]]):
        labels = self.classes_
        rows = []
        for item in values:
            row = np.zeros(len(labels), dtype=int)
            for idx, label in enumerate(labels):
                if label in item:
                    row[idx] = 1
            rows.append(row)
        arr = np.vstack(rows) if rows else np.zeros((0, len(labels)), dtype=int)
        if self.sparse_output:
            from scipy import sparse

            return sparse.csr_matrix(arr)
        return arr


def parse_year(series: pd.Series) -> pd.Series:
    cleaned = series.fillna("")
    cleaned = cleaned.astype(str).str.replace(r"\(|\)|\[|\]|\s+", "", regex=True)
    cleaned = cleaned.str.replace(r"[^0-9]", "", regex=True)
    parsed = pd.to_numeric(cleaned, errors="coerce")
    valid_lower = 1888
    valid_upper = pd.Timestamp.now().year + 1
    return parsed.where(parsed.between(valid_lower, valid_upper), np.nan)


def parse_duration(series: pd.Series) -> pd.Series:
    cleaned = series.fillna("").astype(str)
    cleaned = cleaned.str.replace("min", "", case=False, regex=False)
    cleaned = cleaned.str.replace("mins", "", case=False, regex=False)
    cleaned = cleaned.str.replace("minutes", "", case=False, regex=False)
    cleaned = cleaned.str.replace(r"[^0-9]", "", regex=True)
    parsed = pd.to_numeric(cleaned, errors="coerce")
    return parsed.where(parsed > 0, np.nan)


def validate_dataset(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("The dataset is empty. Please verify the CSV file contents.")
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"The dataset is missing required columns: {missing}")


def load_dataset(data_path: Optional[Path] = None) -> pd.DataFrame:
    dataset_path = find_dataset_path(data_path) if data_path is not None else find_dataset_path()
    try:
        df = pd.read_csv(dataset_path, encoding="latin1")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Dataset file not found: {dataset_path}. Place the IMDb Movies India CSV in the data folder."
        ) from exc
    except Exception as exc:  # pragma: no cover - environment-specific decode errors
        raise ValueError(f"Could not read dataset at {dataset_path}: {exc}") from exc
    validate_dataset(df)
    return df


def describe_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.isna()
        .sum()
        .rename("missing_count")
        .to_frame()
        .assign(missing_percentage=lambda frame: (frame["missing_count"] / len(df)) * 100)
        .sort_values(["missing_count", "missing_percentage"], ascending=False)
    )


def prepare_training_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    cleaned = df.copy()
    original_rows = len(cleaned)
    initial_duplicates = cleaned.duplicated().sum()
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    rows_removed_duplicates = initial_duplicates

    cleaned["Year"] = parse_year(cleaned["Year"])
    cleaned["Duration"] = parse_duration(cleaned["Duration"])
    cleaned["Rating"] = pd.to_numeric(cleaned["Rating"], errors="coerce")
    clean_rows_before_invalid_rating = len(cleaned)
    valid_rating_mask = cleaned["Rating"].notna() & cleaned["Rating"].between(0, 10)
    cleaned = cleaned.loc[valid_rating_mask].reset_index(drop=True)
    rows_removed_invalid_rating = clean_rows_before_invalid_rating - len(cleaned)

    feature_frame = cleaned[FEATURE_COLUMNS].copy()
    for col in ["Director", "Actor 1", "Actor 2", "Actor 3"]:
        feature_frame[col] = feature_frame[col].fillna("Unknown").astype(str).str.strip()
        feature_frame[col] = feature_frame[col].replace({"": "Unknown", "nan": "Unknown", "None": "Unknown", "NaN": "Unknown"})
    feature_frame["Genre"] = feature_frame["Genre"].fillna("Unknown").astype(str).str.strip()
    feature_frame["Genre"] = feature_frame["Genre"].replace({"": "Unknown", "nan": "Unknown", "None": "Unknown", "NaN": "Unknown"})

    target = cleaned["Rating"].astype(float).reset_index(drop=True)
    feature_frame = feature_frame.reset_index(drop=True)
    if len(feature_frame) != len(target):
        raise ValueError("Feature and target lengths do not align after cleaning.")

    return {
        "full_cleaned": cleaned,
        "feature_frame": feature_frame,
        "target": target,
        "summary": {
            "original_row_count": original_rows,
            "duplicate_rows_removed": rows_removed_duplicates,
            "invalid_rating_rows_removed": rows_removed_invalid_rating,
            "valid_target_row_count": len(target),
            "feature_columns": FEATURE_COLUMNS,
        },
    }


def build_preprocessor() -> ColumnTransformer:
    numeric_features = ["Year", "Duration"]
    categorical_features = ["Director", "Actor 1", "Actor 2", "Actor 3"]
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            ),
            (
                "genre",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                        ("genre_transform", GenreTransformer()),
                    ]
                ),
                ["Genre"],
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", max_categories=30, sparse_output=False)),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )


def build_model_candidates() -> Dict[str, Any]:
    return {
        "DummyRegressor": DummyRegressor(strategy="mean"),
        "LinearRegression": LinearRegression(),
        "DecisionTreeRegressor": DecisionTreeRegressor(random_state=RANDOM_SEED, max_depth=10, min_samples_leaf=5),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=80, random_state=RANDOM_SEED, n_jobs=-1),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=RANDOM_SEED, n_estimators=150),
    }


def evaluate_model(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    cv = KFold(n_splits=2, shuffle=True, random_state=RANDOM_SEED)
    val_predictions = cross_val_predict(pipeline, X_train, y_train, cv=cv, method="predict")
    val_mae = mean_absolute_error(y_train, val_predictions)
    val_rmse = math.sqrt(mean_squared_error(y_train, val_predictions))
    val_r2 = r2_score(y_train, val_predictions)

    test_predictions = pipeline.predict(X_test)
    test_mae = mean_absolute_error(y_test, test_predictions)
    test_rmse = math.sqrt(mean_squared_error(y_test, test_predictions))
    test_r2 = r2_score(y_test, test_predictions)

    return {
        "Validation_MAE": float(val_mae),
        "Validation_RMSE": float(val_rmse),
        "Validation_R2": float(val_r2),
        "Test_MAE": float(test_mae),
        "Test_RMSE": float(test_rmse),
        "Test_R2": float(test_r2),
    }


def train_and_score_models(data: pd.DataFrame) -> Dict[str, Any]:
    prepared = prepare_training_dataframe(data)
    feature_frame = prepared["feature_frame"]
    target = prepared["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        feature_frame, target, test_size=0.2, random_state=RANDOM_SEED
    )

    comparison_rows: List[Dict[str, Any]] = []
    fitted_pipelines: Dict[str, Pipeline] = {}
    for model_name, regressor in build_model_candidates().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("regressor", regressor),
            ]
        )
        pipeline.fit(X_train, y_train)
        fitted_pipelines[model_name] = pipeline
        metrics = evaluate_model(pipeline, X_train, y_train, X_test, y_test)
        comparison_rows.append({"Model": model_name, **metrics})

    selected_name = min(comparison_rows, key=lambda row: row["Validation_MAE"])["Model"]
    best_pipeline = fitted_pipelines[selected_name]
    final_metrics = evaluate_model(best_pipeline, X_train, y_train, X_test, y_test)
    comparison_df = pd.DataFrame(comparison_rows)
    selected_row = comparison_df[comparison_df["Model"] == selected_name].copy()
    selected_row["Test_MAE"] = final_metrics["Test_MAE"]
    selected_row["Test_RMSE"] = final_metrics["Test_RMSE"]
    selected_row["Test_R2"] = final_metrics["Test_R2"]
    comparison_df.loc[comparison_df["Model"] == selected_name, ["Test_MAE", "Test_RMSE", "Test_R2"]] = [
        final_metrics["Test_MAE"],
        final_metrics["Test_RMSE"],
        final_metrics["Test_R2"],
    ]
    comparison_df = comparison_df[["Model", "Validation_MAE", "Validation_RMSE", "Validation_R2", "Test_MAE", "Test_RMSE", "Test_R2"]]
    comparison_df.to_csv(REPORT_PATH, index=False)

    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "target_column": "Rating",
        "input_feature_columns": FEATURE_COLUMNS,
        "selected_model_name": selected_name,
        "random_seed": RANDOM_SEED,
        "dataset_row_count": len(data),
        "valid_target_row_count": len(target),
        "training_row_count": len(X_train),
        "test_row_count": len(X_test),
        "evaluation_metrics": final_metrics,
        "training_timestamp": pd.Timestamp.now().isoformat(),
        "feature_engineering_description": (
            "Parsed year from strings like '(2019)', duration from strings like '109 min', "
            "handled missing numeric values, filled unknown categorical labels, and encoded genre via multi-label binarization."
        ),
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_pipeline, f)

    return {
        "selected_model_name": selected_name,
        "comparison_df": comparison_df,
        "best_pipeline": best_pipeline,
        "metrics": final_metrics,
        "metadata": metadata,
    }


def load_model(model_path: Optional[Path] = None):
    path = Path(model_path) if model_path is not None else MODEL_PATH
    if not path.exists():
        raise FileNotFoundError("Model artifact is missing. Please run: python train.py")
    with open(path, "rb") as f:
        return pickle.load(f)


def validate_prediction_input(data: Dict[str, Any]) -> Dict[str, Any]:
    required = {
        "Year": (int, float),
        "Duration": (int, float),
        "Genre": str,
        "Director": str,
        "Actor 1": str,
        "Actor 2": str,
        "Actor 3": str,
    }
    cleaned = {}
    for key, expected_type in required.items():
        value = data.get(key)
        if value is None or (isinstance(value, str) and not value.strip()):
            cleaned[key] = "Unknown" if key in {"Director", "Actor 1", "Actor 2", "Actor 3"} else "Unknown"
            continue
        if isinstance(value, (list, tuple)):
            value = ", ".join(str(item).strip() for item in value if str(item).strip())
        if key in {"Year", "Duration"}:
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{key} must be a numeric value.") from exc
            if key == "Year" and not (1888 <= numeric <= pd.Timestamp.now().year + 1):
                raise ValueError(f"Year must be between 1888 and {pd.Timestamp.now().year + 1}.")
            if key == "Duration" and numeric <= 0:
                raise ValueError("Duration must be greater than zero.")
            cleaned[key] = numeric
        else:
            text = str(value).strip()
            if not text:
                cleaned[key] = "Unknown"
            else:
                cleaned[key] = text
    return cleaned


def predict_single_rating(model, row: Dict[str, Any]) -> float:
    prepared = validate_prediction_input(row)
    frame = pd.DataFrame([{
        "Year": prepared["Year"],
        "Duration": prepared["Duration"],
        "Genre": prepared["Genre"],
        "Director": prepared["Director"],
        "Actor 1": prepared["Actor 1"],
        "Actor 2": prepared["Actor 2"],
        "Actor 3": prepared["Actor 3"],
    }])
    prediction = model.predict(frame)
    return float(prediction[0])


def load_metadata() -> Dict[str, Any]:
    if not METADATA_PATH.exists():
        return {}
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model_comparison() -> pd.DataFrame:
    if not REPORT_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(REPORT_PATH)
