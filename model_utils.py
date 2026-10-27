"""Dataset validation, model training, persistence, and prediction helpers."""

from __future__ import annotations

import json
import math
import pickle
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
FEATURE_NAMES = (
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
)
FEATURE_LABELS = {
    "sepal_length": "Sepal length",
    "sepal_width": "Sepal width",
    "petal_length": "Petal length",
    "petal_width": "Petal width",
}
RANDOM_SEED = 42
SUPPORTED_DATASET_SUFFIXES = {
    ".csv",
    ".tsv",
    ".txt",
    ".xlsx",
    ".xls",
    ".xlsm",
    ".parquet",
    ".json",
}
NON_DATASET_FILENAMES = {"requirements.txt", "readme.txt", "license.txt", "notice.txt"}
TARGET_ALIASES = {"species", "target", "class", "label", "variety"}
FEATURE_ALIASES = {
    "sepallength": "sepal_length",
    "sepallengthcm": "sepal_length",
    "sepal_length": "sepal_length",
    "sepalwidth": "sepal_width",
    "sepalwidthcm": "sepal_width",
    "sepal_width": "sepal_width",
    "petallength": "petal_length",
    "petallengthcm": "petal_length",
    "petal_length": "petal_length",
    "petalwidth": "petal_width",
    "petalwidthcm": "petal_width",
    "petal_width": "petal_width",
}
MODEL_PRIORITY = (
    "Logistic regression",
    "Decision tree",
    "K-nearest neighbors",
    "Support vector machine",
    "Random forest",
    "Dummy classifier",
)


class DatasetError(ValueError):
    """Raised when a local dataset cannot be identified or validated."""


@dataclass(frozen=True)
class DatasetBundle:
    frame: pd.DataFrame
    path: Path
    feature_names: tuple[str, ...]
    target_column: str
    original_columns: tuple[str, ...]
    identifier_columns: tuple[str, ...]
    ignored_columns: tuple[str, ...]
    units: dict[str, str]


@dataclass
class TrainingResult:
    selected_model_name: str
    selected_model: Pipeline
    comparison: pd.DataFrame
    metadata: dict


def _column_key(name: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).strip().casefold())


def _is_identifier_column(name: str) -> bool:
    key = _column_key(name)
    return (
        key in {"id", "index", "rowid", "recordid"}
        or key.startswith("unnamed")
        or key.endswith("identifier")
        or key.endswith("id")
    )


def find_dataset(dataset_path: str | Path | None = None) -> Path:
    """Find one supported dataset in the project root or its data folder."""
    if dataset_path is not None:
        candidate = Path(dataset_path).expanduser()
        if not candidate.is_absolute():
            candidate = ROOT_DIR / candidate
        if not candidate.is_file():
            raise DatasetError(f"Dataset file does not exist: {candidate}")
        if candidate.suffix.casefold() not in SUPPORTED_DATASET_SUFFIXES:
            raise DatasetError(
                f"Unsupported dataset format '{candidate.suffix}' for {candidate.name}."
            )
        return candidate.resolve()

    candidates: list[Path] = []
    for folder in (DATA_DIR, ROOT_DIR):
        if folder.is_dir():
            candidates.extend(
                path.resolve()
                for path in folder.iterdir()
                if path.is_file()
                and path.suffix.casefold() in SUPPORTED_DATASET_SUFFIXES
                and not path.name.startswith(".")
                and path.name.casefold() not in NON_DATASET_FILENAMES
            )
    candidates = sorted(set(candidates), key=lambda path: str(path).casefold())
    if not candidates:
        raise DatasetError(
            "No dataset was found. Place one CSV, Excel, JSON, or Parquet dataset "
            f"in '{DATA_DIR}' or the project folder."
        )
    if len(candidates) > 1:
        listed = "\n".join(f"  - {path}" for path in candidates)
        raise DatasetError(
            "More than one dataset was found, so automatic selection is ambiguous. "
            "Keep one dataset in the project/data folder or pass an explicit path:\n"
            f"{listed}"
        )
    return candidates[0]


def _read_dataset(path: Path) -> pd.DataFrame:
    suffix = path.suffix.casefold()
    try:
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix == ".tsv":
            return pd.read_csv(path, sep="\t")
        if suffix == ".txt":
            return pd.read_csv(path, sep=None, engine="python")
        if suffix in {".xlsx", ".xls", ".xlsm"}:
            return pd.read_excel(path)
        if suffix == ".parquet":
            return pd.read_parquet(path)
        if suffix == ".json":
            return pd.read_json(path)
    except ImportError as exc:
        raise DatasetError(
            f"Cannot read {path.name}: a required file-format package is missing. "
            "Install the project's requirements and try again."
        ) from exc
    except (OSError, ValueError, UnicodeError) as exc:
        raise DatasetError(f"Could not read dataset '{path}': {exc}") from exc
    raise DatasetError(f"Unsupported dataset format: {path.suffix}")


def validate_dataset_frame(
    frame: pd.DataFrame, path: str | Path | None = None
) -> DatasetBundle:
    """Validate schema, normalize recognized names, and preserve all usable rows."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise DatasetError("The dataset is empty or is not a tabular data frame.")
    if frame.columns.duplicated().any():
        raise DatasetError("The dataset has duplicate column names; rename them first.")

    original_columns = tuple(str(column) for column in frame.columns)
    source_by_key: dict[str, list[str]] = {}
    for column in original_columns:
        source_by_key.setdefault(_column_key(column), []).append(column)

    resolved_features: dict[str, str] = {}
    for key, canonical in FEATURE_ALIASES.items():
        for source_column in source_by_key.get(_column_key(key), []):
            if canonical in resolved_features and resolved_features[canonical] != source_column:
                raise DatasetError(
                    f"Multiple columns map to '{canonical}': "
                    f"'{resolved_features[canonical]}' and '{source_column}'."
                )
            resolved_features[canonical] = source_column

    missing_features = [name for name in FEATURE_NAMES if name not in resolved_features]
    if missing_features:
        raise DatasetError(
            "Could not confidently identify all four Iris measurement columns. "
            f"Missing: {', '.join(missing_features)}. "
            f"Found columns: {', '.join(original_columns)}."
        )

    target_candidates = [
        column
        for column in original_columns
        if _column_key(column) in TARGET_ALIASES
    ]
    if len(target_candidates) != 1:
        if not target_candidates:
            raise DatasetError(
                "Could not confidently identify the species/target column. Expected a "
                "column named Species, Target, Class, Label, or Variety; "
                f"found: {', '.join(original_columns)}."
            )
        raise DatasetError(
            "More than one possible target column was found: "
            f"{', '.join(target_candidates)}. Rename the intended target column "
            "to 'species' and remove the ambiguity."
        )
    source_target = target_candidates[0]
    if source_target in resolved_features.values():
        raise DatasetError("The target column cannot also be a measurement feature.")

    normalized = frame.copy()
    rename_map = {
        source: canonical for canonical, source in resolved_features.items()
    }
    rename_map[source_target] = "species"
    normalized = normalized.rename(columns=rename_map)

    for feature in FEATURE_NAMES:
        original_values = normalized[feature]
        converted = pd.to_numeric(original_values, errors="coerce")
        invalid = original_values.notna() & converted.isna()
        if invalid.any():
            examples = original_values[invalid].astype(str).head(3).tolist()
            raise DatasetError(
                f"Feature '{feature}' contains non-numeric values such as {examples}."
            )
        finite_values = converted.dropna().to_numpy(dtype=float)
        if not np.isfinite(finite_values).all():
            raise DatasetError(
                f"Feature '{feature}' contains an infinite value; correct the source data."
            )
        if converted.notna().sum() == 0:
            raise DatasetError(
                f"Feature '{feature}' contains no usable numeric observations."
            )
        normalized[feature] = converted.astype(float)

    target = normalized["species"]
    missing_target = target.isna() | target.astype("string").str.strip().eq("")
    if missing_target.any():
        raise DatasetError(
            f"The target column '{source_target}' contains "
            f"{int(missing_target.sum())} missing/blank label(s). "
            "Supervised training requires a label for every retained row."
        )
    labels = target.astype(str).str.strip()
    label_lookup: dict[str, str] = {}
    for label in labels:
        label_lookup.setdefault(label.casefold(), label)
    normalized["species"] = labels.str.casefold().map(label_lookup)
    if normalized["species"].nunique() < 2:
        raise DatasetError("At least two distinct species labels are required.")

    normalized_labels = {
        label.casefold().replace("_", "-").removeprefix("iris-")
        for label in normalized["species"].unique()
    }
    iris_species = {"setosa", "versicolor", "virginica"}
    if normalized_labels.intersection(iris_species) and not iris_species.issubset(
        normalized_labels
    ):
        missing = sorted(iris_species - normalized_labels)
        raise DatasetError(
            "This appears to be the standard Iris dataset but is missing expected "
            f"species: {', '.join(missing)}."
        )

    identifier_columns = tuple(
        column
        for column in original_columns
        if column not in resolved_features.values()
        and column != source_target
        and _is_identifier_column(column)
    )
    used_columns = set(resolved_features.values()) | {source_target}
    ignored_columns = tuple(
        column
        for column in original_columns
        if column not in used_columns and column not in identifier_columns
    )
    units = {
        feature: (
            "cm"
            if _column_key(resolved_features[feature]).endswith("cm")
            else "unspecified"
        )
        for feature in FEATURE_NAMES
    }
    resolved_path = (
        Path(path).expanduser().resolve()
        if path is not None
        else Path("<in-memory dataset>")
    )
    return DatasetBundle(
        frame=normalized,
        path=resolved_path,
        feature_names=FEATURE_NAMES,
        target_column=source_target,
        original_columns=original_columns,
        identifier_columns=identifier_columns,
        ignored_columns=ignored_columns,
        units=units,
    )


def load_dataset(dataset_path: str | Path | None = None) -> DatasetBundle:
    path = find_dataset(dataset_path)
    return validate_dataset_frame(_read_dataset(path), path)


def _build_models(training_rows: int, random_seed: int) -> dict[str, Pipeline]:
    neighbors = max(1, min(5, training_rows))
    return {
        "Dummy classifier": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("classifier", DummyClassifier(strategy="most_frequent")),
            ]
        ),
        "Logistic regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(max_iter=2000, random_state=random_seed),
                ),
            ]
        ),
        "K-nearest neighbors": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", KNeighborsClassifier(n_neighbors=neighbors)),
            ]
        ),
        "Decision tree": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    DecisionTreeClassifier(random_state=random_seed),
                ),
            ]
        ),
        "Random forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        random_state=random_seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Support vector machine": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    CalibratedClassifierCV(
                        estimator=SVC(kernel="rbf", random_state=random_seed),
                        method="sigmoid",
                        cv=3,
                        ensemble=False,
                    ),
                ),
            ]
        ),
    }


def _select_model(cv_scores: dict[str, float | None]) -> tuple[str, str]:
    available = {name: score for name, score in cv_scores.items() if score is not None}
    if available:
        highest = max(available.values())
        similarly_scoring = [
            name
            for name in MODEL_PRIORITY
            if name in available and highest - available[name] <= 0.01
        ]
        selected = similarly_scoring[0]
        return (
            selected,
            "Highest mean stratified training cross-validation macro F1-score; "
            "models within 0.01 are resolved using the documented simplicity/"
            "interpretability order.",
        )
    selected = MODEL_PRIORITY[0]
    return (
        selected,
        "Training cross-validation was not possible with the available class counts. "
        "Selected using the predefined simplicity order, without using test-set scores.",
    )


def train_models(
    bundle: DatasetBundle,
    test_size: float = 0.2,
    random_seed: int = RANDOM_SEED,
) -> TrainingResult:
    frame = bundle.frame
    X = frame.loc[:, list(bundle.feature_names)]
    y = frame["species"]
    if not 0 < test_size < 1:
        raise ValueError("test_size must be greater than 0 and less than 1.")
    if len(frame) < 4:
        raise DatasetError(
            "At least four labeled rows are required for a train/test split."
        )
    class_counts = y.value_counts()
    if class_counts.min() < 2:
        raise DatasetError(
            "At least two examples per species are required to keep every class "
            "available for training and evaluation. Current class counts: "
            f"{class_counts.to_dict()}."
        )

    test_rows = math.ceil(len(frame) * test_size)
    stratify = (
        y
        if class_counts.min() >= 2
        and test_rows >= y.nunique()
        and len(frame) - test_rows >= y.nunique()
        else None
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_seed,
        stratify=stratify,
    )

    models = _build_models(len(X_train), random_seed)
    training_class_counts = y_train.value_counts()
    cv_folds = (
        min(5, int(training_class_counts.min()))
        if training_class_counts.min() >= 2
        else 0
    )
    cv = (
        StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
        if cv_folds
        else None
    )
    cv_means: dict[str, float | None] = {}
    cv_stds: dict[str, float | None] = {}
    fitted_models: dict[str, Pipeline] = {}
    predictions: dict[str, np.ndarray] = {}
    for name, estimator in models.items():
        if cv is not None:
            scores = cross_val_score(
                estimator,
                X_train,
                y_train,
                cv=cv,
                scoring="f1_macro",
                n_jobs=None,
                error_score="raise",
            )
            cv_means[name] = float(scores.mean())
            cv_stds[name] = float(scores.std())
        else:
            cv_means[name] = None
            cv_stds[name] = None
        estimator.fit(X_train, y_train)
        fitted_models[name] = estimator
        predictions[name] = estimator.predict(X_test)

    selected_name, selection_criteria = _select_model(cv_means)
    labels = [str(label) for label in fitted_models[selected_name].classes_]
    rows = []
    for name, predicted in predictions.items():
        rows.append(
            {
                "model": name,
                "cv_macro_f1_mean": cv_means[name],
                "cv_macro_f1_std": cv_stds[name],
                "test_accuracy": float(accuracy_score(y_test, predicted)),
                "test_precision_macro": float(
                    precision_score(y_test, predicted, average="macro", zero_division=0)
                ),
                "test_recall_macro": float(
                    recall_score(y_test, predicted, average="macro", zero_division=0)
                ),
                "test_f1_macro": float(
                    f1_score(y_test, predicted, average="macro", zero_division=0)
                ),
            }
        )
    comparison = pd.DataFrame(rows)

    selected_prediction = predictions[selected_name]
    selected_metrics = {
        "accuracy": float(accuracy_score(y_test, selected_prediction)),
        "precision_macro": float(
            precision_score(
                y_test, selected_prediction, average="macro", zero_division=0
            )
        ),
        "recall_macro": float(
            recall_score(y_test, selected_prediction, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_test, selected_prediction, average="macro", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(
            y_test, selected_prediction, labels=labels
        ).tolist(),
        "classification_report": classification_report(
            y_test,
            selected_prediction,
            labels=labels,
            target_names=labels,
            output_dict=True,
            zero_division=0,
        ),
    }
    metadata = {
        "selected_model": selected_name,
        "feature_names": list(bundle.feature_names),
        "target_column": bundle.target_column,
        "normalized_target_column": "species",
        "class_labels": labels,
        "training_configuration": {
            "random_seed": random_seed,
            "test_size": test_size,
            "split_strategy": "stratified" if stratify is not None else "random",
            "cross_validation": (
                f"{cv_folds}-fold stratified cross-validation on training data"
                if cv is not None
                else "Not available: insufficient examples per training class"
            ),
            "cross_validation_folds": cv_folds,
            "metrics_averaging": "macro",
            "preprocessing": (
                "Median imputation fitted inside each training pipeline; "
                "standard scaling for logistic regression, KNN, and SVM only."
            ),
        },
        "cross_validation_macro_f1": cv_means,
        "cross_validation_macro_f1_std": cv_stds,
        "test_metrics": selected_metrics,
        "dataset_filename": bundle.path.name,
        "dataset_path": str(bundle.path),
        "dataset_shape": list(bundle.frame.shape),
        "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_selection_criteria": selection_criteria,
        "model_priority_for_similar_scores": list(MODEL_PRIORITY),
        "data_quality": {
            "missing_values_by_column": {
                str(column): int(count)
                for column, count in bundle.frame.isna().sum().items()
            },
            "duplicate_rows_retained": int(bundle.frame.duplicated().sum()),
            "identifier_columns_excluded": list(bundle.identifier_columns),
            "other_columns_ignored": list(bundle.ignored_columns),
        },
        "units": bundle.units,
    }
    return TrainingResult(
        selected_model_name=selected_name,
        selected_model=fitted_models[selected_name],
        comparison=comparison,
        metadata=metadata,
    )


def save_model(model: Pipeline, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination)
    return destination


def load_model(path: str | Path) -> Pipeline:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"Trained model not found at '{source}'. Run `python train.py` first."
        )
    try:
        return joblib.load(source)
    except (
        EOFError,
        ImportError,
        KeyError,
        OSError,
        pickle.UnpicklingError,
        ValueError,
    ) as exc:
        raise RuntimeError(
            f"Could not load model artifact '{source}'. Retrain it with `python train.py`."
        ) from exc


def save_metadata(metadata: dict, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, allow_nan=False)
        file.write("\n")
    return destination


def load_metadata(path: str | Path) -> dict:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"Model metadata not found at '{source}'. Run `python train.py` first."
        )
    try:
        with source.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Could not read model metadata '{source}'. Run `python train.py` again."
        ) from exc
    if metadata.get("feature_names") != list(FEATURE_NAMES):
        raise RuntimeError(
            "Model metadata feature order does not match the application schema. "
            "Retrain with `python train.py`."
        )
    return metadata


def validate_prediction_input(
    values: Mapping[str, object] | Sequence[object] | pd.DataFrame,
    feature_names: Sequence[str] = FEATURE_NAMES,
) -> pd.DataFrame:
    expected = tuple(feature_names)
    if expected != FEATURE_NAMES:
        raise ValueError(
            f"Unsupported model feature order {expected}; expected {FEATURE_NAMES}."
        )
    if isinstance(values, pd.DataFrame):
        if tuple(values.columns) != expected:
            raise ValueError(
                "Prediction input feature names and order must exactly match training: "
                f"{list(expected)}."
            )
        row = values.copy()
        if len(row) != 1:
            raise ValueError("Provide exactly one flower measurement row for prediction.")
    elif isinstance(values, Mapping):
        if set(values) != set(expected):
            raise ValueError(
                f"Prediction input must contain exactly these features: {list(expected)}."
            )
        row = pd.DataFrame([[values[name] for name in expected]], columns=expected)
    else:
        if isinstance(values, (str, bytes)) or len(values) != len(expected):
            raise ValueError(
                f"Prediction input must contain exactly {len(expected)} measurements."
            )
        row = pd.DataFrame([list(values)], columns=expected)

    for feature in expected:
        value = row.iloc[0][feature]
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"'{feature}' must be a finite numeric measurement.")
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"'{feature}' must be a finite numeric measurement."
            ) from exc
        if not math.isfinite(numeric_value):
            raise ValueError(f"'{feature}' must be finite; received {value!r}.")
        row.loc[row.index[0], feature] = numeric_value
    return row.astype(float)


def predict_species(
    model: Pipeline,
    values: Mapping[str, object] | Sequence[object] | pd.DataFrame,
    feature_names: Sequence[str] = FEATURE_NAMES,
) -> str:
    inputs = validate_prediction_input(values, feature_names)
    predictions = model.predict(inputs)
    if len(predictions) != 1:
        raise RuntimeError("The model did not return exactly one species prediction.")
    result = str(predictions[0])
    model_classes = {str(label) for label in getattr(model, "classes_", [])}
    if model_classes and result not in model_classes:
        raise RuntimeError(f"The model returned an unknown class label: {result}.")
    return result


def predict_probabilities(
    model: Pipeline,
    values: Mapping[str, object] | Sequence[object] | pd.DataFrame,
    feature_names: Sequence[str] = FEATURE_NAMES,
) -> dict[str, float] | None:
    inputs = validate_prediction_input(values, feature_names)
    predict_proba = getattr(model, "predict_proba", None)
    if predict_proba is None:
        return None
    probabilities = np.asarray(predict_proba(inputs), dtype=float)
    classes = getattr(model, "classes_", None)
    class_count = len(classes) if classes is not None else 0
    if probabilities.shape != (1, class_count):
        raise RuntimeError("The model returned probabilities with an unexpected shape.")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise RuntimeError("The model returned invalid class probabilities.")
    return {
        str(label): float(probability)
        for label, probability in zip(classes, probabilities[0], strict=True)
    }
