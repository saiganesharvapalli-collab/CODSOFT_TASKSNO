from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from model_utils import (
    FEATURE_NAMES,
    DatasetError,
    load_dataset,
    load_metadata,
    load_model,
    predict_probabilities,
    predict_species,
    save_metadata,
    save_model,
    train_models,
    validate_dataset_frame,
    validate_prediction_input,
)


@pytest.fixture(scope="module")
def actual_dataset():
    return load_dataset()


@pytest.fixture(scope="module")
def actual_training(actual_dataset):
    return train_models(actual_dataset)


def test_loads_the_user_provided_dataset(actual_dataset):
    assert actual_dataset.path.name == "IRIS.csv"
    assert actual_dataset.frame.shape == (150, 5)
    assert actual_dataset.feature_names == FEATURE_NAMES
    assert actual_dataset.target_column == "species"
    assert set(actual_dataset.frame["species"]) == {
        "Iris-setosa",
        "Iris-versicolor",
        "Iris-virginica",
    }


def test_schema_aliases_and_identifier_columns_are_handled(actual_dataset):
    source = actual_dataset.frame.rename(
        columns={
            "sepal_length": "SepalLengthCm",
            "sepal_width": "sepal width",
            "petal_length": "petal_length",
            "petal_width": "PetalWidthCm",
            "species": "Species",
        }
    ).copy()
    source.insert(0, "Id", range(1, len(source) + 1))
    source["Species"] = source["Species"].str.upper().str.strip()

    result = validate_dataset_frame(source)

    assert result.feature_names == FEATURE_NAMES
    assert result.target_column == "Species"
    assert result.identifier_columns == ("Id",)
    assert result.units["sepal_length"] == "cm"
    assert result.frame["species"].nunique() == 3


def test_missing_feature_values_are_preserved_for_training_imputation(actual_dataset):
    frame = actual_dataset.frame.copy()
    frame.loc[0, "sepal_length"] = np.nan
    bundle = validate_dataset_frame(frame, actual_dataset.path)

    assert pd.isna(bundle.frame.loc[0, "sepal_length"])
    result = train_models(bundle)
    assert result.selected_model.predict(
        bundle.frame.loc[[0], list(FEATURE_NAMES)]
    )[0] in result.metadata["class_labels"]


def test_invalid_measurement_text_is_rejected(actual_dataset):
    frame = actual_dataset.frame.copy()
    frame["petal_length"] = frame["petal_length"].astype(object)
    frame.loc[0, "petal_length"] = "not-a-number"

    with pytest.raises(DatasetError, match="non-numeric"):
        validate_dataset_frame(frame)


def test_missing_species_label_is_rejected(actual_dataset):
    frame = actual_dataset.frame.copy()
    frame.loc[0, "species"] = None

    with pytest.raises(DatasetError, match="missing/blank"):
        validate_dataset_frame(frame)


def test_prediction_inputs_require_exact_feature_order():
    row = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=FEATURE_NAMES)
    assert tuple(validate_prediction_input(row).columns) == FEATURE_NAMES

    with pytest.raises(ValueError, match="order"):
        validate_prediction_input(row[list(reversed(FEATURE_NAMES))])


@pytest.mark.parametrize(
    "values",
    [
        [5.1, 3.5, 1.4, float("nan")],
        [5.1, 3.5, 1.4, float("inf")],
        [5.1, 3.5, 1.4, "invalid"],
        [5.1, 3.5, 1.4],
    ],
)
def test_invalid_prediction_values_are_rejected(values):
    with pytest.raises(ValueError):
        validate_prediction_input(values)


def test_model_training_and_metrics_use_all_three_species(
    actual_dataset, actual_training
):
    assert set(actual_training.metadata["class_labels"]) == set(
        actual_dataset.frame["species"]
    )
    assert len(actual_training.comparison) == 6
    assert actual_training.metadata["training_configuration"]["cross_validation_folds"] >= 2
    assert 0 <= actual_training.metadata["test_metrics"]["accuracy"] <= 1
    assert (
        actual_training.metadata["test_metrics"]["confusion_matrix"]
        and len(actual_training.metadata["test_metrics"]["confusion_matrix"]) == 3
    )


def test_model_persistence_and_prediction_probabilities(tmp_path, actual_training):
    model_path = tmp_path / "best_model.pkl"
    save_model(actual_training.selected_model, model_path)
    restored = load_model(model_path)
    values = [5.1, 3.5, 1.4, 0.2]

    prediction = predict_species(restored, values)
    probabilities = predict_probabilities(restored, values)

    assert isinstance(prediction, str)
    assert prediction in actual_training.metadata["class_labels"]
    assert probabilities is not None
    assert set(probabilities) == set(actual_training.metadata["class_labels"])
    assert all(np.isfinite(value) and 0 <= value <= 1 for value in probabilities.values())
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_metadata_is_saved_and_loaded_with_feature_order(tmp_path):
    metadata_path = tmp_path / "model_metadata.json"
    expected = {"feature_names": list(FEATURE_NAMES), "selected_model": "test"}
    save_metadata(expected, metadata_path)

    assert load_metadata(metadata_path) == expected
    assert json.loads(metadata_path.read_text(encoding="utf-8")) == expected


def test_missing_model_and_metadata_files_report_actionable_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="python train.py"):
        load_model(tmp_path / "missing.pkl")
    with pytest.raises(FileNotFoundError, match="python train.py"):
        load_metadata(tmp_path / "missing.json")


def test_corrupt_model_artifact_is_reported(tmp_path):
    corrupt_path = tmp_path / "corrupt.pkl"
    corrupt_path.write_bytes(b"not a pickle file")

    with pytest.raises(RuntimeError, match="Retrain"):
        load_model(corrupt_path)


def test_small_fitted_pipeline_prediction_returns_a_known_class():
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000)),
        ]
    )
    training = pd.DataFrame(
        [
            [5.0, 3.5, 1.5, 0.2],
            [5.1, 3.4, 1.6, 0.3],
            [6.0, 2.8, 4.5, 1.3],
            [6.1, 2.9, 4.6, 1.4],
        ],
        columns=FEATURE_NAMES,
    )
    model.fit(training, ["setosa", "setosa", "versicolor", "versicolor"])

    assert predict_species(model, training.iloc[0].to_dict()) in {
        "setosa",
        "versicolor",
    }


def test_streamlit_dashboard_can_predict():
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=20).run()
    assert not app.exception

    app.radio[0].set_value("Predict species").run()
    assert not app.exception
    app.button[0].click().run()

    assert not app.exception
    assert app.success
    assert "Predicted species" in app.success[0].value
