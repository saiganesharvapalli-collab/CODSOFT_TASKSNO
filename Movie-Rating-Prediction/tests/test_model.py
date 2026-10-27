from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_utils import (
    FEATURE_COLUMNS,
    MODEL_PATH,
    RANDOM_SEED,
    build_model_candidates,
    build_preprocessor,
    load_dataset,
    load_model,
    parse_duration,
    parse_year,
    predict_single_rating,
    prepare_training_dataframe,
    train_and_score_models,
    validate_prediction_input,
)


def test_dataset_loading_real_data():
    df = load_dataset()
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] > 0
    assert set(["Name", "Year", "Duration", "Genre", "Rating", "Votes", "Director", "Actor 1", "Actor 2", "Actor 3"]).issubset(df.columns)


def test_year_parsing():
    values = pd.Series(["(2019)", "N/A", "2022", "2009"])
    parsed = parse_year(values)
    assert parsed.iloc[0] == 2019.0
    assert pd.isna(parsed.iloc[1])
    assert parsed.iloc[2] == 2022.0


def test_duration_parsing():
    values = pd.Series(["109 min", "N/A", "102 mins", "90 minutes"])
    parsed = parse_duration(values)
    assert parsed.iloc[0] == 109.0
    assert pd.isna(parsed.iloc[1])
    assert parsed.iloc[2] == 102.0
    assert parsed.iloc[3] == 90.0


def test_missing_value_handling_pipeline():
    df = pd.DataFrame(
        {
            "Name": ["Movie A", "Movie B", "Movie C"],
            "Year": ["(2020)", np.nan, "(2021)"],
            "Duration": ["120 min", "N/A", "100 min"],
            "Genre": ["Drama, Romance", np.nan, "Comedy"],
            "Rating": [7.2, np.nan, 6.0],
            "Votes": ["1000", "200", "3000"],
            "Director": ["Director A", np.nan, "Director C"],
            "Actor 1": ["Actor A", "Actor D", "Actor E"],
            "Actor 2": ["Actor B", "Actor F", "Actor G"],
            "Actor 3": ["Actor C", "Actor H", "Actor I"],
        }
    )
    prepared = prepare_training_dataframe(df)
    assert prepared["summary"]["valid_target_row_count"] == 2
    assert prepared["feature_frame"].shape[0] == 2
    assert prepared["feature_frame"]["Year"].notna().all()


def test_genre_preprocessor():
    preprocessor = build_preprocessor()
    data = pd.DataFrame(
        {
            "Year": [2020.0, 2021.0],
            "Duration": [120.0, 90.0],
            "Genre": ["Drama, Romance", "Comedy"],
            "Director": ["Dir A", "Dir B"],
            "Actor 1": ["A1", "A2"],
            "Actor 2": ["A3", "A4"],
            "Actor 3": ["A5", "A6"],
        }
    )
    transformed = preprocessor.fit_transform(data)
    assert transformed.shape[0] == 2
    assert transformed.shape[1] > 0


def test_unknown_director_actor_categories():
    data = pd.DataFrame(
        {
            "Year": [2019, 2020],
            "Duration": [110, 95],
            "Genre": ["Drama", "Crime"],
            "Director": ["Unknown Director", "Another Director"],
            "Actor 1": ["Unknown Actor", "Actor X"],
            "Actor 2": ["Actor Y", "Actor Z"],
            "Actor 3": ["Actor W", "Actor Q"],
        }
    )
    transformed = build_preprocessor().fit_transform(data)
    assert transformed.shape[0] == 2
    assert np.isfinite(transformed.data).all()


def test_model_training_and_prediction_small_fixture():
    df = pd.DataFrame(
        {
            "Name": [f"Movie {i}" for i in range(20)],
            "Year": [2015 + (i % 5) for i in range(20)],
            "Duration": [100 + i for i in range(20)],
            "Genre": ["Drama" if i % 2 == 0 else "Comedy" for i in range(20)],
            "Rating": [5.5 + (i % 5) * 0.4 for i in range(20)],
            "Votes": [1000 + i * 20 for i in range(20)],
            "Director": [f"Dir {i % 4}" for i in range(20)],
            "Actor 1": [f"Actor {i % 6}" for i in range(20)],
            "Actor 2": [f"Actor {i % 5}" for i in range(20)],
            "Actor 3": [f"Actor {i % 7}" for i in range(20)],
        }
    )
    results = train_and_score_models(df)
    assert results["selected_model_name"] in build_model_candidates()
    assert set(["Validation_MAE", "Validation_RMSE", "Validation_R2"]).issubset(results["comparison_df"].columns)
    model = results["best_pipeline"]
    sample_row = {"Year": 2022, "Duration": 120, "Genre": "Drama", "Director": "Dir 1", "Actor 1": "Actor 2", "Actor 2": "Actor 3", "Actor 3": "Actor 5"}
    prediction = predict_single_rating(model, sample_row)
    assert np.isfinite(prediction)
    assert isinstance(prediction, float)


def test_model_serialization_and_loading(tmp_path):
    model = build_preprocessor().fit(pd.DataFrame({"Year": [2019], "Duration": [100], "Genre": ["Drama"], "Director": ["Dir"], "Actor 1": ["A1"], "Actor 2": ["A2"], "Actor 3": ["A3"]}))
    artifact_path = tmp_path / "test_model.pkl"
    with open(artifact_path, "wb") as f:
        pickle.dump(model, f)
    with open(artifact_path, "rb") as f:
        loaded = pickle.load(f)
    assert loaded is not None


def test_missing_model_artifact_handling():
    missing = Path("./missing_model.pkl")
    with pytest.raises(FileNotFoundError):
        load_model(missing)


def test_prediction_input_validation():
    invalid = {"Year": "not a year", "Duration": "oops", "Genre": "Drama", "Director": "Dir", "Actor 1": "A1", "Actor 2": "A2", "Actor 3": "A3"}
    with pytest.raises(ValueError):
        validate_prediction_input(invalid)


@pytest.mark.skipif(not Path("data/IMDb Movies India.csv").exists(), reason="Real dataset is not available.")
def test_full_integration_with_real_dataset():
    results = train_and_score_models(load_dataset())
    assert results["selected_model_name"]
    assert MODEL_PATH.exists()
    loaded_model = load_model(MODEL_PATH)
    assert loaded_model is not None
