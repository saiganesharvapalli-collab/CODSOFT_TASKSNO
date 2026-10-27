from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from model_utils import (
    FEATURE_COLUMNS,
    MODEL_PATH,
    OUTPUTS_DIR,
    REPORT_PATH,
    load_dataset,
    load_metadata,
    load_model,
    load_model_comparison,
    predict_single_rating,
)


st.set_page_config(page_title="Movie Rating Prediction", layout="wide")


def safe_load_model():
    if not MODEL_PATH.exists():
        st.warning("The model artifact is missing. Please run `python train.py` from the project root.")
        return None
    try:
        return load_model(MODEL_PATH)
    except FileNotFoundError:
        st.warning("The model artifact is missing. Please run `python train.py` from the project root.")
        return None


@st.cache_data
def get_dataset_summary():
    try:
        df = load_dataset()
    except Exception:
        return None
    valid_ratings = pd.to_numeric(df["Rating"], errors="coerce").notna() & pd.to_numeric(df["Rating"], errors="coerce").between(0, 10)
    return {
        "rows": len(df),
        "valid_rated_movies": int(valid_ratings.sum()),
        "missing_values": int(df.isna().sum().sum()),
        "model_name": load_metadata().get("selected_model_name", "Not available"),
    }


@st.cache_data
def get_model_metrics():
    metadata = load_metadata()
    metrics = metadata.get("evaluation_metrics", {})
    if not metrics:
        return {}
    return {
        "MAE": metrics.get("Test_MAE"),
        "RMSE": metrics.get("Test_RMSE"),
        "R²": metrics.get("Test_R2"),
    }


with st.sidebar:
    st.title("Navigation")
    selected_section = st.radio(
        "Go to",
        [
            "Home",
            "Movie Rating Prediction",
            "Model Performance",
            "Dataset Insights",
            "About",
        ],
        label_visibility="collapsed",
    )

st.title("Movie Rating Prediction")
st.caption("Machine learning project using the IMDb Indian Movies dataset")

if selected_section == "Home":
    summary = get_dataset_summary()
    if summary is None:
        st.error("Dataset could not be loaded. Please check the data folder and run the project from the root directory.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Dataset size", f"{summary['rows']:,}")
        col2.metric("Valid rated movies", f"{summary['valid_rated_movies']:,}")
        col3.metric("Missing values", f"{summary['missing_values']:,}")
        col4.metric("Selected model", summary["model_name"])

        metrics = get_model_metrics()
        if metrics:
            mcol1, mcol2, mcol3 = st.columns(3)
            mcol1.metric("Test MAE", f"{metrics['MAE']:.3f}")
            mcol2.metric("Test RMSE", f"{metrics['RMSE']:.3f}")
            mcol3.metric("Test R²", f"{metrics['R²']:.3f}")
        else:
            st.info("Training has not been run yet. Use `python train.py` to generate the model and evaluation metrics.")

if selected_section == "Movie Rating Prediction":
    model = safe_load_model()
    if model is None:
        st.stop()

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        year = col1.number_input("Release year", min_value=1888, max_value=2100, value=2020, step=1)
        duration = col2.number_input("Duration (minutes)", min_value=1, max_value=500, value=120, step=1)

        genre = st.text_input("Genre or genres", value="Drama, Romance")
        director = st.text_input("Director", value="Nandita Roy")
        actor1 = st.text_input("Actor 1", value="Amitabh Bachchan")
        actor2 = st.text_input("Actor 2", value="Taapsee Pannu")
        actor3 = st.text_input("Actor 3", value="Ayushmann Khurrana")

        submitted = st.form_submit_button("Predict rating")

    if submitted:
        try:
            input_row = {
                "Year": year,
                "Duration": duration,
                "Genre": genre,
                "Director": director,
                "Actor 1": actor1,
                "Actor 2": actor2,
                "Actor 3": actor3,
            }
            estimate = predict_single_rating(model, input_row)
            metrics = get_model_metrics()
            st.subheader("Estimated rating")
            st.metric("Predicted IMDb rating", f"{estimate:.2f}")
            st.caption("This is an estimate based on the training data and not a guaranteed factual rating.")
            if metrics:
                st.caption(f"Historical test MAE: {metrics['MAE']:.3f}. This is an average error on held-out data, not a confidence interval for a single movie.")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

if selected_section == "Model Performance":
    comparison_df = load_model_comparison()
    if comparison_df.empty:
        st.info("The model has not been trained and evaluated yet. Run `python train.py` first.")
    else:
        st.dataframe(comparison_df, use_container_width=True)
        st.caption("Lower MAE/RMSE is better, while R² closer to 1 is stronger; R² can be negative.")

        if "Actual_vs_predicted" in [p.name for p in OUTPUTS_DIR.iterdir()]:
            pass

if selected_section == "Dataset Insights":
    dataset = load_dataset()
    try:
        if (OUTPUTS_DIR / "rating_distribution.png").exists():
            st.image(str(OUTPUTS_DIR / "rating_distribution.png"), caption="Rating distribution")
        if (OUTPUTS_DIR / "duration_distribution.png").exists():
            st.image(str(OUTPUTS_DIR / "duration_distribution.png"), caption="Duration distribution")
        if (OUTPUTS_DIR / "genre_counts.png").exists():
            st.image(str(OUTPUTS_DIR / "genre_counts.png"), caption="Most common genres")
        if (OUTPUTS_DIR / "genre_average_rating.png").exists():
            st.image(str(OUTPUTS_DIR / "genre_average_rating.png"), caption="Average rating by genre")
        if (OUTPUTS_DIR / "duration_vs_rating.png").exists():
            st.image(str(OUTPUTS_DIR / "duration_vs_rating.png"), caption="Duration versus rating")
        if (OUTPUTS_DIR / "release_year_distribution.png").exists():
            st.image(str(OUTPUTS_DIR / "release_year_distribution.png"), caption="Release-year distribution")
        if (OUTPUTS_DIR / "missing_values_heatmap.png").exists():
            st.image(str(OUTPUTS_DIR / "missing_values_heatmap.png"), caption="Missing values")
    except Exception as exc:
        st.warning(f"Some charts could not be displayed: {exc}")

if selected_section == "About":
    st.markdown(
        """
        This project predicts the IMDb rating of Indian movies using details that are usually known before or at the time of release.

        - Target: Rating (continuous numeric variable)
        - Main features: release year, duration, genre, director, actor 1, actor 2, actor 3
        - Excluded: Vote count and movie title are not used in the primary regression model because they reflect audience response after release or are not reliable predictors.
        - Models compared: DummyRegressor, LinearRegression, DecisionTreeRegressor, RandomForestRegressor, GradientBoostingRegressor
        - Evaluation metrics: MAE, RMSE, R²
        - Model selection strategy: lowest validation MAE using training-only validation before the final test evaluation
        """
    )
