"""Streamlit dashboard for exploring data and predicting Iris species."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from model_utils import (
    FEATURE_LABELS,
    FEATURE_NAMES,
    ROOT_DIR,
    DatasetError,
    find_dataset,
    load_dataset,
    load_metadata,
    load_model,
    predict_probabilities,
    predict_species,
)

st.set_page_config(
    page_title="Iris flower classification",
    page_icon="🌸",
    layout="wide",
)

MODEL_PATH = ROOT_DIR / "models" / "best_model.pkl"
METADATA_PATH = ROOT_DIR / "models" / "model_metadata.json"
COMPARISON_PATH = ROOT_DIR / "reports" / "model_comparison.csv"
CHARTS_DIR = ROOT_DIR / "outputs" / "charts"


@st.cache_data(ttl="10m", max_entries=2, show_spinner="Loading the Iris dataset...")
def cached_dataset(path: str, modified_time: float):
    del modified_time
    return load_dataset(path)


@st.cache_resource(max_entries=2, show_spinner=False)
def cached_model(path: str, modified_time: float):
    del modified_time
    return load_model(path)


@st.cache_data(ttl="10m", max_entries=2, show_spinner=False)
def cached_metadata(path: str, modified_time: float):
    del modified_time
    return load_metadata(path)


@st.cache_data(ttl="10m", max_entries=2, show_spinner=False)
def cached_comparison(path: str, modified_time: float) -> pd.DataFrame:
    del modified_time
    return pd.read_csv(path)


def _file_revision(path: Path) -> float:
    return path.stat().st_mtime if path.is_file() else 0.0


def _species_title(label: str) -> str:
    return label.replace("Iris-", "").replace("_", " ").title()


def _metric_row(metrics: dict) -> None:
    with st.container(horizontal=True):
        st.metric("Accuracy", f"{metrics['accuracy']:.1%}", border=True)
        st.metric(
            "Macro precision", f"{metrics['precision_macro']:.1%}", border=True
        )
        st.metric("Macro recall", f"{metrics['recall_macro']:.1%}", border=True)
        st.metric("Macro F1-score", f"{metrics['f1_macro']:.1%}", border=True)


try:
    dataset_path = find_dataset()
    dataset = cached_dataset(str(dataset_path), _file_revision(dataset_path))
except DatasetError as exc:
    st.error(f"Dataset problem: {exc}")
    st.stop()

model = None
metadata = None
model_error = None
try:
    model = cached_model(str(MODEL_PATH), _file_revision(MODEL_PATH))
    metadata = cached_metadata(str(METADATA_PATH), _file_revision(METADATA_PATH))
except (FileNotFoundError, RuntimeError) as exc:
    model_error = str(exc)

with st.sidebar:
    st.title("Iris classifier")
    page = st.radio(
        "Project sections",
        [
            "Home",
            "Predict species",
            "Dataset explorer",
            "Model performance",
            "About",
        ],
        label_visibility="visible",
    )
    st.caption("A supervised machine-learning project using the local dataset.")

frame = dataset.frame
species_labels = sorted(frame["species"].unique(), key=str.casefold)

if page == "Home":
    st.title("Iris flower classification")
    st.write(
        "Explore the measurements in the provided Iris dataset and use a trained "
        "machine-learning model to estimate a flower's species."
    )
    st.subheader("What does classification mean?")
    st.write(
        "A classifier learns patterns from labeled examples. Here, four flower "
        "measurements are used to predict one of the species present in the dataset."
    )

    with st.container(horizontal=True):
        st.metric("Dataset records", f"{len(frame):,}", border=True)
        st.metric("Input measurements", len(FEATURE_NAMES), border=True)
        st.metric("Recognized species", frame["species"].nunique(), border=True)
        st.metric(
            "Selected model",
            metadata["selected_model"] if metadata else "Not trained",
            border=True,
        )

    st.subheader("Species in this dataset")
    st.write(", ".join(_species_title(label) for label in species_labels))
    st.subheader("How it works")
    st.write(
        "1. The dataset is validated and its measurement columns are identified.  "
        "2. Candidate classifiers are compared using training-only cross-validation.  "
        "3. The selected model is evaluated once on a held-out test set.  "
        "4. Enter four measurements to receive a prediction."
    )
    if metadata:
        st.subheader("Selected model's held-out test results")
        _metric_row(metadata["test_metrics"])
    else:
        st.warning(
            "No trained model is available. Run `python train.py` from the project "
            "folder before viewing model metrics or making predictions."
        )

elif page == "Predict species":
    st.title("Predict an Iris species")
    st.write(
        "Enter the four measurements in the same order used to train the model. "
        "Units are not shown because the dataset's column names do not specify them."
    )
    if model is None or metadata is None:
        st.error(model_error or "The trained model is not available.")
        st.info("Run `python train.py` from the project folder, then refresh this page.")
    else:
        with st.form("prediction_form"):
            columns = st.columns(2)
            measurements = {}
            for position, feature in enumerate(FEATURE_NAMES):
                with columns[position % 2]:
                    measurements[feature] = st.number_input(
                        FEATURE_LABELS[feature],
                        value=float(frame[feature].median()),
                        format="%.2f",
                        step=0.1,
                        key=f"input_{feature}",
                    )
            submitted = st.form_submit_button(
                "Predict species", type="primary", icon=":material/search:"
            )
        if submitted:
            try:
                prediction = predict_species(
                    model, measurements, metadata["feature_names"]
                )
                st.success(f"Predicted species: **{_species_title(prediction)}**")
                probabilities = predict_probabilities(
                    model, measurements, metadata["feature_names"]
                )
                if probabilities:
                    probability_frame = pd.DataFrame(
                        {
                            "Species": [
                                _species_title(label) for label in probabilities
                            ],
                            "Probability": list(probabilities.values()),
                        }
                    )
                    st.subheader("Model class probabilities")
                    st.bar_chart(
                        probability_frame,
                        x="Species",
                        y="Probability",
                        y_label="Probability",
                    )
                st.caption(
                    "This prediction is produced by the trained model and is not "
                    "guaranteed to be correct."
                )
            except ValueError as exc:
                st.error(f"Invalid measurements: {exc}")

elif page == "Dataset explorer":
    st.title("Dataset explorer")
    st.caption(f"Loaded from `{dataset.path}`")
    with st.container(horizontal=True):
        st.metric("Rows", frame.shape[0], border=True)
        st.metric("Columns", frame.shape[1], border=True)
        st.metric("Duplicate rows", int(frame.duplicated().sum()), border=True)
        st.metric("Missing cells", int(frame.isna().sum().sum()), border=True)

    with st.container(border=True):
        st.subheader("First five records")
        st.dataframe(frame.head(), hide_index=True)
    with st.container(border=True):
        st.subheader("Column data types")
        st.dataframe(
            pd.DataFrame(
                {"column": frame.columns.astype(str), "data_type": frame.dtypes.astype(str)}
            ),
            hide_index=True,
        )
    with st.container(border=True):
        st.subheader("Missing values by column")
        st.dataframe(
            frame.isna().sum().rename("missing_values").to_frame(),
            hide_index=False,
        )
    with st.container(border=True):
        st.subheader("Species distribution")
        counts = (
            frame["species"]
            .value_counts()
            .rename_axis("species")
            .reset_index(name="flower_count")
        )
        st.bar_chart(counts, x="species", y="flower_count", y_label="Flowers")
    with st.container(border=True):
        st.subheader("Measurement summary")
        st.dataframe(frame[list(FEATURE_NAMES)].describe().T)

    chart_files = (
        ("feature_distributions.png", "Measurement distributions"),
        ("feature_correlation_heatmap.png", "Feature correlation"),
        ("petal_length_vs_petal_width.png", "Petal measurements by species"),
        ("sepal_length_vs_sepal_width.png", "Sepal measurements by species"),
        ("measurement_boxplots_by_species.png", "Measurement ranges by species"),
    )
    if all((CHARTS_DIR / filename).is_file() for filename, _ in chart_files):
        for start in range(0, len(chart_files), 2):
            chart_columns = st.columns(2)
            for column, (filename, title) in zip(
                chart_columns, chart_files[start : start + 2], strict=False
            ):
                with column:
                    st.subheader(title)
                    st.image(str(CHARTS_DIR / filename))
    else:
        st.info("Run `python eda.py` to generate the dataset visualization charts.")

elif page == "Model performance":
    st.title("Model performance")
    if metadata is None:
        st.warning(
            model_error or "Train the models with `python train.py` to see results."
        )
    else:
        st.write(f"Selected model: **{metadata['selected_model']}**")
        st.caption(metadata["model_selection_criteria"])
        st.subheader("Held-out test-set metrics")
        _metric_row(metadata["test_metrics"])
        st.write(
            "Accuracy is the share of correct predictions. Macro precision measures "
            "prediction correctness per class; macro recall measures class coverage; "
            "macro F1 is the harmonic mean of macro precision and recall."
        )
        st.subheader("Cross-validation versus test results")
        st.write(
            "Cross-validation macro F1 is calculated from folds within the training "
            "set and guides model selection. Test metrics use the separate held-out "
            "test set and are reported only for final evaluation."
        )
        if COMPARISON_PATH.is_file():
            comparison = cached_comparison(
                str(COMPARISON_PATH), _file_revision(COMPARISON_PATH)
            )
            st.dataframe(comparison, hide_index=True)
        else:
            st.warning("The comparison report is missing. Run `python train.py` again.")
        confusion_chart = CHARTS_DIR / "selected_model_confusion_matrix.png"
        if confusion_chart.is_file():
            st.image(str(confusion_chart), caption="Selected model test-set confusion matrix")
        st.subheader("Classification report")
        st.dataframe(
            pd.DataFrame(metadata["test_metrics"]["classification_report"]).T,
            hide_index=False,
        )

elif page == "About":
    st.title("About this project")
    st.write(
        "The objective is to classify Iris flowers from their sepal and petal "
        "measurements using supervised multiclass learning."
    )
    st.subheader("Dataset and schema")
    st.write(
        f"The application uses `{dataset.path.name}` from the local project folder. "
        "Its provenance is not documented in the supplied file. Four measurement "
        "columns are mapped to the canonical input order; the species label is the "
        "target. Identifier and other unrelated columns are never model inputs."
    )
    st.write(", ".join(FEATURE_LABELS[name] for name in FEATURE_NAMES))
    st.subheader("Algorithms evaluated")
    st.write(
        "Dummy classifier, logistic regression, K-nearest neighbors, decision tree, "
        "random forest, and support vector machine."
    )
    st.subheader("Preprocessing and evaluation")
    st.write(
        "Missing measurement values are median-imputed inside each model pipeline. "
        "Imputation and scaling are fitted only on training data. Logistic regression, "
        "KNN, and SVM use standardized measurements; tree models do not. Macro "
        "precision, recall, and F1 give each species equal weight."
    )
    st.subheader("Limitations and future work")
    st.write(
        "This is a small, classic dataset and its provenance and measurement units "
        "are not specified by the supplied file. Test metrics can vary with a "
        "different split and are not a guarantee for future flowers. Future work "
        "could add external validation, more labeled samples, and model calibration."
    )
