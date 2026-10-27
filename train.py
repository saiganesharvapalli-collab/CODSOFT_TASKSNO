"""Train, compare, evaluate, and persist the Iris classifiers."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from model_utils import (
    ROOT_DIR,
    DatasetError,
    load_dataset,
    save_metadata,
    save_model,
    train_models,
)


def create_training_charts(result, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = result.metadata["test_metrics"]
    display = ConfusionMatrixDisplay(
        confusion_matrix=np.asarray(metrics["confusion_matrix"]),
        display_labels=result.metadata["class_labels"],
    )
    display.plot(cmap="Blues", colorbar=False, values_format="d")
    plt.title(f"Test-set confusion matrix — {result.selected_model_name}")
    plt.xlabel("Predicted species")
    plt.ylabel("Actual species")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "selected_model_confusion_matrix.png", dpi=160)
    plt.close()

    classifier = result.selected_model.named_steps["classifier"]
    importances = getattr(classifier, "feature_importances_", None)
    if importances is not None:
        importance_frame = pd.DataFrame(
            {"feature": result.metadata["feature_names"], "importance": importances}
        ).sort_values("importance", ascending=True)
        axis = importance_frame.plot.barh(
            x="feature",
            y="importance",
            legend=False,
            color="#2878a5",
            figsize=(8, 4.5),
        )
        axis.set_title(f"Feature importance — {result.selected_model_name}")
        axis.set_xlabel("Relative importance")
        axis.set_ylabel("Measurement")
        plt.tight_layout()
        plt.savefig(output_dir / "selected_model_feature_importance.png", dpi=160)
        plt.close()


def main() -> int:
    try:
        dataset = load_dataset()
        print(f"Dataset: {dataset.path.name}")
        print(f"Path: {dataset.path}")
        print(f"Shape: {dataset.frame.shape[0]} rows x {dataset.frame.shape[1]} columns")
        print(f"Features: {', '.join(dataset.feature_names)}")
        print(f"Target: {dataset.target_column}")
        print(f"Species counts: {dataset.frame['species'].value_counts().to_dict()}")

        result = train_models(dataset)
        models_dir = ROOT_DIR / "models"
        reports_dir = ROOT_DIR / "reports"
        charts_dir = ROOT_DIR / "outputs" / "charts"
        models_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)

        create_training_charts(result, charts_dir)
        result.comparison.sort_values(
            by=["cv_macro_f1_mean", "test_f1_macro"],
            ascending=[False, False],
            na_position="last",
        ).to_csv(reports_dir / "model_comparison.csv", index=False)
        save_model(result.selected_model, models_dir / "best_model.pkl")
        save_metadata(result.metadata, models_dir / "model_metadata.json")

        print(f"\nSelected model: {result.selected_model_name}")
        print(f"Selection criteria: {result.metadata['model_selection_criteria']}")
        print("\nActual held-out test results (macro precision/recall/F1):")
        print(
            result.comparison[
                [
                    "model",
                    "cv_macro_f1_mean",
                    "test_accuracy",
                    "test_precision_macro",
                    "test_recall_macro",
                    "test_f1_macro",
                ]
            ].to_string(index=False, float_format=lambda value: f"{value:.3f}")
        )
        print(f"\nSaved model: {models_dir / 'best_model.pkl'}")
        print(f"Saved metadata: {models_dir / 'model_metadata.json'}")
        print(f"Saved comparison: {reports_dir / 'model_comparison.csv'}")
        print(f"Saved performance charts: {charts_dir}")
        return 0
    except DatasetError as exc:
        print(f"Dataset error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
