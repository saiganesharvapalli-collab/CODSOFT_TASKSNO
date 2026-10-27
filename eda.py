"""Summarize the local Iris dataset and save non-interactive EDA charts."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from model_utils import (
    FEATURE_LABELS,
    ROOT_DIR,
    DatasetError,
    load_dataset,
)


def create_eda_charts(frame, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_paths = []
    palette = "colorblind"

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, feature in zip(axes.flat, FEATURE_LABELS, strict=True):
        sns.histplot(
            data=frame,
            x=feature,
            hue="species",
            kde=True,
            element="step",
            stat="density",
            common_norm=False,
            palette=palette,
            ax=axis,
        )
        axis.set_title(f"{FEATURE_LABELS[feature]} distribution by species")
        axis.set_xlabel(FEATURE_LABELS[feature])
        axis.set_ylabel("Density")
    figure.suptitle("Iris measurement distributions", y=1.02, fontsize=15)
    figure.tight_layout()
    path = output_dir / "feature_distributions.png"
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    chart_paths.append(path)

    figure, axis = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        frame[list(FEATURE_LABELS)].corr(),
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        square=True,
        linewidths=0.5,
        xticklabels=[FEATURE_LABELS[name] for name in FEATURE_LABELS],
        yticklabels=[FEATURE_LABELS[name] for name in FEATURE_LABELS],
        ax=axis,
    )
    axis.set_title("Correlation between Iris measurements")
    figure.tight_layout()
    path = output_dir / "feature_correlation_heatmap.png"
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    chart_paths.append(path)

    scatter_specs = (
        (
            "petal_length",
            "petal_width",
            "Petal length and petal width by species",
            "petal_length_vs_petal_width.png",
        ),
        (
            "sepal_length",
            "sepal_width",
            "Sepal length and sepal width by species",
            "sepal_length_vs_sepal_width.png",
        ),
    )
    for x_feature, y_feature, title, filename in scatter_specs:
        figure, axis = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            data=frame,
            x=x_feature,
            y=y_feature,
            hue="species",
            style="species",
            s=75,
            palette=palette,
            ax=axis,
        )
        axis.set_title(title)
        axis.set_xlabel(FEATURE_LABELS[x_feature])
        axis.set_ylabel(FEATURE_LABELS[y_feature])
        axis.legend(title="Species", loc="best")
        figure.tight_layout()
        path = output_dir / filename
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        chart_paths.append(path)

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, feature in zip(axes.flat, FEATURE_LABELS, strict=True):
        sns.boxplot(
            data=frame,
            x="species",
            y=feature,
            hue="species",
            legend=False,
            palette=palette,
            ax=axis,
        )
        axis.set_title(f"{FEATURE_LABELS[feature]} by species")
        axis.set_xlabel("Species")
        axis.set_ylabel(FEATURE_LABELS[feature])
        axis.tick_params(axis="x", rotation=18)
    figure.suptitle("Measurement ranges by species", y=1.02, fontsize=15)
    figure.tight_layout()
    path = output_dir / "measurement_boxplots_by_species.png"
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    chart_paths.append(path)
    return chart_paths


def main() -> int:
    try:
        dataset = load_dataset()
    except DatasetError as exc:
        print(f"Dataset error: {exc}", file=sys.stderr)
        return 1

    frame = dataset.frame
    print(f"Dataset filename: {dataset.path.name}")
    print(f"Dataset path: {dataset.path}")
    print(f"Shape: {frame.shape}")
    print("\nFirst five records:")
    print(frame.head().to_string(index=False))
    print("\nColumns and data types:")
    print(frame.dtypes.to_string())
    print("\nMissing values by column:")
    print(frame.isna().sum().to_string())
    print(f"\nDuplicate rows (retained): {int(frame.duplicated().sum())}")
    print("\nDescriptive statistics:")
    print(frame[list(FEATURE_LABELS)].describe().to_string())
    species_counts = frame["species"].value_counts()
    is_balanced = species_counts.max() / species_counts.min() <= 1.2
    print("\nSpecies counts:")
    print(species_counts.to_string())
    print(f"Classes reasonably balanced (largest/smallest <= 1.2): {is_balanced}")
    print("\nModel feature order:")
    print(", ".join(dataset.feature_names))
    print(f"Identifier columns excluded: {list(dataset.identifier_columns)}")
    print(f"Other columns ignored for modeling: {list(dataset.ignored_columns)}")

    output_dir = ROOT_DIR / "outputs" / "charts"
    chart_paths = create_eda_charts(frame, output_dir)
    print("\nSaved charts:")
    for chart_path in chart_paths:
        print(f"  {chart_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
