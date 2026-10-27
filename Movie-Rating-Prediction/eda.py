from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from model_utils import OUTPUTS_DIR, describe_missing_values, load_dataset


def ensure_outputs_dir() -> Path:
    OUTPUTS_DIR.mkdir(exist_ok=True, parents=True)
    return OUTPUTS_DIR


def save_plot(path: Path, fig) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def generate_eda_plots() -> None:
    df = load_dataset()
    ensure_outputs_dir()

    print(f"Dataset shape: {df.shape}")
    print("Column names:", list(df.columns))
    print("Data types:\n", df.dtypes)
    print("Missing values:\n", describe_missing_values(df))
    print(f"Duplicate rows: {df.duplicated().sum()}")

    rating = pd.to_numeric(df["Rating"], errors="coerce")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(rating.dropna(), bins=30, kde=True, ax=ax)
    ax.set_title("IMDb rating distribution")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    save_plot(OUTPUTS_DIR / "rating_distribution.png", fig)

    year = pd.to_numeric(df["Year"].astype(str).str.extract(r"(\d{4})")[0], errors="coerce")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(year.dropna(), bins=30, ax=ax)
    ax.set_title("Movie release year distribution")
    ax.set_xlabel("Year")
    ax.set_ylabel("Count")
    save_plot(OUTPUTS_DIR / "release_year_distribution.png", fig)

    duration = df["Duration"].astype(str).str.extract(r"(\d+)")[0]
    duration = pd.to_numeric(duration, errors="coerce")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(duration.dropna(), bins=30, ax=ax)
    ax.set_title("Movie duration distribution")
    ax.set_xlabel("Duration (minutes)")
    ax.set_ylabel("Count")
    save_plot(OUTPUTS_DIR / "duration_distribution.png", fig)

    genre_counts = df["Genre"].fillna("Unknown").astype(str).str.split(",").explode().str.strip().value_counts().head(12)
    fig, ax = plt.subplots(figsize=(10, 6))
    genre_counts.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title("Most common genres")
    ax.set_xlabel("Genre")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    save_plot(OUTPUTS_DIR / "genre_counts.png", fig)

    genre_rating = df[["Genre", "Rating"]].copy()
    genre_rating["Genre"] = genre_rating["Genre"].fillna("Unknown").astype(str)
    genre_rating = genre_rating.assign(Genre=genre_rating["Genre"].str.split(",")).explode("Genre")
    genre_rating["Genre"] = genre_rating["Genre"].str.strip()
    genre_rating = genre_rating.dropna(subset=["Rating"])
    genre_rating = genre_rating.groupby("Genre").filter(lambda g: len(g) >= 5)
    genre_summary = genre_rating.groupby("Genre")["Rating"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    genre_summary.head(12).plot(kind="bar", ax=ax, color="darkorange")
    ax.set_title("Average rating by genre (minimum 5 ratings)")
    ax.set_xlabel("Genre")
    ax.set_ylabel("Average rating")
    plt.xticks(rotation=45, ha="right")
    save_plot(OUTPUTS_DIR / "genre_average_rating.png", fig)

    duration_rating = df[["Duration", "Rating"]].copy()
    duration_rating["Duration"] = pd.to_numeric(duration_rating["Duration"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    duration_rating = duration_rating.dropna()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=duration_rating, x="Duration", y="Rating", alpha=0.3, ax=ax)
    ax.set_title("Duration vs. rating")
    ax.set_xlabel("Duration (minutes)")
    ax.set_ylabel("Rating")
    save_plot(OUTPUTS_DIR / "duration_vs_rating.png", fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(df.isna(), cbar=False, ax=ax)
    ax.set_title("Missing values heatmap")
    save_plot(OUTPUTS_DIR / "missing_values_heatmap.png", fig)

    print("EDA charts saved to:", OUTPUTS_DIR)


if __name__ == "__main__":
    generate_eda_plots()
