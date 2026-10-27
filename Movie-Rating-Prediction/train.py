from __future__ import annotations

from model_utils import load_dataset, train_and_score_models


def main() -> None:
    dataset = load_dataset()
    results = train_and_score_models(dataset)
    print(f"Selected model: {results['selected_model_name']}")
    print(results["metrics"])


if __name__ == "__main__":
    main()
