from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
	accuracy_score,
	confusion_matrix,
	f1_score,
	precision_score,
	recall_score,
	roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from model_utils import MODEL_PATH, build_pipeline, load_dataset


def build_model_candidates():
	return {
		"Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
		"Decision Tree": DecisionTreeClassifier(random_state=42),
		"Random Forest": RandomForestClassifier(
			n_estimators=300,
			random_state=42,
			n_jobs=-1,
		),
	}


def train_and_save(output_path=MODEL_PATH):
	features, target = load_dataset()
	train_features, validation_features, train_target, validation_target = (
		train_test_split(
			features,
			target,
			test_size=0.2,
			random_state=42,
			stratify=target,
		)
	)

	results = []
	validation_confusion_matrices = {}
	candidates = build_model_candidates()

	for model_name, estimator in candidates.items():
		pipeline = build_pipeline(clone(estimator))
		pipeline.fit(train_features, train_target)
		predictions = pipeline.predict(validation_features)
		probabilities = pipeline.predict_proba(validation_features)[:, 1]

		results.append(
			{
				"model_name": model_name,
				"accuracy": float(accuracy_score(validation_target, predictions)),
				"precision": float(
					precision_score(validation_target, predictions, zero_division=0)
				),
				"recall": float(
					recall_score(validation_target, predictions, zero_division=0)
				),
				"f1": float(f1_score(validation_target, predictions, zero_division=0)),
				"roc_auc": float(roc_auc_score(validation_target, probabilities)),
			}
		)
		validation_confusion_matrices[model_name] = confusion_matrix(
			validation_target,
			predictions,
			labels=[0, 1],
		).tolist()

	results.sort(key=lambda result: (result["roc_auc"], result["f1"]), reverse=True)
	selected_name = results[0]["model_name"]

	# Refit the selected pipeline on all available rows after validation is complete.
	final_model = build_pipeline(clone(candidates[selected_name]))
	final_model.fit(features, target)

	artifact = {
		"model": final_model,
		"model_name": selected_name,
		"features": features.columns.tolist(),
		"model_comparison": results,
		"validation_metrics": results[0],
		"validation_confusion_matrices": validation_confusion_matrices,
		"validation": {
			"strategy": "Stratified 80/20 hold-out split",
			"test_size": 0.2,
			"random_state": 42,
			"selection_rule": "Highest validation ROC-AUC; F1-score breaks ties",
		},
	}

	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	joblib.dump(artifact, output_path)

	print(f"Saved final model: {selected_name}")
	print(f"Artifact: {output_path}")
	print("Validation results:")
	print(
		pd.DataFrame(results).to_string(
			index=False,
			float_format=lambda value: f"{value:.3f}",
		)
	)
	return artifact


if __name__ == "__main__":
	train_and_save()