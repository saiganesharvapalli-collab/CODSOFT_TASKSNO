import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from model_utils import load_dataset, load_model, load_raw_dataset, predict_passenger
from train import train_and_save


class ModelWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary_directory = tempfile.TemporaryDirectory()
        cls.model_path = Path(cls.temporary_directory.name) / "titanic_model.pkl"
        train_and_save(cls.model_path)

    @classmethod
    def tearDownClass(cls):
        cls.temporary_directory.cleanup()

    def test_saved_model_loads_with_validation_results(self):
        artifact = load_model(self.model_path)
        self.assertIn(artifact["model_name"], [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
        ])
        self.assertEqual(len(artifact["model_comparison"]), 3)
        self.assertIn("roc_auc", artifact["validation_metrics"])

    def test_loaded_model_predicts_validated_passenger(self):
        artifact = load_model(self.model_path)
        passenger = {
            "pclass": 1,
            "sex": "female",
            "age": 32.0,
            "sibsp": 0,
            "parch": 0,
            "fare": 80.0,
            "embarked": "C",
        }

        prediction, probability = predict_passenger(artifact["model"], passenger)

        self.assertIn(prediction, (0, 1))
        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)

    def test_prediction_without_probability_support(self):
        class PredictionOnlyModel:
            def predict(self, features):
                return [1]

        passenger = {
            "pclass": 1,
            "sex": "female",
            "age": 32.0,
            "sibsp": 0,
            "parch": 0,
            "fare": 80.0,
            "embarked": "C",
        }

        prediction, probability = predict_passenger(
            PredictionOnlyModel(), passenger
        )

        self.assertEqual(prediction, 1)
        self.assertIsNone(probability)

    def test_prediction_rejects_invalid_passenger_values(self):
        artifact = load_model(self.model_path)
        passenger = {
            "pclass": 4,
            "sex": "female",
            "age": 32.0,
            "sibsp": 0,
            "parch": 0,
            "fare": 80.0,
            "embarked": "C",
        }

        with self.assertRaisesRegex(ValueError, "Passenger class"):
            predict_passenger(artifact["model"], passenger)

    def test_training_rejects_non_binary_target_labels(self):
        dataset = load_raw_dataset()
        dataset["survived"] = dataset["survived"].astype(float)
        dataset.loc[0, "survived"] = 0.5

        with patch("model_utils.load_raw_dataset", return_value=dataset):
            with self.assertRaisesRegex(ValueError, "only 0 and 1"):
                load_dataset()

    def test_missing_model_has_a_clear_error(self):
        missing_path = Path(self.temporary_directory.name) / "missing.pkl"
        with self.assertRaisesRegex(FileNotFoundError, "Run `python train.py` first"):
            load_model(missing_path)

    def test_streamlit_views_and_prediction_render(self):
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path), default_timeout=20).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.header[0].value, "Predict Titanic Survival")
        self.assertTrue(app.info)

        app.radio[0].set_value("Model performance").run()
        self.assertFalse(app.exception)
        self.assertEqual(app.header[0].value, "Model Performance")
        self.assertEqual(len(app.metric), 2)
        self.assertEqual(len(app.dataframe), 1)

        app.radio[0].set_value("Dataset insights").run()
        self.assertFalse(app.exception)
        self.assertEqual(app.header[0].value, "Dataset Insights")
        dataset = load_raw_dataset()
        survivor_count = int(dataset["survived"].sum())
        expected_metrics = [
            f"{len(dataset):,}",
            f"{survivor_count:,}",
            f"{len(dataset) - survivor_count:,}",
            f"{dataset['survived'].mean():.1%}",
        ]
        self.assertEqual([metric.value for metric in app.metric], expected_metrics)

        app.radio[0].set_value("Survival prediction").run()
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.metric), 2)
        probabilities = [float(metric.value.rstrip("%")) for metric in app.metric]
        self.assertAlmostEqual(sum(probabilities), 100.0, delta=0.2)
        self.assertTrue(app.success or app.warning)
        self.assertEqual(len(app.dataframe), 1)


if __name__ == "__main__":
    unittest.main()