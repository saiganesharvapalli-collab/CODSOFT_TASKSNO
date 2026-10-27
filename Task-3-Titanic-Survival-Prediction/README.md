# Titanic Survival Prediction

A small machine-learning project that compares three classifiers on the Titanic passenger dataset and provides an interactive prediction and evaluation dashboard.

## Project files

- `data/Titanic-Dataset.csv`: the source dataset used for training and analysis.
- `model_utils.py`: shared dataset loading, preprocessing, artifact loading, validation, and prediction functions.
- `train.py`: validation comparison and final model training.
- `app.py`: Streamlit prediction form, model dashboard, and dataset visualizations.
- `eda.py`: standalone exploratory data visualizations.
- `tests/test_model.py`: automated model loading, prediction, and input validation tests.

## Setup

From the project directory, create and activate a virtual environment, then install the project dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run

Train and compare the models, then start the dashboard:

```powershell
python train.py
streamlit run app.py
```

Run the exploratory charts or automated tests with:

```powershell
python eda.py
python -m unittest discover -s tests -v
```

## Modeling and validation

The training script compares Logistic Regression, Decision Tree, and Random Forest using the same stratified 80/20 train/validation split (`random_state=42`). It reports accuracy, precision, recall, F1-score, ROC-AUC, and a confusion matrix for each model. The selected model is the one with the highest validation ROC-AUC; F1-score breaks a tie.

The validation results are used for model selection, so they are not an independent final test estimate. After selection, the chosen pipeline is refit on all available rows and saved to `models/titanic_model.pkl`. The dashboard shows the saved validation results and confusion matrix, not scores recalculated on the refitted model.

Training and prediction use the same scikit-learn pipeline: numeric columns are median-imputed and scaled, while categorical columns are imputed with the most frequent value and one-hot encoded. Unknown categories are ignored by the encoder, and the trained pipeline is saved with the classifier so inference cannot accidentally skip or change preprocessing.

The dashboard's dataset charts and model scores are computed from the included CSV when training runs; no external or generated passenger records are used.