# Movie Rating Prediction

## Overview
This project uses the IMDb Indian Movies dataset to predict a movie's IMDb rating from information that is typically known before or around release time, such as release year, duration, genre, director, and lead actors. The project includes data cleaning, exploratory analysis, model comparison, evaluation, serialization, and a beginner-friendly Streamlit dashboard.

## Project objective
The aim is to build a regression model that estimates an Indian movie's rating on the IMDb scale. The model uses only practical pre-release or release-era features and avoids using vote count as the main predictor because votes reflect audience response after release.

## Dataset description
- Source: Kaggle IMDb Indian Movies dataset
- File: `data/IMDb Movies India.csv`
- Shape: 15,509 rows x 10 columns
- Columns: `Name`, `Year`, `Duration`, `Genre`, `Rating`, `Votes`, `Director`, `Actor 1`, `Actor 2`, `Actor 3`
- Target variable: `Rating`
- Validation requirement: the CSV is read with `encoding="latin1"` because UTF-8 decoding fails on some characters.

## Features used for prediction
The primary model uses:
- `Year`
- `Duration`
- `Genre`
- `Director`
- `Actor 1`
- `Actor 2`
- `Actor 3`

Columns excluded from the main prediction model:
- `Name`: movie title is treated as an identifier rather than a strong predictive feature.
- `Votes`: excluded because vote counts are an audience-activity signal after the movie has already been released and can distort a pre-release prediction task.
- `Rating`: never used as an input feature.

## Project architecture
- `app.py`: Streamlit dashboard for prediction and overview.
- `train.py`: model training and evaluation workflow.
- `eda.py`: exploratory data analysis and output plots.
- `model_utils.py`: dataset validation, cleaning, preprocessing, training, evaluation, serialization, and prediction logic.
- `tests/test_model.py`: automated tests for dataset integrity and model behavior.
- `models/best_model.pkl`: saved trained pipeline.
- `models/model_metadata.json`: evaluation metadata and model details.
- `reports/model_comparison.csv`: all model comparison metrics.
- `outputs/`: generated EDA charts and visuals.

## Folder structure
```text
Movie-Rating-Prediction/
├── app.py
├── train.py
├── eda.py
├── model_utils.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── IMDb Movies India.csv
├── models/
│   ├── best_model.pkl
│   └── model_metadata.json
├── reports/
│   └── model_comparison.csv
├── outputs/
│   └── generated charts
├── tests/
│   └── test_model.py
└── .venv/
```

## Algorithms compared
- DummyRegressor (baseline)
- LinearRegression
- DecisionTreeRegressor
- RandomForestRegressor
- GradientBoostingRegressor

## Preprocessing methodology
- Read dataset with `encoding="latin1"`.
- Validate the expected schema before processing.
- Remove duplicate rows.
- Parse `Year` values from strings like `(2019)`.
- Parse `Duration` values from strings like `109 min`.
- Convert `Rating` to numeric and drop invalid or missing target values.
- Fill missing categorical values with `Unknown`.
- Encode genre with a multi-label representation instead of treating the entire combined genre string as one category.
- Use separate train/test splits before fitting preprocessing.
- Fit all preprocessing and model steps only on the training data.

## Evaluation metrics
The model is evaluated with:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² score

Lower MAE and RMSE are better. R² can be negative; it is not a percentage accuracy metric.

## Actual measured results
The real training run completed successfully using the actual dataset. The selected model was `GradientBoostingRegressor`.

Validation metrics:
- Validation MAE: 0.9734
- Validation RMSE: 1.2464
- Validation R²: 0.1913

Test metrics:
- Test MAE: 0.9314
- Test RMSE: 1.1958
- Test R²: 0.2308

Baseline comparison:
- DummyRegressor test MAE: 1.1072
- Selected model test MAE: 0.9314
- Improvement: yes, the selected model improved the baseline.

## Windows PowerShell installation
```powershell
cd C:\Users\saiga\OneDrive\Desktop\Movie-Rating-Prediction
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Dataset placement
Place the Kaggle CSV in the project folder using this exact path:
```text
data/IMDb Movies India.csv
```
If the file is missing, the project stops with a clear error instead of fabricating data.

## Training command
```powershell
python train.py
```

## EDA command
```powershell
python eda.py
```

## Testing command
```powershell
pytest -q
```

## Streamlit launch command
```powershell
streamlit run app.py
```

## Example usage
1. Activate the virtual environment.
2. Put the dataset in `data/IMDb Movies India.csv`.
3. Run `python train.py` to generate the model and evaluation files.
4. Run `streamlit run app.py`.
5. Open the prediction page and enter values like:
   - Year: `2020`
   - Duration: `120`
   - Genre: `Drama, Romance`
   - Director: `Nandita Roy`
   - Actor 1: `Amitabh Bachchan`
   - Actor 2: `Taapsee Pannu`
   - Actor 3: `Ayushmann Khurrana`
6. Submit the form to get an estimated rating.

## Limitations
- The dataset is historical and may reflect selection bias and genre patterns in Indian cinema.
- Some movies are missing ratings, directors, actors, or duration information.
- Movie ratings are subjective and depend on audience preferences and review culture.
- The model is a historical prediction tool, not a guarantee of real-world film success.
- Random splitting may not fully reflect a director/actor leakage concern; group-aware or time-aware validation could be added in future work.

## Future improvements
- Add group-aware validation by director or actor.
- Test a time-aware split for a more realistic evaluation.
- Add a richer feature set, including cast popularity or production metadata when available.
- Compare more advanced models and hyperparameter tuning.
- Add an explicit training report notebook for presentation use.

## GitHub upload instructions
1. Open PowerShell in the project folder.
2. Run:
```powershell
git init
git add .
git commit -m "Initial movie rating prediction project"
git branch -M main
git remote add origin https://github.com/<your-user>/<your-repo>.git
git push -u origin main
```
3. Check the size of the dataset and the trained model before uploading large files. This project keeps the data file and model artifact local unless you intentionally want to add them to a public repository.

## Machine learning workflow summary
The basic workflow is:
1. Load and validate the data.
2. Clean and transform the raw features.
3. Split the data into training and test sets.
4. Train multiple regression models.
5. Compare results using validation and held-out test metrics.
6. Save the best pipeline.
7. Use the saved model to make predictions on new data.
8. Review error metrics and explain the model's strengths and limitations.

This workflow is useful in interviews or presentations because it shows how data cleaning, preprocessing, evaluation, and deployment work together in a complete ML project.
