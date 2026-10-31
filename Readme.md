# Medical Insurance Cost Prediction

An end-to-end Machine Learning project that predicts individual medical insurance charges using customer demographic and lifestyle information.

The project explores multiple regression algorithms, evaluates their performance using cross-validation, and selects a Random Forest model with a logarithmically transformed target variable.

## Project Overview

Medical insurance charges can vary significantly depending on factors such as age, BMI, smoking habits, and other demographic characteristics.

The objective of this project is to develop a regression model that estimates medical insurance charges based on available customer information.

**Problem type:** Supervised Machine Learning — Regression

**Selected algorithm:** Random Forest Regressor

**Target transformation:** Logarithmic transformation using `log1p` and `expm1`

**Primary evaluation metric:** Mean Absolute Error (MAE)

## Dataset

The project uses the Medical Cost Personal Dataset available on Kaggle.

Dataset: [Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)

The dataset contains information about insurance beneficiaries.

| Feature | Description |
|---|---|
| age | Age of the beneficiary |
| sex | Gender of the beneficiary |
| bmi | Body Mass Index |
| children | Number of dependent children |
| smoker | Smoking status |
| region | Residential region |
| charges | Individual medical insurance charges (target) |

## Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the target distribution, identify patterns, and investigate relationships between the features and insurance charges.

Key observations:

- The target variable `charges` exhibits a positively skewed distribution.
- Smoking status has a strong association with insurance charges.
- Smokers generally have substantially higher insurance charges than non-smokers.
- Age and BMI show positive associations with insurance charges.
- Potential high-charge outliers were investigated rather than automatically removed, as they may represent genuine observations.
- The relationship between age and charges was also examined across smoking categories.

Visualizations included target distribution analysis, boxplots, scatterplots, and correlation analysis.

## Data Preprocessing

The following preprocessing steps were implemented:

- Duplicate record removal.
- Train-test split using an 80:20 ratio.
- Numerical features retained through a numerical preprocessing pipeline.
- Categorical features encoded using `OneHotEncoder`.
- Preprocessing and model training integrated into a Scikit-learn pipeline.

The fitted preprocessing pipeline is included in the serialized model artifact.

## Model Development

Multiple regression algorithms were evaluated:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Polynomial Regression
- Decision Tree Regression
- Random Forest Regression
- Gradient Boosting Regression

Hyperparameter tuning was performed for the tree-based models using `GridSearchCV`.

Five-fold cross-validation with shuffled `KFold` splits was used for model evaluation.

### Target Transformation Experiment

Since insurance charges exhibited positive skewness, a logarithmic target transformation was investigated.

`TransformedTargetRegressor` was used to:

1. Apply `log1p` to the target during model training.
2. Automatically apply `expm1` to predictions.
3. Evaluate predictions on the original insurance-charge scale.

The log-transformed Random Forest achieved a lower cross-validation MAE than the original-target Random Forest.

### Model Selection: Original vs Log-Transformed Target

The Random Forest model trained with a log-transformed target (`log1p`) was selected over the original-target Random Forest.

Across five-fold cross-validation:

- Mean MAE improved from **2,547.04 to 2,110.82**, representing approximately a **17.1% reduction**.
- MAE improved in all five validation folds, indicating that the improvement was consistent across the evaluated splits.
- Mean RMSE remained nearly unchanged (4,579.47 vs 4,580.19).
- Mean R² remained approximately 0.844 for both models.

Although the log-transformed model showed slightly higher fold-to-fold variability in MAE, its consistently lower absolute errors justified the selection.

The model was therefore selected based primarily on its improved MAE, while acknowledging that large prediction errors remain an area for improvement.

## Model Evaluation

### Five-fold Cross-validation Results

Selected Random Forest configuration with logarithmic target transformation:

| Metric | Mean | Standard Deviation |
|---|---:|---:|
| MAE | 2,110.82 | 113.62 |
| RMSE | 4,580.19 | 218.803 |
| R² | 0.844 | 0.026 |

The original-target Random Forest achieved a cross-validation MAE of 2,547.04.

The log-transformed model reduced cross-validation MAE by approximately 17.12%.

### Exploratory Test-set Results

| Metric | Original-target RF | Log-transformed RF |
|---|---:|---:|
| MAE | 2,384.14 | 1,937.33 |
| RMSE | 4,203.37 | 4,186.49 |
| R² | 0.9038 | 0.9046 |

The log-transformed model achieved a lower test-set MAE.

**Evaluation note:** The test set was examined during earlier experimentation and model comparisons. Therefore, these test metrics should be treated as exploratory rather than as an untouched, independent estimate of generalization performance. Cross-validation was used for model selection.

## Residual Analysis

Residual analysis was conducted to investigate prediction errors.

Residuals were calculated as:

`Residual = Actual Charges - Predicted Charges`

Key observations:

- The model performs reasonably well for many lower-charge observations.
- Prediction errors increase for certain higher-charge observations.
- The model tends to underestimate charges in some higher-charge ranges.
- The ₹20,000–₹30,000 charge group showed particularly high prediction errors in the exploratory test-set analysis.

The logarithmic target transformation improved overall MAE but did not eliminate all prediction errors.

## Project Structure

```text
medical-insurance-cost-prediction/
│
├── notebooks/
│   └── medical_insurance_cost_prediction.ipynb
│
├── models/
│   └── insurance_cost_pipeline.joblib
│
├── src/
│   └── predict.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation and Setup

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/medical-insurance-cost-prediction.git

cd medical-insurance-cost-prediction
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Making Predictions

The trained model is saved as a Joblib artifact.

You can generate predictions using the reusable prediction function:

```python
from src.predict import predict_insurance_cost

prediction = predict_insurance_cost(
    age=35,
    sex="male",
    bmi=28.5,
    children=1,
    smoker="no",
    region="northwest"
)

print(f"Predicted insurance charge: {prediction:,.2f}")
```

The model automatically handles preprocessing and inverse transformation of the predicted target.

## Reproducing the Project

To reproduce the experiments:

1. Clone this repository.
2. Install the dependencies listed in `requirements.txt`.
3. Download the dataset from Kaggle.
4. Open the notebook in Jupyter Notebook or VS Code.
5. Run the notebook cells sequentially to reproduce EDA, preprocessing, model development, and evaluation.

The saved model artifact can be used directly for inference without retraining.

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Jupyter Notebook
- Git and GitHub

## Limitations and Future Improvements

- The dataset contains a limited number of observations.
- Available features may not capture all factors influencing individual insurance charges.
- The model has higher prediction errors for certain expensive insurance cases.
- The dataset is observational, so relationships between features and charges should not be interpreted as causal.
- Further evaluation on an untouched external dataset would provide stronger evidence of generalization.

Potential future improvements include model deployment through FastAPI, integration with a frontend application, and evaluation on additional datasets.

## Disclaimer

This project is intended for educational and machine learning demonstration purposes only. Predictions are estimates based on the supplied dataset and should not be considered actual insurance quotations or financial advice.
