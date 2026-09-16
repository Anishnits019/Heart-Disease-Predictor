# Cardiovascular Disease Predictor

A machine learning project that estimates the probability of cardiovascular disease from demographic, examination, and lifestyle information. The repository contains a modular training pipeline, persisted preprocessing and model artifacts, and a Streamlit dashboard for interactive predictions.

> **Important:** This application is for educational and demonstration purposes only. Its output is not a medical diagnosis and must not replace evaluation by a qualified healthcare professional.

## Features

- End-to-end data pipeline for ingestion, validation, feature extraction, transformation, and model training
- CatBoost classification model trained for cardiovascular disease prediction
- Persisted preprocessing pipeline reused during inference
- Streamlit dashboard with probability and low, moderate, or high risk bands
- Model evaluation results and generated training artifacts stored in the repository

## Project Structure

```text
.
├── app.py                         # Streamlit prediction dashboard
├── main.py                        # Training pipeline entry point
├── model.pkl                      # Root-level trained model used by the app
├── requirements.txt               # Python dependencies
├── data_schema/schema.yaml        # Dataset schema and column definitions
├── cardio/
│   ├── components/                # Ingestion, validation, extraction, transformation, training
│   ├── entity/                    # Pipeline configuration and artifact entities
│   ├── exception/                 # Custom exception handling
│   ├── logging/                   # Application logging
│   └── utils/                     # Serialization, evaluation, and metrics helpers
├── Artifacts/                     # Timestamped pipeline outputs
│   └── <timestamp>/
│       ├── data_feature_extraction/
│       ├── data_ingestion/
│       ├── data_transformation/
│       │   └── transformed_object/preprocessing.pkl
│       └── model_trainer/trained_model/model.pkl
├── Model_Result/                  # Model evaluation output
└── Notebooks/                     # Exploratory analysis and visualizations
```

## Machine Learning Pipeline

The training workflow is started from `main.py` and runs through these stages:

1. **Data ingestion**: loads the cardiovascular dataset and creates train/test splits.
2. **Data validation**: checks the input data against `data_schema/schema.yaml` and generates drift reports.
3. **Feature extraction**: derives BMI, pulse pressure, mean arterial pressure, blood-pressure indicators, lifestyle indicators, age bins, and BMI categories.
4. **Data transformation**: applies the persisted preprocessing contract used by the model.
5. **Model training**: evaluates a CatBoost classifier and saves the trained model.

## Preprocessing Contract

The dashboard does not fit preprocessing again. It loads `preprocessing.pkl` and calls `transform()` on the user input, which prevents data leakage and keeps inference consistent with training.

The model input contains these 13 fields in the training order:

```text
age, height, weight, ap_hi, ap_lo, bmi,
gender, cholesterol, gluc, smoke, alco, active,
bmi_category
```

The persisted transformer applies:

- `RobustScaler` to the numeric fields
- `OrdinalEncoder` to `bmi_category`
- `KNNImputer` after column transformation

The dashboard calculates BMI and BMI category from the raw user inputs before applying this transformer. Age is entered in years, matching the values in the feature-extraction dataset.

## Installation

Create and activate a virtual environment from the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On macOS, the project environment can also be invoked directly:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

## Run the Dashboard

Start the Streamlit application from the project root:

```bash
.venv/bin/streamlit run app.py
```

Streamlit will display a local URL, normally:

```text
http://localhost:8501
```

Enter the patient information, select **Predict Cardiovascular Risk Probability**, and review the probability returned by the trained model.

## Run the Training Pipeline

To retrain the pipeline and generate a new timestamped artifact set:

```bash
.venv/bin/python main.py
```

The generated artifacts are written below `Artifacts/<timestamp>/`. The dashboard looks for the root-level `model.pkl` first and otherwise searches the timestamped model artifacts. It uses the most recent available preprocessing artifact.

## Model Output

The dashboard reports the class-1 probability from `predict_proba()`. The current presentation bands are:

- **Low risk**: below 35%
- **Moderate risk**: 35% to below 65%
- **High risk**: 65% or above

These bands are presentation thresholds, not clinically validated risk categories.

## Verification

A representative inference check confirmed that the persisted preprocessing pipeline produces a 13-feature transformed input and that the trained model returns a probability successfully.

## Development Notes

- Keep the model and preprocessing artifact versions aligned when deploying a new model.
- Do not call `fit()` or `fit_transform()` from the prediction path.
- Install `catboost` and `scikit-learn` in the runtime environment because they are required to deserialize and execute the trained artifacts.
- Avoid committing secrets from `.env` files or database configuration.
