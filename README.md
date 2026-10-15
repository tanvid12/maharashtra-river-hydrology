# Multi-Sensor Remote Sensing & Hydrological Forecasting of Reservoir Dynamics in Maharashtra (2016–2025)

This repository contains the dataset, feature-engineering workflow, and machine-learning pipeline for evaluating reservoir surface-water area prediction using historical water extent and rainfall observations. The broader research context involves satellite remote sensing data from Landsat 8/9 and Sentinel-2 and precipitation data from CHIRPS.

## Project Structure

```text
koyna-reservoir-dynamics/
├── run.py
├── Koyna_Decadal_TimeSeries_2016_2025.csv
├── requirements.txt
├── README.md
└── outputs/
    ├── Koyna_forecast_evaluation.png
    ├── Koyna_model_metrics.csv
    ├── Koyna_feature_importance.csv
    └── Koyna_test_predictions.csv
```

- `run.py`: Feature engineering, chronological train/test splitting, Linear Regression and Random Forest model training, evaluation, and feature-importance calculation.
- `Koyna_Decadal_TimeSeries_2016_2025.csv`: Input time-series dataset containing reservoir water-area and rainfall observations.
- `requirements.txt`: Python dependencies.
- `outputs/Koyna_forecast_evaluation.png`: Model evaluation visualization.
- `outputs/Koyna_model_metrics.csv`: Evaluation metrics for the trained models.
- `outputs/Koyna_feature_importance.csv`: Random Forest feature-importance results.
- `outputs/Koyna_test_predictions.csv`: Actual and predicted water-area values for the test period.

## Installation and Setup

### 1. Clone the repository

Replace `YOUR-USERNAME` with your GitHub username.

```bash
git clone https://github.com/YOUR-USERNAME/koyna-reservoir-dynamics.git
cd koyna-reservoir-dynamics
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Run the pipeline

```bash
python run.py
```

The script generates the evaluation metrics, feature-importance results, test predictions, and visualization in the `outputs/` directory.

## Modeling Workflow

1. Load and validate the input time-series data.
2. Prepare chronological observations and engineer lagged rainfall and water-area features.
3. Construct the next-observation prediction target.
4. Train Linear Regression and Random Forest regression models.
5. Evaluate predictions using R², Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).
6. Export model metrics, predictions, feature importance, and the evaluation plot.

## Evaluation Period

The current implementation uses observations with target dates from 2016–2023 for training and January 2024–November 2025 for testing. The exact periods and sample counts are determined by the data available after preprocessing.

## Reproducibility and Scope

This repository provides a reproducible implementation of the current modeling workflow. Results depend on the supplied dataset, preprocessing, feature definitions, and chronological split.

The current implementation predicts the next available observation using the preceding available observations. If the input series contains missing months, the next observation should not automatically be interpreted as the next calendar month.

The model results represent predictive evaluation, not proof of causal relationships between rainfall and reservoir surface-water extent.

## Research Context

The broader research investigates reservoir surface-water dynamics using satellite-derived water indices and precipitation observations. Confirm that the input dataset and its metadata document the actual satellite products, rainfall source, processing methods, and units used before attributing every observation to a particular sensor or product.

## Citation

If you use this repository in a research paper, cite the repository URL and its release/version where available. A permanent DOI can be added after archiving a release.

## License

No license has been specified yet. Add an appropriate license before permitting reuse or redistribution of the code and data.