
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error
)


def run_reservoir_pipeline(csv_path, reservoir_name="Koyna"):

    print(f"\n{'=' * 20} Processing: {reservoir_name} {'=' * 20}")

    # 1. Load and validate the dataset
    df = pd.read_csv(csv_path)

    required_columns = ["date", "water_area_km2", "rainfall_mm"]
    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing CSV columns: {sorted(missing_columns)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["water_area_km2"] = pd.to_numeric(
        df["water_area_km2"], errors="coerce"
    )
    df["rainfall_mm"] = pd.to_numeric(
        df["rainfall_mm"], errors="coerce"
    )

    df = (
        df.dropna(subset=required_columns)
        .sort_values("date")
        .drop_duplicates(subset="date")
        .reset_index(drop=True)
    )

    # 2. Engineer features and next-period target
    df["Area_t"] = df["water_area_km2"]
    df["Area_t_1"] = df["water_area_km2"].shift(1)

    df["Rain_Lag0"] = df["rainfall_mm"]
    df["Rain_Lag1"] = df["rainfall_mm"].shift(1)
    df["Rain_Lag2"] = df["rainfall_mm"].shift(2)

    # Target is the water area at the next available observation.
    df["Target_Area_t_plus_1"] = df["water_area_km2"].shift(-1)

    # The target date, not the feature date, determines the split.
    df["target_date"] = df["date"].shift(-1)

    features = [
        "Area_t",
        "Area_t_1",
        "Rain_Lag0",
        "Rain_Lag1",
        "Rain_Lag2"
    ]

    df_model = df.dropna(
        subset=features + ["Target_Area_t_plus_1", "target_date"]
    ).copy()

    if df_model.empty:
        raise ValueError("No usable rows remain after feature engineering.")

    # 3. Explicit chronological train/test split
    test_start = pd.Timestamp("2024-01-01")
    test_end = pd.Timestamp("2025-12-31")

    train_df = df_model[df_model["target_date"] < test_start].copy()

    test_df = df_model[
        (df_model["target_date"] >= test_start)
        & (df_model["target_date"] <= test_end)
    ].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            "Training or test data is empty. Check the dates in the CSV."
        )

    X_train = train_df[features]
    y_train = train_df["Target_Area_t_plus_1"]

    X_test = test_df[features]
    y_test = test_df["Target_Area_t_plus_1"]

    print(f"\nTraining observations: {len(train_df)}")
    print(
        f"Training target dates: {train_df['target_date'].min().date()} "
        f"to {train_df['target_date'].max().date()}"
    )
    print(f"Test observations: {len(test_df)}")
    print(
        f"Test target dates: {test_df['target_date'].min().date()} "
        f"to {test_df['target_date'].max().date()}"
    )

    # 4. Train the models
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=6,
        random_state=42
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    # 5. Evaluate models
    metrics = pd.DataFrame({
        "Model": [
            "Multiple Linear Regression (MLR)",
            "Random Forest (RF)"
        ],
        "R2": [
            r2_score(y_test, y_pred_lr),
            r2_score(y_test, y_pred_rf)
        ],
        "MAE (km2)": [
            mean_absolute_error(y_test, y_pred_lr),
            mean_absolute_error(y_test, y_pred_rf)
        ],
        "RMSE (km2)": [
            np.sqrt(mean_squared_error(y_test, y_pred_lr)),
            np.sqrt(mean_squared_error(y_test, y_pred_rf))
        ]
    })

    print("\n--- Model Evaluation ---")
    print(metrics.to_string(index=False))

    # 6. Random Forest feature importance
    importance = pd.DataFrame({
        "Feature": features,
        "Importance": rf.feature_importances_
    }).sort_values("Importance", ascending=False)

    print("\n--- Random Forest Feature Importances ---")
    print(importance.to_string(index=False))

    # 7. Export predictions and results
    os.makedirs("outputs", exist_ok=True)

    predictions = pd.DataFrame({
        "target_date": test_df["target_date"].values,
        "observed_area_km2": y_test.values,
        "random_forest_prediction_km2": y_pred_rf,
        "linear_regression_prediction_km2": y_pred_lr
    })

    predictions.to_csv(
        f"outputs/{reservoir_name}_test_predictions.csv",
        index=False
    )
    metrics.to_csv(
        f"outputs/{reservoir_name}_model_metrics.csv",
        index=False
    )
    importance.to_csv(
        f"outputs/{reservoir_name}_feature_importance.csv",
        index=False
    )

    # 8. Plot using the actual target dates
    plot_dates = pd.to_datetime(test_df["target_date"])

    plt.figure(figsize=(11, 5.5), dpi=300)

    plt.plot(
        plot_dates, y_test.values,
        label="Observed Area (km²)",
        color="black", linewidth=1.5, marker="o", markersize=4
    )
    plt.plot(
        plot_dates, y_pred_rf,
        label="Forecast (Random Forest)",
        color="crimson", linestyle="--", linewidth=1.5
    )
    plt.plot(
        plot_dates, y_pred_lr,
        label="Forecast (Linear Regression)",
        color="dodgerblue", linestyle=":", linewidth=1.5
    )

    plt.title(
        f"{reservoir_name} Reservoir Surface-Water Prediction "
        f"({plot_dates.min():%Y-%m} to {plot_dates.max():%Y-%m})"
    )
    plt.xlabel("Target Date")
    plt.ylabel("Surface-Water Area (km²)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    plt.savefig(
        f"outputs/{reservoir_name}_forecast_evaluation.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print("\nFiles saved in the outputs/ folder.")


if __name__ == "__main__":
    dataset_file = "Koyna_Decadal_TimeSeries_2016_2025.csv"

    if os.path.exists(dataset_file):
        run_reservoir_pipeline(dataset_file, reservoir_name="Koyna")
    else:
        print(f"Error: {dataset_file} not found in the current folder.")
