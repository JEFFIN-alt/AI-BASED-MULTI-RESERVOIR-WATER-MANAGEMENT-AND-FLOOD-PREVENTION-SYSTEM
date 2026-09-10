import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


TRAIN_PATH = "data/processed/model_splits/train.csv"
VAL_PATH = "data/processed/model_splits/validation.csv"
TEST_PATH = "data/processed/model_splits/test.csv"

MODEL_PATH = "models/hist_gradient_boosting_v3_features.joblib"
PRED_PATH = "data/processed/model_results/histgb_v3_test_predictions.csv"


BASE_LAGS = [
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_4",
    "lag_5",
    "lag_6",
    "lag_7",
]


def prepare_data(path):
    df = pd.read_csv(path)
    df["target_date"] = pd.to_datetime(df["target_date"])

    # ---------------------------------------------------------
    # TARGET = TOMORROW'S CHANGE IN WATER LEVEL
    # ---------------------------------------------------------
    df["target_delta"] = (
        df["target_water_level"]
        - df["lag_1_water_level"]
    )

    # ---------------------------------------------------------
    # TEMPORAL / HYDROLOGICAL FEATURES
    # ---------------------------------------------------------

    # Water-level changes
    df["water_level_change_1d"] = (
        df["lag_1_water_level"]
        - df["lag_2_water_level"]
    )

    df["water_level_change_3d"] = (
        df["lag_1_water_level"]
        - df["lag_4_water_level"]
    )

    df["water_level_change_7d"] = (
        df["lag_1_water_level"]
        - df["lag_7_water_level"]
    )

    # Storage changes
    df["storage_change_1d"] = (
        df["lag_1_live_storage"]
        - df["lag_2_live_storage"]
    )

    df["storage_change_3d"] = (
        df["lag_1_live_storage"]
        - df["lag_4_live_storage"]
    )

    df["storage_change_7d"] = (
        df["lag_1_live_storage"]
        - df["lag_7_live_storage"]
    )

    # Inflow changes
    df["inflow_change_1d"] = (
        df["lag_1_inflow"]
        - df["lag_2_inflow"]
    )

    df["inflow_change_3d"] = (
        df["lag_1_inflow"]
        - df["lag_4_inflow"]
    )

    df["inflow_change_7d"] = (
        df["lag_1_inflow"]
        - df["lag_7_inflow"]
    )

    # Outflow changes
    df["outflow_change_1d"] = (
        df["lag_1_total_outflow"]
        - df["lag_2_total_outflow"]
    )

    df["outflow_change_3d"] = (
        df["lag_1_total_outflow"]
        - df["lag_4_total_outflow"]
    )

    df["outflow_change_7d"] = (
        df["lag_1_total_outflow"]
        - df["lag_7_total_outflow"]
    )

    # ---------------------------------------------------------
    # ROLLING / AGGREGATED HYDROLOGICAL SIGNALS
    # ---------------------------------------------------------

    # Rainfall accumulation
    rainfall_cols_3 = [
        "lag_1_rainfall",
        "lag_2_rainfall",
        "lag_3_rainfall",
    ]

    rainfall_cols_7 = [
        f"lag_{i}_rainfall"
        for i in range(1, 8)
    ]

    df["rainfall_3d_sum"] = df[rainfall_cols_3].sum(
        axis=1, min_count=1
    )

    df["rainfall_7d_sum"] = df[rainfall_cols_7].sum(
        axis=1, min_count=1
    )

    # Mean inflow
    inflow_cols_3 = [
        f"lag_{i}_inflow"
        for i in range(1, 4)
    ]

    inflow_cols_7 = [
        f"lag_{i}_inflow"
        for i in range(1, 8)
    ]

    df["inflow_3d_mean"] = df[inflow_cols_3].mean(axis=1)
    df["inflow_7d_mean"] = df[inflow_cols_7].mean(axis=1)

    # Mean outflow
    outflow_cols_3 = [
        f"lag_{i}_total_outflow"
        for i in range(1, 4)
    ]

    outflow_cols_7 = [
        f"lag_{i}_total_outflow"
        for i in range(1, 8)
    ]

    df["outflow_3d_mean"] = df[outflow_cols_3].mean(axis=1)
    df["outflow_7d_mean"] = df[outflow_cols_7].mean(axis=1)

    # ---------------------------------------------------------
    # INFLOW / OUTFLOW BALANCE
    # ---------------------------------------------------------

    df["water_balance_1d"] = (
        df["lag_1_inflow"]
        - df["lag_1_total_outflow"]
    )

    df["water_balance_3d"] = (
        df["inflow_3d_mean"]
        - df["outflow_3d_mean"]
    )

    df["water_balance_7d"] = (
        df["inflow_7d_mean"]
        - df["outflow_7d_mean"]
    )

    # ---------------------------------------------------------
    # TREND FEATURES
    # ---------------------------------------------------------

    df["water_level_trend"] = (
        df["water_level_change_7d"] / 7.0
    )

    df["storage_trend"] = (
        df["storage_change_7d"] / 7.0
    )

    df["inflow_trend"] = (
        df["inflow_change_7d"] / 7.0
    )

    df["outflow_trend"] = (
        df["outflow_change_7d"] / 7.0
    )

    return df


def main():

    print("========== LOADING DATA ==========")

    train = prepare_data(TRAIN_PATH)
    val = prepare_data(VAL_PATH)
    test = prepare_data(TEST_PATH)

    print("Train:", train.shape)
    print("Validation:", val.shape)
    print("Test:", test.shape)

    # ---------------------------------------------------------
    # RESERVOIR ENCODING
    # ---------------------------------------------------------

    reservoirs = sorted(
        train["reservoir"].dropna().unique()
    )

    reservoir_map = {
        name: i for i, name in enumerate(reservoirs)
    }

    for df in [train, val, test]:
        df["reservoir_code"] = (
            df["reservoir"]
            .map(reservoir_map)
            .astype(float)
        )

    # ---------------------------------------------------------
    # FEATURE SELECTION
    # ---------------------------------------------------------

    lag_features = [
        c for c in train.columns
        if c.startswith("lag_")
    ]

    engineered_features = [
        "water_level_change_1d",
        "water_level_change_3d",
        "water_level_change_7d",

        "storage_change_1d",
        "storage_change_3d",
        "storage_change_7d",

        "inflow_change_1d",
        "inflow_change_3d",
        "inflow_change_7d",

        "outflow_change_1d",
        "outflow_change_3d",
        "outflow_change_7d",

        "rainfall_3d_sum",
        "rainfall_7d_sum",

        "inflow_3d_mean",
        "inflow_7d_mean",

        "outflow_3d_mean",
        "outflow_7d_mean",

        "water_balance_1d",
        "water_balance_3d",
        "water_balance_7d",

        "water_level_trend",
        "storage_trend",
        "inflow_trend",
        "outflow_trend",
    ]

    features = (
        lag_features
        + engineered_features
        + ["reservoir_code"]
    )

    print("\n========== FEATURES ==========")
    print("Base lag features:", len(lag_features))
    print("Engineered features:", len(engineered_features))
    print("Total features:", len(features))

    for feature in features:
        print(" ", feature)

    X_train = train[features]
    X_val = val[features]
    X_test = test[features]

    y_train = train["target_delta"]
    y_val = val["target_delta"]
    y_test = test["target_delta"]

    # ---------------------------------------------------------
    # TRAIN
    # ---------------------------------------------------------

    print("\n========== TRAINING ==========")

    train_mask = y_train.notna()

    X_train_fit = X_train.loc[train_mask]
    y_train_fit = y_train.loc[train_mask]

    print("Training rows:", len(X_train_fit))
    print("Excluded missing targets:",
          len(X_train) - len(X_train_fit))

    model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42
    )

    model.fit(
        X_train_fit,
        y_train_fit
    )

    print("Training complete.")

    # ---------------------------------------------------------
    # EVALUATION FUNCTION
    # ---------------------------------------------------------

    def evaluate(name, df, X, y):

        mask = (
            y.notna()
            & df["lag_1_water_level"].notna()
        )

        pred_delta = model.predict(
            X.loc[mask]
        )

        today_level = df.loc[
            mask, "lag_1_water_level"
        ].to_numpy()

        actual_level = df.loc[
            mask, "target_water_level"
        ].to_numpy()

        pred_level = (
            today_level + pred_delta
        )

        mae = mean_absolute_error(
            actual_level,
            pred_level
        )

        rmse = np.sqrt(
            mean_squared_error(
                actual_level,
                pred_level
            )
        )

        r2 = r2_score(
            actual_level,
            pred_level
        )

        print(f"\n========== {name} ==========")
        print("Evaluated rows:", len(actual_level))
        print("MAE: ", round(mae, 6))
        print("RMSE:", round(rmse, 6))
        print("R²:  ", round(r2, 6))

        return mask, pred_level, actual_level

    val_mask, _, _ = evaluate(
        "VALIDATION",
        val,
        X_val,
        y_val
    )

    test_mask, test_pred_level, test_actual_level = evaluate(
        "TEST",
        test,
        X_test,
        y_test
    )

    # ---------------------------------------------------------
    # SAVE TEST PREDICTIONS
    # ---------------------------------------------------------

    predictions = test.loc[
        test_mask,
        [
            "reservoir",
            "target_date",
            "target_water_level",
            "lag_1_water_level"
        ]
    ].copy()

    predictions["prediction"] = test_pred_level

    predictions["actual_delta"] = (
        predictions["target_water_level"]
        - predictions["lag_1_water_level"]
    )

    predictions["predicted_delta"] = (
        predictions["prediction"]
        - predictions["lag_1_water_level"]
    )

    predictions["abs_error"] = (
        predictions["target_water_level"]
        - predictions["prediction"]
    ).abs()

    os.makedirs(
        os.path.dirname(MODEL_PATH),
        exist_ok=True
    )

    os.makedirs(
        os.path.dirname(PRED_PATH),
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    predictions.to_csv(
        PRED_PATH,
        index=False
    )

    print("\n========== SAVED ==========")
    print("Model:", MODEL_PATH)
    print("Predictions:", PRED_PATH)

    # ---------------------------------------------------------
    # RESERVOIR PERFORMANCE
    # ---------------------------------------------------------

    reservoir_results = (
        predictions
        .groupby("reservoir")
        .agg(
            samples=("prediction", "size"),
            MAE=("abs_error", "mean")
        )
        .sort_values("MAE", ascending=False)
    )

    print("\n========== MAE BY RESERVOIR ==========")
    print(
        reservoir_results.to_string(
            float_format=lambda x: f"{x:.6f}"
        )
    )


if __name__ == "__main__":
    main()
