import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


TRAIN_PATH = "data/processed/model_splits/train.csv"
VAL_PATH = "data/processed/model_splits/validation.csv"
TEST_PATH = "data/processed/model_splits/test.csv"

MODEL_PATH = "models/hist_gradient_boosting_v2_delta.joblib"
PRED_PATH = "data/processed/model_results/histgb_v2_test_predictions.csv"


def prepare_data(path):
    df = pd.read_csv(path)

    df["target_date"] = pd.to_datetime(df["target_date"])

    # Delta target:
    # tomorrow's water level - today's water level
    df["target_delta"] = (
        df["target_water_level"] - df["lag_1_water_level"]
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
    # FEATURES
    # ---------------------------------------------------------

    lag_features = [
        c for c in train.columns
        if c.startswith("lag_")
    ]

    # Encode reservoir consistently across all datasets
    reservoirs = sorted(train["reservoir"].dropna().unique())
    reservoir_map = {
        name: i for i, name in enumerate(reservoirs)
    }

    for df in [train, val, test]:
        df["reservoir_code"] = (
            df["reservoir"]
            .map(reservoir_map)
            .astype(float)
        )

    features = lag_features + ["reservoir_code"]

    print("\n========== FEATURES ==========")
    print("Feature count:", len(features))

    for f in features:
        print(" ", f)

    X_train = train[features]
    X_val = val[features]
    X_test = test[features]

    y_train = train["target_delta"]
    y_val = val["target_delta"]
    y_test = test["target_delta"]

    # ---------------------------------------------------------
    # TRAINING
    # ---------------------------------------------------------

    print("\n========== TRAINING ==========")

    train_mask = y_train.notna()

    X_train_fit = X_train.loc[train_mask]
    y_train_fit = y_train.loc[train_mask]

    print("Training rows:", len(X_train))
    print("Valid delta targets:", len(y_train_fit))
    print("Excluded missing targets:",
          len(X_train) - len(y_train_fit))

    model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42
    )

    model.fit(X_train_fit, y_train_fit)

    print("Training complete.")

    # ---------------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------------

    print("\n========== VALIDATION ==========")

    val_mask = y_val.notna()

    val_pred_delta = model.predict(X_val.loc[val_mask])

    val_actual_level = val.loc[
        val_mask, "target_water_level"
    ].to_numpy()

    val_today_level = val.loc[
        val_mask, "lag_1_water_level"
    ].to_numpy()

    val_pred_level = val_today_level + val_pred_delta

    val_mae = mean_absolute_error(
        val_actual_level,
        val_pred_level
    )

    val_rmse = np.sqrt(
        mean_squared_error(
            val_actual_level,
            val_pred_level
        )
    )

    val_r2 = r2_score(
        val_actual_level,
        val_pred_level
    )

    print("Evaluated rows:", len(val_actual_level))
    print("MAE: ", round(val_mae, 4))
    print("RMSE:", round(val_rmse, 4))
    print("R²:  ", round(val_r2, 4))

    # ---------------------------------------------------------
    # TEST
    # ---------------------------------------------------------

    print("\n========== TEST ==========")

    test_mask = (
        y_test.notna()
        & test["lag_1_water_level"].notna()
    )

    test_pred_delta = model.predict(
        X_test.loc[test_mask]
    )

    test_today_level = test.loc[
        test_mask, "lag_1_water_level"
    ].to_numpy()

    test_actual_level = test.loc[
        test_mask, "target_water_level"
    ].to_numpy()

    test_pred_level = (
        test_today_level + test_pred_delta
    )

    test_mae = mean_absolute_error(
        test_actual_level,
        test_pred_level
    )

    test_rmse = np.sqrt(
        mean_squared_error(
            test_actual_level,
            test_pred_level
        )
    )

    test_r2 = r2_score(
        test_actual_level,
        test_pred_level
    )

    print("Evaluated rows:", len(test_actual_level))
    print("MAE: ", round(test_mae, 4))
    print("RMSE:", round(test_rmse, 4))
    print("R²:  ", round(test_r2, 4))

    # ---------------------------------------------------------
    # SAVE PREDICTIONS
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

    predictions["actual_delta"] = (
        predictions["target_water_level"]
        - predictions["lag_1_water_level"]
    )

    predictions["predicted_delta"] = test_pred_delta

    predictions["prediction"] = test_pred_level

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

    joblib.dump(model, MODEL_PATH)

    predictions.to_csv(
        PRED_PATH,
        index=False
    )

    print("\n========== SAVED ==========")
    print("Model:", MODEL_PATH)
    print("Predictions:", PRED_PATH)

    # ---------------------------------------------------------
    # BY RESERVOIR
    # ---------------------------------------------------------

    predictions["error"] = (
        predictions["target_water_level"]
        - predictions["prediction"]
    )

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
    print(reservoir_results.to_string())


if __name__ == "__main__":
    main()
