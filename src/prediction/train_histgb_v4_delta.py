import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATA_DIR = Path("data/processed/model_splits_v4")
MODEL_DIR = Path("models")
RESULT_DIR = Path("data/processed/model_results")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def load_data(name):
    return pd.read_csv(DATA_DIR / f"{name}.csv")


def prepare(df):
    df = df.copy()

    # Tomorrow's CHANGE in water level
    df["target_delta"] = (
        df["target_water_level"]
        - df["lag_1_water_level"]
    )

    return df


def main():

    print("========== LOADING V4 DELTA DATA ==========")

    train = prepare(load_data("train"))
    val = prepare(load_data("validation"))

    print("Train:", train.shape)
    print("Validation:", val.shape)

    # ---------------------------------------------------------
    # FEATURES
    # ---------------------------------------------------------

    excluded = {
        "reservoir",
        "target_date",
        "target_water_level",
        "target_delta",
    }

    features = [
        c for c in train.columns
        if c not in excluded
    ]

    print("\n========== FEATURES ==========")
    print("Feature count:", len(features))

    X_train = train[features]
    X_val = val[features]

    y_train = train["target_delta"]
    y_val = val["target_delta"]

    # ---------------------------------------------------------
    # TRAIN
    # ---------------------------------------------------------

    print("\n========== TRAINING ==========")

    train_mask = (
        y_train.notna()
        & train["lag_1_water_level"].notna()
    )

    X_train_fit = X_train.loc[train_mask]
    y_train_fit = y_train.loc[train_mask]

    print("Training rows:", len(X_train_fit))
    print(
        "Excluded:",
        len(train) - len(X_train_fit)
    )

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
    # VALIDATION
    # ---------------------------------------------------------

    print("\n========== VALIDATION ==========")

    val_mask = (
        y_val.notna()
        & val["lag_1_water_level"].notna()
    )

    pred_delta = model.predict(
        X_val.loc[val_mask]
    )

    today_level = val.loc[
        val_mask,
        "lag_1_water_level"
    ].to_numpy()

    actual_level = val.loc[
        val_mask,
        "target_water_level"
    ].to_numpy()

    predicted_level = (
        today_level + pred_delta
    )

    mae = mean_absolute_error(
        actual_level,
        predicted_level
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual_level,
            predicted_level
        )
    )

    r2 = r2_score(
        actual_level,
        predicted_level
    )

    persistence = today_level

    persistence_mae = mean_absolute_error(
        actual_level,
        persistence
    )

    print("Evaluated rows:", len(actual_level))
    print(f"V4 Delta MAE:       {mae:.6f}")
    print(f"V4 Delta RMSE:      {rmse:.6f}")
    print(f"V4 Delta R²:        {r2:.6f}")
    print(f"Persistence MAE:    {persistence_mae:.6f}")
    print(
        f"Improvement:        "
        f"{persistence_mae - mae:.6f}"
    )

    # ---------------------------------------------------------
    # MOVEMENT ANALYSIS
    # ---------------------------------------------------------

    result = val.loc[
        val_mask,
        [
            "reservoir",
            "target_date",
            "target_water_level",
            "lag_1_water_level"
        ]
    ].copy()

    result["prediction"] = predicted_level

    result["actual_delta"] = (
        result["target_water_level"]
        - result["lag_1_water_level"]
    )

    result["predicted_delta"] = (
        result["prediction"]
        - result["lag_1_water_level"]
    )

    result["ml_error"] = (
        result["prediction"]
        - result["target_water_level"]
    ).abs()

    result["persistence_error"] = (
        result["lag_1_water_level"]
        - result["target_water_level"]
    ).abs()

    result["movement"] = (
        result["actual_delta"].abs()
    )

    bins = [
        -np.inf,
        0.1,
        0.5,
        1,
        2,
        5,
        np.inf
    ]

    labels = [
        "<=0.1",
        "0.1-0.5",
        "0.5-1",
        "1-2",
        "2-5",
        ">5"
    ]

    result["movement_bucket"] = pd.cut(
        result["movement"],
        bins=bins,
        labels=labels
    )

    print("\n========== MOVEMENT PERFORMANCE ==========")

    movement = result.groupby(
        "movement_bucket",
        observed=False
    ).agg(
        samples=("actual_delta", "size"),
        ml_mae=("ml_error", "mean"),
        persistence_mae=("persistence_error", "mean")
    )

    movement["ML_advantage"] = (
        movement["persistence_mae"]
        - movement["ml_mae"]
    )

    print(movement.to_string())

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------

    model_path = (
        MODEL_DIR /
        "hist_gradient_boosting_v4_delta.joblib"
    )

    result_path = (
        RESULT_DIR /
        "histgb_v4_delta_validation.csv"
    )

    joblib.dump(model, model_path)
    result.to_csv(result_path, index=False)

    print("\n========== SAVED ==========")
    print("Model:", model_path)
    print("Validation:", result_path)


if __name__ == "__main__":
    main()
