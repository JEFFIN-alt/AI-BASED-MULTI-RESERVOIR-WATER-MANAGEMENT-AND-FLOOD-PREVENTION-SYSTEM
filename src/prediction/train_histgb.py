import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATA_DIR = Path("data/processed/model_splits")
MODEL_DIR = Path("models")
RESULT_DIR = Path("data/processed/model_results")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def load_data(name):
    return pd.read_csv(DATA_DIR / f"{name}.csv")


def prepare_features(train, validation, test):

    lag_cols = [
        c for c in train.columns
        if c.startswith("lag_")
    ]

    reservoirs = sorted(
        train["reservoir"].dropna().unique()
    )

    reservoir_map = {
        name: i for i, name in enumerate(reservoirs)
    }

    for df in [train, validation, test]:
        df["reservoir_code"] = (
            df["reservoir"]
            .map(reservoir_map)
            .astype("float64")
        )

    feature_cols = lag_cols + ["reservoir_code"]

    X_train = train[feature_cols]
    X_val = validation[feature_cols]
    X_test = test[feature_cols]

    y_train = train["target_water_level"]
    y_val = validation["target_water_level"]
    y_test = test["target_water_level"]

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        feature_cols
    )


def evaluate(name, model, X, y):

    valid = y.notna()

    predictions = model.predict(X.loc[valid])
    actual = y.loc[valid]

    mae = mean_absolute_error(
        actual,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predictions
        )
    )

    r2 = r2_score(
        actual,
        predictions
    )

    print(f"\n========== {name} ==========")
    print("Evaluated rows:", len(actual))
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")

    return predictions, valid


def main():

    print("========== LOADING DATA ==========")

    train = load_data("train")
    validation = load_data("validation")
    test = load_data("test")

    print("Train:", train.shape)
    print("Validation:", validation.shape)
    print("Test:", test.shape)

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        feature_cols
    ) = prepare_features(
        train,
        validation,
        test
    )

    print("\n========== FEATURES ==========")
    print("Feature count:", len(feature_cols))
    print("Features:")

    for feature in feature_cols:
        print(" ", feature)

    print("\n========== TRAINING ==========")

    train_valid = y_train.notna()

    X_train_clean = X_train.loc[train_valid]
    y_train_clean = y_train.loc[train_valid]

    print("Training rows:", len(X_train))
    print(
        "Training rows with valid target:",
        len(X_train_clean)
    )
    print(
        "Excluded training rows with missing target:",
        len(X_train) - len(X_train_clean)
    )

    model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42
    )

    model.fit(
        X_train_clean,
        y_train_clean
    )

    print("Training complete.")

    val_predictions, val_valid = evaluate(
        "VALIDATION",
        model,
        X_val,
        y_val
    )

    test_predictions, test_valid = evaluate(
        "TEST",
        model,
        X_test,
        y_test
    )

    model_path = (
        MODEL_DIR /
        "hist_gradient_boosting_v1.joblib"
    )

    joblib.dump(
        model,
        model_path
    )

    test_results = test.loc[test_valid].copy()

    test_results["prediction"] = test_predictions

    test_results["error"] = (
        test_results["target_water_level"]
        - test_results["prediction"]
    )

    test_results["absolute_error"] = (
        test_results["error"].abs()
    )

    prediction_path = (
        RESULT_DIR /
        "histgb_test_predictions.csv"
    )

    test_results.to_csv(
        prediction_path,
        index=False
    )

    print("\n========== SAVED ==========")
    print("Model:", model_path)
    print("Predictions:", prediction_path)


if __name__ == "__main__":
    main()
