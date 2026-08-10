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


def main():

    print("========== LOADING V4 DATA ==========")

    train = load_data("train")
    val = load_data("validation")
    test = load_data("test")

    print("Train:", train.shape)
    print("Validation:", val.shape)
    print("Test:", test.shape)

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

    # Make sure validation/test have exactly the same features
    X_train = train[features]
    X_val = val[features]
    X_test = test[features]

    y_train = train["target_water_level"]
    y_val = val["target_water_level"]
    y_test = test["target_water_level"]

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
        "Excluded rows:",
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
    # EVALUATION
    # ---------------------------------------------------------

    def evaluate(name, df, X, y):

        mask = (
            y.notna()
            & df["lag_1_water_level"].notna()
        )

        predictions = model.predict(
            X.loc[mask]
        )

        actual = y.loc[mask]

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
        print(f"MAE:  {mae:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"R²:   {r2:.6f}")

        return predictions, mask

    val_pred, val_mask = evaluate(
        "VALIDATION",
        val,
        X_val,
        y_val
    )

    test_pred, test_mask = evaluate(
        "TEST",
        test,
        X_test,
        y_test
    )

    # ---------------------------------------------------------
    # SAVE TEST PREDICTIONS
    # ---------------------------------------------------------

    result = test.loc[test_mask, [
        "reservoir",
        "target_date",
        "target_water_level",
        "lag_1_water_level"
    ]].copy()

    result["prediction"] = test_pred

    result["persistence"] = (
        result["lag_1_water_level"]
    )

    result["ml_error"] = (
        result["prediction"] -
        result["target_water_level"]
    ).abs()

    result["persistence_error"] = (
        result["persistence"] -
        result["target_water_level"]
    ).abs()

    result.to_csv(
        RESULT_DIR / "histgb_v4_test_predictions.csv",
        index=False
    )

    # ---------------------------------------------------------
    # SAVE MODEL
    # ---------------------------------------------------------

    model_path = (
        MODEL_DIR /
        "hist_gradient_boosting_v4.joblib"
    )

    joblib.dump(
        model,
        model_path
    )

    print("\n========== SAVED ==========")
    print("Model:", model_path)
    print(
        "Predictions:",
        RESULT_DIR / "histgb_v4_test_predictions.csv"
    )


if __name__ == "__main__":
    main()
