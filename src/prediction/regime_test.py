import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from train_histgb_v3 import prepare_data


DATA_DIR = Path("data/processed/model_splits")
MODEL_DIR = Path("models")

V3_MODEL = MODEL_DIR / "hist_gradient_boosting_v3_features.joblib"
REGIME_MODEL = MODEL_DIR / "movement_regime_classifier.joblib"


def build_features(train):
    reservoirs = sorted(train["reservoir"].dropna().unique())

    reservoir_map = {
        name: i for i, name in enumerate(reservoirs)
    }

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

    features = lag_features + engineered_features + ["reservoir_code"]

    return reservoir_map, features


def add_reservoir_code(df, reservoir_map):
    df = df.copy()

    df["reservoir_code"] = (
        df["reservoir"]
        .map(reservoir_map)
        .astype(float)
    )

    return df


def main():

    print("========== LOADING MODELS ==========")

    ml_model = joblib.load(V3_MODEL)
    regime_model = joblib.load(REGIME_MODEL)

    print("V3 model loaded")
    print("Regime classifier loaded")

    print("\n========== LOADING DATA ==========")

    train = prepare_data(DATA_DIR / "train.csv")
    validation = prepare_data(DATA_DIR / "validation.csv")
    test = prepare_data(DATA_DIR / "test.csv")

    reservoir_map, features = build_features(train)

    train = add_reservoir_code(train, reservoir_map)
    validation = add_reservoir_code(validation, reservoir_map)
    test = add_reservoir_code(test, reservoir_map)

    # Use the exact feature order stored in V3 model
    if hasattr(ml_model, "feature_names_in_"):
        features = list(ml_model.feature_names_in_)

    print("Feature count:", len(features))
    print("Feature order: OK")

    # =========================================================
    # VALIDATION
    # =========================================================

    print("\n========== VALIDATION ==========")

    valid = (
        validation["target_water_level"].notna()
        & validation["lag_1_water_level"].notna()
    )

    val = validation.loc[valid].copy()

    X_val = val[features]

    actual_val = val["target_water_level"].to_numpy()
    persistence_val = val["lag_1_water_level"].to_numpy()

    ml_delta_val = ml_model.predict(X_val)

    ml_val = (
        persistence_val + ml_delta_val
    )

    regime_probability_val = regime_model.predict_proba(X_val)[:, 1]

    print("Validation rows:", len(val))

    print(
        "Persistence MAE:",
        mean_absolute_error(actual_val, persistence_val)
    )

    print(
        "ML MAE:",
        mean_absolute_error(actual_val, ml_val)
    )

    # =========================================================
    # THRESHOLD SEARCH
    # =========================================================

    print("\n========== REGIME THRESHOLD SEARCH ==========")

    best_threshold = None
    best_mae = float("inf")

    for threshold in np.arange(0.10, 0.91, 0.05):

        use_ml = regime_probability_val >= threshold

        hybrid = np.where(
            use_ml,
            ml_val,
            persistence_val
        )

        score = mean_absolute_error(
            actual_val,
            hybrid
        )

        print(
            f"threshold={threshold:.2f} "
            f"MAE={score:.6f} "
            f"ML rows={use_ml.sum()} "
            f"Persistence rows={(~use_ml).sum()}"
        )

        if score < best_mae:
            best_mae = score
            best_threshold = float(threshold)

    print("\n========== BEST THRESHOLD ==========")
    print("Threshold:", best_threshold)
    print("Validation MAE:", best_mae)

    # =========================================================
    # TEST
    # =========================================================

    print("\n========== FINAL TEST ==========")

    valid = (
        test["target_water_level"].notna()
        & test["lag_1_water_level"].notna()
    )

    test_eval = test.loc[valid].copy()

    X_test = test_eval[features]

    actual = test_eval["target_water_level"].to_numpy()

    persistence = (
        test_eval["lag_1_water_level"].to_numpy()
    )

    ml_delta = ml_model.predict(X_test)

    ml_prediction = (
        persistence + ml_delta
    )

    probability = (
        regime_model.predict_proba(X_test)[:, 1]
    )

    use_ml = probability >= best_threshold

    hybrid = np.where(
        use_ml,
        ml_prediction,
        persistence
    )

    print("Test rows:", len(actual))
    print("Frozen threshold:", best_threshold)

    print("\nPERSISTENCE")
    print(
        f"MAE:  {mean_absolute_error(actual, persistence):.6f}"
    )
    print(
        f"RMSE: {np.sqrt(mean_squared_error(actual, persistence)):.6f}"
    )
    print(
        f"R²:   {r2_score(actual, persistence):.6f}"
    )

    print("\nML V3")
    print(
        f"MAE:  {mean_absolute_error(actual, ml_prediction):.6f}"
    )
    print(
        f"RMSE: {np.sqrt(mean_squared_error(actual, ml_prediction)):.6f}"
    )
    print(
        f"R²:   {r2_score(actual, ml_prediction):.6f}"
    )

    print("\nREGIME HYBRID")
    print(
        f"MAE:  {mean_absolute_error(actual, hybrid):.6f}"
    )
    print(
        f"RMSE: {np.sqrt(mean_squared_error(actual, hybrid)):.6f}"
    )
    print(
        f"R²:   {r2_score(actual, hybrid):.6f}"
    )

    print("\n========== USAGE ==========")
    print("ML predictions used:", int(use_ml.sum()))
    print(
        "Persistence used:",
        int((~use_ml).sum())
    )

    hybrid_error = np.abs(actual - hybrid)
    persistence_error = np.abs(actual - persistence)

    print("\nHybrid beats persistence:",
          int((hybrid_error < persistence_error).sum()))

    print("Hybrid beats ML:",
          int((hybrid_error < np.abs(actual - ml_prediction)).sum()))

    # =========================================================
    # SAVE
    # =========================================================

    result = test_eval[
        ["reservoir", "target_date", "target_water_level"]
    ].copy()

    result["regime_probability"] = probability
    result["ml_prediction"] = ml_prediction
    result["persistence_prediction"] = persistence
    result["hybrid_prediction"] = hybrid
    result["use_ml"] = use_ml

    output = Path(
        "data/processed/model_results/regime_hybrid_test.csv"
    )

    result.to_csv(output, index=False)

    print("\n========== SAVED ==========")
    print(output)


if __name__ == "__main__":
    main()
