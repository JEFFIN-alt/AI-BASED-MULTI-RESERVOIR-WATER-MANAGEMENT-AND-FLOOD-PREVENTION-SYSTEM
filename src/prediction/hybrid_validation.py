import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.metrics import mean_absolute_error


DATA_PATH = Path(
    "data/processed/model_splits/validation.csv"
)

MODEL_PATH = Path(
    "models/hist_gradient_boosting_v3_features.joblib"
)

RESULT_PATH = Path(
    "data/processed/model_results/hybrid_validation.csv"
)


def prepare_data(path):

    df = pd.read_csv(path)

    df["target_date"] = pd.to_datetime(
        df["target_date"]
    )

    # Target delta used during v3 training
    df["target_delta"] = (
        df["target_water_level"]
        - df["lag_1_water_level"]
    )

    # --------------------------------------------------
    # WATER LEVEL CHANGES
    # --------------------------------------------------

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

    # --------------------------------------------------
    # STORAGE CHANGES
    # --------------------------------------------------

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

    # --------------------------------------------------
    # INFLOW CHANGES
    # --------------------------------------------------

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

    # --------------------------------------------------
    # OUTFLOW CHANGES
    # --------------------------------------------------

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

    # --------------------------------------------------
    # RAINFALL
    # --------------------------------------------------

    rainfall_cols_3 = [
        "lag_1_rainfall",
        "lag_2_rainfall",
        "lag_3_rainfall",
    ]

    rainfall_cols_7 = [
        f"lag_{i}_rainfall"
        for i in range(1, 8)
    ]

    df["rainfall_3d_sum"] = df[
        rainfall_cols_3
    ].sum(axis=1, min_count=1)

    df["rainfall_7d_sum"] = df[
        rainfall_cols_7
    ].sum(axis=1, min_count=1)

    # --------------------------------------------------
    # INFLOW MEANS
    # --------------------------------------------------

    inflow_cols_3 = [
        f"lag_{i}_inflow"
        for i in range(1, 4)
    ]

    inflow_cols_7 = [
        f"lag_{i}_inflow"
        for i in range(1, 8)
    ]

    df["inflow_3d_mean"] = df[
        inflow_cols_3
    ].mean(axis=1)

    df["inflow_7d_mean"] = df[
        inflow_cols_7
    ].mean(axis=1)

    # --------------------------------------------------
    # OUTFLOW MEANS
    # --------------------------------------------------

    outflow_cols_3 = [
        f"lag_{i}_total_outflow"
        for i in range(1, 4)
    ]

    outflow_cols_7 = [
        f"lag_{i}_total_outflow"
        for i in range(1, 8)
    ]

    df["outflow_3d_mean"] = df[
        outflow_cols_3
    ].mean(axis=1)

    df["outflow_7d_mean"] = df[
        outflow_cols_7
    ].mean(axis=1)

    # --------------------------------------------------
    # WATER BALANCE
    # --------------------------------------------------

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

    # --------------------------------------------------
    # TRENDS
    # --------------------------------------------------

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

    print("========== LOADING VALIDATION ==========")

    df = prepare_data(DATA_PATH)

    model = joblib.load(MODEL_PATH)

    print("Validation rows:", len(df))

    # --------------------------------------------------
    # RESERVOIR ENCODING
    # --------------------------------------------------

    reservoirs = sorted(
        df["reservoir"].dropna().unique()
    )

    reservoir_map = {
        name: i
        for i, name in enumerate(reservoirs)
    }

    df["reservoir_code"] = (
        df["reservoir"]
        .map(reservoir_map)
        .astype(float)
    )

    # --------------------------------------------------
    # EXACT V3 FEATURE LIST
    # --------------------------------------------------

    lag_features = [
        c for c in df.columns
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

    print("Features:", len(features))

    # --------------------------------------------------
    # VERIFY MODEL FEATURES
    # --------------------------------------------------

    expected = list(
        model.feature_names_in_
    )

    if features != expected:

        print("\nERROR: Feature mismatch!")

        print("\nExpected by model:")
        for f in expected:
            print(" ", f)

        print("\nProvided:")
        for f in features:
            print(" ", f)

        raise ValueError(
            "Feature order does not match trained model."
        )

    print("Feature order: OK")

    X = df[features]

    # --------------------------------------------------
    # VALID ROWS
    # --------------------------------------------------

    valid = (
        df["target_water_level"].notna()
        & df["lag_1_water_level"].notna()
    )

    print("Evaluated rows:", valid.sum())

    # --------------------------------------------------
    # ML PREDICTION
    # --------------------------------------------------

    pred_delta = model.predict(
        X.loc[valid]
    )

    today_level = df.loc[
        valid,
        "lag_1_water_level"
    ].to_numpy()

    actual = df.loc[
        valid,
        "target_water_level"
    ].to_numpy()

    ml_prediction = (
        today_level + pred_delta
    )

    persistence_prediction = today_level

    ml_error = np.abs(
        actual - ml_prediction
    )

    persistence_error = np.abs(
        actual - persistence_prediction
    )

    # --------------------------------------------------
    # BASIC COMPARISON
    # --------------------------------------------------

    print("\n========== VALIDATION PERFORMANCE ==========")

    print(
        f"ML MAE:          "
        f"{ml_error.mean():.6f}"
    )

    print(
        f"Persistence MAE: "
        f"{persistence_error.mean():.6f}"
    )

    print(
        f"ML wins:         "
        f"{(ml_error < persistence_error).sum()}"
    )

    print(
        f"Persistence wins:"
        f" {(persistence_error < ml_error).sum()}"
    )

    # --------------------------------------------------
    # TEST HYBRID THRESHOLDS
    # --------------------------------------------------

    print("\n========== HYBRID THRESHOLD SEARCH ==========")

    movement = np.abs(
        df.loc[
            valid,
            "water_level_change_1d"
        ].to_numpy()
    )

    thresholds = [
        0.0,
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.40,
        0.50,
        0.75,
        1.00,
        1.50,
        2.00,
    ]

    results = []

    for threshold in thresholds:

        # Stable -> persistence
        # Moving -> ML

        use_ml = movement > threshold

        hybrid_prediction = np.where(
            use_ml,
            ml_prediction,
            persistence_prediction
        )

        hybrid_mae = mean_absolute_error(
            actual,
            hybrid_prediction
        )

        ml_count = int(
            use_ml.sum()
        )

        results.append({
            "threshold": threshold,
            "mae": hybrid_mae,
            "ml_rows": ml_count,
            "persistence_rows": len(actual) - ml_count,
        })

        print(
            f"threshold={threshold:>4.2f}  "
            f"MAE={hybrid_mae:.6f}  "
            f"ML rows={ml_count}"
        )

    results_df = pd.DataFrame(results)

    best = results_df.loc[
        results_df["mae"].idxmin()
    ]

    print("\n========== BEST HYBRID ==========")

    print(
        f"Threshold: "
        f"{best['threshold']}"
    )

    print(
        f"Validation MAE: "
        f"{best['mae']:.6f}"
    )

    print(
        f"ML rows: "
        f"{int(best['ml_rows'])}"
    )

    print(
        f"Persistence rows: "
        f"{int(best['persistence_rows'])}"
    )

    # --------------------------------------------------
    # SAVE
    # --------------------------------------------------

    RESULT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        RESULT_PATH,
        index=False
    )

    print("\n========== SAVED ==========")

    print(RESULT_PATH)


if __name__ == "__main__":
    main()
