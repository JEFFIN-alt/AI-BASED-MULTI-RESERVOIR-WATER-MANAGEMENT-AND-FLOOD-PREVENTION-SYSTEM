import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


TEST_PATH = Path(
    "data/processed/model_splits/test.csv"
)

MODEL_PATH = Path(
    "models/hist_gradient_boosting_v3_features.joblib"
)

RESULT_PATH = Path(
    "data/processed/model_results/hybrid_test.csv"
)

THRESHOLD = 0.25


def prepare_data(path):

    df = pd.read_csv(path)

    df["target_date"] = pd.to_datetime(
        df["target_date"]
    )

    # Target delta
    df["target_delta"] = (
        df["target_water_level"]
        - df["lag_1_water_level"]
    )

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

    # Rainfall
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

    # Inflow means
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

    # Outflow means
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

    # Water balance
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

    # Trends
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


def metrics(actual, prediction):

    mae = mean_absolute_error(
        actual,
        prediction
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            prediction
        )
    )

    r2 = r2_score(
        actual,
        prediction
    )

    return mae, rmse, r2


def main():

    print("========== LOADING TEST ==========")

    df = prepare_data(TEST_PATH)

    model = joblib.load(MODEL_PATH)

    print("Test rows:", len(df))

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
    # EXACT V3 FEATURES
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

    # Verify exact feature order
    expected = list(
        model.feature_names_in_
    )

    if features != expected:

        print("\nERROR: Feature mismatch!")

        print("Expected:")
        print(expected)

        print("\nProvided:")
        print(features)

        raise ValueError(
            "Feature order does not match trained model."
        )

    print("Feature order: OK")

    X = df[features]

    # --------------------------------------------------
    # VALID TEST ROWS
    # --------------------------------------------------

    valid = (
        df["target_water_level"].notna()
        & df["lag_1_water_level"].notna()
    )

    print("Evaluated rows:", valid.sum())

    # --------------------------------------------------
    # PREDICTIONS
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

    reservoirs_valid = df.loc[
        valid,
        "reservoir"
    ].to_numpy()

    dates_valid = df.loc[
        valid,
        "target_date"
    ].to_numpy()

    # ML predicted level
    ml_prediction = (
        today_level + pred_delta
    )

    # Persistence prediction
    persistence_prediction = today_level

    # Movement used by hybrid rule
    movement = np.abs(
        df.loc[
            valid,
            "water_level_change_1d"
        ].to_numpy()
    )

    # Frozen threshold selected from validation
    use_ml = movement > THRESHOLD

    hybrid_prediction = np.where(
        use_ml,
        ml_prediction,
        persistence_prediction
    )

    # --------------------------------------------------
    # OVERALL RESULTS
    # --------------------------------------------------

    ml_mae, ml_rmse, ml_r2 = metrics(
        actual,
        ml_prediction
    )

    p_mae, p_rmse, p_r2 = metrics(
        actual,
        persistence_prediction
    )

    h_mae, h_rmse, h_r2 = metrics(
        actual,
        hybrid_prediction
    )

    print("\n========== FINAL TEST ==========")

    print(f"Rows: {len(actual)}")
    print(f"Frozen threshold: {THRESHOLD}")

    print("\nPERSISTENCE")
    print(f"MAE:  {p_mae:.6f}")
    print(f"RMSE: {p_rmse:.6f}")
    print(f"R²:   {p_r2:.6f}")

    print("\nML V3")
    print(f"MAE:  {ml_mae:.6f}")
    print(f"RMSE: {ml_rmse:.6f}")
    print(f"R²:   {ml_r2:.6f}")

    print("\nHYBRID")
    print(f"MAE:  {h_mae:.6f}")
    print(f"RMSE: {h_rmse:.6f}")
    print(f"R²:   {h_r2:.6f}")

    # --------------------------------------------------
    # WIN COUNTS
    # --------------------------------------------------

    ml_error = np.abs(
        actual - ml_prediction
    )

    persistence_error = np.abs(
        actual - persistence_prediction
    )

    hybrid_error = np.abs(
        actual - hybrid_prediction
    )

    print("\n========== WIN COUNTS ==========")

    print(
        "ML wins:",
        int(
            (ml_error < persistence_error).sum()
        )
    )

    print(
        "Persistence wins:",
        int(
            (persistence_error < ml_error).sum()
        )
    )

    print(
        "Hybrid beats persistence:",
        int(
            (hybrid_error < persistence_error).sum()
        )
    )

    print(
        "Hybrid beats ML:",
        int(
            (hybrid_error < ml_error).sum()
        )
    )

    print(
        "ML predictions used:",
        int(use_ml.sum())
    )

    print(
        "Persistence predictions used:",
        int((~use_ml).sum())
    )

    # --------------------------------------------------
    # BY RESERVOIR
    # --------------------------------------------------

    result_df = pd.DataFrame({
        "reservoir": reservoirs_valid,
        "target_date": dates_valid,
        "actual": actual,
        "ml_prediction": ml_prediction,
        "persistence_prediction":
            persistence_prediction,
        "hybrid_prediction":
            hybrid_prediction,
        "movement": movement,
        "ml_error": ml_error,
        "persistence_error":
            persistence_error,
        "hybrid_error": hybrid_error,
        "use_ml": use_ml,
    })

    print("\n========== MAE BY RESERVOIR ==========")

    reservoir_results = []

    for reservoir, g in result_df.groupby(
        "reservoir"
    ):

        reservoir_results.append({
            "reservoir": reservoir,
            "samples": len(g),
            "ml_mae": g["ml_error"].mean(),
            "persistence_mae":
                g["persistence_error"].mean(),
            "hybrid_mae":
                g["hybrid_error"].mean(),
            "ml_used":
                int(g["use_ml"].sum()),
        })

    reservoir_results = pd.DataFrame(
        reservoir_results
    ).sort_values(
        "hybrid_mae",
        ascending=False
    )

    print(
        reservoir_results.to_string(
            index=False
        )
    )

    # --------------------------------------------------
    # MOVEMENT BUCKETS
    # --------------------------------------------------

    print(
        "\n========== PERFORMANCE BY MOVEMENT =========="
    )

    bins = [
        -np.inf,
        0.1,
        0.5,
        1.0,
        2.0,
        5.0,
        np.inf,
    ]

    labels = [
        "<=0.1",
        "0.1-0.5",
        "0.5-1",
        "1-2",
        "2-5",
        ">5",
    ]

    result_df["movement_bucket"] = pd.cut(
        result_df["movement"],
        bins=bins,
        labels=labels,
    )

    movement_results = []

    for bucket, g in result_df.groupby(
        "movement_bucket",
        observed=True
    ):

        movement_results.append({
            "movement_bucket": str(bucket),
            "samples": len(g),
            "ml_mae":
                g["ml_error"].mean(),
            "persistence_mae":
                g["persistence_error"].mean(),
            "hybrid_mae":
                g["hybrid_error"].mean(),
            "ml_used":
                int(g["use_ml"].sum()),
        })

    movement_results = pd.DataFrame(
        movement_results
    )

    print(
        movement_results.to_string(
            index=False
        )
    )

    # --------------------------------------------------
    # LARGEST HYBRID ERRORS
    # --------------------------------------------------

    print(
        "\n========== LARGEST HYBRID ERRORS =========="
    )

    worst = result_df.sort_values(
        "hybrid_error",
        ascending=False
    ).head(20)

    print(
        worst[
            [
                "reservoir",
                "target_date",
                "actual",
                "ml_prediction",
                "persistence_prediction",
                "hybrid_prediction",
                "movement",
                "hybrid_error",
                "use_ml",
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------
    # SAVE
    # --------------------------------------------------

    RESULT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result_df.to_csv(
        RESULT_PATH,
        index=False
    )

    print("\n========== SAVED ==========")
    print(RESULT_PATH)


if __name__ == "__main__":
    main()
