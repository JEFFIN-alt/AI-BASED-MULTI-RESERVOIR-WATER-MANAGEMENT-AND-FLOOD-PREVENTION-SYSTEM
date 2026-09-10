import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix

DATA_DIR = Path("data/processed/model_splits")
MODEL_DIR = Path("models")

THRESHOLD = 0.5


def prepare_features(path):

    df = pd.read_csv(path)
    df["target_date"] = pd.to_datetime(df["target_date"])

    # Water-level changes
    df["water_level_change_1d"] = (
        df["lag_1_water_level"] - df["lag_2_water_level"]
    )

    df["water_level_change_3d"] = (
        df["lag_1_water_level"] - df["lag_4_water_level"]
    )

    df["water_level_change_7d"] = (
        df["lag_1_water_level"] - df["lag_7_water_level"]
    )

    # Storage changes
    df["storage_change_1d"] = (
        df["lag_1_live_storage"] - df["lag_2_live_storage"]
    )

    df["storage_change_3d"] = (
        df["lag_1_live_storage"] - df["lag_4_live_storage"]
    )

    df["storage_change_7d"] = (
        df["lag_1_live_storage"] - df["lag_7_live_storage"]
    )

    # Inflow changes
    df["inflow_change_1d"] = (
        df["lag_1_inflow"] - df["lag_2_inflow"]
    )

    df["inflow_change_3d"] = (
        df["lag_1_inflow"] - df["lag_4_inflow"]
    )

    df["inflow_change_7d"] = (
        df["lag_1_inflow"] - df["lag_7_inflow"]
    )

    # Outflow changes
    df["outflow_change_1d"] = (
        df["lag_1_total_outflow"] -
        df["lag_2_total_outflow"]
    )

    df["outflow_change_3d"] = (
        df["lag_1_total_outflow"] -
        df["lag_4_total_outflow"]
    )

    df["outflow_change_7d"] = (
        df["lag_1_total_outflow"] -
        df["lag_7_total_outflow"]
    )

    # Rainfall
    rainfall_3 = [
        "lag_1_rainfall",
        "lag_2_rainfall",
        "lag_3_rainfall"
    ]

    rainfall_7 = [
        f"lag_{i}_rainfall"
        for i in range(1, 8)
    ]

    df["rainfall_3d_sum"] = df[rainfall_3].sum(
        axis=1, min_count=1
    )

    df["rainfall_7d_sum"] = df[rainfall_7].sum(
        axis=1, min_count=1
    )

    # Inflow means
    inflow_3 = [
        f"lag_{i}_inflow"
        for i in range(1, 4)
    ]

    inflow_7 = [
        f"lag_{i}_inflow"
        for i in range(1, 8)
    ]

    df["inflow_3d_mean"] = df[inflow_3].mean(axis=1)
    df["inflow_7d_mean"] = df[inflow_7].mean(axis=1)

    # Outflow means
    outflow_3 = [
        f"lag_{i}_total_outflow"
        for i in range(1, 4)
    ]

    outflow_7 = [
        f"lag_{i}_total_outflow"
        for i in range(1, 8)
    ]

    df["outflow_3d_mean"] = df[outflow_3].mean(axis=1)
    df["outflow_7d_mean"] = df[outflow_7].mean(axis=1)

    # Water balance
    df["water_balance_1d"] = (
        df["lag_1_inflow"] -
        df["lag_1_total_outflow"]
    )

    df["water_balance_3d"] = (
        df["inflow_3d_mean"] -
        df["outflow_3d_mean"]
    )

    df["water_balance_7d"] = (
        df["inflow_7d_mean"] -
        df["outflow_7d_mean"]
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


def main():

    print("========== LOADING REGIME DATA ==========")

    train = prepare_features(
        DATA_DIR / "train.csv"
    )

    val = prepare_features(
        DATA_DIR / "validation.csv"
    )

    print("Train:", train.shape)
    print("Validation:", val.shape)

    # ---------------------------------------------------------
    # RESERVOIR ENCODING
    # ---------------------------------------------------------

    reservoirs = sorted(
        train["reservoir"].dropna().unique()
    )

    reservoir_map = {
        name: i
        for i, name in enumerate(reservoirs)
    }

    for df in [train, val]:

        df["reservoir_code"] = (
            df["reservoir"]
            .map(reservoir_map)
            .astype(float)
        )

    # ---------------------------------------------------------
    # FEATURES
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
    print("Feature count:", len(features))

    # ---------------------------------------------------------
    # TARGET
    # ---------------------------------------------------------

    train["delta"] = (
        train["target_water_level"]
        - train["lag_1_water_level"]
    )

    val["delta"] = (
        val["target_water_level"]
        - val["lag_1_water_level"]
    )

    train["significant"] = (
        train["delta"].abs() >= THRESHOLD
    ).astype(int)

    val["significant"] = (
        val["delta"].abs() >= THRESHOLD
    ).astype(int)

    mask_train = (
        train["target_water_level"].notna()
        & train["lag_1_water_level"].notna()
    )

    mask_val = (
        val["target_water_level"].notna()
        & val["lag_1_water_level"].notna()
    )

    X_train = train.loc[mask_train, features]
    y_train = train.loc[mask_train, "significant"]

    X_val = val.loc[mask_val, features]
    y_val = val.loc[mask_val, "significant"]

    print("\n========== CLASS DISTRIBUTION ==========")
    print("Normal:", (y_train == 0).sum())
    print("Significant:", (y_train == 1).sum())
    print("Significant rate:", round(y_train.mean() * 100, 2), "%")

    # ---------------------------------------------------------
    # TRAIN
    # ---------------------------------------------------------

    print("\n========== TRAINING CLASSIFIER ==========")

    classifier = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42
    )

    classifier.fit(
        X_train,
        y_train
    )

    print("Training complete.")

    # ---------------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------------

    print("\n========== VALIDATION ==========")

    probabilities = classifier.predict_proba(
        X_val
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    print(
        classification_report(
            y_val,
            predictions,
            target_names=[
                "normal",
                "significant"
            ],
            zero_division=0
        )
    )

    print("Confusion matrix:")
    print(
        confusion_matrix(
            y_val,
            predictions
        )
    )

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------

    path = (
        MODEL_DIR /
        "movement_regime_classifier.joblib"
    )

    joblib.dump(
        classifier,
        path
    )

    print("\n========== SAVED ==========")
    print("Model:", path)


if __name__ == "__main__":
    main()
