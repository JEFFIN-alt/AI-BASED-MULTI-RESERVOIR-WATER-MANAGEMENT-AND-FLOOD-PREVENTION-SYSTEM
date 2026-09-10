import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


TEST_PATH = "data/processed/model_splits/test.csv"
PRED_PATH = "data/processed/model_results/histgb_v3_test_predictions.csv"


def main():
    print("========== LOADING ==========")

    test = pd.read_csv(TEST_PATH)
    pred = pd.read_csv(PRED_PATH)

    print("Test rows:", len(test))
    print("Prediction rows:", len(pred))

    # Make sure both files use the same row identity
    keys = ["reservoir", "target_date"]

    test["target_date"] = pd.to_datetime(test["target_date"])
    pred["target_date"] = pd.to_datetime(pred["target_date"])

    # Keep only the columns we need from predictions
    pred_cols = keys + ["prediction"]
    pred = pred[pred_cols]

    df = test.merge(
        pred,
        on=keys,
        how="inner"
    )

    print("Matched rows:", len(df))

    # Remove rows where target or prediction is unavailable
    df = df.dropna(subset=["target_water_level", "prediction"]).copy()

    # Persistence prediction = previous day's water level
    df["persistence"] = df["lag_1_water_level"]

    df = df.dropna(subset=["persistence"]).copy()

    # Errors
    df["ml_error"] = df["prediction"] - df["target_water_level"]
    df["persistence_error"] = (
        df["persistence"] - df["target_water_level"]
    )

    df["ml_abs_error"] = df["ml_error"].abs()
    df["persistence_abs_error"] = df["persistence_error"].abs()

    # Water-level movement
    df["water_level_change"] = (
        df["target_water_level"] - df["lag_1_water_level"]
    )

    df["abs_water_level_change"] = df["water_level_change"].abs()

    # Which method wins?
    df["winner"] = np.where(
        df["ml_abs_error"] < df["persistence_abs_error"],
        "ML",
        np.where(
            df["ml_abs_error"] > df["persistence_abs_error"],
            "Persistence",
            "Tie"
        )
    )

    print("\n========== OVERALL ==========")

    print("Rows evaluated:", len(df))

    print(
        "ML MAE:",
        round(df["ml_abs_error"].mean(), 6)
    )

    print(
        "Persistence MAE:",
        round(df["persistence_abs_error"].mean(), 6)
    )

    print("\nWinner count:")
    print(df["winner"].value_counts().to_string())

    print("\nML win rate:")
    print(
        round(
            (df["winner"] == "ML").mean() * 100,
            2
        ),
        "%"
    )

    # --------------------------------------------------
    # BY RESERVOIR
    # --------------------------------------------------

    print("\n========== BY RESERVOIR ==========")

    reservoir_results = (
        df.groupby("reservoir")
        .agg(
            samples=("target_water_level", "size"),
            ml_mae=("ml_abs_error", "mean"),
            persistence_mae=("persistence_abs_error", "mean"),
            ml_wins=("winner", lambda x: (x == "ML").sum())
        )
        .reset_index()
    )

    reservoir_results["ml_win_rate"] = (
        reservoir_results["ml_wins"]
        / reservoir_results["samples"]
        * 100
    )

    reservoir_results["improvement"] = (
        reservoir_results["persistence_mae"]
        - reservoir_results["ml_mae"]
    )

    reservoir_results = reservoir_results.sort_values(
        "improvement",
        ascending=False
    )

    print(
        reservoir_results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # --------------------------------------------------
    # BY WATER LEVEL MOVEMENT
    # --------------------------------------------------

    print("\n========== BY WATER-LEVEL MOVEMENT ==========")

    bins = [-np.inf, 0.1, 0.5, 1.0, 2.0, 5.0, np.inf]

    labels = [
        "<=0.1",
        "0.1-0.5",
        "0.5-1",
        "1-2",
        "2-5",
        ">5"
    ]

    df["movement_bucket"] = pd.cut(
        df["abs_water_level_change"],
        bins=bins,
        labels=labels
    )

    movement_results = (
        df.groupby("movement_bucket", observed=False)
        .agg(
            samples=("target_water_level", "size"),
            ml_mae=("ml_abs_error", "mean"),
            persistence_mae=("persistence_abs_error", "mean"),
            ml_wins=("winner", lambda x: (x == "ML").sum())
        )
        .reset_index()
    )

    movement_results["ml_win_rate"] = (
        movement_results["ml_wins"]
        / movement_results["samples"]
        * 100
    )

    movement_results["improvement"] = (
        movement_results["persistence_mae"]
        - movement_results["ml_mae"]
    )

    print(
        movement_results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # --------------------------------------------------
    # LARGEST ML ERRORS
    # --------------------------------------------------

    print("\n========== LARGEST ML ERRORS ==========")

    worst = df.sort_values(
        "ml_abs_error",
        ascending=False
    ).head(20)

    print(
        worst[
            [
                "reservoir",
                "target_date",
                "lag_1_water_level",
                "target_water_level",
                "prediction",
                "persistence",
                "ml_abs_error",
                "persistence_abs_error",
                "water_level_change",
                "winner"
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------
    # CASES WHERE ML HELPS MOST
    # --------------------------------------------------

    df["ml_advantage"] = (
        df["persistence_abs_error"]
        - df["ml_abs_error"]
    )

    print("\n========== ML BIGGEST WINS ==========")

    best = df.sort_values(
        "ml_advantage",
        ascending=False
    ).head(15)

    print(
        best[
            [
                "reservoir",
                "target_date",
                "target_water_level",
                "prediction",
                "persistence",
                "ml_abs_error",
                "persistence_abs_error",
                "ml_advantage",
                "water_level_change"
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------
    # SAVE
    # --------------------------------------------------

    output = "data/processed/model_results/v3_error_analysis.csv"

    df.to_csv(output, index=False)

    print("\n========== SAVED ==========")
    print(output)


if __name__ == "__main__":
    main()
