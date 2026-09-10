import pandas as pd
import numpy as np
from pathlib import Path

INPUT_DIR = Path("data/processed/model_splits")
OUTPUT_DIR = Path("data/processed/model_splits_v4")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def add_v4_features(df):
    df = df.copy()

    df["target_date"] = pd.to_datetime(df["target_date"])

    # =========================================================
    # WATER LEVEL DYNAMICS
    # =========================================================

    df["level_delta_1"] = (
        df["lag_1_water_level"] -
        df["lag_2_water_level"]
    )

    df["level_delta_2"] = (
        df["lag_2_water_level"] -
        df["lag_3_water_level"]
    )

    df["level_delta_3"] = (
        df["lag_3_water_level"] -
        df["lag_4_water_level"]
    )

    # Acceleration / deceleration
    df["level_acceleration_1"] = (
        df["level_delta_1"] -
        df["level_delta_2"]
    )

    df["level_acceleration_2"] = (
        df["level_delta_2"] -
        df["level_delta_3"]
    )

    # Short-term change
    df["level_change_2"] = (
        df["lag_1_water_level"] -
        df["lag_3_water_level"]
    )

    df["level_change_5"] = (
        df["lag_1_water_level"] -
        df["lag_6_water_level"]
    )

    # =========================================================
    # LEVEL VOLATILITY
    # =========================================================

    level_deltas = pd.concat(
        [
            df["lag_1_water_level"] - df["lag_2_water_level"],
            df["lag_2_water_level"] - df["lag_3_water_level"],
            df["lag_3_water_level"] - df["lag_4_water_level"],
            df["lag_4_water_level"] - df["lag_5_water_level"],
            df["lag_5_water_level"] - df["lag_6_water_level"],
            df["lag_6_water_level"] - df["lag_7_water_level"],
        ],
        axis=1
    )

    df["level_delta_std"] = level_deltas.std(axis=1)
    df["level_delta_abs_mean"] = level_deltas.abs().mean(axis=1)
    df["level_delta_max"] = level_deltas.abs().max(axis=1)

    # =========================================================
    # RAINFALL EVENT FEATURES
    # =========================================================

    rain_cols = [
        f"lag_{i}_rainfall"
        for i in range(1, 8)
    ]

    rain = df[rain_cols]

    df["rainfall_recent_max"] = rain.max(axis=1)
    df["rainfall_recent_mean"] = rain.mean(axis=1)

    df["rainfall_nonzero_count"] = (
        (rain > 0).sum(axis=1)
    )

    df["rainfall_heavy_count"] = (
        (rain >= 10).sum(axis=1)
    )

    df["rainfall_recent_1"] = df["lag_1_rainfall"]

    df["rainfall_recent_3"] = (
        df[
            [
                "lag_1_rainfall",
                "lag_2_rainfall",
                "lag_3_rainfall"
            ]
        ].sum(axis=1, min_count=1)
    )

    # Recency-weighted rainfall
    weights = np.array(
        [7, 6, 5, 4, 3, 2, 1],
        dtype=float
    )

    rain_values = rain.to_numpy(dtype=float)

    weighted_sum = np.nansum(
        rain_values * weights,
        axis=1
    )

    weight_available = np.sum(
        ~np.isnan(rain_values) * weights,
        axis=1
    )

    df["rainfall_weighted_7"] = np.where(
        weight_available > 0,
        weighted_sum / weight_available,
        np.nan
    )

    # =========================================================
    # INFLOW DYNAMICS
    # =========================================================

    df["inflow_delta_1"] = (
        df["lag_1_inflow"] -
        df["lag_2_inflow"]
    )

    df["inflow_delta_3"] = (
        df["lag_1_inflow"] -
        df["lag_4_inflow"]
    )

    inflow_cols = [
        f"lag_{i}_inflow"
        for i in range(1, 8)
    ]

    inflow = df[inflow_cols]

    df["inflow_recent_max"] = inflow.max(axis=1)
    df["inflow_recent_min"] = inflow.min(axis=1)
    df["inflow_recent_std"] = inflow.std(axis=1)

    # =========================================================
    # OUTFLOW DYNAMICS
    # =========================================================

    df["outflow_delta_1"] = (
        df["lag_1_total_outflow"] -
        df["lag_2_total_outflow"]
    )

    df["outflow_delta_3"] = (
        df["lag_1_total_outflow"] -
        df["lag_4_total_outflow"]
    )

    outflow_cols = [
        f"lag_{i}_total_outflow"
        for i in range(1, 8)
    ]

    outflow = df[outflow_cols]

    df["outflow_recent_max"] = outflow.max(axis=1)
    df["outflow_recent_std"] = outflow.std(axis=1)

    # =========================================================
    # WATER BALANCE / PRESSURE
    # =========================================================

    df["balance_recent"] = (
        df["lag_1_inflow"] -
        df["lag_1_total_outflow"]
    )

    df["balance_3_mean"] = (
        df[
            ["lag_1_inflow",
             "lag_2_inflow",
             "lag_3_inflow"]
        ].mean(axis=1)
        -
        df[
            ["lag_1_total_outflow",
             "lag_2_total_outflow",
             "lag_3_total_outflow"]
        ].mean(axis=1)
    )

    df["balance_7_mean"] = (
        inflow.mean(axis=1)
        -
        outflow.mean(axis=1)
    )

    # =========================================================
    # INFLOW / OUTFLOW RATIO
    # =========================================================

    denominator = (
        df["lag_1_total_outflow"].abs() + 1e-6
    )

    df["inflow_outflow_ratio"] = (
        df["lag_1_inflow"] / denominator
    )

    # =========================================================
    # STORAGE DYNAMICS
    # =========================================================

    df["storage_delta_1"] = (
        df["lag_1_live_storage"] -
        df["lag_2_live_storage"]
    )

    df["storage_delta_3"] = (
        df["lag_1_live_storage"] -
        df["lag_4_live_storage"]
    )

    # =========================================================
    # SEASONAL FEATURES
    # =========================================================

    month = df["target_date"].dt.month

    df["month_sin"] = np.sin(
        2 * np.pi * month / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * month / 12
    )

    # Monsoon-season indicator
    df["monsoon_flag"] = month.isin(
        [6, 7, 8, 9]
    ).astype(int)

    # =========================================================
    # EXTREME-MOVEMENT FLAGS
    # =========================================================

    df["rapid_change_flag"] = (
        df["level_delta_abs_mean"] > 0.5
    ).astype(int)

    df["large_change_flag"] = (
        df["level_delta_max"] > 2.0
    ).astype(int)

    df["rain_event_flag"] = (
        df["rainfall_recent_max"] >= 10
    ).astype(int)

    return df


def main():

    print("=" * 60)
    print("V4 FEATURE PREPARATION")
    print("=" * 60)

    for split in ["train", "validation", "test"]:

        input_path = INPUT_DIR / f"{split}.csv"
        output_path = OUTPUT_DIR / f"{split}.csv"

        print(f"\nProcessing {split}...")

        df = pd.read_csv(input_path)

        print("Original:", df.shape)

        df = add_v4_features(df)

        df.to_csv(
            output_path,
            index=False
        )

        print("V4:", df.shape)
        print("Saved:", output_path)

        # Feature sanity check
        v4_cols = [
            c for c in df.columns
            if c not in [
                "reservoir",
                "target_date",
                "target_water_level"
            ]
        ]

        print("Feature columns:", len(v4_cols))
        print(
            "Rows with NaN:",
            df[v4_cols].isna().any(axis=1).sum()
        )

    print("\n" + "=" * 60)
    print("V4 DATA PREPARATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
