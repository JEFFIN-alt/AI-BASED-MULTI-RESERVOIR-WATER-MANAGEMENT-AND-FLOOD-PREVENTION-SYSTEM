import pandas as pd
import numpy as np


INPUT = "data/processed/kerala_reservoir_validated.csv"
OUTPUT = "data/processed/inflow_anomaly_classification.csv"


print("=" * 90)
print("MILESTONE 8.3 — INFLOW ANOMALY CLASSIFICATION ENGINE")
print("=" * 90)


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(INPUT)

df["date"] = pd.to_datetime(df["date"], errors="coerce")

numeric_cols = [
    "inflow",
    "water_level",
    "live_storage",
    "rainfall",
    "total_outflow",
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# ============================================================
# 2. SORT TEMPORALLY
# ============================================================

df = df.sort_values(
    ["reservoir", "date"]
).reset_index(drop=True)


# ============================================================
# 3. TEMPORAL FEATURES
# ============================================================

group = df.groupby("reservoir", group_keys=False)

df["prev_inflow"] = group["inflow"].shift(1)
df["next_inflow"] = group["inflow"].shift(-1)

df["prev2_inflow"] = group["inflow"].shift(2)
df["next2_inflow"] = group["inflow"].shift(-2)


# ============================================================
# 4. BASIC ANOMALY FEATURES
# ============================================================

df["abs_inflow_water_level_diff"] = (
    df["inflow"] - df["water_level"]
).abs()

df["relative_inflow_water_level_diff_pct"] = (
    df["abs_inflow_water_level_diff"]
    / df["water_level"].abs().replace(0, np.nan)
) * 100


df["prev_ratio"] = (
    df["inflow"]
    / df["prev_inflow"].replace(0, np.nan)
)

df["next_ratio"] = (
    df["inflow"]
    / df["next_inflow"].replace(0, np.nan)
)


df["neighbor_mean"] = (
    df[["prev_inflow", "next_inflow"]]
    .mean(axis=1)
)

df["neighbor_max"] = (
    df[["prev_inflow", "next_inflow"]]
    .max(axis=1)
)


df["neighbor_ratio"] = (
    df["inflow"]
    / df["neighbor_mean"].replace(0, np.nan)
)


# ============================================================
# 5. EXTREME DEFINITION
# ============================================================

df["extreme_inflow"] = (
    df["inflow"] > 100
)


# ============================================================
# 6. ISOLATED SPIKE
# ============================================================

df["isolated_spike"] = (
    (df["inflow"] > 100)
    &
    (
        df["neighbor_max"] < df["inflow"] * 0.25
    )
)


# ============================================================
# 7. SUSTAINED EXTREME
# ============================================================

df["sustained_extreme"] = (
    (df["inflow"] > 100)
    &
    (
        (df["prev_inflow"] > 100)
        |
        (df["next_inflow"] > 100)
    )
)


# ============================================================
# 8. LEVEL-LIKE VALUE
# ============================================================

df["level_like"] = (
    (df["inflow"] > 100)
    &
    (
        df["abs_inflow_water_level_diff"] <= 5
    )
)


# ============================================================
# 9. RAINFALL SUPPORT
# ============================================================

df["rainfall_supported"] = (
    df["rainfall"].fillna(0) >= 20
)


# ============================================================
# 10. PHYSICAL RESPONSE
# ============================================================

df["physical_response"] = (
    (
        df["live_storage"].notna()
    )
    &
    (
        df["total_outflow"].notna()
    )
)


# ============================================================
# 11. CLASSIFICATION
# ============================================================

conditions = [
    # --------------------------------------------------------
    # LEVEL-LIKE ERROR
    # --------------------------------------------------------
    (
        df["extreme_inflow"]
        &
        df["level_like"]
        &
        df["isolated_spike"]
    ),

    # --------------------------------------------------------
    # STRONG ISOLATED SUSPICIOUS SPIKE
    # --------------------------------------------------------
    (
        df["extreme_inflow"]
        &
        df["isolated_spike"]
        &
        ~df["rainfall_supported"]
    ),

    # --------------------------------------------------------
    # GENUINE / SUPPORTED EXTREME
    # --------------------------------------------------------
    (
        df["extreme_inflow"]
        &
        (
            df["sustained_extreme"]
            |
            df["rainfall_supported"]
        )
    ),

    # --------------------------------------------------------
    # OTHER EXTREME
    # --------------------------------------------------------
    (
        df["extreme_inflow"]
    ),
]


choices = [
    "LEVEL_LIKE_ERROR",
    "SUSPICIOUS_SPIKE",
    "GENUINE_EXTREME",
    "UNCERTAIN",
]


df["anomaly_class"] = np.select(
    conditions,
    choices,
    default="NORMAL"
)


# ============================================================
# 12. CONFIDENCE
# ============================================================

df["classification_confidence"] = "LOW"

df.loc[
    df["anomaly_class"] == "LEVEL_LIKE_ERROR",
    "classification_confidence"
] = "HIGH"

df.loc[
    df["anomaly_class"] == "SUSPICIOUS_SPIKE",
    "classification_confidence"
] = "HIGH"

df.loc[
    df["anomaly_class"] == "GENUINE_EXTREME",
    "classification_confidence"
] = "MEDIUM"


# ============================================================
# 13. SAVE
# ============================================================

df.to_csv(
    OUTPUT,
    index=False
)


# ============================================================
# 14. REPORT
# ============================================================

print("\nClassification counts:")
print(
    df["anomaly_class"]
    .value_counts()
    .to_string()
)


print("\nExtreme classification:")
print(
    df.loc[
        df["extreme_inflow"],
        "anomaly_class"
    ]
    .value_counts()
    .to_string()
)


print("\nConfidence:")
print(
    df["classification_confidence"]
    .value_counts()
    .to_string()
)


print("\nLevel-like records:")
print(
    df["level_like"].sum()
)


print("\nIsolated spikes:")
print(
    df["isolated_spike"].sum()
)


print("\nSustained extremes:")
print(
    df["sustained_extreme"].sum()
)


print("\nRainfall-supported extremes:")
print(
    (
        df["extreme_inflow"]
        &
        df["rainfall_supported"]
    ).sum()
)


print("\n" + "=" * 90)
print("SAVED")
print("=" * 90)

print(OUTPUT)

print("=" * 90)
print("MILESTONE 8.3 COMPLETE")
print("=" * 90)
