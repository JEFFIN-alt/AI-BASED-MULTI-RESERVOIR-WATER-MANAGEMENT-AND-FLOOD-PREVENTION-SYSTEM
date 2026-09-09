import pandas as pd
from pathlib import Path

INPUT = Path("data/processed/kerala_reservoir_validated.csv")
OUTPUT = Path("data/processed/kerala_reservoir_annotated.csv")

df = pd.read_csv(INPUT)

df["date"] = pd.to_datetime(df["date"], errors="coerce")

for col in [
    "inflow",
    "water_level",
    "live_storage",
    "rainfall",
    "total_outflow",
]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.sort_values(["reservoir", "date"]).reset_index(drop=True)

df["prev_inflow"] = df.groupby("reservoir")["inflow"].shift(1)
df["next_inflow"] = df.groupby("reservoir")["inflow"].shift(-1)

df["abs_level_difference"] = (
    df["inflow"] - df["water_level"]
).abs()

# Strong source-field anomaly:
# inflow is extremely close to water level.
level_like = (
    df["inflow"].notna()
    & df["water_level"].notna()
    & (df["abs_level_difference"] <= 2)
    & (df["inflow"] > 100)
)

# Isolated extreme spike:
# current inflow is >100 and both neighboring inflows
# are less than 10% of the current value.
isolated_spike = (
    df["inflow"].notna()
    & (df["inflow"] > 100)
    & df["prev_inflow"].notna()
    & df["next_inflow"].notna()
    & (df["prev_inflow"] < df["inflow"] * 0.1)
    & (df["next_inflow"] < df["inflow"] * 0.1)
)

df["anomaly_class"] = "NORMAL"
df["anomaly_reason"] = ""

df.loc[level_like, "anomaly_class"] = "LEVEL_LIKE_ERROR"
df.loc[level_like, "anomaly_reason"] = (
    "Inflow is extremely close to water level and "
    "is inconsistent with neighboring inflow observations."
)

spike_mask = isolated_spike & ~level_like

df.loc[spike_mask, "anomaly_class"] = "SUSPICIOUS_SPIKE"
df.loc[spike_mask, "anomaly_reason"] = (
    "Extreme isolated inflow spike with neighboring "
    "inflow values far lower."
)

df["training_eligible"] = ~df["anomaly_class"].isin(
    ["LEVEL_LIKE_ERROR", "SUSPICIOUS_SPIKE"]
)

print("=" * 90)
print("MILESTONE 9.1 — ANOMALY POLICY")
print("=" * 90)

print("\nTotal rows:", len(df))

print("\nAnomaly classes:")
print(df["anomaly_class"].value_counts().to_string())

print("\nTraining eligibility:")
print(df["training_eligible"].value_counts().to_string())

print("\nExcluded rows by class:")
print(
    df.loc[~df["training_eligible"], "anomaly_class"]
    .value_counts()
    .to_string()
)

print("\nExamples of excluded records:")
print(
    df.loc[
        ~df["training_eligible"],
        [
            "date",
            "reservoir",
            "inflow",
            "water_level",
            "prev_inflow",
            "next_inflow",
            "anomaly_class",
        ],
    ]
    .head(30)
    .to_string(index=False)
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)

print("\nSaved:")
print(OUTPUT)

print("\nDONE")
