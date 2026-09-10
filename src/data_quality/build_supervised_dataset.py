from pathlib import Path

import pandas as pd
import numpy as np


INPUT = Path(
    "data/processed/kerala_reservoir_clean.csv"
)

OUTPUT = Path(
    "data/processed/forecasting_supervised_7day.csv"
)


# ============================================================
# MILESTONE 10.7 — SUPERVISED FORECASTING DATASET V2
# ============================================================

print("=" * 100)
print("MILESTONE 10.7 — SUPERVISED FORECASTING DATASET V2")
print("=" * 100)


# ============================================================
# CONFIGURATION
# ============================================================

HISTORY_DAYS = 7

TARGET_HORIZONS = {
    "target_1d": 1,
    "target_3d": 3,
    "target_7d": 7,
}

DYNAMIC_FEATURES = [
    "inflow",
    "water_level",
    "live_storage",
    "rainfall",
    "total_outflow",
]

STATIC_FEATURES = [
    "latitude",
    "longitude",
    "FRL",
    "MWL",
    "blue_level",
    "orange_level",
    "red_level",
    "rule_level",
]


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce",
)

for col in DYNAMIC_FEATURES + STATIC_FEATURES:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce",
    )

df = df.sort_values(
    ["reservoir", "date"]
).reset_index(drop=True)


print("\nInput dataset:")
print("Rows:", len(df))
print("Reservoirs:", df["reservoir"].nunique())


# ============================================================
# INFLOW AVAILABILITY
# ============================================================

inflow_counts = (
    df.groupby("reservoir")["inflow"]
    .apply(lambda x: x.notna().sum())
    .sort_values()
)

print("\nInflow availability:")
print(inflow_counts.to_string())


# ============================================================
# BUILD STATIC FEATURE LOOKUP
#
# Static features are constant by reservoir, but some fields
# are completely unavailable for certain reservoirs.
#
# We therefore:
#   1. take the first available value for each static field
#   2. allow missing static fields
#   3. keep a static availability mask
#
# This prevents entire reservoirs from being discarded merely
# because one optional threshold field is unavailable.
# ============================================================

static_lookup = {}

for reservoir, group in df.groupby("reservoir", sort=False):

    values = {}
    availability = {}

    for feature in STATIC_FEATURES:

        series = group[feature].dropna()

        if len(series) > 0:
            values[feature] = float(series.iloc[0])
            availability[feature] = 1
        else:
            values[feature] = 0.0
            availability[feature] = 0

    static_lookup[reservoir] = {
        "values": values,
        "availability": availability,
    }


# ============================================================
# BUILD SAMPLES
# ============================================================

samples = []

diagnostics = []


for reservoir, group in df.groupby(
    "reservoir",
    sort=False,
):

    group = (
        group
        .sort_values("date")
        .reset_index(drop=True)
    )

    inflow_count = group["inflow"].notna().sum()

    # --------------------------------------------------------
    # Skip reservoirs without enough inflow observations
    # --------------------------------------------------------

    if inflow_count < 20:

        print(
            f"\nSkipping {reservoir}: "
            f"insufficient inflow observations ({inflow_count})"
        )

        continue

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    candidate_count = 0
    continuous_count = 0
    complete_history_count = 0

    target_1d_count = 0
    target_3d_count = 0
    target_7d_count = 0

    all_targets_count = 0

    static_info = static_lookup[reservoir]

    static_values = static_info["values"]
    static_availability = static_info["availability"]

    # --------------------------------------------------------
    # Iterate over possible history endpoints
    # --------------------------------------------------------

    for end_idx in range(
        HISTORY_DAYS - 1,
        len(group),
    ):

        candidate_count += 1

        history_start = (
            end_idx - HISTORY_DAYS + 1
        )

        history = group.iloc[
            history_start:end_idx + 1
        ]

        current = group.iloc[end_idx]

        # ----------------------------------------------------
        # Check 7 consecutive calendar days
        #
        # IMPORTANT:
        # Do NOT use:
        #
        # history["date"].diff().dt.days
        #
        # because that caused a native pandas segfault in
        # the current environment.
        # ----------------------------------------------------

        history_dates = (
            history["date"]
            .to_numpy(dtype="datetime64[D]")
        )

        date_diffs = (
            history_dates[1:]
            - history_dates[:-1]
        ).astype("timedelta64[D]").astype(np.int64)

        if len(date_diffs) != HISTORY_DAYS - 1:
            continue

        if not np.all(date_diffs == 1):
            continue

        continuous_count += 1

        # ----------------------------------------------------
        # Require complete dynamic history
        # ----------------------------------------------------

        if history[DYNAMIC_FEATURES].isna().any().any():
            continue

        complete_history_count += 1

        # ----------------------------------------------------
        # Target dates
        # ----------------------------------------------------

        current_date = current["date"]

        target_rows = {}

        valid_targets = True

        for target_name, horizon in TARGET_HORIZONS.items():

            target_date = (
                current_date
                + pd.Timedelta(days=horizon)
            )

            matches = group[
                group["date"] == target_date
            ]

            if len(matches) != 1:
                valid_targets = False
                break

            target_row = matches.iloc[0]

            if pd.isna(target_row["inflow"]):
                valid_targets = False
                break

            target_rows[target_name] = target_row

            if target_name == "target_1d":
                target_1d_count += 1

            elif target_name == "target_3d":
                target_3d_count += 1

            elif target_name == "target_7d":
                target_7d_count += 1

        if not valid_targets:
            continue

        all_targets_count += 1

        # ----------------------------------------------------
        # Flatten dynamic history
        # ----------------------------------------------------

        sample = {
            "date": current_date,
            "reservoir": reservoir,
        }

        # ----------------------------------------------------
        # 7-day dynamic features
        #
        # day_1 = oldest
        # day_7 = current
        # ----------------------------------------------------

        for day_idx in range(HISTORY_DAYS):

            row = history.iloc[day_idx]

            day_number = day_idx + 1

            for feature in DYNAMIC_FEATURES:

                sample[
                    f"{feature}_day_{day_number}"
                ] = float(row[feature])

        # ----------------------------------------------------
        # Static features
        #
        # Missing static fields are represented by 0.0.
        # Availability flags tell the model whether the value
        # actually exists.
        # ----------------------------------------------------

        for feature in STATIC_FEATURES:

            sample[
                f"static_{feature}"
            ] = static_values[feature]

            sample[
                f"static_{feature}_available"
            ] = static_availability[feature]

        # ----------------------------------------------------
        # Targets
        # ----------------------------------------------------

        sample["target_1d"] = float(
            target_rows["target_1d"]["inflow"]
        )

        sample["target_3d"] = float(
            target_rows["target_3d"]["inflow"]
        )

        sample["target_7d"] = float(
            target_rows["target_7d"]["inflow"]
        )

        samples.append(sample)

    # --------------------------------------------------------
    # Reservoir diagnostics
    # --------------------------------------------------------

    diagnostics.append(
        {
            "reservoir": reservoir,
            "candidate": candidate_count,
            "history_continuous": continuous_count,
            "history_features_complete": complete_history_count,
            "target_1d_valid": target_1d_count,
            "target_3d_valid": target_3d_count,
            "target_7d_valid": target_7d_count,
            "all_targets_valid": all_targets_count,
            "final_samples": all_targets_count,
        }
    )


# ============================================================
# CREATE DATAFRAME
# ============================================================

result = pd.DataFrame(samples)


# ============================================================
# DIAGNOSTICS
# ============================================================

print("\n")
print("=" * 100)
print("FILTER DIAGNOSTICS")
print("=" * 100)

if diagnostics:

    diagnostics_df = pd.DataFrame(
        diagnostics
    )

    print(
        diagnostics_df.to_string(
            index=False
        )
    )


# ============================================================
# SAFETY CHECK
# ============================================================

if result.empty:

    raise RuntimeError(
        "No supervised samples were created."
    )


# ============================================================
# SORT
# ============================================================

result = result.sort_values(
    ["reservoir", "date"]
).reset_index(drop=True)


# ============================================================
# TARGET STATISTICS
# ============================================================

print("\n")
print("=" * 100)
print("SUPERVISED DATASET SUMMARY")
print("=" * 100)

print("\nSamples:", len(result))

print(
    "Reservoirs:",
    result["reservoir"].nunique(),
)

print(
    "Date:",
    result["date"].min(),
    "->",
    result["date"].max(),
)

print("\nSamples by reservoir:")

print(
    result["reservoir"]
    .value_counts()
    .sort_index()
    .to_string()
)


print("\nTarget statistics:")

print(
    result[
        [
            "target_1d",
            "target_3d",
            "target_7d",
        ]
    ]
    .describe()
    .to_string()
)


print("\nTarget missing values:")

print(
    result[
        [
            "target_1d",
            "target_3d",
            "target_7d",
        ]
    ]
    .isna()
    .sum()
    .to_string()
)


# ============================================================
# FEATURE INFORMATION
# ============================================================

target_columns = [
    "target_1d",
    "target_3d",
    "target_7d",
]

excluded_columns = [
    "date",
    "reservoir",
]

feature_columns = [
    col
    for col in result.columns
    if col not in target_columns
    and col not in excluded_columns
]


print("\nFeature count:", len(feature_columns))


# ============================================================
# DYNAMIC FEATURE CHECK
# ============================================================

print("\nExpected dynamic feature columns:")

for feature in DYNAMIC_FEATURES:

    columns = [
        col
        for col in result.columns
        if col.startswith(
            feature + "_day_"
        )
    ]

    print(
        f"{feature:<24}: "
        f"{len(columns)}"
    )


# ============================================================
# STATIC FEATURE CHECK
# ============================================================

print("\nStatic feature columns:")

for feature in STATIC_FEATURES:

    column = (
        f"static_{feature}"
    )

    availability_column = (
        f"static_{feature}_available"
    )

    print(
        f"{column:<35}: "
        f"{'YES' if column in result.columns else 'NO'}"
    )

    print(
        f"{availability_column:<35}: "
        f"{'YES' if availability_column in result.columns else 'NO'}"
    )


# ============================================================
# LEAKAGE CHECK
# ============================================================

print("\n")
print("=" * 100)
print("LEAKAGE CHECK")
print("=" * 100)

leakage_terms = [
    "target",
    "future",
]

leakage_columns = []

for column in feature_columns:

    lower = column.lower()

    if any(
        term in lower
        for term in leakage_terms
    ):
        leakage_columns.append(column)


if leakage_columns:

    print(
        "FAIL — possible leakage columns:"
    )

    for column in leakage_columns:
        print(" ", column)

else:

    print(
        "PASS — no target/future columns "
        "found in feature matrix."
    )


# ============================================================
# FINAL SHAPE
# ============================================================

print("\nFinal shape:", result.shape)


# ============================================================
# SAVE
# ============================================================

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

result.to_csv(
    OUTPUT,
    index=False,
)

print("\nSaved:")
print(OUTPUT)

print("\nDONE")
