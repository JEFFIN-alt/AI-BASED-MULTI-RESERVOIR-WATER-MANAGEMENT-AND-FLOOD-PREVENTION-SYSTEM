from pathlib import Path

import pandas as pd


INPUT = Path("data/processed/kerala_reservoir_clean.csv")

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


print("=" * 100)
print("MILESTONE 10.6.2 — SUPERVISED FILTER DIAGNOSTIC")
print("=" * 100)


results = []


for reservoir, group in df.groupby("reservoir"):

    group = group.sort_values("date").reset_index(drop=True)

    counters = {
        "candidate": 0,
        "history_continuous": 0,
        "history_features_complete": 0,
        "target_1d_valid": 0,
        "target_3d_valid": 0,
        "target_7d_valid": 0,
        "all_targets_valid": 0,
        "static_valid": 0,
        "final_samples": 0,
    }

    if group["inflow"].notna().sum() < 20:
        print(f"\n{reservoir}: insufficient inflow")
        continue


    for end_idx in range(
        HISTORY_DAYS - 1,
        len(group),
    ):

        counters["candidate"] += 1

        history_start = end_idx - HISTORY_DAYS + 1

        history = group.iloc[
            history_start:end_idx + 1
        ]

        current = group.iloc[end_idx]

        # ----------------------------------------------------
        # 1. Seven consecutive calendar days
        # ----------------------------------------------------

        history_dates = history["date"]

        date_diffs = (
            history_dates.diff()
            .dropna()
            .dt.days
        )

        if len(date_diffs) != HISTORY_DAYS - 1:
            continue

        if not (date_diffs == 1).all():
            continue

        counters["history_continuous"] += 1


        # ----------------------------------------------------
        # 2. Complete dynamic history
        # ----------------------------------------------------

        if history[DYNAMIC_FEATURES].isna().any().any():
            continue

        counters["history_features_complete"] += 1


        # ----------------------------------------------------
        # 3. Check each target independently
        # ----------------------------------------------------

        current_date = current["date"]

        target_valid = {}

        for target_name, horizon in TARGET_HORIZONS.items():

            target_date = (
                current_date
                + pd.Timedelta(days=horizon)
            )

            match = group[
                group["date"] == target_date
            ]

            valid = (
                len(match) == 1
                and pd.notna(match.iloc[0]["inflow"])
            )

            target_valid[target_name] = valid

            if valid:
                counters[target_name + "_valid"] += 1


        # ----------------------------------------------------
        # 4. All targets
        # ----------------------------------------------------

        all_targets = all(
            target_valid.values()
        )

        if not all_targets:
            continue

        counters["all_targets_valid"] += 1


        # ----------------------------------------------------
        # 5. Static features
        # ----------------------------------------------------

        static_valid = True

        for feature in STATIC_FEATURES:

            if pd.isna(current[feature]):
                static_valid = False
                break

        if not static_valid:
            continue

        counters["static_valid"] += 1


        # ----------------------------------------------------
        # 6. Final sample
        # ----------------------------------------------------

        counters["final_samples"] += 1


    results.append({
        "reservoir": reservoir,
        **counters,
    })


result = pd.DataFrame(results)

print("\n")
print(result.to_string(index=False))


print("\n")
print("=" * 100)
print("FILTER LOSS ANALYSIS")
print("=" * 100)


for _, row in result.iterrows():

    print(f"\n{row['reservoir']}")

    print(
        f"  Candidate endpoints:       "
        f"{row['candidate']}"
    )

    print(
        f"  7-day continuous:          "
        f"{row['history_continuous']}"
    )

    print(
        f"  Complete history:          "
        f"{row['history_features_complete']}"
    )

    print(
        f"  Valid 1-day target:        "
        f"{row['target_1d_valid']}"
    )

    print(
        f"  Valid 3-day target:        "
        f"{row['target_3d_valid']}"
    )

    print(
        f"  Valid 7-day target:        "
        f"{row['target_7d_valid']}"
    )

    print(
        f"  All targets valid:         "
        f"{row['all_targets_valid']}"
    )

    print(
        f"  Static features valid:     "
        f"{row['static_valid']}"
    )

    print(
        f"  FINAL samples:             "
        f"{row['final_samples']}"
    )


print("\nDONE")
