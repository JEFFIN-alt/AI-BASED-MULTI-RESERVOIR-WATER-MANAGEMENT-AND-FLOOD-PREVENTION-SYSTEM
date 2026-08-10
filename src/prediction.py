import pandas as pd
from pathlib import Path

INPUT = Path("data/processed/kerala_reservoir_validated.csv")
OUTPUT = Path("data/processed/ml_sequences_v3.csv")

WINDOW = 7

FEATURES = [
    "water_level",
    "live_storage",
    "inflow",
    "rainfall",
    "total_outflow",
]


def create_sequences(df):
    sequences = []

    for reservoir, group in df.groupby("reservoir"):
        group = group.sort_values("date").reset_index(drop=True)

        for i in range(WINDOW, len(group)):
            history = group.iloc[i - WINDOW:i]
            target = group.iloc[i]

            # All 7 historical observations must be consecutive.
            history_dates = history["date"].tolist()

            history_ok = all(
                (history_dates[j] - history_dates[j - 1]).days == 1
                for j in range(1, len(history_dates))
            )

            # Target must be the immediate next calendar day.
            target_ok = (
                (target["date"] - history_dates[-1]).days == 1
            )

            if not history_ok or not target_ok:
                continue

            row = {
                "reservoir": reservoir,
                "target_date": target["date"],
                "target_water_level": target["water_level"],
            }

            for lag in range(1, WINDOW + 1):
                source = group.iloc[i - lag]

                for feature in FEATURES:
                    row[f"lag_{lag}_{feature}"] = source[feature]

            sequences.append(row)

    return pd.DataFrame(sequences)


def main():
    print(f"Loading: {INPUT}")

    df = pd.read_csv(INPUT)
    df["date"] = pd.to_datetime(df["date"])

    print(f"Input rows: {len(df):,}")
    print(f"Input reservoirs: {df['reservoir'].nunique()}")

    result = create_sequences(df)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)

    print(f"\nCreated: {OUTPUT}")
    print(f"Output rows: {len(result):,}")
    print(f"Output columns: {len(result.columns)}")
    print(f"Output reservoirs: {result['reservoir'].nunique()}")

    if not result.empty:
        print(
            f"Date range: "
            f"{result['target_date'].min()} → "
            f"{result['target_date'].max()}"
        )

        print("\nSequences per reservoir:")
        print(
            result.groupby("reservoir")
            .size()
            .sort_index()
            .to_string()
        )


if __name__ == "__main__":
    main()
