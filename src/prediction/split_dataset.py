import pandas as pd
from pathlib import Path

INPUT = Path("data/processed/ml_sequences_v3.csv")
OUTPUT_DIR = Path("data/processed/model_splits")

TRAIN_END = pd.Timestamp("2023-12-31")
VAL_END = pd.Timestamp("2024-12-31")


def main():
    df = pd.read_csv(INPUT)
    df["target_date"] = pd.to_datetime(df["target_date"])

    train = df[df["target_date"] <= TRAIN_END].copy()

    validation = df[
        (df["target_date"] > TRAIN_END)
        & (df["target_date"] <= VAL_END)
    ].copy()

    test = df[df["target_date"] > VAL_END].copy()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train.to_csv(OUTPUT_DIR / "train.csv", index=False)
    validation.to_csv(OUTPUT_DIR / "validation.csv", index=False)
    test.to_csv(OUTPUT_DIR / "test.csv", index=False)

    print("========== GLOBAL CHRONOLOGICAL SPLIT ==========")

    for name, data in [
        ("TRAIN", train),
        ("VALIDATION", validation),
        ("TEST", test),
    ]:
        print(f"\n{name}")
        print("Rows:", len(data))
        print(
            "Date:",
            data["target_date"].min().date(),
            "to",
            data["target_date"].max().date()
        )
        print("Reservoirs:", data["reservoir"].nunique())

        print("\nRows by reservoir:")
        print(data.groupby("reservoir").size().to_string())

    print("\nFiles created:")
    print(OUTPUT_DIR / "train.csv")
    print(OUTPUT_DIR / "validation.csv")
    print(OUTPUT_DIR / "test.csv")


if __name__ == "__main__":
    main()
