from pathlib import Path

import pandas as pd


# ============================================================
# MILESTONE 10.8 — TEMPORAL TRAIN / VALIDATION / TEST SPLIT
# ============================================================

INPUT = Path(
    "data/processed/forecasting_supervised_7day.csv"
)

OUTPUT_DIR = Path(
    "data/processed/splits"
)

TRAIN_OUTPUT = OUTPUT_DIR / "train.csv"
VAL_OUTPUT = OUTPUT_DIR / "validation.csv"
TEST_OUTPUT = OUTPUT_DIR / "test.csv"


print("=" * 100)
print("MILESTONE 10.8 — TEMPORAL TRAIN / VALIDATION / TEST SPLIT")
print("=" * 100)


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_END = pd.Timestamp("2023-12-31")
VAL_END = pd.Timestamp("2024-12-31")


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce",
)

df = df.sort_values(
    ["date", "reservoir"]
).reset_index(drop=True)


print("\nInput dataset:")
print("Rows:", len(df))
print("Reservoirs:", df["reservoir"].nunique())
print("Date range:", df["date"].min(), "->", df["date"].max())


# ============================================================
# CHECK INVALID DATES
# ============================================================

invalid_dates = df["date"].isna().sum()

print("\nInvalid dates:", invalid_dates)

if invalid_dates > 0:
    raise RuntimeError(
        "Dataset contains invalid dates."
    )


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

train = df[
    df["date"] <= TRAIN_END
].copy()

validation = df[
    (df["date"] > TRAIN_END)
    & (df["date"] <= VAL_END)
].copy()

test = df[
    df["date"] > VAL_END
].copy()


# ============================================================
# BASIC SUMMARY
# ============================================================

print("\n")
print("=" * 100)
print("SPLIT SUMMARY")
print("=" * 100)

print("\nTRAIN")
print("Rows:", len(train))
print(
    "Date:",
    train["date"].min(),
    "->",
    train["date"].max(),
)
print(
    "Reservoirs:",
    train["reservoir"].nunique(),
)

print("\nVALIDATION")
print("Rows:", len(validation))
print(
    "Date:",
    validation["date"].min(),
    "->",
    validation["date"].max(),
)
print(
    "Reservoirs:",
    validation["reservoir"].nunique(),
)

print("\nTEST")
print("Rows:", len(test))
print(
    "Date:",
    test["date"].min(),
    "->",
    test["date"].max(),
)
print(
    "Reservoirs:",
    test["reservoir"].nunique(),
)


# ============================================================
# ROW CONSERVATION
# ============================================================

print("\n")
print("=" * 100)
print("ROW CONSERVATION CHECK")
print("=" * 100)

total_split_rows = (
    len(train)
    + len(validation)
    + len(test)
)

print("Original rows:", len(df))
print("Split rows:", total_split_rows)

if total_split_rows != len(df):

    raise RuntimeError(
        "FAIL — split rows do not equal input rows."
    )

print("PASS — all rows accounted for.")


# ============================================================
# DATE RANGE CHECK
# ============================================================

print("\n")
print("=" * 100)
print("DATE ORDER CHECK")
print("=" * 100)

checks = {
    "train_max <= validation_min":
        train["date"].max()
        < validation["date"].min(),

    "validation_max < test_min":
        validation["date"].max()
        < test["date"].min(),
}


for name, passed in checks.items():

    print(
        f"{name}:",
        "PASS" if passed else "FAIL"
    )

    if not passed:
        raise RuntimeError(
            f"FAIL — {name}"
        )


# ============================================================
# OVERLAPPING DATES
# ============================================================

print("\n")
print("=" * 100)
print("DATE OVERLAP CHECK")
print("=" * 100)

train_dates = set(train["date"])
validation_dates = set(validation["date"])
test_dates = set(test["date"])

train_val_overlap = (
    train_dates & validation_dates
)

train_test_overlap = (
    train_dates & test_dates
)

val_test_overlap = (
    validation_dates & test_dates
)

print(
    "Train / Validation overlap:",
    len(train_val_overlap)
)

print(
    "Train / Test overlap:",
    len(train_test_overlap)
)

print(
    "Validation / Test overlap:",
    len(val_test_overlap)
)

if (
    train_val_overlap
    or train_test_overlap
    or val_test_overlap
):

    raise RuntimeError(
        "FAIL — date overlap detected."
    )

print("PASS — no date overlap.")


# ============================================================
# RESERVOIR COVERAGE
# ============================================================

print("\n")
print("=" * 100)
print("RESERVOIR COVERAGE")
print("=" * 100)

all_reservoirs = set(
    df["reservoir"]
)

for name, dataset in [
    ("train", train),
    ("validation", validation),
    ("test", test),
]:

    reservoirs = set(
        dataset["reservoir"]
    )

    missing = (
        all_reservoirs - reservoirs
    )

    print(
        f"{name:<12}: "
        f"{len(reservoirs)}/{len(all_reservoirs)} reservoirs"
    )

    if missing:

        print(
            "  Missing:",
            sorted(missing)
        )


# ============================================================
# SAMPLES BY RESERVOIR
# ============================================================

print("\n")
print("=" * 100)
print("SAMPLES BY RESERVOIR")
print("=" * 100)

coverage = pd.DataFrame({
    "train": train["reservoir"].value_counts(),
    "validation": validation["reservoir"].value_counts(),
    "test": test["reservoir"].value_counts(),
}).fillna(0).astype(int)

print(
    coverage.sort_index().to_string()
)


# ============================================================
# TARGET COLUMNS
# ============================================================

TARGET_COLUMNS = [
    "target_1d",
    "target_3d",
    "target_7d",
]


print("\n")
print("=" * 100)
print("TARGET DISTRIBUTION")
print("=" * 100)

for name, dataset in [
    ("TRAIN", train),
    ("VALIDATION", validation),
    ("TEST", test),
]:

    print(f"\n{name}")

    print(
        dataset[TARGET_COLUMNS]
        .describe()
        .to_string()
    )


# ============================================================
# TARGET MISSINGNESS
# ============================================================

print("\n")
print("=" * 100)
print("TARGET MISSINGNESS")
print("=" * 100)

for name, dataset in [
    ("train", train),
    ("validation", validation),
    ("test", test),
]:

    missing = (
        dataset[TARGET_COLUMNS]
        .isna()
        .sum()
        .sum()
    )

    print(
        f"{name:<12}: {missing}"
    )

    if missing > 0:

        raise RuntimeError(
            f"FAIL — missing targets in {name}."
        )


# ============================================================
# FEATURE / TARGET SEPARATION CHECK
# ============================================================

print("\n")
print("=" * 100)
print("LEAKAGE CHECK")
print("=" * 100)

metadata_columns = [
    "date",
    "reservoir",
]

feature_columns = [
    col
    for col in df.columns
    if col not in TARGET_COLUMNS
    and col not in metadata_columns
]

leakage_columns = []

for column in feature_columns:

    lower = column.lower()

    if (
        "target_" in lower
        or "future" in lower
    ):
        leakage_columns.append(
            column
        )


if leakage_columns:

    print(
        "FAIL — possible leakage:"
    )

    for column in leakage_columns:
        print(" ", column)

    raise RuntimeError(
        "Feature leakage detected."
    )

print(
    "PASS — feature matrix contains "
    "no target/future columns."
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

train.to_csv(
    TRAIN_OUTPUT,
    index=False,
)

validation.to_csv(
    VAL_OUTPUT,
    index=False,
)

test.to_csv(
    TEST_OUTPUT,
    index=False,
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 100)
print("MILESTONE 10.8 COMPLETE")
print("=" * 100)

print(
    "\nSaved:"
)

print(
    "TRAIN      :",
    TRAIN_OUTPUT
)

print(
    "VALIDATION :",
    VAL_OUTPUT
)

print(
    "TEST       :",
    TEST_OUTPUT
)

print("\nDONE")
