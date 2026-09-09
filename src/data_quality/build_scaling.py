from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ============================================================
# MILESTONE 10.9 — LEAKAGE-SAFE FEATURE SCALING
# ============================================================

INPUT_DIR = Path("data/processed/splits")
OUTPUT_DIR = Path("data/processed/scaled")

TRAIN_INPUT = INPUT_DIR / "train.csv"
VALID_INPUT = INPUT_DIR / "validation.csv"
TEST_INPUT = INPUT_DIR / "test.csv"

TRAIN_OUTPUT = OUTPUT_DIR / "train_scaled.csv"
VALID_OUTPUT = OUTPUT_DIR / "validation_scaled.csv"
TEST_OUTPUT = OUTPUT_DIR / "test_scaled.csv"

FEATURE_SCALER_OUTPUT = OUTPUT_DIR / "feature_scaler.pkl"
TARGET_SCALER_OUTPUT = OUTPUT_DIR / "target_scaler.pkl"


print("=" * 100)
print("MILESTONE 10.9 — LEAKAGE-SAFE FEATURE SCALING")
print("=" * 100)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_COLUMNS = [
    "target_1d",
    "target_3d",
    "target_7d",
]


# These are the actual forecasting inputs created by
# Milestone 10.7.
#
# Each dynamic feature has 7 historical values.
# Static features include availability indicators.
# ============================================================

DYNAMIC_BASE_FEATURES = [
    "inflow",
    "water_level",
    "live_storage",
    "rainfall",
    "total_outflow",
]

STATIC_BASE_FEATURES = [
    "static_latitude",
    "static_longitude",
    "static_FRL",
    "static_MWL",
    "static_blue_level",
    "static_orange_level",
    "static_red_level",
    "static_rule_level",
]

STATIC_AVAILABILITY_FEATURES = [
    "static_latitude_available",
    "static_longitude_available",
    "static_FRL_available",
    "static_MWL_available",
    "static_blue_level_available",
    "static_orange_level_available",
    "static_red_level_available",
    "static_rule_level_available",
]


# ============================================================
# LOAD DATA
# ============================================================

train = pd.read_csv(TRAIN_INPUT)
validation = pd.read_csv(VALID_INPUT)
test = pd.read_csv(TEST_INPUT)


for df in [train, validation, test]:
    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )


print("\nInput datasets:")
print("TRAIN      :", train.shape)
print("VALIDATION :", validation.shape)
print("TEST       :", test.shape)


# ============================================================
# IDENTIFY FEATURE COLUMNS
# ============================================================

all_columns = train.columns.tolist()

dynamic_columns = []

for base in DYNAMIC_BASE_FEATURES:
    matching = [
        col
        for col in all_columns
        if col.startswith(base + "_")
    ]

    matching = sorted(
        matching,
        key=lambda x: int(x.rsplit("_", 1)[-1]),
    )

    dynamic_columns.extend(matching)


static_columns = []

for base in STATIC_BASE_FEATURES:
    if base in all_columns:
        static_columns.append(base)


availability_columns = []

for base in STATIC_AVAILABILITY_FEATURES:
    if base in all_columns:
        availability_columns.append(base)


FEATURE_COLUMNS = (
    dynamic_columns
    + static_columns
    + availability_columns
)


# ============================================================
# VALIDATE FEATURE DISCOVERY
# ============================================================

print("\n" + "=" * 100)
print("FEATURE DISCOVERY")
print("=" * 100)

print("\nDynamic feature columns:")

for base in DYNAMIC_BASE_FEATURES:
    cols = [
        c for c in dynamic_columns
        if c.startswith(base + "_")
    ]

    print(
        f"{base:25s}: {len(cols)}"
    )

print("\nStatic feature columns:")

for col in static_columns:
    print(
        f"{col:35s}: YES"
    )

print("\nStatic availability columns:")

for col in availability_columns:
    print(
        f"{col:35s}: YES"
    )


if not dynamic_columns:
    raise RuntimeError(
        "No dynamic forecasting feature columns found."
    )


if not static_columns:
    raise RuntimeError(
        "No static forecasting feature columns found."
    )


missing_from_validation = [
    c for c in FEATURE_COLUMNS
    if c not in validation.columns
]

missing_from_test = [
    c for c in FEATURE_COLUMNS
    if c not in test.columns
]

if missing_from_validation:
    raise RuntimeError(
        "Validation is missing feature columns:\n"
        + "\n".join(missing_from_validation)
    )

if missing_from_test:
    raise RuntimeError(
        "Test is missing feature columns:\n"
        + "\n".join(missing_from_test)
    )


# ============================================================
# VERIFY TARGET COLUMNS
# ============================================================

for target in TARGET_COLUMNS:

    if target not in train.columns:
        raise RuntimeError(
            f"Missing target in TRAIN: {target}"
        )

    if target not in validation.columns:
        raise RuntimeError(
            f"Missing target in VALIDATION: {target}"
        )

    if target not in test.columns:
        raise RuntimeError(
            f"Missing target in TEST: {target}"
        )


# ============================================================
# CONVERT NUMERIC FEATURES
# ============================================================

for df in [train, validation, test]:

    for col in FEATURE_COLUMNS + TARGET_COLUMNS:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )


# ============================================================
# MISSING VALUE POLICY
# ============================================================
#
# IMPORTANT:
#
# We do NOT use future data to fill missing values.
#
# For each feature:
#
#     median = calculated ONLY from TRAIN
#
# The same train median is then used for:
#
#     TRAIN
#     VALIDATION
#     TEST
#
# Availability columns are special:
#
#     missing availability = 0
#
# ============================================================

print("\n" + "=" * 100)
print("MISSING VALUE POLICY")
print("=" * 100)


continuous_features = (
    dynamic_columns
    + static_columns
)


train_medians = {}

for col in continuous_features:

    median = train[col].median()

    if pd.isna(median):

        raise RuntimeError(
            f"Cannot calculate training median for: {col}"
        )

    train_medians[col] = float(median)


# ============================================================
# APPLY TRAIN-ONLY IMPUTATION
# ============================================================

for col in continuous_features:

    median = train_medians[col]

    train[col] = train[col].fillna(median)

    validation[col] = validation[col].fillna(median)

    test[col] = test[col].fillna(median)


# Availability indicators:
#
# If missing, treat as unavailable.
# ============================================================

for col in availability_columns:

    train[col] = train[col].fillna(0)

    validation[col] = validation[col].fillna(0)

    test[col] = test[col].fillna(0)


# ============================================================
# FIT FEATURE SCALER — TRAIN ONLY
# ============================================================

print("\n" + "=" * 100)
print("FITTING FEATURE SCALER")
print("=" * 100)

feature_scaler = StandardScaler()

feature_scaler.fit(
    train[FEATURE_COLUMNS]
)


# ============================================================
# TRANSFORM FEATURES
# ============================================================

train_features_scaled = feature_scaler.transform(
    train[FEATURE_COLUMNS]
)

validation_features_scaled = feature_scaler.transform(
    validation[FEATURE_COLUMNS]
)

test_features_scaled = feature_scaler.transform(
    test[FEATURE_COLUMNS]
)


# ============================================================
# FIT TARGET SCALER — TRAIN ONLY
# ============================================================

print("\n" + "=" * 100)
print("FITTING TARGET SCALER")
print("=" * 100)

target_scaler = StandardScaler()

target_scaler.fit(
    train[TARGET_COLUMNS]
)


# ============================================================
# TRANSFORM TARGETS
# ============================================================

train_targets_scaled = target_scaler.transform(
    train[TARGET_COLUMNS]
)

validation_targets_scaled = target_scaler.transform(
    validation[TARGET_COLUMNS]
)

test_targets_scaled = target_scaler.transform(
    test[TARGET_COLUMNS]
)


# ============================================================
# BUILD OUTPUT DATAFRAMES
# ============================================================

def build_scaled_dataframe(
    original_df,
    scaled_features,
    scaled_targets,
):
    result = original_df.copy()

    # Make scaled feature columns floating-point
    for i, col in enumerate(FEATURE_COLUMNS):
        result[col] = scaled_features[:, i].astype(np.float64)

    # Make scaled target columns floating-point
    for i, col in enumerate(TARGET_COLUMNS):
        result[col] = scaled_targets[:, i].astype(np.float64)

    return result


train_scaled = build_scaled_dataframe(
    train,
    train_features_scaled,
    train_targets_scaled,
)

validation_scaled = build_scaled_dataframe(
    validation,
    validation_features_scaled,
    validation_targets_scaled,
)

test_scaled = build_scaled_dataframe(
    test,
    test_features_scaled,
    test_targets_scaled,
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# SAVE DATASETS
# ============================================================

train_scaled.to_csv(
    TRAIN_OUTPUT,
    index=False,
)

validation_scaled.to_csv(
    VALID_OUTPUT,
    index=False,
)

test_scaled.to_csv(
    TEST_OUTPUT,
    index=False,
)


# ============================================================
# SAVE PREPROCESSORS
# ============================================================

preprocessor_metadata = {
    "feature_columns": FEATURE_COLUMNS,
    "continuous_features": continuous_features,
    "availability_columns": availability_columns,
    "target_columns": TARGET_COLUMNS,
    "train_medians": train_medians,
    "feature_scaler": feature_scaler,
}


with open(
    FEATURE_SCALER_OUTPUT,
    "wb",
) as f:

    pickle.dump(
        preprocessor_metadata,
        f,
    )


with open(
    TARGET_SCALER_OUTPUT,
    "wb",
) as f:

    pickle.dump(
        target_scaler,
        f,
    )


# ============================================================
# AUDIT
# ============================================================

print("\n" + "=" * 100)
print("MILESTONE 10.9 — SCALING AUDIT")
print("=" * 100)


print("\nOutput shapes:")

print(
    "TRAIN      :",
    train_scaled.shape,
)

print(
    "VALIDATION :",
    validation_scaled.shape,
)

print(
    "TEST       :",
    test_scaled.shape,
)


# ------------------------------------------------------------
# NaN check
# ------------------------------------------------------------

print("\nNaN check:")

train_nan = train_scaled[
    FEATURE_COLUMNS + TARGET_COLUMNS
].isna().sum().sum()

validation_nan = validation_scaled[
    FEATURE_COLUMNS + TARGET_COLUMNS
].isna().sum().sum()

test_nan = test_scaled[
    FEATURE_COLUMNS + TARGET_COLUMNS
].isna().sum().sum()


print(
    "TRAIN      :",
    train_nan,
)

print(
    "VALIDATION :",
    validation_nan,
)

print(
    "TEST       :",
    test_nan,
)


if train_nan != 0:
    raise RuntimeError(
        "NaN values remain in TRAIN."
    )

if validation_nan != 0:
    raise RuntimeError(
        "NaN values remain in VALIDATION."
    )

if test_nan != 0:
    raise RuntimeError(
        "NaN values remain in TEST."
    )


# ------------------------------------------------------------
# Date preservation
# ------------------------------------------------------------

print("\nDate preservation:")

print(
    "TRAIN      :",
    train_scaled["date"].min(),
    "->",
    train_scaled["date"].max(),
)

print(
    "VALIDATION :",
    validation_scaled["date"].min(),
    "->",
    validation_scaled["date"].max(),
)

print(
    "TEST       :",
    test_scaled["date"].min(),
    "->",
    test_scaled["date"].max(),
)


# ------------------------------------------------------------
# Reservoir preservation
# ------------------------------------------------------------

print("\nReservoir coverage:")

print(
    "TRAIN      :",
    train_scaled["reservoir"].nunique(),
)

print(
    "VALIDATION :",
    validation_scaled["reservoir"].nunique(),
)

print(
    "TEST       :",
    test_scaled["reservoir"].nunique(),
)


# ------------------------------------------------------------
# Train scaler statistics
# ------------------------------------------------------------

print("\nTraining scaled feature statistics:")

train_scaled_features = train_scaled[
    FEATURE_COLUMNS
]

print(
    train_scaled_features.describe()
    .loc[["mean", "std"]]
    .round(4)
    .to_string()
)


# ------------------------------------------------------------
# Verify target scaling
# ------------------------------------------------------------

print("\nTraining scaled target statistics:")

print(
    train_scaled[TARGET_COLUMNS]
    .describe()
    .loc[["mean", "std"]]
    .round(4)
    .to_string()
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 100)
print("MILESTONE 10.9 COMPLETE")
print("=" * 100)

print("\nSaved:")

print(
    "TRAIN      :",
    TRAIN_OUTPUT,
)

print(
    "VALIDATION :",
    VALID_OUTPUT,
)

print(
    "TEST       :",
    TEST_OUTPUT,
)

print(
    "FEATURE PREPROCESSOR :",
    FEATURE_SCALER_OUTPUT,
)

print(
    "TARGET SCALER        :",
    TARGET_SCALER_OUTPUT,
)

print("\nDONE")
