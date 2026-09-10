from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# MILESTONE 11.5 — LSTM SEQUENCE PREPARATION
# ============================================================

print("=" * 100)
print("MILESTONE 11.5 — LSTM SEQUENCE PREPARATION")
print("=" * 100)


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

BASE = Path("data/processed")

INPUT_DIR = BASE / "scaled"
OUTPUT_DIR = BASE / "lstm"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


TRAIN_FILE = INPUT_DIR / "train_scaled.csv"
VAL_FILE = INPUT_DIR / "validation_scaled.csv"
TEST_FILE = INPUT_DIR / "test_scaled.csv"


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

HISTORY_DAYS = 7

DYNAMIC_FEATURES = [
    "inflow",
    "water_level",
    "live_storage",
    "rainfall",
    "total_outflow",
]

STATIC_FEATURES = [
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

TARGETS = [
    "target_1d",
    "target_3d",
    "target_7d",
]


# ============================================================
# Helper functions
# ============================================================

def load_dataset(path):
    print(f"\nLoading: {path}")

    df = pd.read_csv(path)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce",
        )

    print("Shape:", df.shape)

    return df


def discover_sequence_columns(df):
    """
    Locate the 7 daily columns for every dynamic feature.
    """

    sequence_columns = {}

    for feature in DYNAMIC_FEATURES:

        columns = [
            f"{feature}_day_{day}"
            for day in range(1, HISTORY_DAYS + 1)
        ]

        missing = [
            column
            for column in columns
            if column not in df.columns
        ]

        if missing:
            raise RuntimeError(
                f"Missing sequence columns for {feature}: "
                f"{missing}"
            )

        sequence_columns[feature] = columns

    return sequence_columns


def build_dynamic_tensor(df, sequence_columns):
    """
    Convert flattened columns into:

        (samples, 7 days, 5 dynamic features)

    Day ordering is preserved:

        day_1
        day_2
        ...
        day_7
    """

    n_samples = len(df)
    n_features = len(DYNAMIC_FEATURES)

    tensor = np.zeros(
        (
            n_samples,
            HISTORY_DAYS,
            n_features,
        ),
        dtype=np.float32,
    )

    for feature_index, feature in enumerate(DYNAMIC_FEATURES):

        columns = sequence_columns[feature]

        values = df[columns].to_numpy(
            dtype=np.float32
        )

        tensor[:, :, feature_index] = values

    return tensor


def build_static_matrix(df):
    """
    Static reservoir features remain outside the temporal
    sequence and are supplied separately to the model.
    """

    values = df[
        STATIC_FEATURES
    ].to_numpy(
        dtype=np.float32
    )

    return values


def build_static_availability_matrix(df):
    """
    Availability flags tell the model whether a static
    threshold/value was originally available.
    """

    values = df[
        STATIC_AVAILABILITY_FEATURES
    ].to_numpy(
        dtype=np.float32
    )

    return values


def build_targets(df):

    values = df[
        TARGETS
    ].to_numpy(
        dtype=np.float32
    )

    return values


def check_finite(name, array):

    if not np.isfinite(array).all():

        bad = np.sum(
            ~np.isfinite(array)
        )

        raise RuntimeError(
            f"{name} contains {bad} "
            f"NaN or infinite values."
        )


def verify_dataset(
    name,
    df,
    dynamic,
    static,
    availability,
    targets,
):

    print("\n" + "-" * 90)
    print(f"{name} VERIFICATION")
    print("-" * 90)

    print("Rows:", len(df))

    print(
        "Dynamic tensor:",
        dynamic.shape,
    )

    print(
        "Static matrix:",
        static.shape,
    )

    print(
        "Availability matrix:",
        availability.shape,
    )

    print(
        "Targets:",
        targets.shape,
    )

    expected_dynamic = (
        len(df),
        HISTORY_DAYS,
        len(DYNAMIC_FEATURES),
    )

    expected_static = (
        len(df),
        len(STATIC_FEATURES),
    )

    expected_availability = (
        len(df),
        len(STATIC_AVAILABILITY_FEATURES),
    )

    expected_targets = (
        len(df),
        len(TARGETS),
    )

    if dynamic.shape != expected_dynamic:
        raise RuntimeError(
            f"{name} dynamic shape mismatch: "
            f"{dynamic.shape} != {expected_dynamic}"
        )

    if static.shape != expected_static:
        raise RuntimeError(
            f"{name} static shape mismatch: "
            f"{static.shape} != {expected_static}"
        )

    if availability.shape != expected_availability:
        raise RuntimeError(
            f"{name} availability shape mismatch: "
            f"{availability.shape} != "
            f"{expected_availability}"
        )

    if targets.shape != expected_targets:
        raise RuntimeError(
            f"{name} target shape mismatch: "
            f"{targets.shape} != {expected_targets}"
        )

    check_finite(
        f"{name} dynamic tensor",
        dynamic,
    )

    check_finite(
        f"{name} static matrix",
        static,
    )

    check_finite(
        f"{name} availability matrix",
        availability,
    )

    check_finite(
        f"{name} targets",
        targets,
    )

    print("Finite-value check: PASS")


def save_numpy(name, array, filename):

    path = OUTPUT_DIR / filename

    np.save(path, array)

    print(
        f"Saved {name}: {path} "
        f"{array.shape}"
    )


# ============================================================
# Load datasets
# ============================================================

train = load_dataset(TRAIN_FILE)
validation = load_dataset(VAL_FILE)
test = load_dataset(TEST_FILE)


# ============================================================
# Dataset summary
# ============================================================

print("\n" + "=" * 100)
print("DATASET SUMMARY")
print("=" * 100)

print(
    "TRAIN:",
    train.shape,
)

print(
    "VALIDATION:",
    validation.shape,
)

print(
    "TEST:",
    test.shape,
)


# ============================================================
# Required column checks
# ============================================================

required_columns = (
    ["date", "reservoir"]
    + STATIC_FEATURES
    + STATIC_AVAILABILITY_FEATURES
    + TARGETS
)

for dataset_name, df in [
    ("TRAIN", train),
    ("VALIDATION", validation),
    ("TEST", test),
]:

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise RuntimeError(
            f"{dataset_name} missing required columns: "
            f"{missing}"
        )


# ============================================================
# Discover temporal columns
# ============================================================

sequence_columns = discover_sequence_columns(train)


print("\n" + "=" * 100)
print("SEQUENCE FEATURE DISCOVERY")
print("=" * 100)

for feature in DYNAMIC_FEATURES:

    print(
        f"{feature:25s}: "
        f"{len(sequence_columns[feature])} columns"
    )

    print(
        "  ",
        sequence_columns[feature]
    )


print("\nDynamic feature count:", len(DYNAMIC_FEATURES))
print("History length:", HISTORY_DAYS)


# ============================================================
# Build tensors
# ============================================================

print("\n" + "=" * 100)
print("BUILDING LSTM TENSORS")
print("=" * 100)


train_dynamic = build_dynamic_tensor(
    train,
    sequence_columns,
)

val_dynamic = build_dynamic_tensor(
    validation,
    sequence_columns,
)

test_dynamic = build_dynamic_tensor(
    test,
    sequence_columns,
)


train_static = build_static_matrix(train)

val_static = build_static_matrix(validation)

test_static = build_static_matrix(test)


train_availability = (
    build_static_availability_matrix(train)
)

val_availability = (
    build_static_availability_matrix(validation)
)

test_availability = (
    build_static_availability_matrix(test)
)


train_targets = build_targets(train)

val_targets = build_targets(validation)

test_targets = build_targets(test)


# ============================================================
# Verify tensors
# ============================================================

verify_dataset(
    "TRAIN",
    train,
    train_dynamic,
    train_static,
    train_availability,
    train_targets,
)

verify_dataset(
    "VALIDATION",
    validation,
    val_dynamic,
    val_static,
    val_availability,
    val_targets,
)

verify_dataset(
    "TEST",
    test,
    test_dynamic,
    test_static,
    test_availability,
    test_targets,
)


# ============================================================
# Date integrity
# ============================================================

print("\n" + "=" * 100)
print("DATE INTEGRITY")
print("=" * 100)

for name, df in [
    ("TRAIN", train),
    ("VALIDATION", validation),
    ("TEST", test),
]:

    dates = df["date"]

    print(
        f"{name:12s}: "
        f"{dates.min()} -> {dates.max()}"
    )

    if dates.isna().any():

        raise RuntimeError(
            f"{name} contains invalid dates."
        )


train_dates = set(
    train["date"].dt.normalize()
)

val_dates = set(
    validation["date"].dt.normalize()
)

test_dates = set(
    test["date"].dt.normalize()
)


train_val_overlap = (
    train_dates & val_dates
)

train_test_overlap = (
    train_dates & test_dates
)

val_test_overlap = (
    val_dates & test_dates
)


print(
    "Train / Validation overlap:",
    len(train_val_overlap),
)

print(
    "Train / Test overlap:",
    len(train_test_overlap),
)

print(
    "Validation / Test overlap:",
    len(val_test_overlap),
)


if (
    train_val_overlap
    or train_test_overlap
    or val_test_overlap
):

    raise RuntimeError(
        "Temporal overlap detected."
    )

print(
    "Temporal separation: PASS"
)


# ============================================================
# Reservoir integrity
# ============================================================

print("\n" + "=" * 100)
print("RESERVOIR INTEGRITY")
print("=" * 100)

for name, df in [
    ("TRAIN", train),
    ("VALIDATION", validation),
    ("TEST", test),
]:

    reservoirs = sorted(
        df["reservoir"]
        .dropna()
        .unique()
        .tolist()
    )

    print(
        f"{name:12s}: "
        f"{len(reservoirs)} reservoirs"
    )


train_reservoirs = set(
    train["reservoir"].dropna()
)

val_reservoirs = set(
    validation["reservoir"].dropna()
)

test_reservoirs = set(
    test["reservoir"].dropna()
)


if not (
    train_reservoirs
    == val_reservoirs
    == test_reservoirs
):

    raise RuntimeError(
        "Reservoir coverage differs "
        "between train/validation/test."
    )

print(
    "Reservoir coverage: PASS"
)


# ============================================================
# Verify day ordering
# ============================================================

print("\n" + "=" * 100)
print("DAY ORDER VERIFICATION")
print("=" * 100)

print(
    "Tensor axis 1 represents:",
    "day_1 -> day_7"
)

print(
    "Tensor axis 2 represents:",
    DYNAMIC_FEATURES,
)

# Verify flattened → tensor mapping on first sample.

first_row = train.iloc[0]

for feature_index, feature in enumerate(
    DYNAMIC_FEATURES
):

    original_values = first_row[
        sequence_columns[feature]
    ].to_numpy(
        dtype=np.float32
    )

    tensor_values = train_dynamic[
        0,
        :,
        feature_index,
    ]

    if not np.allclose(
        original_values,
        tensor_values,
        equal_nan=False,
    ):

        raise RuntimeError(
            f"Day ordering mismatch for "
            f"{feature}"
        )

print(
    "Flattened-to-tensor mapping: PASS"
)


# ============================================================
# Print final shapes
# ============================================================

print("\n" + "=" * 100)
print("FINAL LSTM DATASET SHAPES")
print("=" * 100)

print(
    "TRAIN dynamic:",
    train_dynamic.shape,
)

print(
    "TRAIN static:",
    train_static.shape,
)

print(
    "TRAIN static availability:",
    train_availability.shape,
)

print(
    "TRAIN targets:",
    train_targets.shape,
)

print()

print(
    "VALIDATION dynamic:",
    val_dynamic.shape,
)

print(
    "VALIDATION static:",
    val_static.shape,
)

print(
    "VALIDATION static availability:",
    val_availability.shape,
)

print(
    "VALIDATION targets:",
    val_targets.shape,
)

print()

print(
    "TEST dynamic:",
    test_dynamic.shape,
)

print(
    "TEST static:",
    test_static.shape,
)

print(
    "TEST static availability:",
    test_availability.shape,
)

print(
    "TEST targets:",
    test_targets.shape,
)


# ============================================================
# Save tensors
# ============================================================

print("\n" + "=" * 100)
print("SAVING LSTM DATA")
print("=" * 100)

save_numpy(
    "TRAIN dynamic",
    train_dynamic,
    "X_train_dynamic.npy",
)

save_numpy(
    "VALIDATION dynamic",
    val_dynamic,
    "X_validation_dynamic.npy",
)

save_numpy(
    "TEST dynamic",
    test_dynamic,
    "X_test_dynamic.npy",
)


save_numpy(
    "TRAIN static",
    train_static,
    "X_train_static.npy",
)

save_numpy(
    "VALIDATION static",
    val_static,
    "X_validation_static.npy",
)

save_numpy(
    "TEST static",
    test_static,
    "X_test_static.npy",
)


save_numpy(
    "TRAIN static availability",
    train_availability,
    "X_train_static_availability.npy",
)

save_numpy(
    "VALIDATION static availability",
    val_availability,
    "X_validation_static_availability.npy",
)

save_numpy(
    "TEST static availability",
    test_availability,
    "X_test_static_availability.npy",
)


save_numpy(
    "TRAIN targets",
    train_targets,
    "y_train.npy",
)

save_numpy(
    "VALIDATION targets",
    val_targets,
    "y_validation.npy",
)

save_numpy(
    "TEST targets",
    test_targets,
    "y_test.npy",
)


# ============================================================
# Save metadata
# ============================================================

metadata = {
    "history_days": HISTORY_DAYS,
    "dynamic_features": DYNAMIC_FEATURES,
    "static_features": STATIC_FEATURES,
    "static_availability_features":
        STATIC_AVAILABILITY_FEATURES,
    "targets": TARGETS,
    "train_samples": len(train),
    "validation_samples": len(validation),
    "test_samples": len(test),
    "dynamic_shape": list(
        train_dynamic.shape
    ),
    "static_shape": list(
        train_static.shape
    ),
    "static_availability_shape": list(
        train_availability.shape
    ),
    "target_shape": list(
        train_targets.shape
    ),
}

metadata_path = (
    OUTPUT_DIR / "lstm_metadata.json"
)

import json

with open(
    metadata_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        metadata,
        f,
        indent=2,
    )

print(
    "Saved metadata:",
    metadata_path,
)


# ============================================================
# Final integrity checks
# ============================================================

print("\n" + "=" * 100)
print("FINAL INTEGRITY CHECKS")
print("=" * 100)


if len(train_dynamic) != len(train):
    raise RuntimeError(
        "TRAIN sample count mismatch."
    )

if len(val_dynamic) != len(validation):
    raise RuntimeError(
        "VALIDATION sample count mismatch."
    )

if len(test_dynamic) != len(test):
    raise RuntimeError(
        "TEST sample count mismatch."
    )


if (
    train_targets.shape[1]
    != len(TARGETS)
):

    raise RuntimeError(
        "TRAIN target count mismatch."
    )


if (
    val_targets.shape[1]
    != len(TARGETS)
):

    raise RuntimeError(
        "VALIDATION target count mismatch."
    )


if (
    test_targets.shape[1]
    != len(TARGETS)
):

    raise RuntimeError(
        "TEST target count mismatch."
    )


print(
    "Sample counts: PASS"
)

print(
    "Target dimensions: PASS"
)

print(
    "No NaN/Inf: PASS"
)

print(
    "Temporal separation: PASS"
)

print(
    "Reservoir coverage: PASS"
)

print(
    "Day ordering: PASS"
)


print("\n" + "=" * 100)
print("MILESTONE 11.5 COMPLETE")
print("=" * 100)

print("\nSaved LSTM data to:")
print(OUTPUT_DIR)

print("\nDONE")
