from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# MILESTONE 11.1 — PERSISTENCE FORECASTING BASELINE
# ============================================================

print("=" * 100)
print("MILESTONE 11.1 — PERSISTENCE FORECASTING BASELINE")
print("=" * 100)


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

TRAIN_PATH = Path(
    "data/processed/splits/train.csv"
)

VALIDATION_PATH = Path(
    "data/processed/splits/validation.csv"
)

TEST_PATH = Path(
    "data/processed/splits/test.csv"
)

OUTPUT_DIR = Path(
    "data/processed/baselines"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

TARGETS = [
    "target_1d",
    "target_3d",
    "target_7d",
]

LATEST_INFLOW = "inflow_day_7"


# ------------------------------------------------------------
# Load datasets
# ------------------------------------------------------------

train = pd.read_csv(TRAIN_PATH)
validation = pd.read_csv(VALIDATION_PATH)
test = pd.read_csv(TEST_PATH)


print("\nInput datasets:")
print(f"TRAIN      : {train.shape}")
print(f"VALIDATION : {validation.shape}")
print(f"TEST       : {test.shape}")


# ------------------------------------------------------------
# Validate required columns
# ------------------------------------------------------------

required_columns = [
    "date",
    "reservoir",
    LATEST_INFLOW,
    *TARGETS,
]

for dataset_name, dataset in [
    ("TRAIN", train),
    ("VALIDATION", validation),
    ("TEST", test),
]:

    missing = [
        column
        for column in required_columns
        if column not in dataset.columns
    ]

    if missing:
        raise RuntimeError(
            f"{dataset_name} is missing required columns: {missing}"
        )


# ------------------------------------------------------------
# Date conversion
# ------------------------------------------------------------

for dataset in [train, validation, test]:

    dataset["date"] = pd.to_datetime(
        dataset["date"],
        errors="coerce",
    )

    if dataset["date"].isna().any():
        raise RuntimeError(
            "Invalid dates detected."
        )


# ------------------------------------------------------------
# Persistence prediction
# ------------------------------------------------------------

def evaluate_persistence(dataset, dataset_name):

    rows = []

    print("\n" + "=" * 100)
    print(f"{dataset_name} — PERSISTENCE BASELINE")
    print("=" * 100)

    for target in TARGETS:

        valid = dataset[
            dataset[LATEST_INFLOW].notna()
            & dataset[target].notna()
        ].copy()

        y_true = valid[target].astype(float)
        y_pred = valid[LATEST_INFLOW].astype(float)

        mae = mean_absolute_error(
            y_true,
            y_pred,
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        )

        r2 = r2_score(
            y_true,
            y_pred,
        )

        rows.append(
            {
                "dataset": dataset_name,
                "target": target,
                "samples": len(valid),
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
            }
        )

        print(
            f"{target:10s} | "
            f"samples={len(valid):5d} | "
            f"MAE={mae:10.4f} | "
            f"RMSE={rmse:10.4f} | "
            f"R2={r2:10.4f}"
        )

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# Evaluate
# ------------------------------------------------------------

train_results = evaluate_persistence(
    train,
    "TRAIN",
)

validation_results = evaluate_persistence(
    validation,
    "VALIDATION",
)

test_results = evaluate_persistence(
    test,
    "TEST",
)


results = pd.concat(
    [
        train_results,
        validation_results,
        test_results,
    ],
    ignore_index=True,
)


# ------------------------------------------------------------
# Save overall results
# ------------------------------------------------------------

results_path = (
    OUTPUT_DIR
    / "persistence_baseline_results.csv"
)

results.to_csv(
    results_path,
    index=False,
)


# ------------------------------------------------------------
# Save test predictions
# ------------------------------------------------------------

test_predictions = test[
    [
        "date",
        "reservoir",
        LATEST_INFLOW,
        *TARGETS,
    ]
].copy()

for target in TARGETS:

    prediction_column = (
        target.replace(
            "target",
            "prediction",
        )
    )

    test_predictions[prediction_column] = (
        test_predictions[LATEST_INFLOW]
    )


predictions_path = (
    OUTPUT_DIR
    / "persistence_test_predictions.csv"
)

test_predictions.to_csv(
    predictions_path,
    index=False,
)


# ------------------------------------------------------------
# Test summary
# ------------------------------------------------------------

print("\n" + "=" * 100)
print("TEST SET SUMMARY")
print("=" * 100)

print(
    results[
        results["dataset"] == "TEST"
    ].to_string(index=False)
)


# ------------------------------------------------------------
# Integrity checks
# ------------------------------------------------------------

print("\n" + "=" * 100)
print("INTEGRITY CHECKS")
print("=" * 100)

print(
    "Train rows preserved:",
    len(train),
)

print(
    "Validation rows preserved:",
    len(validation),
)

print(
    "Test rows preserved:",
    len(test),
)

if len(test_predictions) != len(test):
    raise RuntimeError(
        "Test prediction row count mismatch."
    )

print(
    "Prediction row count:",
    len(test_predictions),
)

print(
    "\nPrediction rule:"
)

print(
    "prediction_1d = inflow_day_7"
)

print(
    "prediction_3d = inflow_day_7"
)

print(
    "prediction_7d = inflow_day_7"
)


# ------------------------------------------------------------
# Final
# ------------------------------------------------------------

print("\n" + "=" * 100)
print("MILESTONE 11.1 COMPLETE")
print("=" * 100)

print("\nSaved:")
print(
    "RESULTS     :",
    results_path,
)

print(
    "PREDICTIONS :",
    predictions_path,
)

print("\nDONE")
