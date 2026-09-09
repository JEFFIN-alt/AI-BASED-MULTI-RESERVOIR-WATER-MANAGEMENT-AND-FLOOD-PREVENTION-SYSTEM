from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# MILESTONE 11.2 — 7-DAY ROLLING MEAN FORECASTING BASELINE
# ============================================================

print("=" * 100)
print("MILESTONE 11.2 — 7-DAY ROLLING MEAN FORECASTING BASELINE")
print("=" * 100)


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

SPLIT_DIR = Path("data/processed/splits")
OUTPUT_DIR = Path("data/processed/baselines")

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

INFLOW_COLUMNS = [
    f"inflow_day_{day}"
    for day in range(1, 8)
]


# ------------------------------------------------------------
# Load datasets
# ------------------------------------------------------------

train = pd.read_csv(
    SPLIT_DIR / "train.csv"
)

validation = pd.read_csv(
    SPLIT_DIR / "validation.csv"
)

test = pd.read_csv(
    SPLIT_DIR / "test.csv"
)


print("\nInput datasets:")
print("TRAIN      :", train.shape)
print("VALIDATION :", validation.shape)
print("TEST       :", test.shape)


# ------------------------------------------------------------
# Validate columns
# ------------------------------------------------------------

required_columns = [
    "date",
    "reservoir",
    *INFLOW_COLUMNS,
    *TARGETS,
]

for name, dataset in [
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
            f"{name} missing required columns: {missing}"
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
# Build rolling-mean prediction
# ------------------------------------------------------------

def evaluate(dataset, dataset_name):

    # Require complete 7-day inflow history
    valid = dataset.dropna(
        subset=INFLOW_COLUMNS + TARGETS
    ).copy()

    # Mean of the seven historical inflows
    valid["rolling_mean_prediction"] = (
        valid[INFLOW_COLUMNS]
        .astype(float)
        .mean(axis=1)
    )

    rows = []

    print("\n" + "=" * 100)
    print(f"{dataset_name} — 7-DAY ROLLING MEAN")
    print("=" * 100)

    for target in TARGETS:

        y_true = valid[target].astype(float)
        y_pred = valid["rolling_mean_prediction"].astype(float)

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

    return (
        pd.DataFrame(rows),
        valid,
    )


# ------------------------------------------------------------
# Evaluate all splits
# ------------------------------------------------------------

train_results, train_predictions = evaluate(
    train,
    "TRAIN",
)

validation_results, validation_predictions = evaluate(
    validation,
    "VALIDATION",
)

test_results, test_predictions = evaluate(
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
# Save results
# ------------------------------------------------------------

results_path = (
    OUTPUT_DIR
    / "rolling_mean_baseline_results.csv"
)

results.to_csv(
    results_path,
    index=False,
)


# ------------------------------------------------------------
# Save test predictions
# ------------------------------------------------------------

prediction_columns = [
    "date",
    "reservoir",
    *INFLOW_COLUMNS,
    *TARGETS,
    "rolling_mean_prediction",
]

test_output = test_predictions[
    prediction_columns
].copy()

for target in TARGETS:

    prediction_name = (
        target.replace(
            "target",
            "prediction",
        )
        + "_rolling_mean"
    )

    test_output[prediction_name] = (
        test_output["rolling_mean_prediction"]
    )


predictions_path = (
    OUTPUT_DIR
    / "rolling_mean_test_predictions.csv"
)

test_output.to_csv(
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
    "Train rows:",
    len(train),
)

print(
    "Validation rows:",
    len(validation),
)

print(
    "Test rows:",
    len(test),
)

print(
    "Test prediction rows:",
    len(test_output),
)

if len(test_output) != len(test):
    raise RuntimeError(
        "Prediction row count does not match test dataset."
    )

print(
    "\nPrediction rule:"
)

print(
    "prediction = mean(inflow_day_1 ... inflow_day_7)"
)


# ------------------------------------------------------------
# Final
# ------------------------------------------------------------

print("\n" + "=" * 100)
print("MILESTONE 11.2 COMPLETE")
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
