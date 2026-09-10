from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# MILESTONE 11.3 — TREE-BASED FORECASTING BASELINE
# ============================================================

print("=" * 100)
print("MILESTONE 11.3 — TREE-BASED FORECASTING BASELINE")
print("=" * 100)


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

TRAIN_PATH = Path(
    "data/processed/scaled/train_scaled.csv"
)

VALIDATION_PATH = Path(
    "data/processed/scaled/validation_scaled.csv"
)

TEST_PATH = Path(
    "data/processed/scaled/test_scaled.csv"
)

OUTPUT_DIR = Path(
    "data/processed/baselines/tree"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


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
# Target columns
# ------------------------------------------------------------

TARGET_COLUMNS = [
    "target_1d",
    "target_3d",
    "target_7d",
]


# ------------------------------------------------------------
# Columns that must NOT be used as features
# ------------------------------------------------------------

EXCLUDE_COLUMNS = {
    "date",
    "reservoir",
    "target_1d",
    "target_3d",
    "target_7d",
}


# ------------------------------------------------------------
# Feature discovery
# ------------------------------------------------------------

FEATURE_COLUMNS = [
    column
    for column in train.columns
    if column not in EXCLUDE_COLUMNS
]


print("\nFeature discovery:")
print("Feature count:", len(FEATURE_COLUMNS))

print("\nExcluded columns:")
for column in sorted(EXCLUDE_COLUMNS):
    if column in train.columns:
        print(" ", column)


# ------------------------------------------------------------
# Verify feature consistency
# ------------------------------------------------------------

missing_in_validation = [
    column
    for column in FEATURE_COLUMNS
    if column not in validation.columns
]

missing_in_test = [
    column
    for column in FEATURE_COLUMNS
    if column not in test.columns
]

if missing_in_validation:
    raise RuntimeError(
        "Validation dataset is missing features: "
        + str(missing_in_validation)
    )

if missing_in_test:
    raise RuntimeError(
        "Test dataset is missing features: "
        + str(missing_in_test)
    )


# ------------------------------------------------------------
# Convert features to numeric
# ------------------------------------------------------------

for dataframe_name, dataframe in [
    ("train", train),
    ("validation", validation),
    ("test", test),
]:

    dataframe[FEATURE_COLUMNS] = (
        dataframe[FEATURE_COLUMNS]
        .apply(pd.to_numeric, errors="coerce")
    )


# ------------------------------------------------------------
# Missing-value check
# ------------------------------------------------------------

print("\nMissing values in feature matrix:")

print(
    "TRAIN:",
    int(train[FEATURE_COLUMNS].isna().sum().sum())
)

print(
    "VALIDATION:",
    int(validation[FEATURE_COLUMNS].isna().sum().sum())
)

print(
    "TEST:",
    int(test[FEATURE_COLUMNS].isna().sum().sum())
)


# ------------------------------------------------------------
# HistGradientBoosting can handle NaN values.
# ------------------------------------------------------------

X_train = train[FEATURE_COLUMNS]
X_validation = validation[FEATURE_COLUMNS]
X_test = test[FEATURE_COLUMNS]


# ------------------------------------------------------------
# Model configuration
# ------------------------------------------------------------

MODEL_PARAMETERS = {
    "max_iter": 300,
    "learning_rate": 0.05,
    "max_leaf_nodes": 31,
    "l2_regularization": 1.0,
    "random_state": 42,
}


print("\nModel:")
print("HistGradientBoostingRegressor")

print("\nParameters:")
for key, value in MODEL_PARAMETERS.items():
    print(f"{key}: {value}")


# ------------------------------------------------------------
# Evaluation function
# ------------------------------------------------------------

def evaluate_predictions(
    y_true,
    y_pred,
):
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

    return mae, rmse, r2


# ------------------------------------------------------------
# Storage
# ------------------------------------------------------------

results = []

test_predictions = test[
    [
        "date",
        "reservoir",
    ]
].copy()


# ------------------------------------------------------------
# Train one model per forecasting horizon
# ------------------------------------------------------------

for target in TARGET_COLUMNS:

    print("\n")
    print("=" * 100)
    print(f"TRAINING MODEL — {target}")
    print("=" * 100)

    y_train = pd.to_numeric(
        train[target],
        errors="coerce",
    )

    y_validation = pd.to_numeric(
        validation[target],
        errors="coerce",
    )

    y_test = pd.to_numeric(
        test[target],
        errors="coerce",
    )

    # --------------------------------------------------------
    # Target integrity
    # --------------------------------------------------------

    if y_train.isna().any():
        raise RuntimeError(
            f"Training target contains NaN: {target}"
        )

    if y_validation.isna().any():
        raise RuntimeError(
            f"Validation target contains NaN: {target}"
        )

    if y_test.isna().any():
        raise RuntimeError(
            f"Test target contains NaN: {target}"
        )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = HistGradientBoostingRegressor(
        **MODEL_PARAMETERS
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("Training samples:", len(X_train))
    print("Features:", len(FEATURE_COLUMNS))

    model.fit(
        X_train,
        y_train,
    )

    print("Training complete.")

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    train_pred = model.predict(
        X_train
    )

    validation_pred = model.predict(
        X_validation
    )

    test_pred = model.predict(
        X_test
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    train_mae, train_rmse, train_r2 = (
        evaluate_predictions(
            y_train,
            train_pred,
        )
    )

    validation_mae, validation_rmse, validation_r2 = (
        evaluate_predictions(
            y_validation,
            validation_pred,
        )
    )

    test_mae, test_rmse, test_r2 = (
        evaluate_predictions(
            y_test,
            test_pred,
        )
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\nTRAIN")
    print(
        f"MAE  = {train_mae:.4f}"
    )
    print(
        f"RMSE = {train_rmse:.4f}"
    )
    print(
        f"R2   = {train_r2:.4f}"
    )

    print("\nVALIDATION")
    print(
        f"MAE  = {validation_mae:.4f}"
    )
    print(
        f"RMSE = {validation_rmse:.4f}"
    )
    print(
        f"R2   = {validation_r2:.4f}"
    )

    print("\nTEST")
    print(
        f"MAE  = {test_mae:.4f}"
    )
    print(
        f"RMSE = {test_rmse:.4f}"
    )
    print(
        f"R2   = {test_r2:.4f}"
    )

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    results.extend(
        [
            {
                "dataset": "TRAIN",
                "target": target,
                "samples": len(y_train),
                "MAE": train_mae,
                "RMSE": train_rmse,
                "R2": train_r2,
            },
            {
                "dataset": "VALIDATION",
                "target": target,
                "samples": len(y_validation),
                "MAE": validation_mae,
                "RMSE": validation_rmse,
                "R2": validation_r2,
            },
            {
                "dataset": "TEST",
                "target": target,
                "samples": len(y_test),
                "MAE": test_mae,
                "RMSE": test_rmse,
                "R2": test_r2,
            },
        ]
    )

    # --------------------------------------------------------
    # Store test predictions
    # --------------------------------------------------------

    test_predictions[
        f"prediction_{target}"
    ] = test_pred

    test_predictions[
        f"actual_{target}"
    ] = y_test.to_numpy()

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_path = (
        OUTPUT_DIR
        / f"hist_gradient_boosting_{target}.pkl"
    )

    joblib.dump(
        model,
        model_path,
    )

    print(
        "\nSaved model:",
        model_path,
    )


# ============================================================
# Results dataframe
# ============================================================

results_df = pd.DataFrame(
    results
)


# ------------------------------------------------------------
# Save results
# ------------------------------------------------------------

RESULTS_PATH = (
    OUTPUT_DIR
    / "tree_baseline_results.csv"
)

results_df.to_csv(
    RESULTS_PATH,
    index=False,
)


# ------------------------------------------------------------
# Save predictions
# ------------------------------------------------------------

PREDICTIONS_PATH = (
    OUTPUT_DIR
    / "tree_baseline_test_predictions.csv"
)

test_predictions.to_csv(
    PREDICTIONS_PATH,
    index=False,
)


# ============================================================
# Test summary
# ============================================================

print("\n")
print("=" * 100)
print("TEST SET SUMMARY")
print("=" * 100)

print(
    results_df[
        results_df["dataset"] == "TEST"
    ].to_string(
        index=False
    )
)


# ============================================================
# Integrity checks
# ============================================================

print("\n")
print("=" * 100)
print("INTEGRITY CHECKS")
print("=" * 100)

print(
    "Train rows preserved:",
    len(train)
)

print(
    "Validation rows preserved:",
    len(validation)
)

print(
    "Test rows preserved:",
    len(test)
)

prediction_columns = [
    column
    for column in test_predictions.columns
    if column.startswith("prediction_")
]

print(
    "Prediction columns:",
    len(prediction_columns)
)

print(
    "Prediction rows:",
    len(test_predictions)
)

if len(test_predictions) != len(test):
    raise RuntimeError(
        "Prediction row count does not match test row count."
    )

if test_predictions[prediction_columns].isna().any().any():
    raise RuntimeError(
        "NaN values found in predictions."
    )

print(
    "PASS — prediction integrity checks passed."
)


# ============================================================
# Leakage check
# ============================================================

print("\n")
print("=" * 100)
print("LEAKAGE CHECK")
print("=" * 100)

future_columns = [
    column
    for column in FEATURE_COLUMNS
    if any(
        keyword in column.lower()
        for keyword in [
            "target",
            "future",
        ]
    )
]

if future_columns:
    raise RuntimeError(
        "Potential leakage columns found: "
        + str(future_columns)
    )

print(
    "PASS — no target/future columns found in feature matrix."
)


# ============================================================
# Complete
# ============================================================

print("\n")
print("=" * 100)
print("MILESTONE 11.3 COMPLETE")
print("=" * 100)

print("\nSaved:")

print(
    "RESULTS     :",
    RESULTS_PATH
)

print(
    "PREDICTIONS :",
    PREDICTIONS_PATH
)

print(
    "MODELS      :",
    OUTPUT_DIR
)

print("\nDONE")
