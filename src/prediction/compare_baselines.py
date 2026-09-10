from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# MILESTONE 11.4 — BASELINE COMPARISON
# ============================================================

print("=" * 100)
print("MILESTONE 11.4 — BASELINE COMPARISON + ORIGINAL-UNIT EVALUATION")
print("=" * 100)


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

TEST_PATH = Path(
    "data/processed/splits/test.csv"
)

TARGET_SCALER_PATH = Path(
    "data/processed/scaled/target_scaler.pkl"
)

PERSISTENCE_PATH = Path(
    "data/processed/baselines/persistence_test_predictions.csv"
)

ROLLING_PATH = Path(
    "data/processed/baselines/rolling_mean_test_predictions.csv"
)

TREE_PATH = Path(
    "data/processed/baselines/tree/tree_baseline_test_predictions.csv"
)

OUTPUT_DIR = Path(
    "data/processed/baselines/comparison"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_RESULTS = (
    OUTPUT_DIR /
    "baseline_comparison_original_units.csv"
)


TARGETS = [
    "target_1d",
    "target_3d",
    "target_7d",
]


# ------------------------------------------------------------
# Load
# ------------------------------------------------------------

test_df = pd.read_csv(TEST_PATH)

target_scaler = joblib.load(
    TARGET_SCALER_PATH
)

persistence = pd.read_csv(
    PERSISTENCE_PATH
)

rolling = pd.read_csv(
    ROLLING_PATH
)

tree = pd.read_csv(
    TREE_PATH
)


print("\nInput test dataset:")
print("Rows:", len(test_df))
print("Columns:", len(test_df.columns))

print("\nPrediction datasets:")
print("Persistence:", persistence.shape)
print("Rolling mean:", rolling.shape)
print("Tree:", tree.shape)


# ------------------------------------------------------------
# Row-count checks
# ------------------------------------------------------------

expected_rows = len(test_df)

for name, df in [
    ("Persistence", persistence),
    ("Rolling mean", rolling),
    ("Tree", tree),
]:

    if len(df) != expected_rows:
        raise RuntimeError(
            f"{name} row count mismatch: "
            f"{len(df)} != {expected_rows}"
        )


# ------------------------------------------------------------
# Row alignment checks
# ------------------------------------------------------------

for name, prediction_df in [
    ("Persistence", persistence),
    ("Rolling mean", rolling),
    ("Tree", tree),
]:

    if "date" in prediction_df.columns:

        if not (
            prediction_df["date"]
            .astype(str)
            .reset_index(drop=True)
            .equals(
                test_df["date"]
                .astype(str)
                .reset_index(drop=True)
            )
        ):

            raise RuntimeError(
                f"{name}: date ordering mismatch."
            )

    if "reservoir" in prediction_df.columns:

        if not (
            prediction_df["reservoir"]
            .astype(str)
            .reset_index(drop=True)
            .equals(
                test_df["reservoir"]
                .astype(str)
                .reset_index(drop=True)
            )
        ):

            raise RuntimeError(
                f"{name}: reservoir ordering mismatch."
            )


print("\nRow alignment:")
print(
    "PASS — all prediction files align with test dataset."
)


# ============================================================
# TREE MODEL
# ============================================================

print("\n")
print("=" * 100)
print("TREE PREDICTION SCALING")
print("=" * 100)


tree_prediction_columns = [
    "prediction_target_1d",
    "prediction_target_3d",
    "prediction_target_7d",
]

tree_actual_columns = [
    "actual_target_1d",
    "actual_target_3d",
    "actual_target_7d",
]


# ------------------------------------------------------------
# Check columns
# ------------------------------------------------------------

for column in (
    tree_prediction_columns +
    tree_actual_columns
):

    if column not in tree.columns:

        raise RuntimeError(
            f"Missing tree column: {column}"
        )


# ------------------------------------------------------------
# Build 3-column scaled matrices
# ------------------------------------------------------------

tree_predictions_scaled = tree[
    tree_prediction_columns
].to_numpy(
    dtype=float
)

tree_actual_scaled = tree[
    tree_actual_columns
].to_numpy(
    dtype=float
)


if np.isnan(
    tree_predictions_scaled
).any():

    raise RuntimeError(
        "Tree predictions contain NaN."
    )


if np.isnan(
    tree_actual_scaled
).any():

    raise RuntimeError(
        "Tree actual values contain NaN."
    )


print(
    "Scaled prediction shape:",
    tree_predictions_scaled.shape
)

print(
    "Scaled actual shape:",
    tree_actual_scaled.shape
)


# ------------------------------------------------------------
# IMPORTANT:
#
# target_scaler was fitted on ALL THREE targets together.
# Therefore inverse-transform all three simultaneously.
# ------------------------------------------------------------

tree_predictions_original = (
    target_scaler.inverse_transform(
        tree_predictions_scaled
    )
)

tree_actual_original = (
    target_scaler.inverse_transform(
        tree_actual_scaled
    )
)


print(
    "Inverse-transformed prediction shape:",
    tree_predictions_original.shape
)

print(
    "Inverse-transformed actual shape:",
    tree_actual_original.shape
)


# ------------------------------------------------------------
# Verify tree actual values against test targets
# ------------------------------------------------------------

test_actual_matrix = test_df[
    TARGETS
].to_numpy(
    dtype=float
)


if not np.allclose(
    tree_actual_original,
    test_actual_matrix,
    rtol=1e-5,
    atol=1e-5
):

    raise RuntimeError(
        "Tree actual values do not match "
        "test targets after inverse scaling."
    )


print(
    "Tree actual target verification: PASS"
)


# ============================================================
# METRIC CALCULATION
# ============================================================

results = []


for target_index, target in enumerate(TARGETS):

    print("\n")
    print("-" * 100)
    print(
        f"EVALUATING {target}"
    )
    print("-" * 100)


    # --------------------------------------------------------
    # Actual target
    # --------------------------------------------------------

    actual = test_actual_matrix[
        :,
        target_index
    ]


    # --------------------------------------------------------
    # Persistence prediction
    # --------------------------------------------------------

    persistence_column = (
        f"prediction_"
        f"{target.replace('target_', '')}"
    )

    if persistence_column not in persistence.columns:

        raise RuntimeError(
            f"Missing persistence column: "
            f"{persistence_column}"
        )


    persistence_prediction = (
        persistence[
            persistence_column
        ].to_numpy(
            dtype=float
        )
    )


    # --------------------------------------------------------
    # Rolling prediction
    # --------------------------------------------------------

    rolling_column = (
        f"prediction_"
        f"{target.replace('target_', '')}"
        f"_rolling_mean"
    )

    if rolling_column not in rolling.columns:

        raise RuntimeError(
            f"Missing rolling column: "
            f"{rolling_column}"
        )


    rolling_prediction = (
        rolling[
            rolling_column
        ].to_numpy(
            dtype=float
        )
    )


    # --------------------------------------------------------
    # Tree prediction
    # --------------------------------------------------------

    tree_prediction = (
        tree_predictions_original[
            :,
            target_index
        ]
    )


    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    models = {

        "Persistence":
            persistence_prediction,

        "RollingMean":
            rolling_prediction,

        "HistGradientBoosting":
            tree_prediction,
    }


    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    for model_name, prediction in models.items():

        if len(prediction) != len(actual):

            raise RuntimeError(
                f"{model_name}/{target}: "
                "prediction length mismatch."
            )


        if np.isnan(prediction).any():

            raise RuntimeError(
                f"{model_name}/{target}: "
                "prediction contains NaN."
            )


        mae = mean_absolute_error(
            actual,
            prediction
        )


        rmse = np.sqrt(
            mean_squared_error(
                actual,
                prediction
            )
        )


        r2 = r2_score(
            actual,
            prediction
        )


        results.append(
            {
                "dataset": "TEST",
                "model": model_name,
                "target": target,
                "samples": len(actual),
                "MAE_original_units": mae,
                "RMSE_original_units": rmse,
                "R2": r2,
            }
        )


# ------------------------------------------------------------
# Results dataframe
# ------------------------------------------------------------

results_df = pd.DataFrame(
    results
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n")
print("=" * 100)
print(
    "TEST SET BASELINE COMPARISON — ORIGINAL INFLOW UNITS"
)
print("=" * 100)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# BEST MODEL
# ============================================================

print("\n")
print("=" * 100)
print("BEST MODEL BY FORECAST HORIZON")
print("=" * 100)


for target in TARGETS:

    subset = results_df[
        results_df["target"] == target
    ]


    best = subset.loc[
        subset["R2"].idxmax()
    ]


    print(
        f"{target}: "
        f"{best['model']} | "
        f"R²={best['R2']:.6f} | "
        f"MAE={best['MAE_original_units']:.6f} | "
        f"RMSE={best['RMSE_original_units']:.6f}"
    )


# ============================================================
# MODEL RANKING
# ============================================================

print("\n")
print("=" * 100)
print("MODEL RANKING BY R²")
print("=" * 100)


for target in TARGETS:

    print(f"\n{target}")

    subset = results_df[
        results_df["target"] == target
    ].sort_values(
        "R2",
        ascending=False
    )


    for rank, (_, row) in enumerate(
        subset.iterrows(),
        start=1
    ):

        print(
            f"{rank}. "
            f"{row['model']} "
            f"(R²={row['R2']:.6f})"
        )


# ============================================================
# INTEGRITY CHECKS
# ============================================================

print("\n")
print("=" * 100)
print("INTEGRITY CHECKS")
print("=" * 100)


print(
    "Test rows:",
    len(test_df)
)

print(
    "Persistence rows:",
    len(persistence)
)

print(
    "Rolling rows:",
    len(rolling)
)

print(
    "Tree rows:",
    len(tree)
)


if (
    len(test_df)
    == len(persistence)
    == len(rolling)
    == len(tree)
):

    print(
        "PASS — all prediction datasets "
        "have matching row counts."
    )

else:

    raise RuntimeError(
        "FAIL — prediction row counts do not match."
    )


print(
    "Tree inverse scaling: PASS"
)

print(
    "Tree actual target verification: PASS"
)

print(
    "Original-unit evaluation: PASS"
)


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    OUTPUT_RESULTS,
    index=False
)


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 100)
print("MILESTONE 11.4 COMPLETE")
print("=" * 100)

print("\nSaved:")
print(
    "RESULTS:",
    OUTPUT_RESULTS
)

print("\nDONE")
