# ============================================================
# MILESTONE 11.17
# NON-NEGATIVE LSTM + HUBER LOSS
# ============================================================

import os
import copy
import random
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

MAX_EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 12

HIDDEN_SIZE = 64
DROPOUT = 0.20

OUTPUT_MODEL_DIR = "models/lstm_pytorch_v2_nonnegative"
OUTPUT_RESULTS_DIR = "results/lstm_pytorch_v2_nonnegative"

os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_RESULTS_DIR, exist_ok=True)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 100)
print("MILESTONE 11.17 — NON-NEGATIVE LSTM + HUBER LOSS")
print("=" * 100)

print()
print("Device:")
print(device)


# ============================================================
# LOAD PREPARED LSTM DATA
# ============================================================

print()
print("=" * 100)
print("LOADING PREPARED LSTM DATA")
print("=" * 100)

X_train_dynamic = np.load(
    "data/processed/lstm/X_train_dynamic.npy"
)

X_validation_dynamic = np.load(
    "data/processed/lstm/X_validation_dynamic.npy"
)

X_test_dynamic = np.load(
    "data/processed/lstm/X_test_dynamic.npy"
)

y_train = np.load(
    "data/processed/lstm/y_train.npy"
)

y_validation = np.load(
    "data/processed/lstm/y_validation.npy"
)

y_test = np.load(
    "data/processed/lstm/y_test.npy"
)


print()
print("DATASET SHAPES")
print("-" * 100)

print(
    f"TRAIN dynamic       : {X_train_dynamic.shape}"
)

print(
    f"VALIDATION dynamic  : {X_validation_dynamic.shape}"
)

print(
    f"TEST dynamic        : {X_test_dynamic.shape}"
)

print(
    f"TRAIN targets       : {y_train.shape}"
)

print(
    f"VALIDATION targets  : {y_validation.shape}"
)

print(
    f"TEST targets        : {y_test.shape}"
)


# ============================================================
# FINITE VALUE CHECK
# ============================================================

print()
print("=" * 100)
print("FINITE VALUE CHECK")
print("=" * 100)

arrays = {
    "X_train": X_train_dynamic,
    "X_validation": X_validation_dynamic,
    "X_test": X_test_dynamic,
    "y_train": y_train,
    "y_validation": y_validation,
    "y_test": y_test,
}

for name, arr in arrays.items():

    passed = np.isfinite(arr).all()

    print(
        f"{name:20s}: "
        f"{'PASS' if passed else 'FAIL'}"
    )

    if not passed:
        raise ValueError(
            f"Non-finite values found in {name}"
        )


# ============================================================
# PYTORCH TENSORS
# ============================================================

print()
print("=" * 100)
print("CONVERTING TO PYTORCH TENSORS")
print("=" * 100)

X_train_tensor = torch.tensor(
    X_train_dynamic,
    dtype=torch.float32
)

X_validation_tensor = torch.tensor(
    X_validation_dynamic,
    dtype=torch.float32
)

X_test_tensor = torch.tensor(
    X_test_dynamic,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train,
    dtype=torch.float32
)

y_validation_tensor = torch.tensor(
    y_validation,
    dtype=torch.float32
)

y_test_tensor = torch.tensor(
    y_test,
    dtype=torch.float32
)


# ============================================================
# DATA LOADERS
# ============================================================

train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

validation_dataset = TensorDataset(
    X_validation_tensor,
    y_validation_tensor
)

test_dataset = TensorDataset(
    X_test_tensor,
    y_test_tensor
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# MODEL
# ============================================================

class NonNegativeReservoirLSTM(nn.Module):

    def __init__(
        self,
        input_size=5,
        hidden_size=64,
        dropout=0.20,
        output_size=3
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(
            hidden_size,
            32
        )

        self.relu = nn.ReLU()

        # ----------------------------------------------------
        # IMPORTANT CHANGE FROM V2
        #
        # Softplus guarantees non-negative predictions.
        # ----------------------------------------------------

        self.output = nn.Sequential(
            nn.Linear(32, output_size),
            nn.Softplus()
        )


    def forward(self, x):

        lstm_output, _ = self.lstm(x)

        # Last timestep
        x = lstm_output[:, -1, :]

        x = self.dropout(x)

        x = self.fc1(x)

        x = self.relu(x)

        x = self.output(x)

        return x


model = NonNegativeReservoirLSTM(
    input_size=5,
    hidden_size=HIDDEN_SIZE,
    dropout=DROPOUT,
    output_size=3
).to(device)


# ============================================================
# MODEL SUMMARY
# ============================================================

print()
print("=" * 100)
print("MODEL CONFIGURATION")
print("=" * 100)

print(model)

total_parameters = sum(
    p.numel()
    for p in model.parameters()
)

trainable_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print()
print(f"Total parameters     : {total_parameters:,}")
print(f"Trainable parameters : {trainable_parameters:,}")

print()
print("Output constraint:")
print("Softplus → prediction >= 0")


# ============================================================
# LOSS / OPTIMIZER
# ============================================================

criterion = nn.HuberLoss(
    delta=1.0
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

print()
print("=" * 100)
print("TRAINING LSTM V2.1")
print("=" * 100)

print(f"Max epochs       : {MAX_EPOCHS}")
print(f"Batch size       : {BATCH_SIZE}")
print(f"Learning rate    : {LEARNING_RATE}")
print("Loss             : HuberLoss(delta=1.0)")
print(f"Early stopping   : {EARLY_STOPPING_PATIENCE}")


best_val_loss = float("inf")
best_epoch = 0
best_state = None
patience_counter = 0

history = []


for epoch in range(1, MAX_EPOCHS + 1):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_losses = []

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        predictions = model(X_batch)

        loss = criterion(
            predictions,
            y_batch
        )

        loss.backward()

        optimizer.step()

        train_losses.append(
            loss.item()
        )

    train_loss = float(
        np.mean(train_losses)
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    validation_losses = []

    with torch.no_grad():

        for X_batch, y_batch in validation_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            predictions = model(X_batch)

            loss = criterion(
                predictions,
                y_batch
            )

            validation_losses.append(
                loss.item()
            )

    validation_loss = float(
        np.mean(validation_losses)
    )


    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": validation_loss
    })


    print(
        f"Epoch {epoch:03d}/{MAX_EPOCHS} | "
        f"train_loss={train_loss:.6f} | "
        f"val_loss={validation_loss:.6f}"
    )


    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if validation_loss < best_val_loss:

        best_val_loss = validation_loss
        best_epoch = epoch

        best_state = copy.deepcopy(
            model.state_dict()
        )

        patience_counter = 0

    else:

        patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:

            print()
            print(
                f"Early stopping at epoch {epoch}."
            )

            break


# ============================================================
# LOAD BEST MODEL
# ============================================================

print()
print("=" * 100)
print("LOADING BEST MODEL")
print("=" * 100)

model.load_state_dict(best_state)

print(
    f"Best epoch           : {best_epoch}"
)

print(
    f"Best validation loss : {best_val_loss:.6f}"
)


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict(model, loader):

    model.eval()

    predictions = []

    with torch.no_grad():

        for X_batch, _ in loader:

            X_batch = X_batch.to(device)

            output = model(X_batch)

            predictions.append(
                output.cpu().numpy()
            )

    return np.concatenate(
        predictions,
        axis=0
    )


pred_train_scaled = predict(
    model,
    train_loader
)

pred_validation_scaled = predict(
    model,
    validation_loader
)

pred_test_scaled = predict(
    model,
    test_loader
)


# ============================================================
# SCALED EVALUATION
# ============================================================

def evaluate_predictions(
    actual,
    prediction
):

    rows = []

    for i, target in enumerate(
        ["target_1d", "target_3d", "target_7d"]
    ):

        y_true = actual[:, i]
        y_pred = prediction[:, i]

        rows.append({

            "target": target,

            "samples": len(y_true),

            "MAE":
                mean_absolute_error(
                    y_true,
                    y_pred
                ),

            "RMSE":
                np.sqrt(
                    mean_squared_error(
                        y_true,
                        y_pred
                    )
                ),

            "R2":
                r2_score(
                    y_true,
                    y_pred
                )
        })

    return pd.DataFrame(rows)


print()
print("=" * 100)
print("SCALED-UNIT EVALUATION")
print("=" * 100)

train_scaled_metrics = evaluate_predictions(
    y_train,
    pred_train_scaled
)

validation_scaled_metrics = evaluate_predictions(
    y_validation,
    pred_validation_scaled
)

test_scaled_metrics = evaluate_predictions(
    y_test,
    pred_test_scaled
)

print()
print("TRAIN")
print(
    train_scaled_metrics.to_string(
        index=False
    )
)

print()
print("VALIDATION")
print(
    validation_scaled_metrics.to_string(
        index=False
    )
)

print()
print("TEST")
print(
    test_scaled_metrics.to_string(
        index=False
    )
)


# ============================================================
# ORIGINAL UNIT INVERSE TRANSFORM
# ============================================================

print()
print("=" * 100)
print("ORIGINAL-UNIT EVALUATION")
print("=" * 100)

target_scaler = joblib.load(
    "data/processed/scaled/target_scaler.pkl"
)


pred_train_original = (
    target_scaler.inverse_transform(
        pred_train_scaled
    )
)

pred_validation_original = (
    target_scaler.inverse_transform(
        pred_validation_scaled
    )
)

pred_test_original = (
    target_scaler.inverse_transform(
        pred_test_scaled
    )
)

actual_train_original = (
    target_scaler.inverse_transform(
        y_train
    )
)

actual_validation_original = (
    target_scaler.inverse_transform(
        y_validation
    )
)

actual_test_original = (
    target_scaler.inverse_transform(
        y_test
    )
)


train_original_metrics = evaluate_predictions(
    actual_train_original,
    pred_train_original
)

validation_original_metrics = evaluate_predictions(
    actual_validation_original,
    pred_validation_original
)

test_original_metrics = evaluate_predictions(
    actual_test_original,
    pred_test_original
)


print()
print("TRAIN — ORIGINAL INFLOW UNITS")
print(
    train_original_metrics.to_string(
        index=False
    )
)

print()
print("VALIDATION — ORIGINAL INFLOW UNITS")
print(
    validation_original_metrics.to_string(
        index=False
    )
)

print()
print("TEST — ORIGINAL INFLOW UNITS")
print(
    test_original_metrics.to_string(
        index=False
    )
)


# ============================================================
# PHYSICAL VALIDITY
# ============================================================

print()
print("=" * 100)
print("PHYSICAL VALIDITY CHECK")
print("=" * 100)

for i, target in enumerate(
    ["target_1d", "target_3d", "target_7d"]
):

    predictions = pred_test_original[:, i]

    negative_count = int(
        np.sum(predictions < 0)
    )

    negative_percentage = (
        100.0 *
        negative_count /
        len(predictions)
    )

    print(
        f"{target}: "
        f"{negative_count}/{len(predictions)} "
        f"negative "
        f"({negative_percentage:.2f}%)"
    )

    print(
        f"  minimum prediction = "
        f"{predictions.min():.6f}"
    )

    print(
        f"  maximum prediction = "
        f"{predictions.max():.6f}"
    )


# ============================================================
# PREDICTION DISTRIBUTION
# ============================================================

print()
print("=" * 100)
print("TEST PREDICTION DISTRIBUTION")
print("=" * 100)

prediction_rows = []

for i, target in enumerate(
    ["target_1d", "target_3d", "target_7d"]
):

    actual = actual_test_original[:, i]
    prediction = pred_test_original[:, i]

    prediction_rows.append({

        f"{target}_actual_mean":
            np.mean(actual),

        f"{target}_prediction_mean":
            np.mean(prediction),

        f"{target}_bias":
            np.mean(prediction - actual),

        f"{target}_actual_std":
            np.std(actual),

        f"{target}_prediction_std":
            np.std(prediction),

        f"{target}_min_prediction":
            np.min(prediction),

        f"{target}_max_prediction":
            np.max(prediction),

        f"{target}_negative_count":
            np.sum(prediction < 0)
    })


prediction_distribution = pd.DataFrame(
    prediction_rows
)

print(
    prediction_distribution.to_string(
        index=False
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

model_path = (
    f"{OUTPUT_MODEL_DIR}/best_model.pt"
)

torch.save(
    {
        "model_state_dict":
            model.state_dict(),

        "input_size": 5,

        "hidden_size":
            HIDDEN_SIZE,

        "dropout":
            DROPOUT,

        "output_size": 3,

        "best_epoch":
            best_epoch,

        "best_validation_loss":
            best_val_loss,

        "seed":
            SEED
    },
    model_path
)


# ============================================================
# SAVE HISTORY
# ============================================================

history_path = (
    f"{OUTPUT_RESULTS_DIR}/training_history.csv"
)

pd.DataFrame(history).to_csv(
    history_path,
    index=False
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics_rows = []

for split_name, metrics in [
    ("TRAIN", train_original_metrics),
    ("VALIDATION", validation_original_metrics),
    ("TEST", test_original_metrics)
]:

    for _, row in metrics.iterrows():

        metrics_rows.append({

            "dataset":
                split_name,

            "target":
                row["target"],

            "samples":
                row["samples"],

            "MAE":
                row["MAE"],

            "RMSE":
                row["RMSE"],

            "R2":
                row["R2"]
        })


metrics_df = pd.DataFrame(
    metrics_rows
)

metrics_path = (
    f"{OUTPUT_RESULTS_DIR}/lstm_v2_nonnegative_metrics_original_units.csv"
)

metrics_df.to_csv(
    metrics_path,
    index=False
)


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame({

    "target_1d_actual":
        actual_test_original[:, 0],

    "target_1d_prediction":
        pred_test_original[:, 0],

    "target_3d_actual":
        actual_test_original[:, 1],

    "target_3d_prediction":
        pred_test_original[:, 1],

    "target_7d_actual":
        actual_test_original[:, 2],

    "target_7d_prediction":
        pred_test_original[:, 2]
})

prediction_path = (
    f"{OUTPUT_RESULTS_DIR}/test_predictions_original_units.csv"
)

prediction_df.to_csv(
    prediction_path,
    index=False
)


# ============================================================
# INTEGRITY CHECKS
# ============================================================

print()
print("=" * 100)
print("INTEGRITY CHECKS")
print("=" * 100)

assert len(pred_test_original) == len(y_test)

assert pred_test_original.shape == (
    len(y_test),
    3
)

assert np.isfinite(
    pred_test_original
).all()

assert np.isfinite(
    test_original_metrics[
        ["MAE", "RMSE", "R2"]
    ].to_numpy()
).all()

# Softplus should guarantee positive scaled
# predictions before inverse scaling.
assert np.all(
    pred_test_scaled > 0
)

print(
    "PASS — row counts preserved"
)

print(
    "PASS — prediction dimensions correct"
)

print(
    "PASS — no NaN/Inf in predictions"
)

print(
    "PASS — original-unit inverse scaling"
)

print(
    "PASS — Softplus output is strictly positive"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 100)
print("MILESTONE 11.17 — NON-NEGATIVE LSTM COMPLETE")
print("=" * 100)

print()
print(
    f"Best epoch: {best_epoch}"
)

print(
    f"Best validation loss: "
    f"{best_val_loss:.6f}"
)

print()
print("TEST RESULTS — ORIGINAL INFLOW UNITS")

print(
    test_original_metrics.to_string(
        index=False
    )
)

print()
print("Saved:")

print(
    f"MODEL       : {model_path}"
)

print(
    f"HISTORY     : {history_path}"
)

print(
    f"METRICS     : {metrics_path}"
)

print(
    f"PREDICTIONS : {prediction_path}"
)

print()
print("DONE")
