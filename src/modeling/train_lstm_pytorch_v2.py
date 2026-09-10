import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

DATA_DIR = "data/processed/lstm"
MODEL_DIR = "models/lstm_pytorch_v2"
RESULT_DIR = "results/lstm_pytorch_v2"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

BATCH_SIZE = 128
MAX_EPOCHS = 100
LEARNING_RATE = 0.001
EARLY_STOPPING = 12

DEVICE = torch.device("cpu")


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# HEADER
# ============================================================

print("=" * 100)
print("MILESTONE 11.14 — SIMPLE LSTM + HUBER LOSS")
print("=" * 100)

print()
print("Device:")
print(DEVICE)


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 100)
print("LOADING PREPARED LSTM DATA")
print("=" * 100)

X_train = np.load(
    f"{DATA_DIR}/X_train_dynamic.npy"
).astype(np.float32)

X_val = np.load(
    f"{DATA_DIR}/X_validation_dynamic.npy"
).astype(np.float32)

X_test = np.load(
    f"{DATA_DIR}/X_test_dynamic.npy"
).astype(np.float32)

y_train = np.load(
    f"{DATA_DIR}/y_train.npy"
).astype(np.float32)

y_val = np.load(
    f"{DATA_DIR}/y_validation.npy"
).astype(np.float32)

y_test = np.load(
    f"{DATA_DIR}/y_test.npy"
).astype(np.float32)


print()
print("DATASET SHAPES")
print("-" * 100)

print(f"TRAIN dynamic       : {X_train.shape}")
print(f"VALIDATION dynamic  : {X_val.shape}")
print(f"TEST dynamic        : {X_test.shape}")

print(f"TRAIN targets       : {y_train.shape}")
print(f"VALIDATION targets  : {y_val.shape}")
print(f"TEST targets        : {y_test.shape}")


# ============================================================
# FINITE VALUE CHECK
# ============================================================

print()
print("=" * 100)
print("FINITE VALUE CHECK")
print("=" * 100)

arrays = {
    "X_train": X_train,
    "X_validation": X_val,
    "X_test": X_test,
    "y_train": y_train,
    "y_validation": y_val,
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

X_train_t = torch.from_numpy(X_train)
X_val_t = torch.from_numpy(X_val)
X_test_t = torch.from_numpy(X_test)

y_train_t = torch.from_numpy(y_train)
y_val_t = torch.from_numpy(y_val)
y_test_t = torch.from_numpy(y_test)


# ============================================================
# DATA LOADERS
# ============================================================

train_dataset = TensorDataset(
    X_train_t,
    y_train_t
)

val_dataset = TensorDataset(
    X_val_t,
    y_val_t
)

test_dataset = TensorDataset(
    X_test_t,
    y_test_t
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
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

class SimpleReservoirLSTM(nn.Module):

    def __init__(
        self,
        input_size=5,
        hidden_size=64,
        output_size=3
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.2)

        self.fc1 = nn.Linear(
            hidden_size,
            32
        )

        self.relu = nn.ReLU()

        self.output = nn.Linear(
            32,
            output_size
        )

    def forward(self, x):

        sequence_output, _ = self.lstm(x)

        last_hidden = sequence_output[:, -1, :]

        x = self.dropout(last_hidden)

        x = self.fc1(x)

        x = self.relu(x)

        x = self.output(x)

        return x


model = SimpleReservoirLSTM().to(DEVICE)


# ============================================================
# MODEL INFORMATION
# ============================================================

print()
print("=" * 100)
print("MODEL CONFIGURATION")
print("=" * 100)

print(model)

total_params = sum(
    p.numel()
    for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print()
print(f"Total parameters     : {total_params:,}")
print(f"Trainable parameters : {trainable_params:,}")


# ============================================================
# LOSS + OPTIMIZER
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
print("TRAINING LSTM V2")
print("=" * 100)

print(
    f"Max epochs       : {MAX_EPOCHS}"
)

print(
    f"Batch size       : {BATCH_SIZE}"
)

print(
    f"Learning rate    : {LEARNING_RATE}"
)

print(
    f"Loss             : HuberLoss(delta=1.0)"
)

print(
    f"Early stopping   : {EARLY_STOPPING}"
)


best_val_loss = float("inf")
best_epoch = 0
patience = 0

history = []


for epoch in range(1, MAX_EPOCHS + 1):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_losses = []

    for xb, yb in train_loader:

        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        optimizer.zero_grad()

        predictions = model(xb)

        loss = criterion(
            predictions,
            yb
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

    val_losses = []

    with torch.no_grad():

        for xb, yb in val_loader:

            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            predictions = model(xb)

            loss = criterion(
                predictions,
                yb
            )

            val_losses.append(
                loss.item()
            )

    val_loss = float(
        np.mean(val_losses)
    )


    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss
    })


    print(
        f"Epoch {epoch:03d}/{MAX_EPOCHS} | "
        f"train_loss={train_loss:.6f} | "
        f"val_loss={val_loss:.6f}"
    )


    # --------------------------------------------------------
    # CHECKPOINT
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        best_epoch = epoch

        patience = 0

        torch.save(
            model.state_dict(),
            f"{MODEL_DIR}/best_model.pt"
        )

    else:

        patience += 1


    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if patience >= EARLY_STOPPING:

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

model.load_state_dict(
    torch.load(
        f"{MODEL_DIR}/best_model.pt",
        map_location=DEVICE
    )
)

model.eval()

print(
    f"Best epoch           : {best_epoch}"
)

print(
    f"Best validation loss : {best_val_loss:.6f}"
)


# ============================================================
# PREDICTIONS
# ============================================================

def predict(loader):

    predictions = []

    with torch.no_grad():

        for xb, _ in loader:

            xb = xb.to(DEVICE)

            output = model(xb)

            predictions.append(
                output.cpu().numpy()
            )

    return np.concatenate(
        predictions,
        axis=0
    )


train_predictions = predict(
    train_loader
)

val_predictions = predict(
    val_loader
)

test_predictions = predict(
    test_loader
)


# ============================================================
# METRICS
# ============================================================

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def calculate_metrics(
    actual,
    prediction
):

    rows = []

    targets = [
        "target_1d",
        "target_3d",
        "target_7d"
    ]

    for i, target in enumerate(targets):

        a = actual[:, i]
        p = prediction[:, i]

        mae = mean_absolute_error(
            a,
            p
        )

        rmse = np.sqrt(
            mean_squared_error(
                a,
                p
            )
        )

        r2 = r2_score(
            a,
            p
        )

        rows.append({
            "target": target,
            "samples": len(a),
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

    return pd.DataFrame(rows)


print()
print("=" * 100)
print("SCALED-UNIT EVALUATION")
print("=" * 100)

train_metrics = calculate_metrics(
    y_train,
    train_predictions
)

val_metrics = calculate_metrics(
    y_val,
    val_predictions
)

test_metrics = calculate_metrics(
    y_test,
    test_predictions
)

print()
print("TRAIN")
print(train_metrics.to_string(index=False))

print()
print("VALIDATION")
print(val_metrics.to_string(index=False))

print()
print("TEST")
print(test_metrics.to_string(index=False))


# ============================================================
# ORIGINAL UNIT CONVERSION
# ============================================================

target_scaler_path = (
    "data/processed/scaled/target_scaler.pkl"
)

import joblib

target_scaler = joblib.load(
    target_scaler_path
)

train_pred_original = (
    target_scaler.inverse_transform(
        train_predictions
    )
)

val_pred_original = (
    target_scaler.inverse_transform(
        val_predictions
    )
)

test_pred_original = (
    target_scaler.inverse_transform(
        test_predictions
    )
)

train_actual_original = (
    target_scaler.inverse_transform(
        y_train
    )
)

val_actual_original = (
    target_scaler.inverse_transform(
        y_val
    )
)

test_actual_original = (
    target_scaler.inverse_transform(
        y_test
    )
)


print()
print("=" * 100)
print("ORIGINAL-UNIT EVALUATION")
print("=" * 100)

train_original_metrics = calculate_metrics(
    train_actual_original,
    train_pred_original
)

val_original_metrics = calculate_metrics(
    val_actual_original,
    val_pred_original
)

test_original_metrics = calculate_metrics(
    test_actual_original,
    test_pred_original
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
    val_original_metrics.to_string(
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
# SAVE HISTORY
# ============================================================

history_df = pd.DataFrame(history)

history_df.to_csv(
    f"{RESULT_DIR}/training_history.csv",
    index=False
)


# ============================================================
# SAVE METRICS
# ============================================================

all_metrics = []

for dataset_name, metrics in [
    ("TRAIN", train_original_metrics),
    ("VALIDATION", val_original_metrics),
    ("TEST", test_original_metrics)
]:

    temp = metrics.copy()

    temp.insert(
        0,
        "dataset",
        dataset_name
    )

    all_metrics.append(temp)

metrics_df = pd.concat(
    all_metrics,
    ignore_index=True
)

metrics_df.to_csv(
    f"{RESULT_DIR}/lstm_v2_metrics_original_units.csv",
    index=False
)


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

test_prediction_df = pd.DataFrame({

    "target_1d_actual":
        test_actual_original[:, 0],

    "target_1d_prediction":
        test_pred_original[:, 0],

    "target_3d_actual":
        test_actual_original[:, 1],

    "target_3d_prediction":
        test_pred_original[:, 1],

    "target_7d_actual":
        test_actual_original[:, 2],

    "target_7d_prediction":
        test_pred_original[:, 2],
})

test_prediction_df.to_csv(
    f"{RESULT_DIR}/test_predictions_original_units.csv",
    index=False
)


# ============================================================
# INTEGRITY CHECKS
# ============================================================

print()
print("=" * 100)
print("INTEGRITY CHECKS")
print("=" * 100)

assert len(train_predictions) == len(y_train)
assert len(val_predictions) == len(y_val)
assert len(test_predictions) == len(y_test)

assert train_predictions.shape[1] == 3
assert val_predictions.shape[1] == 3
assert test_predictions.shape[1] == 3

assert np.isfinite(
    test_predictions
).all()

assert np.isfinite(
    test_pred_original
).all()

print("PASS — row counts preserved")
print("PASS — prediction dimensions correct")
print("PASS — no NaN/Inf in predictions")
print("PASS — original-unit inverse scaling")


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 100)
print("MILESTONE 11.14 — SIMPLE LSTM + HUBER COMPLETE")
print("=" * 100)

print()
print(
    f"Best epoch: {best_epoch}"
)

print(
    f"Best validation loss: {best_val_loss:.6f}"
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
    f"MODEL       : {MODEL_DIR}/best_model.pt"
)

print(
    f"HISTORY     : {RESULT_DIR}/training_history.csv"
)

print(
    f"METRICS     : "
    f"{RESULT_DIR}/lstm_v2_metrics_original_units.csv"
)

print(
    f"PREDICTIONS : "
    f"{RESULT_DIR}/test_predictions_original_units.csv"
)

print()
print("DONE")
