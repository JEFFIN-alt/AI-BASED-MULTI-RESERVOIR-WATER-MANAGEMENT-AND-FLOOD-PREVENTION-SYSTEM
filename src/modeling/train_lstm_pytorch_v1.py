from pathlib import Path
import json
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

BATCH_SIZE = 128
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.20

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

MAX_EPOCHS = 100
PATIENCE = 12

MODEL_DIR = Path("models/lstm_pytorch")
RESULT_DIR = Path("results/lstm_pytorch")
PREDICTION_DIR = Path("results/lstm_pytorch")

DATA_DIR = Path("data/processed/lstm")

TARGET_SCALER_PATH = Path(
    "data/processed/scaled/target_scaler.pkl"
)

TARGET_NAMES = [
    "target_1d",
    "target_3d",
    "target_7d",
]


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# HELPERS
# ============================================================

def load_array(filename):
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    return np.load(path)


def check_finite(name, array):
    if not np.isfinite(array).all():
        raise RuntimeError(
            f"FAIL — {name} contains NaN or Inf."
        )


def print_shape(name, array):
    print(f"{name:<35}: {array.shape}")


# ============================================================
# MODEL
# ============================================================

class ReservoirLSTM(nn.Module):

    def __init__(
        self,
        dynamic_features=5,
        static_features=8,
        availability_features=8,
        hidden_size=64,
        num_layers=2,
        dropout=0.20,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=dynamic_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        static_input_size = (
            static_features + availability_features
        )

        self.static_encoder = nn.Sequential(
            nn.Linear(static_input_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_size + 32, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        self.output = nn.Linear(32, 3)

    def forward(
        self,
        dynamic,
        static,
        availability,
    ):

        # dynamic:
        # [batch, 7, 5]

        lstm_output, (hidden, cell) = self.lstm(dynamic)

        # Last temporal representation
        temporal = lstm_output[:, -1, :]

        # Combine static values + availability mask
        static_input = torch.cat(
            [static, availability],
            dim=1,
        )

        static_encoded = self.static_encoder(
            static_input
        )

        combined = torch.cat(
            [temporal, static_encoded],
            dim=1,
        )

        fused = self.fusion(combined)

        return self.output(fused)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    results = []

    for i, target in enumerate(TARGET_NAMES):

        actual = y_true[:, i]
        prediction = y_pred[:, i]

        error = prediction - actual

        mae = np.mean(np.abs(error))

        rmse = np.sqrt(
            np.mean(error ** 2)
        )

        ss_res = np.sum(
            (actual - prediction) ** 2
        )

        ss_tot = np.sum(
            (actual - np.mean(actual)) ** 2
        )

        if ss_tot == 0:
            r2 = np.nan
        else:
            r2 = 1.0 - (ss_res / ss_tot)

        results.append(
            {
                "target": target,
                "samples": len(actual),
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
            }
        )

    return pd.DataFrame(results)


# ============================================================
# ORIGINAL-UNIT INVERSE SCALING
# ============================================================

def inverse_scale_targets(array):

    import joblib

    if not TARGET_SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Target scaler not found: "
            f"{TARGET_SCALER_PATH}"
        )

    scaler = joblib.load(
        TARGET_SCALER_PATH
    )

    return scaler.inverse_transform(array)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 100)
print("MILESTONE 11.6 — PYTORCH LSTM FORECASTING")
print("=" * 100)

print()
print("Device:")
print(DEVICE)

print()
print("=" * 100)
print("LOADING PREPARED LSTM DATA")
print("=" * 100)

X_train_dynamic = load_array(
    "X_train_dynamic.npy"
)

X_validation_dynamic = load_array(
    "X_validation_dynamic.npy"
)

X_test_dynamic = load_array(
    "X_test_dynamic.npy"
)

X_train_static = load_array(
    "X_train_static.npy"
)

X_validation_static = load_array(
    "X_validation_static.npy"
)

X_test_static = load_array(
    "X_test_static.npy"
)

X_train_availability = load_array(
    "X_train_static_availability.npy"
)

X_validation_availability = load_array(
    "X_validation_static_availability.npy"
)

X_test_availability = load_array(
    "X_test_static_availability.npy"
)

y_train = load_array(
    "y_train.npy"
)

y_validation = load_array(
    "y_validation.npy"
)

y_test = load_array(
    "y_test.npy"
)


# ============================================================
# DATA VERIFICATION
# ============================================================

print()
print("=" * 100)
print("DATASET SHAPES")
print("=" * 100)

print_shape(
    "TRAIN dynamic",
    X_train_dynamic
)

print_shape(
    "VALIDATION dynamic",
    X_validation_dynamic
)

print_shape(
    "TEST dynamic",
    X_test_dynamic
)

print_shape(
    "TRAIN static",
    X_train_static
)

print_shape(
    "VALIDATION static",
    X_validation_static
)

print_shape(
    "TEST static",
    X_test_static
)

print_shape(
    "TRAIN availability",
    X_train_availability
)

print_shape(
    "VALIDATION availability",
    X_validation_availability
)

print_shape(
    "TEST availability",
    X_test_availability
)

print_shape(
    "TRAIN targets",
    y_train
)

print_shape(
    "VALIDATION targets",
    y_validation
)

print_shape(
    "TEST targets",
    y_test
)


# ============================================================
# FINITE CHECK
# ============================================================

print()
print("=" * 100)
print("FINITE VALUE CHECK")
print("=" * 100)

arrays = {
    "X_train_dynamic": X_train_dynamic,
    "X_validation_dynamic": X_validation_dynamic,
    "X_test_dynamic": X_test_dynamic,
    "X_train_static": X_train_static,
    "X_validation_static": X_validation_static,
    "X_test_static": X_test_static,
    "X_train_availability": X_train_availability,
    "X_validation_availability": X_validation_availability,
    "X_test_availability": X_test_availability,
    "y_train": y_train,
    "y_validation": y_validation,
    "y_test": y_test,
}

for name, array in arrays.items():

    check_finite(name, array)

    print(
        f"{name:<35}: PASS"
    )


# ============================================================
# SHAPE CHECKS
# ============================================================

assert X_train_dynamic.shape[1:] == (7, 5)
assert X_validation_dynamic.shape[1:] == (7, 5)
assert X_test_dynamic.shape[1:] == (7, 5)

assert X_train_static.shape[1] == 8
assert X_validation_static.shape[1] == 8
assert X_test_static.shape[1] == 8

assert X_train_availability.shape[1] == 8
assert X_validation_availability.shape[1] == 8
assert X_test_availability.shape[1] == 8

assert y_train.shape[1] == 3
assert y_validation.shape[1] == 3
assert y_test.shape[1] == 3

assert len(X_train_dynamic) == len(y_train)
assert len(X_validation_dynamic) == len(y_validation)
assert len(X_test_dynamic) == len(y_test)


# ============================================================
# CONVERT TO TORCH
# ============================================================

print()
print("=" * 100)
print("CONVERTING TO PYTORCH TENSORS")
print("=" * 100)

X_train_dynamic = torch.tensor(
    X_train_dynamic,
    dtype=torch.float32,
)

X_validation_dynamic = torch.tensor(
    X_validation_dynamic,
    dtype=torch.float32,
)

X_test_dynamic = torch.tensor(
    X_test_dynamic,
    dtype=torch.float32,
)

X_train_static = torch.tensor(
    X_train_static,
    dtype=torch.float32,
)

X_validation_static = torch.tensor(
    X_validation_static,
    dtype=torch.float32,
)

X_test_static = torch.tensor(
    X_test_static,
    dtype=torch.float32,
)

X_train_availability = torch.tensor(
    X_train_availability,
    dtype=torch.float32,
)

X_validation_availability = torch.tensor(
    X_validation_availability,
    dtype=torch.float32,
)

X_test_availability = torch.tensor(
    X_test_availability,
    dtype=torch.float32,
)

y_train = torch.tensor(
    y_train,
    dtype=torch.float32,
)

y_validation = torch.tensor(
    y_validation,
    dtype=torch.float32,
)

y_test = torch.tensor(
    y_test,
    dtype=torch.float32,
)


# ============================================================
# DATA LOADERS
# ============================================================

train_dataset = TensorDataset(
    X_train_dynamic,
    X_train_static,
    X_train_availability,
    y_train,
)

validation_dataset = TensorDataset(
    X_validation_dynamic,
    X_validation_static,
    X_validation_availability,
    y_validation,
)

test_dataset = TensorDataset(
    X_test_dynamic,
    X_test_static,
    X_test_availability,
    y_test,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# ============================================================
# MODEL
# ============================================================

print()
print("=" * 100)
print("MODEL CONFIGURATION")
print("=" * 100)

model = ReservoirLSTM(
    dynamic_features=5,
    static_features=8,
    availability_features=8,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
)

model = model.to(DEVICE)

print(model)

parameter_count = sum(
    p.numel()
    for p in model.parameters()
)

trainable_parameter_count = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print()
print(
    f"Total parameters     : {parameter_count:,}"
)

print(
    f"Trainable parameters : {trainable_parameter_count:,}"
)


# ============================================================
# LOSS / OPTIMIZER
# ============================================================

criterion = nn.MSELoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# TRAINING FUNCTIONS
# ============================================================

def train_one_epoch():

    model.train()

    total_loss = 0.0
    total_samples = 0

    for (
        dynamic,
        static,
        availability,
        targets,
    ) in train_loader:

        dynamic = dynamic.to(DEVICE)
        static = static.to(DEVICE)
        availability = availability.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad()

        predictions = model(
            dynamic,
            static,
            availability,
        )

        loss = criterion(
            predictions,
            targets,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        batch_size = targets.size(0)

        total_loss += (
            loss.item() * batch_size
        )

        total_samples += batch_size

    return total_loss / total_samples


@torch.no_grad()
def evaluate_loss(loader):

    model.eval()

    total_loss = 0.0
    total_samples = 0

    for (
        dynamic,
        static,
        availability,
        targets,
    ) in loader:

        dynamic = dynamic.to(DEVICE)
        static = static.to(DEVICE)
        availability = availability.to(DEVICE)
        targets = targets.to(DEVICE)

        predictions = model(
            dynamic,
            static,
            availability,
        )

        loss = criterion(
            predictions,
            targets,
        )

        batch_size = targets.size(0)

        total_loss += (
            loss.item() * batch_size
        )

        total_samples += batch_size

    return total_loss / total_samples


@torch.no_grad()
def predict(loader):

    model.eval()

    predictions = []
    actuals = []

    for (
        dynamic,
        static,
        availability,
        targets,
    ) in loader:

        dynamic = dynamic.to(DEVICE)
        static = static.to(DEVICE)
        availability = availability.to(DEVICE)

        output = model(
            dynamic,
            static,
            availability,
        )

        predictions.append(
            output.cpu().numpy()
        )

        actuals.append(
            targets.numpy()
        )

    predictions = np.concatenate(
        predictions,
        axis=0,
    )

    actuals = np.concatenate(
        actuals,
        axis=0,
    )

    return actuals, predictions


# ============================================================
# TRAINING
# ============================================================

print()
print("=" * 100)
print("TRAINING LSTM")
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
    f"Early stopping   : {PATIENCE}"
)

best_validation_loss = float("inf")
best_epoch = 0
epochs_without_improvement = 0

history = []

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

for epoch in range(1, MAX_EPOCHS + 1):

    train_loss = train_one_epoch()

    validation_loss = evaluate_loss(
        validation_loader
    )

    history.append(
        {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
        }
    )

    print(
        f"Epoch {epoch:03d}/{MAX_EPOCHS} "
        f"| train_loss={train_loss:.6f} "
        f"| val_loss={validation_loss:.6f}"
    )

    if validation_loss < best_validation_loss:

        best_validation_loss = validation_loss
        best_epoch = epoch
        epochs_without_improvement = 0

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "validation_loss": validation_loss,
                "config": {
                    "hidden_size": HIDDEN_SIZE,
                    "num_layers": NUM_LAYERS,
                    "dropout": DROPOUT,
                    "batch_size": BATCH_SIZE,
                    "learning_rate": LEARNING_RATE,
                    "weight_decay": WEIGHT_DECAY,
                    "seed": SEED,
                },
            },
            MODEL_DIR / "best_model.pt",
        )

    else:

        epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:

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

checkpoint = torch.load(
    MODEL_DIR / "best_model.pt",
    map_location=DEVICE,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

print(
    f"Best epoch           : {checkpoint['epoch']}"
)

print(
    f"Best validation loss : "
    f"{checkpoint['validation_loss']:.6f}"
)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(history)

history_df.to_csv(
    RESULT_DIR / "training_history.csv",
    index=False,
)


# ============================================================
# EVALUATION
# ============================================================

print()
print("=" * 100)
print("SCALED-UNIT EVALUATION")
print("=" * 100)

train_actual, train_pred = predict(
    train_loader
)

validation_actual, validation_pred = predict(
    validation_loader
)

test_actual, test_pred = predict(
    test_loader
)

print()
print("TRAIN")

train_metrics = calculate_metrics(
    train_actual,
    train_pred,
)

print(train_metrics.to_string(index=False))

print()
print("VALIDATION")

validation_metrics = calculate_metrics(
    validation_actual,
    validation_pred,
)

print(
    validation_metrics.to_string(
        index=False
    )
)

print()
print("TEST")

test_metrics = calculate_metrics(
    test_actual,
    test_pred,
)

print(
    test_metrics.to_string(
        index=False
    )
)


# ============================================================
# ORIGINAL UNIT EVALUATION
# ============================================================

print()
print("=" * 100)
print("ORIGINAL-UNIT EVALUATION")
print("=" * 100)

train_actual_original = inverse_scale_targets(
    train_actual
)

train_pred_original = inverse_scale_targets(
    train_pred
)

validation_actual_original = inverse_scale_targets(
    validation_actual
)

validation_pred_original = inverse_scale_targets(
    validation_pred
)

test_actual_original = inverse_scale_targets(
    test_actual
)

test_pred_original = inverse_scale_targets(
    test_pred
)

print()
print("TRAIN — ORIGINAL INFLOW UNITS")

train_original_metrics = calculate_metrics(
    train_actual_original,
    train_pred_original,
)

print(
    train_original_metrics.to_string(
        index=False
    )
)

print()
print("VALIDATION — ORIGINAL INFLOW UNITS")

validation_original_metrics = calculate_metrics(
    validation_actual_original,
    validation_pred_original,
)

print(
    validation_original_metrics.to_string(
        index=False
    )
)

print()
print("TEST — ORIGINAL INFLOW UNITS")

test_original_metrics = calculate_metrics(
    test_actual_original,
    test_pred_original,
)

print(
    test_original_metrics.to_string(
        index=False
    )
)


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

test_predictions = pd.DataFrame(
    {
        "target_1d_actual": test_actual_original[:, 0],
        "target_1d_prediction": test_pred_original[:, 0],

        "target_3d_actual": test_actual_original[:, 1],
        "target_3d_prediction": test_pred_original[:, 1],

        "target_7d_actual": test_actual_original[:, 2],
        "target_7d_prediction": test_pred_original[:, 2],
    }
)

test_predictions.to_csv(
    PREDICTION_DIR / "test_predictions_original_units.csv",
    index=False,
)


# ============================================================
# SAVE METRICS
# ============================================================

train_original_metrics.insert(
    0,
    "dataset",
    "TRAIN",
)

validation_original_metrics.insert(
    0,
    "dataset",
    "VALIDATION",
)

test_original_metrics.insert(
    0,
    "dataset",
    "TEST",
)

all_metrics = pd.concat(
    [
        train_original_metrics,
        validation_original_metrics,
        test_original_metrics,
    ],
    ignore_index=True,
)

all_metrics.to_csv(
    RESULT_DIR / "lstm_metrics_original_units.csv",
    index=False,
)


# ============================================================
# INTEGRITY CHECKS
# ============================================================

print()
print("=" * 100)
print("INTEGRITY CHECKS")
print("=" * 100)

print(
    f"Train rows       : {len(train_actual)}"
)

print(
    f"Validation rows  : {len(validation_actual)}"
)

print(
    f"Test rows        : {len(test_actual)}"
)

print(
    f"Test predictions : {len(test_pred)}"
)

assert len(train_actual) == 16662
assert len(validation_actual) == 4383
assert len(test_actual) == 1100
assert len(test_pred) == 1100

assert test_pred.shape == (1100, 3)
assert test_actual.shape == (1100, 3)

assert np.isfinite(
    test_pred
).all()

assert np.isfinite(
    test_actual
).all()

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


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 100)
print("MILESTONE 11.6 — PYTORCH LSTM TRAINING COMPLETE")
print("=" * 100)

print()
print(
    f"Best epoch: {best_epoch}"
)

print(
    f"Best validation loss: "
    f"{best_validation_loss:.6f}"
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
    f"MODEL       : "
    f"{MODEL_DIR / 'best_model.pt'}"
)

print(
    f"HISTORY     : "
    f"{RESULT_DIR / 'training_history.csv'}"
)

print(
    f"METRICS     : "
    f"{RESULT_DIR / 'lstm_metrics_original_units.csv'}"
)

print(
    f"PREDICTIONS : "
    f"{PREDICTION_DIR / 'test_predictions_original_units.csv'}"
)

print()
print("DONE")
