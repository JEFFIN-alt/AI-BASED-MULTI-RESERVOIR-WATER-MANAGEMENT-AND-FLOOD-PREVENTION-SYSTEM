from pathlib import Path

import json
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ============================================================
# MILESTONE 11.6 — LSTM V1 FORECASTING MODEL
# ============================================================

print("=" * 100)
print("MILESTONE 11.6 — LSTM V1 FORECASTING MODEL")
print("=" * 100)


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

BASE = Path("data/processed/lstm")
MODEL_DIR = Path("data/processed/models/lstm_v1")

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

BATCH_SIZE = 64
EPOCHS = 100

LEARNING_RATE = 0.001

LSTM_UNITS = 64
DENSE_UNITS = 32

PATIENCE = 12


# ------------------------------------------------------------
# Load arrays
# ------------------------------------------------------------

print("\n" + "=" * 100)
print("LOADING LSTM DATA")
print("=" * 100)


X_train_dynamic = np.load(
    BASE / "X_train_dynamic.npy"
).astype(np.float32)

X_val_dynamic = np.load(
    BASE / "X_validation_dynamic.npy"
).astype(np.float32)

X_test_dynamic = np.load(
    BASE / "X_test_dynamic.npy"
).astype(np.float32)


X_train_static = np.load(
    BASE / "X_train_static.npy"
).astype(np.float32)

X_val_static = np.load(
    BASE / "X_validation_static.npy"
).astype(np.float32)

X_test_static = np.load(
    BASE / "X_test_static.npy"
).astype(np.float32)


X_train_availability = np.load(
    BASE / "X_train_static_availability.npy"
).astype(np.float32)

X_val_availability = np.load(
    BASE / "X_validation_static_availability.npy"
).astype(np.float32)

X_test_availability = np.load(
    BASE / "X_test_static_availability.npy"
).astype(np.float32)


y_train = np.load(
    BASE / "y_train.npy"
).astype(np.float32)

y_val = np.load(
    BASE / "y_validation.npy"
).astype(np.float32)

y_test = np.load(
    BASE / "y_test.npy"
).astype(np.float32)


# ------------------------------------------------------------
# Shape summary
# ------------------------------------------------------------

print("\nTRAIN")
print("Dynamic:", X_train_dynamic.shape)
print("Static:", X_train_static.shape)
print("Availability:", X_train_availability.shape)
print("Targets:", y_train.shape)

print("\nVALIDATION")
print("Dynamic:", X_val_dynamic.shape)
print("Static:", X_val_static.shape)
print("Availability:", X_val_availability.shape)
print("Targets:", y_val.shape)

print("\nTEST")
print("Dynamic:", X_test_dynamic.shape)
print("Static:", X_test_static.shape)
print("Availability:", X_test_availability.shape)
print("Targets:", y_test.shape)


# ============================================================
# Integrity checks
# ============================================================

print("\n" + "=" * 100)
print("INPUT INTEGRITY CHECKS")
print("=" * 100)


def check_finite(name, array):

    if not np.isfinite(array).all():

        count = np.sum(
            ~np.isfinite(array)
        )

        raise RuntimeError(
            f"{name} contains {count} "
            "NaN/Inf values."
        )

    print(
        f"{name:30s}: PASS"
    )


arrays = {
    "X_train_dynamic": X_train_dynamic,
    "X_val_dynamic": X_val_dynamic,
    "X_test_dynamic": X_test_dynamic,
    "X_train_static": X_train_static,
    "X_val_static": X_val_static,
    "X_test_static": X_test_static,
    "X_train_availability": X_train_availability,
    "X_val_availability": X_val_availability,
    "X_test_availability": X_test_availability,
    "y_train": y_train,
    "y_val": y_val,
    "y_test": y_test,
}


for name, array in arrays.items():

    check_finite(
        name,
        array,
    )


# ------------------------------------------------------------
# Sample-count checks
# ------------------------------------------------------------

if not (
    len(X_train_dynamic)
    == len(X_train_static)
    == len(X_train_availability)
    == len(y_train)
):

    raise RuntimeError(
        "TRAIN sample counts do not match."
    )


if not (
    len(X_val_dynamic)
    == len(X_val_static)
    == len(X_val_availability)
    == len(y_val)
):

    raise RuntimeError(
        "VALIDATION sample counts do not match."
    )


if not (
    len(X_test_dynamic)
    == len(X_test_static)
    == len(X_test_availability)
    == len(y_test)
):

    raise RuntimeError(
        "TEST sample counts do not match."
    )


print(
    "Sample alignment: PASS"
)


# ============================================================
# Model architecture
# ============================================================

print("\n" + "=" * 100)
print("BUILDING LSTM V1")
print("=" * 100)


# ------------------------------------------------------------
# Dynamic temporal branch
# ------------------------------------------------------------

dynamic_input = keras.Input(
    shape=(
        X_train_dynamic.shape[1],
        X_train_dynamic.shape[2],
    ),
    name="dynamic_input",
)


x_dynamic = layers.LSTM(
    LSTM_UNITS,
    return_sequences=False,
    name="lstm",
)(dynamic_input)


x_dynamic = layers.Dense(
    DENSE_UNITS,
    activation="relu",
    name="dynamic_dense",
)(x_dynamic)


# ------------------------------------------------------------
# Static branch
# ------------------------------------------------------------

static_input = keras.Input(
    shape=(
        X_train_static.shape[1],
    ),
    name="static_input",
)


x_static = layers.Dense(
    16,
    activation="relu",
    name="static_dense",
)(static_input)


# ------------------------------------------------------------
# Static availability branch
# ------------------------------------------------------------

availability_input = keras.Input(
    shape=(
        X_train_availability.shape[1],
    ),
    name="availability_input",
)


x_availability = layers.Dense(
    8,
    activation="relu",
    name="availability_dense",
)(availability_input)


# ------------------------------------------------------------
# Merge
# ------------------------------------------------------------

merged = layers.Concatenate(
    name="feature_fusion"
)(
    [
        x_dynamic,
        x_static,
        x_availability,
    ]
)


merged = layers.Dense(
    32,
    activation="relu",
    name="fusion_dense",
)(merged)


merged = layers.Dropout(
    0.20,
    name="dropout",
)(merged)


# ------------------------------------------------------------
# Three forecasting outputs
# ------------------------------------------------------------

output_1d = layers.Dense(
    1,
    name="target_1d",
)(merged)


output_3d = layers.Dense(
    1,
    name="target_3d",
)(merged)


output_7d = layers.Dense(
    1,
    name="target_7d",
)(merged)


model = keras.Model(
    inputs=[
        dynamic_input,
        static_input,
        availability_input,
    ],
    outputs=[
        output_1d,
        output_3d,
        output_7d,
    ],
    name="reservoir_lstm_v1",
)


# ============================================================
# Compile
# ============================================================

optimizer = keras.optimizers.Adam(
    learning_rate=LEARNING_RATE
)


model.compile(
    optimizer=optimizer,
    loss={
        "target_1d": "mse",
        "target_3d": "mse",
        "target_7d": "mse",
    },
    metrics={
        "target_1d": [
            keras.metrics.MeanAbsoluteError(
                name="mae"
            )
        ],
        "target_3d": [
            keras.metrics.MeanAbsoluteError(
                name="mae"
            )
        ],
        "target_7d": [
            keras.metrics.MeanAbsoluteError(
                name="mae"
            )
        ],
    },
)


print("\nModel summary:\n")

model.summary()


# ============================================================
# Callbacks
# ============================================================

best_model_path = (
    MODEL_DIR / "best_lstm_v1.keras"
)


callbacks = [

    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1,
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1,
    ),

    keras.callbacks.ModelCheckpoint(
        filepath=str(best_model_path),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),
]


# ============================================================
# Prepare target dictionary
# ============================================================

train_targets = {
    "target_1d": y_train[:, 0],
    "target_3d": y_train[:, 1],
    "target_7d": y_train[:, 2],
}


val_targets = {
    "target_1d": y_val[:, 0],
    "target_3d": y_val[:, 1],
    "target_7d": y_val[:, 2],
}


# ============================================================
# Train
# ============================================================

print("\n" + "=" * 100)
print("TRAINING LSTM V1")
print("=" * 100)

print(
    f"Epochs: {EPOCHS}"
)

print(
    f"Batch size: {BATCH_SIZE}"
)

print(
    f"Learning rate: {LEARNING_RATE}"
)

print(
    f"LSTM units: {LSTM_UNITS}"
)

print(
    f"Early stopping patience: {PATIENCE}"
)


history = model.fit(

    x=[
        X_train_dynamic,
        X_train_static,
        X_train_availability,
    ],

    y=train_targets,

    validation_data=(
        [
            X_val_dynamic,
            X_val_static,
            X_val_availability,
        ],
        val_targets,
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=callbacks,

    verbose=1,

    shuffle=True,
)


# ============================================================
# Save final model
# ============================================================

final_model_path = (
    MODEL_DIR / "final_lstm_v1.keras"
)

model.save(
    final_model_path
)


# ============================================================
# Save training history
# ============================================================

history_path = (
    MODEL_DIR / "training_history.json"
)


history_serializable = {
    key: [
        float(value)
        for value in values
    ]
    for key, values
    in history.history.items()
}


with open(
    history_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        history_serializable,
        f,
        indent=2,
    )


# ============================================================
# Training summary
# ============================================================

best_epoch = int(
    np.argmin(
        history.history["val_loss"]
    )
) + 1


best_val_loss = float(
    np.min(
        history.history["val_loss"]
    )
)


final_train_loss = float(
    history.history["loss"][-1]
)


final_val_loss = float(
    history.history["val_loss"][-1]
)


print("\n" + "=" * 100)
print("TRAINING SUMMARY")
print("=" * 100)

print(
    "Epochs completed:",
    len(history.history["loss"]),
)

print(
    "Best epoch:",
    best_epoch,
)

print(
    "Best validation loss:",
    best_val_loss,
)

print(
    "Final training loss:",
    final_train_loss,
)

print(
    "Final validation loss:",
    final_val_loss,
)


# ============================================================
# Final integrity
# ============================================================

print("\n" + "=" * 100)
print("MILESTONE 11.6 COMPLETE")
print("=" * 100)

print("\nSaved:")

print(
    "BEST MODEL:",
    best_model_path,
)

print(
    "FINAL MODEL:",
    final_model_path,
)

print(
    "HISTORY:",
    history_path,
)

print("\nDONE")
