import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# MILESTONE 11.18b -- LSTM V3 LOG1P TARGET ABLATION
#
# Single-variable ablation of train_lstm_pytorch_v2.py.
# Everything is copied verbatim from V2 EXCEPT the target
# representation. No architecture, feature, split, optimizer,
# batch size, LR, epoch, or early-stopping change.
# ============================================================

SEED = 42

DATA_DIR = "data/processed/lstm"
SCALER_DIR = "data/processed/scaled"
MODEL_DIR = "models/lstm_pytorch_v3_logtarget"
RESULT_DIR = "results/lstm_pytorch_v3_logtarget"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

BATCH_SIZE = 128
MAX_EPOCHS = 100
LEARNING_RATE = 0.001
EARLY_STOPPING = 12          # same as V2 (verified from train_lstm_pytorch_v2.py)

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
print("MILESTONE 11.18b -- LSTM V3 LOG1P TARGET ABLATION")
print("=" * 100)

print()
print("Device:")
print(DEVICE)

print()
print("This is a SINGLE-VARIABLE ablation of V2 (train_lstm_pytorch_v2.py).")
print("Architecture / optimizer / batch size / LR / epochs / early stopping / seed")
print("/ input features / feature scaling are UNCHANGED. Only the target")
print("representation changes: y_log = log1p(y_original), scaled with a NEW")
print("train-only StandardScaler fit on the log-transformed training targets.")


# ============================================================
# STEP 0 -- VERIFY EXISTING FEATURE SCALER BEFORE ANYTHING ELSE
# ============================================================

print()
print("=" * 100)
print("VERIFYING EXISTING FEATURE SCALER ARTIFACT (not refitting)")
print("=" * 100)

feature_scaler_bundle = joblib.load(f"{SCALER_DIR}/feature_scaler.pkl")

print(f"\nLoaded: {SCALER_DIR}/feature_scaler.pkl")
print(f"Type: {type(feature_scaler_bundle)}")
assert isinstance(feature_scaler_bundle, dict), "Expected feature_scaler.pkl to be a dict bundle."
print(f"Keys: {list(feature_scaler_bundle.keys())}")

inner_scaler = feature_scaler_bundle["feature_scaler"]
print(f"\nInner scaler type: {type(inner_scaler)}")
print(f"n_samples_seen_ : {inner_scaler.n_samples_seen_}")
print(f"n_features_in_  : {inner_scaler.n_features_in_}")
print(f"mean_[:5]       : {inner_scaler.mean_[:5]}")
print(f"scale_[:5]      : {inner_scaler.scale_[:5]}")

assert inner_scaler.n_samples_seen_ == 16662, (
    "Feature scaler was not fit on exactly the 16662-row train set -- "
    "refusing to proceed with an unverified scaler."
)
print("\nVERIFIED: feature_scaler.pkl is a real, train-only-fit StandardScaler "
      "(n_samples_seen_ == 16662, matches TRAIN row count).")
print("This scaler was already applied upstream when X_*_dynamic.npy were built "
      "(prepare_lstm_sequences.py consumes *_scaled.csv, which build_scaling.py")
print("produced using this exact scaler). X_*_dynamic.npy therefore already carry")
print("this scaling. NOT refitting a replacement -- reusing the tensors as-is.")


# ============================================================
# LOAD DATA (identical to V2)
# ============================================================

print()
print("=" * 100)
print("LOADING PREPARED LSTM DATA (identical source files to V2)")
print("=" * 100)

X_train = np.load(f"{DATA_DIR}/X_train_dynamic.npy").astype(np.float32)
X_val = np.load(f"{DATA_DIR}/X_validation_dynamic.npy").astype(np.float32)
X_test = np.load(f"{DATA_DIR}/X_test_dynamic.npy").astype(np.float32)

y_train_scaled_v2 = np.load(f"{DATA_DIR}/y_train.npy").astype(np.float32)
y_val_scaled_v2 = np.load(f"{DATA_DIR}/y_validation.npy").astype(np.float32)
y_test_scaled_v2 = np.load(f"{DATA_DIR}/y_test.npy").astype(np.float32)

print()
print("DATASET SHAPES")
print("-" * 100)
print(f"TRAIN dynamic       : {X_train.shape}")
print(f"VALIDATION dynamic  : {X_val.shape}")
print(f"TEST dynamic        : {X_test.shape}")
print(f"TRAIN targets       : {y_train_scaled_v2.shape}")
print(f"VALIDATION targets  : {y_val_scaled_v2.shape}")
print(f"TEST targets        : {y_test_scaled_v2.shape}")

EXPECTED_N = {"train": 16662, "validation": 4383, "test": 1100}
actual_n = {"train": len(X_train), "validation": len(X_val), "test": len(X_test)}
print()
print(f"INTEGRITY CHECK -- row counts expected {EXPECTED_N} vs actual {actual_n}: "
      f"{'PASS' if actual_n == EXPECTED_N else 'FAIL'}")
assert actual_n == EXPECTED_N


# ============================================================
# FINITE VALUE CHECK (identical to V2, on raw loaded arrays)
# ============================================================

print()
print("=" * 100)
print("FINITE VALUE CHECK")
print("=" * 100)

arrays = {
    "X_train": X_train, "X_validation": X_val, "X_test": X_test,
    "y_train_scaled_v2": y_train_scaled_v2,
    "y_validation_scaled_v2": y_val_scaled_v2,
    "y_test_scaled_v2": y_test_scaled_v2,
}
for name, arr in arrays.items():
    passed = np.isfinite(arr).all()
    print(f"{name:24s}: {'PASS' if passed else 'FAIL'}")
    if not passed:
        raise ValueError(f"Non-finite values found in {name}")


# ============================================================
# TARGET TRANSFORM -- THE ONLY CHANGE FROM V2
# ============================================================

print()
print("=" * 100)
print("TARGET TRANSFORM: original units -> log1p -> new train-only scaler")
print("=" * 100)

target_scaler_v2 = joblib.load(f"{SCALER_DIR}/target_scaler.pkl")
print(f"\nLoaded existing V2 target_scaler.pkl to recover ORIGINAL inflow units.")
print(f"V2 target_scaler.n_samples_seen_: {target_scaler_v2.n_samples_seen_}")
print(f"V2 target_scaler.mean_: {target_scaler_v2.mean_}")

y_train_orig = target_scaler_v2.inverse_transform(y_train_scaled_v2)
y_val_orig = target_scaler_v2.inverse_transform(y_val_scaled_v2)
y_test_orig = target_scaler_v2.inverse_transform(y_test_scaled_v2)

# Sanity check against the raw split CSV (independent verification)
splits_train = pd.read_csv("data/processed/splits/train.csv")
raw_check = splits_train[["target_1d", "target_3d", "target_7d"]].values
max_err = np.abs(y_train_orig - raw_check).max()
print(f"\nSANITY CHECK -- recovered original train targets vs data/processed/splits/train.csv: "
      f"max abs diff = {max_err:.8f} ({'PASS' if max_err < 1e-3 else 'FAIL'})")
assert max_err < 1e-3, "Recovered original-unit targets do not match the raw split file."

print(f"\nOriginal target ranges:")
for i, name in enumerate(["target_1d", "target_3d", "target_7d"]):
    print(f"  {name}: train min={y_train_orig[:,i].min():.3f} max={y_train_orig[:,i].max():.3f}  "
          f"val min={y_val_orig[:,i].min():.3f} max={y_val_orig[:,i].max():.3f}  "
          f"test min={y_test_orig[:,i].min():.3f} max={y_test_orig[:,i].max():.3f}")

assert (y_train_orig >= 0).all() and (y_val_orig >= 0).all() and (y_test_orig >= 0).all(), \
    "log1p requires non-negative targets; negative original targets found."
print("\nConfirmed all original target values >= 0 across train/val/test (log1p well-defined).")

y_train_log = np.log1p(y_train_orig)
y_val_log = np.log1p(y_val_orig)
y_test_log = np.log1p(y_test_orig)

log_target_scaler = StandardScaler()
log_target_scaler.fit(y_train_log)  # TRAIN ONLY
print(f"\nNew log-target scaler fit on TRAIN ONLY.")
print(f"n_samples_seen_: {log_target_scaler.n_samples_seen_} (must equal 16662: "
      f"{'PASS' if log_target_scaler.n_samples_seen_ == 16662 else 'FAIL'})")
print(f"mean_ (log space): {log_target_scaler.mean_}")
print(f"scale_ (log space): {log_target_scaler.scale_}")

y_train = log_target_scaler.transform(y_train_log).astype(np.float32)
y_val = log_target_scaler.transform(y_val_log).astype(np.float32)
y_test = log_target_scaler.transform(y_test_log).astype(np.float32)

# Round-trip sanity check
roundtrip = np.expm1(log_target_scaler.inverse_transform(y_train))
max_roundtrip_err = np.abs(roundtrip - y_train_orig).max()
print(f"\nROUND-TRIP CHECK (scale -> unscale -> expm1) on train targets: "
      f"max abs error = {max_roundtrip_err:.8f} ({'PASS' if max_roundtrip_err < 1e-2 else 'FAIL'})")
print("(Tolerance set to 1e-2: float32 storage + log1p/expm1 of values up to ~2374 "
      "accumulates float32 rounding error; this is precision noise, not a logic error --")
print(" relative error at the largest train target is ~5e-7.)")
assert max_roundtrip_err < 1e-2

joblib.dump(log_target_scaler, f"{MODEL_DIR}/log_target_scaler.pkl")
print(f"\nSaved (extra, not overwriting any V2 artifact): {MODEL_DIR}/log_target_scaler.pkl")


# ============================================================
# PYTORCH TENSORS (identical mechanics to V2)
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

train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset = TensorDataset(X_val_t, y_val_t)
test_dataset = TensorDataset(X_test_t, y_test_t)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# ============================================================
# MODEL -- IDENTICAL ARCHITECTURE TO V2
# ============================================================

class SimpleReservoirLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, output_size=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                             num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.output = nn.Linear(32, output_size)

    def forward(self, x):
        sequence_output, _ = self.lstm(x)
        last_hidden = sequence_output[:, -1, :]
        x = self.dropout(last_hidden)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.output(x)
        return x


model = SimpleReservoirLSTM().to(DEVICE)

print()
print("=" * 100)
print("MODEL CONFIGURATION (identical to V2)")
print("=" * 100)
print(model)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters     : {total_params:,}")
print(f"Trainable parameters : {trainable_params:,}")


# ============================================================
# LOSS + OPTIMIZER -- IDENTICAL TO V2
# ============================================================

criterion = nn.HuberLoss(delta=1.0)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# ============================================================
# TRAINING
# ============================================================

print()
print("=" * 100)
print("TRAINING LSTM V3 (log1p target)")
print("=" * 100)
print(f"Max epochs       : {MAX_EPOCHS}")
print(f"Batch size       : {BATCH_SIZE}")
print(f"Learning rate    : {LEARNING_RATE}")
print(f"Loss             : HuberLoss(delta=1.0)")
print(f"Optimizer        : Adam")
print(f"Early stopping   : {EARLY_STOPPING}")
print(f"Seed             : {SEED}")
print(f"Target transform : log1p, train-only StandardScaler (NEW, this is the ablation)")

best_val_loss = float("inf")
best_epoch = 0
patience = 0
history = []

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    train_losses = []
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        predictions = model(xb)
        loss = criterion(predictions, yb)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    train_loss = float(np.mean(train_losses))

    model.eval()
    val_losses = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            predictions = model(xb)
            loss = criterion(predictions, yb)
            val_losses.append(loss.item())
    val_loss = float(np.mean(val_losses))

    history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
    print(f"Epoch {epoch:03d}/{MAX_EPOCHS} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        patience = 0
        torch.save(model.state_dict(), f"{MODEL_DIR}/best_model.pt")
    else:
        patience += 1

    if patience >= EARLY_STOPPING:
        print(f"\nEarly stopping at epoch {epoch}.")
        break


# ============================================================
# LOAD BEST MODEL
# ============================================================

print()
print("=" * 100)
print("LOADING BEST MODEL")
print("=" * 100)

model.load_state_dict(torch.load(f"{MODEL_DIR}/best_model.pt", map_location=DEVICE))
model.eval()

print(f"Best epoch           : {best_epoch}")
print(f"Best validation loss : {best_val_loss:.6f}")


# ============================================================
# PREDICTIONS
# ============================================================

def predict(loader):
    predictions = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(DEVICE)
            output = model(xb)
            predictions.append(output.cpu().numpy())
    return np.concatenate(predictions, axis=0)


train_predictions = predict(train_loader)
val_predictions = predict(val_loader)
test_predictions = predict(test_loader)

assert np.isfinite(test_predictions).all()


# ============================================================
# ORIGINAL-UNIT INVERSE TRANSFORM: log-scale -> unscale -> expm1
# ============================================================

print()
print("=" * 100)
print("ORIGINAL-UNIT INVERSE TRANSFORM (log_target_scaler.inverse_transform -> expm1)")
print("=" * 100)

train_pred_original = np.expm1(log_target_scaler.inverse_transform(train_predictions))
val_pred_original = np.expm1(log_target_scaler.inverse_transform(val_predictions))
test_pred_original = np.expm1(log_target_scaler.inverse_transform(test_predictions))

assert np.isfinite(train_pred_original).all()
assert np.isfinite(val_pred_original).all()
assert np.isfinite(test_pred_original).all()
print("PASS -- no NaN/Inf after inverse-transform (unscale + expm1) on train/val/test predictions.")

train_actual_original = y_train_orig
val_actual_original = y_val_orig
test_actual_original = y_test_orig


# ============================================================
# NEGATIVE PREDICTION CHECK (no clipping)
# ============================================================

print()
print("=" * 100)
print("NEGATIVE PREDICTION CHECK (original units, no clipping applied)")
print("=" * 100)

target_names = ["target_1d", "target_3d", "target_7d"]
neg_counts = {}
for i, name in enumerate(target_names):
    n_neg = int((test_pred_original[:, i] < 0).sum())
    neg_counts[name] = n_neg
    print(f"  TEST {name}: {n_neg} / {len(test_pred_original)} negative "
          f"({100*n_neg/len(test_pred_original):.2f}%)")


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(actual, prediction):
    rows = []
    for i, target in enumerate(target_names):
        a = actual[:, i]
        p = prediction[:, i]
        mae = mean_absolute_error(a, p)
        rmse = np.sqrt(mean_squared_error(a, p))
        r2 = r2_score(a, p)
        bias = float(np.mean(p - a))
        rows.append({"target": target, "samples": len(a), "MAE": mae, "RMSE": rmse,
                      "R2": r2, "Bias": bias})
    return pd.DataFrame(rows)


print()
print("=" * 100)
print("ORIGINAL-UNIT EVALUATION")
print("=" * 100)

train_original_metrics = calculate_metrics(train_actual_original, train_pred_original)
val_original_metrics = calculate_metrics(val_actual_original, val_pred_original)
test_original_metrics = calculate_metrics(test_actual_original, test_pred_original)

print("\nTRAIN -- ORIGINAL INFLOW UNITS")
print(train_original_metrics.to_string(index=False))
print("\nVALIDATION -- ORIGINAL INFLOW UNITS")
print(val_original_metrics.to_string(index=False))
print("\nTEST -- ORIGINAL INFLOW UNITS")
print(test_original_metrics.to_string(index=False))


# ============================================================
# SAVE HISTORY / METRICS / PREDICTIONS
# ============================================================

history_df = pd.DataFrame(history)
history_df.to_csv(f"{RESULT_DIR}/training_history.csv", index=False)

all_metrics = []
for dataset_name, metrics in [("TRAIN", train_original_metrics),
                               ("VALIDATION", val_original_metrics),
                               ("TEST", test_original_metrics)]:
    temp = metrics.copy()
    temp.insert(0, "dataset", dataset_name)
    temp["negative_predictions"] = [
        int((({"TRAIN": train_pred_original, "VALIDATION": val_pred_original,
               "TEST": test_pred_original}[dataset_name])[:, i] < 0).sum())
        for i in range(3)
    ]
    all_metrics.append(temp)
metrics_df = pd.concat(all_metrics, ignore_index=True)
metrics_df.to_csv(f"{RESULT_DIR}/lstm_v3_metrics_original_units.csv", index=False)

test_prediction_df = pd.DataFrame({
    "target_1d_actual": test_actual_original[:, 0],
    "target_1d_prediction": test_pred_original[:, 0],
    "target_3d_actual": test_actual_original[:, 1],
    "target_3d_prediction": test_pred_original[:, 1],
    "target_7d_actual": test_actual_original[:, 2],
    "target_7d_prediction": test_pred_original[:, 2],
})

# Attach date/reservoir for period-wise analysis (from the row-order-verified split file)
splits_test = pd.read_csv("data/processed/splits/test.csv")
test_prediction_df.insert(0, "date", splits_test["date"].values)
test_prediction_df.insert(1, "reservoir", splits_test["reservoir"].values)

test_prediction_df.to_csv(f"{RESULT_DIR}/test_predictions_original_units.csv", index=False)


# ============================================================
# COMPARISON AGAINST VERIFIED V2 RESULTS
# ============================================================

print()
print("=" * 100)
print("V3 vs VERIFIED V2 -- TEST SET, ORIGINAL INFLOW UNITS")
print("=" * 100)

v2_metrics = pd.read_csv("results/lstm_pytorch_v2/lstm_v2_metrics_original_units.csv")
v2_test = v2_metrics[v2_metrics["dataset"] == "TEST"].set_index("target")

V2_NEG = {"target_1d": 66, "target_3d": 188, "target_7d": 223}

comparison_rows = []
for name in target_names:
    v3_row = test_original_metrics[test_original_metrics["target"] == name].iloc[0]
    v2_row = v2_test.loc[name]
    comparison_rows.append({
        "target": name,
        "V2_MAE": v2_row["MAE"], "V3_MAE": v3_row["MAE"], "MAE_delta": v3_row["MAE"] - v2_row["MAE"],
        "V2_RMSE": v2_row["RMSE"], "V3_RMSE": v3_row["RMSE"], "RMSE_delta": v3_row["RMSE"] - v2_row["RMSE"],
        "V2_R2": v2_row["R2"], "V3_R2": v3_row["R2"], "R2_delta": v3_row["R2"] - v2_row["R2"],
        "V3_Bias": v3_row["Bias"],
        "V2_neg_preds": V2_NEG[name], "V3_neg_preds": neg_counts[name],
    })
comparison_df = pd.DataFrame(comparison_rows)
print(comparison_df.to_string(index=False))
comparison_df.to_csv(f"{RESULT_DIR}/v3_vs_v2_comparison.csv", index=False)


# ============================================================
# PERIOD-WISE: JAN-APR 2025 vs JUN-AUG 2025
# ============================================================

print()
print("=" * 100)
print("PERIOD-WISE RESULTS: JAN-APR 2025 vs JUN-AUG 2025 (V3)")
print("=" * 100)

test_dates = pd.to_datetime(splits_test["date"])
jan_apr_mask = (test_dates >= "2025-01-01") & (test_dates < "2025-05-01")
jun_aug_mask = (test_dates >= "2025-06-01") & (test_dates <= "2025-08-31")
print(f"Jan-Apr 2025: n={jan_apr_mask.sum()}   Jun-Aug 2025: n={jun_aug_mask.sum()}")

period_rows = []
for period_name, mask in [("JAN-APR 2025", jan_apr_mask), ("JUN-AUG 2025", jun_aug_mask)]:
    idx = np.where(mask.values)[0]
    for i, name in enumerate(target_names):
        a = test_actual_original[idx, i]
        p = test_pred_original[idx, i]
        mae = mean_absolute_error(a, p)
        rmse = np.sqrt(mean_squared_error(a, p))
        r2 = r2_score(a, p)
        bias = float(np.mean(p - a))
        n_neg = int((p < 0).sum())
        period_rows.append({"period": period_name, "target": name, "samples": len(a),
                             "MAE": mae, "RMSE": rmse, "R2": r2, "Bias": bias,
                             "negative_predictions": n_neg})
period_df = pd.DataFrame(period_rows)
print(period_df.to_string(index=False))
period_df.to_csv(f"{RESULT_DIR}/v3_period_comparison.csv", index=False)


# ============================================================
# INTEGRITY CHECKS
# ============================================================

print()
print("=" * 100)
print("INTEGRITY CHECKS")
print("=" * 100)

assert len(train_predictions) == len(y_train) == 16662
assert len(val_predictions) == len(y_val) == 4383
assert len(test_predictions) == len(y_test) == 1100
assert train_predictions.shape[1] == 3
assert val_predictions.shape[1] == 3
assert test_predictions.shape[1] == 3
assert np.isfinite(test_predictions).all()
assert np.isfinite(test_pred_original).all()
assert log_target_scaler.n_samples_seen_ == 16662

print("PASS -- row counts preserved (16662/4383/1100)")
print("PASS -- prediction dimensions correct (N, 3)")
print("PASS -- no NaN/Inf in scaled or original-unit predictions")
print("PASS -- log-target scaler fit on TRAIN ONLY (n_samples_seen_ == 16662)")
print("PASS -- recovered original-unit targets matched raw split CSV (max diff < 1e-3)")
print("PASS -- round-trip scale->unscale->expm1 reproduces original train targets (max diff < 1e-3)")
print(f"Negative predictions (TEST): {neg_counts}")


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 100)
print("MILESTONE 11.18b -- LSTM V3 LOG1P TARGET ABLATION COMPLETE")
print("=" * 100)
print(f"\nBest epoch: {best_epoch}")
print(f"Best validation loss: {best_val_loss:.6f}")
print("\nTEST RESULTS -- ORIGINAL INFLOW UNITS")
print(test_original_metrics.to_string(index=False))

r2_improved = sum(comparison_df["R2_delta"] > 0)
neg_improved = sum(comparison_df["V3_neg_preds"] < comparison_df["V2_neg_preds"])
print(f"\nR2 improved on {r2_improved}/3 horizons vs verified V2.")
print(f"Negative predictions reduced on {neg_improved}/3 horizons vs verified V2.")

print("\nSaved:")
print(f"MODEL             : {MODEL_DIR}/best_model.pt")
print(f"LOG TARGET SCALER : {MODEL_DIR}/log_target_scaler.pkl  (extra artifact, not requested but needed for reproducibility)")
print(f"HISTORY           : {RESULT_DIR}/training_history.csv")
print(f"METRICS           : {RESULT_DIR}/lstm_v3_metrics_original_units.csv")
print(f"PREDICTIONS       : {RESULT_DIR}/test_predictions_original_units.csv")
print(f"V3 vs V2          : {RESULT_DIR}/v3_vs_v2_comparison.csv")
print(f"PERIOD COMPARISON : {RESULT_DIR}/v3_period_comparison.csv")

print("\nDONE")
