import os
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch_geometric.nn import GCNConv

# ============================================================
# CONFIGURATIONS
# ============================================================
SEED = 42
BATCH_SIZE = 32
MAX_EPOCHS = 100
LEARNING_RATE = 0.001
EARLY_STOPPING = 12
DEVICE = torch.device("cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

PROJECT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT / "models" / "gcn_lstm_v1_1_identity"
RESULT_DIR = PROJECT / "results" / "gcn_lstm_v1_1_identity"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 100)
print("IMPLEMENTING GCN-LSTM V1.1 (IDENTITY GRAPH)")
print("=" * 100)

def fail(msg):
    print(f"FAIL: {msg}")
    sys.exit(1)

# Load metadata
train_meta = pd.read_csv(PROJECT / "data/processed/splits/train.csv")
val_meta = pd.read_csv(PROJECT / "data/processed/splits/validation.csv")
test_meta = pd.read_csv(PROJECT / "data/processed/splits/test.csv")

res_train = sorted(train_meta["reservoir"].unique())
NODE_ORDER = res_train
res_to_idx = {r: i for i, r in enumerate(NODE_ORDER)}

# ============================================================
# ABLATION: IDENTITY GRAPH CONSTRUCTION
# ============================================================
print("\n--- IDENTITY GRAPH VERIFICATION ---")
# Create self-loops for all 16 reservoirs
edge_index_np = np.array([[i, i] for i in range(16)], dtype=np.int64).T

# Verifications
off_diagonal = False
self_loops_count = 0
for u, v in edge_index_np.T:
    if u != v:
        off_diagonal = True
    else:
        self_loops_count += 1

if off_diagonal:
    fail("Graph contains off-diagonal edges!")
if self_loops_count != 16 or edge_index_np.shape[1] != 16:
    fail(f"Expected exactly 16 self-loops, found {self_loops_count}")

print("- The graph contains only self-loops")
print("- No off-diagonal edges exist")
print("- All 16 reservoirs have exactly one self-loop")
print("- Preprocessing is identical to V1")

# ============================================================
# DATA PREPARATION (IDENTICAL TO V1)
# ============================================================
train_dates = sorted(train_meta["date"].unique())
val_dates = sorted(val_meta["date"].unique())
test_dates = sorted(test_meta["date"].unique())

def pivot_data(meta, X_flat, y_flat, dates):
    n_dates = len(dates)
    X_pivot = np.zeros((n_dates, 16, 7, 5), dtype=np.float32)
    y_pivot = np.zeros((n_dates, 16, 3), dtype=np.float32)
    mask_pivot = np.zeros((n_dates, 16), dtype=bool)
    date_to_idx = {d: i for i, d in enumerate(dates)}
    
    for idx, row in meta.iterrows():
        d_idx = date_to_idx[row["date"]]
        r_idx = res_to_idx[row["reservoir"]]
        X_pivot[d_idx, r_idx] = X_flat[idx]
        y_pivot[d_idx, r_idx] = y_flat[idx]
        mask_pivot[d_idx, r_idx] = True
    return X_pivot, y_pivot, mask_pivot

X_train_flat = np.load(PROJECT / "data/processed/lstm/X_train_dynamic.npy").astype(np.float32)
y_train_flat = np.load(PROJECT / "data/processed/lstm/y_train.npy").astype(np.float32)
X_val_flat = np.load(PROJECT / "data/processed/lstm/X_validation_dynamic.npy").astype(np.float32)
y_val_flat = np.load(PROJECT / "data/processed/lstm/y_validation.npy").astype(np.float32)
X_test_flat = np.load(PROJECT / "data/processed/lstm/X_test_dynamic.npy").astype(np.float32)
y_test_flat = np.load(PROJECT / "data/processed/lstm/y_test.npy").astype(np.float32)

target_scaler = joblib.load(PROJECT / "data/processed/scaled/target_scaler.pkl")
log_target_scaler = joblib.load(PROJECT / "models/lstm_pytorch_v3_logtarget/log_target_scaler.pkl")

# Reconstruct original targets from scaled
y_tr_orig = target_scaler.inverse_transform(y_train_flat)
y_va_orig = target_scaler.inverse_transform(y_val_flat)
y_te_orig = target_scaler.inverse_transform(y_test_flat)

y_tr_log = np.log1p(y_tr_orig)
y_va_log = np.log1p(y_va_orig)
y_te_log = np.log1p(y_te_orig)

y_tr_scaled = log_target_scaler.transform(y_tr_log).astype(np.float32)
y_va_scaled = log_target_scaler.transform(y_va_log).astype(np.float32)
y_te_scaled = log_target_scaler.transform(y_te_log).astype(np.float32)

X_tr, y_tr, m_tr = pivot_data(train_meta, X_train_flat, y_tr_scaled, train_dates)
X_va, y_va, m_va = pivot_data(val_meta, X_val_flat, y_va_scaled, val_dates)
X_te, y_te, m_te = pivot_data(test_meta, X_test_flat, y_te_scaled, test_dates)

X_tr_t = torch.from_numpy(X_tr)
y_tr_t = torch.from_numpy(y_tr)
m_tr_t = torch.from_numpy(m_tr)

X_va_t = torch.from_numpy(X_va)
y_va_t = torch.from_numpy(y_va)
m_va_t = torch.from_numpy(m_va)

X_te_t = torch.from_numpy(X_te)
y_te_t = torch.from_numpy(y_te)
m_te_t = torch.from_numpy(m_te)

edge_index_t = torch.from_numpy(edge_index_np).long().to(DEVICE)

train_dataset = TensorDataset(X_tr_t, y_tr_t, m_tr_t)
val_dataset = TensorDataset(X_va_t, y_va_t, m_va_t)
test_dataset = TensorDataset(X_te_t, y_te_t, m_te_t)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ============================================================
# MODEL & LOSS (IDENTICAL TO V1)
# ============================================================
class GCNLSTM(nn.Module):
    def __init__(self, edge_index):
        super().__init__()
        self.edge_index = edge_index
        # shared GCNConv(5,32), ReLU
        self.gcn = GCNConv(5, 32, add_self_loops=True, normalize=True)
        self.relu1 = nn.ReLU()
        # per-node LSTM(32,64,1 layer)
        self.lstm = nn.LSTM(input_size=32, hidden_size=64, num_layers=1, batch_first=True)
        # Linear(64,32)
        self.fc1 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        # Linear(32,3)
        self.output = nn.Linear(32, 3)
        
    def forward(self, x):
        B, N, T, F = x.shape
        x_gcn_in = x.permute(0, 2, 1, 3)
        gcn_out = self.gcn(x_gcn_in, self.edge_index)
        gcn_out = self.relu1(gcn_out)
        lstm_in = gcn_out.permute(0, 2, 1, 3).contiguous()
        lstm_in = lstm_in.view(B * N, T, 32)
        seq_out, _ = self.lstm(lstm_in)
        last_hidden = seq_out[:, -1, :]
        out = self.fc1(last_hidden)
        out = self.relu2(out)
        out = self.output(out)
        return out.view(B, N, 3)

model = GCNLSTM(edge_index=edge_index_t).to(DEVICE)

class MaskedHuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.criterion = nn.HuberLoss(delta=delta, reduction='none')
        
    def forward(self, pred, target, mask):
        loss = self.criterion(pred, target)
        mask_expanded = mask.unsqueeze(-1).expand_as(loss)
        masked_loss = loss * mask_expanded
        num_valid = mask_expanded.sum()
        if num_valid > 0:
            return masked_loss.sum() / num_valid
        else:
            return masked_loss.sum() * 0.0

criterion = MaskedHuberLoss(delta=1.0)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ============================================================
# TRAINING (IDENTICAL TO V1)
# ============================================================
best_val_loss = float("inf")
patience = 0

print("\n--- STARTING TRAINING ---")
for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    train_losses = []
    for xb, yb, mb in train_loader:
        xb, yb, mb = xb.to(DEVICE), yb.to(DEVICE), mb.to(DEVICE)
        optimizer.zero_grad()
        predictions = model(xb)
        loss = criterion(predictions, yb, mb)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    
    val_losses = []
    model.eval()
    with torch.no_grad():
        for xb, yb, mb in val_loader:
            xb, yb, mb = xb.to(DEVICE), yb.to(DEVICE), mb.to(DEVICE)
            predictions = model(xb)
            loss = criterion(predictions, yb, mb)
            val_losses.append(loss.item())
            
    t_loss = np.mean(train_losses)
    v_loss = np.mean(val_losses)
    
    if v_loss < best_val_loss:
        best_val_loss = v_loss
        patience = 0
        torch.save(model.state_dict(), MODEL_DIR / "best_model.pt")
    else:
        patience += 1
        
    if patience >= EARLY_STOPPING:
        print(f"Early stopping at epoch {epoch}")
        break

# ============================================================
# EVALUATION (IDENTICAL TO V1)
# ============================================================
model.load_state_dict(torch.load(MODEL_DIR / "best_model.pt"))
model.eval()

def predict(loader):
    preds, actuals, masks = [], [], []
    with torch.no_grad():
        for xb, yb, mb in loader:
            xb = xb.to(DEVICE)
            out = model(xb)
            preds.append(out.cpu().numpy())
            actuals.append(yb.numpy())
            masks.append(mb.numpy())
            
    preds = np.concatenate(preds, axis=0)
    actuals = np.concatenate(actuals, axis=0)
    masks = np.concatenate(masks, axis=0)
    
    valid_preds = preds[masks]
    valid_actuals = actuals[masks]
    return valid_preds, valid_actuals

test_preds, test_actuals = predict(test_loader)
test_preds_orig = np.expm1(log_target_scaler.inverse_transform(test_preds))
test_actuals_orig = np.expm1(log_target_scaler.inverse_transform(test_actuals))

target_names = ["target_1d", "target_3d", "target_7d"]
metrics_rows = []
for i, name in enumerate(target_names):
    p = test_preds_orig[:, i]
    a = test_actuals_orig[:, i]
    metrics_rows.append({
        "Model": "V1.1",
        "Horizon": name,
        "MAE": float(mean_absolute_error(a, p)),
        "RMSE": float(np.sqrt(mean_squared_error(a, p))),
        "R2": float(r2_score(a, p)),
        "Bias": float(np.mean(p - a)),
        "Negative Predictions": int((p < 0).sum())
    })

metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv(RESULT_DIR / "gcn_lstm_v1_1_identity_metrics.csv", index=False)

# Period-wise comparison
test_dates_full = pd.to_datetime(test_meta['date'])
jan_apr_mask = (test_dates_full >= "2025-01-01") & (test_dates_full < "2025-05-01")
jun_aug_mask = (test_dates_full >= "2025-06-01") & (test_dates_full <= "2025-08-31")

period_rows = []
for period, p_mask in [("Jan-Apr 2025", jan_apr_mask), ("Jun-Aug 2025", jun_aug_mask)]:
    if p_mask.sum() == 0: continue
    for i, name in enumerate(target_names):
        p = test_preds_orig[p_mask, i]
        a = test_actuals_orig[p_mask, i]
        period_rows.append({
            "period": period,
            "target": name,
            "MAE": float(mean_absolute_error(a, p)),
            "RMSE": float(np.sqrt(mean_squared_error(a, p))),
            "R2": float(r2_score(a, p)),
            "Bias": float(np.mean(p - a))
        })
        
period_df = pd.DataFrame(period_rows)
period_df.to_csv(RESULT_DIR / "gcn_lstm_v1_1_identity_period_metrics.csv", index=False)

print("\nMILESTONE 12.5 COMPLETE")
print("\n" + metrics_df.to_string(index=False))

# Calculate recovery outcome
v3_r2 = {"target_1d": 0.759, "target_3d": 0.654, "target_7d": 0.501}
v1_r2 = {"target_1d": -0.706, "target_3d": -0.802, "target_7d": -0.487}

print("\n--- DIRECT COMPARISON ---")
for i, name in enumerate(target_names):
    m_r2 = metrics_df.iloc[i]["R2"]
    print(f"{name} R2 -> V3: {v3_r2[name]:.3f} | V1.1: {m_r2:.3f} | V1: {v1_r2[name]:.3f}")
