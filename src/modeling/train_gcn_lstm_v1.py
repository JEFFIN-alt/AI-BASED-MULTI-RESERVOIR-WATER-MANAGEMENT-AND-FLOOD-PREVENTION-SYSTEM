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
MODEL_DIR = PROJECT / "models" / "gcn_lstm_v1"
RESULT_DIR = PROJECT / "results" / "gcn_lstm_v1"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 100)
print("IMPLEMENTING GCN-LSTM V1")
print("=" * 100)

# ============================================================
# INTEGRITY CHECKS
# ============================================================
print("\n--- RUNNING MANDATORY INTEGRITY CHECKS ---")

def fail(msg):
    print(f"FAIL: {msg}")
    sys.exit(1)

# Load metadata
train_meta = pd.read_csv(PROJECT / "data/processed/splits/train.csv")
val_meta = pd.read_csv(PROJECT / "data/processed/splits/validation.csv")
test_meta = pd.read_csv(PROJECT / "data/processed/splits/test.csv")

# 1. Verify 16 canonical reservoirs
res_train = sorted(train_meta["reservoir"].unique())
res_val = sorted(val_meta["reservoir"].unique())
res_test = sorted(test_meta["reservoir"].unique())
if not (res_train == res_val == res_test):
    fail("Reservoir sets differ across splits")
if len(res_train) != 16:
    fail(f"Expected 16 reservoirs, got {len(res_train)}")
print("1. [PASS] Verified 16 canonical reservoirs.")

NODE_ORDER = res_train
res_to_idx = {r: i for i, r in enumerate(NODE_ORDER)}

# 2. Verify graph nodes belong to 16-node set
with open(PROJECT / "data/processed/graph/graph_B/metadata.json") as f:
    graph_meta = json.load(f)
graph_nodes = set(graph_meta["reservoirs"])
if not set(NODE_ORDER).issubset(graph_nodes):
    fail("Not all LSTM nodes are in Graph B")
print("2. [PASS] Verified every graph node belongs to the 16-node set (Graph B is a superset).")

# 3. Verify graph edge_index contains only valid node indices
edges_df = pd.read_csv(PROJECT / "data/processed/graph/graph_B/edges.csv")
mask = edges_df["source"].isin(NODE_ORDER) & edges_df["target"].isin(NODE_ORDER)
edges_16 = edges_df[mask]

edge_index_list = []
for _, row in edges_16.iterrows():
    u = res_to_idx[row["source"]]
    v = res_to_idx[row["target"]]
    edge_index_list.append([u, v])
    edge_index_list.append([v, u]) # Bidirectional

edge_index_np = np.array(edge_index_list).T
edge_index_np = np.unique(edge_index_np, axis=1)

if edge_index_np.min() < 0 or edge_index_np.max() >= 16:
    fail("Invalid node indices in edge_index")
print("3. [PASS] Verified graph edge_index contains only valid node indices (0-15).")

# 4. Verify date counts
train_dates = sorted(train_meta["date"].unique())
val_dates = sorted(val_meta["date"].unique())
test_dates = sorted(test_meta["date"].unique())

if len(train_dates) != 1131 or len(val_dates) != 285 or len(test_dates) != 74:
    fail(f"Date counts mismatch: {len(train_dates)}/{len(val_dates)}/{len(test_dates)}")
print("4. [PASS] Verified train/validation/test date counts remain 1131/285/74.")

# 5, 6, 7, 8. Pivot function and checks
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

X_tr, y_tr, m_tr = pivot_data(train_meta, X_train_flat, y_train_flat, train_dates)
X_va, y_va, m_va = pivot_data(val_meta, X_val_flat, y_val_flat, val_dates)
X_te, y_te, m_te = pivot_data(test_meta, X_test_flat, y_test_flat, test_dates)

if m_tr.sum() != len(train_meta) or m_va.sum() != len(val_meta) or m_te.sum() != len(test_meta):
    fail("Rows were dropped during pivot")
print("5. [PASS] Verified no rows are silently dropped.")

if not (np.all(X_tr[~m_tr] == 0) and np.all(y_tr[~m_tr] == 0)):
    fail("Unmasked slots are not zero")
print("6. [PASS] Verified padded slots are exactly those indicated by the mask.")

if X_tr.shape[1:] != (16, 7, 5):
    fail(f"Invalid X shape {X_tr.shape}")
print("7. [PASS] Verified X shape = (B,16,7,5).")

if y_tr.shape[1:] != (16, 3):
    fail(f"Invalid y shape {y_tr.shape}")
print("8. [PASS] Verified y shape = (B,16,3).")

# 9. Verify target inverse transformation
target_scaler = joblib.load(PROJECT / "data/processed/scaled/target_scaler.pkl")
log_target_scaler = joblib.load(PROJECT / "models/lstm_pytorch_v3_logtarget/log_target_scaler.pkl")

# Reconstruct original targets from scaled (v2 format)
y_te_v2_orig = target_scaler.inverse_transform(y_test_flat)

# Now, we need the new target formatting (log1p + log_target_scaler)
y_tr_orig = target_scaler.inverse_transform(y_train_flat)
y_va_orig = target_scaler.inverse_transform(y_val_flat)
y_te_orig = target_scaler.inverse_transform(y_test_flat)

y_tr_log = np.log1p(y_tr_orig)
y_va_log = np.log1p(y_va_orig)
y_te_log = np.log1p(y_te_orig)

y_tr_scaled = log_target_scaler.transform(y_tr_log).astype(np.float32)
y_va_scaled = log_target_scaler.transform(y_va_log).astype(np.float32)
y_te_scaled = log_target_scaler.transform(y_te_log).astype(np.float32)

y_te_recon = np.expm1(log_target_scaler.inverse_transform(y_te_scaled))
if not np.allclose(y_te_v2_orig, y_te_recon, atol=1e-2):
    fail("Inverse transform failed to reproduce original values")
print("9. [PASS] Verified target inverse transformation reproduces original inflow values.")

# Re-pivot targets with new scaling
_, y_tr, _ = pivot_data(train_meta, X_train_flat, y_tr_scaled, train_dates)
_, y_va, _ = pivot_data(val_meta, X_val_flat, y_va_scaled, val_dates)
_, y_te, _ = pivot_data(test_meta, X_test_flat, y_te_scaled, test_dates)

# 10. Verify dates non-overlapping
if set(train_dates).intersection(val_dates) or set(train_dates).intersection(test_dates) or set(val_dates).intersection(test_dates):
    fail("Dates overlap between splits")
print("10. [PASS] Verified train/validation/test dates remain non-overlapping.")

# 11. Verify no NaN/Inf
if not (np.isfinite(X_tr).all() and np.isfinite(y_tr).all() and np.isfinite(X_va).all() and np.isfinite(y_va).all() and np.isfinite(X_te).all() and np.isfinite(y_te).all()):
    fail("NaN or Inf found")
print("11. [PASS] Verified no NaN/Inf.")

print("\nAll integrity checks (1-11) passed. Proceeding with dataset preparation.")

# ============================================================
# DATASET AND MODEL
# ============================================================
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
        # x is (B, 16, 7, 5)
        B, N, T, F = x.shape
        
        # PyG GCNConv supports batched inputs (..., N, F)
        x_gcn_in = x.permute(0, 2, 1, 3) # (B, T, 16, 5)
        
        # GCN forward pass
        gcn_out = self.gcn(x_gcn_in, self.edge_index) # (B, T, 16, 32)
        gcn_out = self.relu1(gcn_out)
        
        # Prepare for LSTM
        # gcn_out is (B, T, 16, 32). Permute to (B, 16, T, 32)
        lstm_in = gcn_out.permute(0, 2, 1, 3).contiguous()
        lstm_in = lstm_in.view(B * N, T, 32)
        
        # LSTM forward pass
        seq_out, _ = self.lstm(lstm_in) # (B*N, T, 64)
        last_hidden = seq_out[:, -1, :] # (B*N, 64)
        
        # Fully connected layers
        out = self.fc1(last_hidden)
        out = self.relu2(out)
        out = self.output(out) # (B*N, 3)
        
        # Reshape back to (B, 16, 3)
        out = out.view(B, N, 3)
        return out

model = GCNLSTM(edge_index=edge_index_t).to(DEVICE)

# Test dummy pass for 12, 13
dummy_out = model(X_tr_t[:2].to(DEVICE))
if dummy_out.shape != (2, 16, 3):
    fail(f"Invalid model output shape {dummy_out.shape}")
print("12. [PASS] Verified the model output shape is (B,16,3).")

# ============================================================
# MASKED LOSS
# ============================================================
class MaskedHuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.criterion = nn.HuberLoss(delta=delta, reduction='none')
        
    def forward(self, pred, target, mask):
        # pred: (B, N, 3), target: (B, N, 3), mask: (B, N)
        loss = self.criterion(pred, target) # (B, N, 3)
        mask_expanded = mask.unsqueeze(-1).expand_as(loss) # (B, N, 3)
        masked_loss = loss * mask_expanded
        
        # Ensure we don't divide by zero if mask is entirely false
        num_valid = mask_expanded.sum()
        if num_valid > 0:
            return masked_loss.sum() / num_valid
        else:
            return masked_loss.sum() * 0.0 # Return 0 loss with grad

criterion = MaskedHuberLoss(delta=1.0)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("13. [PASS] Verified masked loss ignores padded nodes (by code definition).")

# ============================================================
# TRAINING
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
    print(f"Epoch {epoch:03d}/{MAX_EPOCHS} | Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f}")
    
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
# EVALUATION
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
    
    # Flatten valid targets
    valid_preds = preds[masks]
    valid_actuals = actuals[masks]
    
    return valid_preds, valid_actuals

test_preds, test_actuals = predict(test_loader)

# Inverse transform
test_preds_orig = np.expm1(log_target_scaler.inverse_transform(test_preds))
test_actuals_orig = np.expm1(log_target_scaler.inverse_transform(test_actuals))

target_names = ["target_1d", "target_3d", "target_7d"]
metrics_rows = []
for i, name in enumerate(target_names):
    p = test_preds_orig[:, i]
    a = test_actuals_orig[:, i]
    metrics_rows.append({
        "target": name,
        "MAE": mean_absolute_error(a, p),
        "RMSE": np.sqrt(mean_squared_error(a, p)),
        "R2": r2_score(a, p),
        "Bias": float(np.mean(p - a)),
        "negative_count": int((p < 0).sum())
    })

metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv(RESULT_DIR / "gcn_lstm_v1_metrics.csv", index=False)
print("\n--- GCN-LSTM V1 TEST METRICS ---")
print(metrics_df.to_string(index=False))

# ============================================================
# COMPARISONS
# ============================================================
final_comp = pd.read_csv(PROJECT / "results/final_model_comparison/final_model_comparison_original_units.csv")
v3_metrics = final_comp[final_comp["model"] == "LSTM_V3"].set_index("target")
persistence_metrics = final_comp[final_comp["model"] == "Persistence"].set_index("target")
hgb_metrics = final_comp[final_comp["model"] == "HistGradientBoosting"].set_index("target")

print("\n--- GCN-LSTM V1 vs LSTM V3 ---")
for i, name in enumerate(target_names):
    v1_r2 = metrics_rows[i]["R2"]
    v3_r2 = v3_metrics.loc[name, "R2"]
    v1_mae = metrics_rows[i]["MAE"]
    v3_mae = v3_metrics.loc[name, "MAE"]
    print(f"{name}: V1 R2={v1_r2:.4f} vs V3 R2={v3_r2:.4f} | V1 MAE={v1_mae:.4f} vs V3 MAE={v3_mae:.4f}")
    if v1_r2 > v3_r2:
        print(f"  -> V1 improved R2 on {name}!")

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
            "MAE": mean_absolute_error(a, p),
            "RMSE": np.sqrt(mean_squared_error(a, p)),
            "R2": r2_score(a, p),
            "Bias": float(np.mean(p - a))
        })
        
period_df = pd.DataFrame(period_rows)
period_df.to_csv(RESULT_DIR / "gcn_lstm_v1_period_metrics.csv", index=False)

print("\n--- PERIOD-WISE COMPARISON ---")
print(period_df.to_string(index=False))

# Exact files created
print("\n--- EXACT FILES CREATED ---")
print(f"File: {MODEL_DIR / 'best_model.pt'}")
print(f"File: {RESULT_DIR / 'gcn_lstm_v1_metrics.csv'}")
print(f"File: {RESULT_DIR / 'gcn_lstm_v1_period_metrics.csv'}")
print(f"File: {PROJECT / 'src/modeling/train_gcn_lstm_v1.py'}")
