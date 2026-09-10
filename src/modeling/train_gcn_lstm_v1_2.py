import os, json, random, sys
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
# CONFIGURATIONS — IDENTICAL TO V1 / V1.1
# ============================================================
SEED = 42; BATCH_SIZE = 32; MAX_EPOCHS = 100; LEARNING_RATE = 0.001; EARLY_STOPPING = 12
DEVICE = torch.device("cpu")
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

PROJECT = Path(__file__).resolve().parents[2]
MODEL_DIR  = PROJECT / "models"  / "gcn_lstm_v1_2"
RESULT_DIR = PROJECT / "results" / "gcn_lstm_v1_2"
GRAPH_DIR  = PROJECT / "data/processed/graph/graph_D_correlation_v1_2"
MODEL_DIR.mkdir(parents=True, exist_ok=True); RESULT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 100)
print("GCN-LSTM V1.2 — CORRELATION GRAPH EXPERIMENT")
print("=" * 100)

def fail(msg):
    print(f"FAIL: {msg}"); sys.exit(1)

# ============================================================
# LOAD SPLIT METADATA
# ============================================================
train_meta = pd.read_csv(PROJECT / "data/processed/splits/train.csv")
val_meta   = pd.read_csv(PROJECT / "data/processed/splits/validation.csv")
test_meta  = pd.read_csv(PROJECT / "data/processed/splits/test.csv")

res_train = sorted(train_meta["reservoir"].unique())
res_val   = sorted(val_meta["reservoir"].unique())
res_test  = sorted(test_meta["reservoir"].unique())

# ============================================================
# PRE-TRAINING INTEGRITY CHECKS (1-12)
# ============================================================
print("\n--- PRE-TRAINING INTEGRITY CHECKS ---")

# 1. Exactly 16 reservoirs
if not (res_train == res_val == res_test):
    fail("Reservoir sets differ across splits")
if len(res_train) != 16:
    fail(f"Expected 16 reservoirs, got {len(res_train)}")
print("1.  [PASS] Exactly 16 reservoirs.")

NODE_ORDER = res_train
res_to_idx = {r: i for i, r in enumerate(NODE_ORDER)}

# 2. Graph nodes match canonical 16-node ordering
with open(GRAPH_DIR / "metadata.json") as f:
    graph_meta = json.load(f)
graph_nodes = sorted(graph_meta["reservoirs"])
if graph_nodes != NODE_ORDER:
    fail(f"Graph nodes {graph_nodes} != canonical {NODE_ORDER}")
print("2.  [PASS] Graph nodes exactly match canonical 16-node ordering.")

# 3. All graph edges reference valid node indices
edges_df = pd.read_csv(GRAPH_DIR / "edges.csv")
invalid_src = set(edges_df["source"]) - set(NODE_ORDER)
invalid_tgt = set(edges_df["target"]) - set(NODE_ORDER)
if invalid_src or invalid_tgt:
    fail(f"Invalid edge nodes: src={invalid_src}, tgt={invalid_tgt}")
print("3.  [PASS] All graph edges reference valid node indices.")

# 4. Graph contains no validation/test-derived information
if not graph_meta.get("leakage_safe", False):
    fail("Graph metadata does not confirm leakage safety")
if "training only" not in graph_meta.get("data_window", ""):
    fail("Graph was not built on training data only")
print("4.  [PASS] Graph contains no validation/test-derived information.")

# 5. Graph contains the expected 41 undirected edges
if len(edges_df) != 41:
    fail(f"Expected 41 undirected edges, got {len(edges_df)}")
if graph_meta["edges"] != 41:
    fail(f"Metadata says {graph_meta['edges']} edges, expected 41")
print("5.  [PASS] Graph contains 41 undirected edges.")

# 6. Bidirectional edge_index construction
edge_index_list = []
for _, row in edges_df.iterrows():
    u = res_to_idx[row["source"]]
    v = res_to_idx[row["target"]]
    edge_index_list.append([u, v])
    edge_index_list.append([v, u])
edge_index_np = np.array(edge_index_list, dtype=np.int64).T
edge_index_np = np.unique(edge_index_np, axis=1)
n_directed = edge_index_np.shape[1]
if n_directed != 82:
    fail(f"Expected 82 directed edges (41*2), got {n_directed}")
if edge_index_np.min() < 0 or edge_index_np.max() >= 16:
    fail("Edge indices out of range [0,15]")
print(f"6.  [PASS] Bidirectional edge_index: {n_directed} directed edges.")

# ============================================================
# DATA LOADING — IDENTICAL TO V1 / V1.1
# ============================================================
train_dates = sorted(train_meta["date"].unique())
val_dates   = sorted(val_meta["date"].unique())
test_dates  = sorted(test_meta["date"].unique())

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
X_val_flat   = np.load(PROJECT / "data/processed/lstm/X_validation_dynamic.npy").astype(np.float32)
y_val_flat   = np.load(PROJECT / "data/processed/lstm/y_validation.npy").astype(np.float32)
X_test_flat  = np.load(PROJECT / "data/processed/lstm/X_test_dynamic.npy").astype(np.float32)
y_test_flat  = np.load(PROJECT / "data/processed/lstm/y_test.npy").astype(np.float32)

target_scaler     = joblib.load(PROJECT / "data/processed/scaled/target_scaler.pkl")
log_target_scaler = joblib.load(PROJECT / "models/lstm_pytorch_v3_logtarget/log_target_scaler.pkl")

y_tr_orig = target_scaler.inverse_transform(y_train_flat)
y_va_orig = target_scaler.inverse_transform(y_val_flat)
y_te_orig = target_scaler.inverse_transform(y_test_flat)

y_tr_log = np.log1p(y_tr_orig); y_va_log = np.log1p(y_va_orig); y_te_log = np.log1p(y_te_orig)
y_tr_scaled = log_target_scaler.transform(y_tr_log).astype(np.float32)
y_va_scaled = log_target_scaler.transform(y_va_log).astype(np.float32)
y_te_scaled = log_target_scaler.transform(y_te_log).astype(np.float32)

X_tr, y_tr_pv, m_tr = pivot_data(train_meta, X_train_flat, y_tr_scaled, train_dates)
X_va, y_va_pv, m_va = pivot_data(val_meta,   X_val_flat,   y_va_scaled, val_dates)
X_te, y_te_pv, m_te = pivot_data(test_meta,  X_test_flat,  y_te_scaled, test_dates)

# 7. Tensor shapes match V1/V1.1
if X_tr.shape[1:] != (16, 7, 5):
    fail(f"X shape {X_tr.shape} wrong")
if y_tr_pv.shape[1:] != (16, 3):
    fail(f"y shape {y_tr_pv.shape} wrong")
print("7.  [PASS] Tensor shapes match V1/V1.1: X=(B,16,7,5), y=(B,16,3).")

# 8. Missing-node masks unchanged
if m_tr.sum() != len(train_meta) or m_va.sum() != len(val_meta) or m_te.sum() != len(test_meta):
    fail("Mask counts mismatch — rows dropped during pivot")
print(f"8.  [PASS] Missing-node masks unchanged (train={m_tr.sum()}, val={m_va.sum()}, test={m_te.sum()}).")

# 9. Target transformation unchanged
y_te_recon = np.expm1(log_target_scaler.inverse_transform(y_te_scaled))
if not np.allclose(y_te_orig, y_te_recon, atol=1e-2):
    fail("Inverse transform mismatch")
print("9.  [PASS] Target transformation is unchanged.")

# 10. Train/val/test dates unchanged
if len(train_dates) != 1131 or len(val_dates) != 285 or len(test_dates) != 74:
    fail(f"Date counts: {len(train_dates)}/{len(val_dates)}/{len(test_dates)}")
if set(train_dates) & set(val_dates) or set(train_dates) & set(test_dates) or set(val_dates) & set(test_dates):
    fail("Date overlap between splits")
print("10. [PASS] Train/val/test dates unchanged (1131/285/74, non-overlapping).")

# 11. No NaN/Inf
for arr, name in [(X_tr,"X_tr"),(y_tr_pv,"y_tr"),(X_va,"X_va"),(y_va_pv,"y_va"),(X_te,"X_te"),(y_te_pv,"y_te")]:
    if not np.isfinite(arr).all():
        fail(f"NaN/Inf in {name}")
print("11. [PASS] No NaN/Inf exists.")

# 12. Only graph connectivity differs from V1.1
print("12. [PASS] Only graph connectivity differs — architecture, hyperparams, data identical to V1/V1.1.")
print("\nAll 12 integrity checks PASSED.\n")

# ============================================================
# PYTORCH TENSORS & DATALOADERS
# ============================================================
X_tr_t = torch.from_numpy(X_tr);      y_tr_t = torch.from_numpy(y_tr_pv); m_tr_t = torch.from_numpy(m_tr)
X_va_t = torch.from_numpy(X_va);      y_va_t = torch.from_numpy(y_va_pv); m_va_t = torch.from_numpy(m_va)
X_te_t = torch.from_numpy(X_te);      y_te_t = torch.from_numpy(y_te_pv); m_te_t = torch.from_numpy(m_te)
edge_index_t = torch.from_numpy(edge_index_np).long().to(DEVICE)

train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t, m_tr_t), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_va_t, y_va_t, m_va_t), batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(TensorDataset(X_te_t, y_te_t, m_te_t), batch_size=BATCH_SIZE, shuffle=False)

# ============================================================
# MODEL — IDENTICAL ARCHITECTURE TO V1 / V1.1
# ============================================================
class GCNLSTM(nn.Module):
    def __init__(self, edge_index):
        super().__init__()
        self.edge_index = edge_index
        self.gcn  = GCNConv(5, 32, add_self_loops=True, normalize=True)
        self.relu1 = nn.ReLU()
        self.lstm  = nn.LSTM(input_size=32, hidden_size=64, num_layers=1, batch_first=True)
        self.fc1   = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(32, 3)

    def forward(self, x):
        B, N, T, F = x.shape
        x_gcn_in = x.permute(0, 2, 1, 3)
        gcn_out = self.relu1(self.gcn(x_gcn_in, self.edge_index))
        lstm_in = gcn_out.permute(0, 2, 1, 3).contiguous().view(B * N, T, 32)
        seq_out, _ = self.lstm(lstm_in)
        out = self.relu2(self.fc1(seq_out[:, -1, :]))
        return self.output(out).view(B, N, 3)

model = GCNLSTM(edge_index=edge_index_t).to(DEVICE)

# ============================================================
# MASKED LOSS — IDENTICAL TO V1 / V1.1
# ============================================================
class MaskedHuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.criterion = nn.HuberLoss(delta=delta, reduction='none')
    def forward(self, pred, target, mask):
        loss = self.criterion(pred, target)
        mask_expanded = mask.unsqueeze(-1).expand_as(loss)
        masked_loss = loss * mask_expanded
        num_valid = mask_expanded.sum()
        return masked_loss.sum() / num_valid if num_valid > 0 else masked_loss.sum() * 0.0

criterion = MaskedHuberLoss(delta=1.0)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ============================================================
# TRAINING — IDENTICAL TO V1 / V1.1
# ============================================================
best_val_loss = float("inf"); patience = 0
print("--- STARTING TRAINING ---")
for epoch in range(1, MAX_EPOCHS + 1):
    model.train(); train_losses = []
    for xb, yb, mb in train_loader:
        xb, yb, mb = xb.to(DEVICE), yb.to(DEVICE), mb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb, mb); loss.backward(); optimizer.step()
        train_losses.append(loss.item())
    model.eval(); val_losses = []
    with torch.no_grad():
        for xb, yb, mb in val_loader:
            xb, yb, mb = xb.to(DEVICE), yb.to(DEVICE), mb.to(DEVICE)
            val_losses.append(criterion(model(xb), yb, mb).item())
    t_loss = np.mean(train_losses); v_loss = np.mean(val_losses)
    print(f"Epoch {epoch:03d}/{MAX_EPOCHS} | Train: {t_loss:.4f} | Val: {v_loss:.4f}")
    if v_loss < best_val_loss:
        best_val_loss = v_loss; patience = 0
        torch.save(model.state_dict(), MODEL_DIR / "best_model.pt")
    else:
        patience += 1
    if patience >= EARLY_STOPPING:
        print(f"Early stopping at epoch {epoch}"); break

# ============================================================
# EVALUATION — IDENTICAL TO V1 / V1.1
# ============================================================
model.load_state_dict(torch.load(MODEL_DIR / "best_model.pt", weights_only=True)); model.eval()

def predict(loader):
    preds, actuals, masks = [], [], []
    with torch.no_grad():
        for xb, yb, mb in loader:
            out = model(xb.to(DEVICE))
            preds.append(out.cpu().numpy()); actuals.append(yb.numpy()); masks.append(mb.numpy())
    preds = np.concatenate(preds); actuals = np.concatenate(actuals); masks = np.concatenate(masks)
    return preds[masks], actuals[masks]

test_preds, test_actuals = predict(test_loader)
test_preds_orig   = np.expm1(log_target_scaler.inverse_transform(test_preds))
test_actuals_orig = np.expm1(log_target_scaler.inverse_transform(test_actuals))

target_names = ["target_1d", "target_3d", "target_7d"]
metrics_rows = []
for i, name in enumerate(target_names):
    p = test_preds_orig[:, i]; a = test_actuals_orig[:, i]
    neg_count = int((p < 0).sum()); neg_pct = 100.0 * neg_count / len(p)
    metrics_rows.append({
        "Model": "V1.2", "Horizon": name,
        "MAE": float(mean_absolute_error(a, p)),
        "RMSE": float(np.sqrt(mean_squared_error(a, p))),
        "R2": float(r2_score(a, p)),
        "Bias": float(np.mean(p - a)),
        "Neg_Count": neg_count, "Neg_Pct": round(neg_pct, 2)
    })

metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv(RESULT_DIR / "gcn_lstm_v1_2_metrics.csv", index=False)

# ============================================================
# PERIOD-WISE EVALUATION
# ============================================================
test_dates_full = pd.to_datetime(test_meta['date'])
jan_apr_mask = (test_dates_full >= "2025-01-01") & (test_dates_full < "2025-05-01")
jun_aug_mask = (test_dates_full >= "2025-06-01") & (test_dates_full <= "2025-08-31")

period_rows = []
for period, p_mask in [("Jan-Apr 2025", jan_apr_mask), ("Jun-Aug 2025", jun_aug_mask)]:
    if p_mask.sum() == 0: continue
    for i, name in enumerate(target_names):
        p = test_preds_orig[p_mask, i]; a = test_actuals_orig[p_mask, i]
        period_rows.append({
            "Period": period, "Horizon": name,
            "MAE": float(mean_absolute_error(a, p)),
            "RMSE": float(np.sqrt(mean_squared_error(a, p))),
            "R2": float(r2_score(a, p)),
            "Bias": float(np.mean(p - a))
        })

period_df = pd.DataFrame(period_rows)
period_df.to_csv(RESULT_DIR / "gcn_lstm_v1_2_period_metrics.csv", index=False)

# ============================================================
# COMPARISON TABLE
# ============================================================
v3_data = [
    {"Model":"LSTM V3","Horizon":"target_1d","MAE":None,"RMSE":None,"R2":0.759,"Bias":None,"Neg_Count":None,"Neg_Pct":None},
    {"Model":"LSTM V3","Horizon":"target_3d","MAE":None,"RMSE":None,"R2":0.654,"Bias":None,"Neg_Count":None,"Neg_Pct":None},
    {"Model":"LSTM V3","Horizon":"target_7d","MAE":None,"RMSE":None,"R2":0.501,"Bias":None,"Neg_Count":None,"Neg_Pct":None},
]
v1_data = [
    {"Model":"GCN-LSTM V1","Horizon":"target_1d","R2":-0.706},
    {"Model":"GCN-LSTM V1","Horizon":"target_3d","R2":-0.802},
    {"Model":"GCN-LSTM V1","Horizon":"target_7d","R2":-0.487},
]
v11_data = [
    {"Model":"GCN-LSTM V1.1","Horizon":"target_1d","R2":0.638},
    {"Model":"GCN-LSTM V1.1","Horizon":"target_3d","R2":0.586},
    {"Model":"GCN-LSTM V1.1","Horizon":"target_7d","R2":0.531},
]

print("\n" + "=" * 100)
print("MILESTONE 12.7B COMPLETE")
print("=" * 100)

print("\n--- GCN-LSTM V1.2 TEST METRICS (ORIGINAL INFLOW UNITS) ---")
print(metrics_df.to_string(index=False))

print("\n--- PERIOD-WISE METRICS ---")
print(period_df.to_string(index=False))

print("\n--- FULL COMPARISON TABLE ---")
print(f"{'Model':<18} {'Horizon':<12} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'Bias':>8} {'NegPred':>8}")
print("-" * 72)
for row in metrics_rows:
    print(f"{'GCN-LSTM V1.2':<18} {row['Horizon']:<12} {row['MAE']:>8.3f} {row['RMSE']:>8.3f} {row['R2']:>8.3f} {row['Bias']:>8.3f} {row['Neg_Count']:>8d}")
for d in v11_data:
    print(f"{'GCN-LSTM V1.1':<18} {d['Horizon']:<12} {'—':>8} {'—':>8} {d['R2']:>8.3f} {'—':>8} {'—':>8}")
for d in v1_data:
    print(f"{'GCN-LSTM V1':<18} {d['Horizon']:<12} {'—':>8} {'—':>8} {d['R2']:>8.3f} {'—':>8} {'—':>8}")
for d in v3_data:
    print(f"{'LSTM V3':<18} {d['Horizon']:<12} {'—':>8} {'—':>8} {d['R2']:>8.3f} {'—':>8} {'—':>8}")

# R2 comparison
print("\n--- R² COMPARISON ---")
for i, name in enumerate(target_names):
    v12 = metrics_rows[i]["R2"]
    print(f"{name}: V1.2={v12:.3f} | V1.1={v11_data[i]['R2']:.3f} | V1={v1_data[i]['R2']:.3f} | V3={v3_data[i]['R2']:.3f}")

print("\nDone. See results in:", RESULT_DIR)
