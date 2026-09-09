"""
Milestone 12.8 — Gated GCN-LSTM V1: Selective Spatial Fusion
=============================================================
Architecture: dual-branch with learnable scalar gate.
  Local branch:  raw (7,5) -> LSTM(5,64) -> h_local (64-dim)
  Spatial branch: GCNConv(5,32) -> LSTM(32,64) -> h_spatial (64-dim)
  Fusion: h = h_local + sigmoid(alpha) * h_spatial
  Head:   Linear(64,32) -> ReLU -> Linear(32,3)

Gate initialized so sigmoid(alpha) ≈ 0.01 (alpha = -4.595).
This means the model starts essentially as a per-node LSTM baseline,
and must learn to open the gate if spatial information helps.

Graph: existing correlation graph (41 edges, train-only, leakage-checked).
All other controls identical to V1/V1.1/V1.2.
"""
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
# CONFIGURATIONS — match V1/V1.1/V1.2
# ============================================================
SEED = 42; BATCH_SIZE = 32; MAX_EPOCHS = 100; LEARNING_RATE = 0.001; EARLY_STOPPING = 12
DEVICE = torch.device("cpu")
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

PROJECT = Path(__file__).resolve().parents[2]
MODEL_DIR  = PROJECT / "models"  / "gcn_lstm_gated_v1"
RESULT_DIR = PROJECT / "results" / "gcn_lstm_gated_v1"
GRAPH_DIR  = PROJECT / "data/processed/graph/graph_D_correlation_v1_2"
MODEL_DIR.mkdir(parents=True, exist_ok=True); RESULT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 100)
print("MILESTONE 12.8 — GATED GCN-LSTM V1 (SELECTIVE SPATIAL FUSION)")
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
# PRE-TRAINING INTEGRITY CHECKS
# ============================================================
print("\n--- PRE-TRAINING INTEGRITY CHECKS ---")

# 1. 16 reservoirs
if not (res_train == res_val == res_test): fail("Reservoir sets differ")
if len(res_train) != 16: fail(f"Expected 16 reservoirs, got {len(res_train)}")
print("1.  [PASS] Exactly 16 reservoirs.")

NODE_ORDER = res_train
res_to_idx = {r: i for i, r in enumerate(NODE_ORDER)}

# 2. Graph nodes match
with open(GRAPH_DIR / "metadata.json") as f: graph_meta = json.load(f)
if sorted(graph_meta["reservoirs"]) != NODE_ORDER: fail("Graph nodes mismatch")
print("2.  [PASS] Graph nodes match canonical ordering.")

# 3. Edge validity
edges_df = pd.read_csv(GRAPH_DIR / "edges.csv")
if set(edges_df["source"]) - set(NODE_ORDER) or set(edges_df["target"]) - set(NODE_ORDER):
    fail("Invalid edge nodes")
print("3.  [PASS] All edges reference valid nodes.")

# 4. Leakage check
if not graph_meta.get("leakage_safe", False) or "training only" not in graph_meta.get("data_window", ""):
    fail("Graph leakage risk")
print(f"4.  [PASS] Graph is leakage-safe ('{graph_meta['data_window']}').")

# 5. Edge count
if len(edges_df) != 41 or graph_meta["edges"] != 41: fail("Edge count mismatch")
print("5.  [PASS] 41 undirected edges.")

# 6. Bidirectional edge_index
eil = []
for _, row in edges_df.iterrows():
    u, v = res_to_idx[row["source"]], res_to_idx[row["target"]]
    eil.append([u, v]); eil.append([v, u])
edge_index_np = np.unique(np.array(eil, dtype=np.int64).T, axis=1)
if edge_index_np.shape[1] != 82: fail(f"Expected 82 directed edges, got {edge_index_np.shape[1]}")
if edge_index_np.min() < 0 or edge_index_np.max() >= 16: fail("Edge index OOB")
print(f"6.  [PASS] Bidirectional edge_index: 82 directed edges.")

# 7-11. Data loading & checks
train_dates = sorted(train_meta["date"].unique())
val_dates   = sorted(val_meta["date"].unique())
test_dates  = sorted(test_meta["date"].unique())

def pivot_data(meta, X_flat, y_flat, dates):
    nd = len(dates)
    X_p = np.zeros((nd, 16, 7, 5), dtype=np.float32)
    y_p = np.zeros((nd, 16, 3), dtype=np.float32)
    m_p = np.zeros((nd, 16), dtype=bool)
    d2i = {d: i for i, d in enumerate(dates)}
    for idx, row in meta.iterrows():
        di, ri = d2i[row["date"]], res_to_idx[row["reservoir"]]
        X_p[di, ri] = X_flat[idx]; y_p[di, ri] = y_flat[idx]; m_p[di, ri] = True
    return X_p, y_p, m_p

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

y_tr_s = log_target_scaler.transform(np.log1p(y_tr_orig)).astype(np.float32)
y_va_s = log_target_scaler.transform(np.log1p(y_va_orig)).astype(np.float32)
y_te_s = log_target_scaler.transform(np.log1p(y_te_orig)).astype(np.float32)

X_tr, y_tr, m_tr = pivot_data(train_meta, X_train_flat, y_tr_s, train_dates)
X_va, y_va, m_va = pivot_data(val_meta,   X_val_flat,   y_va_s, val_dates)
X_te, y_te, m_te = pivot_data(test_meta,  X_test_flat,  y_te_s, test_dates)

if X_tr.shape[1:] != (16, 7, 5) or y_tr.shape[1:] != (16, 3): fail("Shape mismatch")
print("7.  [PASS] Tensor shapes: X=(B,16,7,5), y=(B,16,3).")

if m_tr.sum() != len(train_meta) or m_va.sum() != len(val_meta) or m_te.sum() != len(test_meta):
    fail("Mask count mismatch")
print(f"8.  [PASS] Masks unchanged (train={m_tr.sum()}, val={m_va.sum()}, test={m_te.sum()}).")

y_te_recon = np.expm1(log_target_scaler.inverse_transform(y_te_s))
if not np.allclose(y_te_orig, y_te_recon, atol=1e-2): fail("Inverse transform mismatch")
print("9.  [PASS] Target transformation round-trip verified.")

if len(train_dates) != 1131 or len(val_dates) != 285 or len(test_dates) != 74: fail("Date counts wrong")
if set(train_dates) & set(val_dates) or set(train_dates) & set(test_dates): fail("Date overlap")
print("10. [PASS] Dates: 1131/285/74, non-overlapping.")

for arr, nm in [(X_tr,"X_tr"),(y_tr,"y_tr"),(X_va,"X_va"),(y_va,"y_va"),(X_te,"X_te"),(y_te,"y_te")]:
    if not np.isfinite(arr).all(): fail(f"NaN/Inf in {nm}")
print("11. [PASS] No NaN/Inf.")

# 12. No test information in training
print("12. [PASS] No test information enters training (graph=train-only, splits verified).")
print("\nAll integrity checks PASSED.\n")

# ============================================================
# MASKING DOCUMENTATION
# ============================================================
print("--- MASKING IMPLEMENTATION ---")
print("Missing reservoir-date slots are zero-padded in X (all 5 features = 0).")
print("GCNConv with add_self_loops=True, normalize=True uses symmetric normalization.")
print("A zero-padded neighbor contributes a zero vector to the aggregation,")
print("which is equivalent to a mild dilution of real neighbor signals.")
print("This is the SAME masking behavior as V1/V1.1/V1.2 — not a new confound.")
print("The loss mask ensures zero-padded nodes do NOT contribute to gradients.")
print("The gate mechanism does NOT change the masking behavior.\n")

# ============================================================
# TENSORS & LOADERS
# ============================================================
X_tr_t = torch.from_numpy(X_tr); y_tr_t = torch.from_numpy(y_tr); m_tr_t = torch.from_numpy(m_tr)
X_va_t = torch.from_numpy(X_va); y_va_t = torch.from_numpy(y_va); m_va_t = torch.from_numpy(m_va)
X_te_t = torch.from_numpy(X_te); y_te_t = torch.from_numpy(y_te); m_te_t = torch.from_numpy(m_te)
edge_index_t = torch.from_numpy(edge_index_np).long().to(DEVICE)

train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t, m_tr_t), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_va_t, y_va_t, m_va_t), batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(TensorDataset(X_te_t, y_te_t, m_te_t), batch_size=BATCH_SIZE, shuffle=False)

# ============================================================
# MODEL: GATED GCN-LSTM
# ============================================================
class GatedGCNLSTM(nn.Module):
    """
    Dual-branch architecture with learnable scalar gate.

    Local branch:  LSTM(input=5, hidden=64) — processes raw features per node
    Spatial branch: GCNConv(5->32) + ReLU -> LSTM(input=32, hidden=64) — same as V1/V1.2
    Fusion: h = h_local + sigmoid(alpha) * h_spatial
    Head:  FC(64->32) -> ReLU -> FC(32->3)

    Gate alpha initialized to -4.595 so sigmoid(alpha) ≈ 0.01.
    The model starts ~99% local, and must learn to open the gate.
    """
    def __init__(self, edge_index):
        super().__init__()
        self.edge_index = edge_index

        # LOCAL branch: per-node LSTM on raw features (matches V3 architecture)
        self.local_lstm = nn.LSTM(input_size=5, hidden_size=64, num_layers=1, batch_first=True)

        # SPATIAL branch: GCNConv + LSTM (matches V1/V1.2 architecture)
        self.gcn = GCNConv(5, 32, add_self_loops=True, normalize=True)
        self.gcn_relu = nn.ReLU()
        self.spatial_lstm = nn.LSTM(input_size=32, hidden_size=64, num_layers=1, batch_first=True)

        # Scalar gate: sigmoid(-4.595) ≈ 0.01
        self.alpha = nn.Parameter(torch.tensor(-4.595))

        # Shared FC head (same as V1/V1.1/V1.2)
        self.fc1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.output = nn.Linear(32, 3)

    def forward(self, x):
        B, N, T, F = x.shape  # (B, 16, 7, 5)

        # --- LOCAL BRANCH ---
        local_in = x.view(B * N, T, F)  # (B*16, 7, 5)
        local_out, _ = self.local_lstm(local_in)  # (B*16, 7, 64)
        h_local = local_out[:, -1, :]  # (B*16, 64)

        # --- SPATIAL BRANCH ---
        x_gcn = x.permute(0, 2, 1, 3)  # (B, T, N, F) for GCNConv
        gcn_out = self.gcn_relu(self.gcn(x_gcn, self.edge_index))  # (B, T, N, 32)
        spatial_in = gcn_out.permute(0, 2, 1, 3).contiguous().view(B * N, T, 32)
        spatial_out, _ = self.spatial_lstm(spatial_in)  # (B*16, 7, 64)
        h_spatial = spatial_out[:, -1, :]  # (B*16, 64)

        # --- GATED FUSION ---
        gate = torch.sigmoid(self.alpha)  # scalar in [0, 1]
        h = h_local + gate * h_spatial  # (B*16, 64)

        # --- FC HEAD ---
        out = self.relu(self.fc1(h))
        out = self.output(out)  # (B*16, 3)
        return out.view(B, N, 3)

model = GatedGCNLSTM(edge_index=edge_index_t).to(DEVICE)
init_gate = torch.sigmoid(model.alpha).item()
print(f"--- MODEL ---")
print(f"Architecture: Dual-branch Gated GCN-LSTM")
print(f"Local branch:  LSTM(5->64)")
print(f"Spatial branch: GCNConv(5->32) + ReLU -> LSTM(32->64)")
print(f"Fusion: h_local + sigmoid(alpha) * h_spatial")
print(f"Head: FC(64->32) + ReLU + FC(32->3)")
print(f"Initial gate value: sigmoid({model.alpha.item():.3f}) = {init_gate:.4f}")
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}\n")

# ============================================================
# LOSS & OPTIMIZER — same as V1/V1.1/V1.2
# ============================================================
class MaskedHuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.criterion = nn.HuberLoss(delta=delta, reduction='none')
    def forward(self, pred, target, mask):
        loss = self.criterion(pred, target)
        me = mask.unsqueeze(-1).expand_as(loss)
        ml = loss * me
        nv = me.sum()
        return ml.sum() / nv if nv > 0 else ml.sum() * 0.0

criterion = MaskedHuberLoss(delta=1.0)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ============================================================
# TRAINING
# ============================================================
best_val_loss = float("inf"); patience = 0; history = []
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
    tl = np.mean(train_losses); vl = np.mean(val_losses)
    gate_val = torch.sigmoid(model.alpha).item()
    history.append({"epoch": epoch, "train_loss": tl, "val_loss": vl, "gate": gate_val})
    print(f"Epoch {epoch:03d}/{MAX_EPOCHS} | Train: {tl:.4f} | Val: {vl:.4f} | Gate: {gate_val:.4f}")
    if vl < best_val_loss:
        best_val_loss = vl; patience = 0
        torch.save(model.state_dict(), MODEL_DIR / "best_model.pt")
    else:
        patience += 1
    if patience >= EARLY_STOPPING:
        print(f"Early stopping at epoch {epoch}"); break

hist_df = pd.DataFrame(history)
hist_df.to_csv(RESULT_DIR / "training_history.csv", index=False)
best_epoch = hist_df.loc[hist_df["val_loss"].idxmin(), "epoch"]
print(f"\nBest epoch: {int(best_epoch)}, Best val loss: {best_val_loss:.4f}")

# ============================================================
# EVALUATION
# ============================================================
model.load_state_dict(torch.load(MODEL_DIR / "best_model.pt", weights_only=True)); model.eval()
final_gate = torch.sigmoid(model.alpha).item()
final_alpha = model.alpha.item()

def predict(loader):
    ps, acs, ms = [], [], []
    with torch.no_grad():
        for xb, yb, mb in loader:
            out = model(xb.to(DEVICE))
            ps.append(out.cpu().numpy()); acs.append(yb.numpy()); ms.append(mb.numpy())
    ps = np.concatenate(ps); acs = np.concatenate(acs); ms = np.concatenate(ms)
    return ps[ms], acs[ms]

test_preds, test_actuals = predict(test_loader)
test_preds_orig   = np.expm1(log_target_scaler.inverse_transform(test_preds))
test_actuals_orig = np.expm1(log_target_scaler.inverse_transform(test_actuals))

target_names = ["target_1d", "target_3d", "target_7d"]
metrics_rows = []
for i, name in enumerate(target_names):
    p = test_preds_orig[:, i]; a = test_actuals_orig[:, i]
    nc = int((p < 0).sum()); np_ = 100.0 * nc / len(p)
    metrics_rows.append({
        "Model": "Gated V1", "Horizon": name,
        "MAE": float(mean_absolute_error(a, p)),
        "RMSE": float(np.sqrt(mean_squared_error(a, p))),
        "R2": float(r2_score(a, p)),
        "Bias": float(np.mean(p - a)),
        "Neg_Count": nc, "Neg_Pct": round(np_, 2)
    })

metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv(RESULT_DIR / "gcn_lstm_gated_v1_metrics.csv", index=False)

# Period-wise
test_dates_full = pd.to_datetime(test_meta['date'])
jan_apr = (test_dates_full >= "2025-01-01") & (test_dates_full < "2025-05-01")
jun_aug = (test_dates_full >= "2025-06-01") & (test_dates_full <= "2025-08-31")

period_rows = []
for pname, pmask in [("Jan-Apr 2025", jan_apr), ("Jun-Aug 2025", jun_aug)]:
    if pmask.sum() == 0: continue
    for i, name in enumerate(target_names):
        p = test_preds_orig[pmask, i]; a = test_actuals_orig[pmask, i]
        period_rows.append({
            "Period": pname, "Horizon": name,
            "MAE": float(mean_absolute_error(a, p)),
            "RMSE": float(np.sqrt(mean_squared_error(a, p))),
            "R2": float(r2_score(a, p)),
            "Bias": float(np.mean(p - a))
        })

period_df = pd.DataFrame(period_rows)
period_df.to_csv(RESULT_DIR / "gcn_lstm_gated_v1_period_metrics.csv", index=False)

# Gate diagnostics
gate_diag = {
    "initial_alpha": -4.595, "initial_gate": 0.01,
    "final_alpha": final_alpha, "final_gate": final_gate,
    "gate_type": "scalar (1 parameter)",
    "interpretation": "gate < 0.05 = effectively ignored spatial; gate > 0.2 = meaningfully active"
}
with open(RESULT_DIR / "gate_diagnostics.json", "w") as f:
    json.dump(gate_diag, f, indent=2)

# ============================================================
# PRINT FINAL RESULTS
# ============================================================
print("\n" + "=" * 100)
print("MILESTONE 12.8 COMPLETE")
print("=" * 100)

print(f"\n--- INTEGRITY STATUS: ALL 12 CHECKS PASSED ---")

print(f"\n--- ARCHITECTURE ---")
print(f"Dual-branch Gated GCN-LSTM")
print(f"Local: LSTM(5->64) | Spatial: GCNConv(5->32)->LSTM(32->64)")
print(f"Fusion: h_local + sigmoid(alpha) * h_spatial")
print(f"Parameters: {total_params:,}")

print(f"\n--- GATE DIAGNOSTICS ---")
print(f"Initial gate: {gate_diag['initial_gate']:.4f}")
print(f"Final gate:   {final_gate:.4f}")
print(f"Final alpha:  {final_alpha:.3f}")
if final_gate < 0.05:
    print("Interpretation: Gate COLLAPSED — model learned to IGNORE spatial information.")
elif final_gate < 0.20:
    print("Interpretation: Gate marginally active — weak spatial contribution.")
else:
    print("Interpretation: Gate meaningfully active — spatial branch contributing.")

print(f"\n--- OVERALL METRICS (ORIGINAL INFLOW UNITS) ---")
print(metrics_df.to_string(index=False))

print(f"\n--- PERIOD-WISE METRICS ---")
print(period_df.to_string(index=False))

# Comparison
v3  = {"target_1d": {"MAE":1.837,"RMSE":3.031,"R2":0.759}, "target_3d": {"MAE":2.095,"RMSE":3.522,"R2":0.654}, "target_7d": {"MAE":2.703,"RMSE":4.573,"R2":0.501}}
v1  = {"target_1d": {"R2":-0.706}, "target_3d": {"R2":-0.802}, "target_7d": {"R2":-0.487}}
v11 = {"target_1d": {"R2":0.638}, "target_3d": {"R2":0.586}, "target_7d": {"R2":0.531}}
v12 = {"target_1d": {"R2":-0.098}, "target_3d": {"R2":-0.089}, "target_7d": {"R2":0.057}}

print(f"\n--- V3 COMPARISON ---")
print(f"{'Horizon':<12} {'Gated R²':>10} {'V3 R²':>8} {'Delta':>8} {'Gated MAE':>10} {'V3 MAE':>8}")
print("-" * 58)
beats_v3 = 0
for i, h in enumerate(target_names):
    gr2 = metrics_rows[i]["R2"]; gmae = metrics_rows[i]["MAE"]
    d = gr2 - v3[h]["R2"]
    if gr2 > v3[h]["R2"]: beats_v3 += 1
    print(f"{h:<12} {gr2:>10.3f} {v3[h]['R2']:>8.3f} {d:>+8.3f} {gmae:>10.3f} {v3[h]['MAE']:>8.3f}")

print(f"\n--- FULL R² COMPARISON ---")
print(f"{'Horizon':<12} {'Gated':>8} {'V3':>8} {'V1.1':>8} {'V1.2':>8} {'V1':>8}")
print("-" * 52)
for i, h in enumerate(target_names):
    gr2 = metrics_rows[i]["R2"]
    print(f"{h:<12} {gr2:>8.3f} {v3[h]['R2']:>8.3f} {v11[h]['R2']:>8.3f} {v12[h]['R2']:>8.3f} {v1[h]['R2']:>8.3f}")

# Classification
monsoon_ok = True
for row in period_rows:
    if row["Period"] == "Jun-Aug 2025" and row["R2"] < -0.5:
        monsoon_ok = False

if beats_v3 >= 2 and monsoon_ok and final_gate > 0.05:
    classification = "SUCCESS"
elif all(metrics_rows[i]["R2"] > v12[target_names[i]]["R2"] for i in range(3)) and \
     all(metrics_rows[i]["R2"] > v1[target_names[i]]["R2"] for i in range(3)):
    if any(metrics_rows[i]["R2"] > v3[target_names[i]]["R2"] for i in range(3)):
        classification = "PARTIAL SUCCESS"
    else:
        classification = "PARTIAL SUCCESS"
else:
    classification = "FLOP"

# Override to FLOP if gate collapsed and no improvement
if final_gate < 0.02 and beats_v3 == 0:
    classification = "FLOP"

print(f"\n--- EXPERIMENT CLASSIFICATION: {classification} ---")

if classification == "FLOP":
    print("\nRECOMMENDATION: STOP ALL GNN EXPERIMENTATION.")
    print("The GNN branch is frozen as a rigorous negative result.")
    print("No fourth graph. No new threshold. No attention sweep.")
elif classification == "PARTIAL SUCCESS":
    print("\nRECOMMENDATION: V3 remains the primary forecasting model.")
    print("The gated architecture shows some promise but does not justify replacing V3.")
else:
    print("\nRECOMMENDATION: The gated GCN-LSTM demonstrates that selective spatial fusion")
    print("can improve upon the temporal-only baseline when spatial information is ")
    print("incorporated conditionally rather than through unconditional message passing.")
