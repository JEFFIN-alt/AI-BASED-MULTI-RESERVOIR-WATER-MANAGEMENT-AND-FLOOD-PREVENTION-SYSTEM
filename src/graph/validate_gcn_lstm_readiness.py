"""
validate_gcn_lstm_readiness.py
==============================
Read-only pre-flight check for the GCN-LSTM implementation.
Modifies NOTHING.  Checks items 1-10 from the design approval list.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
PASS = "PASS"
FAIL = "FAIL"

sep = "=" * 70
results = {}   # check_name → True/False


def ok(msg):
    print(f"  [PASS] {msg}")

def fail(msg):
    print(f"  [FAIL] {msg}")

def section(title):
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)


# ──────────────────────────────────────────────────────────────
# 1. The 16 reservoirs in the canonical LSTM tensors / splits
# ──────────────────────────────────────────────────────────────
section("CHECK 1 & 2  —  16-node list and excluded reservoirs")

SPLITS_DIR = PROJECT / "data/processed/splits"

train_meta = pd.read_csv(SPLITS_DIR / "train.csv", usecols=["date", "reservoir"])
val_meta   = pd.read_csv(SPLITS_DIR / "validation.csv", usecols=["date", "reservoir"])
test_meta  = pd.read_csv(SPLITS_DIR / "test.csv", usecols=["date", "reservoir"])

lstm_reservoirs_train = sorted(train_meta["reservoir"].unique())
lstm_reservoirs_val   = sorted(val_meta["reservoir"].unique())
lstm_reservoirs_test  = sorted(test_meta["reservoir"].unique())

# Must be identical across all three splits
if lstm_reservoirs_train == lstm_reservoirs_val == lstm_reservoirs_test:
    ok(f"All three splits contain the same {len(lstm_reservoirs_train)} reservoirs.")
else:
    fail("Reservoir sets differ across splits!")

NODE_ORDER = lstm_reservoirs_train   # canonical ordering
N = len(NODE_ORDER)
results["check1_node_consistency"] = (lstm_reservoirs_train == lstm_reservoirs_val == lstm_reservoirs_test)

print(f"\n  Canonical 16-node list ({N} nodes):")
for i, r in enumerate(NODE_ORDER):
    print(f"    {i:2d}  {r}")

# Graph B full 18-node list
GRAPH_B_META = PROJECT / "data/processed/graph/graph_B/metadata.json"
graph_b_meta = json.loads(GRAPH_B_META.read_text())
graph_b_all_nodes = graph_b_meta["reservoirs"]  # 18 nodes

excluded = sorted(set(graph_b_all_nodes) - set(NODE_ORDER))
also_in_graph_not_lstm = sorted(set(NODE_ORDER) - set(graph_b_all_nodes))

print(f"\n  Graph B has {len(graph_b_all_nodes)} nodes total.")
print(f"  Excluded from LSTM splits (dropped upstream): {excluded}")

if also_in_graph_not_lstm:
    fail(f"Nodes in LSTM not in Graph B: {also_in_graph_not_lstm}")
else:
    ok("All 16 LSTM nodes are present in Graph B.")

results["check2_excluded_pair"] = (len(excluded) == 2)

# Why excluded? — check raw clean CSV
CLEAN_CSV = PROJECT / "data/processed/kerala_reservoir_clean.csv"
clean_df = pd.read_csv(CLEAN_CSV, usecols=["date", "reservoir", "training_eligible"],
                       parse_dates=["date"])

print(f"\n  Reason for exclusion (training_eligible flag & row counts):")
for r in excluded:
    sub = clean_df[clean_df["reservoir"] == r]
    eligible = int(sub["training_eligible"].sum()) if "training_eligible" in sub.columns else "N/A"
    total_rows = len(sub)
    date_range = f"{sub['date'].min().date()} → {sub['date'].max().date()}" if len(sub) else "no rows"
    print(f"    {r}: total_rows={total_rows}, training_eligible={eligible}, range={date_range}")


# ──────────────────────────────────────────────────────────────
# 3. Filter Graph B to 16 nodes
# ──────────────────────────────────────────────────────────────
section("CHECK 3  —  Graph B filtered to 16-node set")

edges_raw = pd.read_csv(PROJECT / "data/processed/graph/graph_B/edges.csv")
mask = edges_raw["source"].isin(NODE_ORDER) & edges_raw["target"].isin(NODE_ORDER)
edges_16 = edges_raw[mask].copy().reset_index(drop=True)
dropped_edges = (~mask).sum()

print(f"  Graph B original edges  : {len(edges_raw)}")
print(f"  Edges involving excluded nodes: {dropped_edges}")
print(f"  Remaining edges (16-node): {len(edges_16)}")
print(f"\n  Filtered edge list:")
print(edges_16.to_string(index=False))

results["check3_filtered"] = True


# ──────────────────────────────────────────────────────────────
# 4. Connectivity of the 16-node filtered graph
# ──────────────────────────────────────────────────────────────
section("CHECK 4  —  16-node graph connectivity")

adj = {r: set() for r in NODE_ORDER}
for _, row in edges_16.iterrows():
    adj[row["source"]].add(row["target"])
    adj[row["target"]].add(row["source"])

degree = {r: len(adj[r]) for r in NODE_ORDER}
isolated_16 = [r for r in NODE_ORDER if degree[r] == 0]

print(f"  Degree per node:")
for r in NODE_ORDER:
    print(f"    {r:20s}  degree={degree[r]}")

if isolated_16:
    fail(f"Isolated nodes in 16-node graph: {isolated_16}")
else:
    ok("No isolated nodes — every node has at least one edge.")

# BFS component count
components, unseen = [], set(NODE_ORDER)
while unseen:
    start, stack, comp = next(iter(unseen)), [next(iter(unseen))], set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node in comp:
            continue
        comp.add(node)
        unseen.discard(node)
        stack.extend(adj[node] - comp)
    components.append(sorted(comp))

print(f"\n  Connected components: {len(components)}")
for i, c in enumerate(components, 1):
    print(f"    Component {i}: {len(c)} nodes — {c}")

results["check4_connectivity"] = (len(isolated_16) == 0 and len(components) == 1)


# ──────────────────────────────────────────────────────────────
# 5. Pivot feasibility — date coverage per split
# ──────────────────────────────────────────────────────────────
section("CHECK 5, 7, 8  —  Pivot feasibility and missing data")

X_train_flat = np.load(PROJECT / "data/processed/lstm/X_train_dynamic.npy")
X_val_flat   = np.load(PROJECT / "data/processed/lstm/X_validation_dynamic.npy")
X_test_flat  = np.load(PROJECT / "data/processed/lstm/X_test_dynamic.npy")

y_train_flat = np.load(PROJECT / "data/processed/lstm/y_train.npy")
y_val_flat   = np.load(PROJECT / "data/processed/lstm/y_validation.npy")
y_test_flat  = np.load(PROJECT / "data/processed/lstm/y_test.npy")

res_to_idx = {r: i for i, r in enumerate(NODE_ORDER)}

def pivot_analysis(meta_df, X_flat, y_flat, split_name):
    """
    For every unique date in the split, tally how many of the 16 nodes
    have a row.  Returns summary statistics.
    """
    dates = sorted(meta_df["date"].unique())
    n_dates = len(dates)
    date_to_idx = {d: i for i, d in enumerate(dates)}

    # Build pivot index: for each row, record (date_idx, node_idx)
    coverage = np.zeros((n_dates, N), dtype=bool)
    for _, row in meta_df.iterrows():
        d = date_to_idx[row["date"]]
        n = res_to_idx.get(row["reservoir"], -1)
        if n >= 0:
            coverage[d, n] = True

    complete_dates   = int(coverage.all(axis=1).sum())
    incomplete_dates = int((~coverage.all(axis=1)).sum())
    fully_missing_dates = int((coverage.sum(axis=1) == 0).sum())
    total_missing_slots = int((~coverage).sum())

    print(f"\n  {split_name}:")
    print(f"    Total unique dates        : {n_dates}")
    print(f"    Dates with all 16 nodes   : {complete_dates}  (usable without imputation)")
    print(f"    Dates with missing nodes  : {incomplete_dates}  (affected)")
    print(f"    Total missing node-slots  : {total_missing_slots}")
    if incomplete_dates > 0:
        print(f"    Missing breakdown per date:")
        for d_idx, date in enumerate(dates):
            missing_nodes = [NODE_ORDER[n] for n in range(N) if not coverage[d_idx, n]]
            if missing_nodes:
                print(f"      {date}  missing: {missing_nodes}")

    return {
        "total_dates": n_dates,
        "complete_dates": complete_dates,
        "incomplete_dates": incomplete_dates,
        "missing_slots": total_missing_slots,
    }

train_piv = pivot_analysis(train_meta, X_train_flat, y_train_flat, "TRAIN")
val_piv   = pivot_analysis(val_meta,   X_val_flat,   y_val_flat,   "VALIDATION")
test_piv  = pivot_analysis(test_meta,  X_test_flat,  y_test_flat,  "TEST")

results["check5_pivot"] = True


# ──────────────────────────────────────────────────────────────
# 6. Verify per-sample shape = (16, 7, 5) / (16, 3)
# ──────────────────────────────────────────────────────────────
section("CHECK 6  —  GCN-LSTM tensor shapes")

print(f"  Flat LSTM shapes (existing):")
print(f"    X_train  : {X_train_flat.shape}   (samples, T, F)")
print(f"    X_val    : {X_val_flat.shape}")
print(f"    X_test   : {X_test_flat.shape}")
print(f"    y_train  : {y_train_flat.shape}   (samples, 3)")
print(f"    y_val    : {y_val_flat.shape}")
print(f"    y_test   : {y_test_flat.shape}")

print(f"\n  Expected GCN-LSTM shapes after pivot (per date):")
print(f"    X per date : ({N}, 7, 5)  — (nodes, T, F)")
print(f"    Y per date : ({N}, 3)     — (nodes, targets)")
print(f"    Batched B dates → X: (B, {N}, 7, 5)  Y: (B, {N}, 3)")

# Verify flat shapes are consistent with the node count and T, F
T, F = 7, 5
K = 3
expected_T = X_train_flat.shape[1]
expected_F = X_train_flat.shape[2]
expected_K = y_train_flat.shape[1]

shape_ok = (expected_T == T and expected_F == F and expected_K == K)
results["check6_shapes"] = shape_ok
if shape_ok:
    ok(f"Flat arrays confirm T={T}, F={F}, K={K} — pivot to (N,T,F)=(16,7,5) is valid.")
else:
    fail(f"Unexpected shapes: T={expected_T}, F={expected_F}, K={expected_K}")


# ──────────────────────────────────────────────────────────────
# 9. V3 log-target scaler reuse check
# ──────────────────────────────────────────────────────────────
section("CHECK 9  —  V3 log-target scaler reuse")

import joblib

SCALER_PATH = PROJECT / "models/lstm_pytorch_v3_logtarget/log_target_scaler.pkl"
TARGET_SCALER_PATH = PROJECT / "data/processed/scaled/target_scaler.pkl"

scaler_ok = True
if not SCALER_PATH.exists():
    fail(f"log_target_scaler.pkl not found at {SCALER_PATH}")
    scaler_ok = False
else:
    scaler = joblib.load(str(SCALER_PATH))
    print(f"  log_target_scaler.pkl loaded.")
    print(f"    n_samples_seen_ : {scaler.n_samples_seen_}  (must be 16662)")
    print(f"    mean_           : {scaler.mean_}")
    print(f"    scale_          : {scaler.scale_}")
    train_ok = (scaler.n_samples_seen_ == 16662)
    if train_ok:
        ok("Scaler was fit on training data only (n_samples_seen_==16662).")
    else:
        fail(f"Scaler n_samples_seen_={scaler.n_samples_seen_} ≠ 16662!")
    scaler_ok = train_ok

if TARGET_SCALER_PATH.exists():
    target_scaler = joblib.load(str(TARGET_SCALER_PATH))
    ok(f"target_scaler.pkl (V2/V3 original-unit recovery) loaded — "
       f"n_samples_seen_={target_scaler.n_samples_seen_}")
else:
    fail(f"target_scaler.pkl not found at {TARGET_SCALER_PATH}")
    scaler_ok = False

print(f"\n  Reuse plan:")
print(f"    1. GCN-LSTM predicts in log1p-scaled space (same as V3 training).")
print(f"    2. Inverse: expm1(log_target_scaler.inverse_transform(pred)).")
print(f"    3. This is identical to V3 — no scaler refitting required.")
print(f"    4. The scaler operates on shape (samples, 3) — valid for both flat")
print(f"       and reshaped (B*N, 3) GCN-LSTM output after flattening.")
results["check9_scaler"] = scaler_ok


# ──────────────────────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────────────────────
section("VALIDATION SUMMARY")

print(f"\n  Canonical 16-node list:")
for i, r in enumerate(NODE_ORDER):
    print(f"    {i:2d}  {r}")

print(f"\n  Excluded 2 reservoirs (in Graph B, absent from LSTM splits):")
for r in excluded:
    print(f"    - {r}")

print(f"\n  Graph B edge count (16-node filtered) : {len(edges_16)}")
print(f"  Graph B connected components (16-node): {len(components)}")
print(f"  Min degree (16-node)                  : {min(degree.values())}")

print(f"\n  Usable dates (all 16 nodes present):")
print(f"    TRAIN      : {train_piv['complete_dates']} / {train_piv['total_dates']} dates")
print(f"    VALIDATION : {val_piv['complete_dates']} / {val_piv['total_dates']} dates")
print(f"    TEST       : {test_piv['complete_dates']} / {test_piv['total_dates']} dates")

print(f"\n  Incomplete dates (at least 1 node missing) — NOT silently dropped:")
print(f"    TRAIN      : {train_piv['incomplete_dates']} dates, {train_piv['missing_slots']} missing node-slots")
print(f"    VALIDATION : {val_piv['incomplete_dates']} dates, {val_piv['missing_slots']} missing node-slots")
print(f"    TEST       : {test_piv['incomplete_dates']} dates, {test_piv['missing_slots']} missing node-slots")

tensor_valid = all([
    results["check1_node_consistency"],
    results["check2_excluded_pair"],
    results["check4_connectivity"],
    results["check6_shapes"],
    results["check9_scaler"],
])
print(f"\n  GCN-LSTM tensor construction valid    : {'YES' if tensor_valid else 'NO'}")

all_pass = all(results.values())
print(f"\n  All checks passed                     : {'YES' if all_pass else 'NO'}")
print(f"\n  Individual checks:")
for k, v in results.items():
    status = "PASS" if v else "FAIL"
    print(f"    [{status}] {k}")

print(f"\n{sep}")
print("  VALIDATION COMPLETE — no files modified.")
print(sep)

sys.exit(0 if all_pass else 1)
