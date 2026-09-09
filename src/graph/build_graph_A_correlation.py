"""
Milestone 12.3 — Graph A: Correlation-based reservoir graph
============================================================
Edge rule  : positive Pearson inflow correlation >= 0.70
             AND >= 1000 common co-observed dates
Data window: training period only  (date <= TRAIN_END = 2023-12-31)
             to avoid temporal leakage into the test window (2025-01-01+).

Outputs (data/processed/graph/graph_A/):
    edges.csv            — one row per edge
    correlation_all.csv  — full pairwise table (including sub-threshold pairs)
    metadata.json        — graph statistics

Results (results/graph/graph_A/):
    degree.csv
"""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths & hyper-parameters
# ---------------------------------------------------------------------------
PROJECT = Path(__file__).resolve().parents[2]

DATA_FILE  = PROJECT / "data/processed/kerala_reservoir_clean.csv"
GRAPH_DIR  = PROJECT / "data/processed/graph/graph_A"
RESULT_DIR = PROJECT / "results/graph/graph_A"

# Split boundary — must match src/data_quality/build_temporal_split.py
TRAIN_END = pd.Timestamp("2023-12-31")

MIN_OVERLAP     = 1_000   # minimum co-observed dates
MIN_CORRELATION = 0.70    # minimum positive Pearson r


# ---------------------------------------------------------------------------
def main() -> None:
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load & filter to training window only (leakage prevention)
    # -----------------------------------------------------------------------
    df = pd.read_csv(DATA_FILE, parse_dates=["date"])
    df_train = df[df["date"] <= TRAIN_END].copy()

    print(f"Full dataset   : {df['date'].min().date()} → {df['date'].max().date()}  ({len(df):,} rows)")
    print(f"Training window: {df_train['date'].min().date()} → {df_train['date'].max().date()}  ({len(df_train):,} rows)")

    reservoirs = sorted(df_train["reservoir"].dropna().unique())
    print(f"Reservoirs in training window: {len(reservoirs)}")

    # -----------------------------------------------------------------------
    # Pivot: one column per reservoir, indexed by date
    # -----------------------------------------------------------------------
    pivot = (
        df_train
        .pivot_table(index="date", columns="reservoir", values="inflow", aggfunc="mean")
        .reindex(columns=reservoirs)
        .sort_index()
    )

    # -----------------------------------------------------------------------
    # Pairwise correlation
    # -----------------------------------------------------------------------
    edges = []
    corr_rows = []

    for a, b in combinations(reservoirs, 2):
        pair    = pivot[[a, b]].dropna()
        overlap = len(pair)
        corr    = pair[a].corr(pair[b]) if overlap >= 2 else np.nan

        corr_rows.append({
            "reservoir_a"   : a,
            "reservoir_b"   : b,
            "overlap_days"  : overlap,
            "correlation"   : corr,
            "abs_correlation": abs(corr) if pd.notna(corr) else np.nan,
        })

        if overlap >= MIN_OVERLAP and pd.notna(corr) and corr >= MIN_CORRELATION:
            edges.append({
                "source"      : a,
                "target"      : b,
                "correlation" : round(corr, 6),
                "overlap_days": overlap,
            })

    corr_df  = pd.DataFrame(corr_rows)
    edges_df = pd.DataFrame(edges) if edges else pd.DataFrame(
        columns=["source", "target", "correlation", "overlap_days"]
    )

    # -----------------------------------------------------------------------
    # Degree
    # -----------------------------------------------------------------------
    degree = {r: 0 for r in reservoirs}
    for _, row in edges_df.iterrows():
        degree[row["source"]] += 1
        degree[row["target"]] += 1

    degree_df = pd.DataFrame(
        {"reservoir": reservoirs, "degree": [degree[r] for r in reservoirs]}
    )

    # -----------------------------------------------------------------------
    # Connected components (BFS)
    # -----------------------------------------------------------------------
    adj = {r: set() for r in reservoirs}
    for _, row in edges_df.iterrows():
        adj[row["source"]].add(row["target"])
        adj[row["target"]].add(row["source"])

    components, unseen = [], set(reservoirs)
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

    isolated = [r for r in reservoirs if degree[r] == 0]

    # -----------------------------------------------------------------------
    # Persist
    # -----------------------------------------------------------------------
    corr_df.to_csv(GRAPH_DIR / "correlation_all.csv", index=False)
    edges_df.to_csv(GRAPH_DIR / "edges.csv", index=False)
    degree_df.to_csv(RESULT_DIR / "degree.csv", index=False)

    metadata = {
        "graph"                   : "A — correlation",
        "data_window"             : f"≤ {TRAIN_END.date()} (training only)",
        "leakage_safe"            : True,
        "nodes"                   : len(reservoirs),
        "reservoirs"              : reservoirs,
        "edges"                   : len(edges_df),
        "min_overlap_days"        : MIN_OVERLAP,
        "min_positive_correlation": MIN_CORRELATION,
        "connected_components"    : len(components),
        "components"              : components,
        "isolated_reservoirs"     : isolated,
        "edge_rule"               : (
            f"positive Pearson inflow correlation >= {MIN_CORRELATION} "
            f"with >= {MIN_OVERLAP} common training-period dates"
        ),
    }

    with open(GRAPH_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("MILESTONE 12.3 — GRAPH A: CORRELATION  (leakage-safe)")
    print("=" * 70)
    print(f"Nodes            : {len(reservoirs)}")
    print(f"Edges            : {len(edges_df)}")
    print(f"Isolated         : {isolated or 'None'}")
    print(f"Components       : {len(components)}")
    print()
    print("Degree (sorted):")
    print(degree_df.sort_values("degree", ascending=False).to_string(index=False))
    print()
    print("Edges (sorted by correlation):")
    if not edges_df.empty:
        print(edges_df.sort_values("correlation", ascending=False).to_string(index=False))
    else:
        print("  (no edges)")

    print("\nSaved:")
    for p in [GRAPH_DIR / "edges.csv", GRAPH_DIR / "correlation_all.csv",
              GRAPH_DIR / "metadata.json", RESULT_DIR / "degree.csv"]:
        print(" ", p)


if __name__ == "__main__":
    main()
