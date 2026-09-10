import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths & hyper-parameters
# ---------------------------------------------------------------------------
PROJECT = Path(__file__).resolve().parents[2]

DATA_FILE  = PROJECT / "data/processed/splits/train.csv"
GRAPH_DIR  = PROJECT / "data/processed/graph/graph_D_correlation_v1_2"

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
MIN_OVERLAP     = 365   # minimum co-observed dates (1 year)
MIN_CORRELATION = 0.55  # minimum positive Pearson r

def main():
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load & filter to training window only (leakage prevention)
    # The train split already only contains training data (date <= 2023-12-31)
    # -----------------------------------------------------------------------
    df_train = pd.read_csv(DATA_FILE, parse_dates=["date"])
    
    # Restrict to the 16 reservoirs present in the LSTM pipeline
    # The train split already only contains these 16.
    reservoirs = sorted(df_train["reservoir"].dropna().unique())
    print(f"Reservoirs in training window: {len(reservoirs)}")

    # -----------------------------------------------------------------------
    # Pivot: one column per reservoir, indexed by date
    # -----------------------------------------------------------------------
    pivot = (
        df_train
        .pivot_table(index="date", columns="reservoir", values="inflow_day_1", aggfunc="mean")
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
    # Degree & Density
    # -----------------------------------------------------------------------
    degree = {r: 0 for r in reservoirs}
    for _, row in edges_df.iterrows():
        degree[row["source"]] += 1
        degree[row["target"]] += 1

    degree_df = pd.DataFrame(
        {"reservoir": reservoirs, "degree": [degree[r] for r in reservoirs]}
    )

    max_edges = len(reservoirs) * (len(reservoirs) - 1) / 2
    density = len(edges_df) / max_edges if max_edges > 0 else 0

    # -----------------------------------------------------------------------
    # Connected components (BFS)
    # -----------------------------------------------------------------------
    adj = {r: set() for r in reservoirs}
    for _, row in edges_df.iterrows():
        adj[row["source"]].add(row["target"])
        adj[row["target"]].add(row["source"])

    components, unseen = [], set(reservoirs)
    while unseen:
        start = next(iter(unseen))
        stack, comp = [start], set()
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
    edges_df.to_csv(GRAPH_DIR / "edges.csv", index=False)

    metadata = {
        "graph"                   : "D \u2014 correlation_v1_2",
        "data_window"             : "training only (via train.csv split)",
        "leakage_safe"            : True,
        "nodes"                   : len(reservoirs),
        "reservoirs"              : reservoirs,
        "edges"                   : len(edges_df),
        "min_overlap_days"        : MIN_OVERLAP,
        "min_positive_correlation": MIN_CORRELATION,
        "connected_components"    : len(components),
        "components"              : components,
        "isolated_reservoirs"     : isolated,
        "min_degree"              : int(degree_df['degree'].min()),
        "max_degree"              : int(degree_df['degree'].max()),
        "average_degree"          : float(degree_df['degree'].mean()),
        "density"                 : float(density),
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
    print("\nMILESTONE 12.7A COMPLETE\n")
    print("GRAPH:")
    print(f"Nodes: {metadata['nodes']}")
    print(f"Edges: {metadata['edges']}")
    print(f"Threshold: {metadata['min_positive_correlation']}")
    print(f"Minimum overlap: {metadata['min_overlap_days']}")
    print(f"Connected components: {metadata['connected_components']}")
    print(f"Isolated nodes: {len(metadata['isolated_reservoirs'])}")
    print(f"Min degree: {metadata['min_degree']}")
    print(f"Max degree: {metadata['max_degree']}")
    print(f"Average degree: {metadata['average_degree']:.2f}")
    print(f"Density: {metadata['density']:.3f}")
    print(f"Leakage status: SAFE (computed entirely on training subset)\n")

if __name__ == "__main__":
    main()
