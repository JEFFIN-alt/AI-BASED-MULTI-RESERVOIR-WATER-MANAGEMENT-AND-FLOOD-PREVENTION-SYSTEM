"""
Milestone 12.3 — Graph C: Hybrid (correlation + geographic k-NN)
=================================================================
Edge rule  : union of Graph-A edges (correlation >= 0.70, overlap >= 1000)
             and Graph-B edges (k=3 geographic nearest neighbours).
             Each edge is annotated with its source(s): correlation,
             geo, or both.

This script READS the already-produced Graph A and Graph B edge files
rather than recomputing from scratch, so it must be run after both
build_graph_A_correlation.py and build_graph_B_geo_knn.py.

Outputs (data/processed/graph/graph_C/):
    edges.csv        — source, target, edge_type, correlation (if any),
                       distance_km (if any)
    metadata.json

Results (results/graph/graph_C/):
    degree.csv
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT = Path(__file__).resolve().parents[2]

GRAPH_A_DIR = PROJECT / "data/processed/graph/graph_A"
GRAPH_B_DIR = PROJECT / "data/processed/graph/graph_B"
GRAPH_C_DIR = PROJECT / "data/processed/graph/graph_C"
RESULT_DIR  = PROJECT / "results/graph/graph_C"

TRAIN_END = pd.Timestamp("2023-12-31")


# ---------------------------------------------------------------------------
def canonical_pair(a: str, b: str) -> tuple:
    return (min(a, b), max(a, b))


# ---------------------------------------------------------------------------
def main() -> None:
    GRAPH_C_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load Graph A edges
    # -----------------------------------------------------------------------
    edges_A = pd.read_csv(GRAPH_A_DIR / "edges.csv")
    meta_A  = json.loads((GRAPH_A_DIR / "metadata.json").read_text())
    reservoirs = meta_A["reservoirs"]   # canonical node list

    corr_pairs = {
        canonical_pair(r["source"], r["target"]): r
        for _, r in edges_A.iterrows()
    }

    # -----------------------------------------------------------------------
    # Load Graph B edges
    # -----------------------------------------------------------------------
    edges_B = pd.read_csv(GRAPH_B_DIR / "edges.csv")

    geo_pairs = {
        canonical_pair(r["source"], r["target"]): r
        for _, r in edges_B.iterrows()
    }

    # -----------------------------------------------------------------------
    # Union
    # -----------------------------------------------------------------------
    all_pairs = set(corr_pairs) | set(geo_pairs)

    edges = []
    for pair in sorted(all_pairs):
        a, b   = pair
        in_A   = pair in corr_pairs
        in_B   = pair in geo_pairs

        if in_A and in_B:
            edge_type = "both"
        elif in_A:
            edge_type = "correlation"
        else:
            edge_type = "geo"

        row = {
            "source"      : a,
            "target"      : b,
            "edge_type"   : edge_type,
            "correlation" : round(corr_pairs[pair]["correlation"], 6) if in_A else np.nan,
            "distance_km" : round(float(geo_pairs[pair]["distance_km"]), 3) if in_B else np.nan,
        }
        edges.append(row)

    edges_df = pd.DataFrame(edges)

    # -----------------------------------------------------------------------
    # Degree
    # -----------------------------------------------------------------------
    degree = {r: 0 for r in reservoirs}
    for _, row in edges_df.iterrows():
        if row["source"] in degree:
            degree[row["source"]] += 1
        if row["target"] in degree:
            degree[row["target"]] += 1

    degree_df = pd.DataFrame(
        {"reservoir": reservoirs, "degree": [degree[r] for r in reservoirs]}
    )

    # -----------------------------------------------------------------------
    # Connected components (BFS)
    # -----------------------------------------------------------------------
    adj = {r: set() for r in reservoirs}
    for _, row in edges_df.iterrows():
        if row["source"] in adj:
            adj[row["source"]].add(row["target"])
        if row["target"] in adj:
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
    # Edge-type breakdown
    # -----------------------------------------------------------------------
    type_counts = edges_df["edge_type"].value_counts().to_dict()

    # -----------------------------------------------------------------------
    # Persist
    # -----------------------------------------------------------------------
    edges_df.to_csv(GRAPH_C_DIR / "edges.csv", index=False)
    degree_df.to_csv(RESULT_DIR / "degree.csv", index=False)

    metadata = {
        "graph"               : "C — hybrid (correlation + geo k-NN)",
        "data_window"         : f"training only <= {TRAIN_END.date()}",
        "leakage_safe"        : True,
        "sources"             : {
            "graph_A": str(GRAPH_A_DIR / "edges.csv"),
            "graph_B": str(GRAPH_B_DIR / "edges.csv"),
        },
        "edge_rule"           : "union of Graph-A (correlation>=0.70, overlap>=1000) and Graph-B (k=3 geo k-NN)",
        "nodes"               : len(reservoirs),
        "reservoirs"          : reservoirs,
        "edges"               : len(edges_df),
        "edge_type_counts"    : type_counts,
        "connected_components": len(components),
        "components"          : components,
        "isolated_reservoirs" : isolated,
    }

    with open(GRAPH_C_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("MILESTONE 12.3 — GRAPH C: HYBRID (correlation ∪ geo k-NN)")
    print("=" * 70)
    print(f"Nodes         : {len(reservoirs)}")
    print(f"Edges         : {len(edges_df)}")
    print(f"  correlation only : {type_counts.get('correlation', 0)}")
    print(f"  geo only         : {type_counts.get('geo', 0)}")
    print(f"  both             : {type_counts.get('both', 0)}")
    print(f"Isolated      : {isolated or 'None'}")
    print(f"Components    : {len(components)}")
    print()
    print("Degree (sorted):")
    print(degree_df.sort_values("degree", ascending=False).to_string(index=False))
    print()
    print("Edges:")
    print(
        edges_df.sort_values(["edge_type", "correlation"], ascending=[True, False])
        .to_string(index=False)
    )

    print("\nSaved:")
    for p in [GRAPH_C_DIR / "edges.csv", GRAPH_C_DIR / "metadata.json",
              RESULT_DIR / "degree.csv"]:
        print(" ", p)


if __name__ == "__main__":
    main()
