import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]

DATA_FILE = PROJECT / "data/processed/kerala_reservoir_clean.csv"
GRAPH_DIR = PROJECT / "data/processed/graph"
RESULT_DIR = PROJECT / "results/graph"

MIN_OVERLAP = 1000
MIN_CORRELATION = 0.70


def main():
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_FILE)
    df["date"] = pd.to_datetime(df["date"])

    reservoirs = sorted(df["reservoir"].dropna().unique())

    # One coordinate per reservoir.
    coords = (
        df.groupby("reservoir")[["latitude", "longitude"]]
        .first()
        .reindex(reservoirs)
    )

    # Align inflow values by common dates.
    pivot = (
        df.pivot_table(
            index="date",
            columns="reservoir",
            values="inflow",
            aggfunc="mean",
        )
        .reindex(columns=reservoirs)
        .sort_index()
    )

    edges = []
    correlation_rows = []

    for a, b in combinations(reservoirs, 2):
        pair = pivot[[a, b]].dropna()

        overlap = len(pair)

        if overlap >= 2:
            corr = pair[a].corr(pair[b])
        else:
            corr = np.nan

        correlation_rows.append(
            {
                "reservoir_a": a,
                "reservoir_b": b,
                "overlap_days": overlap,
                "correlation": corr,
                "abs_correlation": abs(corr) if pd.notna(corr) else np.nan,
            }
        )

        # Candidate graph: strong positive relationship + substantial overlap.
        if (
            overlap >= MIN_OVERLAP
            and pd.notna(corr)
            and corr >= MIN_CORRELATION
        ):
            edges.append(
                {
                    "source": a,
                    "target": b,
                    "correlation": corr,
                    "overlap_days": overlap,
                }
            )

    correlation_df = pd.DataFrame(correlation_rows)
    edges_df = pd.DataFrame(edges)

    if edges_df.empty:
        edges_df = pd.DataFrame(
            columns=["source", "target", "correlation", "overlap_days"]
        )

    # Undirected degree.
    degree = {r: 0 for r in reservoirs}

    for _, row in edges_df.iterrows():
        degree[row["source"]] += 1
        degree[row["target"]] += 1

    degree_df = pd.DataFrame(
        {
            "reservoir": reservoirs,
            "degree": [degree[r] for r in reservoirs],
        }
    )

    # Connectivity using simple BFS.
    adjacency = {r: set() for r in reservoirs}

    for _, row in edges_df.iterrows():
        a, b = row["source"], row["target"]
        adjacency[a].add(b)
        adjacency[b].add(a)

    components = []
    unseen = set(reservoirs)

    while unseen:
        start = next(iter(unseen))
        stack = [start]
        component = set()

        while stack:
            node = stack.pop()

            if node in component:
                continue

            component.add(node)
            unseen.discard(node)

            stack.extend(adjacency[node] - component)

        components.append(sorted(component))

    # Save outputs.
    correlation_df.to_csv(
        GRAPH_DIR / "correlation_matrix.csv",
        index=False,
    )

    edges_df.to_csv(
        GRAPH_DIR / "reservoir_edges.csv",
        index=False,
    )

    degree_df.to_csv(
        RESULT_DIR / "graph_degree.csv",
        index=False,
    )

    metadata = {
        "nodes": len(reservoirs),
        "reservoirs": reservoirs,
        "candidate_edges": len(edges_df),
        "min_overlap_days": MIN_OVERLAP,
        "min_positive_correlation": MIN_CORRELATION,
        "connected_components": len(components),
        "components": components,
        "isolated_reservoirs": [
            r for r in reservoirs if degree[r] == 0
        ],
        "edge_rule": (
            "positive Pearson inflow correlation >= 0.70 "
            "with >= 1000 common dates"
        ),
    }

    with open(
        GRAPH_DIR / "graph_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, indent=2)

    print("=" * 90)
    print("MILESTONE 12.2 — RESERVOIR GRAPH CONSTRUCTION")
    print("=" * 90)

    print(f"\nNodes: {len(reservoirs)}")
    print(f"Candidate edges: {len(edges_df)}")
    print(f"Minimum overlap: {MIN_OVERLAP} days")
    print(f"Minimum positive correlation: {MIN_CORRELATION}")

    print("\nDegree:")
    print(degree_df.sort_values("degree", ascending=False).to_string(index=False))

    print("\nConnected components:", len(components))

    for i, component in enumerate(components, 1):
        print(f"  Component {i}: {len(component)} nodes")
        print("   ", ", ".join(component))

    print("\nIsolated reservoirs:")
    isolated = [r for r in reservoirs if degree[r] == 0]
    print(isolated if isolated else "None")

    print("\nEdges:")
    print(
        edges_df.sort_values(
            "correlation",
            ascending=False,
        ).to_string(index=False)
    )

    print("\nSaved:")
    print(GRAPH_DIR / "reservoir_edges.csv")
    print(GRAPH_DIR / "correlation_matrix.csv")
    print(GRAPH_DIR / "graph_metadata.json")
    print(RESULT_DIR / "graph_degree.csv")


if __name__ == "__main__":
    main()
