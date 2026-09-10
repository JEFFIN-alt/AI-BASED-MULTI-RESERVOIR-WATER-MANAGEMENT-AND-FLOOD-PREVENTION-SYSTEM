"""
Milestone 12.3 — Graph B: Geographic k-nearest-neighbour graph
===============================================================
Edge rule  : each reservoir is connected to its k=3 geographically
             nearest reservoirs (Haversine great-circle distance).
             Edges are undirected; the union of all k-NN directed
             pairs is taken, so some nodes may have degree > k.

Data window: coordinates taken from training period only
             (date <= TRAIN_END) — consistent with Graph A.

Outputs (data/processed/graph/graph_B/):
    edges.csv        — source, target, distance_km
    distances.csv    — full N×N pairwise distance table
    metadata.json

Results (results/graph/graph_B/):
    degree.csv
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths & hyper-parameters
# ---------------------------------------------------------------------------
PROJECT = Path(__file__).resolve().parents[2]

DATA_FILE  = PROJECT / "data/processed/kerala_reservoir_clean.csv"
GRAPH_DIR  = PROJECT / "data/processed/graph/graph_B"
RESULT_DIR = PROJECT / "results/graph/graph_B"

TRAIN_END = pd.Timestamp("2023-12-31")
K         = 3   # neighbours per reservoir


# ---------------------------------------------------------------------------
# Haversine distance (km) — pure numpy, no scipy dependency needed
# ---------------------------------------------------------------------------
def haversine_matrix(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Return N×N symmetric distance matrix in kilometres."""
    R     = 6_371.0
    lat_r = np.radians(lats)
    lon_r = np.radians(lons)

    dlat = lat_r[:, None] - lat_r[None, :]
    dlon = lon_r[:, None] - lon_r[None, :]

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat_r[:, None]) * np.cos(lat_r[None, :]) * np.sin(dlon / 2) ** 2
    )
    return R * 2 * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------
def main() -> None:
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Coordinates — first recorded lat/lon per reservoir in training window
    # -----------------------------------------------------------------------
    df = pd.read_csv(DATA_FILE, parse_dates=["date"])
    df_train = df[df["date"] <= TRAIN_END].copy()

    coords = (
        df_train.groupby("reservoir")[["latitude", "longitude"]]
        .first()
        .dropna()
        .sort_index()
    )

    reservoirs = coords.index.tolist()
    lats       = coords["latitude"].values
    lons       = coords["longitude"].values
    n          = len(reservoirs)

    print(f"Reservoirs with coordinates: {n}")
    print(coords.to_string())

    # -----------------------------------------------------------------------
    # Pairwise Haversine distances
    # -----------------------------------------------------------------------
    dist_mat = haversine_matrix(lats, lons)
    np.fill_diagonal(dist_mat, np.inf)   # exclude self-loops

    dist_df = pd.DataFrame(dist_mat, index=reservoirs, columns=reservoirs)

    # -----------------------------------------------------------------------
    # k-NN edges (directed → undirected union)
    # -----------------------------------------------------------------------
    directed_pairs = set()
    for i, r in enumerate(reservoirs):
        nn_idx = np.argsort(dist_mat[i])[:K]
        for j in nn_idx:
            pair = (min(r, reservoirs[j]), max(r, reservoirs[j]))
            directed_pairs.add(pair)

    edges = []
    for a, b in sorted(directed_pairs):
        i = reservoirs.index(a)
        j = reservoirs.index(b)
        edges.append({
            "source"     : a,
            "target"     : b,
            "distance_km": round(float(dist_mat[i, j]), 3),
        })

    edges_df = pd.DataFrame(edges)

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
    dist_df.round(3).to_csv(GRAPH_DIR / "distances.csv")
    edges_df.to_csv(GRAPH_DIR / "edges.csv", index=False)
    degree_df.to_csv(RESULT_DIR / "degree.csv", index=False)

    metadata = {
        "graph"               : "B — geographic k-NN",
        "data_window"         : f"coordinates from training window <= {TRAIN_END.date()}",
        "leakage_safe"        : True,
        "k"                   : K,
        "distance_metric"     : "Haversine (great-circle) km",
        "edge_rule"           : f"undirected union of {K}-NN directed pairs by Haversine distance",
        "nodes"               : n,
        "reservoirs"          : reservoirs,
        "edges"               : len(edges_df),
        "connected_components": len(components),
        "components"          : components,
        "isolated_reservoirs" : isolated,
    }

    with open(GRAPH_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("MILESTONE 12.3 — GRAPH B: GEOGRAPHIC k-NN  (k=3)")
    print("=" * 70)
    print(f"Nodes       : {n}")
    print(f"Edges       : {len(edges_df)}")
    print(f"Isolated    : {isolated or 'None'}")
    print(f"Components  : {len(components)}")
    print()
    print("Degree (sorted):")
    print(degree_df.sort_values("degree", ascending=False).to_string(index=False))
    print()
    print("Edges (sorted by distance):")
    print(edges_df.sort_values("distance_km").to_string(index=False))

    print("\nSaved:")
    for p in [GRAPH_DIR / "edges.csv", GRAPH_DIR / "distances.csv",
              GRAPH_DIR / "metadata.json", RESULT_DIR / "degree.csv"]:
        print(" ", p)


if __name__ == "__main__":
    main()
