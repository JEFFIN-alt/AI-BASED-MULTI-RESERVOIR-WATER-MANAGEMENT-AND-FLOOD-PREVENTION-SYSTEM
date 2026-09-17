"""
Stage 14 — GNN ADVISORY (spatial dependency representation)
===========================================================

Builds the structured, read-only ``gnn_advisory`` block that the authoritative
state payload exposes.

SCIENTIFIC POSITION (read this before using any value below)
-----------------------------------------------------------
The Gated GCN-LSTM V1 is a **statistical spatial-dependency representation**
trained on the 16-reservoir Kerala inflow dataset. It:

  * does **not** discover causal relationships,
  * does **not** prove hydraulic connectivity or routing,
  * does **not** control gates, and is **not** a safety mechanism,
  * is **not** part of the control path in any way.

A published, controlled experiment (``results/gcn_lstm_gated_v1/``) showed the
graph branch underperforming the temporal LSTM baseline, and the learned fusion
gate collapsed to ~0.015 — i.e. the model itself learned to ignore most of the
spatial signal. The GNN is therefore exposed as **advisory context only**.

Everything in this module is derived from real model inference or from the
graph's own recorded artifacts. Nothing is invented: when a value cannot be
produced legitimately the block reports ``UNAVAILABLE`` with a reason.

READ-ONLY. This module never writes a file, never touches the control path and
never imports the MPC, the SafetyLayer, the capacity guard or the physics.
"""

from __future__ import annotations

import datetime
import math
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.modeling.gnn_inference import (
    CANONICAL_NODE_ORDER,
    EMBEDDING_DIM,
    NUM_NODES,
)

MODEL_NAME = "Gated GCN-LSTM V1"
MODEL_VERSION = "gcn_lstm_gated_v1"
GRAPH_NAME = "Graph D (correlation_v1_2)"

#: How the exposed similarity is defined. Deliberately explicit: this is a
#: cosine between two LEARNED vectors, nothing hydraulic and nothing causal.
SIMILARITY_LABEL = (
    "embedding similarity (cosine between learned node representations) — "
    "NOT hydraulic influence, NOT causal influence, NOT physical connectivity"
)

GRAPH_PROVENANCE_NOTE = (
    "Statistical graph: an undirected edge joins two reservoirs whose inflow "
    "series have a positive Pearson correlation at or above the recorded "
    "threshold over at least the recorded number of common TRAINING-period "
    "dates. It is a correlation structure, not a hydraulic or routing network."
)

#: Why the full similarity matrix must be read with care.
ZERO_PADDED_ARTEFACT_NOTE = (
    "The model zero-pads nodes with no live input (its documented training "
    "contract). Two such nodes therefore receive IDENTICAL inputs and produce "
    "near-identical representations, i.e. an embedding similarity near 1.0 that "
    "reflects identical padding and carries no spatial meaning. Only pairs "
    "marked `both_nodes_have_live_input` are ranked as relationships; the full "
    "matrix is exposed for transparency and must be read with this in mind."
)

#: The authoritative PHYSICAL control topology (Stage 3), stated so that the
#: two structures can never be conflated. This is the verified live cascade,
#: separate from the 16-node experimental statistical graph above.
PHYSICAL_CONTROL_TOPOLOGY = {
    "description": (
        "Authoritative live cascade used by ReservoirNetwork for the Digital "
        "Twin. This is the PHYSICAL control topology."
    ),
    "nodes": ["Anayirankal", "Ponmudi", "Idamalayar", "Idukki"],
    "links": [
        {"from": "Anayirankal", "to": "Ponmudi", "routing_delay_steps": 2, "attenuation": 0.90},
        {"from": "Ponmudi", "to": "Idamalayar", "routing_delay_steps": 1, "attenuation": 0.85},
        {"from": "Idamalayar", "to": "Idukki", "routing_delay_steps": 1, "attenuation": 0.80},
    ],
    "terminal": "Idukki",
    "is_statistical_graph": False,
}

#: Provenance note making the distinction impossible to miss.
TWO_GRAPHS_NOTE = (
    "Two different structures are in play and must not be conflated: the "
    "PHYSICAL control topology (4 nodes, verified routing delays and "
    "attenuations, authoritative for simulation and control) and the "
    "EXPERIMENTAL STATISTICAL graph (16 nodes, 41 undirected correlations, "
    "used only to represent spatial dependency for the advisory model). The "
    "statistical graph is NOT the physical topology and vice versa."
)

ADVISORY_DISCLAIMER = (
    "ADVISORY ONLY. This block is not an input to the MPC, the SafetyLayer, "
    "the DownstreamCapacityGuard or the physics. Removing or altering it "
    "cannot change any control decision or any reservoir state."
)

#: Frozen validation metrics of the active experimental GNN, copied from
#: ``results/gcn_lstm_gated_v1/gcn_lstm_gated_v1_metrics.csv``. They are
#: published unchanged (a test asserts they equal the artifact), precisely so
#: the advisory block cannot be used to overstate the model.
VALIDATION_METRICS_SOURCE = "results/gcn_lstm_gated_v1/gcn_lstm_gated_v1_metrics.csv"
VALIDATION_METRICS = {
    "target_1d": {"MAE": 2.342946, "RMSE": 3.764655, "R2": 0.628784},
    "target_3d": {"MAE": 2.353850, "RMSE": 3.869269, "R2": 0.582139},
    "target_7d": {"MAE": 2.736001, "RMSE": 4.524306, "R2": 0.511439},
}
VALIDATION_METRICS_NOTE = (
    "Held-out test metrics of the experimental GNN, reported for transparency. "
    "They are NOT the validated forecasting baseline: the frozen LSTM V3 "
    "remains the validated forecast source, and its metrics are published "
    "separately. The validated V3 metrics do not apply to this model."
)

#: Statuses used by the block.
STATUS_AVAILABLE = "AVAILABLE"
STATUS_UNAVAILABLE = "UNAVAILABLE"
STATUS_WARMUP = "WARMUP_INSUFFICIENT_HISTORY"


def _finite(value, digits: int) -> Optional[float]:
    """Round to ``digits`` or return None — never emit a non-finite number."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return round(f, digits)


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Cosine similarity between every pair of node representations.

    ``similarity(i, j) = cosine(embedding_i, embedding_j)``.

    A zero-norm row (a node with no live input can still carry a non-zero
    representation, but a degenerate one could be zero) yields 0.0 rather than
    NaN, so the payload stays finite.
    """
    emb = np.asarray(embeddings, dtype=np.float64)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    safe = np.where(norms == 0.0, 1.0, norms)
    unit = emb / safe
    similarity = unit @ unit.T
    similarity[norms[:, 0] == 0.0, :] = 0.0
    similarity[:, norms[:, 0] == 0.0] = 0.0
    return similarity


def top_relationships(
    similarity: np.ndarray,
    node_names: Sequence[str],
    live_nodes: Sequence[str],
    top_k: int = 8,
) -> List[Dict]:
    """
    The strongest embedding-similarity pairs, ranked over LIVE-INFORMED nodes.

    Only pairs where BOTH endpoints received live data are ranked. This matters:
    the model's documented contract zero-pads absent nodes, so two zero-padded
    nodes receive identical inputs and therefore produce near-identical
    representations — a cosine close to 1.0 that reflects identical padding, not
    spatial structure. Ranking them would be a numerical artefact masquerading
    as a finding, so they are excluded from the ranking (and the caveat is
    stated in the block).

    Reported as "embedding similarity" only. A high value means two learned
    representations point in similar directions — nothing more.
    """
    live = set(live_nodes)
    pairs: List[Dict] = []
    n = len(node_names)
    for i in range(n):
        if node_names[i] not in live:
            continue
        for j in range(i + 1, n):
            if node_names[j] not in live:
                continue
            pairs.append({
                "source": node_names[i],
                "target": node_names[j],
                "embedding_similarity": _finite(similarity[i, j], 4),
                "both_nodes_have_live_input": True,
            })
    pairs.sort(key=lambda p: (p["embedding_similarity"] is None,
                              -(p["embedding_similarity"] or 0.0)))
    return pairs[:top_k]


def unavailable_block(status: str, reason: str, graph: Optional[dict] = None) -> Dict:
    """
    The honest empty case. No value is guessed or carried over from a previous
    inference: every field is null and the reason is stated.
    """
    graph = graph or {}
    return {
        "status": status,
        "reason": reason,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "graph": GRAPH_NAME,
        "graph_nodes": graph.get("graph_nodes"),
        "graph_undirected_edges": graph.get("undirected_edges"),
        "graph_directed_edges": graph.get("directed_edges"),
        "inference_timestamp": None,
        "embedding_dimensions": None,
        "node_embeddings": {},
        "nodes_with_live_input": [],
        "nodes_zero_padded": [],
        "embedding_similarity": None,
        "relationship_summary": None,
        "gate_value": None,
        "inference_latency_ms": None,
        "validation_metrics": VALIDATION_METRICS,
        "validation_metrics_source": VALIDATION_METRICS_SOURCE,
        "graph_provenance": graph or {"available": False},
        "physical_control_topology": PHYSICAL_CONTROL_TOPOLOGY,
        "advisory_only": True,
        "affects_control": False,
        "disclaimer": ADVISORY_DISCLAIMER,
    }


def empty_block(reason: str = "NOT_COMPUTED") -> Dict:
    """A block for a state that has never produced an advisory result."""
    return unavailable_block(STATUS_UNAVAILABLE, reason)


def build_gnn_advisory(
    *,
    embeddings: np.ndarray,
    gate_value: float,
    live_input_nodes: Sequence[str],
    graph_provenance: dict,
    inference_latency_ms: Optional[float] = None,
    inference_timestamp: Optional[str] = None,
) -> Dict:
    """
    Assemble the advisory block from a REAL inference result.

    Parameters
    ----------
    embeddings : np.ndarray
        ``(16, EMBEDDING_DIM)`` fused node representations in canonical order.
    gate_value : float
        ``sigmoid(alpha)`` — the learned scalar gate on the spatial branch.
    live_input_nodes : sequence of str
        Which nodes actually received live data. The remaining nodes are
        zero-padded by the model's documented contract, so their
        representations are NOT informed by live measurements and are reported
        as such rather than presented as live analysis.
    graph_provenance : dict
        Output of ``LiveGNNForecaster.graph_provenance()``.
    """
    emb = np.asarray(embeddings, dtype=np.float64)
    if emb.ndim != 2 or emb.shape[1] != EMBEDDING_DIM:
        return unavailable_block(
            STATUS_UNAVAILABLE,
            f"UNEXPECTED_EMBEDDING_SHAPE:{getattr(emb, 'shape', None)}",
            graph_provenance,
        )
    if not np.isfinite(emb).all():
        return unavailable_block(STATUS_UNAVAILABLE, "NON_FINITE_EMBEDDINGS", graph_provenance)

    node_names = list(CANONICAL_NODE_ORDER)
    similarity = cosine_similarity_matrix(emb)

    live = [n for n in node_names if n in set(live_input_nodes)]
    zero_padded = [n for n in node_names if n not in set(live_input_nodes)]

    embeddings_block = {
        name: [_finite(v, 4) for v in emb[idx]]
        for idx, name in enumerate(node_names[: emb.shape[0]])
    }

    return {
        "status": STATUS_AVAILABLE,
        "reason": None,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "graph": GRAPH_NAME,
        "graph_nodes": graph_provenance.get("graph_nodes", NUM_NODES),
        "graph_undirected_edges": graph_provenance.get("undirected_edges"),
        "graph_directed_edges": graph_provenance.get("directed_edges"),
        "inference_timestamp": inference_timestamp or datetime.datetime.now().isoformat(),
        "embedding_dimensions": int(emb.shape[1]),
        "node_embeddings": embeddings_block,
        "nodes_with_live_input": live,
        "nodes_zero_padded": zero_padded,
        "embedding_similarity": {
            "label": SIMILARITY_LABEL,
            "method": "cosine similarity between learned node representations",
            "node_order": node_names[: emb.shape[0]],
            "matrix": [[_finite(v, 3) for v in row] for row in similarity],
            "matrix_note": ZERO_PADDED_ARTEFACT_NOTE,
            "top_relationships": top_relationships(
                similarity, node_names[: emb.shape[0]], live
            ),
            "ranking_restricted_to": "pairs where both nodes received live input",
            "pairs_with_live_input": len(live) * (len(live) - 1) // 2,
        },
        "relationship_summary": {
            "graph_type": "statistical correlation graph",
            "graph_construction": graph_provenance.get("edge_rule"),
            "graph_nodes": graph_provenance.get("graph_nodes", NUM_NODES),
            "graph_undirected_edges": graph_provenance.get("undirected_edges"),
            "graph_directed_edges": graph_provenance.get("directed_edges"),
            "connected_components": graph_provenance.get("connected_components"),
            "isolated_reservoirs": graph_provenance.get("isolated_reservoirs", []),
            "learned_spatial_gate": _finite(gate_value, 6),
            "gate_interpretation": (
                "scalar gate on the spatial branch (sigmoid(alpha)). Near zero "
                "means the model itself learned to rely mostly on the local "
                "temporal branch; it is not a measure of physical influence."
            ),
            "nodes_with_live_input": live,
            "nodes_zero_padded": zero_padded,
            "note": GRAPH_PROVENANCE_NOTE,
        },
        "gate_value": _finite(gate_value, 6),
        "inference_latency_ms": _finite(inference_latency_ms, 3),
        "validation_metrics": VALIDATION_METRICS,
        "validation_metrics_source": VALIDATION_METRICS_SOURCE,
        "validation_metrics_note": VALIDATION_METRICS_NOTE,
        "graph_provenance": graph_provenance,
        "physical_control_topology": PHYSICAL_CONTROL_TOPOLOGY,
        "two_graphs_note": TWO_GRAPHS_NOTE,
        "advisory_only": True,
        "affects_control": False,
        "disclaimer": ADVISORY_DISCLAIMER,
    }


def default_block() -> Dict:
    """
    The block carried by the twin payload when the advisory is not attached.

    Same schema as the real block (so no frontend has to branch), with every
    value null and an explicit reason. This is the ``state_adapter`` default.
    """
    return unavailable_block(STATUS_UNAVAILABLE, "NO_GNN_ADVISORY_PROVENANCE")
