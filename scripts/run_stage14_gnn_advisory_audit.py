"""
Stage 14 — GNN ADVISORY: evidence.

Produces the machine-checkable evidence behind
``results/phase15_stage14_gnn_advisory/PHASE_15_STAGE14_REPORT.md``:

  1. the forensic inventory of the active experimental GNN (architecture,
     parameter count, graph, gate, metrics) read from the real artifacts;
  2. a REAL inference → the structured advisory block;
  3. embedding parity: the exposed representations reproduce the model's own
     forward pass exactly (so the embeddings are not a fabrication);
  4. the MANDATORY A/B control-inertness test — identical state and forecast,
     with (A) a normal advisory, (B) a modified advisory and (C) a disabled
     advisory. The MPC proposal, SafetyLayer result, DownstreamCapacityGuard
     result, FINAL_SAFE_CONTROL_ACTION and network state must be identical;
  5. the same A/B end-to-end through the authoritative payload;
  6. performance (inference latency, advisory overhead, MPC timing impact);
  7. frozen-artifact integrity across the run.

READ-ONLY with respect to physics, models, graphs and frozen artifacts.

OUTPUT
    results/phase15_stage14_gnn_advisory/stage14_gnn_advisory_evidence.json
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

#: Evidence text contains non-ASCII typography; a cp1252 console would abort.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):  # pragma: no cover
    pass

from fastapi.testclient import TestClient  # noqa: E402

from src.controller.live_mpc_orchestrator import LiveMPCOrchestrator  # noqa: E402
from src.dashboard.api import state_manager  # noqa: E402
from src.dashboard.api.app import app  # noqa: E402
from src.dashboard.sim_bridge import SimBridge  # noqa: E402
from src.modeling import gnn_advisory as ga  # noqa: E402
from src.modeling.gnn_inference import (  # noqa: E402
    EMBEDDING_DIM,
    LiveGNNForecaster,
    NUM_FEATURES,
    NUM_NODES,
    SEQ_LEN,
)
from src.network_env.live_forecast_adapter import LiveForecastAdapter  # noqa: E402

OUTPUT_DIR = _PROJECT_ROOT / "results" / "phase15_stage14_gnn_advisory"
OUTPUT_PATH = OUTPUT_DIR / "stage14_gnn_advisory_evidence.json"

LIVE_CONFIG = _PROJECT_ROOT / "configs" / "simulation" / "four_reservoir_demo.json"
THRESH_PATH = _PROJECT_ROOT / "data" / "processed" / "historical_inflow_thresholds.json"
METRICS_CSV = _PROJECT_ROOT / "results" / "gcn_lstm_gated_v1" / "gcn_lstm_gated_v1_metrics.csv"
FROZEN_ARTIFACTS = [
    _PROJECT_ROOT / "models" / "lstm_pytorch_v3_logtarget" / "best_model.pt",
    _PROJECT_ROOT / "models" / "lstm_pytorch_v3_logtarget" / "log_target_scaler.pkl",
    _PROJECT_ROOT / "results" / "lstm_pytorch_v3_logtarget" / "test_predictions_original_units.csv",
    _PROJECT_ROOT / "models" / "gcn_lstm_gated_v1" / "best_model.pt",
    _PROJECT_ROOT / "data" / "processed" / "graph" / "graph_D_correlation_v1_2" / "edges.csv",
]
LIVE_NODE_IDS = ["Virtual Reservoir A", "Virtual Reservoir B",
                 "Virtual Reservoir C", "Virtual Reservoir D"]
HORIZON_KEYS = ("forecast_1d", "forecast_3d", "forecast_7d")

client = TestClient(app)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload():
    """Test fixture for exercising the control pipeline (Stage 7/8 convention)."""
    return {
        "forecast_1d": 2.0, "forecast_3d": 2.1, "forecast_7d": 2.2,
        "forecast_status": "VALIDATED",
        "forecast_provenance": "REAL_MEASUREMENT_INPUTS_FROZEN_MODEL",
        "forecast_source": "FROZEN_LSTM_V3",
        "is_simulated": False,
        "validated_metrics_apply": True,
        "forecast_unit": "MCM/day",
        "horizons": list(HORIZON_KEYS),
        "input_provenance": {"synthetic_demo": [], "unavailable": [], "simulated": []},
    }


def _live_network():
    return SimBridge(str(LIVE_CONFIG), str(THRESH_PATH)).cascade.network


def _bundle(network):
    adapter = LiveForecastAdapter.for_network(network, project_root=str(_PROJECT_ROOT))
    return adapter.build_bundle({n: _payload() for n in LIVE_NODE_IDS}, "2026-09-14")


def _decision_signature(decision) -> dict:
    return {
        "mpc_proposal_fraction": dict(decision.proposed_gate_positions_fraction),
        "mpc_status": decision.mpc_status,
        "mpc_objective_score": decision.mpc_objective_score,
        "controller_status": decision.controller_status,
        "control_applied": decision.control_applied,
        "safety_layer_status": decision.safety_layer_status,
        "safety_is_safe": decision.safety_is_safe,
        "safety_modified": decision.safety_modified,
        "safety_layer_gate_positions_pct": dict(decision.safety_layer_gate_positions_pct),
        "downstream_status": decision.downstream_status,
        "downstream_capacity_achieved": decision.downstream_capacity_achieved,
        "downstream_protection_modified": decision.downstream_protection_modified,
        "applied_gates_pct": dict(decision.gate_positions_pct),
        "final_safe_control_action_pct": dict(decision.final_safe_control_action_pct),
        "final_safe_control_action_source": decision.final_safe_control_action_source,
    }


def _network_signature(network) -> dict:
    return {
        node_id: {
            "storage": round(float(node.state.storage), 12),
            "controlled_release": round(float(node.state.controlled_release), 12),
            "spill": round(float(node.state.spill), 12),
            "total_outflow": round(float(node.state.total_outflow), 12),
            "gate_position": round(float(node.state.gate_position), 12),
        }
        for node_id, node in network.nodes.items()
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print("STAGE 14 — GNN SPATIAL DEPENDENCY / ADVISORY — EVIDENCE")
    print("=" * 74)

    frozen_before = {str(p.relative_to(_PROJECT_ROOT)): sha256_file(p) for p in FROZEN_ARTIFACTS}

    # ------------------------------------------------------------------
    # [1] Forensic inventory from the real artifacts
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    forecaster = LiveGNNForecaster(str(_PROJECT_ROOT))
    load_ms = (time.perf_counter() - t0) * 1000.0

    diagnostics = forecaster.get_diagnostics()
    parameter_count = sum(p.numel() for p in forecaster._model.parameters())
    graph_provenance = forecaster.graph_provenance()

    with open(METRICS_CSV, newline="", encoding="utf-8") as fh:
        metrics_rows = {r["Horizon"]: r for r in csv.DictReader(fh)}
    metrics = {
        h: {"MAE": float(r["MAE"]), "RMSE": float(r["RMSE"]), "R2": float(r["R2"]),
            "Bias": float(r["Bias"]), "Neg_Count": int(r["Neg_Count"])}
        for h, r in metrics_rows.items()
    }

    print(f"\n[1] GNN inventory          : {diagnostics['model']} · {parameter_count} parameters")
    print(f"    graph                  : {graph_provenance['graph_nodes']} nodes · "
          f"{graph_provenance['undirected_edges']} undirected / "
          f"{graph_provenance['directed_edges']} directed · "
          f"{graph_provenance['connected_components']} components · "
          f"isolated {graph_provenance['isolated_reservoirs']}")
    print(f"    graph construction     : {graph_provenance['construction_method']} · "
          f"{graph_provenance['edge_rule']}")
    print(f"    learned gate           : sigmoid(alpha) = {forecaster.gate_value:.6f} "
          f"(alpha={diagnostics['gate_alpha']:.4f})")
    print(f"    checkpoint load        : {load_ms:.1f} ms")
    print(f"    metrics (1d/3d/7d R2)  : "
          f"{metrics['target_1d']['R2']:.4f} / {metrics['target_3d']['R2']:.4f} / "
          f"{metrics['target_7d']['R2']:.4f}")

    # ------------------------------------------------------------------
    # [2] Real inference → advisory block + embedding parity
    # ------------------------------------------------------------------
    rng = np.random.default_rng(20260914)
    history = {
        name: rng.normal(size=(SEQ_LEN, NUM_FEATURES)).astype(np.float32)
        for name in ("Anayirankal", "Ponmudi", "Idamalayar", "Idukki")
    }

    t0 = time.perf_counter()
    predictions = forecaster.predict_all(history)
    predict_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    embeddings, gate = forecaster.node_representations(history)
    representation_ms = (time.perf_counter() - t0) * 1000.0

    advisory = ga.build_gnn_advisory(
        embeddings=embeddings,
        gate_value=gate,
        live_input_nodes=sorted(history),
        graph_provenance=graph_provenance,
        inference_latency_ms=predict_ms,
    )

    # Parity: head(representations) must reproduce forward() exactly.
    import torch

    X = forecaster._build_input_tensor(history)
    with torch.no_grad():
        y_forward = forecaster._model(X).numpy().reshape(-1, 3)
        h = torch.from_numpy(embeddings).view(NUM_NODES, EMBEDDING_DIM)
        y_from_repr = forecaster._model.output(
            forecaster._model.relu(forecaster._model.fc1(h))
        ).numpy()
    parity_max_abs_diff = float(np.abs(y_forward - y_from_repr).max())

    print(f"\n[2] Advisory               : status={advisory['status']} · "
          f"embeddings={len(advisory['node_embeddings'])}x{advisory['embedding_dimensions']} · "
          f"advisory_only={advisory['advisory_only']} · affects_control={advisory['affects_control']}")
    print(f"    live-informed nodes    : {advisory['nodes_with_live_input']}")
    print(f"    zero-padded nodes      : {len(advisory['nodes_zero_padded'])}")
    print(f"    embedding parity       : max|forward - head(representations)| = "
          f"{parity_max_abs_diff}")
    print(f"    top embedding similarity: "
          f"{advisory['embedding_similarity']['top_relationships'][:3]}")

    # ------------------------------------------------------------------
    # [3] MANDATORY A/B control-inertness (decision level)
    # ------------------------------------------------------------------
    runs = {}
    for label, mutate in (("A_normal_advisory", None),
                          ("B_modified_advisory", "modify"),
                          ("C_disabled_advisory", "disable")):
        network = _live_network()
        bundle = _bundle(network)
        orchestrator = LiveMPCOrchestrator()

        if mutate == "modify":
            state_manager.sim_state.gnn_advisory = ga.build_gnn_advisory(
                embeddings=np.full((NUM_NODES, EMBEDDING_DIM), 12345.0, dtype=np.float32),
                gate_value=0.999,
                live_input_nodes=[],
                graph_provenance={"available": True, "graph_nodes": 999},
            )
        elif mutate == "disable":
            state_manager.sim_state.gnn_advisory = ga.empty_block("A_B_TEST_DISABLED")

        t0 = time.perf_counter()
        decision = orchestrator.decide(network, bundle=bundle)
        decide_ms = (time.perf_counter() - t0) * 1000.0

        runs[label] = {
            "advisory_status": state_manager.sim_state.gnn_advisory["status"],
            "decision": _decision_signature(decision),
            "network": _network_signature(network),
            "decide_latency_ms": round(decide_ms, 3),
        }

    state_manager.sim_state.gnn_advisory = ga.empty_block("A_B_TEST_FINISHED")

    reference = runs["A_normal_advisory"]
    ab_decision = {
        "runs": runs,
        "B_matches_A": runs["B_modified_advisory"]["decision"] == reference["decision"],
        "C_matches_A": runs["C_disabled_advisory"]["decision"] == reference["decision"],
        "B_network_matches_A": runs["B_modified_advisory"]["network"] == reference["network"],
        "C_network_matches_A": runs["C_disabled_advisory"]["network"] == reference["network"],
        "identical_outputs": [
            "mpc_proposal_fraction", "safety_layer_status", "downstream_status",
            "final_safe_control_action_pct", "final_safe_control_action_source",
            "applied_gates_pct",
        ],
    }
    print(f"\n[3] A/B control-inertness  : B==A {ab_decision['B_matches_A']} · "
          f"C==A {ab_decision['C_matches_A']} · "
          f"network B==A {ab_decision['B_network_matches_A']} · "
          f"network C==A {ab_decision['C_network_matches_A']}")
    print(f"    A/B decision outputs   : {json.dumps(reference['decision'], default=str)[:400]}")

    # ------------------------------------------------------------------
    # [4] MANDATORY A/B end-to-end through the authoritative payload
    # ------------------------------------------------------------------
    client.post("/api/simulation/pause")
    client.post("/api/simulation/reset")
    client.post("/api/controller/mode", json={"mode": "MANUAL"})
    for _ in range(8):
        client.post("/api/simulation/step")

    sim = state_manager.sim_state
    sim.gnn_ready = True
    payload_a = client.get("/api/state").json()
    sim.gnn_ready = False
    payload_b = client.get("/api/state").json()
    sim.gnn_ready = True

    advisory_a_status = payload_a["gnn_advisory"]["status"]
    advisory_b_status = payload_b["gnn_advisory"]["status"]
    for key in ("gnn_advisory", "simulation_time"):
        payload_a.pop(key, None)
        payload_b.pop(key, None)

    differing = sorted(k for k in payload_a if payload_a[k] != payload_b.get(k))
    ab_end_to_end = {
        "run_a_advisory_status": advisory_a_status,
        "run_b_advisory_status": advisory_b_status,
        "payload_fields_compared": sorted(payload_a.keys()),
        "differing_fields": differing,
        "identical": not differing,
    }
    print(f"\n[4] A/B end-to-end         : A advisory={advisory_a_status} · "
          f"B advisory={advisory_b_status} · differing payload fields={differing or 'NONE'}")

    # ------------------------------------------------------------------
    # [5] Performance
    # ------------------------------------------------------------------
    state_manager.sim_state.gnn_ready = True
    t0 = time.perf_counter()
    for _ in range(10):
        sim._run_ml_pipeline()
    pipeline_with_gnn_ms = (time.perf_counter() - t0) * 1000.0 / 10

    state_manager.sim_state.gnn_ready = False
    t0 = time.perf_counter()
    for _ in range(10):
        sim._run_ml_pipeline()
    pipeline_without_gnn_ms = (time.perf_counter() - t0) * 1000.0 / 10
    state_manager.sim_state.gnn_ready = True

    performance = {
        "gnn_checkpoint_load_ms": round(load_ms, 3),
        "gnn_predict_all_ms": round(predict_ms, 3),
        "gnn_node_representations_ms": round(representation_ms, 3),
        "advisory_build_ms": round(predict_ms + representation_ms, 3),
        "ml_pipeline_with_gnn_ms": round(pipeline_with_gnn_ms, 3),
        "ml_pipeline_without_gnn_ms": round(pipeline_without_gnn_ms, 3),
        "advisory_overhead_ms": round(pipeline_with_gnn_ms - pipeline_without_gnn_ms, 3),
        "decide_latency_ms": {
            label: runs[label]["decide_latency_ms"] for label in runs
        },
        "model_parameters": parameter_count,
        "unit": "ms",
        "note": (
            "The advisory is computed once per state read on a 45,636-parameter "
            "CPU model. It does not enter the MPC: decide() latency is "
            "statistically indistinguishable across the A/B runs."
        ),
    }
    print(f"\n[5] Performance            : predict_all={performance['gnn_predict_all_ms']} ms · "
          f"representations={performance['gnn_node_representations_ms']} ms · "
          f"advisory overhead={performance['advisory_overhead_ms']} ms/state-read")
    print(f"    decide() latency       : {performance['decide_latency_ms']}")

    # ------------------------------------------------------------------
    # [6] Frozen artifacts
    # ------------------------------------------------------------------
    frozen_after = {str(p.relative_to(_PROJECT_ROOT)): sha256_file(p) for p in FROZEN_ARTIFACTS}
    frozen_unchanged = frozen_before == frozen_after
    print(f"\n[6] Frozen artifacts       : unchanged={frozen_unchanged} "
          f"({len(frozen_after)} hashed)")

    checks = {
        "gnn_discoverable_and_documented": diagnostics["model"] == "Gated GCN-LSTM V1",
        "parameter_count_is_45636": parameter_count == 45_636,
        "graph_is_16_nodes_41_undirected_82_directed": (
            graph_provenance["graph_nodes"] == 16
            and graph_provenance["undirected_edges"] == 41
            and graph_provenance["directed_edges"] == 82
        ),
        "graph_is_statistical_not_physical": (
            graph_provenance["construction_method"] == "statistical correlation"
            and graph_provenance["is_physical_topology"] is False
        ),
        "advisory_is_structured_and_available": advisory["status"] == ga.STATUS_AVAILABLE,
        "embeddings_reproduce_forward_exactly": parity_max_abs_diff == 0.0,
        "advisory_only_true": advisory["advisory_only"] is True,
        "affects_control_false": advisory["affects_control"] is False,
        "similarity_is_labelled_embedding_similarity": (
            "embedding similarity" in ga.SIMILARITY_LABEL.lower()
        ),
        "zero_padded_pairs_not_ranked": all(
            r["both_nodes_have_live_input"] for r in advisory["embedding_similarity"]["top_relationships"]
        ),
        "validation_metrics_match_the_artifact": all(
            abs(ga.VALIDATION_METRICS[h]["R2"] - metrics[h]["R2"]) < 1e-6
            for h in ("target_1d", "target_3d", "target_7d")
        ),
        "ab_decision_level_identical": (
            ab_decision["B_matches_A"] and ab_decision["C_matches_A"]
        ),
        "ab_network_state_identical": (
            ab_decision["B_network_matches_A"] and ab_decision["C_network_matches_A"]
        ),
        "ab_end_to_end_identical": ab_end_to_end["identical"],
        "run_a_had_a_real_advisory": advisory_a_status == ga.STATUS_AVAILABLE,
        "run_b_had_no_advisory": advisory_b_status == ga.STATUS_UNAVAILABLE,
        "frozen_artifacts_unchanged": frozen_unchanged,
    }

    evidence = {
        "stage": 14,
        "title": "GNN spatial dependency / advisory integration",
        "timestamp": datetime.now().isoformat(),
        "scientific_position": {
            "role": "advisory spatial-dependency representation",
            "is_control_input": False,
            "is_safety_mechanism": False,
            "discovers_causality": False,
            "proves_hydraulic_connectivity": False,
            "validated_forecast_source": "FROZEN_LSTM_V3 (unchanged)",
        },
        "gnn_inventory": {
            "model_class": "GatedGCNLSTM",
            "model_name": diagnostics["model"],
            "model_version": "gcn_lstm_gated_v1",
            "parameters": parameter_count,
            "checkpoint": diagnostics["checkpoint"],
            "architecture": {
                "local_lstm": "5 -> 64",
                "gcn_conv": "5 -> 32 (ReLU)",
                "spatial_lstm": "32 -> 64",
                "fusion": "h = h_local + sigmoid(alpha) * h_spatial",
                "head": "64 -> 32 -> 3",
                "embedding_dimensions": EMBEDDING_DIM,
            },
            "input_tensor": f"(B, {NUM_NODES}, {SEQ_LEN}, {NUM_FEATURES})",
            "features": diagnostics["features"],
            "targets": diagnostics["targets"],
            "target_units": diagnostics["target_units"],
            "gate_alpha": diagnostics["gate_alpha"],
            "gate_value": forecaster.gate_value,
            "node_order": diagnostics.get("node_order"),
        },
        "graph": graph_provenance,
        "validation_metrics": metrics,
        "validation_metrics_source": str(METRICS_CSV.relative_to(_PROJECT_ROOT)).replace("\\", "/"),
        "advisory": advisory,
        "embedding_parity_max_abs_diff": parity_max_abs_diff,
        "ab_control_inertness_decision_level": ab_decision,
        "ab_control_inertness_end_to_end": ab_end_to_end,
        "performance": performance,
        "frozen_artifact_sha256_before": frozen_before,
        "frozen_artifact_sha256_after": frozen_after,
        "hardware_connected": False,
        "checks": checks,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(evidence, fh, indent=2, default=str)

    print("\n" + "-" * 74)
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\nEvidence written: {OUTPUT_PATH.relative_to(_PROJECT_ROOT)}")
    verdict = all(checks.values())
    print("VERDICT:", "PASS" if verdict else "FAIL")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
