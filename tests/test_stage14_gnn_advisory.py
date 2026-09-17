"""
Stage 14 — GNN SPATIAL DEPENDENCY / ADVISORY: tests.

Proves the Gated GCN-LSTM V1 is integrated as an ADVISORY spatial-dependency
component and that it cannot reach control:

  * the implementation, graph and metrics are discoverable and documented;
  * real inference produces a structured advisory block with provenance;
  * ``advisory_only == True`` and ``affects_control == False`` everywhere;
  * the advisory cannot change gates, bypass the MPC / SafetyLayer /
    DownstreamCapacityGuard, or perturb ReservoirNetwork or the
    MassBalanceMonitor;
  * the frozen LSTM V3 remains the validated forecast source;
  * the 16-node statistical graph is never presented as the 4-node physical
    control topology (and the declared physical topology is cross-checked
    against the live network's own connections);
  * similarity is labelled as embedding similarity only;
  * no browser-side and no Streamlit-side inference exists;
  * the MANDATORY A/B control-inertness test: identical state + forecast with
    (A) a normal advisory and (B) a modified/disabled advisory produce an
    identical MPC proposal, SafetyLayer result, capacity-guard result,
    FINAL_SAFE_CONTROL_ACTION and network state.
"""

import ast
import copy
import csv
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.controller.live_mpc_orchestrator import LiveMPCOrchestrator  # noqa: E402
from src.dashboard.api import state_manager  # noqa: E402
from src.dashboard.api.app import app  # noqa: E402
from src.dashboard.sim_bridge import SimBridge  # noqa: E402
from src.modeling import gnn_advisory as ga  # noqa: E402
from src.modeling.gnn_inference import (  # noqa: E402
    CANONICAL_NODE_ORDER,
    EMBEDDING_DIM,
    GatedGCNLSTM,
    LiveGNNForecaster,
    NUM_NODES,
    SEQ_LEN,
    NUM_FEATURES,
)
from src.network_env.live_forecast_adapter import LiveForecastAdapter  # noqa: E402

client = TestClient(app)

PROJECT_ROOT = str(_PROJECT_ROOT)
LIVE_CONFIG = _PROJECT_ROOT / "configs" / "simulation" / "four_reservoir_demo.json"
THRESH_PATH = _PROJECT_ROOT / "data" / "processed" / "historical_inflow_thresholds.json"
METRICS_CSV = _PROJECT_ROOT / "results" / "gcn_lstm_gated_v1" / "gcn_lstm_gated_v1_metrics.csv"
GRAPH_DIR = _PROJECT_ROOT / "data" / "processed" / "graph" / "graph_D_correlation_v1_2"
CHECKPOINT = _PROJECT_ROOT / "models" / "gcn_lstm_gated_v1" / "best_model.pt"
GATE_DIAG = _PROJECT_ROOT / "results" / "gcn_lstm_gated_v1" / "gate_diagnostics.json"

APP_PATH = _PROJECT_ROOT / "src" / "dashboard" / "app.py"
WEB_DIR = _PROJECT_ROOT / "src" / "dashboard" / "web"
ROUTES_PATH = _PROJECT_ROOT / "src" / "dashboard" / "api" / "routes.py"
STATE_MANAGER_PATH = _PROJECT_ROOT / "src" / "dashboard" / "api" / "state_manager.py"
ORCHESTRATOR_PATH = _PROJECT_ROOT / "src" / "controller" / "live_mpc_orchestrator.py"
NETWORK_PATH = _PROJECT_ROOT / "src" / "network_env" / "reservoir_network.py"
MASS_BALANCE_PATH = _PROJECT_ROOT / "src" / "network_env" / "mass_balance.py"
ADAPTER_PATH = _PROJECT_ROOT / "src" / "network_env" / "gnn_forecast_adapter.py"

LIVE_NODE_IDS = ["Virtual Reservoir A", "Virtual Reservoir B",
                 "Virtual Reservoir C", "Virtual Reservoir D"]
HORIZON_KEYS = ("forecast_1d", "forecast_3d", "forecast_7d")
FORECAST_UNIT = "MCM/day"

#: Declared physical control topology (Stage 3) — delays / attenuations.
PHYSICAL_DELAYS = [2, 1, 1]
PHYSICAL_ATTENUATION = [0.90, 0.85, 0.80]


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def forecaster():
    return LiveGNNForecaster(PROJECT_ROOT)


@pytest.fixture(scope="module")
def gnn_history():
    """A real-shaped (7, 5) scaled window for the four live reservoirs."""
    rng = np.random.default_rng(20260914)
    return {
        name: rng.normal(size=(SEQ_LEN, NUM_FEATURES)).astype(np.float32)
        for name in ("Anayirankal", "Ponmudi", "Idamalayar", "Idukki")
    }


@pytest.fixture(autouse=True)
def _restore_authoritative_state():
    """
    Stage 14 tests drive the ONE authoritative singleton (stepping it, toggling
    ``gnn_ready``). Left alone, the network can end up saturated and spilling,
    where storages are capped and a further step cannot change them — which
    would break unrelated suites that assert "a step advances physics".

    This finalizer restores STATE ONLY (paused, 50 % storages, MANUAL). It
    weakens no assertion anywhere; it simply stops this module from leaving the
    shared simulation in a degenerate corner.
    """
    yield
    sim = state_manager.sim_state
    sim.gnn_ready = True
    client.post("/api/simulation/pause")
    client.post("/api/simulation/reset")
    client.post("/api/controller/mode", json={"mode": "MANUAL"})


@pytest.fixture
def stepped_sim():
    """The ONE authoritative instance, paused and advanced 8 real steps."""
    client.post("/api/simulation/pause")
    client.post("/api/simulation/reset")
    client.post("/api/controller/mode", json={"mode": "MANUAL"})
    for _ in range(8):
        assert client.post("/api/simulation/step").status_code == 200
    return state_manager.sim_state


def _advisory_from_live() -> dict:
    return client.get("/api/state").json()["gnn_advisory"]


def _payload(*, status="VALIDATED",
             provenance="REAL_MEASUREMENT_INPUTS_FROZEN_MODEL",
             validated=True, simulated=False):
    """
    A forecast payload used ONLY to exercise the control pipeline in tests.

    This mirrors the convention already established by the Stage 7/8/9/10
    suites. It is a test fixture for driving the pipeline, not a claim about
    live forecast provenance.
    """
    return {
        "forecast_1d": 2.0, "forecast_3d": 2.1, "forecast_7d": 2.2,
        "forecast_status": status,
        "forecast_provenance": provenance,
        "forecast_source": "FROZEN_LSTM_V3",
        "is_simulated": simulated,
        "validated_metrics_apply": validated,
        "forecast_unit": FORECAST_UNIT,
        "horizons": list(HORIZON_KEYS),
        "input_provenance": {"synthetic_demo": [], "unavailable": [], "simulated": []},
    }


def _live_network():
    return SimBridge(str(LIVE_CONFIG), str(THRESH_PATH)).cascade.network


def _validated_bundle(network):
    adapter = LiveForecastAdapter.for_network(network, project_root=PROJECT_ROOT)
    return adapter.build_bundle({n: _payload() for n in LIVE_NODE_IDS}, "2026-09-14")


def _decision_signature(decision) -> dict:
    """Every output the A/B test must find identical."""
    return {
        "mpc_proposal_fraction": dict(decision.proposed_gate_positions_fraction),
        "mpc_status": decision.mpc_status,
        "mpc_objective_score": decision.mpc_objective_score,
        "mpc_forecast_used": decision.mpc_forecast_used,
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
    """The raw ReservoirNetwork state: storage, outflow, spill and gate."""
    return {
        node_id: {
            "storage": float(node.state.storage),
            "controlled_release": float(node.state.controlled_release),
            "spill": float(node.state.spill),
            "total_outflow": float(node.state.total_outflow),
            "gate_position": float(node.state.gate_position),
        }
        for node_id, node in network.nodes.items()
    }


def _imported_modules(path: Path) -> set:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            modules.add(node.module or "")
            modules |= {f"{node.module}.{a.name}" for a in node.names}
    return modules


# ===========================================================================
# 1. Discoverable, documented implementation
# ===========================================================================

def test_gnn_implementation_is_discoverable_and_documented(forecaster):
    """Requirement 1 — the active experimental implementation is identifiable."""
    assert CHECKPOINT.exists(), "GNN checkpoint missing"
    assert (GRAPH_DIR / "edges.csv").exists() and (GRAPH_DIR / "metadata.json").exists()
    assert isinstance(forecaster._model, GatedGCNLSTM)

    diag = forecaster.get_diagnostics()
    assert diag["model"] == "Gated GCN-LSTM V1"
    assert diag["graph"] == "Graph D (correlation_v1_2)"
    assert diag["nodes"] == 16
    assert diag["edges_undirected"] == 41
    assert diag["edges_directed"] == 82
    assert diag["sequence_length"] == 7
    assert diag["features"] == ["inflow", "water_level", "live_storage",
                                "rainfall", "total_outflow"]
    assert diag["target_units"] == "MCM/day"


def test_gnn_parameter_count_and_gate_match_the_published_facts(forecaster):
    """The documented 45,636-parameter architecture and the learned gate."""
    total = sum(p.numel() for p in forecaster._model.parameters())
    assert total == 45_636, f"parameter count changed: {total}"

    assert forecaster.gate_value == pytest.approx(0.0151, abs=1e-4)
    assert forecaster._model.alpha.item() == pytest.approx(-4.1779, abs=1e-4)


def test_graph_provenance_records_a_statistical_construction(forecaster):
    """Requirement 15 — the graph's provenance is preserved verbatim."""
    prov = forecaster.graph_provenance()
    assert prov["available"] is True
    assert prov["graph_nodes"] == 16
    assert prov["undirected_edges"] == 41
    assert prov["directed_edges"] == 82
    assert prov["connected_components"] == 2
    assert prov["isolated_reservoirs"] == ["Pamba"]
    assert prov["construction_method"] == "statistical correlation"
    assert prov["min_positive_correlation"] == 0.55
    assert prov["min_overlap_days"] == 365
    assert prov["data_window"] == "training only (via train.csv split)"
    assert prov["leakage_safe"] is True
    assert prov["is_physical_topology"] is False


def test_validation_metrics_match_the_published_artifact():
    """Requirement 5 — the metrics are copied from the artifact, not restated."""
    with open(METRICS_CSV, newline="", encoding="utf-8") as fh:
        rows = {r["Horizon"]: r for r in csv.DictReader(fh)}

    for horizon, values in ga.VALIDATION_METRICS.items():
        row = rows[horizon]
        assert values["MAE"] == pytest.approx(float(row["MAE"]), abs=1e-6)
        assert values["RMSE"] == pytest.approx(float(row["RMSE"]), abs=1e-6)
        assert values["R2"] == pytest.approx(float(row["R2"]), abs=1e-6)

    # And the published gate diagnostics agree with the checkpoint.
    import json
    assert json.loads(GATE_DIAG.read_text())["final_gate"] == pytest.approx(0.0151, abs=1e-4)


# ===========================================================================
# 2. Structured advisory result with provenance
# ===========================================================================

def test_inference_produces_a_structured_advisory_result(forecaster, gnn_history):
    """Requirement 2 — a real inference yields the documented structure."""
    embeddings, gate = forecaster.node_representations(gnn_history)
    block = ga.build_gnn_advisory(
        embeddings=embeddings,
        gate_value=gate,
        live_input_nodes=sorted(gnn_history),
        graph_provenance=forecaster.graph_provenance(),
        inference_latency_ms=forecaster.last_inference_time_ms,
    )

    for key in ("status", "model_name", "model_version", "graph_nodes", "graph_undirected_edges",
                "graph_directed_edges", "inference_timestamp", "embedding_dimensions",
                "node_embeddings", "relationship_summary", "graph_provenance",
                "advisory_only", "affects_control"):
        assert key in block, f"advisory block is missing {key}"

    assert block["status"] == ga.STATUS_AVAILABLE
    assert block["embedding_dimensions"] == EMBEDDING_DIM == 64
    assert len(block["node_embeddings"]) == NUM_NODES
    assert all(len(v) == EMBEDDING_DIM for v in block["node_embeddings"].values())
    assert block["gate_value"] == pytest.approx(0.0151, abs=1e-4)


def test_node_representations_reproduce_the_models_own_forward_pass(forecaster, gnn_history):
    """
    The embeddings are REAL model internals, not a fabrication: the model's own
    FC head applied to the exposed representations reproduces ``forward``'s
    output exactly.
    """
    import torch

    embeddings, gate = forecaster.node_representations(gnn_history)
    X = forecaster._build_input_tensor(gnn_history)
    with torch.no_grad():
        y_forward = forecaster._model(X).numpy().reshape(-1, 3)
        h = torch.from_numpy(embeddings).view(1 * NUM_NODES, EMBEDDING_DIM)
        y_from_representations = forecaster._model.output(
            forecaster._model.relu(forecaster._model.fc1(h))
        ).numpy()

    assert np.abs(y_forward - y_from_representations).max() == 0.0
    assert gate == pytest.approx(forecaster.gate_value)


def test_advisory_contains_provenance_and_flags(forecaster, gnn_history):
    """Requirements 3, 4, 5, 15 — provenance plus the two safety flags."""
    embeddings, gate = forecaster.node_representations(gnn_history)
    block = ga.build_gnn_advisory(
        embeddings=embeddings, gate_value=gate,
        live_input_nodes=sorted(gnn_history),
        graph_provenance=forecaster.graph_provenance(),
    )
    assert block["advisory_only"] is True          # requirement 4
    assert block["affects_control"] is False       # requirement 5
    assert "ADVISORY ONLY" in block["disclaimer"]
    assert block["graph_provenance"]["leakage_safe"] is True
    assert block["validation_metrics_source"] == ga.VALIDATION_METRICS_SOURCE
    assert block["model_name"] == "Gated GCN-LSTM V1"


def test_similarity_is_labelled_as_embedding_similarity_only():
    """
    Requirement 16 — the similarity is named for what it is, and every mention
    of a stronger claim appears only as an explicit NEGATION.
    """
    label = ga.SIMILARITY_LABEL.lower()
    assert "embedding similarity" in label
    for negated in ("not hydraulic influence", "not causal influence",
                    "not physical connectivity"):
        assert negated in label, f"the label must explicitly deny {negated[4:]!r}"

    # The same restriction applies to the strings the UI renders.
    for text in (ga.ADVISORY_DISCLAIMER, ga.GRAPH_PROVENANCE_NOTE, ga.TWO_GRAPHS_NOTE,
                 ga.ZERO_PADDED_ARTEFACT_NOTE):
        lowered = text.lower()
        for forbidden in ("discovers causal", "proves hydraulic", "controls gates",
                          "guarantees flood", "causal discovery"):
            assert forbidden not in lowered, f"unsupported claim in {text[:40]!r}"


def test_similarity_matrix_is_cosine_between_representations(forecaster, gnn_history):
    embeddings, gate = forecaster.node_representations(gnn_history)
    matrix = ga.cosine_similarity_matrix(embeddings)
    assert matrix.shape == (NUM_NODES, NUM_NODES)
    assert np.isfinite(matrix).all()
    assert np.allclose(np.diag(matrix), 1.0, atol=1e-6)

    # Cross-check one entry against a direct cosine computation (float64, to
    # match the matrix's own precision).
    a = np.asarray(embeddings[0], dtype=np.float64)
    b = np.asarray(embeddings[1], dtype=np.float64)
    expected = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert matrix[0, 1] == pytest.approx(expected, abs=1e-6)


def test_zero_padded_nodes_are_disclosed_and_excluded_from_ranking(forecaster, gnn_history):
    """
    The model zero-pads absent nodes, so two padded nodes share identical inputs
    and produce a near-1.0 similarity that means nothing. Those pairs must be
    disclosed and kept out of the relationship ranking.
    """
    embeddings, gate = forecaster.node_representations(gnn_history)
    block = ga.build_gnn_advisory(
        embeddings=embeddings, gate_value=gate,
        live_input_nodes=sorted(gnn_history),
        graph_provenance=forecaster.graph_provenance(),
    )
    live = set(block["nodes_with_live_input"])
    padded = set(block["nodes_zero_padded"])
    assert live == set(gnn_history)
    assert len(live) + len(padded) == NUM_NODES

    for rel in block["embedding_similarity"]["top_relationships"]:
        assert rel["source"] in live and rel["target"] in live, (
            f"a zero-padded pair ({rel['source']}, {rel['target']}) was ranked"
        )
        assert rel["both_nodes_have_live_input"] is True
    assert "zero-pad" in block["embedding_similarity"]["matrix_note"].lower()


def test_unavailable_advisory_invents_nothing():
    """Requirement 2/3 in the negative case — no fabricated values."""
    for block in (ga.empty_block("NO_INFERENCE_YET"),
                  ga.unavailable_block(ga.STATUS_UNAVAILABLE, "MODEL_NOT_LOADED")):
        assert block["status"] != ga.STATUS_AVAILABLE
        assert block["reason"]
        assert block["node_embeddings"] == {}
        assert block["embedding_similarity"] is None
        assert block["embedding_dimensions"] is None
        assert block["node_embeddings"] == {}
        assert block["advisory_only"] is True
        assert block["affects_control"] is False


def test_advisory_survives_bad_input_without_inventing_values(forecaster, gnn_history):
    embeddings, gate = forecaster.node_representations(gnn_history)
    bad = np.full_like(embeddings, np.nan)
    block = ga.build_gnn_advisory(
        embeddings=bad, gate_value=gate,
        live_input_nodes=sorted(gnn_history),
        graph_provenance=forecaster.graph_provenance(),
    )
    assert block["status"] == ga.STATUS_UNAVAILABLE
    assert block["reason"] == "NON_FINITE_EMBEDDINGS"


# ===========================================================================
# 3. The live payload exposes the advisory (display only)
# ===========================================================================

def test_live_payload_exposes_the_advisory_block(stepped_sim):
    block = _advisory_from_live()
    assert block["status"] == ga.STATUS_AVAILABLE
    assert block["advisory_only"] is True
    assert block["affects_control"] is False
    assert block["graph_nodes"] == 16
    assert block["graph_undirected_edges"] == 41
    assert block["graph_directed_edges"] == 82
    assert block["embedding_dimensions"] == 64
    assert len(block["node_embeddings"]) == NUM_NODES
    # Only the four mapped reservoirs have live input in the live cascade.
    assert sorted(block["nodes_with_live_input"]) == sorted(
        ["Anayirankal", "Ponmudi", "Idamalayar", "Idukki"]
    )
    assert block["gate_value"] == pytest.approx(0.0151, abs=1e-4)
    assert isinstance(block["inference_latency_ms"], float)


def test_advisory_reaches_the_websocket_payload_too(stepped_sim):
    rest = client.get("/api/state").json()
    with client.websocket_connect("/ws/state") as ws:
        pushed = ws.receive_json()
    assert set(rest.keys()) == set(pushed.keys())
    assert "gnn_advisory" in pushed
    assert pushed["gnn_advisory"]["advisory_only"] is True
    assert pushed["gnn_advisory"]["affects_control"] is False
    assert pushed["gnn_advisory"]["graph_nodes"] == 16


def test_twin_adapter_carries_the_block_with_a_safe_default():
    from src.dashboard.twin_component.state_adapter import adapt_state_for_twin

    default = adapt_state_for_twin({"reservoirs": {}}, "MANUAL", 0.0)["gnn_advisory"]
    assert default["advisory_only"] is True
    assert default["affects_control"] is False
    assert default["status"] == ga.STATUS_UNAVAILABLE
    # The adapter's degraded fallback must not drift from the canonical schema.
    assert set(default.keys()) == set(ga.default_block().keys())


# ===========================================================================
# 4. The advisory cannot reach control
# ===========================================================================

def test_control_policy_remains_lstm_only_in_source():
    source = STATE_MANAGER_PATH.read_text(encoding="utf-8")
    assert 'policy="lstm_primary"' in source
    assert "gnn_primary" not in source
    assert "risk_envelope" not in source


def test_control_forecasts_are_lstm_only_on_every_step(stepped_sim):
    forecasts = stepped_sim._run_ml_pipeline()
    for fc in forecasts.values():
        assert fc.get("source") == "LSTM_V3"
        assert fc.get("forecast_source") == "FROZEN_LSTM_V3"


def test_guard_refuses_an_advisory_influenced_control_forecast(stepped_sim):
    """
    The fail-closed guard is real: a tampered forecast source MUST be refused
    rather than silently controlled with.
    """
    lstm = {"Virtual Reservoir A": {"forecast_1d": 3.0, "forecast_3d": 4.0, "forecast_7d": 5.0}}
    tampered = copy.deepcopy(lstm)
    tampered["Virtual Reservoir A"]["forecast_1d"] = 99.0

    with pytest.raises(RuntimeError, match="NOT FROM THE VALIDATED MODEL"):
        stepped_sim._assert_control_forecasts_are_lstm_only(tampered, lstm)

    # The untampered case passes.
    stepped_sim._assert_control_forecasts_are_lstm_only(copy.deepcopy(lstm), lstm)


def test_gnn_cannot_change_or_bypass_the_control_chain():
    """Requirements 6, 7, 8, 9 — structural proof at the control boundary."""
    orchestrator = ORCHESTRATOR_PATH.read_text(encoding="utf-8")
    routes = ROUTES_PATH.read_text(encoding="utf-8")
    state_manager_src = STATE_MANAGER_PATH.read_text(encoding="utf-8")

    # The layers that decide gates never see the advisory.
    for text, name in ((orchestrator, "live_mpc_orchestrator.py"),
                       (routes, "routes.py")):
        assert "gnn_advisory" not in text, f"{name} reads the advisory block"
        assert "gnn_forecaster" not in text, f"{name} reaches the GNN model"

    # The control layers never import the GNN.
    for path in (ORCHESTRATOR_PATH, ROUTES_PATH,
                 _PROJECT_ROOT / "src" / "controller" / "mpc_controller.py",
                 _PROJECT_ROOT / "src" / "controller" / "safety.py",
                 _PROJECT_ROOT / "src" / "controller" / "downstream_capacity_guard.py"):
        modules = " ".join(_imported_modules(path)).lower()
        assert "gnn" not in modules, f"{path.name} imports the GNN"

    # The applied gates come from the orchestrator's decision, and nothing else.
    assert "for name, pct in decision.gate_positions_pct.items()" in state_manager_src
    assert "gate_commands[res] = gnn" not in state_manager_src


def test_reservoir_network_and_mass_balance_are_gnn_independent():
    """Requirements 10, 11."""
    for path in (NETWORK_PATH, MASS_BALANCE_PATH):
        source = path.read_text(encoding="utf-8")
        assert "gnn" not in source.lower(), f"{path.name} references the GNN"
        modules = " ".join(_imported_modules(path)).lower()
        assert "gnn" not in modules, f"{path.name} imports the GNN"


def test_frozen_lstm_v3_remains_the_validated_forecast_source(stepped_sim):
    """
    Requirement 12 — the validated baseline is untouched and still reported as
    the forecast source wherever a live forecast exists. Reservoir D carries no
    live forecast by Stage 9 policy, so its entry is explicitly unavailable
    rather than zero-filled.
    """
    state = client.get("/api/state").json()
    assert state["forecast_provenance"]["model"] == "LSTM_V3_LOGTARGET"
    assert state["forecast_provenance"]["model_status"] == "FROZEN_UNMODIFIED"

    sources = {res.get("forecast_source") for res in state["reservoirs"].values()}
    assert sources <= {None, "FROZEN_LSTM_V3"}, f"an unexpected forecast source: {sources}"
    assert "FROZEN_LSTM_V3" in sources

    # The Stage 9 exclusion of Reservoir D is explicit, not a fabricated value.
    assert state["cascade"]["reservoirs"][3]["name"] == "Idukki"

    # And the advisory states plainly that its own metrics do not apply.
    assert "NOT the validated forecasting baseline" in _advisory_from_live()["validation_metrics_note"]


def test_gnn_forecast_is_labelled_experimental_and_advisory():
    """Requirement 13."""
    source = ADAPTER_PATH.read_text(encoding="utf-8")
    for phrase in ("EXPERIMENTAL", "ADVISORY", "does not discover causal relationships",
                   "does not prove hydraulic connectivity"):
        assert phrase in source, f"the adapter docstring omits {phrase!r}"

    inference_source = (_PROJECT_ROOT / "src" / "modeling" / "gnn_inference.py").read_text(
        encoding="utf-8"
    )
    assert "ADVISORY" in inference_source
    assert "Gated GCN-LSTM V1 (ADVISORY)" in inference_source
    assert "(Production)" not in inference_source


# ===========================================================================
# 5. Two graphs, never conflated
# ===========================================================================

def test_physical_topology_is_not_presented_as_the_statistical_graph(stepped_sim):
    """Requirement 14 — and the declared physical topology is cross-checked."""
    block = _advisory_from_live()
    assert block["graph_nodes"] == 16, "the statistical graph must stay 16 nodes"
    assert block["graph_provenance"]["is_physical_topology"] is False
    assert "must not be conflated" in block["two_graphs_note"]

    declared = block["physical_control_topology"]
    assert declared["is_statistical_graph"] is False
    assert len(declared["nodes"]) == 4

    # The declared node order is the AUTHORITATIVE cascade order (A..D), read
    # from the live payload — not an independent list that could drift.
    live_names = [r["name"] for r in client.get("/api/state").json()["cascade"]["reservoirs"]]
    assert live_names == declared["nodes"] == ["Anayirankal", "Ponmudi", "Idamalayar", "Idukki"]
    assert declared["terminal"] == live_names[-1]

    # Cross-check the declaration against the LIVE network's own connections.
    network = _live_network()
    connections = list(network.connections)
    assert len(connections) == 3, "the live cascade has three links"
    assert len(network.nodes) == 4
    assert len(declared["links"]) == 3
    for link, conn, delay, atten in zip(declared["links"], connections,
                                        PHYSICAL_DELAYS, PHYSICAL_ATTENUATION):
        assert link["routing_delay_steps"] == delay == conn.delay
        assert link["attenuation"] == pytest.approx(atten) == pytest.approx(conn.attenuation)


def test_axis_orders_differ_between_the_two_structures(forecaster):
    """The 16-node order is alphabetical; the physical chain is A→B→C→D."""
    prov = forecaster.graph_provenance()
    assert prov["graph_nodes"] == 16
    assert CANONICAL_NODE_ORDER[0] == "Anathode"      # alphabetical, not the cascade
    assert ga.PHYSICAL_CONTROL_TOPOLOGY["nodes"][0] == "Anayirankal"  # the cascade head
    assert ga.PHYSICAL_CONTROL_TOPOLOGY["nodes"][-1] == "Idukki"      # terminal


# ===========================================================================
# 6. No frontend inference
# ===========================================================================

def test_no_browser_side_gnn_inference_exists():
    """Requirement 17."""
    forbidden = ("torch", "tensorflow", "onnx", "GatedGCNLSTM", "LiveGNNForecaster",
                 "gnn_inference", "state_dict", "checkpoint")
    for name in ("index.html", "api.js"):
        source = (WEB_DIR / name).read_text(encoding="utf-8")
        lowered = source.lower()
        for token in forbidden:
            assert token.lower() not in lowered, f"{name} references model runtime {token!r}"

    index = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    assert "state.gnn_advisory || null" in index, "the page must READ the block"
    assert "embedding_similarity" in index, "the page may only render payload values"


def test_no_streamlit_side_authoritative_gnn_inference():
    """Requirement 18."""
    modules = " ".join(_imported_modules(APP_PATH)).lower()
    for token in ("gnn_inference", "gnn_advisory", "livegnnforecaster", "gatedgcn"):
        assert token not in modules, f"the Streamlit page imports {token}"

    source = APP_PATH.read_text(encoding="utf-8")
    assert "gnn_advisory" in source, "Streamlit must display the advisory"
    assert "node_representations" not in source
    assert "cosine" not in source.lower()


# ===========================================================================
# 7. MANDATORY A/B control-inertness test
# ===========================================================================

def _run_a_b_at_decision_level():
    """
    Identical network + identical forecast bundle, twice. Run A with the real
    advisory present; run B with a MODIFIED advisory and then with the advisory
    DISABLED. Every control output must be identical.
    """
    results = {}

    for label, mutate in (
        ("A_normal_advisory", lambda: None),
        ("B_modified_advisory", "modify"),
        ("B_disabled_advisory", "disable"),
    ):
        network = _live_network()
        bundle = _validated_bundle(network)
        orchestrator = LiveMPCOrchestrator()

        if mutate == "modify":
            state_manager.sim_state.gnn_advisory = ga.build_gnn_advisory(
                embeddings=np.zeros((NUM_NODES, EMBEDDING_DIM), dtype=np.float32) + 12345.0,
                gate_value=0.999,
                live_input_nodes=[],
                graph_provenance={"available": True, "graph_nodes": 999},
            )
        elif mutate == "disable":
            state_manager.sim_state.gnn_advisory = ga.empty_block("A_B_TEST_DISABLED")
            state_manager.sim_state.gnn_ready = False

        decision = orchestrator.decide(network, bundle=bundle)
        results[label] = {
            "decision": _decision_signature(decision),
            "network": _network_signature(network),
        }

    # Restore the live advisory state for any later test.
    state_manager.sim_state.gnn_ready = True
    state_manager.sim_state.gnn_advisory = ga.empty_block("A_B_TEST_FINISHED")
    return results


def test_ab_control_inertness_full_pipeline(stepped_sim):
    """
    Requirements 6–11 and 19/20, measured: the control pipeline is IDENTICAL
    with a normal, a modified and a disabled advisory.
    """
    results = _run_a_b_at_decision_level()
    reference = results["A_normal_advisory"]

    for label in ("B_modified_advisory", "B_disabled_advisory"):
        assert results[label] == reference, (
            f"the control pipeline changed when the GNN advisory was {label}: "
            f"{results[label]} != {reference}"
        )

    # Non-triviality check: the pipeline really did produce a decision.
    decision = reference["decision"]
    assert decision["final_safe_control_action_source"]
    assert decision["controller_status"]
    assert decision["safety_layer_status"]
    assert decision["downstream_status"]


def test_ab_control_inertness_on_the_authoritative_singleton(stepped_sim):
    """
    The same A/B through the REAL authoritative object, holding EVERY input
    identical: the simulation is paused, the network, the history buffers, the
    gates and the forecast inputs are untouched between the two captures, so the
    only variable is the GNN advisory itself.

    Run A: GNN available, advisory AVAILABLE.
    Run B: GNN disabled, advisory UNAVAILABLE.

    Every field of the authoritative payload must be identical — not just the
    control blocks. (``simulation_time`` is a wall-clock stamp generated per
    payload and is excluded by design; the identity is ``state_identity``.)
    """
    sim = state_manager.sim_state
    client.post("/api/simulation/pause")
    client.post("/api/controller/mode", json={"mode": "MANUAL"})

    # --- Run A: real advisory -------------------------------------------------
    sim.gnn_ready = True
    run_a = client.get("/api/state").json()
    assert run_a["gnn_advisory"]["status"] == ga.STATUS_AVAILABLE, (
        "run A must carry a real advisory, otherwise this A/B proves nothing"
    )

    # --- Run B: GNN disabled, nothing else touched ----------------------------
    sim.gnn_ready = False
    run_b = client.get("/api/state").json()
    sim.gnn_ready = True

    assert run_b["gnn_advisory"]["status"] == ga.STATUS_UNAVAILABLE
    assert run_b["gnn_advisory"]["reason"] == "MODEL_NOT_LOADED"

    # --- The ONLY differences may be the advisory and the wall clock ----------
    for key in ("gnn_advisory", "simulation_time"):
        run_a.pop(key, None)
        run_b.pop(key, None)

    assert run_a == run_b, (
        "the authoritative payload changed when the GNN advisory was removed: "
        f"{set(k for k in run_a if run_a[k] != run_b.get(k))}"
    )

    # Explicitly name the outputs the requirement asks about.
    assert run_a["reservoirs"] == run_b["reservoirs"]
    assert run_a["control"] == run_b["control"]
    assert run_a["downstream"] == run_b["downstream"]
    assert run_a["mass_balance"] == run_b["mass_balance"]
    assert run_a["cascade"] == run_b["cascade"]
    assert run_a["state_identity"] == run_b["state_identity"]


def test_advisory_carries_no_gate_or_control_field():
    """The advisory schema itself cannot be mistaken for a command."""
    keys = set(ga.default_block().keys())
    for forbidden in ("gate", "gates", "gate_positions_pct", "manual_gates",
                      "controller_action", "target_gates"):
        assert forbidden not in keys, f"the advisory block exposes {forbidden!r}"


def test_advisory_is_not_read_by_the_controller_chain():
    """Requirement 6/7/8/9 — no control module reads the advisory attribute."""
    for path in (ORCHESTRATOR_PATH,
                 _PROJECT_ROOT / "src" / "controller" / "mpc_controller.py",
                 _PROJECT_ROOT / "src" / "controller" / "safety.py",
                 _PROJECT_ROOT / "src" / "controller" / "downstream_capacity_guard.py",
                 _PROJECT_ROOT / "src" / "controller" / "objective.py"):
        source = path.read_text(encoding="utf-8")
        assert "gnn_advisory" not in source, f"{path.name} reads the advisory"
        assert "gnn_forecaster" not in source, f"{path.name} reaches the GNN"
