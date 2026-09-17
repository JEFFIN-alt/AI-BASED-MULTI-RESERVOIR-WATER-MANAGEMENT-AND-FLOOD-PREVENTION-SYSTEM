"""
Stage 15 — END-TO-END ARCHITECTURE / INTEGRATION VERIFICATION: tests.

Proves the complete authoritative architecture works as ONE system, on the real
objects, with the boundary order MEASURED rather than assumed:

    input/state -> frozen LSTM V3 -> LiveForecastAdapter -> NetworkForecastSnapshot
                -> GNN advisory (advisory only)
                -> MPCController -> SafetyLayer -> DownstreamCapacityGuard
                -> FINAL_SAFE_CONTROL_ACTION -> ReservoirNetwork.step()
                -> MassBalanceMonitor -> authoritative state
                -> REST/WebSocket -> Digital Twin / Streamlit

Coverage (Stage 15 §18): A full control cycle · B forecast->MPC contract ·
C GNN advisory->control invariance · D safety ordering · E final action->physics ·
F physics->mass balance · G state->REST/WebSocket · H API->twin ·
I Streamlit->API->backend · J all four reservoirs · K failure paths ·
L single simulation authority.
"""

import ast
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.controller.downstream_capacity_guard import DownstreamCapacityGuard  # noqa: E402
from src.controller.live_mpc_orchestrator import LiveMPCOrchestrator  # noqa: E402
from src.controller.mpc_controller import MPCController  # noqa: E402
from src.controller.safety import SafetyLayer  # noqa: E402
from src.dashboard.api import state_manager  # noqa: E402
from src.dashboard.api.app import app  # noqa: E402
from src.dashboard.sim_bridge import SimBridge  # noqa: E402
from src.modeling import gnn_advisory as ga  # noqa: E402
from src.network_env.live_cascade_adapter import LiveCascadeAdapter  # noqa: E402
from src.network_env.live_forecast_adapter import LiveForecastAdapter  # noqa: E402
from src.network_env.mass_balance import MassBalanceMonitor  # noqa: E402
from src.network_env.reservoir_network import ReservoirNetwork  # noqa: E402

client = TestClient(app)

PROJECT_ROOT = str(_PROJECT_ROOT)
LIVE_CONFIG = _PROJECT_ROOT / "configs" / "simulation" / "four_reservoir_demo.json"
THRESH_PATH = _PROJECT_ROOT / "data" / "processed" / "historical_inflow_thresholds.json"
WEB_DIR = _PROJECT_ROOT / "src" / "dashboard" / "web"
TWIN_PATH = WEB_DIR / "index.html"
API_JS_PATH = WEB_DIR / "api.js"
STREAMLIT_PATH = _PROJECT_ROOT / "src" / "dashboard" / "app.py"
STATE_MANAGER_PATH = _PROJECT_ROOT / "src" / "dashboard" / "api" / "state_manager.py"
ORCHESTRATOR_PATH = _PROJECT_ROOT / "src" / "controller" / "live_mpc_orchestrator.py"

NODES = ["Virtual Reservoir A", "Virtual Reservoir B",
         "Virtual Reservoir C", "Virtual Reservoir D"]
NODE_TO_NAME = {"Virtual Reservoir A": "Anayirankal", "Virtual Reservoir B": "Ponmudi",
                "Virtual Reservoir C": "Idamalayar", "Virtual Reservoir D": "Idukki"}
HORIZONS = ("forecast_1d", "forecast_3d", "forecast_7d")
PHYSICAL_DELAYS = [2, 1, 1]
PHYSICAL_ATTENUATION = [0.90, 0.85, 0.80]


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _restore_authoritative_state():
    """
    This module drives the ONE authoritative singleton. Restore state only
    (paused, 50 % storages, MANUAL) afterwards so no other suite inherits a
    saturated or running simulation.
    """
    yield
    client.post("/api/simulation/pause")
    client.post("/api/simulation/reset")
    client.post("/api/controller/mode", json={"mode": "MANUAL"})


@pytest.fixture
def stepped_sim():
    client.post("/api/simulation/pause")
    client.post("/api/simulation/reset")
    client.post("/api/controller/mode", json={"mode": "MANUAL"})
    for _ in range(8):
        assert client.post("/api/simulation/step").status_code == 200
    return state_manager.sim_state


def _payload(*, status="VALIDATED",
             provenance="REAL_MEASUREMENT_INPUTS_FROZEN_MODEL",
             validated=True, simulated=False, v1=2.0, v3=2.1, v7=2.2):
    """A forecast payload that exercises the pipeline (Stage 7/8/9/10 convention)."""
    return {
        "forecast_1d": v1, "forecast_3d": v3, "forecast_7d": v7,
        "forecast_status": status, "forecast_provenance": provenance,
        "forecast_source": "FROZEN_LSTM_V3", "is_simulated": simulated,
        "validated_metrics_apply": validated, "forecast_unit": "MCM/day",
        "horizons": list(HORIZONS),
        "input_provenance": {"synthetic_demo": [], "unavailable": [], "simulated": []},
    }


def _demo_payload():
    return _payload(status="DEMONSTRATION_ONLY",
                    provenance="SIMULATION_OR_SYNTHETIC_INPUTS_FROZEN_MODEL",
                    validated=False, simulated=True)


def _network():
    return SimBridge(str(LIVE_CONFIG), str(THRESH_PATH)).cascade.network


def _bundle(network, payloads=None, nodes=None):
    adapter = LiveForecastAdapter.for_network(network, project_root=PROJECT_ROOT)
    nodes = nodes if nodes is not None else NODES
    payloads = payloads if payloads is not None else {n: _payload() for n in nodes}
    return adapter.build_bundle(payloads, "2026-09-14")


def _decision_projection(decision) -> dict:
    """Every control output Stage 15 §18C requires to be invariant."""
    return {
        "mpc_proposal": dict(decision.proposed_gate_positions_fraction),
        "mpc_status": decision.mpc_status,
        "safety_layer_status": decision.safety_layer_status,
        "safety_layer_output": dict(decision.safety_layer_gate_positions_pct),
        "downstream_status": decision.downstream_status,
        "downstream_capacity_achieved": decision.downstream_capacity_achieved,
        "downstream_modified": decision.downstream_protection_modified,
        "final_action_pct": dict(decision.final_safe_control_action_pct),
        "final_action_source": decision.final_safe_control_action_source,
        "applied_gates_pct": dict(decision.gate_positions_pct),
    }


def _imports_of(path: Path) -> set:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
            names |= {f"{node.module}.{a.name}" for a in node.names}
    return names


# ===========================================================================
# L. SINGLE SIMULATION AUTHORITY
# ===========================================================================

def test_exactly_one_live_simulation_owner():
    assert state_manager.authoritative_instance_count() == 1

    owners = []
    for path in (_PROJECT_ROOT / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "SimBridge(" in text or "GlobalSimulationState()" in text:
            owners.append(str(path.relative_to(_PROJECT_ROOT)).replace("\\", "/"))
    assert set(owners) == {"src/dashboard/api/state_manager.py"}, owners


def test_the_live_stack_is_the_validated_one():
    cascade = state_manager.sim_state.bridge.cascade
    assert isinstance(cascade, LiveCascadeAdapter)
    assert isinstance(cascade.network, ReservoirNetwork)


def test_no_physics_engine_is_imported_by_the_live_path():
    """Offline engines may exist; none may be reachable from the live path."""
    live_modules = [STATE_MANAGER_PATH,
                    _PROJECT_ROOT / "src" / "dashboard" / "api" / "routes.py",
                    _PROJECT_ROOT / "src" / "dashboard" / "api" / "app.py",
                    STREAMLIT_PATH,
                    _PROJECT_ROOT / "src" / "dashboard" / "sim_bridge.py",
                    _PROJECT_ROOT / "src" / "network_env" / "live_cascade_adapter.py",
                    ORCHESTRATOR_PATH]
    for path in live_modules:
        imports = " ".join(_imports_of(path)).lower()
        for engine in ("simulator.engine", "simulator.environment",
                       "simulationengine", "virtualcascade"):
            assert engine not in imports, f"{path.name} imports {engine}"


def test_dormant_legacy_rule_based_helper_has_no_callers():
    """
    `SimBridge.compute_ai_recommendation` (ForecastAwareController) is a
    documented NON-AUTHORITATIVE offline helper. It must have zero callers.
    """
    callers = []
    for path in list((_PROJECT_ROOT / "src").rglob("*.py")) + \
            list((_PROJECT_ROOT / "tests").rglob("*.py")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "compute_ai_recommendation(" in text and "def compute_ai_recommendation" not in text:
            callers.append(str(path.relative_to(_PROJECT_ROOT)).replace("\\", "/"))
    assert callers == [], f"the legacy advisor is invoked: {callers}"

    # And the live control path never mentions it.
    assert "compute_ai_recommendation" not in STATE_MANAGER_PATH.read_text(encoding="utf-8")


# ===========================================================================
# A. FULL CONTROL CYCLE (measured boundary order)
# ===========================================================================

def test_full_control_cycle_boundary_order_is_measured(monkeypatch):
    """
    Instrument the REAL boundary methods and record the actual call order.
    The authoritative order is MPC -> SafetyLayer -> DownstreamCapacityGuard.
    """
    order = []
    orig_mpc, orig_safety, orig_guard = (MPCController.decide, SafetyLayer.validate,
                                         DownstreamCapacityGuard.evaluate)

    monkeypatch.setattr(MPCController, "decide",
                        lambda self, *a, **kw: (order.append("mpc.decide"),
                                                orig_mpc(self, *a, **kw))[1])
    monkeypatch.setattr(SafetyLayer, "validate",
                        lambda self, *a, **kw: (order.append("safety.validate"),
                                                orig_safety(self, *a, **kw))[1])
    monkeypatch.setattr(DownstreamCapacityGuard, "evaluate",
                        lambda self, *a, **kw: (order.append("downstream_guard.evaluate"),
                                                orig_guard(self, *a, **kw))[1])

    network = _network()
    decision = LiveMPCOrchestrator().decide(network, bundle=_bundle(network),
                                           current_inflows={n: 3.0 for n in NODES})

    assert order[0] == "mpc.decide", order
    assert "safety.validate" in order and order.index("safety.validate") > 0, order
    assert order[-1] == "downstream_guard.evaluate", order
    assert order.index("downstream_guard.evaluate") > order.index("safety.validate"), order
    # The SafetyLayer is invoked once inside the MPC and once by the orchestrator;
    # both precede the capacity guard, so the ordering invariant holds.
    assert order.count("safety.validate") <= 2, order

    assert decision.final_safe_control_action_source
    assert decision.controller_status == "ACTIVE"


def test_full_control_cycle_produces_a_final_action_for_every_reservoir():
    network = _network()
    decision = LiveMPCOrchestrator().decide(network, bundle=_bundle(network),
                                           current_inflows={n: 3.0 for n in NODES})
    assert set(decision.final_safe_control_action_pct) == set(NODES)
    assert all(math.isfinite(float(v)) for v in decision.final_safe_control_action_pct.values())


# ===========================================================================
# B. FORECAST -> MPC CONTRACT
# ===========================================================================

def test_validated_forecast_is_eligible_and_drives_the_mpc():
    network = _network()
    decision = LiveMPCOrchestrator().decide(network, bundle=_bundle(network))
    assert decision.forecast_control_eligible is True
    assert decision.mpc_status == "OPTIMAL"
    assert decision.control_applied is True
    assert decision.controller_status == "ACTIVE"


def test_snapshot_contract_has_four_reservoirs_and_three_horizons():
    network = _network()
    snapshot = _bundle(network).snapshot
    assert snapshot.available_count("1d") == 4
    for node_id in NODES:
        reservoir = snapshot.get(node_id)
        assert reservoir is not None, node_id
        values = [getattr(reservoir, h) for h in ("target_1d", "target_3d", "target_7d")]
        assert all(v is not None for v in values), node_id
        assert all(math.isfinite(float(v)) and float(v) >= 0.0 for v in values), node_id


@pytest.mark.parametrize("label,payloads,nodes", [
    ("demonstration_only", {n: _demo_payload() for n in NODES}, NODES),
    ("missing_reservoir_d", {n: _payload() for n in NODES[:-1]}, NODES[:-1]),
    ("nan_forecast", {n: _payload(v1=float("nan")) for n in NODES}, NODES),
    ("infinite_forecast", {n: _payload(v1=float("inf")) for n in NODES}, NODES),
    ("negative_forecast", {n: _payload(v1=-5.0) for n in NODES}, NODES),
])
def test_invalid_forecast_provenance_cannot_enter_the_validated_mpc(
    label, payloads, nodes
):
    """Requirement §5 — DEMONSTRATION_ONLY / missing / non-finite => NOT INVOKED."""
    network = _network()
    decision = LiveMPCOrchestrator().decide(
        network, bundle=_bundle(network, payloads=payloads, nodes=nodes)
    )
    assert decision.forecast_control_eligible is False, label
    assert decision.mpc_status == "NOT_INVOKED", label
    assert decision.control_applied is False, label
    assert decision.final_safe_control_action_source == "HELD_CURRENT_GATES", label
    # The gates handed on are the CURRENT authoritative gates, not a fabrication.
    current = {n: round(network.nodes[n].state.gate_position * 100.0, 9) for n in NODES}
    held = {n: round(float(v), 9) for n, v in decision.final_safe_control_action_pct.items()}
    assert held == current, label


def test_live_ai_mode_is_blocked_because_reservoir_d_has_no_live_forecast(stepped_sim):
    """
    The live cascade's own forecast contract, measured end to end:

      * the live forecasts are DEMONSTRATION_ONLY (simulation-derived inputs), so
        `validated_metrics_apply` is False for A/B/C;
      * Reservoir D has NO live forecast at all (Stage 9 policy), so the
        coordinated MPC cannot be eligible;

    therefore the provenance gate blocks and the system HOLDS the current gates
    rather than fabricating a forecast or acting on an unvalidated one. The
    payload reports the same verdict as the orchestrator's own decision object.
    """
    client.post("/api/controller/mode", json={"mode": "AI"})
    state_manager.sim_state.step()          # a decision now exists

    decision = state_manager.sim_state.last_control_decision
    assert decision.controller_status == "BLOCKED"
    assert decision.mpc_status == "NOT_INVOKED"
    assert decision.final_safe_control_action_source == "HELD_CURRENT_GATES"
    assert decision.blocked_reason == "FORECAST_NOT_ELIGIBLE_FOR_CONTROL"

    # Both independent reasons are stated per reservoir.
    nodes = decision.forecast_provenance["nodes"]
    assert nodes["Virtual Reservoir D"]["issue"] == "MISSING_FORECAST"
    assert nodes["Virtual Reservoir A"]["declared_status"] == "DEMONSTRATION_ONLY"
    assert nodes["Virtual Reservoir A"]["validated_metrics_apply"] is False

    state = client.get("/api/state").json()
    control = state["control"]
    assert control["forecast_control_eligible"] is False
    assert control["controller_status"] == "BLOCKED"
    assert control["control_applied"] is False
    assert control["final_safe_control_action_source"] == "HELD_CURRENT_GATES"
    assert control["blocked_reason"] == "FORECAST_NOT_ELIGIBLE_FOR_CONTROL"
    assert control["safety_layer_status"] == "NOT_APPLIED_MPC_BLOCKED"
    assert control["downstream_status"] == "NOT_APPLIED_MPC_BLOCKED"
    assert state["cascade"]["reservoirs"][3]["name"] == "Idukki"
    assert state["cascade"]["reservoirs"][3]["terminal"] is True

    # The blocked action is the CURRENT authoritative gates, not a new value.
    current = {n: round(state_manager.sim_state.bridge.cascade.network
                        .nodes[n].state.gate_position * 100.0, 9) for n in NODES}
    held = {n: round(float(v), 9)
            for n, v in decision.final_safe_control_action_pct.items()}
    assert held == current


# ===========================================================================
# C. GNN ADVISORY -> CONTROL INVARIANCE
# ===========================================================================

def test_modified_gnn_advisory_leaves_every_control_output_unchanged(stepped_sim):
    network = _network()
    bundle = _bundle(network)
    sim = state_manager.sim_state
    saved = sim.gnn_advisory

    baseline = _decision_projection(
        LiveMPCOrchestrator().decide(network, bundle=bundle,
                                     current_inflows={n: 3.0 for n in NODES})
    )
    sim.gnn_advisory = ga.build_gnn_advisory(
        embeddings=np.full((16, 64), 12345.0, dtype=np.float32),
        gate_value=0.999, live_input_nodes=[],
        graph_provenance={"available": True, "graph_nodes": 999},
    )
    modified = _decision_projection(
        LiveMPCOrchestrator().decide(network, bundle=bundle,
                                     current_inflows={n: 3.0 for n in NODES})
    )
    sim.gnn_advisory = saved

    assert modified == baseline


def test_unavailable_gnn_leaves_the_payload_and_decision_unchanged(stepped_sim):
    sim = state_manager.sim_state
    client.post("/api/controller/mode", json={"mode": "MANUAL"})

    sim.gnn_ready = True
    with_advisory = client.get("/api/state").json()
    assert with_advisory["gnn_advisory"]["status"] == ga.STATUS_AVAILABLE

    sim.gnn_ready = False
    without = client.get("/api/state").json()
    sim.gnn_ready = True

    assert without["gnn_advisory"]["status"] == ga.STATUS_UNAVAILABLE
    for key in ("gnn_advisory", "simulation_time"):
        with_advisory.pop(key, None)
        without.pop(key, None)
    assert with_advisory == without


def test_gnn_advisory_flags_are_advisory_only(stepped_sim):
    block = client.get("/api/state").json()["gnn_advisory"]
    assert block["advisory_only"] is True
    assert block["affects_control"] is False


# ===========================================================================
# D. SAFETY ORDERING + CORRECTION
# ===========================================================================

def test_unsafe_proposal_is_corrected_before_physics():
    network = _network()
    current = {n: float(network.nodes[n].state.gate_position) for n in NODES}
    unsafe = {"Virtual Reservoir A": 5.0, "Virtual Reservoir B": -3.0,
              "Virtual Reservoir C": float("nan"), "Virtual Reservoir D": 0.95}

    safety_result = SafetyLayer().validate(unsafe, current, NODES)
    guard_result = DownstreamCapacityGuard().evaluate(
        network,
        action_fraction={n: float(safety_result.validated_gates[n]) for n in NODES},
        current_fraction=current, node_ids=NODES, max_gate_change=0.5,
        inflows={n: 3.0 for n in NODES},
    )

    for nid in NODES:
        applied = float(guard_result.action_fraction[nid])
        assert math.isfinite(applied), f"{nid}: a NaN reached the guard output"
        assert 0.0 <= applied <= 1.0, f"{nid}: bounds violated"
        assert abs(applied - current[nid]) <= 0.5 + 1e-9, f"{nid}: rate limit violated"
    assert str(guard_result.status) in ("PROTECTED", "CORRECTED", "FAILED_CLOSED")


def test_the_gate_command_applied_is_the_final_safe_control_action():
    """Requirement §8 — what physics receives is the FINAL action."""
    network = _network()
    decision = LiveMPCOrchestrator().decide(network, bundle=_bundle(network),
                                           current_inflows={n: 3.0 for n in NODES})
    bridge = SimBridge(str(LIVE_CONFIG), str(THRESH_PATH))
    bridge.step({n: 3.0 for n in NODES}, dict(decision.final_safe_control_action_pct),
                action_source=decision.final_safe_control_action_source)

    for nid in NODES:
        received = bridge.cascade.network.nodes[nid].state.gate_position * 100.0
        assert received == pytest.approx(decision.final_safe_control_action_pct[nid], abs=1e-9)


# ===========================================================================
# E/F. PHYSICS -> MASS BALANCE
# ===========================================================================

def test_physics_topology_delays_and_attenuation():
    network = _network()
    assert list(network.processing_order) == NODES
    connections = list(network.connections)
    assert [c.delay for c in connections] == PHYSICAL_DELAYS
    for conn, expected in zip(connections, PHYSICAL_ATTENUATION):
        assert conn.attenuation == pytest.approx(expected)
    assert float(network.downstream_capacity) == pytest.approx(50.0)


def test_physics_advances_exactly_one_timestep_per_step():
    network = _network()
    before = int(network.timestep)
    network.step({n: 0.0 for n in NODES}, {n: 0.0 for n in NODES})
    assert int(network.timestep) - before == 1


def test_mass_balance_passes_after_an_authoritative_step(stepped_sim):
    state = client.get("/api/state").json()
    mb = state["mass_balance"]
    assert mb["status"] == "PASS"
    assert mb["checked"] is True
    assert mb["tolerance"] == 1e-9
    assert abs(float(mb["residual"])) <= 1e-9
    assert mb["reservoirs_checked"] == mb["reservoirs_expected"] == 4


@pytest.mark.parametrize("corruption", ["storage", "outflow", "routing_queue", "non_finite"])
def test_mass_balance_detects_corruption_and_never_repairs_it(corruption):
    """
    Correct protocol: snapshot the PRE-step state, run the real physics step,
    corrupt, then verify. Corrupting after the step is what makes the detected
    residual attributable to the corruption rather than to protocol noise.
    """
    network = _network()
    monitor = MassBalanceMonitor()
    snapshot = monitor.snapshot(network)
    network.step({n: 1.0 for n in NODES}, {n: 0.3 for n in NODES})

    if corruption == "storage":
        network.nodes["Virtual Reservoir C"].state.storage += 7.0
    elif corruption == "outflow":
        network.nodes["Virtual Reservoir D"].state.total_outflow += 3.0
    elif corruption == "routing_queue":
        network.connections[0].queue[-1] += 2.0
    else:
        network.nodes["Virtual Reservoir D"].state.storage = float("nan")

    result = monitor.verify(network, snapshot, applied_inflows={n: 1.0 for n in NODES},
                            applied_gates_fraction={n: 0.3 for n in NODES})
    assert result.status != "PASS", f"{corruption} was not detected"
    assert result.violations, f"{corruption} produced no violation record"
    # The monitor must not have "fixed" anything.
    if corruption == "non_finite":
        assert math.isnan(network.nodes["Virtual Reservoir D"].state.storage)
    if corruption == "storage":
        assert network.nodes["Virtual Reservoir C"].state.storage > 100.0


def test_the_same_protocol_without_corruption_reports_pass():
    """Control case — the protocol itself is not what produces a violation."""
    network = _network()
    monitor = MassBalanceMonitor()
    snapshot = monitor.snapshot(network)
    network.step({n: 1.0 for n in NODES}, {n: 0.3 for n in NODES})
    result = monitor.verify(network, snapshot, applied_inflows={n: 1.0 for n in NODES},
                            applied_gates_fraction={n: 0.3 for n in NODES})
    assert result.status == "PASS"
    assert abs(float(result.residual)) <= 1e-9


# ===========================================================================
# G. STATE -> REST / WEBSOCKET
# ===========================================================================

def test_rest_and_websocket_agree_for_the_same_timestep(stepped_sim):
    client.post("/api/simulation/pause")
    with client.websocket_connect("/ws/state") as ws:
        ws_state = ws.receive_json()
        rest_state = client.get("/api/state").json()

    for key in ("reservoirs", "cascade", "control", "mass_balance", "downstream",
                "forecast_summary", "storm", "simulation", "state_identity",
                "hardware_status", "forecast_provenance"):
        assert ws_state.get(key) == rest_state.get(key), key
    assert ws_state["state_identity"] == rest_state["state_identity"]


def test_pushed_payload_equals_rest_after_a_step(stepped_sim):
    client.post("/api/simulation/pause")
    before = client.get("/api/state").json()["state_identity"]["network_timestep"]
    with client.websocket_connect("/ws/state") as ws:
        ws.receive_json()
        client.post("/api/simulation/step")
        pushed = ws.receive_json()
    rest_after = client.get("/api/state").json()

    assert pushed["reservoirs"] == rest_after["reservoirs"]
    assert pushed["state_identity"] == rest_after["state_identity"]
    assert rest_after["state_identity"]["network_timestep"] - before == 1
    assert pushed["state_identity"] == rest_after["state_identity"]


def test_no_frontend_state_write_endpoint_exists():
    assert client.post("/api/state", json={"storage": 999.0}).status_code == 405
    assert client.put("/api/state", json={"storage": 999.0}).status_code == 405


# ===========================================================================
# H. API -> DIGITAL TWIN
# ===========================================================================

def test_twin_payload_carries_every_block_the_twin_renders(stepped_sim):
    state = client.get("/api/state").json()
    for key in ("reservoirs", "cascade", "control", "mass_balance", "downstream",
                "forecast_summary", "forecast_provenance", "storm", "simulation",
                "state_identity", "hardware_status", "gnn_advisory"):
        assert key in state, key
    assert state["cascade"]["count"] == 4


def test_twin_contains_no_model_or_physics_runtime():
    twin = TWIN_PATH.read_text(encoding="utf-8").lower()
    for token in ("torch", "tensorflow", "onnx", "gatedgcnlstm", "livegnnforecaster",
                  "gnn_inference", "simbridge", "mpc_controller", "reservoir_network",
                  "simulationengine", "virtualcascade", "state_dict"):
        assert token not in twin, f"the twin references {token}"
    api_js = API_JS_PATH.read_text(encoding="utf-8").lower()
    for token in ("torch", "mpc", "safety.validate", "reservoir_network"):
        assert token not in api_js, f"api.js references {token}"


def test_twin_reads_the_advisory_from_the_payload_only():
    twin = TWIN_PATH.read_text(encoding="utf-8")
    assert "state.gnn_advisory" in twin
    assert "embedding_similarity" in twin


# ===========================================================================
# I. STREAMLIT -> API -> BACKEND
# ===========================================================================

def test_streamlit_imports_no_model_engine_or_controller():
    imports = " ".join(_imports_of(STREAMLIT_PATH)).lower()
    for token in ("gnn", "sim_bridge", "simulator", "state_manager", "network_env",
                  "controller", "src.dashboard.api"):
        assert token not in imports, f"Streamlit imports {token}"


def test_streamlit_reads_state_and_only_posts_bound_commands():
    source = STREAMLIT_PATH.read_text(encoding="utf-8")
    assert "/api/state" in source
    assert "urllib.request" in source
    assert "node_representations" not in source
    # No simulation-advancing code. The Stage 13 documentation comments name
    # `ReservoirNetwork.step()` while stating that this page never calls it, so
    # the proof here is the AST: no advancing call exists, and the import-level
    # proof is test_streamlit_imports_no_model_engine_or_controller.
    tree = ast.parse(source)
    advancing = {"step", "init_cascade", "simulation_loop"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else None
            assert name not in advancing, f"Streamlit calls {name}(...)"


def test_streamlit_commands_reach_the_authoritative_backend(stepped_sim):
    """The routes the page posts to are real and move the ONE authoritative sim."""
    response = client.post("/api/gate/reservoir_4", json={"value": 20.0})
    assert response.status_code == 200
    assert state_manager.sim_state.manual_gates["Virtual Reservoir D"] == pytest.approx(20.0)
    assert client.get("/api/state").json()["reservoirs"]["reservoir_4"]["gate"] == \
        pytest.approx(0.20, abs=1e-9)


# ===========================================================================
# J. ALL FOUR RESERVOIRS
# ===========================================================================

def test_all_four_reservoirs_are_first_class(stepped_sim):
    state = client.get("/api/state").json()
    cascade = state["cascade"]
    assert cascade["count"] == 4
    assert [r["name"] for r in cascade["reservoirs"]] == list(NODE_TO_NAME.values())
    assert cascade["terminal"]["name"] == "Idukki"
    for key in ("reservoir_1", "reservoir_2", "reservoir_3", "reservoir_4"):
        assert key in state["reservoirs"]
        assert state["reservoirs"][key].get("node_id")


def test_reservoir_d_is_not_pinned_and_can_change_gate():
    network = _network()
    network.step({n: 3.0 for n in NODES}, {n: 0.5 for n in NODES})
    assert network.nodes["Virtual Reservoir D"].state.gate_position == pytest.approx(0.5)
    network.step({n: 3.0 for n in NODES}, {**{n: 0.5 for n in NODES},
                                           "Virtual Reservoir D": 0.05})
    assert network.nodes["Virtual Reservoir D"].state.gate_position == pytest.approx(0.05)


def test_mpc_action_space_is_1296_candidate_vectors():
    space = LiveMPCOrchestrator().action_space_for(NODES)
    assert space["dimension"] == 4
    assert space["candidate_vectors"] == 1296  # 6 ^ 4 gate levels


# ===========================================================================
# K. FAILURE PATHS
# ===========================================================================

@pytest.mark.parametrize("method,url,body,expected", [
    ("post", "/api/gate/reservoir_1", {"value": "wide open"}, 422),
    ("post", "/api/gate/reservoir_1", '{"value": NaN}', 422),
    ("post", "/api/gate/reservoir_9", {"value": 10.0}, 400),
    ("post", "/api/simulation/speed", {"speed": "fast"}, 422),
    ("post", "/api/storm", {"value": None}, 422),
    ("post", "/api/storm", '{"value": Infinity}', 422),
    ("post", "/api/controller/mode", {"mode": "HACK"}, 422),
    ("post", "/api/state", {"storage": 999.0}, 405),
    ("put", "/api/state", {"storage": 999.0}, 405),
    ("get", "/api/nonexistent", None, 404),
])
def test_malformed_commands_are_rejected(method, url, body, expected):
    if isinstance(body, str):
        response = client.post(url, content=body,
                               headers={"Content-Type": "application/json"})
    else:
        response = getattr(client, method)(url, json=body) if body is not None \
            else client.get(url)
    assert response.status_code == expected


def test_zero_speed_is_clamped_not_accepted_as_zero():
    client.post("/api/simulation/speed", json={"speed": 0.0})
    assert state_manager.sim_state.sim_speed >= 0.05
    assert 1.0 / state_manager.sim_state.sim_speed > 0.0


def test_gate_and_storm_out_of_range_are_clamped_not_injected():
    client.post("/api/gate/reservoir_1", json={"value": 1e9})
    assert state_manager.sim_state.manual_gates["Virtual Reservoir A"] == 100.0
    client.post("/api/gate/reservoir_1", json={"value": -1e9})
    assert state_manager.sim_state.manual_gates["Virtual Reservoir A"] == 0.0


def test_mass_balance_corruption_in_the_live_path_is_reported_not_repaired(stepped_sim):
    """A violation after a real step is surfaced in the payload, never fixed."""
    network = state_manager.sim_state.bridge.cascade.network
    monitor = state_manager.sim_state.bridge.cascade.mass_balance_monitor
    snapshot = monitor.snapshot(network)
    network.nodes["Virtual Reservoir D"].state.total_outflow += 5.0
    result = monitor.verify(network, snapshot,
                            applied_inflows=dict(state_manager.sim_state.manual_inflows),
                            applied_gates_fraction={n: 0.3 for n in NODES})
    assert result.status != "PASS"
    assert network.nodes["Virtual Reservoir D"].state.total_outflow > 5.0  # untouched


def test_unavailable_gnn_does_not_block_the_simulation(stepped_sim):
    sim = state_manager.sim_state
    sim.gnn_ready = False
    try:
        before = client.get("/api/state").json()["state_identity"]["network_timestep"]
        assert client.post("/api/simulation/step").status_code == 200
        after = client.get("/api/state").json()
    finally:
        sim.gnn_ready = True
    assert after["state_identity"]["network_timestep"] - before == 1
    assert after["gnn_advisory"]["status"] == ga.STATUS_UNAVAILABLE
    assert after["mass_balance"]["status"] in ("PASS", "VIOLATION", "NOT_CHECKED")


# ===========================================================================
# HARDWARE BOUNDARY
# ===========================================================================

def test_hardware_remains_disconnected(stepped_sim):
    state = client.get("/api/state").json()
    assert state["hardware_status"] == {
        "esp32": "NOT_CONNECTED", "water_level_sensor": "NOT_CONNECTED",
        "flow_sensor": "NOT_CONNECTED", "gate_actuator": "NOT_CONNECTED",
    }
    assert state["mass_balance"]["hardware_connected"] is False


# ===========================================================================
# SCIENTIFIC CLAIMS
# ===========================================================================

def test_no_unsupported_scientific_claims_in_the_live_surface():
    needles = ("guarantee flood", "flood-proof", "prevents flooding", "causal discovery",
               "discovers causal", "proves hydraulic", "controls real dams",
               "gnn directly controls")
    for path in list(WEB_DIR.rglob("*.html")) + list(WEB_DIR.rglob("*.js")) + \
            [STREAMLIT_PATH]:
        if "vendor" in str(path):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        for needle in needles:
            assert needle not in text, f"{path.name} claims {needle!r}"


def test_payload_label_keeps_the_advisory_and_demonstration_wording(stepped_sim):
    state = client.get("/api/state").json()
    assert state["gnn_advisory"]["advisory_only"] is True
    assert "ADVISORY ONLY" in state["gnn_advisory"]["disclaimer"].upper()
    assert state["forecast_provenance"]["live_forecasts_are_validated"] is False
    assert "embedding similarity" in state["gnn_advisory"]["embedding_similarity"]["label"]
