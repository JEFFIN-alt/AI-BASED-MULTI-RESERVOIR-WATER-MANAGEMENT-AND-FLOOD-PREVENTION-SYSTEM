"""
Stage 9 — Genuinely coordinated four-reservoir control tests.

Proves that the live MPC + SafetyLayer path is a COORDINATED four-reservoir
controller, not four independent controllers and not the legacy behaviour in
which Reservoir D (Idukki) was pinned at 100%.

Topology under test (authoritative):

    A = Anayirankal -> B = Ponmudi -> C = Idamalayar -> D = Idukki

All four reservoirs must be first-class members of the state, the forecast
contract, the MPC action space, the SafetyLayer boundary, the network
transition, the WebSocket state and the Digital Twin payload.
"""

import ast
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.controller.live_mpc_orchestrator import (  # noqa: E402
    ControllerStatus,
    LiveMPCOrchestrator,
)
from src.controller.mpc_controller import MPCController  # noqa: E402
from src.controller.safety import SafetyLayer  # noqa: E402
from src.dashboard.api import state_manager  # noqa: E402
from src.dashboard.api.app import app  # noqa: E402
from src.dashboard.api.state_manager import sim_state as LIVE_SIM  # noqa: E402
from src.dashboard.api.routes import RESERVOIR_ID_TO_NODE  # noqa: E402
from src.dashboard.sim_bridge import SimBridge  # noqa: E402
from src.dashboard.twin_component.state_adapter import (  # noqa: E402
    TWIN_KEY_ORDER,
    adapt_state_for_twin,
)
from src.network_env.live_forecast_adapter import (  # noqa: E402
    FORECAST_UNIT,
    HORIZON_KEYS,
    LiveForecastAdapter,
)
from src.network_env.reservoir_network import ReservoirNetwork  # noqa: E402
from src.network_env.v3_forecast_adapter import ForecastStatus  # noqa: E402

PROJECT_ROOT = str(_PROJECT_ROOT)
client = TestClient(app)

LIVE_CONFIG = _PROJECT_ROOT / "configs" / "simulation" / "four_reservoir_demo.json"
THRESH_PATH = _PROJECT_ROOT / "data" / "processed" / "historical_inflow_thresholds.json"
TOPOLOGY_PATH = _PROJECT_ROOT / "src" / "network_env" / "topology_config.yaml"
ORCH_PATH = _PROJECT_ROOT / "src" / "controller" / "live_mpc_orchestrator.py"
MPC_PATH = _PROJECT_ROOT / "src" / "controller" / "mpc_controller.py"
SAFETY_PATH = _PROJECT_ROOT / "src" / "controller" / "safety.py"
NETWORK_PATH = _PROJECT_ROOT / "src" / "network_env" / "reservoir_network.py"
STATE_MANAGER_PATH = _PROJECT_ROOT / "src" / "dashboard" / "api" / "state_manager.py"
ROUTES_PATH = _PROJECT_ROOT / "src" / "dashboard" / "api" / "routes.py"
TWIN_INDEX_PATH = _PROJECT_ROOT / "src" / "dashboard" / "web" / "index.html"

NODES = [
    "Virtual Reservoir A",
    "Virtual Reservoir B",
    "Virtual Reservoir C",
    "Virtual Reservoir D",
]

#: The validated reservoir -> identity mapping (topology_config.yaml).
VALIDATED_NAMES = {
    "Virtual Reservoir A": "Anayirankal",
    "Virtual Reservoir B": "Ponmudi",
    "Virtual Reservoir C": "Idamalayar",
    "Virtual Reservoir D": "Idukki",
}

FROZEN_ARTIFACTS = [
    _PROJECT_ROOT / "models" / "lstm_pytorch_v3_logtarget" / "best_model.pt",
    _PROJECT_ROOT / "models" / "lstm_pytorch_v3_logtarget" / "log_target_scaler.pkl",
    _PROJECT_ROOT / "data" / "processed" / "scaled" / "feature_scaler.pkl",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _payload(*, status="VALIDATED",
             provenance="REAL_MEASUREMENT_INPUTS_FROZEN_MODEL",
             validated=True, simulated=False):
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


def _demo_payload():
    return _payload(status="DEMONSTRATION_ONLY",
                    provenance="SIMULATION_OR_SYNTHETIC_INPUTS_FROZEN_MODEL",
                    validated=False, simulated=True)


def _live_network() -> ReservoirNetwork:
    return SimBridge(str(LIVE_CONFIG), str(THRESH_PATH)).cascade.network


def _adapter(network) -> LiveForecastAdapter:
    return LiveForecastAdapter.for_network(network, project_root=PROJECT_ROOT)


def _validated_bundle(network, nodes=None):
    adapter = _adapter(network)
    return adapter.build_bundle(
        {n: _payload() for n in (nodes if nodes is not None else NODES)}, "2026-09-14"
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def orchestrator():
    return LiveMPCOrchestrator()


@pytest.fixture
def live_sim():
    """Snapshot/restore the authoritative singleton around a test."""
    sim = LIVE_SIM
    saved = {
        "mode": sim.mode,
        "manual_gates": dict(sim.manual_gates),
        "manual_inflows": dict(sim.manual_inflows),
    }
    try:
        yield sim
    finally:
        sim.mode = saved["mode"]
        sim.manual_gates = saved["manual_gates"]
        sim.manual_inflows = saved["manual_inflows"]
        sim.bridge.init_cascade(50.0)


# ===========================================================================
# 1 & 3. All four reservoirs reach the MPC; the final action has four gates
# ===========================================================================

def test_all_four_reservoirs_reach_the_mpc(orchestrator, monkeypatch):
    """The network the MPC is handed is the FOUR-node authoritative network."""
    network = _live_network()
    seen = {}

    real = MPCController.decide

    def spy(self, net, **kwargs):
        seen["node_ids"] = list(net.processing_order)
        seen["n_nodes"] = len(net.nodes)
        return real(self, net, **kwargs)

    monkeypatch.setattr(MPCController, "decide", spy)
    orchestrator.decide(network, bundle=_validated_bundle(network))

    assert seen["n_nodes"] == 4
    assert seen["node_ids"] == NODES, "the MPC must see A -> B -> C -> D"


def test_final_action_contains_four_gate_decisions(orchestrator):
    network = _live_network()
    decision = orchestrator.decide(network, bundle=_validated_bundle(network))

    assert decision.controller_status == ControllerStatus.ACTIVE.value
    assert set(decision.gate_positions_fraction.keys()) == set(NODES)
    assert set(decision.gate_positions_pct.keys()) == set(NODES)
    assert set(decision.proposed_gate_positions_fraction.keys()) == set(NODES)
    assert set(decision.per_node.keys()) == set(NODES)
    for nid in NODES:
        assert 0.0 <= decision.gate_positions_fraction[nid] <= 1.0


# ===========================================================================
# 4. The MPC candidate/action representation has four dimensions
# ===========================================================================

def test_action_space_is_four_dimensional(orchestrator):
    network = _live_network()
    decision = orchestrator.decide(network, bundle=_validated_bundle(network))

    space = decision.action_space
    assert space["dimension"] == 4
    assert space["node_ids"] == NODES
    assert space["coordinated"] is True
    assert len(space["gate_levels"]) == len(orchestrator.mpc.config.gate_levels)
    # 6 gate levels ^ 4 reservoirs
    assert space["candidate_vectors"] == len(space["gate_levels"]) ** 4
    assert space["candidate_vectors"] == 1296


def test_mpc_scored_the_complete_action_vector(orchestrator):
    """The number of candidates scored equals the size of the 4-D product."""
    network = _live_network()
    decision = orchestrator.decide(network, bundle=_validated_bundle(network))

    assert decision.candidates_evaluated == decision.action_space["candidate_vectors"]


def test_candidate_search_covers_the_joint_cartesian_product(orchestrator, monkeypatch):
    """
    THE COORDINATION PROOF.

    Every candidate the validated MPC simulates carries a gate for ALL FOUR
    reservoirs, and the set of candidates is exactly the Cartesian product of the
    gate levels over the four nodes — i.e. the controller searches JOINT action
    vectors, not four independent per-reservoir choices.
    """
    network = _live_network()
    captured = []

    real = MPCController._simulate_trajectory

    def spy(self, real_network, candidate_gates, inflow_scenarios, node_ids):
        captured.append(dict(candidate_gates))
        return real(self, real_network, candidate_gates, inflow_scenarios, node_ids)

    monkeypatch.setattr(MPCController, "_simulate_trajectory", spy)
    orchestrator.decide(network, bundle=_validated_bundle(network))

    levels = orchestrator.mpc.config.gate_levels
    expected = {
        tuple(sorted(dict(zip(NODES, combo)).items()))
        for combo in itertools.product(levels, repeat=len(NODES))
    }
    observed = {tuple(sorted(c.items())) for c in captured}

    assert len(captured) == len(expected) == 1296
    assert observed == expected
    for candidate in captured:
        assert set(candidate.keys()) == set(NODES), "candidate is not four-dimensional"


def test_mpc_simulates_candidates_on_the_authoritative_network(orchestrator, monkeypatch):
    """
    Each candidate is rolled out on a clone of the VALIDATED ReservoirNetwork:
    the controller receives the authoritative network and never mutates it.
    """
    network = _live_network()
    seen = []
    real = MPCController._simulate_trajectory

    def spy(self, real_network, candidate_gates, inflow_scenarios, node_ids):
        before = {nid: real_network.nodes[nid].state.storage for nid in node_ids}
        trajectory = real(self, real_network, candidate_gates,
                          inflow_scenarios, node_ids)
        after = {nid: real_network.nodes[nid].state.storage for nid in node_ids}
        seen.append({
            "is_network": isinstance(real_network, ReservoirNetwork),
            "is_the_live_network": real_network is network,
            "order": list(real_network.processing_order),
            "nodes": set(candidate_gates.keys()),
            "unmutated": before == after,
        })
        return trajectory

    monkeypatch.setattr(MPCController, "_simulate_trajectory", spy)
    orchestrator.decide(network, bundle=_validated_bundle(network))

    assert seen
    for record in seen:
        assert record["is_network"] is True
        assert record["is_the_live_network"] is True
        assert record["order"] == NODES
        assert record["nodes"] == set(NODES), "candidate is not four-dimensional"
        assert record["unmutated"] is True, "the live network was mutated"


# ===========================================================================
# 2 & 5. D is not pinned; one reservoir's action moves the others
# ===========================================================================

def test_reservoir_d_manual_baseline_is_not_the_legacy_100_percent():
    """
    The hardcoded 'D = 100%' gate is gone.

    NOTE: this asserts on the module that DEFINES the baseline, not on the
    shared ``sim_state`` singleton, whose gate store other tests legitimately
    command and overwrite during a session.
    """
    source = STATE_MANAGER_PATH.read_text(encoding="utf-8")
    assert '"Virtual Reservoir D": 100.0' not in source, "legacy D pin is back"
    assert '"Virtual Reservoir D": 50.0' in source, "D has no explicit baseline"
    assert "LIVE_FORECAST_EXCLUDED" in source

    # The live gate store has four reservoirs and every value is a valid percent.
    assert set(LIVE_SIM.manual_gates) == set(NODES)
    for nid in NODES:
        assert 0.0 <= LIVE_SIM.manual_gates[nid] <= 100.0


def test_reservoir_d_is_commandable_through_the_api(live_sim):
    """D used to be uncommandable: POST /api/gate/reservoir_4 was rejected."""
    assert RESERVOIR_ID_TO_NODE["reservoir_4"] == "Virtual Reservoir D"
    assert set(RESERVOIR_ID_TO_NODE) == {"reservoir_1", "reservoir_2",
                                        "reservoir_3", "reservoir_4"}

    client.post("/api/controller/mode", json={"mode": "MANUAL"})
    response = client.post("/api/gate/reservoir_4", json={"value": 12.5})
    assert response.status_code == 200
    assert LIVE_SIM.manual_gates["Virtual Reservoir D"] == pytest.approx(12.5)

    twin = client.get("/api/state").json()
    assert twin["reservoirs"]["reservoir_4"]["gate"] == pytest.approx(0.125, abs=1e-9)


def test_reservoir_d_gate_is_not_permanently_fixed(live_sim):
    """
    BEHAVIOURAL NON-PINNING PROOF.

    D must track the operator's command like any other reservoir: two different
    commands must produce two different authoritative gate positions, both in
    the payload and in the physics.
    """
    sim = live_sim
    sim.mode = "MANUAL"
    sim.bridge.init_cascade(50.0)
    network = sim.bridge.cascade.network

    observed = []
    for percent in (10.0, 73.0):
        client.post("/api/gate/reservoir_4", json={"value": percent})
        sim.step()
        observed.append({
            "payload": client.get("/api/state").json()["reservoirs"]["reservoir_4"]["gate"],
            "physics": network.nodes["Virtual Reservoir D"].state.gate_position,
        })

    assert observed[0]["payload"] == pytest.approx(0.10, abs=1e-9)
    assert observed[1]["payload"] == pytest.approx(0.73, abs=1e-9)
    assert observed[0]["physics"] == pytest.approx(0.10, abs=1e-9)
    assert observed[1]["physics"] == pytest.approx(0.73, abs=1e-9)
    assert observed[0] != observed[1], "reservoir D is still pinned"


def test_no_special_case_controller_for_reservoir_d():
    """
    D must not have its own controller and must not be excluded from the
    coordinated decision. The only place D is named explicitly is the documented
    LIVE FORECAST inventory (D has no live validated forecast source) — never a
    gate decision.
    """
    for path in (ORCH_PATH, MPC_PATH, SAFETY_PATH):
        source = path.read_text(encoding="utf-8")
        assert "reservoir_d_controller" not in source.lower()
        assert "if nid == \"Virtual Reservoir D\"" not in source
        assert "if node_id == \"Virtual Reservoir D\"" not in source

    sm_source = STATE_MANAGER_PATH.read_text(encoding="utf-8")
    assert "LIVE_FORECAST_EXCLUDED" in sm_source
    assert 'if v_name == "Virtual Reservoir D":' not in sm_source


def test_upstream_action_changes_downstream_state_according_to_the_physics():
    """
    AUTHORITATIVE PHYSICS COUPLING.

    Opening Reservoir A's gate must raise the water that ARRIVES at Reservoir B
    after the validated routing delay, and must eventually move Reservoir D's
    storage. Nothing here is simulated by a test double: it is ReservoirNetwork.
    """
    network = _live_network()
    inflows = {nid: 0.0 for nid in NODES}
    gates_closed = {nid: 0.0 for nid in NODES}
    gates_a_open = dict(gates_closed, **{"Virtual Reservoir A": 1.0})

    closed = _live_network()
    opened = _live_network()

    delay_ab = next(c.delay for c in opened.connections
                    if c.source == "Virtual Reservoir A")
    atten_ab = next(c.attenuation for c in opened.connections
                    if c.source == "Virtual Reservoir A")
    assert delay_ab == 2, "validated A->B routing delay"
    assert atten_ab == pytest.approx(0.90)

    routed_b = []
    for _ in range(delay_ab + 1):
        closed.step(inflows, gates_closed)
        opened.step(inflows, gates_a_open)
        routed_b.append(opened.nodes["Virtual Reservoir B"].state.inflow_routed)

    # BEFORE the delay elapses nothing arrives; AFTER it, A's release does.
    assert routed_b[0] == pytest.approx(0.0)
    assert routed_b[1] == pytest.approx(0.0)
    expected_arrival = opened.nodes["Virtual Reservoir A"].max_release * atten_ab
    assert routed_b[delay_ab] == pytest.approx(expected_arrival, rel=1e-9)

    # The closed run never receives anything.
    assert closed.nodes["Virtual Reservoir B"].state.inflow_routed == pytest.approx(0.0)

    # B's storage therefore differs between the two coordinated actions.
    assert opened.nodes["Virtual Reservoir B"].state.storage != pytest.approx(
        closed.nodes["Virtual Reservoir B"].state.storage
    ) or opened.nodes["Virtual Reservoir B"].state.controlled_release != pytest.approx(
        closed.nodes["Virtual Reservoir B"].state.controlled_release
    )


def test_upstream_action_propagates_all_the_way_to_the_terminal_reservoir():
    """
    A -> B -> C -> D coupling over the full validated delay chain (2 + 1 + 1).

    Two identical networks are run with the same exogenous inflows and the same
    B/C/D gates; ONLY reservoir A's gate differs. Because each reservoir's
    release is limited by the water actually available, A's action changes what
    B can release, which changes what C can release, which changes D's storage.
    The step at which each downstream reservoir first diverges is exactly the
    routing delay of the connection feeding it — the authoritative physics.
    """
    def run(a_gate):
        net = _live_network()
        gates = {nid: 1.0 for nid in NODES}
        gates["Virtual Reservoir D"] = 0.0
        gates["Virtual Reservoir A"] = a_gate
        inflows = {nid: 0.0 for nid in NODES}
        series = []
        for _ in range(8):
            net.step(inflows, gates)
            series.append({
                nid: (net.nodes[nid].state.controlled_release,
                      net.nodes[nid].state.storage)
                for nid in NODES
            })
        return series

    closed = run(0.0)
    opened = run(1.0)

    def first_divergence(nid, field):
        for t in range(len(closed)):
            if abs(closed[t][nid][field] - opened[t][nid][field]) > 1e-12:
                return t + 1
        return None

    # A's action reaches B two days later (validated A->B delay = 2),
    # then C one day later, then D's storage one day after that.
    assert first_divergence("Virtual Reservoir A", 0) == 1
    assert first_divergence("Virtual Reservoir B", 0) == 3
    assert first_divergence("Virtual Reservoir C", 0) == 4
    assert first_divergence("Virtual Reservoir D", 1) == 5

    # The magnitude of the first arrival at B is A's release x attenuation.
    network = _live_network()
    max_release_a = network.nodes["Virtual Reservoir A"].max_release
    assert opened[0]["Virtual Reservoir A"][0] == pytest.approx(max_release_a)
    # At t=3 B releases exactly the water A sent, attenuated by 0.90; in the
    # closed run B has nothing to release at that step.
    assert closed[2]["Virtual Reservoir B"][0] == pytest.approx(0.0)
    assert opened[2]["Virtual Reservoir B"][0] == pytest.approx(
        max_release_a * 0.90, rel=1e-9
    )


# ===========================================================================


def test_terminal_reservoir_is_idukki_and_drives_the_downstream_flow():
    network = _live_network()
    assert network.processing_order[-1] == "Virtual Reservoir D"
    assert VALIDATED_NAMES["Virtual Reservoir D"] == "Idukki"

    network.step({nid: 0.0 for nid in NODES},
                 {nid: 0.0 for nid in NODES})
    assert network.nodes["Virtual Reservoir D"].state.total_outflow == pytest.approx(
        network.nodes["Virtual Reservoir D"].state.controlled_release
        + network.nodes["Virtual Reservoir D"].state.spill
    )


# ===========================================================================
# 6. The SafetyLayer receives the complete four-reservoir proposal
# ===========================================================================

def test_safety_layer_receives_the_complete_four_reservoir_proposal(orchestrator, monkeypatch):
    network = _live_network()
    calls = []
    real = orchestrator.safety.validate

    def spy(proposed, current, node_ids):
        calls.append({
            "proposed": dict(proposed),
            "current": dict(current),
            "node_ids": list(node_ids),
        })
        return real(proposed, current, node_ids)

    monkeypatch.setattr(orchestrator.safety, "validate", spy)
    orchestrator.decide(network, bundle=_validated_bundle(network))

    assert len(calls) == 1
    call = calls[0]
    assert call["node_ids"] == NODES
    assert set(call["proposed"].keys()) == set(NODES)
    assert set(call["current"].keys()) == set(NODES)


def test_safety_layer_returns_a_gate_for_every_reservoir(orchestrator):
    network = _live_network()
    decision = orchestrator.decide(network, bundle=_validated_bundle(network))

    result = SafetyLayer(
        max_gate_change_per_step=orchestrator.mpc.config.max_gate_change
    ).validate(decision.proposed_gate_positions_fraction,
               decision.gate_positions_fraction, NODES)
    assert set(result.validated_gates.keys()) == set(NODES)


# ===========================================================================
# 7. ReservoirNetwork receives all four safe actions
# ===========================================================================

def test_network_receives_the_safety_output_for_all_four_nodes(live_sim, monkeypatch):
    sim = live_sim
    sim.bridge.init_cascade(50.0)
    sim.mode = "MANUAL"
    sim.manual_inflows = {"Virtual Reservoir A": 3.0, "Virtual Reservoir B": 4.0,
                          "Virtual Reservoir C": 90.0, "Virtual Reservoir D": 0.0}
    sim.manual_gates = {n: 0.0 for n in NODES}
    sim.step()

    network = sim.bridge.cascade.network
    monkeypatch.setattr(sim, "_build_live_forecast_bundle",
                        lambda cf: _validated_bundle(network))

    sim.mode = "AI"
    sim.step()

    decision = sim.last_control_decision
    assert decision is not None
    assert decision.controller_status == ControllerStatus.ACTIVE.value

    for nid in NODES:
        assert network.nodes[nid].state.gate_position == pytest.approx(
            decision.gate_positions_fraction[nid], abs=1e-9
        )


def test_every_reservoir_gate_reaches_the_physics(live_sim):
    """Commanding each of the four reservoirs moves exactly that reservoir."""
    sim = live_sim
    sim.mode = "MANUAL"
    sim.bridge.init_cascade(50.0)
    network = sim.bridge.cascade.network

    for index, nid in enumerate(NODES, start=1):
        response = client.post(f"/api/gate/reservoir_{index}", json={"value": 25.0})
        assert response.status_code == 200
        sim.step()
        assert network.nodes[nid].state.gate_position == pytest.approx(0.25, abs=1e-9)


# ===========================================================================
# 8 & 9. Missing D / demonstration forecasts still block the coordination
# ===========================================================================

def test_missing_reservoir_d_forecast_blocks_the_coordinated_mpc(orchestrator):
    network = _live_network()
    bundle = _validated_bundle(network, nodes=[n for n in NODES if n != "Virtual Reservoir D"])
    decision = orchestrator.decide(network, bundle=bundle)

    assert decision.controller_status == ControllerStatus.BLOCKED.value
    assert decision.mpc_status == "NOT_INVOKED"
    assert any("Virtual Reservoir D:MISSING_FORECAST" in r for r in decision.reasons)
    # No fabrication of any kind.
    for nid in NODES:
        assert decision.gate_positions_fraction[nid] == pytest.approx(
            float(network.nodes[nid].state.gate_position)
        )


def test_demonstration_forecast_blocks_the_coordinated_mpc(orchestrator):
    network = _live_network()
    adapter = _adapter(network)
    bundle = adapter.build_bundle({n: _demo_payload() for n in NODES}, "2026-09-14")
    decision = orchestrator.decide(network, bundle=bundle)

    assert decision.controller_status == ControllerStatus.BLOCKED.value
    assert decision.forecast_control_eligible is False
    assert decision.control_applied is False
    assert decision.safety_layer_status == "NOT_APPLIED_MPC_BLOCKED"


def test_no_zero_fill_average_or_carry_forward_for_reservoir_d(orchestrator):
    """
    An absent D forecast must stay absent: the adapter marks all three horizons
    unavailable and nothing substitutes a zero, a mean or a carried value.
    """
    network = _live_network()
    adapter = _adapter(network)
    bundle = adapter.build_bundle(
        {n: _payload() for n in NODES if n != "Virtual Reservoir D"}, "2026-09-14"
    )

    fc_d = bundle.get("Virtual Reservoir D")
    assert fc_d is not None, "D must be explicitly represented, not omitted"
    assert fc_d.status_1d == ForecastStatus.UNAVAILABLE
    assert fc_d.target_1d is None
    assert fc_d.target_3d is None
    assert fc_d.target_7d is None
    for label in ("1d", "3d", "7d"):
        assert fc_d.is_available(label) is False

    # None of the other reservoirs were invented either.
    assert set(bundle.forecasts.keys()) == set(NODES)


# ===========================================================================
# 10. No legacy advisor; 11. the browser cannot bypass the path
# ===========================================================================

def test_legacy_forecast_aware_controller_is_not_authoritative():
    """
    The rule-based ForecastAwareController is not on the live path.

    The name may appear in a docstring explaining that it is NOT used; what must
    not exist is an import or an instantiation in the authoritative state manager.
    """
    source = STATE_MANAGER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
    assert "ForecastAwareController" not in imported
    assert not any("simulator.controllers" in m for m in imported)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id != "ForecastAwareController"

    assert "compute_ai_recommendation" not in source
    assert "self.mpc_orchestrator.decide(" in source
    assert "decision.gate_positions_pct" in source


def test_browser_cannot_bypass_the_coordinated_path():
    assert client.post("/api/state", json={"reservoirs": {}}).status_code == 405
    assert client.put("/api/state", json={"reservoirs": {}}).status_code == 405

    for bad in ("reservoir_5", "reservoir_99", "Virtual Reservoir D", "garbage"):
        assert client.post(f"/api/gate/{bad}", json={"value": 50.0}).status_code == 400


def test_only_the_four_known_reservoirs_are_commandable():
    source = ROUTES_PATH.read_text(encoding="utf-8")
    assert "RESERVOIR_ID_TO_NODE" in source
    assert set(RESERVOIR_ID_TO_NODE.values()) == set(NODES)


def test_gnn_remains_advisory_only():
    source = ORCH_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
    assert not any("gnn" in m.lower() for m in imported)
    assert "gnn_result" not in source


# ===========================================================================
# 12. WebSocket state carries all four reservoirs + controller provenance
# ===========================================================================

def test_websocket_state_has_four_reservoirs_and_provenance():
    client.post("/api/controller/mode", json={"mode": "MANUAL"})
    with client.websocket_connect("/ws/state") as ws:
        state = ws.receive_json()

    assert set(state["reservoirs"].keys()) == set(TWIN_KEY_ORDER)
    assert state["cascade"]["count"] == 4
    assert "control" in state
    assert "safety_layer_status" in state["control"]
    assert "action_space" in state["control"]


def test_twin_payload_declares_the_four_reservoir_cascade():
    twin = client.get("/api/state").json()
    cascade = twin["cascade"]

    assert cascade["count"] == 4
    assert cascade["order"] == list(TWIN_KEY_ORDER)
    assert [r["name"] for r in cascade["reservoirs"]] == [
        "Anayirankal", "Ponmudi", "Idamalayar", "Idukki"
    ]
    assert [r["node_id"] for r in cascade["reservoirs"]] == NODES
    assert cascade["terminal"]["key"] == "reservoir_4"
    assert cascade["terminal"]["name"] == "Idukki"


def test_every_twin_reservoir_carries_its_authoritative_identity():
    twin = client.get("/api/state").json()
    for index, nid in enumerate(NODES, start=1):
        res = twin["reservoirs"][f"reservoir_{index}"]
        assert res["node_id"] == nid
        assert res["repository_name"] == VALIDATED_NAMES[nid]
        assert res["cascade_position"] == index - 1
        assert res["is_terminal"] is (index == 4)


def test_twin_ui_represents_all_four_reservoirs():
    html = TWIN_INDEX_PATH.read_text(encoding="utf-8")
    # a fourth telemetry card, a fourth topology node and a fourth gate slider
    assert "resCard(1), resCard(2), resCard(3), resCard(4)" in html
    assert 'data-ref="tn4"' in html
    assert 'data-ctl="g4"' in html
    assert "cascade.reservoirs" in html
    # the authoritative names, replacing the mismatched legacy three-name list
    assert "'Anayirankal', 'Ponmudi', 'Idamalayar', 'Idukki'" in html
    assert "RESERVOIRS" in html and "MPC ACTION" in html


def test_demo_badge_wiring_is_not_dead_code():
    """Stage 9 bug fix: the badge helper must read the real element cache."""
    html = TWIN_INDEX_PATH.read_text(encoding="utf-8")
    # The dead reference must be gone from the executable code (it is still
    # named in the explanatory comment, which is why this checks assignments).
    assert "= this.refs" not in html
    assert "this.refs ||" not in html
    assert "const refs = this.el || {};" in html


# ===========================================================================
# 13. Canonical units
# ===========================================================================

def test_canonical_units_hold_for_all_four_reservoirs(orchestrator):
    network = _live_network()
    decision = orchestrator.decide(network, bundle=_validated_bundle(network))

    from src.common import units
    for nid in NODES:
        fraction = decision.gate_positions_fraction[nid]
        assert 0.0 <= fraction <= 1.0, "internal gate must be a FRACTION"
        assert decision.gate_positions_pct[nid] == pytest.approx(
            units.gate_fraction_to_percent(fraction), abs=1e-9
        )


def test_no_gate_value_crosses_the_boundary_implicitly():
    for path in (ORCH_PATH, MPC_PATH):
        source = path.read_text(encoding="utf-8")
        assert "/ 100.0" not in source
        assert "* 100.0" not in source


# ===========================================================================
# 14. Determinism
# ===========================================================================

def test_four_reservoir_decision_is_deterministic():
    outcomes = []
    for _ in range(2):
        network = _live_network()
        orch = LiveMPCOrchestrator()
        decision = orch.decide(network, bundle=_validated_bundle(network))
        score = decision.mpc_objective_score
        outcomes.append({
            "gates": {nid: round(decision.gate_positions_fraction[nid], 12)
                      for nid in NODES},
            "score": None if score is None else round(score, 9),
            "candidates": decision.candidates_evaluated,
            "status": decision.controller_status,
        })
    assert outcomes[0] == outcomes[1]


def test_candidate_enumeration_order_is_deterministic(orchestrator):
    network = _live_network()
    first = []
    second = []
    real = MPCController._simulate_trajectory

    def make(sink):
        def spy(self, real_network, candidate_gates, inflow_scenarios, node_ids):
            sink.append(tuple(candidate_gates[nid] for nid in NODES))
            return real(self, real_network, candidate_gates, inflow_scenarios, node_ids)
        return spy

    import unittest.mock as mock
    with mock.patch.object(MPCController, "_simulate_trajectory", make(first)):
        orchestrator.decide(network, bundle=_validated_bundle(network))
    with mock.patch.object(MPCController, "_simulate_trajectory", make(second)):
        LiveMPCOrchestrator().decide(_live_network(), bundle=_validated_bundle(_live_network()))

    assert first == second


# ===========================================================================
# 7 (report) — topology / physics verification
# ===========================================================================

def test_validated_topology_is_a_b_c_d_with_d_terminal():
    network = _live_network()
    assert network.processing_order == NODES
    assert network._terminal_node_id == "Virtual Reservoir D"
    pairs = {(c.source, c.destination) for c in network.connections}
    assert pairs == {
        ("Virtual Reservoir A", "Virtual Reservoir B"),
        ("Virtual Reservoir B", "Virtual Reservoir C"),
        ("Virtual Reservoir C", "Virtual Reservoir D"),
    }


def test_validated_routing_parameters_are_unchanged():
    with open(TOPOLOGY_PATH) as fh:
        cfg = yaml.safe_load(fh)
    conns = {(c["source"], c["destination"]): c for c in cfg["connections"]}

    expected = {
        ("Reservoir_A", "Reservoir_B"): (2, 0.90),
        ("Reservoir_B", "Reservoir_C"): (1, 0.85),
        ("Reservoir_C", "Reservoir_D"): (1, 0.80),
    }
    for key, (delay, atten) in expected.items():
        assert conns[key]["routing_delay_days"]["value"] == delay
        assert conns[key]["attenuation_factor"]["value"] == pytest.approx(atten)

    network = _live_network()
    observed = {(c.source, c.destination): (c.delay, c.attenuation)
                for c in network.connections}
    assert observed[("Virtual Reservoir A", "Virtual Reservoir B")] == (2, pytest.approx(0.90))
    assert observed[("Virtual Reservoir B", "Virtual Reservoir C")] == (1, pytest.approx(0.85))
    assert observed[("Virtual Reservoir C", "Virtual Reservoir D")] == (1, pytest.approx(0.80))


def test_downstream_capacity_is_unchanged():
    network = _live_network()
    assert network.downstream_capacity == pytest.approx(50.0)
    with open(TOPOLOGY_PATH) as fh:
        cfg = yaml.safe_load(fh)
    assert cfg["topology"]["downstream_capacity_mcm_day"]["value"] == pytest.approx(50.0)


def test_mass_balance_holds_under_four_reservoir_control():
    network = _live_network()
    inflows = {"Virtual Reservoir A": 3.0, "Virtual Reservoir B": 4.0,
               "Virtual Reservoir C": 90.0, "Virtual Reservoir D": 1.0}
    gates = {"Virtual Reservoir A": 0.4, "Virtual Reservoir B": 0.35,
             "Virtual Reservoir C": 0.5, "Virtual Reservoir D": 0.25}
    for _ in range(10):
        network.step(inflows, gates)

    balance = network.mass_balance_check()
    assert abs(balance["residual_error"]) < 1e-9


def test_non_terminal_spill_is_not_routed_downstream():
    """Spill from A/B/C leaves the network; only the terminal node's is downstream."""
    network = _live_network()
    # Fill A far beyond capacity so it must spill.
    node_a = network.nodes["Virtual Reservoir A"]
    node_a.state.storage = node_a.capacity
    before_b = network.nodes["Virtual Reservoir B"].state.storage
    network.step({"Virtual Reservoir A": node_a.capacity * 2, **{n: 0.0 for n in NODES[1:]}},
                 {n: 0.0 for n in NODES})
    assert node_a.state.spill > 0.0
    assert network.mass_balance_check()["total_nonterminal_spill"] > 0.0
    # B gained nothing from A's spill (A's gate was closed, so nothing was routed).
    assert network.nodes["Virtual Reservoir B"].state.inflow_routed == pytest.approx(0.0)
    assert network.nodes["Virtual Reservoir B"].state.storage == pytest.approx(before_b)


def test_the_physics_source_is_unmodified():
    source = NETWORK_PATH.read_text(encoding="utf-8")
    for banned in ("live_mpc_orchestrator", "SafetyLayer", "gate_levels"):
        assert banned not in source


# ===========================================================================
# 16. Frozen artifacts
# ===========================================================================

def test_stage9_does_not_modify_frozen_artifacts(orchestrator):
    before = {p: _sha(p) for p in FROZEN_ARTIFACTS}
    network = _live_network()
    orchestrator.decide(network, bundle=_validated_bundle(network))
    assert {p: _sha(p) for p in FROZEN_ARTIFACTS} == before


def test_validated_mpc_and_safety_are_unmodified():
    mpc = MPC_PATH.read_text(encoding="utf-8")
    assert "class MPCController" in mpc
    assert "live_mpc_orchestrator" not in mpc
    assert "itertools.product(self.config.gate_levels, repeat=len(node_ids))" in mpc

    safety = SAFETY_PATH.read_text(encoding="utf-8")
    assert "class SafetyLayer" in safety
    assert "live_mpc_orchestrator" not in safety


# ===========================================================================
# BEHAVIOURAL INTEGRATION — four-reservoir proposal -> SafetyLayer -> network
# ===========================================================================

def test_four_reservoir_proposal_through_safety_reaches_the_physics(
    live_sim, monkeypatch
):
    """
    END TO END, on the authoritative simulation.

    A joint four-reservoir MPC proposal that violates the validated SafetyLayer
    is modified by the layer, and the SAFE four-reservoir action — not the raw
    proposal — is what ReservoirNetwork receives. The resulting state must match
    the authoritative physics for all four reservoirs.
    """
    sim = live_sim
    sim.bridge.init_cascade(50.0)
    sim.mode = "MANUAL"
    sim.manual_inflows = {"Virtual Reservoir A": 3.0, "Virtual Reservoir B": 4.0,
                          "Virtual Reservoir C": 90.0, "Virtual Reservoir D": 0.0}
    sim.manual_gates = {n: 0.0 for n in NODES}
    sim.step()

    network = sim.bridge.cascade.network
    for nid in NODES:
        assert network.nodes[nid].state.gate_position == pytest.approx(0.0)

    # A four-reservoir proposal that is unsafe on every count.
    from src.controller.mpc_controller import ControlDecision
    unsafe = ControlDecision(
        gate_positions={
            "Virtual Reservoir A": 5.0,          # out of range -> clamp
            "Virtual Reservoir B": -3.0,         # out of range -> clamp
            "Virtual Reservoir C": float("nan"), # invalid -> 0.1
            "Virtual Reservoir D": 0.9,          # in range but too fast
        },
        objective_score=1.0, status="OPTIMAL", per_node={}, candidates_evaluated=1296,
    )

    monkeypatch.setattr(sim, "_build_live_forecast_bundle",
                        lambda cf: _validated_bundle(network))
    monkeypatch.setattr(sim.mpc_orchestrator.mpc, "decide", lambda *a, **k: unsafe)

    sim.mode = "AI"
    sim.step()

    decision = sim.last_control_decision
    assert decision.controller_status == ControllerStatus.ACTIVE.value
    assert decision.safety_layer_status == "CORRECTED"
    assert decision.safety_modified is True
    assert decision.action_space["dimension"] == 4

    limit = sim.mpc_orchestrator.mpc.config.max_gate_change
    # What the validated SafetyLayer produced (Stage 8 contract).
    safety_expected = {
        "Virtual Reservoir A": limit,    # 5.0 -> clamp 1.0 -> rate-limited from 0.0
        "Virtual Reservoir B": 0.0,      # -3.0 -> clamp 0.0
        "Virtual Reservoir C": 0.1,      # NaN -> conservative fallback
        "Virtual Reservoir D": limit,    # 0.9 -> rate-limited from 0.0
    }

    # 1. The SafetyLayer produced the sanitised four-reservoir action...
    for nid, value in safety_expected.items():
        assert decision.safety_layer_gate_positions_fraction[nid] == pytest.approx(
            value, abs=1e-12
        )
    # ...and the action that is APPLIED is that action or a downstream-tightened
    # version of it (Stage 10), never a looser one.
    for nid, value in safety_expected.items():
        assert decision.gate_positions_fraction[nid] <= value + 1e-12

    # 2. ReservoirNetwork RECEIVED the final action, for all four nodes.
    for nid in NODES:
        assert network.nodes[nid].state.gate_position == pytest.approx(
            decision.gate_positions_fraction[nid], abs=1e-9
        )

    # 3. ...and NOT the raw proposal.
    assert network.nodes["Virtual Reservoir A"].state.gate_position != pytest.approx(1.0)
    assert network.nodes["Virtual Reservoir D"].state.gate_position != pytest.approx(0.9)

    # 4. The resulting physics reflects the applied action for every reservoir.
    for nid in NODES:
        node = network.nodes[nid]
        assert node.state.controlled_release == pytest.approx(
            decision.gate_positions_fraction[nid] * node.max_release, abs=1e-6
        )

    # 5. Mass balance still holds with four-reservoir control.
    assert abs(network.mass_balance_check()["residual_error"]) < 1e-9


def test_coordinated_control_moves_more_than_one_reservoir(live_sim, monkeypatch):
    """The coordinated action is genuinely four-reservoir, not one-reservoir."""
    sim = live_sim
    sim.bridge.init_cascade(50.0)
    sim.mode = "MANUAL"
    sim.manual_gates = {n: 0.0 for n in NODES}
    sim.step()

    network = sim.bridge.cascade.network
    monkeypatch.setattr(sim, "_build_live_forecast_bundle",
                        lambda cf: _validated_bundle(network))

    sim.mode = "AI"
    sim.step()

    decision = sim.last_control_decision
    moved = [
        nid for nid in NODES
        if not math.isclose(network.nodes[nid].state.gate_position, 0.0, abs_tol=1e-12)
    ]
    assert len(moved) >= 2, f"only {moved} moved — control is not coordinated"
    assert set(decision.per_node.keys()) == set(NODES)
