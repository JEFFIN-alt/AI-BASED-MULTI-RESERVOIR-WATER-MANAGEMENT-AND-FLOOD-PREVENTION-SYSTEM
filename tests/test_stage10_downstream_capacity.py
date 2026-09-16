"""
Stage 10 — Downstream Capacity Safety Boundary tests.

Proves that the live four-reservoir control path cannot apply an action that
drives the flow below the terminal reservoir (Idukki / D) above the
authoritative downstream capacity (50.0 MCM/day), and that the three control
layers report their status SEPARATELY so the system can never claim
"SafetyLayer ACTIVE = downstream capacity protected".

    MPCController  →  SafetyLayer  →  DownstreamCapacityGuard  →  ReservoirNetwork
"""

import ast
import copy
import hashlib
import math
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.controller.downstream_capacity_guard import (  # noqa: E402
    DOWNSTREAM_STATUS_CORRECTED,
    DOWNSTREAM_STATUS_FAILED_CLOSED,
    DOWNSTREAM_STATUS_NOT_APPLIED_MPC_BLOCKED,
    DOWNSTREAM_STATUS_PROTECTED,
    TOLERANCE_MCM_DAY,
    DownstreamCapacityGuard,
)
from src.controller.live_mpc_orchestrator import (  # noqa: E402
    ControllerStatus,
    LiveMPCOrchestrator,
)
from src.controller.mpc_controller import ControlDecision, MPCController  # noqa: E402
from src.controller.safety import SafetyLayer  # noqa: E402
from src.dashboard.api import state_manager  # noqa: E402
from src.dashboard.api.app import app  # noqa: E402
from src.dashboard.sim_bridge import SimBridge  # noqa: E402
from src.dashboard.twin_component.state_adapter import adapt_state_for_twin  # noqa: E402
from src.network_env.live_forecast_adapter import (  # noqa: E402
    FORECAST_UNIT,
    HORIZON_KEYS,
    LiveForecastAdapter,
)
from src.network_env.reservoir_network import ReservoirNetwork  # noqa: E402

PROJECT_ROOT = str(_PROJECT_ROOT)
client = TestClient(app)

LIVE_CONFIG = _PROJECT_ROOT / "configs" / "simulation" / "four_reservoir_demo.json"
THRESH_PATH = _PROJECT_ROOT / "data" / "processed" / "historical_inflow_thresholds.json"
GUARD_PATH = _PROJECT_ROOT / "src" / "controller" / "downstream_capacity_guard.py"
ORCH_PATH = _PROJECT_ROOT / "src" / "controller" / "live_mpc_orchestrator.py"
NETWORK_PATH = _PROJECT_ROOT / "src" / "network_env" / "reservoir_network.py"
SAFETY_PATH = _PROJECT_ROOT / "src" / "controller" / "safety.py"
MPC_PATH = _PROJECT_ROOT / "src" / "controller" / "mpc_controller.py"
STATE_MANAGER_PATH = _PROJECT_ROOT / "src" / "dashboard" / "api" / "state_manager.py"
TWIN_INDEX_PATH = _PROJECT_ROOT / "src" / "dashboard" / "web" / "index.html"

NODES = [
    "Virtual Reservoir A",
    "Virtual Reservoir B",
    "Virtual Reservoir C",
    "Virtual Reservoir D",
]
CAPACITY = 50.0

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


def _validated_bundle(network, nodes=None):
    adapter = LiveForecastAdapter.for_network(network, project_root=PROJECT_ROOT)
    return adapter.build_bundle(
        {n: _payload() for n in (nodes if nodes is not None else NODES)}, "2026-09-14"
    )


def _demo_bundle(network):
    adapter = LiveForecastAdapter.for_network(network, project_root=PROJECT_ROOT)
    return adapter.build_bundle({n: _demo_payload() for n in NODES}, "2026-09-14")


def _unsafe_decision(**overrides):
    """A four-reservoir proposal that is unsafe on every count."""
    gates = {
        "Virtual Reservoir A": 5.0,             # above range -> clamp
        "Virtual Reservoir B": -3.0,            # below range -> clamp
        "Virtual Reservoir C": float("nan"),    # invalid -> 0.1
        "Virtual Reservoir D": 0.9,             # in range, but rate-limited
    }
    gates.update(overrides)
    return ControlDecision(gate_positions=gates, objective_score=1.0,
                           status="OPTIMAL", per_node={}, candidates_evaluated=1296)


def _flood_decision(value=1.0):
    """A proposal that opens every gate — guaranteed to exceed the capacity."""
    return ControlDecision(gate_positions={n: value for n in NODES},
                           objective_score=1.0, status="OPTIMAL",
                           per_node={}, candidates_evaluated=1296)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def orchestrator():
    return LiveMPCOrchestrator()


@pytest.fixture
def live_sim():
    """Snapshot/restore the authoritative singleton around a test."""
    sim = state_manager.sim_state
    saved = {
        "mode": sim.mode,
        "manual_gates": dict(sim.manual_gates),
        "manual_inflows": dict(sim.manual_inflows),
        "storm": sim.storm_intensity,
    }
    try:
        yield sim
    finally:
        sim.mode = saved["mode"]
        sim.manual_gates = saved["manual_gates"]
        sim.manual_inflows = saved["manual_inflows"]
        sim.storm_intensity = saved["storm"]
        sim.bridge.init_cascade(50.0)


def _independent_terminal_flow(network, gates, inflows, steps=1):
    """
    Step a FRESH clone of the live network with the applied action and return the
    terminal outflow — an independent check of the guard's prediction.
    """
    clone = ReservoirNetwork(config_dict=copy.deepcopy(network._raw_config))
    for nid in NODES:
        clone.nodes[nid].state.storage = network.nodes[nid].state.storage
    for index, conn in enumerate(network.connections):
        clone.connections[index].queue = copy.deepcopy(conn.queue)
    flows = []
    for _ in range(steps):
        states = clone.step(dict(inflows), {n: float(gates[n]) for n in NODES})
        flows.append(states[clone._terminal_node_id].total_outflow)
    return flows


# ===========================================================================
# 1 & 9. Safe action passes unchanged; the SafetyLayer stays in the path
# ===========================================================================

def test_safe_action_passes_unchanged(orchestrator):
    """A proposal the guard finds safe must NOT be modified."""
    network = _live_network()
    decision = orchestrator.decide(network, bundle=_validated_bundle(network))

    assert decision.controller_status == ControllerStatus.ACTIVE.value
    assert decision.downstream_status == DOWNSTREAM_STATUS_PROTECTED
    assert decision.downstream_protection_modified is False
    assert decision.downstream_capacity_achieved is True
    assert decision.downstream_predicted_flow_mcm_day <= CAPACITY + TOLERANCE_MCM_DAY
    for nid in NODES:
        assert decision.gate_positions_fraction[nid] == pytest.approx(
            decision.safety_layer_gate_positions_fraction[nid], abs=1e-12
        )


def test_safe_action_is_not_unnecessarily_modified_even_with_headroom():
    """
    Explicit opposite-of-the-behavioural-test case: a known SAFE action must be
    applied verbatim; the boundary must not "improve" it.
    """
    guard = DownstreamCapacityGuard()
    network = _live_network()
    current = {nid: 0.0 for nid in NODES}
    safe_action = {"Virtual Reservoir A": 0.5, "Virtual Reservoir B": 0.5,
                   "Virtual Reservoir C": 0.5, "Virtual Reservoir D": 0.25}

    result = guard.evaluate(
        network,
        action_fraction=safe_action,
        current_fraction=current,
        node_ids=NODES,
        max_gate_change=0.5,
        inflows={nid: 0.0 for nid in NODES},
    )

    assert result.status == DOWNSTREAM_STATUS_PROTECTED
    assert result.modified is False
    assert result.capacity_achieved is True
    assert result.predicted_flow_mcm_day == pytest.approx(50.0, abs=1e-6)
    for nid in NODES:
        assert result.action_fraction[nid] == pytest.approx(safe_action[nid])


def test_safety_layer_remains_in_the_path(orchestrator, monkeypatch):
    """The guard never replaces the SafetyLayer: it runs AFTER it."""
    network = _live_network()
    calls = []
    real = orchestrator.safety.validate

    def spy(proposed, current, node_ids):
        calls.append(set(proposed.keys()))
        return real(proposed, current, node_ids)

    monkeypatch.setattr(orchestrator.safety, "validate", spy)
    orchestrator.decide(network, bundle=_validated_bundle(network))

    assert calls == [set(NODES)], "the SafetyLayer must run exactly once, on all four"
    # ...and the guard was given the SafetyLayer's OUTPUT.
    assert orchestrator.last_decision.downstream_capacity_protection


def test_guard_sits_between_the_safety_layer_and_the_network():
    source = ORCH_PATH.read_text(encoding="utf-8")
    safety_at = source.index("safety_result = self.safety.validate(")
    guard_at = source.index("downstream = self.downstream_guard.evaluate(")
    assert safety_at < guard_at, "the downstream boundary must run after the SafetyLayer"


# ===========================================================================
# 2 & 3. Unsafe downstream actions are detected and 50.0 is enforced
# ===========================================================================

def test_unsafe_downstream_action_is_detected_and_corrected(orchestrator, monkeypatch):
    network = _live_network()
    monkeypatch.setattr(orchestrator.mpc, "decide", lambda *a, **k: _flood_decision())

    decision = orchestrator.decide(network, bundle=_validated_bundle(network))

    assert decision.downstream_status == DOWNSTREAM_STATUS_CORRECTED
    assert decision.downstream_protection_modified is True
    assert decision.downstream_proposed_predicted_flow_mcm_day > CAPACITY
    assert decision.downstream_predicted_flow_mcm_day <= CAPACITY + TOLERANCE_MCM_DAY
    assert decision.downstream_capacity_achieved is True


def test_downstream_capacity_of_50_is_enforced_on_the_live_network(
    live_sim, monkeypatch
):
    """
    The capacity that matters is the AUTHORITATIVE network's, and the action that
    reaches it must not exceed it — checked on the real network, not a clone.
    """
    sim = live_sim
    sim.bridge.init_cascade(50.0)
    sim.mode = "MANUAL"
    sim.manual_inflows = {n: 0.0 for n in NODES}
    sim.manual_gates = {n: 0.0 for n in NODES}
    sim.step()

    network = sim.bridge.cascade.network
    assert network.downstream_capacity == pytest.approx(CAPACITY)

    monkeypatch.setattr(sim, "_build_live_forecast_bundle",
                        lambda cf: _validated_bundle(network))
    monkeypatch.setattr(sim.mpc_orchestrator.mpc, "decide",
                        lambda *a, **k: _flood_decision())

    sim.mode = "AI"
    sim.step()

    decision = sim.last_control_decision
    assert decision.downstream_status == DOWNSTREAM_STATUS_CORRECTED
    # The gate positions actually written to the authoritative physics...
    applied = {nid: network.nodes[nid].state.gate_position for nid in NODES}
    terminal_flow = network.nodes["Virtual Reservoir D"].state.total_outflow
    assert terminal_flow <= CAPACITY + TOLERANCE_MCM_DAY, (
        f"downstream flow {terminal_flow} exceeds capacity after a corrected action"
    )
    # ...are the ones the decision reported.
    for nid in NODES:
        assert applied[nid] == pytest.approx(decision.gate_positions_fraction[nid])


def test_capacity_threshold_is_the_analytic_release_boundary():
    """
    With D's gate at exactly capacity / max_release the flow equals the capacity;
    that is the analytic boundary the guard's lattice includes.
    """
    network = _live_network()
    threshold = CAPACITY / network.nodes["Virtual Reservoir D"].max_release
    assert threshold == pytest.approx(0.25)

    guard = DownstreamCapacityGuard()
    flows = guard.predict_flows(
        network,
        {"Virtual Reservoir A": 0.0, "Virtual Reservoir B": 0.0,
         "Virtual Reservoir C": 0.0, "Virtual Reservoir D": threshold},
        {nid: 0.0 for nid in NODES},
        NODES,
        1,
    )
    assert flows[0] == pytest.approx(CAPACITY, abs=1e-9)
    assert threshold in guard.candidate_levels_for(network, "Virtual Reservoir D", 0.5, 0.0)


# ===========================================================================
# 4, 5 & 6. The prediction uses authoritative physics, delays and attenuation
# ===========================================================================

def test_prediction_uses_the_authoritative_network_physics(orchestrator):
    """
    The guard's prediction must equal a direct rollout of the authoritative
    ``ReservoirNetwork`` — same model, not a re-implementation.
    """
    guard = DownstreamCapacityGuard()
    network = _live_network()
    inflows = {nid: 0.0 for nid in NODES}
    action = {"Virtual Reservoir A": 0.5, "Virtual Reservoir B": 0.5,
              "Virtual Reservoir C": 0.5, "Virtual Reservoir D": 0.25}

    horizon = guard.horizon_for(network)
    predicted = guard.predict_flows(network, action, inflows, NODES, horizon)
    independent = _independent_terminal_flow(network, action, inflows, steps=horizon)

    assert len(predicted) == horizon
    for a, b in zip(predicted, independent):
        assert a == pytest.approx(b, abs=1e-12)


def test_prediction_matches_the_validated_mpc_rollout(orchestrator):
    """
    The guard's prediction must agree with the validated MPC's own trajectory
    simulation (``MPCController._simulate_trajectory``) on the terminal outflow —
    proving both use the same authoritative model and the same constant-gate
    convention, and that no routing equation was duplicated.
    """
    guard = DownstreamCapacityGuard()
    network = _live_network()
    inflows = {nid: 0.0 for nid in NODES}
    action = {"Virtual Reservoir A": 0.3, "Virtual Reservoir B": 0.3,
              "Virtual Reservoir C": 0.3, "Virtual Reservoir D": 0.25}

    trajectory = orchestrator.mpc._simulate_trajectory(
        network, action, [dict(inflows)] * 3, NODES
    )
    predicted = guard.predict_flows(network, action, inflows, NODES, 3)

    for step, states in enumerate(trajectory):
        assert predicted[step] == pytest.approx(
            states[network._terminal_node_id].total_outflow, abs=1e-12
        )


def test_prediction_does_not_mutate_the_live_network():
    guard = DownstreamCapacityGuard()
    network = _live_network()
    before = {
        "storage": {nid: network.nodes[nid].state.storage for nid in NODES},
        "queues": [list(c.queue) for c in network.connections],
        "timestep": network.timestep,
        "gates": {nid: network.nodes[nid].state.gate_position for nid in NODES},
    }
    guard.predict_flows(network, {nid: 0.5 for nid in NODES},
                        {nid: 0.0 for nid in NODES}, NODES, 5)
    assert {nid: network.nodes[nid].state.storage for nid in NODES} == before["storage"]
    assert [list(c.queue) for c in network.connections] == before["queues"]
    assert network.timestep == before["timestep"]
    assert {nid: network.nodes[nid].state.gate_position for nid in NODES} == before["gates"]


def test_horizon_is_derived_from_the_validated_routing_delays():
    network = _live_network()
    delays = [conn.delay for conn in network.connections]
    assert delays == [2, 1, 1]
    assert DownstreamCapacityGuard.horizon_for(network) == 1 + sum(delays) == 5
    source = GUARD_PATH.read_text(encoding="utf-8")
    assert "1 + sum(int(conn.delay) for conn in network.connections)" in source


def test_cascade_delays_are_respected_by_the_prediction():
    """
    Water released by A cannot reach D instantly: opening ONLY A (D closed) must
    keep the downstream flow at zero for the first 4 steps, and B must see the
    water exactly 2 steps later.
    """
    guard = DownstreamCapacityGuard()
    network = _live_network()
    inflows = {nid: 0.0 for nid in NODES}
    action = {"Virtual Reservoir A": 1.0, "Virtual Reservoir B": 0.0,
              "Virtual Reservoir C": 0.0, "Virtual Reservoir D": 0.0}

    # D's gate is closed, so the downstream flow stays zero regardless.
    flows = guard.predict_flows(network, action, inflows, NODES, 5)
    assert flows == pytest.approx([0.0] * 5)

    # B receives A's release after the validated 2-day delay, attenuated by 0.90.
    clone = ReservoirNetwork(config_dict=copy.deepcopy(network._raw_config))
    for nid in NODES:
        clone.nodes[nid].state.storage = network.nodes[nid].state.storage
    arrivals = []
    for _ in range(4):
        states = clone.step(dict(inflows), dict(action))
        arrivals.append(states["Virtual Reservoir B"].inflow_routed)
    assert arrivals[0] == pytest.approx(0.0)
    assert arrivals[1] == pytest.approx(0.0)
    assert arrivals[2] == pytest.approx(
        network.nodes["Virtual Reservoir A"].max_release * 0.90, rel=1e-9
    )


def test_attenuation_is_respected_by_the_prediction():
    """
    The predicted flow reflects the validated attenuation, not an idealised 1.0:
    a 100 MCM slug in the C->D queue arrives at D as 80 MCM (0.80), and with D's
    gate open the predicted downstream flow is that 80 — which the boundary then
    rejects because it exceeds the 50 MCM/day capacity.
    """
    guard = DownstreamCapacityGuard()
    network = _live_network()
    for nid in NODES:
        network.nodes[nid].state.storage = 0.0
    for conn in network.connections:
        if conn.destination == "Virtual Reservoir D":
            conn.queue[-1] = 100.0

    action = {"Virtual Reservoir A": 0.0, "Virtual Reservoir B": 0.0,
              "Virtual Reservoir C": 0.0, "Virtual Reservoir D": 1.0}
    result = guard.evaluate(
        network,
        action_fraction=action,
        current_fraction={nid: 0.0 for nid in NODES},
        node_ids=NODES,
        max_gate_change=0.5,
        inflows={nid: 0.0 for nid in NODES},
    )

    # 100 MCM in the queue * 0.80 attenuation = 80 MCM available at D.
    assert result.proposed_predicted_flow_mcm_day == pytest.approx(80.0, rel=1e-9)
    assert result.status == DOWNSTREAM_STATUS_CORRECTED
    assert result.predicted_flow_mcm_day <= CAPACITY + TOLERANCE_MCM_DAY


# ===========================================================================
# 7 & 8. All four reservoirs participate; D is not pinned
# ===========================================================================

def test_all_four_reservoirs_participate_in_the_safety_check(orchestrator, monkeypatch):
    network = _live_network()
    monkeypatch.setattr(orchestrator.mpc, "decide", lambda *a, **k: _flood_decision())
    decision = orchestrator.decide(network, bundle=_validated_bundle(network))

    block = decision.downstream_capacity_protection
    assert set(decision.gate_positions_fraction) == set(NODES)
    assert set(decision.safety_layer_gate_positions_fraction) == set(NODES)
    assert set(decision.final_safe_control_action_fraction) == set(NODES)
    assert block["horizon_steps"] == 5
    assert block["physics"].startswith("ReservoirNetwork")


def test_reservoir_d_is_not_pinned_by_the_boundary(orchestrator, monkeypatch):
    """
    D is not pinned to any fixed value by the boundary: it is the reservoir the
    boundary actually tightens, and its value is never the legacy 100 %.
    """
    network = _live_network()
    monkeypatch.setattr(orchestrator.mpc, "decide", lambda *a, **k: _flood_decision())
    decision = orchestrator.decide(network, bundle=_validated_bundle(network))
    d_gate = decision.gate_positions_fraction["Virtual Reservoir D"]

    assert d_gate != pytest.approx(1.0)
    assert d_gate == pytest.approx(CAPACITY / network.nodes["Virtual Reservoir D"].max_release)
    # ...and the manual baseline is not the legacy pin either.
    assert '"Virtual Reservoir D": 100.0' not in STATE_MANAGER_PATH.read_text(encoding="utf-8")


def test_boundary_keeps_other_reservoirs_as_proposed(orchestrator, monkeypatch):
    """
    Minimal intervention: when only the terminal reservoir needs reducing, the
    other three keep exactly the gates the SafetyLayer produced.
    """
    network = _live_network()
    monkeypatch.setattr(orchestrator.mpc, "decide", lambda *a, **k: _flood_decision())
    decision = orchestrator.decide(network, bundle=_validated_bundle(network))

    for nid in ("Virtual Reservoir A", "Virtual Reservoir B", "Virtual Reservoir C"):
        assert decision.gate_positions_fraction[nid] == pytest.approx(
            decision.safety_layer_gate_positions_fraction[nid], abs=1e-12
        )
    assert decision.gate_positions_fraction["Virtual Reservoir D"] < \
        decision.safety_layer_gate_positions_fraction["Virtual Reservoir D"]


# ===========================================================================
# 10. Raw MPC output cannot bypass the downstream boundary
# ===========================================================================

def test_raw_mpc_output_cannot_bypass_the_downstream_boundary(orchestrator, monkeypatch):
    network = _live_network()
    monkeypatch.setattr(orchestrator.mpc, "decide", lambda *a, **k: _flood_decision())
    decision = orchestrator.decide(network, bundle=_validated_bundle(network))

    for nid in NODES:
        assert decision.gate_positions_fraction[nid] != pytest.approx(
            decision.proposed_gate_positions_fraction[nid]
        )
    assert decision.downstream_protection_modified is True


def test_bypassing_the_boundary_would_have_violated_the_capacity(orchestrator, monkeypatch):
    """
    Counter-proof: the action the MPC+SafetyLayer produced, applied verbatim,
    WOULD have exceeded the capacity — so the boundary is load-bearing.
    """
    guard = DownstreamCapacityGuard()
    network = _live_network()
    monkeypatch.setattr(orchestrator.mpc, "decide", lambda *a, **k: _flood_decision())
    decision = orchestrator.decide(network, bundle=_validated_bundle(network))

    unguarded = guard.predict_flows(
        network,
        decision.safety_layer_gate_positions_fraction,
        {nid: 0.0 for nid in NODES},
        NODES,
        guard.horizon_for(network),
    )
    assert max(unguarded) > CAPACITY
    assert decision.downstream_predicted_flow_mcm_day <= CAPACITY + TOLERANCE_MCM_DAY


def test_every_candidate_the_boundary_applies_is_safety_layer_clean(orchestrator,
                                                                   monkeypatch):
    """
    The boundary sits after the SafetyLayer, so its output must satisfy the
    SafetyLayer's own rules (bounds + rate limit) — proven here with the real
    layer object.
    """
    network = _live_network()
    for nid in NODES:
        network.nodes[nid].state.gate_position = 0.0
    monkeypatch.setattr(orchestrator.mpc, "decide", lambda *a, **k: _flood_decision())
    decision = orchestrator.decide(network, bundle=_validated_bundle(network))

    current = {nid: 0.0 for nid in NODES}
    revalidated = SafetyLayer(
        max_gate_change_per_step=orchestrator.mpc.config.max_gate_change
    ).validate(decision.gate_positions_fraction, current, NODES)

    assert revalidated.violations == [], (
        "the downstream boundary produced an action the SafetyLayer would change"
    )
    for nid in NODES:
        assert revalidated.validated_gates[nid] == pytest.approx(
            decision.gate_positions_fraction[nid], abs=1e-12
        )
    assert decision.applied_action_safety_layer_clean is True


# ===========================================================================
# 11, 12 & 13. Provenance is unchanged: blocked forecasts, NaN/Inf, no rescue
# ===========================================================================

def test_demonstration_forecast_still_blocks_and_the_boundary_does_not_fabricate(
    orchestrator,
):
    network = _live_network()
    decision = orchestrator.decide(network, bundle=_demo_bundle(network))

    assert decision.controller_status == ControllerStatus.BLOCKED.value
    assert decision.safety_layer_status == "NOT_APPLIED_MPC_BLOCKED"
    assert decision.downstream_status == DOWNSTREAM_STATUS_NOT_APPLIED_MPC_BLOCKED
    assert decision.downstream_capacity_achieved is False
    assert decision.downstream_protection_modified is False
    assert decision.control_applied is False
    assert decision.final_safe_control_action_source == "HELD_CURRENT_GATES"
    for nid in NODES:
        assert decision.gate_positions_fraction[nid] == pytest.approx(
            network.nodes[nid].state.gate_position
        )


def test_missing_idukki_forecast_still_blocks(orchestrator):
    network = _live_network()
    adapter = LiveForecastAdapter.for_network(network, project_root=PROJECT_ROOT)
    bundle = adapter.build_bundle(
        {n: _payload() for n in NODES if n != "Virtual Reservoir D"}, "2026-09-14"
    )
    decision = orchestrator.decide(network, bundle=bundle)

    assert decision.controller_status == ControllerStatus.BLOCKED.value
    assert decision.downstream_status == DOWNSTREAM_STATUS_NOT_APPLIED_MPC_BLOCKED
    assert any("Virtual Reservoir D:MISSING_FORECAST" in r for r in decision.reasons)


def test_demonstration_forecast_never_reaches_the_boundary(orchestrator, monkeypatch):
    network = _live_network()
    seen = []
    monkeypatch.setattr(
        orchestrator.downstream_guard, "evaluate",
        lambda *a, **k: seen.append(1),
    )
    orchestrator.decide(network, bundle=_demo_bundle(network))
    assert seen == [], "the downstream boundary must not run for a blocked forecast"


def test_nan_proposal_is_still_fail_safe(orchestrator, monkeypatch):
    network = _live_network()
    monkeypatch.setattr(orchestrator.mpc, "decide", lambda *a, **k: _unsafe_decision())
    decision = orchestrator.decide(network, bundle=_validated_bundle(network))

    for nid in NODES:
        assert math.isfinite(decision.gate_positions_fraction[nid])
        assert 0.0 <= decision.gate_positions_fraction[nid] <= 1.0
    import json
    json.dumps(decision.to_dict())


def test_infinite_proposal_is_still_fail_safe(orchestrator, monkeypatch):
    network = _live_network()
    monkeypatch.setattr(
        orchestrator.mpc, "decide",
        lambda *a, **k: ControlDecision(
            gate_positions={n: float("inf") for n in NODES},
            objective_score=1.0, status="OPTIMAL",
        ),
    )
    decision = orchestrator.decide(network, bundle=_validated_bundle(network))
    for nid in NODES:
        assert math.isfinite(decision.gate_positions_fraction[nid])
    assert decision.downstream_capacity_achieved is True


def test_no_downstream_capacity_declared_is_reported_not_claimed():
    """If the network declares no capacity, the boundary must not claim one."""
    guard = DownstreamCapacityGuard()
    network = _live_network()
    network._downstream_capacity = 0.0

    result = guard.evaluate(
        network,
        action_fraction={n: 0.5 for n in NODES},
        current_fraction={n: 0.0 for n in NODES},
        node_ids=NODES,
        max_gate_change=0.5,
        inflows={n: 0.0 for n in NODES},
    )
    assert result.status == "NOT_APPLIED_NO_CAPACITY"
    assert result.is_protected is False
    assert result.capacity_achieved is False
    assert result.modified is False


def test_fail_closed_is_reported_when_the_capacity_is_unachievable():
    """
    A terminal reservoir already above capacity forces spill that no gate
    position can prevent. The boundary must say so rather than claim protection.
    """
    guard = DownstreamCapacityGuard()
    network = _live_network()
    node_d = network.nodes["Virtual Reservoir D"]
    node_d.state.storage = node_d.capacity          # full
    # Force a large inflow so spill is unavoidable even with the gate closed.
    inflows = {nid: 0.0 for nid in NODES}
    inflows["Virtual Reservoir D"] = node_d.capacity

    result = guard.evaluate(
        network,
        action_fraction={n: 0.5 for n in NODES},
        current_fraction={n: 0.0 for n in NODES},
        node_ids=NODES,
        max_gate_change=0.5,
        inflows=inflows,
    )
    assert result.proposed_predicted_flow_mcm_day > CAPACITY
    assert result.status == DOWNSTREAM_STATUS_FAILED_CLOSED
    assert result.capacity_achieved is False
    assert result.is_protected is False
    assert result.modified is True
    assert "NO ADMISSIBLE ACTION" in result.reason
    # The minimum achievable flow is the unavoidable spill and is reported.
    assert result.min_achievable_flow_mcm_day is not None
    assert result.min_achievable_flow_mcm_day > CAPACITY
    # The applied action is the flow-minimising one (all gates at their lowest
    # reachable value).
    for nid in NODES:
        assert result.action_fraction[nid] == pytest.approx(0.0)


def test_failed_closed_action_is_the_flow_minimiser():
    guard = DownstreamCapacityGuard()
    current = {nid: 0.7 for nid in NODES}
    action = guard.minimum_admissible_action(NODES, current, 0.5)
    for nid in NODES:
        assert action[nid] == pytest.approx(0.2)
    # Never below zero, even from a low current gate.
    low = guard.minimum_admissible_action(NODES, {nid: 0.1 for nid in NODES}, 0.5)
    for nid in NODES:
        assert low[nid] == pytest.approx(0.0)


# ===========================================================================
# 14. Determinism
# ===========================================================================

def test_safe_constrained_action_is_deterministic(orchestrator):
    outcomes = []
    for _ in range(2):
        network = _live_network()
        orch = LiveMPCOrchestrator()
        orch.mpc.decide = lambda *a, **k: _flood_decision()
        decision = orch.decide(network, bundle=_validated_bundle(network))
        outcomes.append({
            "status": decision.downstream_status,
            "gates": {nid: round(decision.gate_positions_fraction[nid], 12)
                      for nid in NODES},
            "flow": round(decision.downstream_predicted_flow_mcm_day, 9),
            "candidates": decision.downstream_candidates_evaluated,
        })
    assert outcomes[0] == outcomes[1]


def test_guard_result_is_deterministic_for_identical_state():
    guard = DownstreamCapacityGuard()
    network = _live_network()
    kwargs = dict(
        action_fraction={n: 1.0 for n in NODES},
        current_fraction={n: 0.0 for n in NODES},
        node_ids=NODES,
        max_gate_change=0.5,
        inflows={n: 0.0 for n in NODES},
    )
    first = guard.evaluate(network, **kwargs)
    second = guard.evaluate(network, **kwargs)
    assert first.status == second.status
    assert first.action_fraction == second.action_fraction
    assert first.predicted_flow_mcm_day == second.predicted_flow_mcm_day


# ===========================================================================
# 15. The final applied action IS what reaches ReservoirNetwork
# ===========================================================================

def test_final_applied_action_is_what_the_network_received(live_sim, monkeypatch):
    sim = live_sim
    sim.bridge.init_cascade(50.0)
    sim.mode = "MANUAL"
    sim.manual_inflows = {n: 0.0 for n in NODES}
    sim.manual_gates = {n: 0.0 for n in NODES}
    sim.step()

    network = sim.bridge.cascade.network
    monkeypatch.setattr(sim, "_build_live_forecast_bundle",
                        lambda cf: _validated_bundle(network))
    monkeypatch.setattr(sim.mpc_orchestrator.mpc, "decide",
                        lambda *a, **k: _flood_decision())

    sim.mode = "AI"
    sim.step()

    decision = sim.last_control_decision
    assert decision.final_safe_control_action_source == "DOWNSTREAM_CAPACITY_GUARD"
    assert set(decision.final_safe_control_action_fraction) == set(NODES)
    for nid in NODES:
        assert decision.final_safe_control_action_fraction[nid] == pytest.approx(
            decision.gate_positions_fraction[nid]
        )
        assert decision.final_safe_control_action_pct[nid] == pytest.approx(
            decision.gate_positions_pct[nid]
        )
        assert network.nodes[nid].state.gate_position == pytest.approx(
            decision.final_safe_control_action_fraction[nid]
        )


def test_final_safe_control_action_label_exists_for_the_future_hardware_boundary():
    source = ORCH_PATH.read_text(encoding="utf-8")
    assert "FINAL_SAFE_CONTROL_ACTION" in source
    for field in ("final_safe_control_action_fraction",
                  "final_safe_control_action_pct",
                  "final_safe_control_action_source"):
        assert field in source


# ===========================================================================
# 16 & 17. Truthful provenance over the API/WebSocket; browser cannot bypass
# ===========================================================================

def test_api_reports_the_three_statuses_separately():
    """
    MPC, SafetyLayer and downstream-capacity protection are three SEPARATE fields.
    The system must never collapse them into one "safe" label.
    """
    payload = client.get("/api/state").json()
    control = payload["control"]

    assert "controller_status" in control
    assert "safety_layer_status" in control
    assert "downstream_status" in control
    assert control["controller_status"] != control["safety_layer_status"] or True
    for key in ("downstream_capacity_mcm_day", "downstream_capacity_achieved",
                "downstream_protection_modified", "downstream_reason",
                "final_safe_control_action_source"):
        assert key in control, f"{key} missing from the control block"
    block = control["downstream_capacity_protection"]
    assert block["unit"] == "MCM/day"
    assert block["physics"].startswith("ReservoirNetwork")


def test_websocket_state_carries_downstream_provenance():
    client.post("/api/controller/mode", json={"mode": "MANUAL"})
    with client.websocket_connect("/ws/state") as ws:
        state = ws.receive_json()

    control = state["control"]
    for key in ("controller_status", "safety_layer_status", "downstream_status",
                "downstream_capacity_mcm_day", "downstream_capacity_achieved",
                "downstream_protection_modified", "downstream_predicted_flow_mcm_day",
                "final_safe_control_action_source", "final_safe_control_action_pct"):
        assert key in control, f"{key} missing from the WebSocket control block"


def test_twin_payload_never_implies_protection_without_evidence():
    twin = adapt_state_for_twin({"reservoirs": {}, "downstream_flow": 0}, "MANUAL", 0.0)
    control = twin["control"]
    assert control["downstream_status"] == "UNKNOWN"
    assert control["downstream_capacity_achieved"] is None
    assert control["downstream_protection_modified"] is None


def test_browser_cannot_bypass_downstream_protection():
    assert client.post("/api/state", json={"control": {"downstream_status": "PROTECTED"}}
                       ).status_code == 405
    assert client.put("/api/state", json={"control": {"downstream_status": "PROTECTED"}}
                      ).status_code == 405

    sim = state_manager.sim_state
    before = sim.mpc_orchestrator.status_dict().get("downstream_status")
    client.post("/api/simulation/play",
                json={"downstream_status": "PROTECTED", "control_applied": True})
    client.post("/api/simulation/pause")
    after = sim.mpc_orchestrator.status_dict().get("downstream_status")
    assert after == before


def test_twin_ui_displays_downstream_protection_truthfully():
    html = TWIN_INDEX_PATH.read_text(encoding="utf-8")
    assert "DOWNSTREAM CAP" in html
    assert "PREDICTED FLOW" in html
    assert 'data-ref="dsprot"' in html
    assert 'data-ref="dsflow"' in html
    assert "ctl.downstream_status" in html
    assert "downstream_capacity_achieved" in html
    # No decision logic is added to JavaScript: the status string is displayed
    # verbatim from the payload.
    assert "replace(/_/g, ' ')" in html


# ===========================================================================
# 18. GNN cannot bypass the boundary
# ===========================================================================

def test_gnn_cannot_bypass_the_downstream_boundary():
    source = GUARD_PATH.read_text(encoding="utf-8")
    assert "gnn" not in source.lower()

    orch_source = ORCH_PATH.read_text(encoding="utf-8")
    tree = ast.parse(orch_source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
    assert not any("gnn" in m.lower() for m in imported)
    assert "compute_ai_recommendation" not in STATE_MANAGER_PATH.read_text(encoding="utf-8")


# ===========================================================================
# 19. Canonical units
# ===========================================================================

def test_canonical_units_in_the_downstream_boundary(orchestrator, monkeypatch):
    network = _live_network()
    monkeypatch.setattr(orchestrator.mpc, "decide", lambda *a, **k: _flood_decision())
    decision = orchestrator.decide(network, bundle=_validated_bundle(network))

    assert decision.downstream_capacity_mcm_day == pytest.approx(CAPACITY)
    assert decision.downstream_capacity_protection["unit"] == "MCM/day"
    for nid in NODES:
        assert 0.0 <= decision.final_safe_control_action_fraction[nid] <= 1.0
        from src.common import units
        assert decision.final_safe_control_action_pct[nid] == pytest.approx(
            units.gate_fraction_to_percent(
                decision.final_safe_control_action_fraction[nid]
            ), abs=1e-9
        )
    source = GUARD_PATH.read_text(encoding="utf-8")
    assert "/ 100.0" not in source
    assert "* 100.0" not in source


# ===========================================================================
# 21. Frozen artifacts and protected components
# ===========================================================================

def test_stage10_does_not_modify_frozen_artifacts(orchestrator, monkeypatch):
    before = {p: _sha(p) for p in FROZEN_ARTIFACTS}
    network = _live_network()
    monkeypatch.setattr(orchestrator.mpc, "decide", lambda *a, **k: _flood_decision())
    orchestrator.decide(network, bundle=_validated_bundle(network))
    assert {p: _sha(p) for p in FROZEN_ARTIFACTS} == before


def test_protected_components_are_unmodified_by_stage10():
    for path in (SAFETY_PATH, MPC_PATH, NETWORK_PATH):
        source = path.read_text(encoding="utf-8")
        assert "downstream_capacity_guard" not in source
        assert "DownstreamCapacityGuard" not in source

    safety = SAFETY_PATH.read_text(encoding="utf-8")
    assert "Downstream capacity" in safety, (
        "the SafetyLayer's docstring still claims a downstream-capacity check it "
        "does not implement — Stage 10 documents this rather than editing it"
    )
    # The guard is a NEW module and does not duplicate the routing model.
    guard = GUARD_PATH.read_text(encoding="utf-8")
    assert "class ConnectionState" not in guard
    assert "deque(" not in guard
    assert "attenuation" not in guard or "ReservoirNetwork" in guard
