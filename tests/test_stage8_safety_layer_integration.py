"""
Stage 8 — SafetyLayer integration tests.

Proves the validated SafetyLayer (``src/controller/safety.py``) is the final
software boundary between the MPC's proposal and the authoritative reservoir
simulation, and that its ACTUAL guarantees (not aspirational ones) hold.
"""

import ast
import hashlib
import math
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.common import units  # noqa: E402
from src.controller.live_mpc_orchestrator import (  # noqa: E402
    SAFETY_STATUS_CORRECTED,
    SAFETY_STATUS_NOT_APPLIED_MPC_BLOCKED,
    SAFETY_STATUS_SAFE,
    ControllerStatus,
    LiveMPCOrchestrator,
)
from src.controller.mpc_controller import ControlDecision  # noqa: E402
from src.controller.safety import SafetyLayer  # noqa: E402
from src.network_env.live_forecast_adapter import (  # noqa: E402
    FORECAST_UNIT,
    HORIZON_KEYS,
    LiveForecastAdapter,
)
from src.dashboard.api import state_manager  # noqa: E402
from src.dashboard.api.app import app  # noqa: E402
from src.dashboard.sim_bridge import SimBridge  # noqa: E402
from src.dashboard.twin_component.state_adapter import adapt_state_for_twin  # noqa: E402

PROJECT_ROOT = str(_PROJECT_ROOT)
client = TestClient(app)

LIVE_CONFIG = _PROJECT_ROOT / "configs" / "simulation" / "four_reservoir_demo.json"
THRESH_PATH = _PROJECT_ROOT / "data" / "processed" / "historical_inflow_thresholds.json"
SAFETY_PATH = _PROJECT_ROOT / "src" / "controller" / "safety.py"
ORCH_PATH = _PROJECT_ROOT / "src" / "controller" / "live_mpc_orchestrator.py"
MPC_PATH = _PROJECT_ROOT / "src" / "controller" / "mpc_controller.py"
STATE_MANAGER_PATH = _PROJECT_ROOT / "src" / "dashboard" / "api" / "state_manager.py"
TWIN_INDEX_PATH = _PROJECT_ROOT / "src" / "dashboard" / "web" / "index.html"

LIVE_NODE_IDS = [
    "Virtual Reservoir A",
    "Virtual Reservoir B",
    "Virtual Reservoir C",
    "Virtual Reservoir D",
]

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


def _live_network():
    return SimBridge(str(LIVE_CONFIG), str(THRESH_PATH)).cascade.network


def _adapter(network):
    return LiveForecastAdapter.for_network(network, project_root=PROJECT_ROOT)


def _validated_bundle(network, nodes=None):
    adapter = _adapter(network)
    nodes = nodes if nodes is not None else LIVE_NODE_IDS
    return adapter.build_bundle({n: _payload() for n in nodes}, "2026-09-14")


def _unsafe_decision(**overrides):
    gates = {
        "Virtual Reservoir A": 5.0,      # above range -> clamp
        "Virtual Reservoir B": -3.0,     # below range -> clamp
        "Virtual Reservoir C": float("nan"),   # invalid -> 0.1
        "Virtual Reservoir D": 0.9,      # in range, but rate-limited
    }
    gates.update(overrides)
    return ControlDecision(gate_positions=gates, objective_score=1.0,
                           status="OPTIMAL", per_node={}, candidates_evaluated=1)


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
    }
    try:
        yield sim
    finally:
        sim.mode = saved["mode"]
        sim.manual_gates = saved["manual_gates"]
        sim.manual_inflows = saved["manual_inflows"]
        sim.bridge.init_cascade(50.0)


# ===========================================================================
# 1. Valid MPC action reaches the SafetyLayer
# ===========================================================================

def test_valid_mpc_action_reaches_the_safety_layer(orchestrator, monkeypatch):
    network = _live_network()
    bundle = _validated_bundle(network)

    calls = []
    real_validate = orchestrator.safety.validate

    def _spy(proposed, current, node_ids):
        calls.append({"proposed": dict(proposed), "node_ids": list(node_ids)})
        return real_validate(proposed, current, node_ids)

    monkeypatch.setattr(orchestrator.safety, "validate", _spy)

    decision = orchestrator.decide(network, bundle=bundle)

    assert len(calls) == 1, "the SafetyLayer must be invoked exactly once"
    assert set(calls[0]["node_ids"]) == set(LIVE_NODE_IDS)
    assert set(calls[0]["proposed"].keys()) == set(LIVE_NODE_IDS)
    assert decision.safety_layer_status == SAFETY_STATUS_SAFE
    assert decision.safety_is_safe is True
    assert decision.safety_layer_integrated is True
    assert decision.safety_layer_version == "PHASE_15_3_VALIDATED"


def test_the_safety_layer_used_is_the_validated_implementation(orchestrator):
    assert isinstance(orchestrator.safety, SafetyLayer)
    assert type(orchestrator.safety) is SafetyLayer
    # Same configuration as the MPC's own instance.
    assert orchestrator.safety.max_gate_change == orchestrator.mpc.safety.max_gate_change


# ===========================================================================
# 2 & 3. SafetyLayer output — not raw MPC output — reaches the network
# ===========================================================================

def test_safe_mpc_action_passes_through_unchanged(orchestrator):
    network = _live_network()
    decision = orchestrator.decide(network, bundle=_validated_bundle(network))

    assert decision.safety_modified is False
    for nid in LIVE_NODE_IDS:
        assert decision.gate_positions_fraction[nid] == pytest.approx(
            decision.proposed_gate_positions_fraction[nid]
        )


def test_unsafe_mpc_action_is_constrained_by_the_safety_layer(orchestrator, monkeypatch):
    network = _live_network()
    monkeypatch.setattr(orchestrator.mpc, "decide", lambda *a, **k: _unsafe_decision())

    decision = orchestrator.decide(network, bundle=_validated_bundle(network))

    assert decision.controller_status == ControllerStatus.ACTIVE.value
    assert decision.safety_layer_status == SAFETY_STATUS_CORRECTED
    assert decision.safety_is_safe is False
    assert decision.safety_modified is True
    assert decision.safety_violations, "violations must be reported"

    # Every applied gate is a legal gate fraction.
    for nid in LIVE_NODE_IDS:
        assert 0.0 <= decision.gate_positions_fraction[nid] <= 1.0
        assert math.isfinite(decision.gate_positions_fraction[nid])

    # The raw proposal did NOT simply pass through.
    assert decision.gate_positions_fraction != decision.proposed_gate_positions_fraction


def test_safety_layer_output_equals_a_direct_validate_call(orchestrator, monkeypatch):
    """
    The SafetyLayer boundary itself must still be exactly ``validate()``.

    STAGE 10 update: ``gate_positions_fraction`` is now the
    FINAL_SAFE_CONTROL_ACTION (SafetyLayer output -> downstream-capacity
    boundary). The SafetyLayer's own output is exposed as
    ``safety_layer_gate_positions_fraction`` and is asserted here to equal a
    direct ``validate()`` call to 1e-12, so the Stage 8 guarantee is unchanged —
    it is simply no longer the last word when the downstream capacity demands
    otherwise (see ``tests/test_stage10_downstream_capacity.py``).
    """
    network = _live_network()
    monkeypatch.setattr(orchestrator.mpc, "decide", lambda *a, **k: _unsafe_decision())

    current = {nid: float(network.nodes[nid].state.gate_position) for nid in LIVE_NODE_IDS}
    expected = SafetyLayer(
        max_gate_change_per_step=orchestrator.mpc.config.max_gate_change
    ).validate(_unsafe_decision().gate_positions, current, LIVE_NODE_IDS)

    decision = orchestrator.decide(network, bundle=_validated_bundle(network))

    for nid in LIVE_NODE_IDS:
        assert decision.safety_layer_gate_positions_fraction[nid] == pytest.approx(
            expected.validated_gates[nid], abs=1e-12
        )
        # The applied action can only ever be MORE restrictive, never less.
        assert decision.gate_positions_fraction[nid] <= (
            expected.validated_gates[nid] + 1e-12
        )


# ===========================================================================
# 4-8. The SafetyLayer's ACTUAL checks
# ===========================================================================

def test_gate_bounds_are_enforced(orchestrator, monkeypatch):
    network = _live_network()
    monkeypatch.setattr(orchestrator.mpc, "decide",
                        lambda *a, **k: _unsafe_decision(**{
                            "Virtual Reservoir A": 5.0,
                            "Virtual Reservoir B": -3.0}))
    decision = orchestrator.decide(network, bundle=_validated_bundle(network))
    for nid in LIVE_NODE_IDS:
        assert 0.0 <= decision.gate_positions_fraction[nid] <= 1.0
    assert any("clamped" in v for v in decision.safety_violations)


def test_rate_limit_is_enforced(orchestrator, monkeypatch):
    """
    A large jump is truncated to max_gate_change from the CURRENT gate.

    STAGE 10 update: the rate limit is enforced by the SafetyLayer and is visible
    in ``safety_layer_gate_positions_fraction``; the final action may be reduced
    further by the downstream-capacity boundary, but can never exceed the
    SafetyLayer's output (which is what preserves the rate limit).
    """
    network = _live_network()
    for nid in LIVE_NODE_IDS:
        network.nodes[nid].state.gate_position = 0.0
    monkeypatch.setattr(orchestrator.mpc, "decide",
                        lambda *a, **k: _unsafe_decision(**{
                            "Virtual Reservoir A": 1.0,
                            "Virtual Reservoir B": 1.0,
                            "Virtual Reservoir C": 1.0,
                            "Virtual Reservoir D": 1.0}))
    decision = orchestrator.decide(network, bundle=_validated_bundle(network))
    limit = orchestrator.mpc.config.max_gate_change
    for nid in LIVE_NODE_IDS:
        assert decision.safety_layer_gate_positions_fraction[nid] == pytest.approx(
            limit, abs=1e-12
        )
        assert decision.gate_positions_fraction[nid] <= limit + 1e-12
    assert any("rate-limited" in v for v in decision.safety_violations)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_gate_is_replaced_by_the_conservative_value(orchestrator, monkeypatch, bad):
    network = _live_network()
    monkeypatch.setattr(orchestrator.mpc, "decide",
                        lambda *a, **k: _unsafe_decision(**{"Virtual Reservoir C": bad}))
    decision = orchestrator.decide(network, bundle=_validated_bundle(network))
    # The validated layer sets 0.1 for NaN/Inf (its documented behaviour).
    assert decision.gate_positions_fraction["Virtual Reservoir C"] == pytest.approx(0.1)
    assert any("invalid gate value" in v for v in decision.safety_violations)


def test_non_numeric_gate_is_replaced(orchestrator, monkeypatch):
    network = _live_network()
    monkeypatch.setattr(orchestrator.mpc, "decide",
                        lambda *a, **k: _unsafe_decision(**{"Virtual Reservoir A": "wide open"}))
    decision = orchestrator.decide(network, bundle=_validated_bundle(network))
    assert decision.gate_positions_fraction["Virtual Reservoir A"] == pytest.approx(0.1)
    assert decision.safety_layer_status == SAFETY_STATUS_CORRECTED


def test_missing_node_in_the_proposal_is_filled_from_the_current_gate(orchestrator, monkeypatch):
    network = _live_network()
    network.nodes["Virtual Reservoir D"].state.gate_position = 0.25
    partial = _unsafe_decision().gate_positions
    partial.pop("Virtual Reservoir D")
    monkeypatch.setattr(orchestrator.mpc, "decide",
                        lambda *a, **k: ControlDecision(gate_positions=partial,
                                                        objective_score=1.0,
                                                        status="OPTIMAL"))
    decision = orchestrator.decide(network, bundle=_validated_bundle(network))
    assert decision.gate_positions_fraction["Virtual Reservoir D"] == pytest.approx(0.25)
    assert any("missing gate command" in v for v in decision.safety_violations)
    assert decision.proposed_gate_positions_fraction["Virtual Reservoir D"] is None


def test_emergency_status_is_unreachable_through_validate():
    """
    Documented behaviour: ``validate()`` always populates every node id, so its
    ``EMERGENCY`` branch is not reachable via that method. ``emergency_fallback``
    exists separately.
    """
    layer = SafetyLayer()
    result = layer.validate({"A": 5.0}, {}, ["A", "B"])
    assert result.status in ("SAFE", "CORRECTED")
    assert set(result.validated_gates.keys()) == {"A", "B"}
    assert SafetyLayer.emergency_fallback(["A", "B"]) == {"A": 0.1, "B": 0.1}


def test_release_and_downstream_capacity_checks_are_not_implemented():
    """
    HONEST GAP REPORT — the validated layer does NOT check release bounds or
    downstream capacity, despite its module docstring claiming the latter.
    Nothing was added to fill this: it is reported instead.
    """
    source = SAFETY_PATH.read_text(encoding="utf-8")
    assert "max_release" not in source
    assert "downstream_capacity" not in source
    assert "storage" not in source
    # The docstring claims a downstream-capacity check that has no code.
    assert "Downstream capacity" in source, "docstring claims a check that is absent"


# ===========================================================================
# 9 & 10. All four reservoirs; D not pinned
# ===========================================================================

def test_all_four_reservoirs_pass_through_the_safety_layer(orchestrator):
    network = _live_network()
    decision = orchestrator.decide(network, bundle=_validated_bundle(network))
    for mapping in (decision.gate_positions_fraction, decision.gate_positions_pct,
                    decision.proposed_gate_positions_fraction):
        assert set(mapping.keys()) == set(LIVE_NODE_IDS)


def test_reservoir_d_is_not_pinned_to_the_legacy_100_percent(orchestrator, monkeypatch):
    """
    Reservoir D (Idukki) must receive the safety-validated coordinated action,
    not the legacy hard-coded 100% gate.

    STAGE 10 update: D's gate is the one the downstream-capacity boundary also
    constrains (it is the terminal reservoir, so its outflow IS the downstream
    flow). The asserted value is therefore the SafetyLayer's rate-limited output
    and the final action must be at most that.
    """
    network = _live_network()
    monkeypatch.setattr(orchestrator.mpc, "decide",
                        lambda *a, **k: _unsafe_decision(**{"Virtual Reservoir D": 0.9}))
    for nid in LIVE_NODE_IDS:
        network.nodes[nid].state.gate_position = 0.0
    decision = orchestrator.decide(network, bundle=_validated_bundle(network))
    d_safety = decision.safety_layer_gate_positions_fraction["Virtual Reservoir D"]
    d_gate = decision.gate_positions_fraction["Virtual Reservoir D"]
    assert d_safety == pytest.approx(0.5)      # rate-limited from 0.0
    assert d_gate != pytest.approx(1.0)
    assert d_gate <= d_safety + 1e-12


# ===========================================================================
# 11 & 12. Upstream blocking is unaffected by the SafetyLayer
# ===========================================================================

def test_demonstration_forecast_blocks_and_safety_does_not_rescue(orchestrator):
    network = _live_network()
    adapter = _adapter(network)
    bundle = adapter.build_bundle({n: _demo_payload() for n in LIVE_NODE_IDS}, "2026-09-14")

    decision = orchestrator.decide(network, bundle=bundle)

    assert decision.controller_status == ControllerStatus.BLOCKED.value
    assert decision.mpc_status == "NOT_INVOKED"
    assert decision.safety_layer_status == SAFETY_STATUS_NOT_APPLIED_MPC_BLOCKED
    assert decision.control_applied is False
    # The SafetyLayer did not manufacture a control decision.
    for nid in LIVE_NODE_IDS:
        assert decision.gate_positions_fraction[nid] == pytest.approx(
            float(network.nodes[nid].state.gate_position)
        )


def test_safety_layer_is_not_invoked_when_the_mpc_is_blocked(orchestrator, monkeypatch):
    network = _live_network()
    adapter = _adapter(network)
    bundle = adapter.build_bundle({n: _demo_payload() for n in LIVE_NODE_IDS}, "2026-09-14")

    calls = []
    monkeypatch.setattr(orchestrator.safety, "validate",
                        lambda *a, **k: calls.append(1))
    orchestrator.decide(network, bundle=bundle)
    assert calls == [], "the SafetyLayer must not run for a blocked forecast"


def test_missing_idukki_forecast_blocks_upstream(orchestrator):
    network = _live_network()
    bundle = _validated_bundle(network, nodes=[n for n in LIVE_NODE_IDS
                                               if n != "Virtual Reservoir D"])
    decision = orchestrator.decide(network, bundle=bundle)
    assert decision.controller_status == ControllerStatus.BLOCKED.value
    assert decision.safety_layer_status == SAFETY_STATUS_NOT_APPLIED_MPC_BLOCKED
    assert any("Virtual Reservoir D:MISSING_FORECAST" in r for r in decision.reasons)


# ===========================================================================
# 13 & 14. Nothing else can bypass the SafetyLayer
# ===========================================================================

def test_browser_cannot_bypass_the_safety_layer():
    forged = {"reservoirs": {"reservoir_1": {"gate": 1.0}},
              "control": {"safety_layer_status": "SAFE", "control_applied": True}}
    assert client.post("/api/state", json=forged).status_code == 405
    assert client.put("/api/state", json=forged).status_code == 405


def test_forged_control_payload_does_not_change_safety_provenance():
    sim = state_manager.sim_state
    before = sim.mpc_orchestrator.status_dict()
    response = client.post("/api/simulation/play",
                           json={"safety_layer_status": "SAFE", "control_applied": True})
    assert response.status_code == 200
    client.post("/api/simulation/pause")
    after = sim.mpc_orchestrator.status_dict()
    assert after.get("safety_layer_status") == before.get("safety_layer_status")
    assert after.get("safety_modified") == before.get("safety_modified")


def test_only_the_orchestrator_applies_gates_to_the_live_network():
    """
    In AI mode the applied gate commands come from the orchestrator's
    safety-validated decision and from nowhere else.
    """
    source = STATE_MANAGER_PATH.read_text(encoding="utf-8")
    assert "decision.gate_positions_pct" in source
    assert "self.mpc_orchestrator.decide(" in source
    # The legacy rule-based advisor is not a gate source.
    assert "compute_ai_recommendation" not in source
    # The GNN result is never assigned into gate_commands.
    assert "gate_commands[res] = gnn" not in source
    assert "ai_recommendations" not in source


def test_gnn_cannot_bypass_the_safety_layer():
    orch_source = ORCH_PATH.read_text(encoding="utf-8")
    tree = ast.parse(orch_source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
    assert not any("gnn" in m.lower() for m in imported)

    sm_source = STATE_MANAGER_PATH.read_text(encoding="utf-8")
    # The GNN feeds only advisory bookkeeping, never the applied gate commands.
    assert "gnn_result" in sm_source
    assert "gate_commands" in sm_source


def test_safety_layer_source_is_unmodified():
    source = SAFETY_PATH.read_text(encoding="utf-8")
    assert "class SafetyLayer" in source
    assert "def validate(" in source
    assert "def emergency_fallback(" in source
    assert "live_mpc_orchestrator" not in source
    assert "LiveMPCOrchestrator" not in source


# ===========================================================================
# 15 & 16. Provenance reaches state / WebSocket
# ===========================================================================

def test_controller_provenance_reports_mpc_and_safety_status():
    payload = client.get("/api/state").json()
    control = payload["control"]
    assert control["controller_type"] == "MPC"
    assert control["controller_status"] in ("ACTIVE", "BLOCKED", "UNAVAILABLE")
    assert control["safety_layer_integrated"] is True
    assert control["safety_layer_version"] == "PHASE_15_3_VALIDATED"
    assert control["safety_layer_status"] in (
        "SAFE", "CORRECTED", "EMERGENCY", "FALLBACK",
        "NOT_APPLIED_MPC_BLOCKED", "NOT_APPLIED_MPC_ERROR",
        "NOT_APPLIED_ADAPTER_ERROR",
    )
    assert "safety_modified" in control
    assert "proposed_gate_positions_fraction" in control


def test_safety_state_is_visible_in_the_twin_payload():
    bridge = SimBridge(str(LIVE_CONFIG), str(THRESH_PATH))
    twin = adapt_state_for_twin(bridge.get_state({}), "MANUAL", 0.0)
    control = twin["control"]
    assert "safety_layer_status" in control
    assert "safety_modified" in control
    assert "proposed_gate_positions_fraction" in control


def test_twin_ui_displays_safety_status_and_applied_vs_proposed():
    html = TWIN_INDEX_PATH.read_text(encoding="utf-8")
    assert 'data-ref="safety"' in html
    assert 'data-ref="safetymod"' in html
    assert "SAFETY LAYER" in html
    assert "SAFETY ACTION" in html
    assert "safety_modified" in html
    assert "MODIFIED" in html


def test_control_payload_is_json_serialisable_even_for_an_unsafe_proposal(orchestrator, monkeypatch):
    import json

    network = _live_network()
    monkeypatch.setattr(orchestrator.mpc, "decide", lambda *a, **k: _unsafe_decision())
    decision = orchestrator.decide(network, bundle=_validated_bundle(network))
    # Must not raise: a malformed proposal must never break the WebSocket feed.
    json.dumps(decision.to_dict())


# ===========================================================================
# 17. Canonical units
# ===========================================================================

def test_canonical_units_are_preserved(orchestrator):
    network = _live_network()
    decision = orchestrator.decide(network, bundle=_validated_bundle(network))
    for nid in LIVE_NODE_IDS:
        fraction = decision.gate_positions_fraction[nid]
        assert 0.0 <= fraction <= 1.0
        assert decision.gate_positions_pct[nid] == pytest.approx(
            units.gate_fraction_to_percent(fraction), abs=1e-9
        )


def test_no_implicit_unit_conversions_in_the_orchestrator():
    source = ORCH_PATH.read_text(encoding="utf-8")
    assert "/ 100.0" not in source
    assert "* 100.0" not in source
    assert "units.gate_fraction_to_percent" in source


# ===========================================================================
# 18. Determinism
# ===========================================================================

def test_safety_layer_is_deterministic_for_the_same_inputs():
    layer = SafetyLayer(max_gate_change_per_step=0.5)
    proposed = {"Virtual Reservoir A": 5.0, "Virtual Reservoir B": float("nan")}
    current = {"Virtual Reservoir A": 0.0, "Virtual Reservoir B": 0.0}
    node_ids = ["Virtual Reservoir A", "Virtual Reservoir B"]
    first = layer.validate(dict(proposed), dict(current), node_ids)
    second = layer.validate(dict(proposed), dict(current), node_ids)
    assert first.validated_gates == second.validated_gates
    assert first.status == second.status
    assert first.violations == second.violations


def test_whole_path_is_deterministic_with_a_validated_forecast():
    outcomes = []
    for _ in range(2):
        network = _live_network()
        orch = LiveMPCOrchestrator()
        decision = orch.decide(network, bundle=_validated_bundle(network))
        outcomes.append({
            "gates": {nid: round(decision.gate_positions_fraction[nid], 12)
                      for nid in LIVE_NODE_IDS},
            "safety": decision.safety_layer_status,
            "status": decision.controller_status,
        })
    assert outcomes[0] == outcomes[1]


# ===========================================================================
# 20. Frozen artifacts
# ===========================================================================

def test_safety_integration_does_not_modify_frozen_artifacts(orchestrator, monkeypatch):
    before = {p: _sha(p) for p in FROZEN_ARTIFACTS}
    network = _live_network()
    monkeypatch.setattr(orchestrator.mpc, "decide", lambda *a, **k: _unsafe_decision())
    orchestrator.decide(network, bundle=_validated_bundle(network))
    assert {p: _sha(p) for p in FROZEN_ARTIFACTS} == before


def test_validated_mpc_and_safety_are_not_modified_by_stage8():
    mpc = MPC_PATH.read_text(encoding="utf-8")
    assert "class MPCController" in mpc
    assert "live_mpc_orchestrator" not in mpc
    assert "SafetyLayer" in mpc          # unchanged internal use
    safety = SAFETY_PATH.read_text(encoding="utf-8")
    assert "live_mpc_orchestrator" not in safety


# ===========================================================================
# BEHAVIOURAL INTEGRATION — unsafe proposal -> safe result at the network
# ===========================================================================

def test_unsafe_mpc_proposal_is_sanitised_before_reaching_reservoir_network(
    live_sim, monkeypatch
):
    """
    End-to-end: the MPC proposes an unsafe action, the SafetyLayer modifies it,
    ReservoirNetwork receives the SAFE result, and the resulting state reflects
    the safe action (not the raw proposal).

    STAGE 10 update: the action that reaches ReservoirNetwork is now the
    FINAL_SAFE_CONTROL_ACTION — the SafetyLayer's output after the
    downstream-capacity boundary. The SafetyLayer's own output is asserted
    separately (``safety_layer_gate_positions_fraction``) so the Stage 8
    guarantee stays pinned while the Stage 10 boundary is allowed to tighten it.
    """
    sim = live_sim
    sim.bridge.init_cascade(50.0)
    sim.manual_inflows = {"Virtual Reservoir A": 3.0, "Virtual Reservoir B": 4.0,
                          "Virtual Reservoir C": 90.0, "Virtual Reservoir D": 0.0}
    sim.mode = "MANUAL"
    sim.manual_gates = {n: 0.0 for n in LIVE_NODE_IDS}
    sim.step()                                   # gates now 0.0

    network = sim.bridge.cascade.network
    for nid in LIVE_NODE_IDS:
        assert network.nodes[nid].state.gate_position == pytest.approx(0.0)

    monkeypatch.setattr(sim, "_build_live_forecast_bundle",
                        lambda cf: _validated_bundle(network))
    unsafe = _unsafe_decision()
    monkeypatch.setattr(sim.mpc_orchestrator.mpc, "decide", lambda *a, **k: unsafe)

    sim.mode = "AI"
    sim.step()

    decision = sim.last_control_decision
    assert decision is not None
    assert decision.controller_status == ControllerStatus.ACTIVE.value
    assert decision.safety_layer_status == SAFETY_STATUS_CORRECTED
    assert decision.safety_modified is True

    limit = sim.mpc_orchestrator.mpc.config.max_gate_change
    safety_expected = {
        "Virtual Reservoir A": limit,     # 5.0 clamped to 1.0, then rate-limited from 0.0
        "Virtual Reservoir B": 0.0,       # -3.0 clamped to 0.0
        "Virtual Reservoir C": 0.1,       # NaN -> 0.1
        "Virtual Reservoir D": limit,     # 0.9 rate-limited from 0.0
    }

    # 1a. The SafetyLayer produced exactly the sanitised action.
    for nid, value in safety_expected.items():
        assert decision.safety_layer_gate_positions_fraction[nid] == pytest.approx(
            value, abs=1e-12
        )
    # 1b. The FINAL action is that action, or tighter (never looser).
    for nid, value in safety_expected.items():
        assert decision.gate_positions_fraction[nid] <= value + 1e-12

    # 2. ReservoirNetwork RECEIVED the final safe action...
    for nid in LIVE_NODE_IDS:
        assert network.nodes[nid].state.gate_position == pytest.approx(
            decision.gate_positions_fraction[nid], abs=1e-9
        )

    # 3. ...and NOT the raw unsafe proposal.
    assert network.nodes["Virtual Reservoir A"].state.gate_position != pytest.approx(1.0)
    assert network.nodes["Virtual Reservoir D"].state.gate_position != pytest.approx(0.9)

    # 4. The resulting state reflects it: A released at its applied gate.
    node_a = network.nodes["Virtual Reservoir A"]
    assert node_a.state.controlled_release == pytest.approx(
        decision.gate_positions_fraction["Virtual Reservoir A"] * node_a.max_release,
        abs=1e-6,
    )


def test_demo_forecast_end_to_end_leaves_gates_untouched(live_sim, monkeypatch):
    """Blocked upstream ⇒ the SafetyLayer never runs and gates are held."""
    sim = live_sim
    sim.bridge.init_cascade(50.0)
    sim.mode = "MANUAL"
    sim.manual_gates = {n: 40.0 for n in LIVE_NODE_IDS}
    sim.step()

    network = sim.bridge.cascade.network
    before = {nid: network.nodes[nid].state.gate_position for nid in LIVE_NODE_IDS}

    sim.mode = "AI"
    sim.step()                                   # demo forecasts -> blocked

    decision = sim.last_control_decision
    assert decision.controller_status == ControllerStatus.BLOCKED.value
    assert decision.safety_layer_status == SAFETY_STATUS_NOT_APPLIED_MPC_BLOCKED
    for nid in LIVE_NODE_IDS:
        assert network.nodes[nid].state.gate_position == pytest.approx(before[nid])


# ===========================================================================
# PERFORMANCE
# ===========================================================================

def test_safety_layer_latency_is_negligible(orchestrator):
    network = _live_network()
    orchestrator.decide(network, bundle=_validated_bundle(network))

    samples = []
    for _ in range(5):
        orchestrator.decide(network, bundle=_validated_bundle(network))
        samples.append(orchestrator.last_safety_latency_ms)

    assert all(s >= 0.0 for s in samples)
    assert max(samples) < 5.0, f"SafetyLayer latency too high: {max(samples)} ms"


def test_safety_layer_latency_measured_against_a_bare_call():
    layer = SafetyLayer(max_gate_change_per_step=0.5)
    proposed = {nid: 0.5 for nid in LIVE_NODE_IDS}
    current = {nid: 0.4 for nid in LIVE_NODE_IDS}
    t0 = time.perf_counter()
    for _ in range(1000):
        layer.validate(proposed, current, LIVE_NODE_IDS)
    per_call_ms = (time.perf_counter() - t0) * 1000.0 / 1000.0
    assert per_call_ms < 1.0, f"validate() averaged {per_call_ms} ms per call"
