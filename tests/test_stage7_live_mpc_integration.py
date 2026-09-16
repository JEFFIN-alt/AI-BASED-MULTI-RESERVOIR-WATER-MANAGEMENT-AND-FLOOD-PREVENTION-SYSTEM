"""
Stage 7 — Live MPC integration tests.

Proves the validated Phase 15.3 MPC is the ONE authoritative live controller,
and that it can be driven ONLY by a forecast whose provenance says
``validated_metrics_apply == True``.
"""

import ast
import hashlib
import math
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.common import units  # noqa: E402
from src.controller.live_mpc_orchestrator import (  # noqa: E402
    SAFETY_LAYER_STATUS,
    ControllerStatus,
    LiveMPCOrchestrator,
)
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

MPC_PATH = _PROJECT_ROOT / "src" / "controller" / "mpc_controller.py"
SAFETY_PATH = _PROJECT_ROOT / "src" / "controller" / "safety.py"
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
             validated=True, simulated=False,
             v1=2.0, v3=2.1, v7=2.2, unit=FORECAST_UNIT):
    return {
        "forecast_1d": v1, "forecast_3d": v3, "forecast_7d": v7,
        "forecast_status": status,
        "forecast_provenance": provenance,
        "forecast_source": "FROZEN_LSTM_V3",
        "is_simulated": simulated,
        "validated_metrics_apply": validated,
        "forecast_unit": unit,
        "horizons": list(HORIZON_KEYS),
        "input_provenance": {
            "measured_historical": ["inflow", "water_level", "live_storage",
                                    "rainfall", "total_outflow"] if validated else [],
            "synthetic_demo": [] if validated else ["water_level", "rainfall"],
            "unavailable": [], "simulated": [] if validated else ["inflow"],
        },
    }


def _demo_payload():
    return _payload(status="DEMONSTRATION_ONLY",
                    provenance="SIMULATION_OR_SYNTHETIC_INPUTS_FROZEN_MODEL",
                    validated=False, simulated=True)


def _live_network():
    bridge = SimBridge(str(LIVE_CONFIG), str(THRESH_PATH))
    return bridge.cascade.network


def _adapter(network):
    return LiveForecastAdapter.for_network(network, project_root=PROJECT_ROOT)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def orchestrator():
    return LiveMPCOrchestrator()


# ===========================================================================
# 1. Validated forecast reaches the MPC
# ===========================================================================

def test_validated_forecast_reaches_the_mpc(orchestrator):
    network = _live_network()
    adapter = _adapter(network)
    bundle = adapter.build_bundle({n: _payload() for n in LIVE_NODE_IDS}, "2026-09-14")

    decision = orchestrator.decide(network, bundle=bundle)

    assert decision.controller_type == "MPC"
    assert decision.controller_status == ControllerStatus.ACTIVE.value
    assert decision.forecast_control_eligible is True
    assert decision.control_applied is True
    assert decision.mpc_status in ("OPTIMAL", "CORRECTED", "SAFE")
    assert decision.candidates_evaluated > 0
    assert decision.mpc_forecast_used is True


def test_orchestrator_output_equals_a_direct_mpc_call(orchestrator):
    """
    The orchestrator must not rewrite, tune or post-process the MPC decision —
    only bound it to [0,1] and convert units.
    """
    network = _live_network()
    adapter = _adapter(network)
    bundle = adapter.build_bundle({n: _payload() for n in LIVE_NODE_IDS}, "2026-09-14")

    direct = orchestrator.mpc.decide(network, forecast_snapshot=bundle.snapshot)
    through = orchestrator.decide(network, bundle=bundle)

    for nid in LIVE_NODE_IDS:
        assert through.gate_positions_fraction[nid] == pytest.approx(
            float(direct.gate_positions[nid]), abs=1e-12
        )
    assert through.mpc_status == direct.status
    assert through.mpc_safety_status == direct.safety_status


# ===========================================================================
# 2 & 3. Demonstration / non-validated forecasts are BLOCKED
# ===========================================================================

def test_demonstration_only_forecast_blocks_the_mpc(orchestrator):
    network = _live_network()
    adapter = _adapter(network)
    bundle = adapter.build_bundle({n: _demo_payload() for n in LIVE_NODE_IDS}, "2026-09-14")

    decision = orchestrator.decide(network, bundle=bundle)

    assert decision.controller_status == ControllerStatus.BLOCKED.value
    assert decision.forecast_control_eligible is False
    assert decision.control_applied is False
    assert decision.mpc_status == "NOT_INVOKED", "the MPC must not be called at all"
    assert decision.blocked_reason == "FORECAST_NOT_ELIGIBLE_FOR_CONTROL"
    assert any("NOT_VALIDATED_METRICS" in r for r in decision.reasons)
    assert any("DEMONSTRATION_ONLY" in r for r in decision.reasons)


def test_validated_metrics_apply_false_blocks_the_mpc(orchestrator):
    network = _live_network()
    adapter = _adapter(network)
    payload = _payload(status="VALIDATED", provenance="REAL_MEASUREMENT_INPUTS_FROZEN_MODEL",
                       validated=False, simulated=False)
    bundle = adapter.build_bundle({n: payload for n in LIVE_NODE_IDS}, "2026-09-14")

    decision = orchestrator.decide(network, bundle=bundle)

    assert decision.controller_status == ControllerStatus.BLOCKED.value
    assert decision.mpc_status == "NOT_INVOKED"
    assert any("NOT_VALIDATED_METRICS" in r for r in decision.reasons)


def test_blocked_decision_holds_current_gates_without_fabricating(orchestrator):
    """A blocked gate holds REAL current gates and says so explicitly."""
    network = _live_network()
    adapter = _adapter(network)
    bundle = adapter.build_bundle({n: _demo_payload() for n in LIVE_NODE_IDS}, "2026-09-14")

    decision = orchestrator.decide(network, bundle=bundle)

    for nid in LIVE_NODE_IDS:
        assert decision.gate_positions_fraction[nid] == pytest.approx(
            float(network.nodes[nid].state.gate_position)
        )
        assert 0.0 <= decision.gate_positions_fraction[nid] <= 1.0


def test_forecast_provenance_is_preserved_on_a_blocked_decision(orchestrator):
    network = _live_network()
    adapter = _adapter(network)
    bundle = adapter.build_bundle({n: _demo_payload() for n in LIVE_NODE_IDS}, "2026-09-14")

    decision = orchestrator.decide(network, bundle=bundle)
    nodes = decision.forecast_provenance["nodes"]

    assert set(nodes.keys()) == set(LIVE_NODE_IDS)
    assert nodes["Virtual Reservoir A"]["declared_status"] == "DEMONSTRATION_ONLY"
    assert nodes["Virtual Reservoir A"]["validated_metrics_apply"] is False
    assert nodes["Virtual Reservoir A"]["is_simulated"] is True


# ===========================================================================
# 4. Missing Idukki forecast is never fabricated
# ===========================================================================

def test_missing_idukki_forecast_blocks_and_is_not_fabricated(orchestrator):
    network = _live_network()
    adapter = _adapter(network)
    # Deliberately omit Reservoir D (Idukki).
    bundle = adapter.build_bundle(
        {n: _payload() for n in LIVE_NODE_IDS if n != "Virtual Reservoir D"}, "2026-09-14"
    )

    forecast_d = bundle.get("Virtual Reservoir D")
    assert forecast_d is not None, "the node must still appear in the snapshot"
    assert forecast_d.target_1d is None
    assert forecast_d.target_3d is None
    assert forecast_d.target_7d is None

    decision = orchestrator.decide(network, bundle=bundle)
    assert decision.controller_status == ControllerStatus.BLOCKED.value
    assert decision.mpc_status == "NOT_INVOKED"
    assert any("Virtual Reservoir D:MISSING_FORECAST" in r for r in decision.reasons)


# ===========================================================================
# 5 & 6. Non-finite and unit-mismatched forecasts are rejected
# ===========================================================================

@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_forecast_is_rejected(orchestrator, bad):
    network = _live_network()
    adapter = _adapter(network)
    payload = _payload()
    payload["forecast_1d"] = bad
    bundle = adapter.build_bundle({n: payload for n in LIVE_NODE_IDS}, "2026-09-14")

    decision = orchestrator.decide(network, bundle=bundle)
    assert decision.controller_status == ControllerStatus.BLOCKED.value
    assert decision.mpc_status == "NOT_INVOKED"
    assert any("HORIZON_UNAVAILABLE" in r or "INVALID_VALUE" in r for r in decision.reasons)


def test_nonfinite_value_in_a_bare_snapshot_is_rejected(orchestrator):
    """Defence in depth: the gate also inspects the snapshot values directly."""
    from src.network_env.v3_forecast_adapter import (
        ForecastStatus, NetworkForecastSnapshot, ReservoirForecast,
    )

    network = _live_network()
    snapshot = NetworkForecastSnapshot(forecast_date="2026-09-14")
    for nid in LIVE_NODE_IDS:
        snapshot.forecasts[nid] = ReservoirForecast(
            reservoir_id=nid, v3_reservoir_name="X", forecast_date="2026-09-14",
            target_1d=float("nan") if nid == "Virtual Reservoir B" else 2.0,
            target_3d=2.1, target_7d=2.2,
            status_1d=ForecastStatus.AVAILABLE, status_3d=ForecastStatus.AVAILABLE,
            status_7d=ForecastStatus.AVAILABLE,
            provenance="status=VALIDATED;source=REAL;is_simulated=False;"
                       "validated_metrics_apply=True;unit=MCM/day;synthetic=none;unavailable=none",
        )
    decision = orchestrator.decide(network, snapshot=snapshot)
    assert decision.controller_status == ControllerStatus.BLOCKED.value
    assert any("INVALID_VALUE" in r for r in decision.reasons)


def test_unit_mismatch_is_rejected(orchestrator):
    network = _live_network()
    adapter = _adapter(network)
    bundle = adapter.build_bundle(
        {n: _payload(unit="m3/s") for n in LIVE_NODE_IDS}, "2026-09-14"
    )

    decision = orchestrator.decide(network, bundle=bundle)
    assert decision.controller_status == ControllerStatus.BLOCKED.value
    assert decision.mpc_status == "NOT_INVOKED"
    assert any("UNIT_MISMATCH" in r for r in decision.reasons)


def test_unit_mismatch_declared_in_provenance_is_rejected(orchestrator):
    network = _live_network()
    adapter = _adapter(network)
    bundle = adapter.build_bundle({n: _payload() for n in LIVE_NODE_IDS}, "2026-09-14")
    # Corrupt the declared unit in the snapshot's provenance string only.
    for fc in bundle.snapshot.forecasts.values():
        fc.provenance = fc.provenance.replace("unit=MCM/day", "unit=m3/s")

    decision = orchestrator.decide(network, snapshot=bundle.snapshot)
    assert decision.controller_status == ControllerStatus.BLOCKED.value
    assert any("UNIT_MISMATCH" in r for r in decision.reasons)


def test_warmup_and_unavailable_statuses_block_the_mpc(orchestrator):
    network = _live_network()
    adapter = _adapter(network)
    for status in ("WARMUP_INSUFFICIENT_HISTORY", "FORECAST_UNAVAILABLE"):
        bundle = adapter.build_bundle(
            {n: _payload(status=status, validated=False) for n in LIVE_NODE_IDS},
            "2026-09-14",
        )
        decision = orchestrator.decide(network, bundle=bundle)
        assert decision.controller_status == ControllerStatus.BLOCKED.value
        assert decision.mpc_status == "NOT_INVOKED"


# ===========================================================================
# 7. MPC actually changes live gate decisions
# ===========================================================================

def test_mpc_actually_changes_live_gate_decisions(monkeypatch):
    """
    End-to-end through the authoritative live simulation: with an ELIGIBLE
    forecast the MPC determines the gates that are actually applied.
    """
    sim = state_manager.sim_state
    previous_mode = sim.mode
    try:
        sim.bridge.init_cascade(50.0)
        sim.mode = "MANUAL"
        sim.manual_inflows = {"Virtual Reservoir A": 3.0, "Virtual Reservoir B": 4.0,
                              "Virtual Reservoir C": 90.0, "Virtual Reservoir D": 0.0}
        # Establish distinctive "wide open" gates first.
        sim.manual_gates = {n: 100.0 for n in LIVE_NODE_IDS}
        sim.step()

        network = sim.bridge.cascade.network
        gates_before = {nid: network.nodes[nid].state.gate_position for nid in LIVE_NODE_IDS}
        assert all(g == pytest.approx(1.0) for g in gates_before.values())

        adapter = _adapter(network)
        bundle = adapter.build_bundle({n: _payload() for n in LIVE_NODE_IDS}, "2026-09-14")
        monkeypatch.setattr(sim, "_build_live_forecast_bundle", lambda cf: bundle)

        sim.mode = "AI"
        sim.step()

        decision = sim.last_control_decision
        assert decision is not None
        assert decision.controller_status == ControllerStatus.ACTIVE.value
        assert decision.control_applied is True

        gates_after = {nid: network.nodes[nid].state.gate_position for nid in LIVE_NODE_IDS}
        assert any(gates_after[nid] != pytest.approx(gates_before[nid]) for nid in LIVE_NODE_IDS), \
            "the MPC decision did not change the applied gates"

        # The applied gates ARE the MPC's gates.
        for nid in LIVE_NODE_IDS:
            assert gates_after[nid] == pytest.approx(decision.gate_positions_fraction[nid])
    finally:
        sim.mode = previous_mode
        sim.manual_gates = {"Virtual Reservoir A": 40.0, "Virtual Reservoir B": 35.0,
                            "Virtual Reservoir C": 50.0, "Virtual Reservoir D": 100.0}
        sim.bridge.init_cascade(50.0)


def test_live_ai_mode_with_demo_forecasts_blocks_and_holds_gates():
    """The default live AI path is BLOCKED because forecasts are demonstration-only."""
    sim = state_manager.sim_state
    previous_mode = sim.mode
    try:
        sim.mode = "AI"
        sim.manual_gates = {"Virtual Reservoir A": 40.0, "Virtual Reservoir B": 35.0,
                            "Virtual Reservoir C": 50.0, "Virtual Reservoir D": 100.0}
        sim.step()
        decision = sim.last_control_decision
        assert decision is not None
        assert decision.controller_status == ControllerStatus.BLOCKED.value
        assert decision.control_applied is False
        assert decision.mpc_status == "NOT_INVOKED"
    finally:
        sim.mode = previous_mode


# ===========================================================================
# 8. The old rule-based advisor is no longer authoritative
# ===========================================================================

def test_legacy_forecast_aware_controller_is_not_called_by_the_live_path():
    source = STATE_MANAGER_PATH.read_text(encoding="utf-8")
    assert "compute_ai_recommendation" not in source, (
        "the live AI path must not use the rule-based ForecastAwareController"
    )


def test_live_ai_step_does_not_use_the_legacy_advisor(monkeypatch):
    sim = state_manager.sim_state
    previous_mode = sim.mode

    def _boom(*args, **kwargs):
        raise AssertionError("ForecastAwareController must not be consulted in AI mode")

    monkeypatch.setattr(sim.bridge, "compute_ai_recommendation", _boom)
    try:
        sim.mode = "AI"
        sim.step()          # must not raise
    finally:
        sim.mode = previous_mode


def test_sim_bridge_advisor_is_marked_non_authoritative():
    source = (_PROJECT_ROOT / "src" / "dashboard" / "sim_bridge.py").read_text(encoding="utf-8")
    assert "NON-AUTHORITATIVE" in source or "non-authoritative" in source


def test_gnn_cannot_make_control_decisions():
    source = (_PROJECT_ROOT / "src" / "controller" / "live_mpc_orchestrator.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
    assert not any("gnn" in m.lower() for m in imported)


# ===========================================================================
# 9. Browser cannot inject authoritative state
# ===========================================================================

def test_no_state_write_endpoint_exists():
    forged = {"reservoirs": {"reservoir_1": {"storage": 9999.0}}, "control": {"controller_status": "ACTIVE"}}
    assert client.post("/api/state", json=forged).status_code == 405
    assert client.put("/api/state", json=forged).status_code == 405


def test_forged_control_block_is_ignored_by_command_endpoints():
    sim = state_manager.sim_state
    forged = {"value": 50.0, "control": {"controller_status": "ACTIVE", "control_applied": True}}
    before = sim.mpc_orchestrator.status_dict().get("controller_status")

    response = client.post("/api/gate/reservoir_1", json=forged)
    assert response.status_code == 200

    after = sim.mpc_orchestrator.status_dict().get("controller_status")
    assert after == before, "a client payload must not change controller provenance"


# ===========================================================================
# 10. All four reservoirs reach the MPC state
# ===========================================================================

def test_all_four_reservoirs_reach_the_mpc_state(orchestrator):
    network = _live_network()
    adapter = _adapter(network)
    bundle = adapter.build_bundle({n: _payload() for n in LIVE_NODE_IDS}, "2026-09-14")

    decision = orchestrator.decide(network, bundle=bundle)

    assert set(decision.per_node.keys()) == set(LIVE_NODE_IDS)
    assert set(decision.gate_positions_fraction.keys()) == set(LIVE_NODE_IDS)
    assert set(decision.gate_positions_pct.keys()) == set(LIVE_NODE_IDS)
    # Reservoir D (Idukki) is a first-class member, not pinned or excluded.
    assert decision.per_node["Virtual Reservoir D"]["controller"] in (
        "MPC_FORECAST_AWARE", "MPC_NO_FORECAST"
    )


# ===========================================================================
# 11. Canonical units
# ===========================================================================

def test_canonical_units_are_preserved(orchestrator):
    network = _live_network()
    adapter = _adapter(network)
    bundle = adapter.build_bundle({n: _payload() for n in LIVE_NODE_IDS}, "2026-09-14")

    decision = orchestrator.decide(network, bundle=bundle)

    for nid in LIVE_NODE_IDS:
        fraction = decision.gate_positions_fraction[nid]
        percent = decision.gate_positions_pct[nid]
        assert 0.0 <= fraction <= 1.0, "internal gate must be a FRACTION"
        assert percent == pytest.approx(units.gate_fraction_to_percent(fraction), abs=1e-9)
        assert 0.0 <= percent <= 100.0
    assert "percent" in decision.to_dict()["gate_unit"]


def test_no_new_implicit_unit_conversions_are_introduced():
    source = (_PROJECT_ROOT / "src" / "controller" / "live_mpc_orchestrator.py").read_text(encoding="utf-8")
    # The single boundary module must be the ONLY place conversion happens.
    assert "/ 100.0" not in source
    assert "* 100.0" not in source
    assert "units.gate_fraction_to_percent" in source


# ===========================================================================
# 12. Controller provenance reaches the WebSocket / API / UI
# ===========================================================================

def test_control_provenance_is_in_the_api_state():
    payload = client.get("/api/state").json()
    assert "control" in payload
    control = payload["control"]
    for key in ("controller_type", "controller_status", "forecast_control_eligible",
                "safety_layer_status", "control_applied"):
        assert key in control
    assert control["controller_type"] == "MPC"
    # Stage 8: the SafetyLayer boundary is now IN the live path.
    assert control["safety_layer_integrated"] is True
    assert control["safety_layer_version"] == "PHASE_15_3_VALIDATED"


def test_control_provenance_is_in_the_twin_payload():
    bridge = SimBridge(str(LIVE_CONFIG), str(THRESH_PATH))
    twin = adapt_state_for_twin(bridge.get_state({}), "MANUAL", 0.0)
    assert "control" in twin
    assert twin["control"]["safety_layer_status"] in (
        "UNKNOWN", "SAFE", "CORRECTED", "EMERGENCY",
        "NOT_APPLIED_MPC_BLOCKED",
    )


def test_twin_payload_never_labels_an_unknown_controller_as_active_mpc():
    twin = adapt_state_for_twin({"reservoirs": {}, "downstream_flow": 0}, "MANUAL", 0.0)
    assert twin["control"]["controller_status"] == "UNKNOWN"
    assert twin["control"]["control_applied"] is False


def test_twin_ui_displays_the_controller_state():
    html = TWIN_INDEX_PATH.read_text(encoding="utf-8")
    assert "data-ref=\"ctlstat\"" in html
    assert "data-ref=\"safety\"" in html
    assert "MPC FORECAST" in html
    assert "SAFETY LAYER" in html
    assert "NOT_INTEGRATED_STAGE_7" in html or "safety_layer_status" in html


# ===========================================================================
# 13. Frozen artifacts untouched
# ===========================================================================

def test_mpc_integration_does_not_modify_frozen_artifacts(orchestrator):
    before = {p: _sha(p) for p in FROZEN_ARTIFACTS}
    network = _live_network()
    adapter = _adapter(network)
    bundle = adapter.build_bundle({n: _payload() for n in LIVE_NODE_IDS}, "2026-09-14")
    orchestrator.decide(network, bundle=bundle)
    after = {p: _sha(p) for p in FROZEN_ARTIFACTS}
    assert before == after


def test_validated_mpc_and_safety_files_are_unmodified():
    mpc = MPC_PATH.read_text(encoding="utf-8")
    assert "class MPCController" in mpc
    assert "class ControlDecision" in mpc
    assert "def decide(" in mpc
    assert "live_mpc_orchestrator" not in mpc
    assert "LiveMPCOrchestrator" not in mpc

    safety = SAFETY_PATH.read_text(encoding="utf-8")
    assert "class SafetyLayer" in safety
    assert "live_mpc_orchestrator" not in safety


def test_reservoir_network_and_v3_adapter_do_not_import_the_orchestrator():
    for rel in ("src/network_env/reservoir_network.py",
                "src/network_env/v3_forecast_adapter.py"):
        assert "live_mpc_orchestrator" not in (_PROJECT_ROOT / rel).read_text(encoding="utf-8")


# ===========================================================================
# 14. Stage 5 decision conditions still hold
# ===========================================================================

def test_demonstration_badge_and_strict_mode_still_present():
    html = TWIN_INDEX_PATH.read_text(encoding="utf-8")
    assert "DEMONSTRATION — MODEL INPUTS SIMULATED" in html
    sm = STATE_MANAGER_PATH.read_text(encoding="utf-8")
    assert "AQUAFLOW_ALLOW_SYNTHETIC_DEMO_INPUTS" in sm


def test_stage3_topology_is_unchanged():
    network = _live_network()
    assert network.processing_order == LIVE_NODE_IDS
    delays = {(c.source, c.destination): c.delay for c in network.connections}
    atten = {(c.source, c.destination): c.attenuation for c in network.connections}
    assert delays[("Virtual Reservoir A", "Virtual Reservoir B")] == 2
    assert delays[("Virtual Reservoir B", "Virtual Reservoir C")] == 1
    assert delays[("Virtual Reservoir C", "Virtual Reservoir D")] == 1
    assert atten[("Virtual Reservoir A", "Virtual Reservoir B")] == pytest.approx(0.90)
    assert atten[("Virtual Reservoir B", "Virtual Reservoir C")] == pytest.approx(0.85)
    assert atten[("Virtual Reservoir C", "Virtual Reservoir D")] == pytest.approx(0.80)


# ===========================================================================
# 15. A validated snapshot passes through without changing values
# ===========================================================================

def test_validated_snapshot_values_pass_through_unchanged(orchestrator):
    network = _live_network()
    adapter = _adapter(network)
    values = (0.0, 1e-9, 2.436246156692505, 91.71626281738281)
    payload = _payload(v1=values[0], v3=values[1], v7=values[2])
    bundle = adapter.build_bundle({n: payload for n in LIVE_NODE_IDS}, "2026-09-14")

    snapshot = bundle.snapshot
    for nid in LIVE_NODE_IDS:
        fc = snapshot.get(nid)
        assert fc.target_1d == values[0]
        assert fc.target_3d == values[1]
        assert fc.target_7d == values[2]

    decision = orchestrator.decide(network, bundle=bundle)
    # The gate did not alter the snapshot it was given.
    for nid in LIVE_NODE_IDS:
        fc = bundle.snapshot.get(nid)
        assert fc.target_1d == values[0]
        assert fc.target_3d == values[1]
        assert fc.target_7d == values[2]
    assert decision.forecast_control_eligible is True


# ===========================================================================
# Determinism / behavioural integration
# ===========================================================================

def test_same_state_and_validated_forecast_gives_the_same_mpc_decision():
    """
    Behavioural proof: identical authoritative state + identical validated
    forecast ⇒ identical, deterministic control decision.
    """
    adapter = _adapter(_live_network())
    bundle_template = adapter.build_bundle(
        {n: _payload() for n in LIVE_NODE_IDS}, "2026-09-14"
    )

    gates = []
    for _ in range(2):
        network = _live_network()          # identical initial state
        orch = LiveMPCOrchestrator()       # fresh, identical controller
        adapter2 = _adapter(network)
        bundle2 = adapter2.build_bundle({n: _payload() for n in LIVE_NODE_IDS}, "2026-09-14")
        decision = orch.decide(network, bundle=bundle2)
        assert decision.controller_status == ControllerStatus.ACTIVE.value
        gates.append({nid: round(decision.gate_positions_fraction[nid], 12) for nid in LIVE_NODE_IDS})

    assert gates[0] == gates[1]
    assert set(bundle_template.snapshot.forecasts.keys()) == set(LIVE_NODE_IDS)


def test_eligibility_is_fail_closed_without_any_snapshot(orchestrator):
    network = _live_network()
    decision = orchestrator.decide(network, snapshot=None)
    assert decision.controller_status == ControllerStatus.BLOCKED.value
    assert decision.mpc_status == "NOT_INVOKED"
    assert "NO_FORECAST_SNAPSHOT" in decision.reasons


def test_status_dict_before_any_decision_is_explicit(orchestrator):
    status = orchestrator.status_dict()
    assert status["controller_status"] == ControllerStatus.UNAVAILABLE.value
    assert status["control_applied"] is False
    assert status["blocked_reason"] == "NO_DECISION_YET"
    # Stage 8: the layer is integrated; before any decision it simply has not run.
    assert status["safety_layer_integrated"] is True
    assert status["safety_layer_status"] == SAFETY_LAYER_STATUS


def test_safety_layer_boundary_is_reported_on_an_active_decision(orchestrator):
    """Stage 8 — the SafetyLayer ran and its real status is reported."""
    network = _live_network()
    adapter = _adapter(network)
    bundle = adapter.build_bundle({n: _payload() for n in LIVE_NODE_IDS}, "2026-09-14")
    decision = orchestrator.decide(network, bundle=bundle)
    assert decision.safety_layer_integrated is True
    assert decision.safety_layer_status in ("SAFE", "CORRECTED", "EMERGENCY", "FALLBACK")
    assert decision.to_dict()["safety_layer_version"] == "PHASE_15_3_VALIDATED"


def test_forecast_values_are_finite_through_the_whole_path(orchestrator):
    network = _live_network()
    adapter = _adapter(network)
    bundle = adapter.build_bundle({n: _payload() for n in LIVE_NODE_IDS}, "2026-09-14")
    decision = orchestrator.decide(network, bundle=bundle)
    for nid in LIVE_NODE_IDS:
        assert math.isfinite(decision.gate_positions_fraction[nid])
        assert math.isfinite(decision.gate_positions_pct[nid])
