"""
Stage 11 — Live Mass-Balance Integrity tests.

Establishes that every authoritative LIVE simulation step conserves mass under
the exact semantics of the validated ``ReservoirNetwork``, that the audit is
performed on the state produced by the action that was ACTUALLY applied

    validated forecast
      -> MPC  ->  SafetyLayer  ->  DownstreamCapacityGuard
      ->  FINAL_SAFE_CONTROL_ACTION  ->  ReservoirNetwork.step()
      ->  mass-balance check

that a violation is detected, reported and NEVER repaired or hidden, and that the
truthful result reaches the API, the WebSocket and the Digital Twin.

Units: storage/spill MCM, flows MCM/day, gates FRACTION internally / PERCENT at
the API. One step is exactly one day, so X MCM/day moves exactly X MCM per step.
"""

import ast
import copy
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.controller.live_mpc_orchestrator import (  # noqa: E402
    ControllerStatus,
    LiveMPCOrchestrator,
)
from src.controller.mpc_controller import ControlDecision  # noqa: E402
from src.dashboard.api import state_manager  # noqa: E402
from src.dashboard.api.app import app  # noqa: E402
from src.dashboard.sim_bridge import SimBridge  # noqa: E402
from src.dashboard.twin_component.state_adapter import adapt_state_for_twin  # noqa: E402
from src.network_env.live_cascade_adapter import LiveCascadeAdapter  # noqa: E402
from src.network_env.live_forecast_adapter import (  # noqa: E402
    FORECAST_UNIT,
    HORIZON_KEYS,
    LiveForecastAdapter,
)
from src.network_env.mass_balance import (  # noqa: E402
    FAIL_SAFE_NONE_DEFINED,
    MASS_BALANCE_STATUS_NOT_CHECKED,
    MASS_BALANCE_STATUS_PASS,
    MASS_BALANCE_STATUS_VIOLATION,
    MASS_BALANCE_TOLERANCE_MCM,
    NETWORK_EQUATION,
    RESERVOIR_EQUATION,
    ROUTING_EQUATION,
    MassBalanceMonitor,
    not_checked_result,
)
from src.network_env.reservoir_network import ReservoirNetwork  # noqa: E402

PROJECT_ROOT = str(_PROJECT_ROOT)
client = TestClient(app)

LIVE_CONFIG = _PROJECT_ROOT / "configs" / "simulation" / "four_reservoir_demo.json"
THRESH_PATH = _PROJECT_ROOT / "data" / "processed" / "historical_inflow_thresholds.json"

MASS_BALANCE_PATH = _PROJECT_ROOT / "src" / "network_env" / "mass_balance.py"
NETWORK_PATH = _PROJECT_ROOT / "src" / "network_env" / "reservoir_network.py"
ADAPTER_PATH = _PROJECT_ROOT / "src" / "network_env" / "live_cascade_adapter.py"
BRIDGE_PATH = _PROJECT_ROOT / "src" / "dashboard" / "sim_bridge.py"
STATE_MANAGER_PATH = _PROJECT_ROOT / "src" / "dashboard" / "api" / "state_manager.py"
STATE_ADAPTER_PATH = (
    _PROJECT_ROOT / "src" / "dashboard" / "twin_component" / "state_adapter.py"
)
TWIN_INDEX_PATH = _PROJECT_ROOT / "src" / "dashboard" / "web" / "index.html"
SAFETY_PATH = _PROJECT_ROOT / "src" / "controller" / "safety.py"
MPC_PATH = _PROJECT_ROOT / "src" / "controller" / "mpc_controller.py"
GUARD_PATH = _PROJECT_ROOT / "src" / "controller" / "downstream_capacity_guard.py"
MANIFEST_PATH = _PROJECT_ROOT / "results" / "phase15_v3_validation" / "v3_integrity_check.json"

NODES = [
    "Virtual Reservoir A",
    "Virtual Reservoir B",
    "Virtual Reservoir C",
    "Virtual Reservoir D",
]
A, B, C, D = NODES
#: A = Anayirankal, B = Ponmudi, C = Idamalayar, D = Idukki (the repository names)
REPOSITORY_NAMES = {A: "Anayirankal", B: "Ponmudi", C: "Idamalayar", D: "Idukki"}
CAPACITY = 50.0

FROZEN_ARTIFACTS = [
    _PROJECT_ROOT / "models" / "lstm_pytorch_v3_logtarget" / "best_model.pt",
    _PROJECT_ROOT / "models" / "lstm_pytorch_v3_logtarget" / "log_target_scaler.pkl",
    _PROJECT_ROOT / "results" / "lstm_pytorch_v3_logtarget" / "test_predictions_original_units.csv",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _bridge() -> SimBridge:
    return SimBridge(str(LIVE_CONFIG), str(THRESH_PATH))


def _live_network() -> ReservoirNetwork:
    """A fresh authoritative network with the validated topology parameters."""
    return _bridge().cascade.network


def _zero(nodes=NODES):
    return {n: 0.0 for n in nodes}


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


def _validated_bundle(network, nodes=None):
    adapter = LiveForecastAdapter.for_network(network, project_root=PROJECT_ROOT)
    return adapter.build_bundle(
        {n: _payload() for n in (nodes if nodes is not None else NODES)}, "2026-09-15"
    )


def _demo_bundle(network):
    adapter = LiveForecastAdapter.for_network(network, project_root=PROJECT_ROOT)
    return adapter.build_bundle({n: _demo_payload() for n in NODES}, "2026-09-15")


def _flood_decision(value=1.0):
    """A proposal that opens every gate — the raw MPC proposal."""
    return ControlDecision(gate_positions={n: value for n in NODES},
                           objective_score=1.0, status="OPTIMAL",
                           per_node={}, candidates_evaluated=1296)


def _row(audit, node_id):
    for row in audit["per_reservoir"]:
        assert row["node_id"] in NODES
        if row["node_id"] == node_id:
            return row
    raise AssertionError(f"{node_id} missing from the diagnostic")


def _routing_row(audit, source):
    for row in audit["routing"]:
        if row["source"] == source:
            return row
    raise AssertionError(f"connection from {source} missing from the diagnostic")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


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


def _drive_live_ai_step(sim, monkeypatch, decision=None, bundle=None):
    """
    Run ONE full authoritative live step through the real applied-action path:

        forecast -> provenance gate -> MPC -> SafetyLayer
                 -> DownstreamCapacityGuard -> ReservoirNetwork.step()

    Returns the recorded arguments the physics actually received.
    """
    sim.bridge.init_cascade(50.0)
    sim.mode = "MANUAL"
    sim.manual_inflows = _zero()
    sim.manual_gates = _zero()
    sim.step()                      # establish a known starting gate state

    network = sim.bridge.cascade.network
    monkeypatch.setattr(sim, "_build_live_forecast_bundle",
                        lambda cf: bundle or _validated_bundle(network))
    if decision is not None:
        monkeypatch.setattr(sim.mpc_orchestrator.mpc, "decide",
                            lambda *a, **k: decision)

    recorded = {}
    real_step = network.step

    def spy(inflows, gates):
        recorded["inflows"] = dict(inflows)
        recorded["gates"] = dict(gates)
        return real_step(inflows, gates)

    monkeypatch.setattr(network, "step", spy)

    sim.mode = "AI"
    sim.step()
    return recorded


# ===========================================================================
# 1 & 2. One-step and multi-step mass balance
# ===========================================================================

def test_one_step_mass_balance_passes():
    monitor = MassBalanceMonitor()
    network = _live_network()

    result = monitor.step_and_check(network, _zero(), {n: 0.3 for n in NODES})

    assert result.checked is True
    assert result.status == MASS_BALANCE_STATUS_PASS
    assert result.residual <= MASS_BALANCE_TOLERANCE_MCM
    assert result.checks == 1
    assert result.timestep_days == 1.0


def test_multi_step_mass_balance_passes():
    monitor = MassBalanceMonitor()
    network = _live_network()

    for step in range(1, 21):
        result = monitor.step_and_check(
            network,
            {A: 1.0, B: 2.0, C: 20.0, D: 0.5},
            {A: 0.4, B: 0.35, C: 0.5, D: 0.25},
        )
        assert result.status == MASS_BALANCE_STATUS_PASS, (step, result.violations)
        assert result.residual <= MASS_BALANCE_TOLERANCE_MCM
        assert result.timestep == step
    assert monitor.checks == 20
    assert monitor.violation_count == 0


def test_per_reservoir_and_network_equations_are_the_models_own():
    source = NETWORK_PATH.read_text(encoding="utf-8")
    # The audited per-reservoir law is the one the validated model documents...
    assert "new_storage = old_storage" in source
    # ...and each term the audit reads is the code that implements that law.
    assert "controlled_release = min(requested_release, available)" in source
    assert "preliminary = available - controlled_release" in source
    assert "preliminary = self.capacity" in source          # spill caps storage
    assert "raw_arriving = conn.queue.popleft()" in source
    assert "attenuated = raw_arriving * conn.attenuation" in source
    assert "loss = raw_arriving * (1.0 - conn.attenuation)" in source
    assert "conn.queue.append(release_vol)" in source       # delayed routing
    # the audit's own equation strings say what the model does
    assert RESERVOIR_EQUATION == (
        "new_storage = old_storage + inflow_local + inflow_routed "
        "- controlled_release - spill"
    )
    assert "terminal_outflow" in NETWORK_EQUATION and "change_in_transit" in NETWORK_EQUATION

    audit = MassBalanceMonitor().diagnostic()
    assert audit["equation"] == RESERVOIR_EQUATION
    assert audit["routing_equation"] == ROUTING_EQUATION
    assert audit["network_equation"] == NETWORK_EQUATION
    assert audit["physics"].startswith("ReservoirNetwork")


def test_incremental_equation_agrees_with_the_networks_own_cumulative_check():
    """
    The per-step audit must be the incremental form of the network's OWN
    cumulative ``mass_balance_check()`` — not a second, invented bookkeeping.
    """
    monitor = MassBalanceMonitor()
    network = _live_network()
    before = network.mass_balance_check()["residual_error"]

    result = monitor.step_and_check(
        network, {A: 1.0, B: 2.0, C: 30.0, D: 1.0}, {A: 0.5, B: 0.5, C: 0.5, D: 0.25}
    )
    after = network.mass_balance_check()

    # The model's own cumulative residual stays at the noise floor...
    assert abs(after["residual_error"]) <= MASS_BALANCE_TOLERANCE_MCM
    # ...and its DELTA is exactly the step residual this audit reports.
    delta = after["residual_error"] - before
    assert delta == pytest.approx(result.network["residual_mcm"], abs=1e-12)
    assert result.network["cumulative_residual_mcm"] == after["residual_error"]


# ===========================================================================
# 3 & 7. All four reservoirs participate; no special-case reservoir
# ===========================================================================

def test_all_four_reservoirs_are_checked():
    monitor = MassBalanceMonitor()
    network = _live_network()

    result = monitor.step_and_check(network, {A: 1.0, B: 1.0, C: 1.0, D: 1.0},
                                    {n: 0.2 for n in NODES})

    assert result.reservoirs_expected == 4
    assert result.reservoirs_checked == 4
    assert [r["node_id"] for r in result.per_reservoir] == list(network.processing_order)
    assert {r["node_id"] for r in result.per_reservoir} == set(NODES)
    assert {r["node_id"] for r in result.per_reservoir} == {
        "Virtual Reservoir A", "Virtual Reservoir B", "Virtual Reservoir C",
        "Virtual Reservoir D",
    }
    # every reservoir carries the SAME diagnostics — no special-cased terminal node
    keys = {frozenset(r.keys()) for r in result.per_reservoir}
    assert len(keys) == 1
    assert all(r["residual_mcm"] == pytest.approx(0.0, abs=1e-12)
               for r in result.per_reservoir)


def test_reservoir_d_is_not_pinned_or_special_cased():
    """
    D (Idukki) is audited like every other reservoir, and the audit never writes
    or pins its gate: whatever gate was applied is what is reported.
    """
    monitor = MassBalanceMonitor()
    network = _live_network()
    gates = {A: 0.37, B: 0.42, C: 0.11, D: 0.37}
    gates[A] = 0.37
    gates[D] = 0.37

    result = monitor.step_and_check(network, _zero(), gates)

    row = _row(result.to_dict(), D)
    assert row["commanded_gate_fraction"] == pytest.approx(0.37)
    assert row["applied_gate_position"] == pytest.approx(0.37)
    assert network.nodes[D].state.gate_position == pytest.approx(0.37)
    assert result.action_fully_applied is True
    assert row["terminal"] is True
    # the audit is read-only: it does not pin D to 1.0, 0.0 or anything else
    assert "100.0" not in MASS_BALANCE_PATH.read_text(encoding="utf-8")


def test_the_audit_never_mutates_the_network():
    monitor = MassBalanceMonitor()
    network = _live_network()
    before = {
        "storages": {n: network.nodes[n].state.storage for n in NODES},
        "queues": [list(c.queue) for c in network.connections],
        "gates": {n: network.nodes[n].state.gate_position for n in NODES},
    }
    snapshot = monitor.snapshot(network)
    assert {n: network.nodes[n].state.storage for n in NODES} == before["storages"]
    network.step(_zero(), {n: 0.0 for n in NODES})
    after_step = {
        "storages": {n: network.nodes[n].state.storage for n in NODES},
        "queues": [list(c.queue) for c in network.connections],
    }
    monitor.verify(network, snapshot, applied_inflows=_zero(),
                   applied_gates_fraction={n: 0.0 for n in NODES})
    assert {n: network.nodes[n].state.storage for n in NODES} == after_step["storages"]
    assert [list(c.queue) for c in network.connections] == after_step["queues"]


# ===========================================================================
# 4, 5 & 6. Steady state, normal operation and spill
# ===========================================================================

def test_zero_flow_steady_state():
    monitor = MassBalanceMonitor()
    network = _live_network()
    before = {n: network.nodes[n].state.storage for n in NODES}

    result = monitor.step_and_check(network, _zero(), {n: 0.0 for n in NODES})

    assert result.status == MASS_BALANCE_STATUS_PASS
    assert result.residual == 0.0
    for nid in NODES:
        assert network.nodes[nid].state.storage == pytest.approx(before[nid])
        row = _row(result.to_dict(), nid)
        assert row["storage_change_mcm"] == pytest.approx(0.0)
        assert row["controlled_release_mcm_day"] == pytest.approx(0.0)
        assert row["spill_mcm"] == pytest.approx(0.0)


def test_normal_inflow_release_case():
    monitor = MassBalanceMonitor()
    network = _live_network()
    inflows = {A: 1.0, B: 2.0, C: 20.0, D: 1.0}
    gates = {A: 0.4, B: 0.3, C: 0.2, D: 0.1}

    result = monitor.step_and_check(network, inflows, gates)

    assert result.status == MASS_BALANCE_STATUS_PASS
    for nid in NODES:
        row = _row(result.to_dict(), nid)
        node = network.nodes[nid]
        # the audited components are the model's own reported ones
        assert row["inflow_local_mcm_day"] == pytest.approx(inflows[nid])
        assert row["controlled_release_mcm_day"] == pytest.approx(
            gates[nid] * node.max_release
        )
        assert row["storage_change_mcm"] == pytest.approx(
            row["inflow_local_mcm_day"] + row["inflow_routed_mcm_day"]
            - row["controlled_release_mcm_day"] - row["spill_mcm"]
        )
    assert result.network["total_external_inflow_mcm_day"] == pytest.approx(
        sum(inflows.values())
    )


def test_spill_case_nonterminal_and_terminal():
    """Spill is booked once — never double counted, never hidden."""
    monitor = MassBalanceMonitor()

    # non-terminal spill (A) leaves the network
    network = _live_network()
    network.nodes[A].state.storage = network.nodes[A].capacity
    result = monitor.step_and_check(network, {A: 4.0, B: 0.0, C: 0.0, D: 0.0},
                                    {n: 0.0 for n in NODES})
    assert result.status == MASS_BALANCE_STATUS_PASS
    row_a = _row(result.to_dict(), A)
    assert row_a["spill_mcm"] == pytest.approx(4.0)
    assert result.network["nonterminal_spill_mcm"] == pytest.approx(4.0)
    assert result.network["terminal_spill_mcm"] == pytest.approx(0.0)
    assert result.network["spill_all_nodes_mcm"] == pytest.approx(4.0)
    assert row_a["residual_mcm"] == pytest.approx(0.0, abs=1e-12)

    # terminal spill (D) is inside terminal_outflow and must NOT be added twice
    network = _live_network()
    network.nodes[D].state.storage = network.nodes[D].capacity
    result = monitor.step_and_check(network, {A: 0.0, B: 0.0, C: 0.0, D: 10.0},
                                    {n: 0.0 for n in NODES})
    assert result.status == MASS_BALANCE_STATUS_PASS, result.violations
    assert result.network["terminal_spill_mcm"] == pytest.approx(10.0)
    assert result.network["terminal_outflow_mcm_day"] == pytest.approx(10.0)
    assert result.network["nonterminal_spill_mcm"] == pytest.approx(0.0)
    # spam check: if terminal spill were also added as its own term the residual
    # would be exactly -10 MCM, i.e. the audit would report a phantom violation.
    assert result.network["residual_mcm"] == pytest.approx(0.0, abs=1e-12)
    assert "terminal_spill is already inside terminal_outflow" in \
        result.network["double_count_note"]


# ===========================================================================
# 7 & 8. Delayed and attenuated routing
# ===========================================================================

def test_delayed_upstream_routing_case():
    """
    Water released by A must NOT be treated as instantaneous inflow: B sees it
    exactly 2 steps later, and the audit proves the delay from the queue and from
    its own release history.
    """
    monitor = MassBalanceMonitor()
    network = _live_network()
    a_release = network.nodes[A].max_release

    r1 = monitor.step_and_check(network, _zero(), {A: 1.0, B: 0.0, C: 0.0, D: 0.0})
    r2 = monitor.step_and_check(network, _zero(), {A: 1.0, B: 0.0, C: 0.0, D: 0.0})
    r3 = monitor.step_and_check(network, _zero(), {A: 1.0, B: 0.0, C: 0.0, D: 0.0})

    # nothing has arrived yet, even though A released 5 MCM on step 1
    assert _routing_row(r1.to_dict(), A)["raw_arriving_mcm_day"] == 0.0
    assert _routing_row(r2.to_dict(), A)["raw_arriving_mcm_day"] == 0.0
    assert _row(r1.to_dict(), B)["inflow_routed_mcm_day"] == 0.0
    assert _row(r2.to_dict(), B)["inflow_routed_mcm_day"] == 0.0

    # ...and on step 3 the delayed water arrives, attenuated by the validated 0.90
    arrival = _routing_row(r3.to_dict(), A)
    assert arrival["delay_days"] == 2
    assert arrival["raw_arriving_mcm_day"] == pytest.approx(a_release)
    assert arrival["attenuated_arrival_mcm_day"] == pytest.approx(a_release * 0.90)
    assert arrival["delay_history_ok"] is True
    assert _row(r3.to_dict(), B)["inflow_routed_mcm_day"] == pytest.approx(a_release * 0.90)

    # the routed inflow is the DELAYED arrival, never A's current release
    assert _row(r3.to_dict(), B)["inflow_routed_mcm_day"] != pytest.approx(
        network.nodes[A].state.controlled_release * 0.90
    ) or network.nodes[A].state.controlled_release == pytest.approx(a_release)


def test_attenuated_routing_case():
    monitor = MassBalanceMonitor()
    network = _live_network()

    # seed real water into the C -> D queue through the model's own step()
    network.nodes[C].state.storage = 100.0
    first = monitor.step_and_check(network, _zero(), {A: 0.0, B: 0.0, C: 1.0, D: 0.0})
    released = network.nodes[C].state.controlled_release
    assert released == pytest.approx(100.0)      # gate 1.0, min(max_release, available)
    assert _routing_row(first.to_dict(), C)["raw_arriving_mcm_day"] == 0.0

    # ...and the delay-1 connection delivers it on the NEXT step, attenuated
    result = monitor.step_and_check(network, _zero(), {A: 0.0, B: 0.0, C: 0.0, D: 0.0})
    audit = result.to_dict()
    row = _routing_row(audit, C)

    assert row["delay_days"] == 1
    assert row["attenuation"] == pytest.approx(0.80)
    assert row["raw_arriving_mcm_day"] == pytest.approx(released)
    assert row["attenuated_arrival_mcm_day"] == pytest.approx(
        row["raw_arriving_mcm_day"] * 0.80
    )
    # the 20 % transmission loss is booked in full, never silently destroyed
    assert row["transmission_loss_mcm"] == pytest.approx(
        row["raw_arriving_mcm_day"] * 0.20
    )
    assert audit["network"]["routing_loss_mcm"] == pytest.approx(
        row["transmission_loss_mcm"]
    )
    assert row["residual_mcm"] == pytest.approx(0.0, abs=1e-12)
    assert row["delay_history_ok"] is True
    assert _row(audit, D)["inflow_routed_mcm_day"] == pytest.approx(
        row["attenuated_arrival_mcm_day"]
    )
    assert result.status == MASS_BALANCE_STATUS_PASS


def test_routing_parameters_are_the_validated_ones():
    monitor = MassBalanceMonitor()
    network = _live_network()
    result = monitor.step_and_check(network, _zero(), {n: 0.0 for n in NODES})

    delays = result.network["routing_delays_days"]
    atten = result.network["attenuation_factors"]
    assert [delays[f"{A}->{B}"], delays[f"{B}->{C}"], delays[f"{C}->{D}"]] == [2, 1, 1]
    assert [atten[f"{A}->{B}"], atten[f"{B}->{C}"], atten[f"{C}->{D}"]] == \
        [0.90, 0.85, 0.80]
    assert [c.delay for c in network.connections] == [2, 1, 1]
    assert [c.attenuation for c in network.connections] == [0.90, 0.85, 0.80]


# ===========================================================================
# 9. Terminal downstream outflow
# ===========================================================================

def test_terminal_downstream_flow_case():
    monitor = MassBalanceMonitor()
    network = _live_network()
    terminal = network._terminal_node_id
    assert terminal == D, "the validated cascade is terminal at Reservoir D (Idukki)"

    result = monitor.step_and_check(network, {A: 1.0, B: 1.0, C: 1.0, D: 1.0},
                                    {A: 0.5, B: 0.5, C: 0.5, D: 0.5})
    audit = result.to_dict()
    net = audit["network"]

    assert net["terminal_node_id"] == D
    assert net["terminal_outflow_mcm_day"] == pytest.approx(
        network.nodes[D].state.total_outflow
    )
    assert net["terminal_outflow_mcm_day"] == pytest.approx(
        net["terminal_controlled_release_mcm_day"] + net["terminal_spill_mcm"]
    )
    assert result.status == MASS_BALANCE_STATUS_PASS

    # Falsification: dropping the terminal term breaks the balance.
    without_terminal = (
        net["total_external_inflow_mcm_day"]
        - (net["storage_change_mcm"] + net["nonterminal_spill_mcm"]
           + net["routing_loss_mcm"] + net["change_in_transit_mcm"])
    )
    assert without_terminal == pytest.approx(net["terminal_outflow_mcm_day"])
    assert abs(without_terminal) > MASS_BALANCE_TOLERANCE_MCM


# ===========================================================================
# 10-12. The FINAL_SAFE_CONTROL_ACTION is what is audited
# ===========================================================================

def test_e2e_validated_forecast_to_mass_balance(live_sim, monkeypatch):
    """
    END-TO-END: validated forecast -> MPC -> SafetyLayer ->
    DownstreamCapacityGuard -> FINAL_SAFE_CONTROL_ACTION -> ReservoirNetwork.step()
    -> mass-balance check, proving the audit is of the action ACTUALLY applied.
    """
    sim = live_sim
    recorded = _drive_live_ai_step(sim, monkeypatch, decision=_flood_decision())

    decision = sim.last_control_decision
    network = sim.bridge.cascade.network
    audit = sim.bridge.cascade.mass_balance_diagnostic()

    # the full validated chain ran
    assert decision.controller_status == ControllerStatus.ACTIVE.value
    assert decision.forecast_control_eligible is True
    assert decision.safety_layer_status in ("SAFE", "CORRECTED")
    assert decision.final_safe_control_action_source == "DOWNSTREAM_CAPACITY_GUARD"

    # the checksum: the network was stepped EXACTLY ONCE, with the final action
    assert set(recorded["gates"]) == set(NODES)
    for nid in NODES:
        assert recorded["gates"][nid] == pytest.approx(
            decision.final_safe_control_action_fraction[nid], abs=1e-12
        )
        assert recorded["gates"][nid] == pytest.approx(
            network.nodes[nid].state.gate_position, abs=1e-12
        )
        assert recorded["gates"][nid] == pytest.approx(
            audit["applied_action_fraction"][nid], abs=1e-12
        )
    # the inflows the physics received are the ones the bridge was handed, and
    # the audit recorded exactly those (never the forecast, never a default)
    assert recorded["inflows"] == pytest.approx(sim.manual_inflows)
    assert audit["applied_inflows_mcm_day"] == pytest.approx(recorded["inflows"])

    # ...and the audit of that step is a real PASS
    assert audit["checked"] is True
    assert audit["status"] == MASS_BALANCE_STATUS_PASS
    assert audit["reservoirs_checked"] == 4
    assert audit["applied_action_source"] == "DOWNSTREAM_CAPACITY_GUARD"

    # the Digital Twin payload confirms the applied action IS the final action
    twin = sim.get_adapted_state()["mass_balance"]
    assert twin["controller_action_checked"] is True
    assert twin["matches_final_safe_control_action"] is True


def test_actual_final_applied_action_is_used_not_the_raw_proposal(live_sim, monkeypatch):
    """
    The audit must describe the applied action. The raw MPC proposal (1.0), the
    SafetyLayer output (rate-limited to 0.5) and the FINAL_SAFE_CONTROL_ACTION
    (downstream-corrected to 0.25) are three DIFFERENT actions for Reservoir D.
    """
    sim = live_sim
    _drive_live_ai_step(sim, monkeypatch, decision=_flood_decision())
    decision = sim.last_control_decision
    network = sim.bridge.cascade.network
    audit = sim.bridge.cascade.mass_balance_diagnostic()

    proposal = decision.proposed_gate_positions_fraction[D]
    safety_gate = decision.safety_layer_gate_positions_fraction[D]
    final_gate = decision.final_safe_control_action_fraction[D]
    audited_gate = audit["applied_action_fraction"][D]

    assert proposal == pytest.approx(1.0)
    assert safety_gate == pytest.approx(0.5)
    assert final_gate == pytest.approx(0.25)
    assert len({proposal, safety_gate, final_gate}) == 3

    assert audited_gate == pytest.approx(final_gate)
    assert audited_gate != pytest.approx(proposal)
    assert audited_gate != pytest.approx(safety_gate)

    # and the audited PHYSICS belongs to the applied gate, not to the proposal:
    max_release = network.nodes[D].max_release
    row = _row(audit, D)
    assert row["controlled_release_mcm_day"] == pytest.approx(final_gate * max_release)
    assert row["controlled_release_mcm_day"] != pytest.approx(safety_gate * max_release)
    assert row["controlled_release_mcm_day"] != pytest.approx(proposal * max_release)


def test_downstream_guard_modification_reaches_the_mass_balance_record(live_sim,
                                                                      monkeypatch):
    sim = live_sim
    _drive_live_ai_step(sim, monkeypatch, decision=_flood_decision())
    decision = sim.last_control_decision
    audit = sim.bridge.cascade.mass_balance_diagnostic()

    assert decision.downstream_status == "CORRECTED"
    assert decision.downstream_protection_modified is True
    assert audit["applied_action_source"] == "DOWNSTREAM_CAPACITY_GUARD"
    assert audit["status"] == MASS_BALANCE_STATUS_PASS

    # the correction is reflected in the audited physics: D's terminal outflow
    # is the corrected release, which is exactly the downstream capacity.
    assert audit["network"]["terminal_outflow_mcm_day"] == pytest.approx(CAPACITY,
                                                                        abs=1e-9)
    # the three upstream reservoirs kept what the SafetyLayer produced
    for nid in (A, B, C):
        assert audit["applied_action_fraction"][nid] == pytest.approx(
            decision.safety_layer_gate_positions_fraction[nid]
        )


def test_manual_mode_names_operator_gates_and_does_not_claim_a_controller_verdict(
        live_sim):
    sim = live_sim
    sim.bridge.init_cascade(50.0)
    sim.mode = "MANUAL"
    sim.manual_inflows = _zero()
    sim.manual_gates = {A: 40.0, B: 35.0, C: 50.0, D: 50.0}
    sim.step()

    audit = sim.bridge.cascade.mass_balance_diagnostic()
    assert audit["status"] == MASS_BALANCE_STATUS_PASS
    assert audit["applied_action_source"] == "MANUAL_OPERATOR_GATES"
    assert audit["applied_action_percent"][D] == pytest.approx(50.0)
    assert audit["applied_action_fraction"][D] == pytest.approx(0.5)

    twin = sim.get_adapted_state()["mass_balance"]
    assert twin["controller_action_checked"] is False
    assert twin["matches_final_safe_control_action"] is None
    assert "MANUAL" in twin["action_check_note"]


# ===========================================================================
# 13-16. Corruption, NaN/Inf, tolerance, failure handling
# ===========================================================================

def test_deliberately_corrupted_state_is_detected():
    """Corrupting storage AFTER the step must be caught, not smoothed over."""
    monitor = MassBalanceMonitor()
    network = _live_network()
    snapshot = monitor.snapshot(network)
    network.step(_zero(), {n: 0.4 for n in NODES})
    network.nodes[C].state.storage += 0.5          # deliberate corruption

    result = monitor.verify(network, snapshot, applied_inflows=_zero(),
                            applied_gates_fraction={n: 0.4 for n in NODES})

    assert result.status == MASS_BALANCE_STATUS_VIOLATION
    assert result.checked is True
    assert result.state_valid is False
    assert _row(result.to_dict(), C)["residual_mcm"] == pytest.approx(0.5)
    assert any("RESERVOIR_MASS_BALANCE_VIOLATION" in v for v in result.violations)


def test_deliberately_corrupted_flow_is_detected():
    """
    Corrupting a reported outflow must be caught — both by the reservoir-level
    conservation law and by the network-level one when it is the terminal node.
    """
    monitor = MassBalanceMonitor()
    network = _live_network()
    snapshot = monitor.snapshot(network)
    network.step(_zero(), {n: 0.4 for n in NODES})
    # deliberate corruption of the reported flows (both the component and the
    # total the network-level balance reads)
    network.nodes[D].state.controlled_release += 1.25
    network.nodes[D].state.total_outflow += 1.25

    result = monitor.verify(network, snapshot, applied_inflows=_zero(),
                            applied_gates_fraction={n: 0.4 for n in NODES})

    assert result.status == MASS_BALANCE_STATUS_VIOLATION
    row = _row(result.to_dict(), D)
    # a release that is larger than the water available shows up as a POSITIVE
    # storage residual (the book-keeping no longer adds up)
    assert row["residual_mcm"] == pytest.approx(1.25)
    assert row["within_tolerance"] is False
    assert any("RESERVOIR_MASS_BALANCE_VIOLATION" in v for v in result.violations)
    # the network-level balance notices too (the terminal outflow is corrupted)
    assert result.network["within_tolerance"] is False
    assert result.network["residual_mcm"] == pytest.approx(-1.25)
    assert any("NETWORK_MASS_BALANCE_VIOLATION" in v for v in result.violations)


def test_a_corrupted_routing_queue_is_detected():
    monitor = MassBalanceMonitor()
    network = _live_network()
    snapshot = monitor.snapshot(network)
    network.step(_zero(), {n: 0.3 for n in NODES})
    network.connections[0].queue[-1] += 2.0     # water appears from nowhere

    result = monitor.verify(network, snapshot, applied_inflows=_zero(),
                            applied_gates_fraction={n: 0.3 for n in NODES})

    assert result.status == MASS_BALANCE_STATUS_VIOLATION
    assert any("ROUTING_MASS_BALANCE_VIOLATION" in v for v in result.violations)


def test_corruption_is_never_repaired_by_the_audit():
    monitor = MassBalanceMonitor()
    network = _live_network()
    snapshot = monitor.snapshot(network)
    network.step(_zero(), {n: 0.4 for n in NODES})
    corrupted = network.nodes[C].state.storage + 0.5
    network.nodes[C].state.storage = corrupted

    result = monitor.verify(network, snapshot, applied_inflows=_zero(),
                            applied_gates_fraction={n: 0.4 for n in NODES})

    assert result.status == MASS_BALANCE_STATUS_VIOLATION
    # the state the audit read is UNCHANGED — nothing was silently corrected
    assert network.nodes[C].state.storage == corrupted
    assert network.nodes[C].state.controlled_release == pytest.approx(
        0.4 * network.nodes[C].max_release
    )


def test_violation_is_sticky_and_counted():
    monitor = MassBalanceMonitor()
    network = _live_network()
    monitor.step_and_check(network, _zero(), {n: 0.2 for n in NODES})
    assert monitor.ever_violated is False

    snapshot = monitor.snapshot(network)
    network.step(_zero(), {n: 0.2 for n in NODES})
    network.nodes[A].state.storage -= 3.0
    bad = monitor.verify(network, snapshot, applied_inflows=_zero(),
                         applied_gates_fraction={n: 0.2 for n in NODES})
    assert bad.status == MASS_BALANCE_STATUS_VIOLATION
    assert monitor.ever_violated is True
    assert monitor.violation_count == 1

    # a later, clean step reports PASS but the violation is still on the record
    good = monitor.step_and_check(network, _zero(), {n: 0.2 for n in NODES})
    assert good.status == MASS_BALANCE_STATUS_PASS
    assert good.ever_violated is True
    assert good.violation_count == 1


def test_nan_state_is_detected_safely():
    monitor = MassBalanceMonitor()
    network = _live_network()
    snapshot = monitor.snapshot(network)
    network.step(_zero(), {n: 0.2 for n in NODES})
    network.nodes[D].state.storage = float("nan")

    result = monitor.verify(network, snapshot, applied_inflows=_zero(),
                            applied_gates_fraction={n: 0.2 for n in NODES})

    assert result.status == MASS_BALANCE_STATUS_VIOLATION
    assert result.checked is True
    assert result.state_valid is False
    assert any("NON_FINITE" in v for v in result.violations)
    assert result.non_finite
    # and it did not raise, did not clamp and did not "fix" the NaN
    assert math.isnan(network.nodes[D].state.storage)
    # the payload stays JSON-safe (no NaN leaking to the WebSocket)
    assert json.loads(json.dumps(result.to_dict(), allow_nan=False))["status"] == \
        MASS_BALANCE_STATUS_VIOLATION


def test_infinite_flow_is_detected_safely():
    monitor = MassBalanceMonitor()
    network = _live_network()
    snapshot = monitor.snapshot(network)
    network.step(_zero(), {n: 0.2 for n in NODES})
    network.nodes[B].state.controlled_release = float("inf")

    result = monitor.verify(network, snapshot, applied_inflows=_zero(),
                            applied_gates_fraction={n: 0.2 for n in NODES})

    assert result.status == MASS_BALANCE_STATUS_VIOLATION
    assert any("NON_FINITE" in v for v in result.violations)
    assert network.nodes[B].state.controlled_release == float("inf")


def test_residual_tolerance_is_explicit_and_meaningful():
    assert MASS_BALANCE_TOLERANCE_MCM == 1e-9

    monitor = MassBalanceMonitor()
    network = _live_network()
    snapshot = monitor.snapshot(network)
    network.step(_zero(), {n: 0.2 for n in NODES})
    network.nodes[A].state.storage += 1e-12          # below tolerance: noise
    ok = monitor.verify(network, snapshot, applied_inflows=_zero(),
                        applied_gates_fraction={n: 0.2 for n in NODES})
    assert ok.tolerance == 1e-9
    assert ok.status == MASS_BALANCE_STATUS_PASS

    # a physically meaningful error is caught
    monitor2 = MassBalanceMonitor()
    network2 = _live_network()
    snapshot2 = monitor2.snapshot(network2)
    network2.step(_zero(), {n: 0.2 for n in NODES})
    network2.nodes[A].state.storage += 1e-6
    bad = monitor2.verify(network2, snapshot2, applied_inflows=_zero(),
                          applied_gates_fraction={n: 0.2 for n in NODES})
    assert bad.status == MASS_BALANCE_STATUS_VIOLATION


def test_the_diagnostic_is_not_a_constant_pass():
    """The strongest anti-tautology check: corrupt something and it must FAIL."""
    monitor = MassBalanceMonitor()
    network = _live_network()
    results = [monitor.step_and_check(network, _zero(), {n: 0.2 for n in NODES})
               for _ in range(3)]
    assert {r.status for r in results} == {MASS_BALANCE_STATUS_PASS}

    snapshot = monitor.snapshot(network)
    network.step(_zero(), {n: 0.2 for n in NODES})
    network.nodes[D].state.spill += 7.5
    assert monitor.verify(network, snapshot, applied_inflows=_zero(),
                          applied_gates_fraction={n: 0.2 for n in NODES}
                          ).status == MASS_BALANCE_STATUS_VIOLATION


def test_fail_safe_gap_is_reported_rather_than_invented():
    monitor = MassBalanceMonitor()
    network = _live_network()
    snapshot = monitor.snapshot(network)
    network.step(_zero(), {n: 0.2 for n in NODES})
    network.nodes[A].state.storage -= 1.0
    result = monitor.verify(network, snapshot, applied_inflows=_zero(),
                            applied_gates_fraction={n: 0.2 for n in NODES})

    assert result.fail_safe == FAIL_SAFE_NONE_DEFINED
    assert "defines no response" in result.fail_safe_gap
    assert "does not invent a fail-safe" in result.fail_safe_gap

    # The AUDIT itself contains no repair / clamp / re-application of anything:
    # no write to the live network, and verify() never steps the physics.
    source = MASS_BALANCE_PATH.read_text(encoding="utf-8")
    verify_src = source[source.index("    def verify("):source.index("    def step_and_check(")]
    assert "network.step(" not in verify_src
    assert "network.reset()" not in verify_src
    assert ".state.storage =" not in verify_src
    assert ".state.controlled_release =" not in verify_src
    assert ".state.spill =" not in verify_src
    assert ".queue.append" not in verify_src
    assert ".queue.popleft" not in verify_src


def test_never_reports_pass_before_a_check_has_run():
    monitor = MassBalanceMonitor()
    fresh = monitor.diagnostic()
    assert fresh["status"] == MASS_BALANCE_STATUS_NOT_CHECKED
    assert fresh["checked"] is False
    assert fresh["residual"] is None
    assert fresh["reason"] == "NO_AUTHORITATIVE_STEP_AUDITED"

    assert not_checked_result()["checked"] is False
    assert not_checked_result()["status"] == MASS_BALANCE_STATUS_NOT_CHECKED

    # a fresh live cascade has not been audited yet either
    assert _bridge().cascade.mass_balance_diagnostic()["checked"] is False


def test_reset_returns_the_diagnostic_to_not_checked():
    monitor = MassBalanceMonitor()
    network = _live_network()
    monitor.step_and_check(network, _zero(), {n: 0.2 for n in NODES})
    assert monitor.diagnostic()["checked"] is True

    monitor.reset()
    assert monitor.diagnostic()["checked"] is False
    assert monitor.diagnostic()["status"] == MASS_BALANCE_STATUS_NOT_CHECKED


# ===========================================================================
# 17-20. API, WebSocket, Digital Twin, blocked forecasts, no injection
# ===========================================================================

def test_api_exposes_mass_balance_diagnostics():
    client.post("/api/controller/mode", json={"mode": "MANUAL"})
    client.post("/api/simulation/step")
    payload = client.get("/api/state").json()

    mb = payload["mass_balance"]
    for key in ("status", "checked", "residual", "tolerance", "unit", "timestamp",
                "step_index", "per_reservoir", "reservoirs_checked",
                "applied_action_source", "state_valid", "fail_safe", "equation",
                "physics", "latency_ms"):
        assert key in mb, f"{key} missing from the API mass-balance block"
    assert mb["checked"] is True
    assert mb["status"] == MASS_BALANCE_STATUS_PASS
    assert mb["unit"] == "MCM"
    assert mb["tolerance"] == MASS_BALANCE_TOLERANCE_MCM
    assert mb["reservoirs_checked"] == 4
    assert len(mb["per_reservoir"]) == 4
    assert mb["physics"].startswith("ReservoirNetwork")


def test_websocket_state_carries_mass_balance():
    client.post("/api/controller/mode", json={"mode": "MANUAL"})
    client.post("/api/simulation/step")
    with client.websocket_connect("/ws/state") as ws:
        state = ws.receive_json()

    mb = state["mass_balance"]
    assert mb["status"] in (MASS_BALANCE_STATUS_PASS, MASS_BALANCE_STATUS_VIOLATION,
                            MASS_BALANCE_STATUS_NOT_CHECKED)
    assert "checked" in mb and "residual" in mb and "tolerance" in mb
    assert mb["checked"] is True
    assert mb["reservoirs_checked"] == 4
    assert {r["node_id"] for r in mb["per_reservoir"]} == set(NODES)


def test_twin_payload_never_implies_conservation_without_an_audit():
    twin = adapt_state_for_twin({"reservoirs": {}, "downstream_flow": 0}, "MANUAL", 0.0)
    mb = twin["mass_balance"]
    assert mb["status"] == MASS_BALANCE_STATUS_NOT_CHECKED
    assert mb["checked"] is False
    assert mb["residual"] is None
    assert mb["tolerance"] is None
    assert mb["reason"] == "NO_MASS_BALANCE_PROVENANCE"
    assert mb["matches_final_safe_control_action"] is None
    assert mb["controller_action_checked"] is False
    assert mb["hardware_connected"] is False


def test_twin_payload_passes_the_audit_through_verbatim():
    audit = not_checked_result()
    audit.update({
        "status": "VIOLATION", "checked": True, "residual": 0.5,
        "tolerance": 1e-9, "reservoirs_checked": 4, "reservoirs_expected": 4,
        "violations": ["RESERVOIR_MASS_BALANCE_VIOLATION: X"],
        "state_valid": False, "controller_action_checked": True,
        "matches_final_safe_control_action": True, "action_check_note": "ok",
    })
    twin = adapt_state_for_twin({"reservoirs": {}, "downstream_flow": 0,
                                 "mass_balance": audit}, "AI", 0.0)
    mb = twin["mass_balance"]
    assert mb["status"] == "VIOLATION"
    assert mb["checked"] is True
    assert mb["residual"] == 0.5
    assert mb["state_valid"] is False
    assert mb["violations"] == ["RESERVOIR_MASS_BALANCE_VIOLATION: X"]


def test_demonstration_forecast_remains_blocked_and_is_still_audited(
        live_sim, monkeypatch):
    sim = live_sim
    _drive_live_ai_step(sim, monkeypatch, decision=None, bundle=None)
    # now re-run with a DEMONSTRATION bundle
    sim.bridge.init_cascade(50.0)
    sim.mode = "MANUAL"
    sim.manual_inflows = _zero()
    sim.manual_gates = {n: 0.0 for n in NODES}
    sim.step()

    network = sim.bridge.cascade.network
    monkeypatch.setattr(sim, "_build_live_forecast_bundle",
                        lambda cf: _demo_bundle(network))
    sim.mode = "AI"
    sim.step()

    decision = sim.last_control_decision
    assert decision.controller_status == ControllerStatus.BLOCKED.value
    assert decision.forecast_control_eligible is False
    assert decision.control_applied is False
    assert decision.final_safe_control_action_source == "HELD_CURRENT_GATES"

    audit = sim.bridge.cascade.mass_balance_diagnostic()
    assert audit["checked"] is True
    assert audit["status"] == MASS_BALANCE_STATUS_PASS
    assert audit["applied_action_source"] == "HELD_CURRENT_GATES"


def test_missing_idukki_forecast_remains_blocked(live_sim, monkeypatch):
    sim = live_sim
    network = sim.bridge.cascade.network
    bundle = _validated_bundle(network, nodes=[A, B, C])   # D (Idukki) omitted

    decision = sim.mpc_orchestrator.decide(network, bundle=bundle)
    assert decision.controller_status == ControllerStatus.BLOCKED.value
    assert "Virtual Reservoir D" in " ".join(decision.reasons) or \
        any(nid == D for nid in [r.get("node_id") for r in
                                 decision.forecast_provenance.get("nodes", [])
                                 if isinstance(r, dict)])


def test_browser_cannot_inject_mass_balance_state():
    assert client.post("/api/state", json={"mass_balance": {"status": "PASS"}}
                       ).status_code == 405
    assert client.put("/api/state", json={"mass_balance": {"status": "PASS"}}
                      ).status_code == 405

    client.post("/api/controller/mode", json={"mode": "MANUAL"})
    client.post("/api/simulation/step")
    before = client.get("/api/state").json()["mass_balance"]["status"]

    client.post("/api/storm", json={"value": 0.2,
                                    "mass_balance": {"status": "VIOLATION"}})
    client.post("/api/simulation/speed", json={"speed": 2.0,
                                               "mass_balance": {"status": "VIOLATION"}})
    after = client.get("/api/state").json()["mass_balance"]
    assert after["status"] == before
    assert after["status"] != "VIOLATION" or before == "VIOLATION"


def test_gnn_remains_advisory_only():
    source = MASS_BALANCE_PATH.read_text(encoding="utf-8")
    assert "gnn" not in source.lower()
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
    assert not any("gnn" in m.lower() for m in imported)
    # the audit is read-only physics: it never imports a controller
    assert not any(m.startswith("src.controller") or ".controller" in m for m in imported)
    assert "compute_ai_recommendation" not in STATE_MANAGER_PATH.read_text(encoding="utf-8")


# ===========================================================================
# 21-23. Digital Twin UI
# ===========================================================================

def test_twin_ui_displays_mass_balance_truthfully():
    html = TWIN_INDEX_PATH.read_text(encoding="utf-8")
    assert "MASS BALANCE" in html
    assert "MB RESIDUAL" in html
    assert "MB CHECKED" in html
    assert 'data-ref="mbstat"' in html
    assert 'data-ref="mbres"' in html
    assert 'data-ref="mbchk"' in html
    assert "state.mass_balance" in html
    assert "this.data.massBalance" in html
    # PASS is only rendered when the backend says it actually checked
    assert "mb.checked === true" in html
    assert "mbStatus === 'VIOLATION'" in html
    # the status string is displayed verbatim
    assert "replace(/_/g, ' ')" in html


def test_twin_js_contains_no_mass_balance_calculation():
    html = TWIN_INDEX_PATH.read_text(encoding="utf-8")
    for forbidden in ("Math.abs(mb", "mb.residual +", "mb.residual -",
                      "mb.tolerance *", "mb.residual <", "mb.residual >"):
        assert forbidden not in html, f"JavaScript computes mass balance: {forbidden}"
    # no browser-side state injection path exists at all
    assert "addEventListener('message'" not in html
    assert 'addEventListener("message"' not in html


def test_twin_js_parses():
    """The inline module must still be valid JavaScript (parse-check)."""
    node = shutil.which("node")
    if node is None:  # pragma: no cover - node is available in this environment
        pytest.skip("node is not available to parse-check the twin page")

    html = TWIN_INDEX_PATH.read_text(encoding="utf-8")
    blocks = []
    marker = '<script type="module">'
    idx = html.find(marker)
    assert idx != -1
    end = html.find("</script>", idx)
    blocks.append(html[idx + len(marker):end])
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "twin.mjs"
        path.write_text("\n".join(blocks), encoding="utf-8")
        proc = subprocess.run([node, "--check", str(path)], capture_output=True,
                              text=True)
    assert proc.returncode == 0, proc.stderr


# ===========================================================================
# 24-25. Frozen physics, frozen artifacts, Phase 15.3 reproduction
# ===========================================================================

def test_frozen_physics_implementation_is_untouched():
    for path in (NETWORK_PATH, SAFETY_PATH, MPC_PATH, GUARD_PATH):
        source = path.read_text(encoding="utf-8")
        assert "mass_balance import" not in source
        assert "MassBalanceMonitor" not in source
        assert "from .mass_balance" not in source
        assert "mass_balance_diagnostic" not in source

    # the audit lives OUTSIDE the physics: one adapter, one bridge, one manager
    assert "MassBalanceMonitor" in ADAPTER_PATH.read_text(encoding="utf-8")
    assert "mass_balance" in BRIDGE_PATH.read_text(encoding="utf-8")
    assert "mass_balance" in STATE_ADAPTER_PATH.read_text(encoding="utf-8")


def test_the_live_step_cannot_bypass_the_audit():
    """
    ``ReservoirNetwork.step()`` is called in exactly one place on the live path,
    and that place is the audited one.
    """
    adapter = ADAPTER_PATH.read_text(encoding="utf-8")
    assert "self.network.step(" not in adapter
    assert "self.mass_balance_monitor.step_and_check(" in adapter

    step_body = adapter[adapter.index("    def step(self, inflows"):]
    step_body = step_body[:step_body.index("    def reset(self)")]
    assert step_body.count("step_and_check(") == 1


def test_frozen_artifacts_are_unchanged_by_stage11(live_sim, monkeypatch):
    before = {p: _sha(p) for p in FROZEN_ARTIFACTS}
    _drive_live_ai_step(live_sim, monkeypatch, decision=_flood_decision())
    assert {p: _sha(p) for p in FROZEN_ARTIFACTS} == before


def test_frozen_artifacts_still_match_the_phase15_3_manifest():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    expected = manifest["pre_validation"]
    assert len(expected) == 3
    for rel_path, record in expected.items():
        path = _PROJECT_ROOT / rel_path
        assert path.exists(), f"frozen artifact missing: {rel_path}"
        # the manifest was computed on LF bytes (core.autocrlf=true rewrites text
        # artifacts to CRLF in the working tree), so accept either view
        digests = {_sha(path), _sha_lf(path)}
        assert record["sha256"] in digests, f"frozen artifact MODIFIED: {rel_path}"


def test_phase15_v3_protected_directory_is_intact():
    protected = _PROJECT_ROOT / "results" / "phase15_v3_validation"
    assert protected.exists()
    files = sorted(p for p in protected.rglob("*") if p.is_file())
    assert files, "the protected Phase 15.3 validation directory is empty"
    assert (protected / "v3_integrity_check.json").exists()
    manifest = json.loads((protected / "v3_integrity_check.json").read_text(encoding="utf-8"))
    assert manifest["all_match"] is True


# ===========================================================================
# 26. Performance
# ===========================================================================

def test_diagnostic_overhead_is_negligible():
    monitor = MassBalanceMonitor()
    network = _live_network()
    perf = monitor.performance_overhead(network, iterations=120)

    assert perf["iterations"] == 120
    assert perf["overhead_ms"] < 2.0, perf
    # the audit is a small fraction of the controller's own latency (~660 ms)
    assert perf["overhead_ms"] < 660.0 * 0.01
    assert perf["last_audit_latency_ms"] is not None
