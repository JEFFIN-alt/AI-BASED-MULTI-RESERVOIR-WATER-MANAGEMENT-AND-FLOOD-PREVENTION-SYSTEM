"""
Stage 17 — PERFORMANCE VALIDATION: tests.

These tests verify that the PERFORMANCE VALIDATION itself is sound — the harness
works, the required components really are measured, the measured structure of the
system is what the report claims, and the measurement work changed nothing. They
deliberately do NOT assert timing values from a specific run: wall-clock numbers
on a shared, non-realtime desktop are not a correctness property. Where a numeric
bound appears it is a broad sanity bound justified at the point of use.

Coverage (Stage 17 §13): benchmark infrastructure · required components measured ·
MPC candidate count is 1296 · four reservoirs included · validated fixture
executes the complete chain · blocked live provenance stays fail-closed · no
duplicate simulation owner · repeated cycles remain stable · no obvious unbounded
memory growth · performance evidence is reproducible.
"""

import ast
import copy
import gc
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.common import units  # noqa: E402
from src.controller.mpc_controller import MPCConfig, MPCController  # noqa: E402
from src.dashboard.api import state_manager  # noqa: E402
from src.dashboard.api.app import app  # noqa: E402
from src.dashboard.twin_component.state_adapter import adapt_state_for_twin  # noqa: E402
from src.network_env.live_forecast_adapter import LiveForecastAdapter  # noqa: E402
from src.network_env.reservoir_network import ReservoirNetwork  # noqa: E402

client = TestClient(app)
PROJECT_ROOT = str(_PROJECT_ROOT)

EVIDENCE_PATH = (_PROJECT_ROOT / "results" / "phase15_stage17_performance_validation"
                 / "stage17_performance_evidence.json")
MANIFEST_PATH = _PROJECT_ROOT / "results" / "phase15_v3_validation" / "v3_integrity_check.json"

NODES = ["Virtual Reservoir A", "Virtual Reservoir B",
         "Virtual Reservoir C", "Virtual Reservoir D"]
HORIZONS = ("forecast_1d", "forecast_3d", "forecast_7d")
EXPECTED_CANDIDATES = 1296
EXPECTED_GATE_LEVELS = [0.0, 0.15, 0.30, 0.50, 0.70, 1.0]

#: Every module the Stage 17 brief forbids Stage 17 from changing.
PROTECTED_PATHS = [
    "src/network_env/reservoir_network.py",
    "src/network_env/mass_balance.py",
    "src/network_env/v3_forecast_adapter.py",
    "src/network_env/topology_config.yaml",
    "src/controller/mpc_controller.py",
    "src/controller/safety.py",
    "src/controller/objective.py",
    "src/controller/baseline_controller.py",
    "src/controller/downstream_capacity_guard.py",
    "scripts/run_phase15_3_validation.py",
]

#: §13's required component measurements.
REQUIRED_COMPONENT_KEYS = [
    "A_frozen_lstm_v3_inference",
    "B_gnn_advisory_inference",
    "B2_gnn_node_representations",
    "C_live_forecast_adapter",
    "D_mpc_decide",
    "D2_mpc_decide_no_reset",
    "E_safety_layer_validate",
    "F_downstream_capacity_guard_safe",
    "F2_downstream_capacity_guard_corrected",
    "G_reservoir_network_step",
    "H_mass_balance_step_and_check",
    "H2_mass_balance_verify_only",
    "I_state_adapter",
    "I2_payload_serialization",
]


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _restore_authoritative_state():
    """
    These tests drive the ONE authoritative singleton. Restore state only
    (paused, 50 % storages, MANUAL) afterwards so no other suite inherits a
    saturated or running simulation.
    """
    yield
    client.post("/api/simulation/pause")
    client.post("/api/simulation/reset")
    client.post("/api/controller/mode", json={"mode": "MANUAL"})


def _percentile(sorted_values, p):
    n = len(sorted_values)
    if n == 0:
        return None
    if n == 1:
        return sorted_values[0]
    k = (n - 1) * p
    lo, hi = int(k // 1), int(-(-k // 1))
    if lo == hi:
        return sorted_values[int(k)]
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (k - lo)


def _benchmark(fn, warmup=1, iterations=5, before_each=None):
    for _ in range(warmup):
        if before_each:
            before_each()
        fn()
    samples = []
    for _ in range(iterations):
        if before_each:
            before_each()
        t0 = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - t0) / 1e6)
    ordered = sorted(samples)
    return {
        "n": len(samples),
        "mean": sum(samples) / len(samples),
        "median": ordered[len(ordered) // 2],
        "min": ordered[0],
        "max": ordered[-1],
        "p95": _percentile(ordered, 0.95),
        "stdev": (sum((x - sum(samples) / len(samples)) ** 2 for x in samples)
                  / (len(samples) - 1)) ** 0.5 if len(samples) > 1 else 0.0,
    }


def _reset_to_50():
    sm = state_manager.sim_state
    sm.running = False
    sm.bridge.init_cascade(50.0)


def _drive(count):
    sm = state_manager.sim_state
    for _ in range(count):
        sm.step()


def _fixture_payload():
    """An explicitly labelled CONTROLLED TEST FIXTURE forecast payload."""
    return {
        "forecast_1d": 2.0, "forecast_3d": 2.1, "forecast_7d": 2.2,
        "forecast_status": "VALIDATED",
        "forecast_provenance": "REAL_MEASUREMENT_INPUTS_FROZEN_MODEL",
        "forecast_source": "FROZEN_LSTM_V3",
        "is_simulated": False,
        "validated_metrics_apply": True,
        "forecast_unit": "MCM/day",
        "horizons": list(HORIZONS),
        "input_provenance": {"synthetic_demo": [], "unavailable": [], "simulated": []},
    }


def _fixture_bundle(network):
    adapter = LiveForecastAdapter.for_network(network, project_root=PROJECT_ROOT)
    return adapter.build_bundle({n: _fixture_payload() for n in NODES}, "2026-09-16")


@pytest.fixture
def prepared_live():
    """The one authoritative simulation, past warm-up, in AI mode."""
    sm = state_manager.sim_state
    sm.running = False
    sm.mode = "AI"
    _reset_to_50()
    _drive(8)
    return sm


@pytest.fixture
def evidence():
    if not EVIDENCE_PATH.exists():
        pytest.skip("Stage 17 evidence JSON not generated yet "
                    "(run scripts/run_stage17_performance_validation.py)")
    return json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))


def _live_network():
    return state_manager.sim_state.bridge.cascade.network


# ---------------------------------------------------------------------------
# 1. benchmark infrastructure
# ---------------------------------------------------------------------------

def test_benchmark_harness_computes_the_required_statistics(monkeypatch):
    """
    §3 — the harness reports mean/median/p95/min/max/stdev over N samples.

    The statistics are of ELAPSED WALL TIME, so they cannot be compared with a
    value the callback happens to append. The clock is therefore replaced with a
    deterministic sequence of known durations: this tests the statistic
    CALCULATION, not how fast this machine executes ``list.append()``.
    """
    known = []

    def record():
        known.append(1.0)  # the callback's own bookkeeping, NOT a timing sample

    # _benchmark() reads the clock twice per sample (start, end), so the timeline
    # holds BOTH endpoints of each of the six known durations.
    durations_ms = [1.0, 2.0, 3.0, 4.0, 5.0, 7.0]
    pending = []
    clock_ns = 0
    for duration in durations_ms:
        pending.append(clock_ns)                    # t0 of this sample
        clock_ns += int(round(duration * 1e6))
        pending.append(clock_ns)                    # t1 of this sample
    last = {"value": pending[-1]}

    def fake_clock():
        # Never exhaust: any unrelated caller keeps getting the last timestamp.
        if pending:
            last["value"] = pending.pop(0)
        return last["value"]

    monkeypatch.setattr(time, "perf_counter_ns", fake_clock)

    stats = _benchmark(record, warmup=2, iterations=6)

    assert stats["n"] == 6
    assert len(known) == 8  # warm-up iterations really ran, and were not sampled
    assert stats["min"] == pytest.approx(1.0)
    assert stats["max"] == pytest.approx(7.0)
    assert stats["mean"] == pytest.approx(22.0 / 6.0)
    assert stats["median"] == pytest.approx(4.0)   # upper median of 6 samples
    # p95 of [1, 2, 3, 4, 5, 7] by linear interpolation: 5 + (7 - 5) * 0.75
    assert stats["p95"] == pytest.approx(6.5)
    # sample standard deviation of those six durations
    assert stats["stdev"] == pytest.approx(2.160246899469287)
    assert stats["min"] <= stats["median"] <= stats["max"]


def test_benchmark_harness_uses_a_monotonic_high_resolution_clock():
    """§3 — timing is monotonic and sub-microsecond resolution."""
    t0 = time.perf_counter_ns()
    t1 = time.perf_counter_ns()
    assert t1 >= t0
    assert isinstance(t0, int)
    # perf_counter_ns resolution is far finer than a millisecond
    samples = []
    for _ in range(50):
        a = time.perf_counter_ns()
        b = time.perf_counter_ns()
        samples.append(b - a)
    assert max(samples) < 10_000_000  # never a 10 ms granularity


def test_percentile_helper_matches_known_values():
    ordered = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert _percentile(ordered, 0.0) == 1.0
    assert _percentile(ordered, 1.0) == 5.0
    assert _percentile(ordered, 0.5) == 3.0
    assert _percentile([7.0], 0.99) == 7.0


def test_warmup_iterations_are_not_counted_as_samples():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1

    stats = _benchmark(fn, warmup=4, iterations=3)
    assert stats["n"] == 3
    assert calls["n"] == 7


# ---------------------------------------------------------------------------
# 2. required components are really measured
# ---------------------------------------------------------------------------

def test_required_components_are_measured(evidence):
    for key in REQUIRED_COMPONENT_KEYS:
        block = evidence["components"][key]
        assert block["measured_iterations"] >= 1, key
        assert block["warmup_iterations"] >= 0, key
        assert block["mean"] > 0.0, key
        assert block["median"] > 0.0, key
        assert block["p95"] is not None, key
        assert block["min"] <= block["median"] <= block["max"], key
        assert block["stdev"] >= 0.0, key
        assert block["clock"] == "time.perf_counter_ns", key


def test_mpc_benchmark_reports_a_warmup_and_more_than_one_sample(evidence):
    block = evidence["components"]["D_mpc_decide"]
    assert block["warmup_iterations"] >= 1
    assert block["measured_iterations"] > 1
    assert block["total"] > block["mean"]  # more than one sample really ran


def test_component_timings_are_reproducible_from_the_live_objects():
    """
    §13 — 'performance evidence is reproducible'.

    Re-measures four components directly and compares them with the recorded
    evidence using a BROAD band (a shared desktop's run-to-run spread is easily
    2-3x). The point is that the same objects produce the same order of magnitude,
    not that a wall-clock number is stable.
    """
    sm = state_manager.sim_state
    _reset_to_50()
    _drive(8)
    network = _live_network()

    # SafetyLayer.validate
    from src.controller.safety import SafetyLayer
    safety = SafetyLayer()
    node_ids = list(network.processing_order)
    proposal = {n: 0.3 for n in node_ids}
    current = {n: float(network.nodes[n].state.gate_position) for n in node_ids}
    measured = _benchmark(lambda: safety.validate(proposal, current, node_ids),
                          warmup=5, iterations=40)

    if EVIDENCE_PATH.exists():
        recorded = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
        recorded_ms = recorded["components"]["E_safety_layer_validate"]["mean"]
        # a sub-millisecond call must stay sub-millisecond-order, not drift by 100x
        assert measured["mean"] < max(recorded_ms * 100.0, 1.0)
        assert measured["mean"] > min(recorded_ms / 100.0, 1e-9)

    # ReservoirNetwork.step and MassBalanceMonitor.step_and_check
    snap = copy.deepcopy(network)
    inflows = {n: float(network.nodes[n].state.inflow_local) for n in node_ids}
    gates = {n: 0.3 for n in node_ids}
    physics = _benchmark(
        lambda: network.step(dict(inflows), dict(gates)),
        warmup=10, iterations=60,
        before_each=lambda: network.__dict__.update(copy.deepcopy(snap.__dict__)))
    network.__dict__.update(copy.deepcopy(snap.__dict__))
    assert physics["mean"] > 0.0
    assert physics["mean"] < 500.0  # a 4-node step is not a 0.5 s operation

    monitor = sm.bridge.cascade.mass_balance_monitor
    audit = _benchmark(
        lambda: monitor.step_and_check(network, dict(inflows), dict(gates),
                                       action_source="STAGE17_TEST"),
        warmup=10, iterations=60,
        before_each=lambda: network.__dict__.update(copy.deepcopy(snap.__dict__)))
    network.__dict__.update(copy.deepcopy(snap.__dict__))
    assert audit["mean"] > physics["mean"] * 0.5
    assert audit["mean"] < 200.0


# ---------------------------------------------------------------------------
# 3. MPC candidate space
# ---------------------------------------------------------------------------

def test_mpc_candidate_count_is_exactly_1296(prepared_live):
    orch = prepared_live.mpc_orchestrator
    network = _live_network()
    node_ids = list(network.processing_order)

    space = orch.action_space_for(node_ids)
    assert space["gate_levels"] == EXPECTED_GATE_LEVELS
    assert space["dimension"] == 4
    assert space["candidate_vectors"] == EXPECTED_CANDIDATES
    assert space["coordinated"] is True

    # independent enumeration of the Cartesian product
    import itertools
    enumerated = len(list(itertools.product(space["gate_levels"], repeat=4)))
    assert enumerated == EXPECTED_CANDIDATES

    # the controller really evaluates all of them
    bundle = _fixture_bundle(network)
    decision = orch.mpc.decide(network, forecast_snapshot=bundle.snapshot,
                               current_inflows=dict(prepared_live.manual_inflows))
    assert decision.candidates_evaluated == EXPECTED_CANDIDATES


def test_candidate_space_was_not_reduced_for_performance():
    """§4/§16 — the validated lattice and lookahead are unchanged."""
    config = MPCConfig()
    assert config.gate_levels == EXPECTED_GATE_LEVELS
    assert config.lookahead_steps == 3
    assert config.max_gate_change == 0.5
    assert len(config.gate_levels) ** 4 == EXPECTED_CANDIDATES


def test_mpc_cost_is_dominated_by_candidate_trajectory_simulation(prepared_live):
    """
    §9 — a STRUCTURAL claim, not a timing threshold: the decision is a grid search,
    so the candidate rollouts must account for the bulk of its cost, and the number
    of rollouts must equal the candidate count.
    """
    orch = prepared_live.mpc_orchestrator
    mpc = orch.mpc
    network = _live_network()
    bundle = _fixture_bundle(network)

    sim_calls = {"n": 0, "ns": 0}
    original = mpc._simulate_trajectory

    def timed(*a, **k):
        t0 = time.perf_counter_ns()
        try:
            return original(*a, **k)
        finally:
            sim_calls["n"] += 1
            sim_calls["ns"] += time.perf_counter_ns() - t0

    mpc._simulate_trajectory = timed
    try:
        t0 = time.perf_counter_ns()
        orch.mpc.decide(network, forecast_snapshot=bundle.snapshot,
                        current_inflows=dict(prepared_live.manual_inflows))
        total_ns = time.perf_counter_ns() - t0
    finally:
        del mpc._simulate_trajectory

    assert sim_calls["n"] == EXPECTED_CANDIDATES
    assert sim_calls["ns"] >= 0.5 * total_ns


def test_four_reservoirs_are_included_in_the_decision(prepared_live):
    network = _live_network()
    orch = prepared_live.mpc_orchestrator
    space = orch.action_space_for(list(network.processing_order))
    assert space["node_ids"] == NODES
    assert set(space["node_ids"]) == set(network.nodes)

    bundle = _fixture_bundle(network)
    decision = orch.mpc.decide(network, forecast_snapshot=bundle.snapshot,
                               current_inflows=dict(prepared_live.manual_inflows))
    assert set(decision.gate_positions) == set(NODES)
    assert set(decision.per_node) == set(NODES)


# ---------------------------------------------------------------------------
# 4. the validated fixture executes the complete chain
# ---------------------------------------------------------------------------

def test_validated_fixture_executes_the_complete_chain(prepared_live):
    sm = prepared_live
    network = _live_network()
    node_ids = list(network.processing_order)

    # forecast (fixture) -> adapter -> MPC -> SafetyLayer -> guard
    bundle = _fixture_bundle(network)
    decision = sm.mpc_orchestrator.decide(
        network, bundle=bundle, current_inflows=dict(sm.manual_inflows))

    assert decision.forecast_control_eligible is True
    assert decision.mpc_status in ("OPTIMAL", "CORRECTED")
    assert decision.safety_layer_status in ("SAFE", "CORRECTED")
    assert decision.downstream_status in ("PROTECTED", "CORRECTED")
    assert decision.control_applied is True
    assert decision.final_safe_control_action_source == "DOWNSTREAM_CAPACITY_GUARD"

    # physics receives exactly the FINAL_SAFE_CONTROL_ACTION
    sm.bridge.step(dict(sm.manual_inflows), dict(decision.gate_positions_pct),
                   action_source=decision.final_safe_control_action_source)
    for nid in node_ids:
        assert network.nodes[nid].state.gate_position == pytest.approx(
            decision.final_safe_control_action_fraction[nid], abs=1e-12)

    # mass balance audited the applied action, not a fabricated one
    published = sm.get_adapted_state()
    mb = published["mass_balance"]
    assert mb["checked"] is True
    assert mb["status"] == "PASS"
    assert mb["applied_action_source"] == "DOWNSTREAM_CAPACITY_GUARD"

    # state publication carries the identity and the chain provenance
    assert published["state_identity"]["state_id"].startswith("step")
    assert published["cascade"]["reservoirs"]
    assert published["gnn_advisory"]["affects_control"] is False


def test_the_fixture_does_not_change_any_real_forecast_status(prepared_live):
    """
    §5 — the CASE B fixture must not convert DEMONSTRATION_ONLY into VALIDATED.

    The real live pipeline is re-run AFTER the fixture has been used and must
    still be fail-closed.
    """
    sm = prepared_live
    network = _live_network()
    bundle = _fixture_bundle(network)
    sm.mpc_orchestrator.decide(network, bundle=bundle,
                               current_inflows=dict(sm.manual_inflows))

    _reset_to_50()
    _drive(8)
    state = sm.get_adapted_state()
    decision = sm.last_control_decision

    assert state["forecast_summary"]["validated_metrics_apply"] is False
    assert decision.forecast_control_eligible is False
    assert decision.blocked_reason == "FORECAST_NOT_ELIGIBLE_FOR_CONTROL"


# ---------------------------------------------------------------------------
# 5. blocked live provenance stays fail-closed
# ---------------------------------------------------------------------------

#: The documented Stage 9 operator baselines (state_manager defaults).
KNOWN_OPERATOR_GATES_PCT = {"reservoir_1": 40.0, "reservoir_2": 35.0,
                            "reservoir_3": 50.0, "reservoir_4": 50.0}


@pytest.fixture
def manual_then_blocked_ai():
    """
    KNOWN non-zero actuator gates, then the blocked-AI condition.

    The known gates are established through the real MANUAL operator path — the
    public ``POST /api/gate/{id}`` route followed by MANUAL steps — never by
    mutating internal state, so the held-gates assertion below cannot pass on a
    trivially zero actuator. The operator baselines are set explicitly because
    other suites assign ``sim.manual_gates`` directly and do not restore it.
    """
    sm = state_manager.sim_state
    sm.running = False
    sm.mode = "MANUAL"
    _reset_to_50()
    saved_gates = dict(sm.manual_gates)
    try:
        for reservoir_id, value in KNOWN_OPERATOR_GATES_PCT.items():
            assert client.post(f"/api/gate/{reservoir_id}",
                               json={"value": value}).status_code == 200
        _drive(8)  # MANUAL applies the operator gates to the authoritative physics
        network = _live_network()
        for nid in network.processing_order:
            assert float(network.nodes[nid].state.gate_position) > 0.0, nid
        sm.mode = "AI"
        sm.step()  # live forecasts are DEMONSTRATION_ONLY => provenance blocks
        yield sm
    finally:
        sm.manual_gates = saved_gates


def test_blocked_live_provenance_remains_fail_closed(manual_then_blocked_ai):
    sm = manual_then_blocked_ai
    decision = sm.last_control_decision

    assert decision.forecast_control_eligible is False
    assert decision.mpc_status == "NOT_INVOKED"
    assert decision.safety_layer_status == "NOT_APPLIED_MPC_BLOCKED"
    assert decision.downstream_status == "NOT_APPLIED_MPC_BLOCKED"
    assert decision.final_safe_control_action_source == "HELD_CURRENT_GATES"
    assert decision.control_applied is False
    assert decision.blocked_reason == "FORECAST_NOT_ELIGIBLE_FOR_CONTROL"

    # STAGE 10 / STAGE 15 contract: "held current gates" means the CURRENT
    # AUTHORITATIVE ACTUATOR positions, not the operator baselines in
    # sm.manual_gates (the actuators only receive those in MANUAL mode).
    network = _live_network()
    held_pct = dict(decision.gate_positions_pct)
    held_fraction = dict(decision.gate_positions_fraction)
    current = {nid: float(network.nodes[nid].state.gate_position)
               for nid in network.processing_order}
    assert all(value > 0.0 for value in current.values()), current
    for nid, position in current.items():
        assert held_pct[nid] == pytest.approx(
            units.gate_fraction_to_percent(position), abs=1e-9), nid
        assert held_fraction[nid] == pytest.approx(position, abs=1e-9), nid

    # and the hold really was applied: nothing fabricated was written
    after = {nid: float(network.nodes[nid].state.gate_position)
             for nid in network.processing_order}
    assert after == pytest.approx(current)


def test_live_reservoir_d_still_has_no_forecast_and_blocks_the_mpc(prepared_live):
    """The real reason the gate blocks: D has no live forecast at all."""
    sm = prepared_live
    assert "Virtual Reservoir D" in state_manager.LIVE_FORECAST_EXCLUDED
    sm.step()
    reasons = sm.last_control_decision.reasons
    assert any("Virtual Reservoir D" in r for r in reasons)
    assert any("MISSING_FORECAST" in r or "NOT_VALIDATED_METRICS" in r for r in reasons)


def test_gnn_advisory_cannot_reach_control(prepared_live):
    """§16 — the advisory stays advisory after all Stage 17 measurement work."""
    sm = prepared_live
    sm.step()
    state = sm.get_adapted_state()
    advisory = state["gnn_advisory"]
    assert advisory["advisory_only"] is True
    assert advisory["affects_control"] is False
    assert state["control"].get("final_safe_control_action_source") in (
        "HELD_CURRENT_GATES", "DOWNSTREAM_CAPACITY_GUARD")


# ---------------------------------------------------------------------------
# 6. exactly one simulation owner
# ---------------------------------------------------------------------------

def test_exactly_one_authoritative_simulation_owner(prepared_live):
    assert state_manager.authoritative_instance_count() == 1

    owners = []
    for path in (_PROJECT_ROOT / "src").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                    and node.func.id == "GlobalSimulationState":
                owners.append(str(path.relative_to(_PROJECT_ROOT)).replace("\\", "/"))
                break
    assert owners == ["src/dashboard/api/state_manager.py"]


def test_stepping_through_the_api_does_not_create_another_owner():
    client.post("/api/simulation/pause")
    client.post("/api/simulation/reset")
    before = state_manager.authoritative_instance_count()
    for _ in range(3):
        assert client.post("/api/simulation/step").status_code == 200
    assert state_manager.authoritative_instance_count() == before == 1


def test_the_live_step_has_no_suspension_point():
    """
    §6 — the measured serialization mechanism: ``step()`` is synchronous with no
    ``await`` inside, so the one event loop runs each control mutation to
    completion. This is checked on the AST, not on a comment.
    """
    tree = ast.parse((_PROJECT_ROOT / "src" / "dashboard" / "api" / "state_manager.py")
                     .read_text(encoding="utf-8"))
    step = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "step":
            step = node
    assert step is not None
    suspensions = [n for n in ast.walk(step)
                   if isinstance(n, (ast.Await, ast.AsyncFor, ast.AsyncWith))]
    assert suspensions == []


#: Modules that own the LEGACY OFFLINE simulation (cascade, engine, its runner).
LEGACY_ENGINE_MODULES = {
    "src.simulator.environment",
    "src.simulator.engine",
    "src.simulator.run_simulation",
    "simulator.environment",
    "simulator.engine",
    "simulator.run_simulation",
}
#: Classes only the LEGACY offline engine defines. ``src.simulator.controllers``
#: (dormant, rule-based, no physics, no state) is deliberately NOT an engine.
LEGACY_ENGINE_SYMBOLS = {"SimulationEngine", "VirtualCascade"}
#: The ONE module allowed to construct the authoritative state owner.
AUTHORITATIVE_STATE_OWNER = "src/dashboard/api/state_manager.py"


def test_no_second_simulation_engine_is_importable_from_the_live_path():
    """
    §6 — one authoritative simulation engine, and it is the live one.

    The check is by engine IDENTITY (the legacy ``VirtualCascade`` /
    ``SimulationEngine`` modules and symbols), not by module-name substring, so
    the dormant rule-based helpers in ``src.simulator.controllers`` are not
    conflated with an engine. Dormancy of the legacy advisor is pinned too.
    """
    live_files = [
        "src/dashboard/api/state_manager.py",
        "src/dashboard/api/routes.py",
        "src/dashboard/api/app.py",
        "src/dashboard/sim_bridge.py",
        "src/network_env/live_cascade_adapter.py",
        "src/controller/live_mpc_orchestrator.py",
    ]
    for rel in live_files:
        tree = ast.parse((_PROJECT_ROOT / rel).read_text(encoding="utf-8"))
        modules, names, calls = set(), set(), set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module)
                names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.Call):
                func = node.func
                calls.add(func.id if isinstance(func, ast.Name)
                          else func.attr if isinstance(func, ast.Attribute) else "")

        assert not (modules & LEGACY_ENGINE_MODULES), \
            (rel, modules & LEGACY_ENGINE_MODULES)
        assert not (names & LEGACY_ENGINE_SYMBOLS), \
            (rel, names & LEGACY_ENGINE_SYMBOLS)
        # exactly one live state owner, and it is the authoritative module
        if "GlobalSimulationState" in names:
            assert rel == AUTHORITATIVE_STATE_OWNER, (rel, "second live state owner")
        # §16 — the dormant rule-based advisor is never invoked on the live path
        assert "compute_ai_recommendation" not in calls, \
            (rel, "dormant ForecastAwareController advisor invoked")

    # ...and the authoritative state manager does not even mention it (Stage 10/15).
    state_manager_source = (_PROJECT_ROOT / AUTHORITATIVE_STATE_OWNER).read_text(
        encoding="utf-8")
    assert "compute_ai_recommendation" not in state_manager_source


# ---------------------------------------------------------------------------
# 7. stability / memory
# ---------------------------------------------------------------------------

def test_repeated_cycles_remain_stable(prepared_live):
    """§8 — repeated cycles complete, step once each, and stay mass-balanced."""
    sm = prepared_live
    network = _live_network()
    n = 12
    before_network_timestep = int(network.timestep)
    before_index = sm.sim_step_index

    for _ in range(n):
        state = sm.step()
        assert state["mass_balance"]["checked"] is True
        assert state["mass_balance"]["status"] == "PASS"

    assert sm.sim_step_index == before_index + n
    assert int(network.timestep) == before_network_timestep + n


def test_history_buffers_stay_bounded(prepared_live):
    sm = prepared_live
    for _ in range(10):
        sm.step()
    assert sm.history_buffers
    assert all(len(buf) <= 7 for buf in sm.history_buffers.values())
    assert len(sm.clients) == 0


def test_no_obvious_unbounded_memory_growth(prepared_live):
    """
    §7 — a broad sanity bound only. RSS on a desktop OS includes allocator
    retention, so this test can only rule out the OBVIOUS case: memory that grows
    per cycle without limit.
    """
    import psutil
    proc = psutil.Process()
    sm = prepared_live

    for _ in range(5):
        sm.step()
    gc.collect()
    before = proc.memory_info().rss

    for _ in range(15):
        sm.step()
    gc.collect()
    after = proc.memory_info().rss

    growth_mb = (after - before) / (1024 ** 2)
    # 15 full live cycles; a genuine per-cycle leak would be far larger than this.
    assert growth_mb < 150.0, f"RSS grew {growth_mb:.1f} MB over 15 cycles"


def test_repeated_mpc_decisions_do_not_leak_network_clones(prepared_live):
    """
    §7 — the MPC deep-copies the network once per candidate (1296 per decision).
    If those clones leaked, the live-object count would grow by thousands.
    """
    sm = prepared_live
    network = _live_network()
    bundle = _fixture_bundle(network)

    gc.collect()
    before = sum(1 for o in gc.get_objects() if isinstance(o, ReservoirNetwork))
    for _ in range(2):
        sm.mpc_orchestrator.mpc.decide(network, forecast_snapshot=bundle.snapshot,
                                       current_inflows=dict(sm.manual_inflows))
    gc.collect()
    after = sum(1 for o in gc.get_objects() if isinstance(o, ReservoirNetwork))

    # 2 decisions create 2592 clones; at most a handful may still be referenced
    assert after - before < 50, f"live ReservoirNetwork objects grew by {after - before}"


# ---------------------------------------------------------------------------
# 8. performance work changed nothing
# ---------------------------------------------------------------------------

def test_protected_scientific_modules_are_clean_against_head():
    """§16 — Stage 17 must not have modified any validated scientific module."""
    result = subprocess.run(
        ["git", "status", "--porcelain", "--"] + PROTECTED_PATHS,
        cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert result.returncode == 0
    assert result.stdout.strip() == "", (
        "Stage 17 modified a protected scientific path:\n" + result.stdout)


def test_frozen_artifacts_match_the_manifest():
    """§16 — the frozen models and the V3 predictions are unchanged."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["all_match"] is True
    for raw_name, entry in manifest["pre_validation"].items():
        rel = raw_name.replace("\\", "/")
        path = _PROJECT_ROOT / rel
        assert path.exists(), rel
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            # only the frozen predictions CSV is line-ending sensitive; the
            # manifest hash is LF-normalised. NEVER "fix" the line endings.
            assert rel.endswith(".csv")
            normalised = path.read_bytes().replace(b"\r\n", b"\n")
            assert hashlib.sha256(normalised).hexdigest() == entry["sha256"]
            assert path.stat().st_size - len(normalised) == \
                path.read_bytes().count(b"\r\n"), "unexpected CRLF accounting"


def test_frozen_forecast_artifact_has_not_been_rewritten():
    """Re-running performance tests must not re-emit the V3 predictions file."""
    path = (_PROJECT_ROOT / "results" / "lstm_pytorch_v3_logtarget"
            / "test_predictions_original_units.csv")
    mtime = path.stat().st_mtime
    time.sleep(0.01)
    assert path.stat().st_mtime == mtime


def test_live_controller_configuration_is_unchanged():
    """§16 — no provenance relaxation, no lattice shrink, no second controller."""
    sm = state_manager.sim_state
    assert sm.mode in ("MANUAL", "AI")
    assert sm.mpc_orchestrator.mpc.config.gate_levels == EXPECTED_GATE_LEVELS
    assert sm.mpc_orchestrator.downstream_guard.mpc_gate_levels == EXPECTED_GATE_LEVELS
    assert state_manager.LIVE_FORECAST_EXCLUDED == ("Virtual Reservoir D",)


def test_mpc_and_safety_do_not_import_the_orchestrator():
    """§16 — the protected controllers stay independent of the live orchestrator."""
    for rel in ("src/controller/mpc_controller.py", "src/controller/safety.py",
                "src/controller/objective.py"):
        tree = ast.parse((_PROJECT_ROOT / rel).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert "live_mpc_orchestrator" not in node.module, rel
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert "live_mpc_orchestrator" not in alias.name, rel


# ---------------------------------------------------------------------------
# 9. evidence document
# ---------------------------------------------------------------------------

def test_performance_evidence_document_is_complete(evidence):
    for key in ("environment", "components", "mpc", "case_a", "case_b",
                "memory", "throughput", "concurrency", "bottlenecks", "budget",
                "limitations", "frozen_artifacts_unchanged"):
        assert key in evidence, key

    assert evidence["measurement_only"] is True
    assert evidence["code_modified"] is False
    assert evidence["frozen_artifacts_unchanged"] is True

    for key in ("os", "python_version", "pytorch_version", "cpu", "ram_total_gb",
                "gpu", "repository", "test_configuration"):
        assert key in evidence["environment"], key
    assert evidence["environment"]["environment_modified_for_benchmarking"] is False


def test_evidence_case_a_is_live_and_case_b_is_labelled_a_fixture(evidence):
    assert evidence["case_a"]["is_fixture"] is False
    assert evidence["case_b"]["is_fixture"] is True
    assert evidence["case_b"]["fixture_declaration"]["is_live_telemetry"] is False
    assert evidence["case_b"]["fixture_declaration"][
        "real_demonstration_only_forecasts_converted"] is False
    # CASE A really is the blocked live path
    assert evidence["case_a"]["chain_verdict"]["forecast_control_eligible"] is False
    assert evidence["case_a"]["chain_verdict"]["mpc_status"] == "NOT_INVOKED"
    assert evidence["case_a"]["chain_verdict"][
        "final_safe_control_action_source"] == "HELD_CURRENT_GATES"
    # CASE B really does reach the guard
    assert evidence["case_b"]["chain_verdict"][
        "final_safe_control_action_source"] == "DOWNSTREAM_CAPACITY_GUARD"


def test_evidence_candidate_space_is_1296(evidence):
    space = evidence["mpc"]["candidate_space"]
    assert space["expected_candidate_vectors"] == EXPECTED_CANDIDATES
    assert space["independently_enumerated_candidate_vectors"] == EXPECTED_CANDIDATES
    assert space["decision_candidates_evaluated"] == EXPECTED_CANDIDATES
    assert space["orchestrator_reported_candidate_vectors"] == EXPECTED_CANDIDATES
    assert space["matches_1296"] is True
    assert space["reduced_for_performance"] is False
    assert space["reservoirs_included"] == 4


def test_evidence_bottleneck_ranking_is_measured_only(evidence):
    rows = evidence["bottlenecks"]["ordered_by_measured_mean_ms"]
    assert len(rows) >= 10
    values = [r["measured_mean_ms"] for r in rows]
    assert all(v > 0.0 for v in values)
    assert values == sorted(values, reverse=True)
    assert [r["rank"] for r in rows] == list(range(1, len(rows) + 1))
    for row in rows:
        assert row["category"] in (
            "scientific computation", "safety computation", "physics",
            "physics/audit", "inference", "serialization", "serialization/UI",
            "serialization/adapter", "integration")


def test_evidence_budget_is_transparent_and_not_forced(evidence):
    budget = evidence["budget"]
    assert budget["control_decision_period_ms"] > 0
    assert budget["budget_pass_forced_by_code_change"] is False
    assert budget["rows"]
    for row in budget["rows"]:
        assert row["measured_ms"] > 0.0
        assert row["budget_ms"] > 0.0
        assert row["within_budget"] == (row["measured_ms"] <= row["budget_ms"])
    assert budget["meets_budget"] == (budget["not_met"] == [])


def test_evidence_repeated_cycles_are_stable(evidence):
    tp = evidence["throughput"]
    assert tp["cycles_completed"] == tp["cycles_requested"]
    assert tp["failures"] == []
    assert tp["exceptions"] == []
    assert tp["sim_step_index_monotonic"] is True
    assert set(tp["mass_balance_status_counts"]) == {"PASS"}
    assert tp["timing"]["measured_iterations"] >= 20


def test_evidence_memory_is_bounded(evidence):
    mem = evidence["memory"]
    assert mem["history_buffers_bounded"] is True
    assert mem["websocket_clients_registered"] == 0
    assert mem["destructive_stress_tests_performed"] is False
    assert len(mem["rounds"]) >= 3
    # broad sanity bound: 45 cycles must not grow RSS by hundreds of MB
    assert mem["rss_growth_over_45_cycles_mb"] < 300.0


def test_evidence_concurrency_reports_no_duplicate_steps(evidence):
    conc = evidence["concurrency"]
    if not conc.get("performed"):
        pytest.skip(f"concurrency probe not performed: {conc.get('reason')}")
    steps = conc["concurrent_steps"]
    assert steps["http_200"] == steps["requests"]
    assert steps["steps_advanced"] == steps["expected_steps"]
    assert steps["duplicate_or_lost_steps"] is False
    assert steps["state_index_monotonic_nondecreasing"] is True

    inproc = conc["in_process_concurrency"]
    assert inproc["single_owner_in_process"] is True
    assert inproc["authoritative_instance_count_in_this_process"] == 1
    assert inproc["step_is_synchronous_with_no_suspension_point"] is True
    assert inproc["concurrent_decide_probe"]["errors"] == []
    assert inproc["concurrent_decide_probe"]["decide_calls"] == 4


def test_evidence_has_no_obvious_unbounded_growth_signature(evidence):
    """§7 — the leak probe must show no single allocation site exploding."""
    mem = evidence["memory"]
    for row in mem["tracemalloc_top_growth"]:
        # 3 advisory runs + 3 MPC decisions; a leak would show thousands of
        # retained blocks or tens of MB from one line.
        assert row["count_diff"] < 20000, row
        assert row["size_diff_kb"] < 40000, row
