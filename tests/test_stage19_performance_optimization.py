"""
STAGE 19 — PERFORMANCE OPTIMISATION OF THE AUTO-CONTROL PIPELINE.

This suite is the GOLDEN EQUIVALENCE NET for a purely mechanical optimisation.
No control policy is changed here: same candidate space, same candidate costs,
same selected action, same safety result, same applied gates — less computation.

P1  ONE rollout clone per MPC decision, reused across all 1,296 candidates
    (the pre-optimisation algorithm rebuilt a `ReservoirNetwork` per candidate;
    that algorithm is re-implemented INDEPENDENTLY in this file as the
    reference, so the two implementations are compared rather than assumed).
P2  Hypothetical rollout logging silenced (`emit_warnings=False`) while the
    authoritative network keeps logging real exceedances.
P3  Per-authoritative-step memoisation of the DISPLAY forecast/GNN-advisory
    work in `GlobalSimulationState.get_adapted_state()`.
P4  Deadline-based `simulation_loop()` scheduling (no cadence drift, no
    catch-up bursts).

The reference rollout below is deliberately written in the PRE-OPTIMISATION
style (fresh clone per candidate) so that it keeps proving equivalence even
after the production code has been refactored.
"""

import asyncio
import contextlib
import copy
import io
import itertools
import logging
import sys
import threading
import time
from collections import deque
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.controller.mpc_controller import MPCController, MPCConfig  # noqa: E402
from src.controller.objective import ObjectiveFunction, ObjectiveWeights  # noqa: E402
from src.controller.safety import SafetyLayer  # noqa: E402
from src.dashboard.api import state_manager  # noqa: E402
from src.dashboard.api.app import app  # noqa: E402
from src.dashboard.sim_bridge import SimBridge  # noqa: E402
from src.network_env.live_forecast_adapter import LiveForecastAdapter  # noqa: E402
from src.network_env.reservoir_network import ReservoirNetwork  # noqa: E402

client = TestClient(app)

PROJECT_ROOT = str(_PROJECT_ROOT)
LIVE_CONFIG = _PROJECT_ROOT / "configs" / "simulation" / "four_reservoir_demo.json"
THRESH_PATH = _PROJECT_ROOT / "data" / "processed" / "historical_inflow_thresholds.json"

NODES = [
    "Virtual Reservoir A",
    "Virtual Reservoir B",
    "Virtual Reservoir C",
    "Virtual Reservoir D",
]
A, B, C, D = NODES

#: Capacity (MCM) per node, from the frozen demo topology.
CAPACITY = {A: 10.82, B: 21.26, C: 348.29, D: 401.43}

#: The MPC lattice the search covers — 6^4 = 1,296 candidates.
GATE_LEVELS = [0.0, 0.15, 0.30, 0.50, 0.70, 1.0]
EXPECTED_CANDIDATES = 6 ** 4


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _fresh_network() -> ReservoirNetwork:
    return SimBridge(str(LIVE_CONFIG), str(THRESH_PATH)).cascade.network


def _payload(v1: float, v3: float, v7: float) -> dict:
    """A forecast payload that satisfies the Stage 7 provenance gate."""
    return {
        "forecast_1d": v1, "forecast_3d": v3, "forecast_7d": v7,
        "forecast_status": "VALIDATED",
        "forecast_provenance": "REAL_MEASUREMENT_INPUTS_FROZEN_MODEL",
        "forecast_source": "FROZEN_LSTM_V3",
        "is_simulated": False, "validated_metrics_apply": True,
        "forecast_unit": "MCM/day",
        "horizons": ["forecast_1d", "forecast_3d", "forecast_7d"],
        "input_provenance": {"synthetic_demo": [], "unavailable": [], "simulated": []},
    }


def _bundle(network: ReservoirNetwork, values=(12.0, 11.5, 10.0)):
    adapter = LiveForecastAdapter.for_network(network, project_root=PROJECT_ROOT)
    payloads = {nid: _payload(*values) for nid in NODES}
    return adapter.build_bundle(payloads, "2026-09-14")


def _set_state(network, storages, gates, queues=None):
    """Place the network in a specific authoritative condition."""
    for nid in NODES:
        network.nodes[nid].state.storage = float(storages[nid])
        network.nodes[nid].state.gate_position = float(gates[nid])
    network.timestep = 7
    for index, conn in enumerate(network.connections):
        values = (queues or {}).get(index, [0.0] * int(conn.delay))
        conn.queue = deque([float(v) for v in values], maxlen=max(int(conn.delay), 1))
    return network


def _fingerprint(network) -> dict:
    """Every mutable bit of the authoritative network, for leak detection."""
    return {
        "timestep": int(network.timestep),
        "storage": {nid: float(network.nodes[nid].state.storage) for nid in NODES},
        "gates": {nid: float(network.nodes[nid].state.gate_position) for nid in NODES},
        "spill": {nid: float(network.nodes[nid].state.spill) for nid in NODES},
        "release": {nid: float(network.nodes[nid].state.controlled_release) for nid in NODES},
        "overflow_count": {nid: int(network.nodes[nid]._overflow_count) for nid in NODES},
        "cumulative_spill": {nid: float(network.nodes[nid]._cumulative_spill) for nid in NODES},
        "queues": [list(conn.queue) for conn in network.connections],
        "queue_maxlen": [int(conn.queue.maxlen or 0) for conn in network.connections],
        "totals": (
            float(network._total_external_inflow),
            float(network._total_routing_loss),
            float(network._total_terminal_outflow),
            float(network._total_nonterminal_spill),
        ),
    }


# ---------------------------------------------------------------------------
# fixture states — including the high-storm R3 scenario from the diagnostic
# ---------------------------------------------------------------------------

def _fixture_baseline():
    """Fresh cascade, closed gates, storm = 0 inflows."""
    network = _fresh_network()
    _set_state(
        network,
        storages={nid: network.nodes[nid].state.storage for nid in NODES},
        gates={nid: 0.0 for nid in NODES},
    )
    inflows = {A: 3.0, B: 4.0, C: 90.0, D: 0.0}
    return network, inflows


def _fixture_rising():
    """Reservoirs filling, partial gates, water already in transit."""
    network = _fresh_network()
    _set_state(
        network,
        storages={A: 0.90 * CAPACITY[A], B: 0.90 * CAPACITY[B],
                  C: 0.90 * CAPACITY[C], D: 0.60 * CAPACITY[D]},
        gates={A: 1.0, B: 0.7, C: 0.3, D: 0.15},
        queues={0: [4.5, 4.2], 1: [9.0], 2: [60.0]},
    )
    inflows = {A: 6.0, B: 8.0, C: 150.0, D: 0.0}
    return network, inflows


def _fixture_high_storm_r3():
    """
    The diagnostic's steady state: storm = 70 %.

    R3 (Virtual Reservoir C) at 100 % storage with a 50 % gate, R4 passing
    exactly the 50 MCM/day the river allows. R3 must NOT be forced open by any
    optimisation: the same action has to come out of both implementations.
    """
    network = _fresh_network()
    _set_state(
        network,
        storages={A: CAPACITY[A], B: CAPACITY[B], C: CAPACITY[C],
                  D: 0.744 * CAPACITY[D]},
        gates={A: 1.0, B: 1.0, C: 0.5, D: 0.25},
        queues={0: [10.0, 10.0], 1: [10.0], 2: [105.0]},
    )
    inflows = {A: 9.3, B: 12.4, C: 279.0, D: 0.0}
    return network, inflows


FIXTURES = {
    "baseline": _fixture_baseline,
    "rising": _fixture_rising,
    "high_storm_r3": _fixture_high_storm_r3,
}


# ---------------------------------------------------------------------------
# the REFERENCE implementation — the pre-optimisation algorithm, frozen here
# ---------------------------------------------------------------------------

def _reference_candidate_costs(network, forecast_snapshot, current_inflows,
                               node_ids=None):
    """
    The PRE-OPTIMISATION grid search, re-implemented independently.

    A BRAND NEW `ReservoirNetwork` is constructed for every candidate — exactly
    what `MPCController._simulate_trajectory` used to do. Nothing in this
    function calls the production rollout code, so it cannot drift with it.

    Returns
    -------
    dict with:
        candidates  : list of candidate gate vectors, in product order
        costs       : matching total_cost, in product order
        best_index  : index of the winning candidate (strict `<`, first wins ties)
        best_gates  : {node_id: gate} of the winning candidate
        best_cost   : objective value of the winning candidate
        trajectories: {index: [[(storage, release, spill) per node] per step]}
    """
    node_ids = list(node_ids or network.processing_order)
    mpc = MPCController()
    objective = ObjectiveFunction()
    capacities = {nid: network.nodes[nid].capacity for nid in node_ids}
    current_gates = {nid: network.nodes[nid].state.gate_position for nid in node_ids}
    scenarios = mpc._build_inflow_scenarios(node_ids, current_inflows, forecast_snapshot)

    candidates, costs, trajectories = [], [], {}
    last = len(GATE_LEVELS) ** len(node_ids) - 1
    sampled = {0, 1, last // 2, last}

    for index, combo in enumerate(itertools.product(GATE_LEVELS, repeat=len(node_ids))):
        candidate = {nid: gate for nid, gate in zip(node_ids, combo)}

        # ---- PRE-OPTIMISATION ROLLOUT: one fresh clone per candidate ----
        clone = ReservoirNetwork(config_dict=copy.deepcopy(network._raw_config))
        for nid in node_ids:
            clone.nodes[nid].state.storage = float(network.nodes[nid].state.storage)
        for i, conn in enumerate(network.connections):
            clone.connections[i].queue = copy.deepcopy(conn.queue)

        trajectory, capture = [], []
        for step_inflows in scenarios:
            states = clone.step(dict(step_inflows), dict(candidate))
            trajectory.append(states)
            capture.append({
                nid: (float(states[nid].storage), float(states[nid].controlled_release),
                      float(states[nid].spill), float(states[nid].total_outflow))
                for nid in node_ids
            })

        result = objective.evaluate_trajectory(
            trajectory, capacities, network.downstream_capacity,
            network._terminal_node_id, previous_gates=dict(current_gates),
        )
        candidates.append(candidate)
        costs.append(float(result["total_cost"]))
        if index in sampled:
            trajectories[index] = capture

    best_index = 0
    best_cost = float("inf")
    for index, cost in enumerate(costs):
        if cost < best_cost:                      # strict `<` — first wins ties
            best_cost = cost
            best_index = index

    return {
        "candidates": candidates,
        "costs": costs,
        "best_index": best_index,
        "best_gates": candidates[best_index],   # raw arg-min, before SafetyLayer
        "best_cost": best_cost,
        "trajectories": trajectories,
    }


def _record_costs(monkeypatch, mpc):
    """
    Record what the PRODUCTION rollout actually evaluated: candidate vector,
    cost, and the trajectory of every candidate.
    """
    rows = []
    original = mpc.objective.evaluate_trajectory

    def recording(trajectory, capacities, downstream_capacity, terminal_node_id,
                  previous_gates=None):
        result = original(trajectory, capacities, downstream_capacity,
                          terminal_node_id, previous_gates)
        candidate = {nid: trajectory[0][nid].gate_position for nid in trajectory[0]}
        capture = [
            {nid: (float(states[nid].storage), float(states[nid].controlled_release),
                   float(states[nid].spill), float(states[nid].total_outflow))
             for nid in states}
            for states in trajectory
        ]
        rows.append((candidate, float(result["total_cost"]), capture))
        return result

    monkeypatch.setattr(mpc.objective, "evaluate_trajectory", recording)
    return rows


# ===========================================================================
# P1 — GOLDEN EQUIVALENCE OF THE REUSED ROLLOUT CLONE
# ===========================================================================

@pytest.mark.parametrize("fixture", list(FIXTURES))
def test_p1_same_candidate_space_costs_action_and_objective(fixture, monkeypatch):
    """
    The optimised rollout must reproduce the reference EXACTLY:

    1. same candidate space (all 1,296 vectors, same product order)
    2. same candidate costs, elementwise
    3. same selected raw MPC action (trajectory arg-min, fed to SafetyLayer),
       same validated gates, and same objective value
    """
    network, inflows = FIXTURES[fixture]()
    snapshot = _bundle(network)

    reference = _reference_candidate_costs(network, snapshot, inflows)

    mpc = MPCController()
    rows = _record_costs(monkeypatch, mpc)
    decision = mpc.decide(network, forecast_snapshot=snapshot, current_inflows=inflows)

    # ---- 1. same candidate space, same order
    assert decision.candidates_evaluated == EXPECTED_CANDIDATES
    assert len(rows) == EXPECTED_CANDIDATES
    assert len(reference["candidates"]) == EXPECTED_CANDIDATES
    for (candidate, _cost, _capture), expected in zip(rows, reference["candidates"]):
        assert candidate == pytest.approx(expected)

    # ---- 2. same candidate costs
    for index, (_candidate, cost, _capture) in enumerate(rows):
        assert cost == pytest.approx(reference["costs"][index], rel=1e-12, abs=1e-12), index

    # ---- 3. `decide()` returned the trajectory arg-min: recompute the
    # arg-min from the RECORDED production rows with the same strict `<`
    # scan and require it to equal the reference arg-min; the production
    # `gate_positions` must equal the SafetyLayer's output for that arg-min.
    raw_best, raw_cost = rows[0][0], rows[0][1]
    for candidate, cost, _capture in rows[1:]:
        if cost < raw_cost:
            raw_best, raw_cost = candidate, cost
    assert raw_best == pytest.approx(reference["best_gates"])
    assert raw_cost == pytest.approx(reference["best_cost"], rel=1e-12, abs=1e-12)
    assert decision.objective_score == pytest.approx(raw_cost, rel=1e-12, abs=1e-12)
    current = {nid: network.nodes[nid].state.gate_position for nid in NODES}
    expected_validated = dict(
        SafetyLayer(max_gate_change_per_step=0.5)
        .validate(dict(raw_best), current, list(NODES))
        .validated_gates)
    assert dict(decision.gate_positions) == pytest.approx(expected_validated)

@pytest.mark.parametrize("fixture", list(FIXTURES))
def test_p1_same_trajectory_result_for_sampled_candidates(fixture, monkeypatch):
    """5. same per-step trajectory (storage, release, spill, outflow) per node."""
    network, inflows = FIXTURES[fixture]()
    snapshot = _bundle(network)
    reference = _reference_candidate_costs(network, snapshot, inflows)

    mpc = MPCController()
    rows = _record_costs(monkeypatch, mpc)
    mpc.decide(network, forecast_snapshot=snapshot, current_inflows=inflows)

    for index, expected in reference["trajectories"].items():
        _candidate, _cost, capture = rows[index]
        assert len(capture) == len(expected)
        for step_index, step in enumerate(expected):
            for nid in NODES:
                assert capture[step_index][nid] == pytest.approx(step[nid], rel=1e-12), (
                    index, step_index, nid)


@pytest.mark.parametrize("fixture", list(FIXTURES))
def test_p1_no_rollout_state_leaks_into_the_authoritative_network(fixture):
    """7. the live network is bit-identical before and after a decision."""
    network, inflows = FIXTURES[fixture]()
    snapshot = _bundle(network)
    before = _fingerprint(network)

    decision = MPCController().decide(network, forecast_snapshot=snapshot,
                                      current_inflows=inflows)

    after = _fingerprint(network)
    assert after == before
    assert decision.gate_positions is not None


def test_p1_tie_breaking_is_unchanged(monkeypatch):
    """
    6. same tie-breaking: with every candidate equally costly, the FIRST
    candidate in product order wins (the selection is a strict `<` scan).
    """
    network, inflows = _fixture_high_storm_r3()
    snapshot = _bundle(network)

    mpc = MPCController()
    original = mpc.objective.evaluate_trajectory
    rows = []

    def flat_cost(trajectory, capacities, downstream_capacity, terminal_node_id,
                  previous_gates=None):
        result = original(trajectory, capacities, downstream_capacity,
                          terminal_node_id, previous_gates)
        result["total_cost"] = 1.0            # every candidate ties
        candidate = {nid: trajectory[0][nid].gate_position for nid in trajectory[0]}
        rows.append(candidate)
        return result

    monkeypatch.setattr(mpc.objective, "evaluate_trajectory", flat_cost)
    mpc.decide(network, forecast_snapshot=snapshot, current_inflows=inflows)

    # The objective saw all 1,296 candidates in product order, so the first
    # row is the first combo; with a strict `<` scan that row must be the
    # arg-min the controller reports.
    assert len(rows) == EXPECTED_CANDIDATES
    first = {nid: GATE_LEVELS[0] for nid in NODES}
    assert rows[0] == first
    # Under the flat cost, `decide()` must pick the FIRST candidate: re-run
    # and check the score plus the safety-consistent output.
    decision = mpc.decide(network, forecast_snapshot=snapshot, current_inflows=inflows)
    assert decision.objective_score == pytest.approx(1.0)
    # The controller's raw arg-min under the flat cost is the first combo;
    # the SafetyLayer may only rate-limit it from the current gates.
    current = {nid: network.nodes[nid].state.gate_position for nid in NODES}
    expected = dict(SafetyLayer(max_gate_change_per_step=0.5)
                    .validate(dict(first), current, list(NODES)).validated_gates)
    assert dict(decision.gate_positions) == pytest.approx(expected)

def test_p1_decide_is_deterministic():
    """The same input yields the same decision twice — no cross-call state."""
    network, inflows = _fixture_high_storm_r3()
    snapshot = _bundle(network)
    mpc = MPCController()
    first = mpc.decide(network, forecast_snapshot=snapshot, current_inflows=inflows)
    second = mpc.decide(network, forecast_snapshot=snapshot, current_inflows=inflows)
    assert first.gate_positions == second.gate_positions
    assert first.objective_score == pytest.approx(second.objective_score)
    assert first.candidates_evaluated == second.candidates_evaluated


def test_p1_candidate_count_is_not_reduced():
    """The search space must still be the full 6^4 lattice."""
    mpc = MPCController()
    assert list(mpc.config.gate_levels) == GATE_LEVELS
    assert mpc.config.lookahead_steps == 3
    assert (len(list(itertools.product(GATE_LEVELS, repeat=len(NODES))))
            == EXPECTED_CANDIDATES)


# ===========================================================================
# P2 — HYPOTHETICAL ROLLOUT LOGGING IS SILENCED, LIVE LOGGING SURVIVES
# ===========================================================================

class _RecordCounter(logging.Handler):
    """Count emitted log records by message prefix (in-memory, read-only)."""

    def __init__(self):
        super().__init__()
        self.counts = {}

    def emit(self, record):
        key = record.getMessage().split(" ")[0][:40]
        self.counts[key] = self.counts.get(key, 0) + 1


def _overflow_counter_handler():
    handler = _RecordCounter()
    net_logger = logging.getLogger("src.network_env.reservoir_network")
    previous_level = net_logger.level
    net_logger.addHandler(handler)
    net_logger.setLevel(logging.WARNING)
    return handler, net_logger, previous_level


def _restore_logger(net_logger, handler, previous_level):
    net_logger.removeHandler(handler)
    net_logger.setLevel(previous_level)


@pytest.mark.parametrize("fixture", list(FIXTURES))
def test_p2_rollout_does_not_emit_hypothetical_warnings(fixture):
    """
    A. a full 1,296-candidate MPC grid search emits NO network warnings,
    while recording the SAME selected action as the (logging) reference.
    """
    network, inflows = FIXTURES[fixture]()
    snapshot = _bundle(network)
    reference = _reference_candidate_costs(network, snapshot, inflows)

    handler, net_logger, previous_level = _overflow_counter_handler()
    try:
        decision = MPCController().decide(network, forecast_snapshot=snapshot,
                                          current_inflows=inflows)
    finally:
        _restore_logger(net_logger, handler, previous_level)

    total = sum(handler.counts.values())
    assert total == 0, handler.counts

    # The silenced rollout still selects the reference action and cost.
    recomputed, best = reference["candidates"][0], reference["costs"][0]
    for candidate, cost in zip(reference["candidates"][1:], reference["costs"][1:]):
        if cost < best:
            recomputed, best = candidate, cost
    assert recomputed == pytest.approx(reference["best_gates"])
    current = {nid: network.nodes[nid].state.gate_position for nid in NODES}
    expected_validated = dict(
        SafetyLayer(max_gate_change_per_step=0.5)
        .validate(dict(recomputed), current, list(NODES)).validated_gates)
    assert dict(decision.gate_positions) == pytest.approx(expected_validated)
    assert decision.objective_score == pytest.approx(best, rel=1e-12, abs=1e-12)


def test_p2_quiet_clone_physics_identical_to_loud_clone():
    """The flag gates emission only — storage, spill, queues all identical."""
    network, _ = _fixture_high_storm_r3()
    gates = {A: 0.7, B: 0.3, C: 1.0, D: 0.15}
    inflows = {A: 9.3, B: 12.4, C: 279.0, D: 0.0}

    loud = ReservoirNetwork(config_dict=copy.deepcopy(network._raw_config))
    quiet = ReservoirNetwork(config_dict=copy.deepcopy(network._raw_config),
                             emit_warnings=False)
    assert loud._emit_warnings is True
    assert quiet._emit_warnings is False
    assert all(node.emit_warnings is True for node in loud.nodes.values())
    assert all(node.emit_warnings is False for node in quiet.nodes.values())

    out = {}
    for label, clone in (("loud", loud), ("quiet", quiet)):
        for nid in NODES:
            clone.nodes[nid].state.storage = float(network.nodes[nid].state.storage)
        for i, conn in enumerate(network.connections):
            clone.connections[i].queue = copy.deepcopy(conn.queue)
        states = []
        for _ in range(3):
            states.append(clone.step(dict(inflows), dict(gates)))
        out[label] = [
            {nid: (s.storage, s.controlled_release, s.spill, s.total_outflow,
                   s.gate_position)
             for nid, s in step_states.items()}
            for step_states in states
        ]

    assert out["quiet"] == out["loud"]
    spill_loud = {nid: clone.nodes[nid]._cumulative_spill for nid in NODES
                  for clone in (loud,)}
    spill_quiet = {nid: clone.nodes[nid]._cumulative_spill for nid in NODES
                   for clone in (quiet,)}
    assert spill_loud == spill_quiet
    overflow_loud = {nid: clone.nodes[nid]._overflow_count for nid in NODES
                     for clone in (loud,)}
    overflow_quiet = {nid: clone.nodes[nid]._overflow_count for nid in NODES
                      for clone in (quiet,)}
    assert overflow_loud == overflow_quiet


def test_p2_live_network_still_warns():
    """
    B. the REAL authoritative network keeps logging real exceedances.

    A live network at full storage with a storm inflow overflows on a REAL
    step and MUST emit the OVERFLOW record (the P1/P2 path never touches this).
    """
    network = _fresh_network()
    for nid in NODES:
        network.nodes[nid].state.storage = float(network.nodes[nid].capacity)
    network.nodes[D].state.gate_position = 0.0

    handler, net_logger, previous_level = _overflow_counter_handler()
    try:
        network.step({A: 3.0, B: 4.0, C: 90.0, D: 0.0},
                     {nid: 0.0 for nid in NODES})
    finally:
        _restore_logger(net_logger, handler, previous_level)

    assert sum(handler.counts.values()) > 0, "live warnings were silenced"
    assert any("OVERFLOW" in key.replace("[", "").replace("]", "")
               or key.startswith("[Virtual") for key in handler.counts)


# ===========================================================================
# P3 — PER-STEP MEMOISATION OF THE DISPLAY FORECAST/ADVISORY PIPELINE
# ===========================================================================

def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


#: Fields produced by WHEN something ran rather than by WHAT it computed:
#: wall-clock stamps and measured durations. They legitimately differ between
#: two reads of identical authoritative state, so equivalence checks exclude
#: them. (`state_id` is excluded as the payload's own identity field.)
VOLATILE_KEYS = (
    "simulation_time", "timestamp", "state_id",
    "inference_timestamp", "inference_latency_ms",
)


def _strip_volatile(payload):
    """Deep copy with wall-clock/measured fields removed (not pipeline output)."""
    if isinstance(payload, dict):
        return {
            key: _strip_volatile(value)
            for key, value in payload.items()
            if key not in VOLATILE_KEYS
        }
    if isinstance(payload, list):
        return [_strip_volatile(item) for item in payload]
    return payload


def _pipeline_counter(sim, monkeypatch):
    """Count real `_run_ml_pipeline()` executions (control AND display paths)."""
    calls = {"n": 0}
    original = sim._run_ml_pipeline

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(sim, "_run_ml_pipeline", counting)
    return calls


@pytest.fixture(autouse=True)
def _restore_authoritative_state():
    """This module drives the ONE authoritative singleton; restore it."""
    yield
    sim = state_manager.sim_state
    sim.running = False
    _quiet(client.post, "/api/simulation/pause")
    _quiet(client.post, "/api/controller/mode",
           json={"mode": "MANUAL", "source": "SIMULATION"})
    _quiet(client.post, "/api/storm", json={"value": 0.0})
    _quiet(client.post, "/api/simulation/reset")
    sim._invalidate_pipeline_cache()


def test_p3_repeated_reads_are_served_from_the_cache(monkeypatch):
    """One computation per authoritative state, then memoised reads."""
    sim = state_manager.sim_state
    _quiet(client.post, "/api/simulation/pause")
    _quiet(client.post, "/api/controller/mode",
           json={"mode": "MANUAL", "source": "SIMULATION"})
    _quiet(client.post, "/api/simulation/reset")

    calls = _pipeline_counter(sim, monkeypatch)
    first = client.get("/api/state").json()
    assert calls["n"] == 1
    second = client.get("/api/state").json()
    third = client.get("/api/state").json()
    assert calls["n"] == 1, "repeated reads must not re-run the pipeline"
    # The cache is in sync with the authoritative fingerprint — never stale.
    assert sim._pipeline_cache_key == sim._display_pipeline_cache_key()
    assert _strip_volatile(first) == _strip_volatile(second) == _strip_volatile(third)


def test_p3_cached_payload_equals_uncached_payload(monkeypatch):
    """1. cached payload == uncached payload for identical authoritative state."""
    sim = state_manager.sim_state
    _quiet(client.post, "/api/simulation/pause")
    _quiet(client.post, "/api/controller/mode",
           json={"mode": "MANUAL", "source": "SIMULATION"})
    _quiet(client.post, "/api/simulation/reset")

    calls = _pipeline_counter(sim, monkeypatch)
    cached = client.get("/api/state").json()
    assert calls["n"] == 1
    client.get("/api/state")                # served from cache
    assert calls["n"] == 1

    sim._invalidate_pipeline_cache()
    uncached = client.get("/api/state").json()
    assert calls["n"] == 2

    assert _strip_volatile(cached) == _strip_volatile(uncached)
    # The forecast/GNN-dependent subtrees must be identical.
    # Recomputing inference changes these measurements, not scientific results.
    volatile_inference = {"inference_timestamp", "inference_latency_ms"}
    assert {k: v for k, v in cached["gnn_advisory"].items() if k not in volatile_inference} == {
        k: v for k, v in uncached["gnn_advisory"].items() if k not in volatile_inference
    }
    assert cached["forecast_provenance"] == uncached["forecast_provenance"]
    assert cached["forecast_summary"] == uncached["forecast_summary"]
    for key in ("reservoir_1", "reservoir_2", "reservoir_3", "reservoir_4"):
        for horizon in ("forecast_1d", "forecast_3d", "forecast_7d"):
            assert (cached["reservoirs"][key][horizon]
                    == uncached["reservoirs"][key][horizon])


def test_p3_cache_does_not_survive_a_reset():
    """2. RESET starts a new run: the memoisation is dropped, not reused."""
    sim = state_manager.sim_state
    _quiet(client.post, "/api/simulation/pause")
    _quiet(client.post, "/api/controller/mode",
           json={"mode": "MANUAL", "source": "SIMULATION"})
    _quiet(client.post, "/api/simulation/reset")
    _quiet(client.post, "/api/simulation/step")  # exercise reset after actual advancement
    step_before_reset = sim.sim_step_index
    network_before_reset = sim.bridge.cascade.network
    client.get("/api/state")                    # warm the cache
    assert sim._pipeline_cache_key is not None

    _quiet(client.post, "/api/simulation/reset")
    assert sim._pipeline_cache_key is None
    assert sim._pipeline_cache is None

    after = client.get("/api/state").json()
    assert sim._pipeline_cache_key == sim._display_pipeline_cache_key()
    assert sim.running is False
    assert sim.bridge.cascade.network is not network_before_reset
    assert sim.sim_step_index == step_before_reset
    assert after["state_identity"]["network_timestep"] == 0
    assert after["state_identity"]["state_id"] == f"step{step_before_reset}-t0"


def test_p3_changing_storm_invalidates_the_cache(monkeypatch):
    """3. a storm change is an authoritative input change."""
    sim = state_manager.sim_state
    _quiet(client.post, "/api/simulation/pause")
    _quiet(client.post, "/api/controller/mode",
           json={"mode": "MANUAL", "source": "SIMULATION"})
    _quiet(client.post, "/api/simulation/reset")

    calls = _pipeline_counter(sim, monkeypatch)
    client.get("/api/state")
    client.get("/api/state")
    assert calls["n"] == 1

    _quiet(client.post, "/api/storm", json={"value": 0.5})
    before = client.get("/api/state").json()
    assert calls["n"] == 2, "storm change must recompute"
    assert sim._pipeline_cache_key == sim._display_pipeline_cache_key()

    _quiet(client.post, "/api/storm", json={"value": 0.0})
    after = client.get("/api/state").json()
    assert calls["n"] == 3
    assert before["storm_intensity"] == pytest.approx(0.5)
    assert after["storm_intensity"] == pytest.approx(0.0)
    assert _strip_volatile(before) != _strip_volatile(after)


def test_p3_changing_forecast_source_invalidates_the_cache(monkeypatch):
    """4. a source change is a different pipeline; never serve the old one."""
    sim = state_manager.sim_state
    _quiet(client.post, "/api/simulation/pause")
    _quiet(client.post, "/api/controller/mode",
           json={"mode": "MANUAL", "source": "SIMULATION"})
    _quiet(client.post, "/api/simulation/reset")

    calls = _pipeline_counter(sim, monkeypatch)
    client.get("/api/state")
    client.get("/api/state")
    assert calls["n"] == 1

    _quiet(client.post, "/api/controller/mode",
           json={"mode": "MANUAL", "source": "VALIDATED_REPLAY"})
    replay = client.get("/api/state").json()
    assert calls["n"] == 2, "forecast source change must recompute"
    assert replay["forecast_source_selection"]["selected"] == "VALIDATED_REPLAY"
    assert replay["forecast_source_selection"]["control_forecast_validated"] is True
    assert sim._pipeline_cache_key == sim._display_pipeline_cache_key()

    _quiet(client.post, "/api/controller/mode",
           json={"mode": "MANUAL", "source": "SIMULATION"})
    live = client.get("/api/state").json()
    assert calls["n"] == 3
    assert live["forecast_source_selection"]["selected"] == "SIMULATION"
    assert live["forecast_source_selection"]["control_forecast_validated"] is False
    assert _strip_volatile(replay) != _strip_volatile(live)


def test_p3_control_path_recomputes_and_never_reads_the_display_cache(monkeypatch):
    """
    A step must run the CONTROL pipeline itself: warming the display cache
    must not short-circuit the forecast the provenance gate consumes.
    """
    sim = state_manager.sim_state
    _quiet(client.post, "/api/simulation/pause")
    _quiet(client.post, "/api/controller/mode",
           json={"mode": "MANUAL", "source": "SIMULATION"})
    _quiet(client.post, "/api/simulation/reset")

    calls = _pipeline_counter(sim, monkeypatch)
    client.get("/api/state")
    client.get("/api/state")
    assert calls["n"] == 1, "precondition: display reads are memoised"

    before = calls["n"]
    _quiet(client.post, "/api/simulation/step")
    # `step()` itself calls `_run_ml_pipeline()` for CONTROL, and the state
    # changed, so the display path recomputes too.
    assert calls["n"] > before, "the control path must not consume the UI cache"
    assert sim._pipeline_cache_key == sim._display_pipeline_cache_key()


def _network_gate_pct() -> dict:
    from src.common import units as _units
    network = state_manager.sim_state.bridge.cascade.network
    return {
        str(nid): float(_units.gate_fraction_to_percent(
            network.nodes[nid].state.gate_position))
        for nid in network.processing_order
    }


def _kinds(events) -> list:
    return [str(e.get("kind")) for e in events or []]


def test_p3_simulation_still_fails_closed_with_a_warm_cache():
    """
    5. SIMULATION -> ineligible for AUTO control -> BLOCKED -> gates HELD.

    The memoisation must not soften the Stage 7 provenance gate.
    """
    sim = state_manager.sim_state
    _quiet(client.post, "/api/simulation/pause")
    _quiet(client.post, "/api/controller/mode",
           json={"mode": "MANUAL", "source": "SIMULATION"})
    _quiet(client.post, "/api/simulation/reset")

    for _ in range(8):                    # warm the SIMULATION forecast window
        _quiet(client.post, "/api/simulation/step")
    _quiet(client.post, "/api/controller/mode", json={"mode": "AI"})

    client.get("/api/state")              # display cache warm (SIMULATION)
    client.get("/api/state")
    assert sim._pipeline_cache_key is not None

    before = _network_gate_pct()
    _quiet(client.post, "/api/simulation/step")
    after = _network_gate_pct()

    state = client.get("/api/state").json()
    control = state["control"]
    assert control["forecast_control_eligible"] is False
    assert control["controller_status"] == "BLOCKED"
    assert control["blocked_reason"] == "FORECAST_NOT_ELIGIBLE_FOR_CONTROL"
    assert control["final_safe_control_action_source"] == "HELD_CURRENT_GATES"
    assert state["auto_control"]["state"] == "AUTO_BLOCKED"
    assert state["forecast_source_selection"]["selected"] == "SIMULATION"
    for nid in NODES:
        assert after[nid] == pytest.approx(before[nid], abs=1e-9), nid
    assert _kinds(state["event_log"])[-1] == "control_blocked"


def test_p3_validated_replay_still_controls_with_a_warm_cache():
    """6. VALIDATED_REPLAY -> eligible -> ACTIVE -> gates really move."""
    sim = state_manager.sim_state
    _quiet(client.post, "/api/simulation/pause")
    _quiet(client.post, "/api/controller/mode",
           json={"mode": "MANUAL", "source": "SIMULATION"})
    _quiet(client.post, "/api/simulation/reset")

    _quiet(client.post, "/api/controller/mode",
           json={"mode": "AI", "source": "VALIDATED_REPLAY"})
    client.get("/api/state")              # display cache warm (replay)
    client.get("/api/state")
    assert sim._pipeline_cache_key is not None

    before = _network_gate_pct()
    _quiet(client.post, "/api/simulation/step")
    after = _network_gate_pct()

    state = client.get("/api/state").json()
    control = state["control"]
    assert control["forecast_control_eligible"] is True
    assert control["controller_status"] == "ACTIVE"
    assert control["control_applied"] is True
    assert state["auto_control"]["state"] == "AUTO_CONTROL_APPLIED"
    assert state["forecast_source_selection"]["control_forecast_validated"] is True
    moved = [nid for nid in NODES if abs(after[nid] - before[nid]) > 1e-9]
    assert moved, "VALIDATED_REPLAY must still move gates"
    for nid in NODES:
        assert after[nid] == pytest.approx(
            control["final_safe_control_action_pct"][nid], abs=1e-9), nid
    # The cache key tracked the step — the memoisation never went stale.
    assert sim._pipeline_cache_key == sim._display_pipeline_cache_key()


def test_p3_no_stale_advisory_is_displayed(monkeypatch):
    """7. after the authoritative state advances, the advisory is recomputed."""
    sim = state_manager.sim_state
    _quiet(client.post, "/api/simulation/pause")
    _quiet(client.post, "/api/controller/mode",
           json={"mode": "MANUAL", "source": "SIMULATION"})
    _quiet(client.post, "/api/simulation/reset")

    calls = _pipeline_counter(sim, monkeypatch)
    warm = client.get("/api/state").json()
    assert calls["n"] == 1
    assert warm["gnn_advisory"] == sim.gnn_advisory

    _quiet(client.post, "/api/simulation/step")
    after_step = calls["n"]
    fresh = client.get("/api/state").json()
    # The step recomputed the pipeline, so the read after it is a cache hit
    # of the CURRENT state's advisory — never the previous run's.
    assert calls["n"] == after_step
    assert fresh["gnn_advisory"] == sim.gnn_advisory
    assert sim._pipeline_cache_key == sim._display_pipeline_cache_key()
    assert int(fresh["state_identity"]["sim_step_index"]) > 0


# ===========================================================================
# P4 — DEADLINE-BASED SIMULATION LOOP (cadence, overruns, no catch-up)
# ===========================================================================

def _drive_loop(monkeypatch, seconds, step_duration, speed):
    """
    Run the REAL `simulation_loop()` with a synthetic `step()` of known cost.

    Returns (timestamps, overrun_count). `broadcast_state()` is replaced by a
    no-op so the measurement isolates scheduling, not serialisation.
    """
    sim = state_manager.sim_state
    stamps = []

    def fake_step():
        stamps.append(time.monotonic())
        time.sleep(step_duration)          # synchronous CPU work, as in production

    async def fake_broadcast():
        await asyncio.sleep(0)

    monkeypatch.setattr(sim, "step", fake_step)
    monkeypatch.setattr(sim, "broadcast_state", fake_broadcast)
    monkeypatch.setattr(sim, "sim_speed", float(speed))
    sim.loop_overrun_count = 0
    sim.last_loop_overrun_s = 0.0
    sim.max_loop_overrun_s = 0.0

    async def drive():
        sim.running = True
        task = asyncio.create_task(sim.simulation_loop())
        try:
            await asyncio.sleep(seconds)
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            sim.running = False

    asyncio.run(drive())
    return stamps, sim.loop_overrun_count


def test_p4_cadence_period_excludes_computation_time(monkeypatch):
    """
    The requested period is 0.2 s (speed 5); a step costs 0.08 s.

    OLD behaviour: interval = 0.2 + 0.08 = 0.28 s (cadence drift).
    NEW behaviour: interval ~ 0.2 s (the sleep covers only the REMAINING time).
    """
    period = 0.2
    computation = 0.08
    stamps, _overruns = _drive_loop(monkeypatch, seconds=1.6,
                                    step_duration=computation, speed=5.0)
    assert len(stamps) >= 4, "the loop must keep stepping"
    intervals = [b - a for a, b in zip(stamps, stamps[1:])]
    mean = sum(intervals) / len(intervals)
    # Drift-free: the mean interval is close to the PERIOD, not to
    # period + computation (which is what the old loop produced).
    assert abs(mean - period) < 0.5 * computation, (
        f"mean interval {mean:.3f}s is not on the {period}s deadline "
        f"(old drifted loop would give ~{period + computation:.3f}s)")


def test_p4_slow_step_overruns_are_recorded_without_catch_up(monkeypatch):
    """
    A step that costs 0.25 s against a 0.1 s deadline:

    * the overrun IS recorded,
    * no catch-up burst happens (steps stay ~0.25 s apart, never back-to-back
      at the 0.1 s rate),
    * no step is duplicated: one iteration = one step.
    """
    period = 0.1                                # speed 10
    computation = 0.25
    seconds = 1.2
    stamps, overruns = _drive_loop(monkeypatch, seconds=seconds,
                                   step_duration=computation, speed=1.0 / period)

    assert len(stamps) >= 3, "the loop must keep stepping when overrunning"
    assert overruns >= 1, "an overrun past the deadline must be recorded"

    # No catch-up: at most span/computation steps could physically fit.
    #
    # The bound is taken over the loop's OWN measured span, not over `seconds`:
    # `step()` is synchronous, so while it runs the event loop is blocked and
    # the driver's timer cannot be serviced. The step that is in flight when the
    # driver's `seconds` window expires therefore still completes and is
    # recorded, which makes the loop's active span longer than `seconds`.
    # (+2 = that in-flight step plus one step-duration of rounding.)
    span = stamps[-1] - stamps[0]
    max_possible = int(span / computation) + 2
    assert len(stamps) <= max_possible, (
        f"{len(stamps)} steps in {span:.3f}s of loop span implies catch-up "
        f"bursts (at {computation}s per step only {max_possible} can fit)")

    # And decisively below what a CATCH-UP loop would produce over the same
    # span: one step per `period` instead of one step per `computation`.
    burst_count = int(span / period) + 1
    assert len(stamps) <= burst_count, (
        f"{len(stamps)} steps in {span:.3f}s is at/near the "
        f"{burst_count}-step catch-up rate ({period}s period)")

    # Steps never collapse onto each other: each interval is a full step.
    intervals = [b - a for a, b in zip(stamps, stamps[1:])]
    assert min(intervals) > 0.5 * computation


def test_p4_overrun_path_still_yields_to_the_event_loop(monkeypatch):
    """
    An OVERRUNNING loop must keep the event loop cooperative even when there is
    no WebSocket client and `broadcast_state()` therefore returns without
    awaiting anything.

    Otherwise the deadline loop degenerates into a tight, never-yielding spin:
    every other task (WebSocket feed, REST commands) is starved and the loop
    can never even be cancelled — a hang, not a slow cadence.

    Here the deadline is 1 ms while `step()` costs 5 ms, so EVERY iteration
    overruns and takes the overrun branch, and `broadcast_state()` deliberately
    contains no await. A watchdog thread stops the loop after 1 s, so this test
    always terminates: if the loop had starved the event loop, the driver's
    0.05 s sleep could only return after that watchdog fired (measured: 1.014 s
    starved vs 0.064 s cooperative).
    """
    sim = state_manager.sim_state
    steps = []

    def slow_step():
        # Synchronous work STRICTLY longer than the deadline, so the positive
        # `remaining` branch is never taken and the overrun branch is hit on
        # every single iteration.
        steps.append(time.monotonic())
        time.sleep(0.005)

    monkeypatch.setattr(sim, "step", slow_step)
    monkeypatch.setattr(sim, "sim_speed", 1000.0)      # 1 ms deadline

    async def no_yield_broadcast():
        return None                # exactly what an empty client set produces

    monkeypatch.setattr(sim, "broadcast_state", no_yield_broadcast)

    async def drive():
        sim.running = True
        watchdog = threading.Timer(1.0, setattr, args=(sim, "running", False))
        watchdog.daemon = True
        watchdog.start()
        task = asyncio.create_task(sim.simulation_loop())
        start = time.monotonic()
        try:
            await asyncio.sleep(0.05)          # another task must be served
            elapsed = time.monotonic() - start
        finally:
            watchdog.cancel()
            sim.running = False
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        return elapsed

    elapsed = asyncio.run(drive())
    assert steps, "the loop must actually step"
    assert elapsed < 0.5, (
        f"the overrun path starved the event loop for {elapsed:.3f}s: a "
        "zero-delay yield is required on the overrun branch")


def test_p4_no_steps_while_paused(monkeypatch):
    """The idle branch must not step: no authoritative state is advanced."""
    sim = state_manager.sim_state
    stamps = []
    monkeypatch.setattr(sim, "step", lambda: stamps.append(time.monotonic()))
    sim.running = False

    async def drive():
        task = asyncio.create_task(sim.simulation_loop())
        await asyncio.sleep(0.8)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(drive())
    assert stamps == []


def test_p4_one_authoritative_step_per_iteration(monkeypatch):
    """
    Running the REAL loop with the REAL step: the physics timestep advances
    once per iteration and never jumps by two (no duplicated steps), and every
    iteration still broadcasts exactly one state.
    """
    sim = state_manager.sim_state
    _quiet(client.post, "/api/simulation/pause")
    _quiet(client.post, "/api/controller/mode",
           json={"mode": "MANUAL", "source": "SIMULATION"})
    _quiet(client.post, "/api/simulation/reset")

    broadcasts = {"n": 0}
    real_broadcast = sim.broadcast_state

    async def counting_broadcast():
        broadcasts["n"] += 1
        return await real_broadcast()

    monkeypatch.setattr(sim, "broadcast_state", counting_broadcast)
    monkeypatch.setattr(sim, "sim_speed", 20.0)      # 0.05 s period

    network_before = int(sim.bridge.cascade.network.timestep)
    steps_before = int(sim.sim_step_index)

    async def drive():
        sim.running = True
        task = asyncio.create_task(sim.simulation_loop())
        await asyncio.sleep(1.0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        sim.running = False

    asyncio.run(drive())

    network_delta = int(sim.bridge.cascade.network.timestep) - network_before
    step_index_delta = int(sim.sim_step_index) - steps_before
    assert network_delta >= 1, "the loop must actually step"
    # One authoritative step per iteration: the physics timestep and the step
    # counter move together, and the broadcast count matches the iterations,
    # so the scheduler never duplicated or dropped an authoritative step.
    assert step_index_delta == network_delta
    assert broadcasts["n"] >= network_delta
    assert broadcasts["n"] <= network_delta + 1
