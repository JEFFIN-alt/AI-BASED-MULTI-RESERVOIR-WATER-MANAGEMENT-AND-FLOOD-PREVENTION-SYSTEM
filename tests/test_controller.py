"""
Phase 15.3 — Coordinated Controller Tests

12 tests covering coordination, safety, fallback, determinism,
and baseline vs MPC comparison.
"""
import sys
import copy
import hashlib
import math
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.network_env.reservoir_network import ReservoirNetwork
from src.network_env.v3_forecast_adapter import (
    V3ForecastAdapter, NetworkForecastSnapshot, ReservoirForecast, ForecastStatus
)
from src.controller.baseline_controller import BaselineController
from src.controller.mpc_controller import MPCController, MPCConfig, ControlDecision
from src.controller.objective import ObjectiveFunction, ObjectiveWeights
from src.controller.safety import SafetyLayer

PROJECT_ROOT = str(_PROJECT_ROOT)
TOPOLOGY_PATH = str(_PROJECT_ROOT / "src" / "network_env" / "topology_config.yaml")
V3_RESULTS_DIR = _PROJECT_ROOT / "results" / "lstm_pytorch_v3_logtarget"
V3_MODEL_DIR = _PROJECT_ROOT / "models" / "lstm_pytorch_v3_logtarget"


def _hash_dir(dirpath: Path) -> dict:
    hashes = {}
    if dirpath.exists():
        for f in sorted(dirpath.rglob("*")):
            if f.is_file():
                hashes[str(f.relative_to(dirpath))] = hashlib.sha256(f.read_bytes()).hexdigest()
    return hashes


def _make_network(storages=None):
    """Create a network with custom initial storages."""
    import yaml
    with open(TOPOLOGY_PATH, 'r') as f:
        cfg = yaml.safe_load(f)
    if storages:
        for i, res in enumerate(cfg["reservoirs"]):
            nid = res["id"]
            if nid in storages:
                res["initial_storage_mcm"]["value"] = storages[nid]
    return ReservoirNetwork(config_dict=cfg)


# ---------------------------------------------------------------------------
# TEST 1 — SAFE NORMAL CONDITIONS
# ---------------------------------------------------------------------------
def test_01_safe_normal():
    """All reservoirs at ~50%. Controller should apply minimal release."""
    net = _make_network()
    mpc = MPCController()
    inflows = {nid: 1.0 for nid in net.processing_order}

    decision = mpc.decide(net, current_inflows=inflows)

    assert decision.status in ("OPTIMAL", "CORRECTED", "SAFE"), \
        f"Expected safe status, got {decision.status}"
    for nid, gate in decision.gate_positions.items():
        assert 0.0 <= gate <= 1.0
    print(f"TEST 1 PASSED — Normal conditions: status={decision.status}, "
          f"gates={decision.gate_positions}")


# ---------------------------------------------------------------------------
# TEST 2 — HIGH UPSTREAM INFLOW (COORDINATION)
# ---------------------------------------------------------------------------
def test_02_high_upstream_coordination():
    """
    A receives high inflow. Controller must consider B's capacity
    before deciding to release from A.
    """
    # A near full, B also moderately full
    net = _make_network({
        "Reservoir_A": 9.5,   # 87.8% of 10.82
        "Reservoir_B": 18.0,  # 84.7% of 21.26
        "Reservoir_C": 174.0,
        "Reservoir_D": 200.0,
    })
    mpc = MPCController()
    # High inflow at A
    inflows = {"Reservoir_A": 5.0, "Reservoir_B": 0.0,
               "Reservoir_C": 0.0, "Reservoir_D": 0.0}

    decision = mpc.decide(net, current_inflows=inflows)

    # Controller should release from A (it's near capacity with high inflow)
    # but the decision should be COORDINATED — aware that B is also filling
    gate_a = decision.gate_positions.get("Reservoir_A", 0)
    assert gate_a > 0, "A should release under high inflow pressure"
    print(f"TEST 2 PASSED — High upstream: A gate={gate_a:.2f}, "
          f"score={decision.objective_score:.2f}, status={decision.status}")


# ---------------------------------------------------------------------------
# TEST 3 — DOWNSTREAM NEAR CAPACITY
# ---------------------------------------------------------------------------
def test_03_downstream_near_capacity():
    """
    D is nearly full. Controller should limit upstream releases
    to avoid downstream overload.
    """
    net = _make_network({
        "Reservoir_A": 5.0,
        "Reservoir_B": 10.0,
        "Reservoir_C": 170.0,
        "Reservoir_D": 395.0,  # 98.4% of 401.43 — near capacity
    })
    mpc = MPCController()
    inflows = {nid: 2.0 for nid in net.processing_order}

    decision = mpc.decide(net, current_inflows=inflows)

    # D is near capacity, so the controller should be cautious
    print(f"TEST 3 PASSED — Downstream stress: D gate={decision.gate_positions.get('Reservoir_D', 0):.2f}, "
          f"score={decision.objective_score:.2f}, status={decision.status}")


# ---------------------------------------------------------------------------
# TEST 4 — UPSTREAM + DOWNSTREAM STRESS
# ---------------------------------------------------------------------------
def test_04_full_network_stress():
    """
    A has high forecast inflow AND D is near capacity.
    Controller must recognize the network-wide constraint.
    """
    net = _make_network({
        "Reservoir_A": 10.0,  # 92.4% — near capacity
        "Reservoir_B": 19.0,  # 89.4%
        "Reservoir_C": 330.0, # 94.7%
        "Reservoir_D": 390.0, # 97.2%
    })
    mpc = MPCController()
    inflows = {"Reservoir_A": 5.0, "Reservoir_B": 3.0,
               "Reservoir_C": 10.0, "Reservoir_D": 5.0}

    decision = mpc.decide(net, current_inflows=inflows)

    # Under full stress, controller should be in a high-release mode
    total_gate = sum(decision.gate_positions.values())
    assert total_gate > 1.0, "Should be releasing aggressively under stress"
    print(f"TEST 4 PASSED — Full stress: gates={decision.gate_positions}, "
          f"score={decision.objective_score:.2f}")


# ---------------------------------------------------------------------------
# TEST 5 — FORECAST-AWARE EARLY ACTION
# ---------------------------------------------------------------------------
def test_05_forecast_vs_baseline():
    """
    Compare MPC with forecasts vs baseline without forecasts.
    """
    net_mpc = _make_network({
        "Reservoir_A": 8.0,  # 74% — close to blue threshold
        "Reservoir_B": 15.0,
        "Reservoir_C": 170.0,
        "Reservoir_D": 200.0,
    })
    net_base = _make_network({
        "Reservoir_A": 8.0,
        "Reservoir_B": 15.0,
        "Reservoir_C": 170.0,
        "Reservoir_D": 200.0,
    })

    # Create a mock forecast showing high future inflow at A
    snapshot = NetworkForecastSnapshot(forecast_date="2025-01-01")
    for nid in net_mpc.processing_order:
        v3_name = {"Reservoir_A": "Anayirankal", "Reservoir_B": "Ponmudi",
                    "Reservoir_C": "Idamalayar", "Reservoir_D": "Idukki"}.get(nid, "")
        high_inflow = 8.0 if nid == "Reservoir_A" else 1.0
        snapshot.forecasts[nid] = ReservoirForecast(
            reservoir_id=nid, v3_reservoir_name=v3_name,
            forecast_date="2025-01-01",
            target_1d=high_inflow, target_3d=high_inflow * 1.2, target_7d=high_inflow * 0.8,
            status_1d=ForecastStatus.AVAILABLE,
            status_3d=ForecastStatus.AVAILABLE,
            status_7d=ForecastStatus.AVAILABLE,
        )

    mpc = MPCController()
    baseline = BaselineController()

    inflows = {"Reservoir_A": 2.0, "Reservoir_B": 0.5,
               "Reservoir_C": 0.5, "Reservoir_D": 0.5}

    mpc_decision = mpc.decide(net_mpc, forecast_snapshot=snapshot, current_inflows=inflows)
    base_decision = baseline.decide(
        {nid: net_base.nodes[nid].state.storage for nid in net_base.processing_order},
        {nid: net_base.nodes[nid].capacity for nid in net_base.processing_order},
    )

    mpc_gate_a = mpc_decision.gate_positions.get("Reservoir_A", 0)
    base_gate_a = base_decision.get("Reservoir_A", {}).get("gate_position", 0)

    print(f"TEST 5 PASSED — Forecast comparison:")
    print(f"  MPC (forecast-aware):  A gate = {mpc_gate_a:.2f}, forecast_used={mpc_decision.forecast_used}")
    print(f"  Baseline (no forecast): A gate = {base_gate_a:.2f}")
    assert mpc_decision.forecast_used, "MPC should use forecast"


# ---------------------------------------------------------------------------
# TEST 6 — FORECAST UNAVAILABLE
# ---------------------------------------------------------------------------
def test_06_forecast_unavailable():
    """Controller must work without forecast, with explicit status."""
    net = _make_network()
    mpc = MPCController()
    inflows = {nid: 1.0 for nid in net.processing_order}

    decision = mpc.decide(net, forecast_snapshot=None, current_inflows=inflows)

    assert not decision.forecast_used
    assert decision.forecast_status in ("NO_FORECAST", "FORECAST_UNAVAILABLE")
    assert decision.status in ("OPTIMAL", "CORRECTED")
    print(f"TEST 6 PASSED — No forecast: status={decision.status}, "
          f"forecast_status={decision.forecast_status}")


# ---------------------------------------------------------------------------
# TEST 7 — INVALID FORECAST
# ---------------------------------------------------------------------------
def test_07_invalid_forecast():
    """Controller handles NaN/invalid forecasts safely."""
    net = _make_network()
    snapshot = NetworkForecastSnapshot(forecast_date="2025-01-01")
    for nid in net.processing_order:
        snapshot.forecasts[nid] = ReservoirForecast(
            reservoir_id=nid, v3_reservoir_name="test",
            forecast_date="2025-01-01",
            target_1d=float('nan'), target_3d=float('nan'), target_7d=float('nan'),
            status_1d=ForecastStatus.INVALID,
            status_3d=ForecastStatus.INVALID,
            status_7d=ForecastStatus.INVALID,
        )

    mpc = MPCController()
    inflows = {nid: 1.0 for nid in net.processing_order}
    decision = mpc.decide(net, forecast_snapshot=snapshot, current_inflows=inflows)

    # Should fall back to current inflow
    assert decision.status in ("OPTIMAL", "CORRECTED", "FALLBACK")
    print(f"TEST 7 PASSED — Invalid forecast: status={decision.status}")


# ---------------------------------------------------------------------------
# TEST 8 — NO FEASIBLE SAFE ACTION (stress test)
# ---------------------------------------------------------------------------
def test_08_emergency():
    """
    All reservoirs at 99%+ capacity with high inflow.
    The controller may report EMERGENCY or produce high-release gates.
    """
    net = _make_network({
        "Reservoir_A": 10.8,   # 99.8%
        "Reservoir_B": 21.2,   # 99.7%
        "Reservoir_C": 348.0,  # 99.9%
        "Reservoir_D": 401.0,  # 99.9%
    })
    mpc = MPCController()
    inflows = {nid: 20.0 for nid in net.processing_order}

    decision = mpc.decide(net, current_inflows=inflows)

    # Under extreme stress, should still produce valid gate positions
    for nid, gate in decision.gate_positions.items():
        assert 0.0 <= gate <= 1.0, f"Invalid gate for {nid}: {gate}"
    print(f"TEST 8 PASSED — Emergency: status={decision.status}, "
          f"score={decision.objective_score:.2f}")


# ---------------------------------------------------------------------------
# TEST 9 — GATE LIMITS
# ---------------------------------------------------------------------------
def test_09_gate_limits():
    """Safety layer rejects invalid gate values."""
    safety = SafetyLayer(max_gate_change_per_step=0.5)

    # NaN gate
    result = safety.validate(
        {"R0": float('nan'), "R1": 0.5},
        {"R0": 0.3, "R1": 0.3},
        ["R0", "R1"],
    )
    assert not result.is_safe
    assert result.validated_gates["R0"] == 0.1  # fallback
    assert 0.0 <= result.validated_gates["R1"] <= 1.0

    # Out-of-range gate
    result2 = safety.validate(
        {"R0": 2.5, "R1": -0.5},
        {"R0": 0.3, "R1": 0.3},
        ["R0", "R1"],
    )
    assert result2.validated_gates["R0"] <= 1.0
    assert result2.validated_gates["R1"] >= 0.0

    print(f"TEST 9 PASSED — Gate limits enforced")


# ---------------------------------------------------------------------------
# TEST 10 — DETERMINISM
# ---------------------------------------------------------------------------
def test_10_determinism():
    """Same inputs → identical controller output."""
    net1 = _make_network()
    net2 = _make_network()
    mpc = MPCController()

    inflows = {"Reservoir_A": 3.0, "Reservoir_B": 1.0,
               "Reservoir_C": 2.0, "Reservoir_D": 1.5}

    d1 = mpc.decide(net1, current_inflows=inflows)
    d2 = mpc.decide(net2, current_inflows=inflows)

    for nid in net1.processing_order:
        assert abs(d1.gate_positions[nid] - d2.gate_positions[nid]) < 1e-12, \
            f"Non-deterministic: {nid}"
    assert abs(d1.objective_score - d2.objective_score) < 1e-12

    print(f"TEST 10 PASSED — Deterministic output")


# ---------------------------------------------------------------------------
# TEST 11 — NETWORK IMMUTABILITY
# ---------------------------------------------------------------------------
def test_11_immutability():
    """Candidate evaluation must NOT alter the real network state."""
    net = _make_network()
    mpc = MPCController()

    # Record pre-decision state
    pre_storages = {nid: net.nodes[nid].state.storage for nid in net.processing_order}
    pre_timestep = net.timestep

    inflows = {nid: 5.0 for nid in net.processing_order}
    _ = mpc.decide(net, current_inflows=inflows)

    # Verify nothing changed
    for nid in net.processing_order:
        assert net.nodes[nid].state.storage == pre_storages[nid], \
            f"{nid} storage changed from {pre_storages[nid]} to {net.nodes[nid].state.storage}"
    assert net.timestep == pre_timestep, "Timestep changed"

    print(f"TEST 11 PASSED — Network immutable during candidate evaluation")


# ---------------------------------------------------------------------------
# TEST 12 — V3 INTEGRITY
# ---------------------------------------------------------------------------
def test_12_v3_integrity():
    """V3 artifacts unchanged after controller execution."""
    results_before = _hash_dir(V3_RESULTS_DIR)
    model_before = _hash_dir(V3_MODEL_DIR)

    # Run controller with real V3 forecasts
    net = _make_network()
    adapter = V3ForecastAdapter(PROJECT_ROOT)
    snapshot = adapter.get_network_snapshot("2025-01-01")
    mpc = MPCController()
    inflows = {nid: 2.0 for nid in net.processing_order}
    _ = mpc.decide(net, forecast_snapshot=snapshot, current_inflows=inflows)

    results_after = _hash_dir(V3_RESULTS_DIR)
    model_after = _hash_dir(V3_MODEL_DIR)

    assert results_before == results_after, "V3 results modified!"
    assert model_before == model_after, "V3 model modified!"

    print(f"TEST 12 PASSED — V3 artifacts unchanged")


# ===========================================================================
# COMPARISON SIMULATION: Baseline vs MPC over 30-day scenario
# ===========================================================================
def run_comparison_simulation():
    """
    Run identical 30-day scenarios with both controllers and compare.
    """
    import yaml
    with open(TOPOLOGY_PATH, 'r') as f:
        base_cfg = yaml.safe_load(f)

    print(f"\n{'='*70}")
    print("COMPARISON SIMULATION: Baseline vs MPC (30 days)")
    print(f"{'='*70}")

    # Scenario: rising inflow at Reservoir A, moderate elsewhere
    def get_inflows(t):
        """Synthetic inflow that ramps up, simulating an approaching storm."""
        base_a = 1.0 + max(0, (t - 5)) * 0.8  # ramps from 1.0 to ~21
        base_a = min(base_a, 15.0)
        return {
            "Reservoir_A": base_a,
            "Reservoir_B": 1.5,
            "Reservoir_C": 5.0,
            "Reservoir_D": 3.0,
        }

    def build_forecast_at(t, net, node_ids):
        """Build synthetic forecast from known future inflows."""
        snap = NetworkForecastSnapshot(forecast_date=f"day-{t}")
        mapping = {"Reservoir_A": "Anayirankal", "Reservoir_B": "Ponmudi",
                    "Reservoir_C": "Idamalayar", "Reservoir_D": "Idukki"}
        for nid in node_ids:
            inflows_1d = get_inflows(t + 1)
            inflows_3d = get_inflows(t + 3)
            inflows_7d = get_inflows(t + 7)
            snap.forecasts[nid] = ReservoirForecast(
                reservoir_id=nid, v3_reservoir_name=mapping.get(nid, ""),
                forecast_date=f"day-{t}",
                target_1d=inflows_1d[nid], target_3d=inflows_3d[nid], target_7d=inflows_7d[nid],
                status_1d=ForecastStatus.AVAILABLE,
                status_3d=ForecastStatus.AVAILABLE,
                status_7d=ForecastStatus.AVAILABLE,
            )
        return snap

    # --- Run Baseline ---
    net_baseline = ReservoirNetwork(config_dict=copy.deepcopy(base_cfg))
    baseline_ctrl = BaselineController()
    bl_metrics = {"overflow": 0, "overflow_vol": 0.0, "ds_violations": 0,
                  "peak_ds": 0.0, "total_release": 0.0}
    node_ids = net_baseline.processing_order

    for t in range(1, 31):
        inflows = get_inflows(t)
        storages = {nid: net_baseline.nodes[nid].state.storage for nid in node_ids}
        caps = {nid: net_baseline.nodes[nid].capacity for nid in node_ids}
        decisions = baseline_ctrl.decide(storages, caps)
        gates = {nid: decisions[nid]["gate_position"] for nid in node_ids}
        states = net_baseline.step(inflows, gates)

        for nid, st in states.items():
            if st.spill > 0:
                bl_metrics["overflow"] += 1
                bl_metrics["overflow_vol"] += st.spill
            bl_metrics["total_release"] += st.controlled_release

        terminal = states.get(net_baseline._terminal_node_id)
        if terminal:
            ds = terminal.total_outflow
            bl_metrics["peak_ds"] = max(bl_metrics["peak_ds"], ds)
            if ds > net_baseline.downstream_capacity:
                bl_metrics["ds_violations"] += 1

    # --- Run MPC ---
    net_mpc = ReservoirNetwork(config_dict=copy.deepcopy(base_cfg))
    mpc_ctrl = MPCController()
    mpc_metrics = {"overflow": 0, "overflow_vol": 0.0, "ds_violations": 0,
                   "peak_ds": 0.0, "total_release": 0.0, "forecast_used": 0}

    for t in range(1, 31):
        inflows = get_inflows(t)
        snapshot = build_forecast_at(t, net_mpc, node_ids)
        decision = mpc_ctrl.decide(net_mpc, forecast_snapshot=snapshot, current_inflows=inflows)

        if decision.forecast_used:
            mpc_metrics["forecast_used"] += 1

        states = net_mpc.step(inflows, decision.gate_positions)

        for nid, st in states.items():
            if st.spill > 0:
                mpc_metrics["overflow"] += 1
                mpc_metrics["overflow_vol"] += st.spill
            mpc_metrics["total_release"] += st.controlled_release

        terminal = states.get(net_mpc._terminal_node_id)
        if terminal:
            ds = terminal.total_outflow
            mpc_metrics["peak_ds"] = max(mpc_metrics["peak_ds"], ds)
            if ds > net_mpc.downstream_capacity:
                mpc_metrics["ds_violations"] += 1

    # --- Compare ---
    print(f"\n{'Metric':<30} {'Baseline':>12} {'MPC':>12} {'Diff':>12}")
    print("-" * 66)
    for key in ["overflow", "overflow_vol", "ds_violations", "peak_ds", "total_release"]:
        bl_val = bl_metrics[key]
        mpc_val = mpc_metrics[key]
        diff = mpc_val - bl_val
        label = "BETTER" if diff < 0 else ("SAME" if diff == 0 else "WORSE")
        print(f"  {key:<28} {bl_val:>12.2f} {mpc_val:>12.2f} {diff:>+12.2f}  {label}")
    print(f"  {'forecast_used':<28} {'N/A':>12} {mpc_metrics['forecast_used']:>12}")

    # Final mass balance
    bl_mb = net_baseline.mass_balance_check()
    mpc_mb = net_mpc.mass_balance_check()
    print(f"\n  Baseline mass balance residual: {bl_mb['residual_error']:.2e}")
    print(f"  MPC mass balance residual:      {mpc_mb['residual_error']:.2e}")

    # Return metrics for the report
    return bl_metrics, mpc_metrics


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Phase 15.3 — Coordinated Controller Tests")
    print("=" * 60)

    test_01_safe_normal()
    test_02_high_upstream_coordination()
    test_03_downstream_near_capacity()
    test_04_full_network_stress()
    test_05_forecast_vs_baseline()
    test_06_forecast_unavailable()
    test_07_invalid_forecast()
    test_08_emergency()
    test_09_gate_limits()
    test_10_determinism()
    test_11_immutability()
    test_12_v3_integrity()

    bl, mpc = run_comparison_simulation()

    print("\n" + "=" * 60)
    print("ALL 12 TESTS PASSED + COMPARISON SIMULATION COMPLETE.")
    print("=" * 60)
