"""
Phase 5 — First Live Test (Deterministic Integration Test)

Verifies that Python state travels correctly through:
  Python → state_adapter → JSON → (ready for) Three.js updateState()

Does NOT directly modify JavaScript variables.
"""
import sys
import json
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.dashboard.twin_component.state_adapter import adapt_state_for_twin
from src.dashboard.sim_bridge import SimBridge


def build_sim_state_snapshot(bridge, forecasts=None):
    """Get a full state dict from SimBridge, exactly as the dashboard does."""
    if forecasts is None:
        forecasts = {}
        for res in bridge.cascade.cascade_order:
            forecasts[res] = {"forecast_1d": 10.0, "forecast_3d": 10.0, "forecast_7d": 10.0}
    return bridge.get_state(forecasts)


def test_water_level_rise():
    """
    Reservoir 1 water_level must change from 0.65 to 0.85
    when we manipulate the underlying simulation storage.
    """
    config_path = str(_PROJECT_ROOT / "configs" / "simulation" / "four_reservoir_demo.json")
    thresh_path = str(_PROJECT_ROOT / "data" / "processed" / "historical_inflow_thresholds.json")

    bridge = SimBridge(config_path, thresh_path)

    # Set Reservoir A to 65% capacity
    res_a = bridge.cascade.reservoirs["Virtual Reservoir A"]
    res_a.state.storage_mcm = res_a.capacity_mcm * 0.65

    sim_state = build_sim_state_snapshot(bridge)
    twin = adapt_state_for_twin(sim_state, "MANUAL", 0.0)

    level_65 = twin["reservoirs"]["reservoir_1"]["water_level"]
    print(f"[TEST] Reservoir 1 water_level at 65% storage: {level_65:.4f}")
    assert abs(level_65 - 0.65) < 0.02, f"Expected ~0.65, got {level_65}"

    # Set Reservoir A to 85% capacity
    res_a.state.storage_mcm = res_a.capacity_mcm * 0.85

    sim_state = build_sim_state_snapshot(bridge)
    twin = adapt_state_for_twin(sim_state, "MANUAL", 0.0)

    level_85 = twin["reservoirs"]["reservoir_1"]["water_level"]
    print(f"[TEST] Reservoir 1 water_level at 85% storage: {level_85:.4f}")
    assert abs(level_85 - 0.85) < 0.02, f"Expected ~0.85, got {level_85}"
    assert level_85 > level_65, "Water level must visibly RISE"

    print("[PASS] Water level rise test passed.")
    return twin


def test_gate_change():
    """
    Gate visual must change from 0.20 to 0.90 when simulation gate changes.
    """
    config_path = str(_PROJECT_ROOT / "configs" / "simulation" / "four_reservoir_demo.json")
    thresh_path = str(_PROJECT_ROOT / "data" / "processed" / "historical_inflow_thresholds.json")

    bridge = SimBridge(config_path, thresh_path)

    # Set gate to 20%
    res_a = bridge.cascade.reservoirs["Virtual Reservoir A"]
    res_a.state.gate_position_pct = 20.0

    sim_state = build_sim_state_snapshot(bridge)
    twin = adapt_state_for_twin(sim_state, "MANUAL", 0.0)

    gate_20 = twin["reservoirs"]["reservoir_1"]["gate"]
    print(f"[TEST] Reservoir 1 gate at 20%: {gate_20:.4f}")
    assert abs(gate_20 - 0.20) < 0.02, f"Expected ~0.20, got {gate_20}"

    # Set gate to 90%
    res_a.state.gate_position_pct = 90.0

    sim_state = build_sim_state_snapshot(bridge)
    twin = adapt_state_for_twin(sim_state, "MANUAL", 0.0)

    gate_90 = twin["reservoirs"]["reservoir_1"]["gate"]
    print(f"[TEST] Reservoir 1 gate at 90%: {gate_90:.4f}")
    assert abs(gate_90 - 0.90) < 0.02, f"Expected ~0.90, got {gate_90}"
    assert gate_90 > gate_20, "Gate must visibly OPEN wider"

    print("[PASS] Gate change test passed.")
    return twin


def test_full_json_contract():
    """Verify the complete JSON output matches the Digital Twin contract."""
    config_path = str(_PROJECT_ROOT / "configs" / "simulation" / "four_reservoir_demo.json")
    thresh_path = str(_PROJECT_ROOT / "data" / "processed" / "historical_inflow_thresholds.json")

    bridge = SimBridge(config_path, thresh_path)
    sim_state = build_sim_state_snapshot(bridge)
    twin = adapt_state_for_twin(sim_state, "AI", 0.5)

    # Validate top-level keys
    required_top = ["reservoirs", "storm_intensity", "downstream_flow",
                    "controller_mode", "simulation_time", "hardware_status", "metadata"]
    for key in required_top:
        assert key in twin, f"Missing top-level key: {key}"
        
    assert twin["metadata"]["flow_unit"] == "m3/s"

    # Validate all 3 reservoirs
    for i in range(1, 4):
        rk = f"reservoir_{i}"
        assert rk in twin["reservoirs"], f"Missing {rk}"
        res = twin["reservoirs"][rk]
        for field in ["water_level", "storage", "inflow", "release", "gate", "risk"]:
            assert field in res, f"Missing field {field} in {rk}"
        # Numeric checks
        for field in ["water_level", "storage", "inflow", "release", "gate"]:
            assert isinstance(res[field], (int, float)), f"{rk}.{field} is not numeric"
        # Risk must be a valid string
        assert res["risk"] in ["normal", "warning", "danger", "critical"], \
            f"{rk}.risk has invalid value: {res['risk']}"

    # Hardware status must all be NOT_CONNECTED
    hw = twin["hardware_status"]
    for key in ["esp32", "water_level_sensor", "flow_sensor", "gate_actuator"]:
        assert hw[key] == "NOT_CONNECTED", f"Hardware {key} should be NOT_CONNECTED, got {hw[key]}"

    # storm_intensity must be numeric
    assert isinstance(twin["storm_intensity"], (int, float))
    # downstream_flow must be numeric
    assert isinstance(twin["downstream_flow"], (int, float))

    # Check unit conversions
    res1 = twin["reservoirs"]["reservoir_1"]
    # 10 MCM/day = 10 * 1e6 / 86400 = 115.7407 m3/s
    assert abs(res1["forecast_1d"] - (10.0 * 1_000_000.0 / 86400.0)) < 0.001

    print("[PASS] Full JSON contract test passed.")
    print(json.dumps(twin, indent=2))
    return twin


if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 5 — DETERMINISTIC INTEGRATION TESTS")
    print("=" * 60)

    test_full_json_contract()
    print()
    test_water_level_rise()
    print()
    test_gate_change()

    print()
    print("=" * 60)
    print("ALL PHASE 5 TESTS PASSED")
    print("=" * 60)
