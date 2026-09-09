import json
import pandas as pd
from pathlib import Path
import sys

# Setup paths
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.simulator.environment import VirtualReservoir, VirtualCascade
from src.simulator.engine import SimulationEngine
from src.simulator.controllers import ReactiveBaselineController

CONFIG_PATH = _PROJECT_ROOT / "configs" / "simulation" / "four_reservoir_demo.json"

def get_base_config():
    with open(CONFIG_PATH, 'r') as f:
        return json.load(f)

def test_mass_conservation_and_negative_storage():
    base_config = get_base_config()
    res_config = base_config["reservoirs"]["Virtual Reservoir A"]
    res_config["initial_storage_mcm"]["value"] = 5.0 # Start with 5 MCM
    res_config["max_release_capacity_mcm_day"]["value"] = 10.0 # Can release 10 MCM
    
    res = VirtualReservoir("Virtual Reservoir A", res_config)
    
    # Try to release 100% (10 MCM), but we only have 5 MCM + 1 MCM inflow = 6 MCM
    res.step(local_inflow_mcm_day=1.0, routed_inflow_mcm_day=0.0, gate_command_pct=100.0)
    
    # Storage should be exactly 0, not negative
    assert res.state.storage_mcm == 0.0
    # Actual release should be 6 MCM (what was available)
    assert res.state.release_mcm_day == 6.0

def test_absolute_capacity_limits():
    base_config = get_base_config()
    res_config = base_config["reservoirs"]["Virtual Reservoir A"]
    capacity = res_config["capacity_mcm"]["value"]
    res_config["initial_storage_mcm"]["value"] = capacity - 1.0 # Start almost full
    
    res = VirtualReservoir("Virtual Reservoir A", res_config)
    
    # Massive inflow, 0% gate
    res.step(local_inflow_mcm_day=10.0, routed_inflow_mcm_day=0.0, gate_command_pct=0.0)
    
    # Storage must not exceed capacity
    assert res.state.storage_mcm == capacity
    # Excess (10 - 1 = 9) must be forcibly spilled
    assert res.state.release_mcm_day == 9.0
    assert res.state.overflow_events == 1

def test_cascade_routing():
    base_config = get_base_config()
    cascade = VirtualCascade(base_config)
    
    # Inject massive inflow only into A. All gates 100%.
    inflows = {
        "Virtual Reservoir A": 5.0,
        "Virtual Reservoir B": 0.0,
        "Virtual Reservoir C": 0.0,
        "Virtual Reservoir D": 0.0
    }
    
    gates = {
        "Virtual Reservoir A": 100.0,
        "Virtual Reservoir B": 100.0,
        "Virtual Reservoir C": 100.0,
        "Virtual Reservoir D": 100.0
    }
    
    # Day 1
    cascade.step(inflows, gates)
    release_A_day1 = cascade.reservoirs["Virtual Reservoir A"].state.release_mcm_day
    
    # B should NOT have received A's release yet due to 1-day routing delay
    assert cascade.reservoirs["Virtual Reservoir B"].state.upstream_routed_inflow == 0.0
    
    # Day 2
    cascade.step(inflows, gates)
    # Now B should receive A's Day 1 release
    assert cascade.reservoirs["Virtual Reservoir B"].state.upstream_routed_inflow == release_A_day1

def test_synthetic_engine_graceful_degradation():
    base_config = get_base_config()
    # Pass empty dict for thresholds, controller should degrade safely
    engine = SimulationEngine(base_config, {})
    controller = ReactiveBaselineController(base_config)
    
    synthetic_params = {
        "duration_days": 5,
        "base_inflows": {"Virtual Reservoir A": 1.0, "Virtual Reservoir B": 1.0, "Virtual Reservoir C": 1.0, "Virtual Reservoir D": 1.0},
        "storm_start": 999,
        "storm_duration": 0,
        "rainfall": 1.0,
        "humidity": 50.0,
        "storm_multiplier": 1.0
    }
    
    df = engine.run(
        scenario_name="Synthetic", 
        sliced_actual=None, 
        sliced_v3=None, 
        controller=controller, 
        is_synthetic=True, 
        synthetic_params=synthetic_params
    )
    
    assert len(df) == 5 * 4 # 5 days * 4 reservoirs
    assert df["decision_reason"].iloc[0].startswith("Reactive Policy:")

if __name__ == "__main__":
    print("Running tests...")
    test_mass_conservation_and_negative_storage()
    test_absolute_capacity_limits()
    test_cascade_routing()
    test_synthetic_engine_graceful_degradation()
    print("All tests passed.")
