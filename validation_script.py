import sys
from pathlib import Path
import json

_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.dashboard.api.state_manager import GlobalSimulationState
from src.dashboard.sim_bridge import SimBridge
from src.simulator.environment import VirtualCascade
import time

def run_tests():
    print("==================================================")
    print("1. INVENTORY OF DATA SOURCES")
    print("==================================================")
    print("- Inflow: VirtualReservoir.state.inflow_mcm_day (from environment.py), supplemented by routed inflow.")
    print("- Release: VirtualReservoir.state.release_mcm_day")
    print("- Storage: VirtualReservoir.state.storage_mcm")
    print("- Water Level: VirtualReservoir.get_simulated_water_level_proxy()")
    print("- Gate Position: VirtualReservoir.state.gate_position_pct")
    print("- Downstream Flow: VirtualCascade.current_downstream_flow")
    print("- Risk: risk_engine.py assess_risk()")
    print("- Storm Intensity: GlobalSimulationState.storm_intensity (scales manual inflows)")
    print("- Controller Mode: GlobalSimulationState.mode ('MANUAL' or 'AI')")
    print("- LSTM: Forecast predictions mapped via v3_forecast_adapter.py (targets 1d, 3d, 7d inflows).")
    
    # 2. NUMERICAL CONSISTENCY
    print("\n==================================================")
    print("2. SIMULATOR NUMERICAL CONSISTENCY TEST & 3. MASS BALANCE")
    print("==================================================")
    
    sim = GlobalSimulationState()
    sim.sim_speed = 1000.0 # run instantly
    
    # Step 0 (Init)
    state = sim.get_adapted_state()
    print("Initial State:")
    
    res_names = ["Virtual Reservoir A", "Virtual Reservoir B", "Virtual Reservoir C"]
    hist = {res: [] for res in res_names}
    
    for step in range(10):
        # record prior storage
        prior_storage = {}
        for r in res_names:
            prior_storage[r] = sim.bridge.cascade.reservoirs[r].state.storage_mcm
            
        sim.step()
        
        for r in res_names:
            res_obj = sim.bridge.cascade.reservoirs[r]
            inflow = res_obj.state.inflow_mcm_day + res_obj.state.upstream_routed_inflow
            outflow = res_obj.state.release_mcm_day
            expected_change = inflow - outflow
            actual_change = res_obj.state.storage_mcm - prior_storage[r]
            
            hist[r].append({
                "step": step+1,
                "prior": prior_storage[r],
                "inflow": inflow,
                "outflow": outflow,
                "final": res_obj.state.storage_mcm,
                "expected_change": expected_change,
                "actual_change": actual_change,
                "error": abs(expected_change - actual_change)
            })
            
    for r in res_names:
        print(f"\n--- {r} ---")
        for h in hist[r][:3]: # print first 3 steps
            print(f"Step {h['step']}: In={h['inflow']:.2f}, Out={h['outflow']:.2f}, Prior={h['prior']:.2f}, Final={h['final']:.2f}, Err={h['error']:.6f}")
            
    print("\n==================================================")
    print("4. CASCADE TEST")
    print("==================================================")
    # Test R1 release affecting R2 inflow
    sim = GlobalSimulationState()
    sim.manual_gates = {"Virtual Reservoir A": 100.0, "Virtual Reservoir B": 0.0, "Virtual Reservoir C": 0.0, "Virtual Reservoir D": 0.0}
    
    print("Step 1 (Gate A 100%):")
    sim.step()
    res_a = sim.bridge.cascade.reservoirs["Virtual Reservoir A"]
    res_b = sim.bridge.cascade.reservoirs["Virtual Reservoir B"]
    res_c = sim.bridge.cascade.reservoirs["Virtual Reservoir C"]
    print(f"A release: {res_a.state.release_mcm_day:.2f}")
    print(f"B routed inflow: {res_b.state.upstream_routed_inflow:.2f}")
    
    print("Step 2:")
    sim.step()
    print(f"A release: {res_a.state.release_mcm_day:.2f}")
    print(f"B routed inflow: {res_b.state.upstream_routed_inflow:.2f}")

    print("\n==================================================")
    print("5. GATE CONTROL TEST")
    print("==================================================")
    for g in [0, 25, 50, 75, 100]:
        sim = GlobalSimulationState()
        sim.manual_gates["Virtual Reservoir A"] = g; sim.manual_gates["Virtual Reservoir D"] = 0
        sim.step()
        r = sim.bridge.cascade.reservoirs["Virtual Reservoir A"]
        print(f"Gate {g}% -> Release: {r.state.release_mcm_day:.2f}, Storage: {r.state.storage_mcm:.2f}")
        
    print("\n==================================================")
    print("6. STORM TEST")
    print("==================================================")
    for s in [0.0, 0.5, 1.0]:
        sim = GlobalSimulationState()
        sim.storm_intensity = s
        sim.step()
        r = sim.bridge.cascade.reservoirs["Virtual Reservoir A"]
        print(f"Storm {s} -> Inflow: {r.state.inflow_mcm_day:.2f}")

    print("\n==================================================")
    print("7. MANUAL VS AI/MPC")
    print("==================================================")
    sim = GlobalSimulationState()
    sim.mode = "MANUAL"
    sim.step()
    print(f"MANUAL Gate A: {sim.bridge.cascade.reservoirs['Virtual Reservoir A'].state.gate_position_pct}")
    
    sim = GlobalSimulationState()
    sim.mode = "AI"
    sim.step()
    print(f"AI Gate A: {sim.bridge.cascade.reservoirs['Virtual Reservoir A'].state.gate_position_pct}")

    print("\n==================================================")
    print("10. STATE ADAPTER TEST")
    print("==================================================")
    ad_state = sim.get_adapted_state()
    print(f"Reservoirs mapped: {list(ad_state['reservoirs'].keys())}")
    for r_key, data in ad_state['reservoirs'].items():
        print(f"{r_key}: wl={data['water_level']:.2f}, st={data['storage']:.2f}, in={data['inflow']:.2f}, rl={data['release']:.2f}, gt={data['gate']:.2f}")
    
    
if __name__ == "__main__":
    run_tests()

