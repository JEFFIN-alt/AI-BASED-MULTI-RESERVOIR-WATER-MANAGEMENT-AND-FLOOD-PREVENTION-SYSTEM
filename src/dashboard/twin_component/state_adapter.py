import datetime

def mcm_per_day_to_m3_per_s(mcm: float) -> float:
    """
    Converts MCM/day (Million Cubic Meters per day) to m³/s (Cubic Meters per second).
    1 MCM = 1,000,000 m³
    1 Day = 86,400 seconds
    """
    if mcm is None:
        return None
    return float(mcm) * (1_000_000.0 / 86400.0)

def adapt_state_for_twin(sim_state, current_mode="MANUAL", storm_intensity=0.0):
    """
    Converts the output of SimBridge.get_state() into the exact JSON schema
    expected by the GLM 3D Reservoir Digital Twin UI.
    """
    
    mapping = {
        "Virtual Reservoir A": "reservoir_1",
        "Virtual Reservoir B": "reservoir_2",
        "Virtual Reservoir C": "reservoir_3"
    }
    
    twin_state = {
        "reservoirs": {},
        "storm_intensity": float(storm_intensity),
        "downstream_flow": mcm_per_day_to_m3_per_s(sim_state.get("downstream_flow", 0.0)),
        "controller_mode": current_mode,
        "simulation_time": datetime.datetime.now().isoformat(),
        "hardware_status": {
            "esp32": "NOT_CONNECTED",
            "water_level_sensor": "NOT_CONNECTED",
            "flow_sensor": "NOT_CONNECTED",
            "gate_actuator": "NOT_CONNECTED"
        },
        "metadata": {
            "flow_unit": "m3/s",
            "storage_unit": "ratio",
            "level_unit": "proxy_ratio"
        }
    }
    
    for res_name, res_data in sim_state.get("reservoirs", {}).items():
        twin_key = mapping.get(res_name)
        if not twin_key:
            continue
            
        twin_state["reservoirs"][twin_key] = {
            "water_level": res_data.get("storage_pct", 0.0) / 100.0,
            "storage": res_data.get("storage_pct", 0.0) / 100.0,
            "inflow": mcm_per_day_to_m3_per_s(res_data.get("inflow", 0.0) + res_data.get("routed_inflow", 0.0)),
            "release": mcm_per_day_to_m3_per_s(res_data.get("outflow", 0.0)),
            "gate": res_data.get("gate_position_pct", 0.0) / 100.0,
            "risk": res_data.get("risk_status", "NORMAL").lower(),
            "forecast_1d": mcm_per_day_to_m3_per_s(res_data.get("forecast_1d")),
            "forecast_3d": mcm_per_day_to_m3_per_s(res_data.get("forecast_3d")),
            "forecast_7d": mcm_per_day_to_m3_per_s(res_data.get("forecast_7d"))
        }
        
    # Provide defaults if missing
    for i in range(1, 4):
        key = f"reservoir_{i}"
        if key not in twin_state["reservoirs"]:
            twin_state["reservoirs"][key] = {
                "water_level": 0.0,
                "storage": 0.0,
                "inflow": 0.0,
                "release": 0.0,
                "gate": 0.0,
                "risk": "normal"
            }
            
    return twin_state
