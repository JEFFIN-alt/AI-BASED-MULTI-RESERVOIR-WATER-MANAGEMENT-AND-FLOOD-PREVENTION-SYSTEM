import pandas as pd
from typing import Dict, List, Any
import json

from .environment import VirtualCascade
from .controllers import BaseController
from .scenarios import get_timestep_inflows, get_timestep_forecasts

class SimulationEngine:
    def __init__(self, config: Dict[str, Any], historical_thresholds: Dict[str, float]):
        self.config = config
        self.historical_thresholds = historical_thresholds
            
    def run(self, scenario_name: str, sliced_actual: pd.DataFrame, sliced_v3: pd.DataFrame, controller: BaseController, is_synthetic: bool = False, synthetic_params: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Executes the discrete daily simulation loop.
        """
        # Pass the config dictionary directly to VirtualCascade
        cascade = VirtualCascade(self.config)
        
        # If synthetic, we simulate a specified number of days instead of relying on sliced_actual dates
        if is_synthetic:
            num_days = synthetic_params.get("duration_days", 30)
            dates = [pd.Timestamp("2025-01-01") + pd.Timedelta(days=i) for i in range(num_days)]
        else:
            dates = sliced_actual["date"].sort_values().unique()
        
        logs = []
        
        # Cache downstream statuses for cascade coordination (start NORMAL)
        downstream_statuses = {res: "NORMAL" for res in self.config["topology"]["cascade_order"]}
        
        for i, current_date in enumerate(dates):
            ts = pd.Timestamp(current_date)
            
            # 1. Load data for timestep
            if is_synthetic:
                from .hydrology import generate_synthetic_inflow, get_synthetic_forecasts
                inflows = {}
                forecasts = {}
                for r_name in cascade.reservoirs.keys():
                    base = synthetic_params.get("base_inflows", {}).get(r_name, 10.0)
                    inflows[r_name] = generate_synthetic_inflow(
                        base_inflow=base,
                        day_index=i,
                        storm_start_day=synthetic_params.get("storm_start", 999),
                        storm_duration=synthetic_params.get("storm_duration", 0),
                        rainfall_multiplier=synthetic_params.get("rainfall", 1.0),
                        humidity_pct=synthetic_params.get("humidity", 50.0),
                        storm_multiplier=synthetic_params.get("storm_multiplier", 1.0)
                    )
                    forecasts[r_name] = get_synthetic_forecasts(
                        res_name=r_name,
                        base_inflow=base,
                        day_index=i,
                        storm_start_day=synthetic_params.get("storm_start", 999),
                        storm_duration=synthetic_params.get("storm_duration", 0),
                        rainfall_multiplier=synthetic_params.get("rainfall", 1.0),
                        humidity_pct=synthetic_params.get("humidity", 50.0),
                        storm_multiplier=synthetic_params.get("storm_multiplier", 1.0)
                    )
            else:
                inflows = get_timestep_inflows(sliced_actual, ts)
                forecasts = get_timestep_forecasts(sliced_v3, ts)
            
            gate_commands = {}
            decisions = {}
            
            # Process reservoirs bottom-up to figure out downstream statuses for coordination
            reversed_order = reversed(self.config["topology"]["cascade_order"])
            
            for i, res_name in enumerate(reversed_order):
                res_obj = cascade.reservoirs[res_name]
                
                # Determine what the downstream status is
                if i == 0: # Terminal reservoir
                    ds_status = "NORMAL" if cascade.current_downstream_flow <= cascade.downstream_capacity else "HIGH RISK"
                else:
                    # Get status of the reservoir immediately below it
                    ds_status = downstream_statuses[self.config["topology"]["cascade_order"][-(i)]] # -i points to the one we just processed
                
                # 4 & 5. Provide info to controller & Generate release decision
                gate, reason = controller.compute_gate(
                    res_name, 
                    res_obj, 
                    forecasts.get(res_name, {}), 
                    self.historical_thresholds, 
                    ds_status
                )
                
                gate_commands[res_name] = gate
                decisions[res_name] = reason
                
                # Predict status for the next guy upstream to look at (simplified logic matching controller)
                if gate == 100.0:
                    downstream_statuses[res_name] = "HIGH RISK"
                elif gate >= 50.0:
                    downstream_statuses[res_name] = "ALERT"
                else:
                    downstream_statuses[res_name] = "NORMAL"

            # 7 & 8. Route flow and Update storage
            storage_before = {name: cascade.reservoirs[name].state.storage_mcm for name in cascade.reservoirs}
            
            cascade.step(inflows, gate_commands)
            
            # 10. Record states
            for name in self.config["topology"]["cascade_order"]:
                state = cascade.reservoirs[name].state
                logs.append({
                    "date": ts,
                    "reservoir": name,
                    "inflow": state.inflow_mcm_day,
                    "upstream_routed_inflow": state.upstream_routed_inflow,
                    "storage_before": storage_before[name],
                    "storage_after": state.storage_mcm,
                    "release": state.release_mcm_day,
                    "gate_position": state.gate_position_pct,
                    "overflow_indicator": 1 if state.storage_mcm >= cascade.reservoirs[name].capacity_mcm and state.overflow_events > 0 else 0,
                    "downstream_flow": cascade.current_downstream_flow if name == self.config["topology"]["cascade_order"][-1] else 0.0,
                    "decision_reason": decisions[name]
                })
                
                # Reset overflow trigger for next step logging
                state.overflow_events = 0
                
        return pd.DataFrame(logs)
