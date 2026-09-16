import sys
import json
from pathlib import Path
from typing import Dict, Any, Tuple

# Resolve root to import risk_engine securely
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
from src.management.risk_engine import assess_risk
from src.common import units

# UNIT CONTRACT (see src/common/units.py)
# ---------------------------------------
# These controllers operate on the LIVE path and therefore emit gate positions
# in the EXTERNAL representation: PERCENT in [0, 100].
# The validated MPCController emits FRACTIONS in [0.0, 1.0].
# Convert only at the boundary, never inline.


class BaseController:
    def __init__(self, config: Dict):
        self.config = config
        # Config stores the environmental floor as a FRACTION (e.g. 0.05).
        # Controllers speak PERCENT, so convert once here through the boundary.
        self.min_flow = units.gate_fraction_to_percent(
            config["minimum_environmental_flow_percent"]["value"]
        )

    def compute_gate(self, env_state: Dict[str, Any], forecast_data: Dict[str, Any], historical_thresholds: Dict[str, float], downstream_status: str) -> Tuple[float, str]:
        raise NotImplementedError

class ReactiveBaselineController(BaseController):
    """
    BASELINE: Ignores forecasts entirely. Reacts strictly to current storage/water-level.
    Provides a fair comparison against the forecast-aware policy.
    """
    def compute_gate(self, virt_name: str, res_obj, forecast_data: Dict, historical_thresholds: Dict, downstream_status: str) -> Tuple[float, str]:
        # Formulate fake current state to use Channel 1 of risk engine
        simulated_level_proxy = res_obj.get_simulated_water_level_proxy()
        
        current_state = {"waterLevel": simulated_level_proxy}
        # Note: metadata expects FRL/Blue/Orange/Red in the same scale as waterLevel.
        # We mapped capacity % to 100 scale, so thresholds are 75, 85, 95
        metadata = {
            "blueLevel": 75.0,
            "orangeLevel": 85.0,
            "redLevel": 95.0,
            "historical_95th_inflow": None # Intentionally block channel 2
        }
        
        # We pass None for forecasts so Risk Engine Channel 2 stays NORMAL
        risk = assess_risk(current_state, metadata, None, None, None)
        wl_status = risk["water_level_status"]
        
        reason = f"Reactive Policy: {wl_status}."
        
        if wl_status == "HIGH RISK":
            gate = 100.0
        elif wl_status == "ALERT":
            gate = 50.0
            # Cascade coordination: if downstream is struggling, reduce release unless we are high risk
            if downstream_status in ["ALERT", "HIGH RISK"]:
                gate = 25.0
                reason += " (Reduced due to downstream risk)"
        elif wl_status == "WATCH":
            gate = 25.0
            if downstream_status in ["ALERT", "HIGH RISK"]:
                gate = self.min_flow
                reason += " (Reduced due to downstream risk)"
        else:
            gate = self.min_flow
            
        return gate, reason

class ForecastAwareController(BaseController):
    """
    FORECAST-AWARE: Uses V3 point forecasts via the validated Risk Engine to proactively 
    buffer storage BEFORE a simulated extreme event arrives.
    """
    def compute_gate(self, virt_name: str, res_obj, forecast_data: Dict, historical_thresholds: Dict, downstream_status: str) -> Tuple[float, str]:
        simulated_level_proxy = res_obj.get_simulated_water_level_proxy()
        current_state = {"waterLevel": simulated_level_proxy}
        
        # Real historical 95th threshold for the mapped reservoir
        real_name = self.config["reservoirs"][virt_name]["repository_derived_source"]
        hist_thresh = historical_thresholds.get(real_name)
        
        metadata = {
            "blueLevel": 75.0,
            "orangeLevel": 85.0,
            "redLevel": 95.0,
            "historical_95th_inflow": hist_thresh
        }
        
        # Pass actual V3 point forecasts to activate Channel 2
        f1 = forecast_data.get("forecast_1d")
        f3 = forecast_data.get("forecast_3d")
        f7 = forecast_data.get("forecast_7d")
        
        # Safely handle missing forecasts
        if f1 is None or f3 is None or f7 is None:
            # Degrade safely to reactive behavior if forecasts are missing/NaN
            risk = assess_risk(current_state, metadata, None, None, None)
        else:
            risk = assess_risk(current_state, metadata, f1, f3, f7)
            
        overall = risk["overall_status"]
        reason = f"Forecast Policy: {overall} ({risk['reason']})"
        
        # Deterministic rules balancing retention vs flood buffer
        if overall == "HIGH RISK":
            gate = 100.0
        elif overall == "ALERT":
            gate = 50.0
            if downstream_status in ["ALERT", "HIGH RISK"]:
                gate = 25.0 # Pre-release carefully
                reason += " (Reduced due to downstream risk)"
        elif overall == "WATCH":
            gate = 25.0
            if downstream_status in ["ALERT", "HIGH RISK"]:
                gate = self.min_flow
                reason += " (Held due to downstream risk)"
        else:
            gate = self.min_flow
            
        return gate, reason
