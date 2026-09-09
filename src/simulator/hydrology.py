from typing import Dict, Any

def generate_synthetic_inflow(
    base_inflow: float, 
    day_index: int, 
    storm_start_day: int, 
    storm_duration: int, 
    rainfall_multiplier: float = 1.0, 
    humidity_pct: float = 50.0, 
    storm_multiplier: float = 1.0
) -> float:
    """
    SYNTHETIC SIMULATION ASSUMPTION — NOT A CALIBRATED HYDROLOGICAL MODEL.
    
    This function generates a synthetic daily inflow volume for stress-testing.
    
    1. DIRECT INFLOW CONTROL — PRIMARY:
       The `base_inflow` is the primary anchor. Without any other modifiers, 
       inflow = base_inflow.
       
    2. RAINFALL — STRESS MODIFIER:
       Rainfall linearly scales the base inflow. 1.0 = normal, 2.0 = heavy rain.
       
    3. HUMIDITY — OPTIONAL SYNTHETIC STRESS FACTOR:
       High humidity slightly compounds the rainfall effect (e.g., saturated ground assumption).
       Again, purely a simulation parameter. 1 + (humidity_pct / 100).
       
    4. STORM MULTIPLIER:
       If the current day falls within the defined storm window, the entire formula 
       is multiplied by the storm_multiplier.
    """
    
    is_storm = (storm_start_day <= day_index < (storm_start_day + storm_duration))
    current_storm_mult = storm_multiplier if is_storm else 1.0
    
    # Humidity factor ranges from 1.0 (0% humidity) to 2.0 (100% humidity)
    humidity_factor = 1.0 + (humidity_pct / 100.0)
    
    synthetic_inflow = base_inflow * rainfall_multiplier * humidity_factor * current_storm_mult
    
    return max(0.0, synthetic_inflow)

def get_synthetic_forecasts(
    res_name: str,
    base_inflow: float,
    day_index: int,
    storm_start_day: int,
    storm_duration: int,
    rainfall_multiplier: float,
    humidity_pct: float,
    storm_multiplier: float
) -> Dict[str, float]:
    """
    Generates synthetic 'perfect' point forecasts to simulate what an ideal
    V3 model *might* predict if it perfectly saw the synthetic weather.
    
    NOTE: These are explicitly labeled as SIMULATION INPUT, NOT an actual V3 prediction.
    If the user requires strict V3 adherence, they must run REAL-V3-REPLAY.
    """
    f1 = generate_synthetic_inflow(base_inflow, day_index + 1, storm_start_day, storm_duration, rainfall_multiplier, humidity_pct, storm_multiplier)
    f3 = generate_synthetic_inflow(base_inflow, day_index + 3, storm_start_day, storm_duration, rainfall_multiplier, humidity_pct, storm_multiplier)
    f7 = generate_synthetic_inflow(base_inflow, day_index + 7, storm_start_day, storm_duration, rainfall_multiplier, humidity_pct, storm_multiplier)
    
    return {
        "forecast_1d": f1,
        "forecast_3d": f3,
        "forecast_7d": f7
    }
