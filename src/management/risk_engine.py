from typing import Dict, Any, List

def assess_risk(
    current_state: Dict[str, Any],
    metadata: Dict[str, Any],
    forecast_1d: float,
    forecast_3d: float,
    forecast_7d: float
) -> Dict[str, str]:
    """
    Evaluates reservoir risk using a dual-channel Point-Forecast Warning Engine.
    
    WARNING CHANNEL 1: Current Water Level vs static warning thresholds.
    WARNING CHANNEL 2: Point Inflow Forecasts vs historical 95th percentile inflow (training data only).
    """
    reasons = []
    
    # ---------------------------------------------------------
    # CHANNEL 1: CURRENT WATER LEVEL STATUS
    # ---------------------------------------------------------
    wl_level = 0  # 0: NORMAL, 1: WATCH, 2: ALERT, 3: HIGH RISK
    wl_status = "NORMAL"
    current_wl = current_state.get('waterLevel')
    
    if current_wl is not None:
        red = metadata.get('redLevel')
        orange = metadata.get('orangeLevel')
        blue = metadata.get('blueLevel')
        
        if red is not None and current_wl >= red:
            wl_level = 3
            wl_status = "HIGH RISK"
            reasons.append(f"Current water level ({current_wl}) exceeds RED warning level ({red}).")
        elif orange is not None and current_wl >= orange:
            wl_level = 2
            wl_status = "ALERT"
            reasons.append(f"Current water level ({current_wl}) exceeds ORANGE warning level ({orange}).")
        elif blue is not None and current_wl >= blue:
            wl_level = 1
            wl_status = "WATCH"
            reasons.append(f"Current water level ({current_wl}) exceeds BLUE warning level ({blue}).")
    else:
        # If waterLevel is missing in telemetry
        wl_status = "UNKNOWN"
        reasons.append("Current water level is missing.")

    if wl_level == 0 and wl_status == "NORMAL":
        reasons.append("Current water level is within safe thresholds.")
        
    # ---------------------------------------------------------
    # CHANNEL 2: INFLOW FORECAST STATUS
    # ---------------------------------------------------------
    inflow_level = 0 # 0: NORMAL, 1: ALERT (Watch mapped to alert for simplicity or we can keep just ALERT)
    inflow_status = "NORMAL"
    inflow_threshold = metadata.get('historical_95th_inflow')
    
    if inflow_threshold is None:
        inflow_status = "INSUFFICIENT_DATA"
        reasons.append("Missing historical inflow threshold for this reservoir.")
    else:
        triggered_horizons = []
        for h, forecast in [('1d', forecast_1d), ('3d', forecast_3d), ('7d', forecast_7d)]:
            if forecast is not None and forecast >= inflow_threshold:
                inflow_level = max(inflow_level, 2)
                inflow_status = "ALERT"
                triggered_horizons.append(f"{h} ({forecast:.2f})")
                
            if forecast is not None and forecast < 0:
                reasons.append(f"Warning: Negative forecast detected at {h}.")
                
        if inflow_level > 0:
            reasons.append(f"Forecast inflow exceeds historical 95th percentile ({inflow_threshold:.2f}) at horizons: {', '.join(triggered_horizons)}.")
        else:
            reasons.append("Forecast inflow is within historical norms.")

    # ---------------------------------------------------------
    # COMBINED OVERALL STATUS
    # ---------------------------------------------------------
    overall_level = max(wl_level, inflow_level if inflow_status != "INSUFFICIENT_DATA" else 0)
    
    if overall_level == 3:
        overall_status = "HIGH RISK"
    elif overall_level == 2:
        overall_status = "ALERT"
    elif overall_level == 1:
        overall_status = "WATCH"
    else:
        overall_status = "NORMAL"
        
    return {
        'overall_status': overall_status,
        'water_level_status': wl_status,
        'inflow_forecast_status': inflow_status,
        'reason': " | ".join(reasons)
    }
