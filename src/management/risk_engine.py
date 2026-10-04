"""Threshold warnings: missing observations cannot establish a NORMAL verdict."""
import math
from numbers import Real
from typing import Any, Dict


def _finite(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def assess_risk(current_state: Dict[str, Any], metadata: Dict[str, Any],
                forecast_1d: float, forecast_3d: float, forecast_7d: float) -> Dict[str, str]:
    reasons = []
    current_wl = current_state.get('waterLevel')
    thresholds = [(metadata.get('redLevel'), 3, 'HIGH RISK', 'RED'),
                  (metadata.get('orangeLevel'), 2, 'ALERT', 'ORANGE'),
                  (metadata.get('blueLevel'), 1, 'WATCH', 'BLUE')]
    available = [entry for entry in thresholds if _finite(entry[0])]
    wl_level, wl_status = 0, 'UNKNOWN'
    if not _finite(current_wl):
        reasons.append('Current water level is missing or non-finite.')
    elif not available:
        reasons.append('Water-level warning thresholds are unavailable.')
    else:
        wl_status = 'NORMAL'
        for threshold, level, status, label in available:
            if current_wl >= threshold:
                wl_level, wl_status = level, status
                reasons.append(f'Current water level ({current_wl}) exceeds {label} warning level ({threshold}).')
                break
        if wl_level == 0:
            reasons.append('Current water level is within supplied warning thresholds.')

    inflow_level, inflow_status = 0, 'INSUFFICIENT_DATA'
    threshold = metadata.get('historical_95th_inflow')
    horizons = [('1d', forecast_1d), ('3d', forecast_3d), ('7d', forecast_7d)]
    if not _finite(threshold) or threshold < 0:
        reasons.append('Missing or invalid historical inflow threshold for this reservoir.')
    else:
        valid = [(h, f) for h, f in horizons if _finite(f) and f >= 0]
        triggered = [(h, f) for h, f in valid if f >= threshold]
        missing = [h for h, f in horizons if not _finite(f) or f < 0]
        if triggered:
            inflow_level, inflow_status = 2, 'ALERT'
            values = ', '.join(f'{h} ({f:.2f})' for h, f in triggered)
            reasons.append(f'Forecast inflow exceeds historical 95th percentile ({threshold:.2f}) at horizons: {values}.')
        elif not missing:
            inflow_status = 'NORMAL'
            reasons.append('Forecast inflow is within historical norms.')
        if missing:
            reasons.append('Missing, non-finite or negative forecast horizons: ' + ', '.join(missing) + '.')

    level = max(wl_level, inflow_level)
    overall = {3: 'HIGH RISK', 2: 'ALERT', 1: 'WATCH'}.get(level)
    if overall is None:
        overall = 'NORMAL' if wl_status == inflow_status == 'NORMAL' else 'UNKNOWN'
    return {'overall_status': overall, 'water_level_status': wl_status,
            'inflow_forecast_status': inflow_status, 'reason': ' | '.join(reasons)}
