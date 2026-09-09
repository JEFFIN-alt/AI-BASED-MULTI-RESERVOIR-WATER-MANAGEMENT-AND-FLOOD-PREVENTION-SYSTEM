"""
data_bridge.py — Adapter that connects validated management-layer loaders
and V3 prediction artifacts to the Streamlit dashboard.

This module does NOT modify any source data. It reads, adapts naming
conventions, and returns clean Python structures for the UI.
"""

import os
import sys
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional

# ---------------------------------------------------------------------------
# Resolve project root so imports work regardless of working directory
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent  # src/dashboard -> src -> project root

sys.path.insert(0, str(_PROJECT_ROOT))

from src.management.management_data_loader import load_reservoir_metadata, get_current_state
from src.management.risk_engine import assess_risk

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
LIVE_JSON = _PROJECT_ROOT / "data" / "raw" / "reservoir" / "Kerala-Dam-Water-Levels" / "live.json"
IRRIGATION_LIVE_JSON = _PROJECT_ROOT / "data" / "raw" / "reservoir" / "Kerala-Dam-Water-Levels" / "irrigation_live.json"
HISTORIC_DATA_DIR = _PROJECT_ROOT / "data" / "raw" / "reservoir" / "Kerala-Dam-Water-Levels" / "historic_data"
HISTORICAL_THRESHOLDS = _PROJECT_ROOT / "data" / "processed" / "historical_inflow_thresholds.json"
V3_PREDICTIONS = _PROJECT_ROOT / "results" / "lstm_pytorch_v3_logtarget" / "test_predictions_original_units.csv"

# The 16 reservoirs that V3 was trained/evaluated on
V3_RESERVOIRS = [
    "Anathode", "Anayirankal", "Banasura Sagar", "Chenkulam",
    "Idamalayar", "Idukki", "Kakkayam", "Kallarkutty",
    "Kundala", "Mattupetty", "Moozhiyar", "Pamba",
    "Pambla", "Ponmudi", "Poringalkuthu", "Sholayar"
]

# V3 validated test-set performance (from V3_FINAL_AUDIT_REPORT.md)
V3_PERFORMANCE = {
    "target_1d": {"MAE": 1.837, "RMSE": 3.031, "R2": 0.759},
    "target_3d": {"MAE": 2.095, "RMSE": 3.522, "R2": 0.654},
    "target_7d": {"MAE": 2.703, "RMSE": 4.573, "R2": 0.501},
}


def load_all_metadata() -> Dict[str, Dict[str, Any]]:
    """Load metadata for all reservoirs. Keys are lowercase."""
    return load_reservoir_metadata(
        str(LIVE_JSON), str(IRRIGATION_LIVE_JSON), str(HISTORICAL_THRESHOLDS)
    )


def load_current_state(reservoir_name: str) -> Dict[str, Any]:
    """Load the latest telemetry for a specific reservoir."""
    # Historic JSON filenames use underscores for spaces
    filename = reservoir_name.replace(" ", "_") + ".json"
    path = HISTORIC_DATA_DIR / filename
    return get_current_state(str(path))


def load_v3_predictions() -> pd.DataFrame:
    """Load the validated V3 test predictions."""
    try:
        df = pd.read_csv(str(V3_PREDICTIONS), parse_dates=["date"])
        return df
    except FileNotFoundError:
        return pd.DataFrame()


def get_latest_v3_forecasts(predictions_df: pd.DataFrame) -> Dict[str, Dict[str, Optional[float]]]:
    """
    For each reservoir, get the LATEST available V3 test prediction.
    Returns dict keyed by reservoir name (original case).
    """
    if predictions_df.empty:
        return {}

    forecasts = {}
    for reservoir in V3_RESERVOIRS:
        res_df = predictions_df[predictions_df["reservoir"] == reservoir]
        if res_df.empty:
            forecasts[reservoir] = {
                "date": None,
                "forecast_1d": None, "forecast_3d": None, "forecast_7d": None,
                "actual_1d": None, "actual_3d": None, "actual_7d": None,
            }
            continue

        latest = res_df.sort_values("date").iloc[-1]
        forecasts[reservoir] = {
            "date": str(latest["date"].date()) if pd.notna(latest["date"]) else None,
            "forecast_1d": float(latest["target_1d_prediction"]) if pd.notna(latest["target_1d_prediction"]) else None,
            "forecast_3d": float(latest["target_3d_prediction"]) if pd.notna(latest["target_3d_prediction"]) else None,
            "forecast_7d": float(latest["target_7d_prediction"]) if pd.notna(latest["target_7d_prediction"]) else None,
            "actual_1d": float(latest["target_1d_actual"]) if pd.notna(latest["target_1d_actual"]) else None,
            "actual_3d": float(latest["target_3d_actual"]) if pd.notna(latest["target_3d_actual"]) else None,
            "actual_7d": float(latest["target_7d_actual"]) if pd.notna(latest["target_7d_actual"]) else None,
        }
    return forecasts


def assess_all_reservoirs() -> List[Dict[str, Any]]:
    """
    Run the validated risk engine on all 16 V3 reservoirs.
    Returns a list of result dicts, one per reservoir.
    """
    metadata = load_all_metadata()
    predictions_df = load_v3_predictions()
    latest_forecasts = get_latest_v3_forecasts(predictions_df)

    results = []
    for reservoir in V3_RESERVOIRS:
        state = load_current_state(reservoir)
        meta = metadata.get(reservoir.lower().strip(), {})
        fc = latest_forecasts.get(reservoir, {})

        risk = assess_risk(
            state, meta,
            fc.get("forecast_1d"),
            fc.get("forecast_3d"),
            fc.get("forecast_7d"),
        )

        results.append({
            "reservoir": reservoir,
            "water_level": state.get("waterLevel"),
            "live_storage": state.get("liveStorage"),
            "telemetry_date": state.get("date"),
            "forecast_date": fc.get("date"),
            "forecast_1d": fc.get("forecast_1d"),
            "forecast_3d": fc.get("forecast_3d"),
            "forecast_7d": fc.get("forecast_7d"),
            "actual_1d": fc.get("actual_1d"),
            "actual_3d": fc.get("actual_3d"),
            "actual_7d": fc.get("actual_7d"),
            "blue_level": meta.get("blueLevel"),
            "orange_level": meta.get("orangeLevel"),
            "red_level": meta.get("redLevel"),
            "frl": meta.get("FRL"),
            "historical_95th_inflow": meta.get("historical_95th_inflow"),
            "overall_status": risk["overall_status"],
            "water_level_status": risk["water_level_status"],
            "inflow_forecast_status": risk["inflow_forecast_status"],
            "reason": risk["reason"],
        })

    return results
