import pandas as pd
from typing import Dict, List, Tuple
from pathlib import Path

# Mapping of virtual names to the real historical reservoirs we draw inflow data from
# This is a SIMULATION ASSUMPTION mapping.
SOURCE_MAPPING = {
    "Virtual Reservoir A": "Anayirankal",
    "Virtual Reservoir B": "Ponmudi",
    "Virtual Reservoir C": "Idamalayar",
    "Virtual Reservoir D": "Idukki"
}

def load_scenario(scenario_name: str, clean_csv_path: str, v3_csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads 30-day slices of historical data and V3 predictions for the requested scenario.
    
    1. NORMAL INFLOW: e.g., March 2025 (Dry/stable)
    2. HEAVY INFLOW: e.g., June 2025 (Monsoon)
    3. EXTREME INFLOW: e.g., August 2024 (Flood anomaly, since V3 test set is 2025, let's pick a very wet period in 2025 like July)
    4. MULTI-RESERVOIR EVENT: Staggered event in 2025.
    
    We must use the V3 test set date range: 2025-01-01 -> 2025-08-22.
    """
    df_actual = pd.read_csv(clean_csv_path, parse_dates=["date"])
    df_v3 = pd.read_csv(v3_csv_path, parse_dates=["date"])
    
    # Filter to only the mapped source reservoirs
    sources = list(SOURCE_MAPPING.values())
    df_actual = df_actual[df_actual["reservoir"].isin(sources)]
    df_v3 = df_v3[df_v3["reservoir"].isin(sources)]
    
    # Define date ranges for scenarios (30 days each)
    scenario_dates = {
        "NORMAL": ("2025-02-01", "2025-03-02"),       # Typically dry
        "HEAVY": ("2025-06-01", "2025-06-30"),        # Monsoon onset
        "EXTREME": ("2025-07-15", "2025-08-13"),      # Peak monsoon / extreme event
        "MULTI_RESERVOIR": ("2025-07-01", "2025-07-30") # Mixed heavy rain
    }
    
    start_date, end_date = scenario_dates.get(scenario_name, ("2025-01-01", "2025-01-30"))
    
    mask_actual = (df_actual["date"] >= start_date) & (df_actual["date"] <= end_date)
    mask_v3 = (df_v3["date"] >= start_date) & (df_v3["date"] <= end_date)
    
    sliced_actual = df_actual[mask_actual].sort_values("date")
    sliced_v3 = df_v3[mask_v3].sort_values("date")
    
    return sliced_actual, sliced_v3

def get_timestep_inflows(sliced_actual: pd.DataFrame, current_date: pd.Timestamp) -> Dict[str, float]:
    """Gets the actual inflow for each virtual reservoir on a specific date."""
    inflows = {}
    day_data = sliced_actual[sliced_actual["date"] == current_date]
    
    for virt_name, real_name in SOURCE_MAPPING.items():
        row = day_data[day_data["reservoir"] == real_name]
        if not row.empty:
            # Prevent negative inflows which are physically invalid (evaporation is separate)
            val = float(row["inflow"].iloc[0])
            inflows[virt_name] = max(0.0, val) if pd.notna(val) else 0.0
        else:
            inflows[virt_name] = 0.0
            
    return inflows

def get_timestep_forecasts(sliced_v3: pd.DataFrame, current_date: pd.Timestamp) -> Dict[str, Dict[str, float]]:
    """Gets the V3 point forecasts available ON current_date."""
    forecasts = {}
    day_data = sliced_v3[sliced_v3["date"] == current_date]
    
    for virt_name, real_name in SOURCE_MAPPING.items():
        row = day_data[day_data["reservoir"] == real_name]
        if not row.empty:
            forecasts[virt_name] = {
                "forecast_1d": float(row["target_1d_prediction"].iloc[0]),
                "forecast_3d": float(row["target_3d_prediction"].iloc[0]),
                "forecast_7d": float(row["target_7d_prediction"].iloc[0])
            }
        else:
            forecasts[virt_name] = {
                "forecast_1d": None,
                "forecast_3d": None,
                "forecast_7d": None
            }
            
    return forecasts
