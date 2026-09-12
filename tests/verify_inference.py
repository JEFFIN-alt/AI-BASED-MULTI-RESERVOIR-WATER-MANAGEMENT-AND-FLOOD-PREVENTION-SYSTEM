import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.modeling.inference import LiveForecaster
from src.dashboard.data_bridge import assess_all_reservoirs

def create_mock_history(valid=True, consecutive=True, missing_feature=False, has_nan=False):
    dates = pd.date_range(start="2026-09-01", periods=7, freq='D')
    if not consecutive:
        dates = [pd.Timestamp("2026-09-01") + pd.Timedelta(days=i*2) for i in range(7)]
        
    df = pd.DataFrame({
        "date": dates,
        "water_level": np.random.uniform(900, 1000, 7),
        "live_storage": np.random.uniform(50, 150, 7),
        "inflow": np.random.uniform(5, 50, 7),
        "rainfall": np.random.uniform(0, 20, 7),
        "total_outflow": np.random.uniform(5, 40, 7)
    })
    
    if missing_feature:
        df = df.drop(columns=["inflow"])
        
    if has_nan:
        df.loc[3, "water_level"] = np.nan
        
    return df

def test_inference():
    print("Testing LiveForecaster initialization...")
    forecaster = LiveForecaster(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    print("SUCCESS: Model and scalers loaded.\n")
    
    print("Testing Valid 7-Day History...")
    df_valid = create_mock_history()
    res = forecaster.predict(df_valid)
    print(f"SUCCESS: Predictions generated: {res}\n")
    
    print("Testing Missing Feature...")
    df_missing = create_mock_history(missing_feature=True)
    try:
        forecaster.predict(df_missing)
        print("FAIL: Should have raised ValueError for missing feature.\n")
    except ValueError as e:
        print(f"SUCCESS: Caught missing feature: {e}\n")
        
    print("Testing Non-Consecutive Dates...")
    df_non_consec = create_mock_history(consecutive=False)
    try:
        forecaster.predict(df_non_consec)
        print("FAIL: Should have raised ValueError for non-consecutive dates.\n")
    except ValueError as e:
        print(f"SUCCESS: Caught non-consecutive dates: {e}\n")
        
    print("Testing NaN Values...")
    df_nan = create_mock_history(has_nan=True)
    try:
        forecaster.predict(df_nan)
        print("FAIL: Should have raised ValueError for NaN.\n")
    except ValueError as e:
        print(f"SUCCESS: Caught NaN: {e}\n")

def test_data_bridge():
    print("Testing Data Bridge fallback and live inference on real data...")
    results = assess_all_reservoirs()
    sources = [r['source'] for r in results]
    live_count = sum(1 for s in sources if s == "LIVE_INFERENCE")
    fallback_count = sum(1 for s in sources if s == "VALIDATED_TEST_PREDICTION")
    unavail_count = sum(1 for s in sources if s == "UNAVAILABLE")
    
    print(f"Total Reservoirs: {len(results)}")
    print(f"Live Inference count: {live_count}")
    print(f"Fallback count: {fallback_count}")
    print(f"Unavailable count: {unavail_count}")
    print("\nSUCCESS: End-to-end data bridge executed.")

if __name__ == "__main__":
    test_inference()
    test_data_bridge()
