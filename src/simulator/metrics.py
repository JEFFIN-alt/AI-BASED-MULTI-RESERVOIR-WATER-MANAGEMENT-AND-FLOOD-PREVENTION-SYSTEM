import pandas as pd
from typing import Dict, Any

def compute_metrics(results_df: pd.DataFrame, cascade_config: Dict[str, Any]) -> Dict[str, float]:
    """
    Computes summary metrics for a simulation run from the daily logs.
    """
    if results_df.empty:
        return {}

    metrics = {}
    
    # 1. Reservoir Safety
    metrics["total_overflow_events"] = int(results_df["overflow_indicator"].sum())
    
    # Threshold violations (Storage > Red Level (95%))
    red_violations = 0
    for res_name in cascade_config["reservoirs"].keys():
        res_data = results_df[results_df["reservoir"] == res_name]
        cap = cascade_config["reservoirs"][res_name]["capacity_mcm"]["value"]
        red_level = cap * 0.95
        red_violations += len(res_data[res_data["storage_after"] > red_level])
        
    metrics["total_red_violations"] = red_violations

    # 2. Water Conservation
    metrics["total_release_mcm"] = float(results_df["release"].sum())
    metrics["total_water_retained_mcm"] = float(results_df.groupby("reservoir")["storage_after"].last().sum())
    metrics["avg_storage_utilization_pct"] = float((results_df["storage_after"] / results_df.apply(
        lambda r: cascade_config["reservoirs"][r["reservoir"]]["capacity_mcm"]["value"], axis=1
    )).mean() * 100)

    # 3. Downstream Safety
    downstream_cap = cascade_config["topology"]["downstream_capacity"]["value"]
    # Downstream flow is only meaningful at the terminal reservoir (D)
    terminal_res = cascade_config["topology"]["cascade_order"][-1]
    terminal_data = results_df[results_df["reservoir"] == terminal_res]
    
    metrics["peak_downstream_flow_mcm"] = float(terminal_data["downstream_flow"].max())
    metrics["downstream_capacity_violations"] = int(len(terminal_data[terminal_data["downstream_flow"] > downstream_cap]))

    return metrics

def compare_runs(baseline_metrics: Dict[str, float], forecast_metrics: Dict[str, float]) -> pd.DataFrame:
    """
    Compares Baseline vs Forecast-Aware runs and calculates absolute/percentage differences.
    """
    comparison = []
    
    for key in baseline_metrics.keys():
        b_val = baseline_metrics[key]
        f_val = forecast_metrics[key]
        
        abs_diff = f_val - b_val
        
        # Calculate percentage improvement (handling div by zero). 
        # Lower is better for violations/overflow/peak flow. Higher is better for retention.
        pct_imp = 0.0
        if b_val != 0:
            if key in ["total_overflow_events", "total_red_violations", "downstream_capacity_violations", "peak_downstream_flow_mcm"]:
                pct_imp = ((b_val - f_val) / b_val) * 100.0 # Positive means reduction in bad things
            else:
                pct_imp = ((f_val - b_val) / abs(b_val)) * 100.0 # Positive means increase
                
        comparison.append({
            "Metric": key,
            "Baseline": b_val,
            "Forecast-Aware": f_val,
            "Absolute Diff": abs_diff,
            "Improvement (%)": pct_imp
        })
        
    return pd.DataFrame(comparison)
