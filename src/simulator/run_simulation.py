import os
import sys
import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# Resolve paths
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.simulator.scenarios import load_scenario
from src.simulator.controllers import ReactiveBaselineController, ForecastAwareController
from src.simulator.engine import SimulationEngine
from src.simulator.metrics import compute_metrics, compare_runs

CONFIG_PATH = _PROJECT_ROOT / "configs" / "simulation" / "four_reservoir_demo.json"
HIST_THRESH_PATH = _PROJECT_ROOT / "data" / "processed" / "historical_inflow_thresholds.json"
CLEAN_CSV_PATH = _PROJECT_ROOT / "data" / "processed" / "kerala_reservoir_clean.csv"
V3_CSV_PATH = _PROJECT_ROOT / "results" / "lstm_pytorch_v3_logtarget" / "test_predictions_original_units.csv"
OUT_DIR = _PROJECT_ROOT / "results" / "phase14_simulation"

def run_experiment():
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
        
    engine = SimulationEngine(str(CONFIG_PATH), str(HIST_THRESH_PATH))
    engine.config_path = str(CONFIG_PATH) # explicitly set for VirtualCascade
    
    scenarios = ["NORMAL", "HEAVY", "EXTREME", "MULTI_RESERVOIR"]
    controllers = {
        "Reactive Baseline": ReactiveBaselineController(config),
        "Forecast-Aware": ForecastAwareController(config)
    }
    
    all_metrics = []
    
    for scenario in scenarios:
        print(f"--- Running Scenario: {scenario} ---")
        sliced_actual, sliced_v3 = load_scenario(scenario, str(CLEAN_CSV_PATH), str(V3_CSV_PATH))
        
        scenario_metrics = {}
        
        for ctrl_name, controller in controllers.items():
            print(f"Executing {ctrl_name}...")
            
            df_results = engine.run(scenario, sliced_actual, sliced_v3, controller)
            
            # Save raw logs
            out_file = OUT_DIR / f"sim_log_{scenario}_{ctrl_name.replace(' ', '_')}.csv"
            df_results.to_csv(out_file, index=False)
            
            # Calculate metrics
            mets = compute_metrics(df_results, config)
            scenario_metrics[ctrl_name] = mets
            
            # Plot storage trajectories
            fig, ax = plt.subplots(figsize=(10, 6))
            for res_name in config["topology"]["cascade_order"]:
                res_data = df_results[df_results["reservoir"] == res_name]
                ax.plot(res_data["date"], res_data["storage_after"], label=res_name)
            
            ax.set_title(f"[{scenario}] {ctrl_name} - Reservoir Storage Trajectories")
            ax.set_ylabel("Storage (MCM)")
            ax.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(OUT_DIR / f"plot_{scenario}_{ctrl_name.replace(' ', '_')}.png")
            plt.close()
            
        # Compare
        comp_df = compare_runs(scenario_metrics["Reactive Baseline"], scenario_metrics["Forecast-Aware"])
        comp_df.insert(0, "Scenario", scenario)
        all_metrics.append(comp_df)
        print(f"Completed {scenario}")

    final_comparison = pd.concat(all_metrics, ignore_index=True)
    final_comparison.to_csv(OUT_DIR / "comparison_metrics.csv", index=False)
    print("\nAll experiments complete. Results saved to:", OUT_DIR)
    
if __name__ == "__main__":
    run_experiment()
