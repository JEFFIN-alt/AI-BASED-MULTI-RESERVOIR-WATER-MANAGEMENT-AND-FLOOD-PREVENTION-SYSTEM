import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
import json

# Setup paths to simulator
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.simulator.engine import SimulationEngine
from src.simulator.controllers import ReactiveBaselineController, ForecastAwareController
from src.simulator.metrics import compute_metrics, compare_runs

# Use the base configuration to initialize UI defaults
CONFIG_PATH = _PROJECT_ROOT / "configs" / "simulation" / "four_reservoir_demo.json"
HIST_THRESH_PATH = _PROJECT_ROOT / "data" / "processed" / "historical_inflow_thresholds.json"
with open(CONFIG_PATH, 'r') as f:
    DEFAULT_CONFIG = json.load(f)

def render_simulation_page():
    st.header("Interactive Simulator Laboratory (Phase 14.4)")
    st.markdown("""
    **VIRTUAL STRESS-TEST ENVIRONMENT**
    - This interface allows you to define *synthetic* scenarios by directly specifying inflow.
    - Rainfall, humidity, and storm multipliers are *synthetic simulation assumptions* used purely to multiply inflow for stress testing. They do NOT represent a calibrated physical hydrological model.
    - All predictions are frozen; V3 artifacts are strictly protected.
    """)
    
    with st.expander("🛠️ Simulation Configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("1. Inflow & Weather (Synthetic Modifiers)")
            
            # Allow individual base inflows
            st.markdown("**Base Inflows (MCM/day)**")
            base_inflows = {}
            bc1, bc2 = st.columns(2)
            with bc1:
                base_inflows["Virtual Reservoir A"] = st.number_input("Reservoir A Base Inflow", value=10.0, min_value=0.0)
                base_inflows["Virtual Reservoir C"] = st.number_input("Reservoir C Base Inflow", value=100.0, min_value=0.0)
            with bc2:
                base_inflows["Virtual Reservoir B"] = st.number_input("Reservoir B Base Inflow", value=20.0, min_value=0.0)
                base_inflows["Virtual Reservoir D"] = st.number_input("Reservoir D Base Inflow", value=200.0, min_value=0.0)
            
            st.markdown("---")
            rainfall = st.slider("Rainfall Multiplier (1.0 = Normal, 2.0 = Heavy)", 1.0, 5.0, 1.0, 0.1)
            humidity = st.slider("Humidity (%)", 0.0, 100.0, 50.0, 5.0)
            
        with col2:
            st.subheader("2. Storm Event (Synthetic)")
            storm_start = st.number_input("Storm Start Day", min_value=1, max_value=30, value=15)
            storm_duration = st.number_input("Storm Duration (days)", min_value=0, max_value=30, value=5)
            storm_mult = st.number_input("Storm Inflow Multiplier", min_value=1.0, max_value=10.0, value=2.0)
            
            st.subheader("3. Initial Storage Conditions")
            start_storage_pct = st.slider("Initial Storage (% of Capacity) - All Reservoirs", 0.0, 100.0, 50.0, 1.0)
            
            st.subheader("4. Controller Setup")
            mode = st.radio("Select Test Mode:", ["Reactive Baseline", "Forecast-Aware", "Compare Both"])
            
    if st.button("🚀 Run Stress Test"):
        run_interactive_simulation(base_inflows, rainfall, humidity, storm_start, storm_duration, storm_mult, start_storage_pct, mode)


def run_interactive_simulation(base_inflows, rainfall, humidity, storm_start, storm_duration, storm_mult, start_storage_pct, mode):
    st.markdown("---")
    st.subheader("Simulation Results")
    
    # Clone default config and apply overrides
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    for res_name in config["reservoirs"]:
        cap = config["reservoirs"][res_name]["capacity_mcm"]["value"]
        config["reservoirs"][res_name]["initial_storage_mcm"]["value"] = cap * (start_storage_pct / 100.0)
        
    synthetic_params = {
        "duration_days": 30,
        "base_inflows": base_inflows,
        "rainfall": rainfall,
        "humidity": humidity,
        "storm_start": storm_start - 1, # 0-indexed in code
        "storm_duration": storm_duration,
        "storm_multiplier": storm_mult
    }
    
    engine = SimulationEngine(config, str(HIST_THRESH_PATH)) # It loads thresholds via path if we don't pass the dict, let's fix that wrapper
    with open(HIST_THRESH_PATH, 'r') as f:
        thresholds = json.load(f)
    engine = SimulationEngine(config, thresholds)
    
    controllers = {}
    if mode in ["Reactive Baseline", "Compare Both"]:
        controllers["Reactive Baseline"] = ReactiveBaselineController(config)
    if mode in ["Forecast-Aware", "Compare Both"]:
        controllers["Forecast-Aware"] = ForecastAwareController(config)
        
    results = {}
    metrics_list = []
    
    for ctrl_name, controller in controllers.items():
        with st.spinner(f"Running {ctrl_name}..."):
            df = engine.run(
                scenario_name="Synthetic", 
                sliced_actual=None, 
                sliced_v3=None, 
                controller=controller,
                is_synthetic=True,
                synthetic_params=synthetic_params
            )
            results[ctrl_name] = df
            metrics = compute_metrics(df, config)
            metrics_list.append((ctrl_name, metrics))
            
    # Display Results
    if mode == "Compare Both":
        st.markdown("### Metrics Comparison")
        comp_df = compare_runs(metrics_list[0][1], metrics_list[1][1])
        st.dataframe(comp_df, use_container_width=True)
        
    st.info("Synthetic scenario — V3 forecast unavailable; Forecast-Aware controller uses safe fallback behavior.")
        
    # Plotting
    for ctrl_name, df in results.items():
        st.markdown(f"### {ctrl_name} Trajectories")
        plot_simulation_results(df, config, synthetic_params, ctrl_name)
        
def plot_simulation_results(df, config, synthetic_params, key_name):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        subplot_titles=("Storage (%)", "Inflow (MCM/day)", "Gate Position (%) & Downstream Flow"),
                        vertical_spacing=0.1)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, res_name in enumerate(config["topology"]["cascade_order"]):
        res_data = df[df["reservoir"] == res_name]
        cap = config["reservoirs"][res_name]["capacity_mcm"]["value"]
        
        # 1. Storage
        fig.add_trace(go.Scatter(x=res_data["date"], y=(res_data["storage_after"]/cap)*100, 
                                 name=f"{res_name} Storage", line=dict(color=colors[i])), row=1, col=1)
        
        # 2. Inflow
        fig.add_trace(go.Scatter(x=res_data["date"], y=res_data["inflow"], 
                                 name=f"{res_name} Inflow", line=dict(color=colors[i], dash='dash')), row=2, col=1)
                                 
        # 3. Gate Position
        fig.add_trace(go.Scatter(x=res_data["date"], y=res_data["gate_position"], 
                                 name=f"{res_name} Gate", line=dict(color=colors[i], dash='dot')), row=3, col=1)
                                 
    # Add storm window shading
    start_date = pd.Timestamp("2025-01-01") + pd.Timedelta(days=synthetic_params["storm_start"])
    end_date = start_date + pd.Timedelta(days=synthetic_params["storm_duration"])
    
    for row in range(1, 4):
        fig.add_vrect(x0=start_date, x1=end_date, fillcolor="rgba(255, 0, 0, 0.1)", layer="below", line_width=0, annotation_text="Storm Window", row=row, col=1)
        
    # Add downstream threshold on row 3
    terminal_res = config["topology"]["cascade_order"][-1]
    terminal_data = df[df["reservoir"] == terminal_res]
    fig.add_trace(go.Scatter(x=terminal_data["date"], y=terminal_data["downstream_flow"], 
                             name="Downstream Flow", line=dict(color='black', width=2)), row=3, col=1)
    fig.add_hline(y=config["topology"]["downstream_capacity"]["value"], line_dash="dash", line_color="red", row=3, col=1, annotation_text="Flood Limit")

    fig.update_layout(height=800, title_text="Virtual Cascade Dynamics", showlegend=True)
    st.plotly_chart(fig, use_container_width=True, key=f"trajectory_plot_{key_name}")
