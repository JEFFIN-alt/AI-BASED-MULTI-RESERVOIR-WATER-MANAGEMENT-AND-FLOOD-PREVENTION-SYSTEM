"""
Streamlit Dashboard for AI-Based Multi-Reservoir Water Management
Decision Support System (Phase 14.3 MVP)
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from data_bridge import assess_all_reservoirs, V3_PERFORMANCE

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Reservoir Decision Support Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Caching Data Load
# ---------------------------------------------------------------------------
@st.cache_data(ttl=60) # Cache for 60 seconds
def get_reservoir_data():
    """Loads all reservoir assessments from the validated data bridge."""
    return assess_all_reservoirs()

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def status_color(status: str) -> str:
    """Returns a CSS color code based on the risk status."""
    status = status.upper()
    if status == "NORMAL":
        return "#28a745"  # Green
    elif status == "WATCH":
        return "#ffc107"  # Yellow
    elif status == "ALERT":
        return "#fd7e14"  # Orange
    elif status == "HIGH RISK":
        return "#dc3545"  # Red
    elif status == "UNKNOWN":
        return "#6c757d"  # Gray
    elif status == "INSUFFICIENT_DATA":
        return "#6c757d"  # Gray
    return "#6c757d"


def create_water_level_gauge(res_data: dict) -> go.Figure:
    """Creates a gauge chart for current water level vs thresholds."""
    wl = res_data.get("water_level")
    blue = res_data.get("blue_level")
    orange = res_data.get("orange_level")
    red = res_data.get("red_level")
    frl = res_data.get("frl")
    
    if wl is None:
        # Create an empty figure if data is missing
        fig = go.Figure()
        fig.update_layout(title="Water Level Data Unavailable")
        return fig
        
    # Determine the maximum value for the gauge
    max_val = max(val for val in [wl, frl, red, orange, blue, wl * 1.1] if val is not None)
    min_val = min(val for val in [wl, blue, orange, red, frl, wl * 0.9] if val is not None)
    
    # Safely handle missing thresholds for the gauge steps
    steps = []
    if blue is not None:
        steps.append({'range': [min_val, blue], 'color': "lightgreen"})
    if blue is not None and orange is not None:
        steps.append({'range': [blue, orange], 'color': "yellow"})
    if orange is not None and red is not None:
        steps.append({'range': [orange, red], 'color': "orange"})
    if red is not None:
        steps.append({'range': [red, max_val], 'color': "red"})

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=wl,
        title={'text': "Current Water Level (m)"},
        gauge={
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': "darkblue"},
            'steps': steps,
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': red if red is not None else max_val
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def create_forecast_bar_chart(res_data: dict) -> go.Figure:
    """Creates a bar chart for V3 point forecasts vs historical threshold."""
    forecasts = [
        res_data.get("forecast_1d"),
        res_data.get("forecast_3d"),
        res_data.get("forecast_7d")
    ]
    labels = ['1-Day', '3-Day', '7-Day']
    
    if all(f is None for f in forecasts):
        fig = go.Figure()
        fig.update_layout(title="Forecast Data Unavailable")
        return fig

    # Replace Nones with 0 for plotting, but we should handle this cleaner if we want
    plot_forecasts = [f if f is not None else 0 for f in forecasts]
    threshold = res_data.get("historical_95th_inflow")

    fig = go.Figure(data=[
        go.Bar(name='Point Forecast', x=labels, y=plot_forecasts, marker_color='dodgerblue')
    ])
    
    if threshold is not None:
        fig.add_hline(
            y=threshold, 
            line_dash="dash", 
            line_color="red", 
            annotation_text=f"Historical 95th Percentile ({threshold:.2f})",
            annotation_position="top left"
        )

    fig.update_layout(
        title="Validated V3 Test Forecast (Inflow Point Forecasts)",
        yaxis_title="Inflow (MCM/day)",
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False
    )
    return fig

# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------
def live_dashboard():
    st.title("AI-Based Multi-Reservoir Water Management")
    st.subheader("Decision Support Dashboard")
    st.markdown("*LSTM V3 Inflow Forecasting + Deterministic Risk Assessment*")
    
    # Load data
    try:
        data = get_reservoir_data()
    except Exception as e:
        st.error(f"Failed to load reservoir data: {e}")
        st.stop()

    if not data:
        st.error("No reservoir data found.")
        st.stop()

    df = pd.DataFrame(data)

    # -----------------------------------------------------------------------
    # TOP SUMMARY
    # -----------------------------------------------------------------------
    st.markdown("---")
    
    # Calculate aggregates safely
    total_res = len(df)
    normal_count = len(df[df['overall_status'] == 'NORMAL'])
    watch_count = len(df[df['overall_status'] == 'WATCH'])
    alert_count = len(df[df['overall_status'] == 'ALERT'])
    high_risk_count = len(df[df['overall_status'] == 'HIGH RISK'])
    unknown_count = len(df[~df['overall_status'].isin(['NORMAL', 'WATCH', 'ALERT', 'HIGH RISK'])])
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Monitored", total_res)
    with col2:
        st.metric("NORMAL", normal_count)
    with col3:
        st.metric("WATCH", watch_count)
    with col4:
        st.metric("ALERT", alert_count)
    with col5:
        st.metric("HIGH RISK", high_risk_count)

    if unknown_count > 0:
         st.warning(f"{unknown_count} reservoir(s) have UNKNOWN or INSUFFICIENT DATA status.")

    # -----------------------------------------------------------------------
    # SYSTEM EXPLANATION & LIMITATIONS (Sidebar)
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.header("System Architecture")
        st.markdown("""
        **16 software-monitored reservoirs**
        
        **Data Flow:**
        1. Historical Reservoir Data
        2. LSTM V3 Model
        3. 1d / 3d / 7d Inflow Point Forecasts
        4. Point-Forecast Warning Engine
        5. Risk Status
        6. Operator Decision Support
        """)
        
        st.markdown("---")
        
        st.header("Model Performance")
        st.markdown("**Validated Test Performance (LSTM V3):**")
        st.markdown(f"- 1-day R² = {V3_PERFORMANCE['target_1d']['R2']}")
        st.markdown(f"- 3-day R² = {V3_PERFORMANCE['target_3d']['R2']}")
        st.markdown(f"- 7-day R² = {V3_PERFORMANCE['target_7d']['R2']}")
        st.caption("*Spatial GNN experiments were evaluated during development but did not outperform the final temporal LSTM V3 model.*")

        st.markdown("---")

        with st.expander("⚠️ Scientific Limitations", expanded=False):
            st.markdown("""
            **This system DOES NOT:**
            - Predict downstream floods
            - Predict future reservoir storage volumes
            - Predict overtopping dates
            - Calculate optimal release volumes
            - Autonomously control real dams
            - Use GNN forecasting
            - Claim that the physical prototype is a scaled hydraulic model
            
            *V3 outputs are point forecasts for the specified day, not cumulative averages.*
            """)
            
        st.markdown("---")
        st.header("Future Phase: Deterministic Control Policy")
        st.markdown("""
        *(Not Currently Operational)*
        - Risk Status → bounded gate position
        """)


    # -----------------------------------------------------------------------
    # MAIN CONTENT: RESERVOIR OVERVIEW
    # -----------------------------------------------------------------------
    st.header("Reservoir Overview")
    
    # Create a nice styled dataframe for the overview
    display_cols = ['reservoir', 'overall_status', 'water_level_status', 'inflow_forecast_status', 'water_level']
    display_df = df[display_cols].copy()
    
    # Basic styling function for Pandas Styler
    def style_status(val):
        color = status_color(val)
        # Using lighter colors for background, or just coloring text. Let's color text.
        return f'color: {color}; font-weight: bold;'
        
    st.dataframe(
        display_df.style.map(style_status, subset=['overall_status', 'water_level_status', 'inflow_forecast_status']),
        use_container_width=True,
        hide_index=True
    )

    # -----------------------------------------------------------------------
    # RESERVOIR SELECTION
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.header("Selected Reservoir Detail")
    
    selected_res_name = st.selectbox("Select a reservoir for detailed analysis:", df['reservoir'].tolist())
    
    # Get the data for the selected reservoir
    res_data = df[df['reservoir'] == selected_res_name].iloc[0].to_dict()
    
    # Layout for details
    detail_col1, detail_col2 = st.columns([1, 1])
    
    with detail_col1:
        st.subheader("Current State")
        st.markdown(f"**Telemetry Date:** {res_data.get('telemetry_date') or 'DATA UNAVAILABLE'}")
        
        # Display Status
        wl_status = res_data.get('water_level_status', 'UNKNOWN')
        st.markdown(f"**Water-Level Status:** <span style='color:{status_color(wl_status)}; font-weight:bold;'>{wl_status}</span>", unsafe_allow_html=True)
        
        # Display threshold values cleanly
        st.markdown("**Warning Thresholds:**")
        t_col1, t_col2, t_col3 = st.columns(3)
        t_col1.metric("Blue", f"{res_data.get('blue_level')} m" if pd.notna(res_data.get('blue_level')) else "N/A")
        t_col2.metric("Orange", f"{res_data.get('orange_level')} m" if pd.notna(res_data.get('orange_level')) else "N/A")
        t_col3.metric("Red", f"{res_data.get('red_level')} m" if pd.notna(res_data.get('red_level')) else "N/A")
        
        # Gauge Chart
        st.plotly_chart(create_water_level_gauge(res_data), use_container_width=True)

    with detail_col2:
        st.subheader("V3 Forecast")
        st.markdown(f"**Forecast Date:** {res_data.get('forecast_date') or 'DATA UNAVAILABLE'}")
        
        inflow_status = res_data.get('inflow_forecast_status', 'UNKNOWN')
        st.markdown(f"**Inflow-Forecast Status:** <span style='color:{status_color(inflow_status)}; font-weight:bold;'>{inflow_status}</span>", unsafe_allow_html=True)
        
        hist_thresh = res_data.get('historical_95th_inflow')
        if hist_thresh is not None:
             st.markdown(f"**Historical Extreme-Inflow Threshold:** {hist_thresh:.2f} MCM/day")
             st.caption("*This threshold was calculated using training-period historical inflow data.*")
        else:
             st.markdown("**Historical Extreme-Inflow Threshold:** DATA UNAVAILABLE")
             
        # Bar Chart
        st.plotly_chart(create_forecast_bar_chart(res_data), use_container_width=True)

    # -----------------------------------------------------------------------
    # RISK ASSESSMENT EXPLANATION
    # -----------------------------------------------------------------------
    st.markdown("### Risk Assessment")
    
    overall_status = res_data.get('overall_status', 'UNKNOWN')
    st.markdown(f"**Overall Status:** <span style='color:{status_color(overall_status)}; font-weight:bold; font-size:1.2em;'>{overall_status}</span>", unsafe_allow_html=True)
    
    st.markdown("**Reason:**")
    st.info(res_data.get('reason', 'No reason provided.'))

    # -----------------------------------------------------------------------
    # PROTOTYPE PLACEHOLDER
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.header("Physical Prototype — Not Connected")
    st.markdown("*This section is reserved for future hardware integration (Phase 14.4+).*")
    
    p_col1, p_col2, p_col3 = st.columns(3)
    p_col1.metric("ESP32", "Not connected")
    p_col2.metric("Sensor", "Not connected")
    p_col3.metric("Servo", "Not connected")


def main():
    st.sidebar.title("Navigation")
    mode = st.sidebar.radio(
        "Select Mode:",
        ("Live Monitoring MVP", "Interactive Simulator Laboratory")
    )

    if mode == "Live Monitoring MVP":
        live_dashboard()
    else:
        from simulation_page import render_simulation_page
        render_simulation_page()

if __name__ == "__main__":
    main()
