"""
Streamlit Dashboard for AI-Based Multi-Reservoir Water Management
Decision Support System — 3D Digital Twin Interface
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.graph_objects as go
import json
import time
from pathlib import Path

from data_bridge import assess_all_reservoirs, V3_PERFORMANCE
from sim_bridge import SimBridge
from twin_component.state_adapter import adapt_state_for_twin

# Setup paths
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
CONFIG_PATH = _PROJECT_ROOT / "configs" / "simulation" / "four_reservoir_demo.json"
THRESH_PATH = _PROJECT_ROOT / "data" / "processed" / "historical_inflow_thresholds.json"

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Reservoir Command Center",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""<style>
    .block-container{padding-top:1rem;padding-bottom:0}
    [data-testid="stAppViewContainer"]{background:#0a0e17}
    header[data-testid="stHeader"]{background:rgba(10,14,23,.9)}
    [data-testid="stSidebar"]{background:#0d1220}
    [data-testid="stSidebar"] .stMarkdown p{color:#b0bec5}
    iframe{border:none!important;border-radius:8px}
    .metric-card {
        background:rgba(20,28,45,0.8);
        border: 1px solid rgba(100,140,200,0.2);
        padding:10px;
        border-radius:6px;
        margin-bottom:10px;
    }
</style>""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "sim_bridge" not in st.session_state:
    st.session_state.sim_bridge = SimBridge(str(CONFIG_PATH), str(THRESH_PATH))

if "sim_running" not in st.session_state:
    st.session_state.sim_running = False

if "sim_tick" not in st.session_state:
    st.session_state.sim_tick = 0

if "sim_history" not in st.session_state:
    st.session_state.sim_history = []
    
@st.cache_data(ttl=60)
def get_reservoir_data():
    return assess_all_reservoirs()

def reset_simulation():
    init_pct = st.session_state.get("init_storage", 50.0)
    st.session_state.sim_bridge.init_cascade(init_pct)
    st.session_state.sim_tick = 0
    st.session_state.sim_history = []
    st.session_state.sim_running = False

# ---------------------------------------------------------------------------
# Helpers for the data view charts
# ---------------------------------------------------------------------------
def create_water_level_gauge(res_data: dict) -> go.Figure:
    wl = res_data.get("water_level")
    blue = res_data.get("blue_level")
    orange = res_data.get("orange_level")
    red = res_data.get("red_level")
    frl = res_data.get("frl")
    if wl is None:
        fig = go.Figure()
        fig.update_layout(title="Water Level Data Unavailable", template="plotly_dark", height=200)
        return fig
    max_val = max(v for v in [wl, frl, red, orange, blue, wl*1.1] if v is not None)
    min_val = min(v for v in [wl, blue, orange, red, frl, wl*0.9] if v is not None)
    steps = []
    if blue is not None: steps.append({'range':[min_val,blue],'color':'rgba(40,167,69,.3)'})
    if blue and orange: steps.append({'range':[blue,orange],'color':'rgba(255,193,7,.3)'})
    if orange and red: steps.append({'range':[orange,red],'color':'rgba(253,126,20,.3)'})
    if red: steps.append({'range':[red,max_val],'color':'rgba(220,53,69,.3)'})
    fig = go.Figure(go.Indicator(mode="gauge+number",value=wl,
        title={'text':"Water Level (m)",'font':{'color':'#b0bec5'}},
        number={'font':{'color':'#e0e6f0'}},
        gauge={'axis':{'range':[min_val,max_val],'tickcolor':'#556677'},
               'bar':{'color':'#3388cc'},'steps':steps,'bgcolor':'#1a2030',
               'threshold':{'line':{'color':'#ff4455','width':3},'thickness':.75,'value':red or max_val}}))
    fig.update_layout(height=200,margin=dict(l=10,r=10,t=40,b=10),paper_bgcolor='rgba(0,0,0,0)',font_color='#b0bec5')
    return fig

def create_forecast_bar_chart(res_data: dict) -> go.Figure:
    forecasts = [res_data.get("forecast_1d"),res_data.get("forecast_3d"),res_data.get("forecast_7d")]
    labels = ['1-Day','3-Day','7-Day']
    if all(f is None for f in forecasts):
        fig = go.Figure()
        fig.update_layout(title="Forecast Data Unavailable",template="plotly_dark",height=200)
        return fig
    plot_f = [f if f is not None else 0 for f in forecasts]
    thresh = res_data.get("historical_95th_inflow")
    fig = go.Figure(data=[go.Bar(x=labels,y=plot_f,marker_color=['#3388cc','#4499dd','#55aaee'])])
    if thresh: fig.add_hline(y=thresh,line_dash="dash",line_color="#ff5566",
        annotation_text=f"95th %ile ({thresh:.1f})",annotation_font_color="#ff5566")
    source_label = res_data.get("source", "UNKNOWN")
    fig.update_layout(title=f"LSTM V3 Inflow Forecast ({source_label})",yaxis_title="MCM/day",height=200,
        margin=dict(l=10,r=10,t=40,b=10),paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(10,14,23,.5)',
        font_color='#b0bec5',showlegend=False)
    return fig

# ===================================================================
# MAIN APPLICATION RENDER
# ===================================================================
def main():
    bridge = st.session_state.sim_bridge

    with st.spinner("Loading LSTM V3 inference engine and assessing reservoirs..."):
        try:
            data = get_reservoir_data()
            df = pd.DataFrame(data)
        except Exception as e:
            st.error(f"Failed to load reservoir data: {e}")
            st.stop()

    if df.empty:
        st.error("No reservoir data found.")
        st.stop()

    # ---------------------------------------------------------------
    # 1. LEFT SIDEBAR (Nav & Reservoir List)
    # ---------------------------------------------------------------
    with st.sidebar:
        st.markdown("### 🛰️ COMMAND CENTER")
        st.caption("AI-Based Multi-Reservoir System")
        st.markdown("---")
        
        # Status summary
        total = len(df)
        normal = len(df[df['overall_status']=='NORMAL'])
        alert_high = len(df[df['overall_status'].isin(['ALERT', 'HIGH RISK'])])
        
        st.markdown(f"**System Health:** {'🟢 Stable' if alert_high == 0 else '🔴 Critical'}")
        st.markdown(f"**Live Inference:** Active (GPU Accelerated)")
        st.markdown(f"**Monitored:** {total} | **Normal:** {normal} | **Alert:** {alert_high}")
        
        st.markdown("---")
        st.markdown("#### Select Reservoir")
        
        selected_real = st.radio(
            "reservoir",
            df['reservoir'].tolist(),
            label_visibility="collapsed",
            format_func=lambda r: f"{'🟢' if df[df['reservoir']==r].iloc[0]['overall_status']=='NORMAL' else '🟡' if df[df['reservoir']==r].iloc[0]['overall_status']=='WATCH' else '🟠' if df[df['reservoir']==r].iloc[0]['overall_status']=='ALERT' else '🔴'} {r}"
        )
        
        # Map real reservoir to virtual cascade for UI linking
        real_to_virtual = {
            "Anayirankal": "Virtual Reservoir A",
            "Ponmudi": "Virtual Reservoir B",
            "Idamalayar": "Virtual Reservoir C",
            "Idukki": "Virtual Reservoir D"
        }
        virtual_target = real_to_virtual.get(selected_real)
        
        st.markdown("---")
        st.markdown("#### Hardware Interface")
        st.caption("FUTURE PHYSICAL INTEGRATION")
        st.markdown("""
        <div class="metric-card" style="font-size:14px; background:rgba(10,14,23,0.9);">
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="color:#b0bec5">ESP32</span><span style="color:#ff5555; font-weight:bold;">● OFFLINE</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="color:#b0bec5">Level Sensor</span><span style="color:#ff5555; font-weight:bold;">● OFFLINE</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="color:#b0bec5">Flow Sensor</span><span style="color:#ff5555; font-weight:bold;">● OFFLINE</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:12px;">
                <span style="color:#b0bec5">Gate Actuator</span><span style="color:#ff5555; font-weight:bold;">● OFFLINE</span>
            </div>
            <div style="border-top:1px solid rgba(255,255,255,0.1); padding-top:8px; display:flex; justify-content:space-between;">
                <span style="color:#e0e6f0">System Mode</span><span style="color:#28a745; font-weight:bold;">● SIMULATION</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ---------------------------------------------------------------
    # MAIN LAYOUT
    # ---------------------------------------------------------------
    # Top Bar
    st.markdown("### 🌐 Reservoir Digital Twin & AI Control")

    # ---------------------------------------------------------------
    # Full-Screen Digital Twin Link
    # ---------------------------------------------------------------
    # The standalone FastAPI + Three.js Digital Twin is the PRIMARY
    # immersive 3D interface. Streamlit is the secondary analytics UI.
    # They run SEPARATE SimBridge instances by design — Streamlit owns
    # its own simulation state in st.session_state, while FastAPI owns
    # its own via state_manager.GlobalSimulationState.
    _fastapi_host = "127.0.0.1"
    _fastapi_port = 8000
    _twin_url = f"http://{_fastapi_host}:{_fastapi_port}"
    st.markdown(
        f'<div style="margin-bottom:12px;">'
        f'<a href="{_twin_url}" target="_blank" '
        f'style="display:inline-block;padding:10px 20px;background:linear-gradient(135deg,#1e3a5f,#0f2847);'
        f'color:#38bdf8;border:1px solid rgba(56,189,248,0.4);border-radius:8px;text-decoration:none;'
        f'font-weight:700;font-size:14px;letter-spacing:0.05em;">'
        f'🖥️ OPEN FULL-SCREEN DIGITAL TWIN</a>'
        f'<span style="margin-left:12px;color:#94a3b8;font-size:12px;">'
        f'PRIMARY 3D DIGITAL TWIN · FastAPI + Three.js</span>'
        f'</div>',
        unsafe_allow_html=True
    )
    
    col_main, col_right = st.columns([3, 1])

    # ---------------------------------------------------------------
    # CENTER AREA (3D Twin)
    # ---------------------------------------------------------------
    with col_main:
        # We render the Three.js scene ONCE. (Triggering reload)
        with open(_THIS_DIR / "twin_component" / "reservoir_twin.html", "r", encoding="utf-8") as f:
            twin_html = f.read()
        components.html(twin_html, height=650, scrolling=False)

    # ---------------------------------------------------------------
    # RIGHT AREA (Controls & AI)
    # ---------------------------------------------------------------
    with col_right:
        st.markdown("#### Simulation Controls")
        
        # Buttons only SET flags in session_state.
        # The actual simulation advance runs AFTER advance_simulation()
        # and all its dependencies (manual_inflows, gate_commands) are
        # fully constructed — see "DEFERRED BUTTON ACTIONS" section below.
        c1, c2, c3 = st.columns(3)
        if c1.button("▶ Play" if not st.session_state.sim_running else "⏸ Pause"):
            st.session_state.sim_running = not st.session_state.sim_running
        if c2.button("⏭ Step"):
            st.session_state.sim_running = False
            st.session_state["_pending_step"] = True
        if c3.button("↻ Reset"):
            reset_simulation()
            
        sim_speed = st.selectbox("Playback Speed", [0.25, 0.5, 1.0, 2.0], index=2)
        
        st.markdown("---")
        control_mode = st.radio("Control Mode", ["Manual Control", "AI/MPC Control"], horizontal=True)
        storm_intensity = st.slider("Storm Mode Intensity", 0.0, 1.0, 0.0, 0.1)
        
        st.markdown("---")
        st.markdown("#### Reservoir Operators")
        
        manual_inflows = {}
        manual_gates = {}
        
        # Show controls for A, B, C
        for res in ["Virtual Reservoir A", "Virtual Reservoir B", "Virtual Reservoir C"]:
            # If the user selected a corresponding real reservoir in the sidebar, keep it expanded
            is_expanded = (res == virtual_target)
            
            with st.expander(f"⚙️ {res}", expanded=is_expanded):
                default_inflow = 10.0 if "A" in res else (20.0 if "B" in res else 100.0)
                current_inflow = default_inflow * (1.0 + (storm_intensity * 3.0))
                manual_inflows[res] = st.slider(f"Inflow", 0.0, default_inflow*5, float(current_inflow), key=f"inf_{res}")
                
                if "Manual" in control_mode:
                    current_gate = bridge.cascade.reservoirs[res].state.gate_position_pct
                    manual_gates[res] = st.slider(f"Gate %", 0.0, 100.0, float(current_gate), key=f"gate_{res}")
                else:
                    st.info("Controlled by AI/MPC")

    # ---------------------------------------------------------------
    # SIMULATION TICK LOGIC
    # ---------------------------------------------------------------
    # We need mock forecasts to feed the AI controller and Risk Engine
    forecasts = {}
    for res in manual_inflows.keys():
        f_base = manual_inflows[res]
        forecasts[res] = {"forecast_1d": f_base, "forecast_3d": f_base, "forecast_7d": f_base}
        
    ai_recommendations = bridge.compute_ai_recommendation(forecasts)

    if "AI" in control_mode:
        gate_commands = {r: ai_recommendations.get(r, 0.0) for r in manual_inflows.keys()}
        gate_commands["Virtual Reservoir D"] = 100.0
    else:
        gate_commands = manual_gates
        gate_commands["Virtual Reservoir D"] = 100.0
        
    def advance_simulation(b):
        """Advance the real simulation by one step via SimBridge."""
        inflows = manual_inflows.copy()
        inflows["Virtual Reservoir D"] = 0.0
        b.step(inflows, gate_commands)
        st.session_state.sim_tick += 1

    # ---------------------------------------------------------------
    # DEFERRED BUTTON ACTIONS
    # ---------------------------------------------------------------
    # The STEP button sets _pending_step = True earlier in the render.
    # Now that advance_simulation() and its captured variables are ready
    # we can safely execute the deferred step.
    if st.session_state.pop("_pending_step", False):
        advance_simulation(bridge)

    if st.session_state.sim_running:
        advance_simulation(bridge)
        
    current_state = bridge.get_state(forecasts)
    current_state["storm_intensity"] = storm_intensity
    
    # Inject real-time slider values for instant visual feedback on gates
    for res, gate_val in gate_commands.items():
        if res in current_state["reservoirs"]:
            current_state["reservoirs"][res]["gate_position_pct"] = gate_val

    # INJECT 3D STATE
    mode_str = "AI" if "AI" in control_mode else "MANUAL"
    adapted_state = adapt_state_for_twin(current_state, mode_str, storm_intensity)
    state_json = json.dumps(adapted_state)
    js_injector = f"""
    <script>
        const frames = window.parent.frames;
        for (let i = 0; i < frames.length; i++) {{
            frames[i].postMessage({{type: "streamlit:render", args: {{state: {state_json}}}}}, "*");
        }}
    </script>
    """
    components.html(js_injector, height=0, width=0)

    # ---------------------------------------------------------------
    # BOTTOM AREA (Telemetry & Forecast)
    # ---------------------------------------------------------------
    st.markdown("---")
    st.markdown(f"### 📊 Telemetry & Forecasts: **{selected_real}**")
    
    res_data = df[df['reservoir'] == selected_real].iloc[0].to_dict()
    
    col_g, col_f, col_ai, col_p = st.columns([1, 1, 1, 1])
    
    with col_g:
        st.plotly_chart(create_water_level_gauge(res_data), use_container_width=True)
        
    with col_f:
        st.plotly_chart(create_forecast_bar_chart(res_data), use_container_width=True)
        
    with col_ai:
        status = res_data.get("overall_status", "UNKNOWN")
        color = "#28a745" if status == "NORMAL" else "#ffc107" if status == "WATCH" else "#fd7e14" if status == "ALERT" else "#dc3545"
        
        st.markdown(f"""
        <div class="metric-card" style="border-left: 4px solid {color}">
            <h4 style="margin:0; padding-bottom:5px; border-bottom:1px solid rgba(255,255,255,0.1)">Live Risk Status</h4>
            <h2 style="color:{color}; margin-top:10px;">{status}</h2>
            <p style="font-size:12px; color:#b0bec5">{res_data.get('reason')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # If virtual equivalent exists, show its simulation state
        if virtual_target:
            virt = current_state["reservoirs"][virtual_target]
            rec = ai_recommendations.get(virtual_target, 0)
            st.markdown(f"""
            <div class="metric-card">
                <h5 style="margin:0;">AI/MPC Recommendation</h5>
                <p style="font-size:12px; margin-bottom:0">Simulated Gate: {virt['gate_position_pct']:.0f}%</p>
                <p style="font-size:12px; font-weight:bold; color:#4a9eff">Optimal Target: {rec:.0f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
    with col_p:
        st.markdown("""
        <div class="metric-card">
            <h4 style="margin:0; padding-bottom:5px; border-bottom:1px solid rgba(255,255,255,0.1)">LSTM V3 Performance</h4>
        """, unsafe_allow_html=True)
        for horizon, metrics in V3_PERFORMANCE.items():
            h_label = horizon.replace("target_", "")
            st.markdown(f"**+{h_label}**: MAE {metrics['MAE']:.2f} | R² {metrics['R2']:.3f}")
        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.sim_running:
        time.sleep(1.0 / sim_speed)
        st.rerun()

if __name__ == "__main__":
    main()
