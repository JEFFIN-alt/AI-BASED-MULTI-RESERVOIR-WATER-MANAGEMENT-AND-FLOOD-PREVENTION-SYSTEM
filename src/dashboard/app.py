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

import urllib.error
import urllib.request

from data_bridge import assess_all_reservoirs, V3_PERFORMANCE

# Setup paths
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent

# ---------------------------------------------------------------------------
# STAGE 4 — SINGLE AUTHORITATIVE SIMULATION
# ---------------------------------------------------------------------------
# Streamlit is a READ-ONLY client of the Digital Twin.
#
# It deliberately does NOT import `sim_bridge.SimBridge` or
# `simulator.engine.SimulationEngine`. It owns no simulation, holds no
# simulation parameters, and can neither produce nor overwrite Digital Twin
# state. The ONE authoritative simulation lives in the FastAPI backend
# (`src/dashboard/api/state_manager.py::sim_state`) — the same instance the
# WebSocket publishes from. Streamlit only *reads* it over HTTP.
#
# `data_bridge` is retained: it is read-only analytics over frozen artifacts
# (forecast + risk assessment) and produces no Digital Twin state.
# ---------------------------------------------------------------------------

#: The authoritative Digital Twin backend (FastAPI + Three.js).
TWIN_HOST = "127.0.0.1"
TWIN_PORT = 8000
TWIN_URL = f"http://{TWIN_HOST}:{TWIN_PORT}"
STATE_URL = f"{TWIN_URL}/api/state"

#: Live-name -> twin reservoir key, for read-only display only.
_VIRTUAL_TO_TWIN_KEY = {
    "Virtual Reservoir A": "reservoir_1",
    "Virtual Reservoir B": "reservoir_2",
    "Virtual Reservoir C": "reservoir_3",
}


def fetch_authoritative_state(timeout: float = 2.0):
    """
    Read the ONE authoritative Digital Twin state over HTTP.

    Returns
    -------
    ``(state, None)`` on success; ``(None, error_message)`` when the
    authoritative backend is unreachable. On failure NOTHING is fabricated —
    the caller renders an explicit "backend offline" notice instead.
    """
    try:
        with urllib.request.urlopen(STATE_URL, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8")), None
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return None, str(exc)

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
# STAGE 4: Streamlit holds NO simulation state. There is no local `sim_bridge`,
# `sim_running`, `sim_tick` or reset — the authoritative simulation is owned
# exclusively by the FastAPI backend.

@st.cache_data(ttl=60)
def get_reservoir_data():
    return assess_all_reservoirs()

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
    # The FastAPI + Three.js Digital Twin is the ONE authoritative
    # simulation and the primary immersive interface. This Streamlit page
    # is a READ-ONLY analytics client: it fetches the authoritative state
    # over HTTP and mirrors it. It never owns or advances a simulation.
    _twin_url = TWIN_URL
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
    # RIGHT AREA (read-only authoritative status)
    # ---------------------------------------------------------------
    # STAGE 4: the simulation controls that used to live here advanced a
    # SECOND, competing simulation owned by Streamlit. They are gone.
    # Control now exists in exactly one place: the authoritative Digital
    # Twin (FastAPI). This panel only READS its state.
    with col_right:
        st.markdown("#### Authoritative Simulation")
        auth_state, auth_error = fetch_authoritative_state()

        if auth_error:
            st.error(
                "Authoritative Digital Twin backend is not reachable.\n\n"
                f"`{STATE_URL}`\n\n"
                "This page is a read-only viewer and holds no simulation of "
                "its own, so no state can be shown until the backend is running."
            )
            st.code("python app.py", language="bash")
        else:
            st.success("Connected — read-only")
            st.caption(
                "State is produced by the ONE authoritative simulation "
                "(`GlobalSimulationState`) and published over `/ws/state`. "
                "All control is delegated to the Digital Twin."
            )
            _mode = auth_state.get("controller_mode", "UNKNOWN")
            _storm = auth_state.get("storm_intensity", 0.0)
            _ds = auth_state.get("downstream_flow", 0.0)
            st.metric("Controller mode", str(_mode))
            st.metric("Storm intensity", f"{float(_storm):.2f}")
            st.metric("Downstream flow", f"{float(_ds):.1f} m³/s")

            # Stage 5/6 — DEMONSTRATION provenance banner.
            # Shown unless the authoritative backend explicitly reports that live
            # forecasts are validated, so a missing provenance block errs on the
            # side of caution rather than implying validity.
            _fp = auth_state.get("forecast_provenance", {}) or {}
            if _fp.get("live_forecasts_are_validated") is not True:
                st.warning(
                    "**DEMONSTRATION — MODEL INPUTS SIMULATED**\n\n"
                    "Live forecasts use simulation-derived inputs and explicitly "
                    "labelled synthetic placeholders for quantities the live "
                    "simulation cannot produce (`water_level` in metres, `rainfall` "
                    "in mm). The validated V3 test metrics do **not** apply."
                )

            # Stage 7/8 — authoritative controller + safety provenance (read-only).
            _ctl = auth_state.get("control", {}) or {}
            st.markdown("---")
            st.caption("AUTHORITATIVE CONTROL PATH")
            st.metric("Controller", f"{_ctl.get('controller_type', 'UNKNOWN')} · "
                                    f"{_ctl.get('controller_status', 'UNKNOWN')}")
            st.metric("Forecast eligible", "YES" if _ctl.get("forecast_control_eligible") else "NO")
            st.metric("Safety layer", str(_ctl.get("safety_layer_status", "UNKNOWN")))
            if _ctl.get("safety_modified"):
                st.caption("Safety layer MODIFIED the MPC proposal before it was applied.")

        st.markdown("---")
        auto_refresh = st.checkbox(
            "Live follow", value=False,
            help="Re-read the authoritative state once per second (read-only).",
        )

    # ---------------------------------------------------------------
    # MIRROR AUTHORITATIVE STATE INTO THE EMBEDDED 3D VIEWER
    # ---------------------------------------------------------------
    # The payload is the authoritative twin schema, produced by the one
    # simulation instance. It is forwarded VERBATIM — Streamlit neither
    # fabricates nor modifies any physical value, and it cannot write back.
    if auth_error:
        st.info(
            "3D viewer is showing its idle state; start the authoritative "
            "backend to mirror live data."
        )
    else:
        state_json = json.dumps(auth_state)
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
        
        # Read-only view of the AUTHORITATIVE gate for the mapped reservoir.
        # Streamlit holds no simulation, so this is the Digital Twin's value.
        twin_key = _VIRTUAL_TO_TWIN_KEY.get(virtual_target) if virtual_target else None
        if twin_key and not auth_error:
            twin_res = auth_state.get("reservoirs", {}).get(twin_key, {})
            gate_ratio = twin_res.get("gate")
            if isinstance(gate_ratio, (int, float)):
                gate_pct = gate_ratio * 100.0
                st.markdown(f"""
            <div class="metric-card">
                <h5 style="margin:0;">Authoritative Gate</h5>
                <p style="font-size:12px; margin-bottom:0">{virtual_target} · from the Digital Twin</p>
                <p style="font-size:18px; font-weight:bold; color:#4a9eff">{gate_pct:.0f}%</p>
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

    # STAGE 4: no local simulation loop exists here any more. "Live follow"
    # merely re-reads the authoritative state (read-only) — it never advances
    # anything. The authoritative backend runs its own loop.
    if auto_refresh:
        time.sleep(1.0)
        st.rerun()

if __name__ == "__main__":
    main()
