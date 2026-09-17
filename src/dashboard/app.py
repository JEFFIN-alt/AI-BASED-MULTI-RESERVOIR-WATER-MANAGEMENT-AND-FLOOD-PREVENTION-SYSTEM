"""
Streamlit Dashboard for AI-Based Multi-Reservoir Water Management
Decision Support System — 3D Digital Twin Interface
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.graph_objects as go
import json
import math
import os
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
#: STAGE 13 — host/port remain module constants with their original defaults;
#: the environment overrides exist so the page can be pointed at an
#: authoritative backend on another port (runtime auditing) without editing
#: code. They change WHERE the page reads/commands, never WHAT it owns: this
#: page still owns no simulation.
TWIN_HOST = os.environ.get("AQUAFLOW_TWIN_HOST", "127.0.0.1")
TWIN_PORT = int(os.environ.get("AQUAFLOW_TWIN_PORT", "8000"))
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
# STAGE 13 — COMMAND PROXY (the only way this page can influence anything)
# ---------------------------------------------------------------------------
# This page is a VIEWER + COMMAND PROXY. It executes no physics, owns no
# simulation object and keeps no authoritative state, so a control here cannot
# "apply" anything locally — it can only ask the authoritative backend to apply
# it. The backend then runs the full validated path
#
#     MPC -> SafetyLayer -> DownstreamCapacityGuard -> ReservoirNetwork.step()
#
# and publishes the resulting state, which this page re-reads and displays.
# There is deliberately NO local mirror of the command result.
#
# The command payloads are forwarded VERBATIM: validation is not duplicated,
# weakened or bypassed here. If the backend rejects a command (400/422) the
# backend's own verdict is shown to the operator instead of a local opinion.
#
#: STAGE 13 — every command this page can send, and the ONLY routes it uses.
#: PLAY / PAUSE / STEP / RESET / SET_SPEED / gate (all four) / storm / mode.
COMMAND_PROXY_ENDPOINTS = (
    "/api/simulation/play",
    "/api/simulation/pause",
    "/api/simulation/step",
    "/api/simulation/reset",
    "/api/simulation/speed",
    "/api/storm",
    "/api/controller/mode",
    "/api/gate/reservoir_1",
    "/api/gate/reservoir_2",
    "/api/gate/reservoir_3",
    "/api/gate/reservoir_4",
)


def post_command(path: str, payload: dict | None = None, timeout: float = 5.0):
    """
    Send ONE bounded command to the authoritative backend.

    Parameters
    ----------
    path : str
        A route from :data:`COMMAND_PROXY_ENDPOINTS` (relative to ``/api``).
    payload : dict | None
        The command body (``None`` for the parameterless simulation commands).
        Forwarded verbatim — never rewritten, clamped or "corrected" here.

    Returns
    -------
    ``(True, response_body)`` when the authoritative backend accepted the
    command; ``(False, message)`` when it rejected it or could not be reached.
    The distinction matters: a rejected command is reported as rejected, and a
    missing backend is reported as unreachable. Nothing is assumed to have
    happened.
    """
    if path not in COMMAND_PROXY_ENDPOINTS:
        # Defensive: this page may only ever call the documented command routes.
        return False, f"refusing to POST to an undocumented route: {path}"

    url = f"{TWIN_URL}{path}"
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
        return True, (json.loads(raw) if raw.strip() else {})
    except urllib.error.HTTPError as exc:
        # The BACKEND rejected the command. Surface its verdict; do not paper
        # over it with a local default.
        try:
            detail = json.loads(exc.read().decode("utf-8"))
        except Exception:
            detail = exc.reason
        return False, f"HTTP {exc.code} — backend rejected the command: {detail}"
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return False, f"authoritative backend unreachable: {exc}"


def dispatch_command(label: str, path: str, payload: dict | None = None) -> None:
    """
    Run one proxied command, remember its outcome, then re-enter the script.

    The re-run is what makes the displayed state *authoritative*: it re-reads
    ``GET /api/state`` and renders whatever the backend now reports, so a
    command is never reflected by local bookkeeping. ``st.session_state`` holds
    only this transient flash message — never physical state.
    """
    ok, detail = post_command(path, payload)
    st.session_state["_cmd_flash"] = {"label": label, "ok": ok, "detail": detail}
    st.rerun()

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
# STAGE 13 — READ-ONLY RENDERERS OF THE AUTHORITATIVE PAYLOAD
# ===================================================================
# These functions only FORMAT what the backend already decided. They perform no
# unit conversion beyond the documented display boundary, derive no physical
# quantity, and never fall back to an invented number: an absent value renders
# as "--".

def flow_text(value) -> str:
    """Format an authoritative flow (m³/s) for display, or ``--`` if absent."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "--"
    if not math.isfinite(float(value)):
        return "--"
    return f"{float(value):.1f}"


def residual_text(value) -> str:
    """Format the mass-balance residual (tiny magnitudes need exponents)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "--"
    if not math.isfinite(float(value)):
        return "--"
    return f"{float(value):.2e}"


def render_command_proxy(auth_state: dict) -> None:
    """
    STAGE 13 — the command proxy.

    Every widget here is operator INTENT held in Streamlit's transient widget
    state. A widget value is sent to the authoritative backend on submit and is
    never applied to anything local. The DISPLAYED physical values always come
    from the payload, never from these widgets.

    Commands are dispatched through :func:`dispatch_command` →
    :func:`post_command`, i.e. over HTTP to the ONE authoritative backend. No
    branch of this function touches a reservoir, a gate, a queue, a spill, a
    timestep, a controller decision or a safety decision.
    """
    sim = auth_state.get("simulation", {}) or {}

    # --- transport: PLAY / PAUSE / STEP ------------------------------
    c1, c2, c3 = st.columns(3)
    if c1.button("▶ PLAY", key="cmd_play", use_container_width=True):
        dispatch_command("PLAY", "/api/simulation/play")
    if c2.button("⏸ PAUSE", key="cmd_pause", use_container_width=True):
        dispatch_command("PAUSE", "/api/simulation/pause")
    if c3.button("⏭ STEP", key="cmd_step", use_container_width=True):
        dispatch_command("STEP", "/api/simulation/step")

    # --- RESET / SET_SPEED -------------------------------------------
    c4, c5 = st.columns(2)
    if c4.button("⟲ RESET", key="cmd_reset", use_container_width=True):
        dispatch_command("RESET", "/api/simulation/reset")

    _speed = sim.get("speed")
    _speed_default = float(_speed) if isinstance(_speed, (int, float)) else 1.0
    with c5:
        speed = st.number_input(
            "Playback speed ×", min_value=0.05, max_value=50.0,
            value=_speed_default, step=0.05, key="cmd_speed",
        )
        if st.button("APPLY SPEED", key="cmd_speed_apply", use_container_width=True):
            dispatch_command("SET_SPEED", "/api/simulation/speed", {"speed": float(speed)})

    # --- controller mode ---------------------------------------------
    _mode = auth_state.get("controller_mode")
    _mode_options = ["MANUAL", "AI"]
    mode = st.radio(
        "Controller mode", _mode_options,
        index=_mode_options.index(_mode) if _mode in _mode_options else 0,
        key="cmd_mode", horizontal=True,
    )
    if st.button("APPLY MODE", key="cmd_mode_apply"):
        dispatch_command("SET_MODE", "/api/controller/mode", {"mode": mode})

    # --- storm intensity ---------------------------------------------
    _storm = auth_state.get("storm_intensity")
    storm = st.slider(
        "Storm intensity", min_value=0.0, max_value=1.0,
        value=float(_storm) if isinstance(_storm, (int, float)) else 0.0,
        step=0.05, key="cmd_storm",
    )
    if st.button("APPLY STORM", key="cmd_storm_apply"):
        dispatch_command("SET_STORM", "/api/storm", {"value": float(storm)})

    # --- gate commands, ALL FOUR reservoirs --------------------------
    # The list of commandable reservoirs is read from the authoritative
    # `cascade` inventory, so Reservoir D (Idukki) is an ordinary commandable
    # reservoir here — never a hardcoded subset.
    st.markdown("**Gate commands (%)**")
    cascade = (auth_state.get("cascade", {}) or {}).get("reservoirs", []) or []
    gate_values: dict[str, float] = {}
    for entry in cascade:
        key = entry.get("key")
        if not key:
            continue
        res = (auth_state.get("reservoirs", {}) or {}).get(key, {}) or {}
        label = str(entry.get("name") or res.get("repository_name") or key)
        if entry.get("terminal"):
            label += " (terminal)"
        _gate = res.get("gate")
        _gate_pct = float(_gate) * 100.0 if isinstance(_gate, (int, float)) else 0.0
        gate_values[key] = st.slider(
            label, min_value=0.0, max_value=100.0,
            value=float(round(_gate_pct)), step=1.0, key=f"cmd_gate_{key}",
        )

    if st.button("APPLY GATES", key="cmd_gates_apply", use_container_width=True):
        # One bounded command per reservoir; the backend validates each one.
        outcomes = [
            (key, *post_command(f"/api/gate/{key}", {"value": float(value)}))
            for key, value in gate_values.items()
        ]
        rejected = [(k, msg) for k, ok, msg in outcomes if not ok]
        st.session_state["_cmd_flash"] = {
            "label": f"GATE COMMANDS ({len(outcomes)})",
            "ok": not rejected,
            "detail": (
                "all accepted by the authoritative backend"
                if not rejected
                else "; ".join(f"{k}: {msg}" for k, msg in rejected)
            ),
        }
        st.rerun()


def render_authoritative_reservoirs(auth_state: dict) -> None:
    """
    STAGE 9/13 — all FOUR authoritative reservoirs, as the backend reports them.

    Names, positions, storage, gates, flows, trends, risks and forecasts are
    rendered from the payload. Nothing here is computed, remembered or defaulted
    to a physical value.
    """
    st.markdown("### 🗺️ Authoritative Four-Reservoir State")
    st.caption(
        "Read verbatim from the ONE authoritative simulation payload "
        "(`GET /api/state`). This page neither computes nor stores these values."
    )
    reservoirs = auth_state.get("reservoirs", {}) or {}
    cascade = (auth_state.get("cascade", {}) or {}).get("reservoirs", []) or []
    # The inventory comes from the payload alone — there is deliberately NO
    # hardcoded reservoir list here, so the page cannot disagree with the
    # authoritative cascade. A payload without an inventory shows nothing
    # rather than a guess.
    order = [
        entry.get("key") for entry in cascade
        if entry.get("key") and entry.get("key") in reservoirs
    ]
    if not order:
        st.info(
            "The authoritative payload carries no reservoir inventory yet; "
            "nothing is displayed rather than an assumed reservoir list."
        )
        return

    columns = st.columns(max(1, len(order)))
    for column, key in zip(columns, order):
        res = reservoirs.get(key, {}) or {}
        name = str(res.get("repository_name") or res.get("node_id") or key)
        _storage = res.get("storage")
        with column:
            st.markdown(f"**{name}**")
            st.caption(
                f"`{key}` · node `{res.get('node_id')}` · "
                f"position {res.get('cascade_position')}"
                + (" · TERMINAL" if res.get("is_terminal") else "")
            )
            st.metric(
                f"{name} · storage",
                f"{float(_storage) * 100.0:.1f} %" if isinstance(_storage, (int, float)) else "--",
            )
            _gate = res.get("gate")
            st.metric(
                f"{name} · gate",
                f"{float(_gate) * 100.0:.0f} %" if isinstance(_gate, (int, float)) else "--",
            )
            st.metric(f"{name} · inflow", f"{flow_text(res.get('inflow'))} m³/s")
            st.metric(f"{name} · release", f"{flow_text(res.get('release'))} m³/s")
            st.metric(f"{name} · trend", str(res.get("trend") or "--"))
            st.metric(f"{name} · risk", str(res.get("risk") or "--"))
            st.caption(
                "forecast "
                + " / ".join(
                    f"{h}: {flow_text(res.get(h))}"
                    for h in ("forecast_1d", "forecast_3d", "forecast_7d")
                )
                + f" m³/s · status `{res.get('forecast_status')}`"
            )


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

        # Transient UI-only flash describing the LAST proxied command. It is a
        # message, not state: it never carries a physical value.
        _flash = st.session_state.pop("_cmd_flash", None)
        if _flash:
            if _flash["ok"]:
                st.success(f"{_flash['label']} → accepted by the authoritative backend")
            else:
                st.error(f"{_flash['label']} → {_flash['detail']}")

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

            # STAGE 12/13 — WHICH authoritative state this is (no second clock).
            _ident = auth_state.get("state_identity", {}) or {}
            _sim = auth_state.get("simulation", {}) or {}
            st.metric("State #", str(_ident.get("state_id") or "--"))
            _running = _sim.get("running")
            st.metric(
                "Simulation",
                "RUNNING" if _running is True else "PAUSED" if _running is False else "--",
            )

            _mode = auth_state.get("controller_mode", "UNKNOWN")
            # STAGE 12/13 — intensity AND level come from the backend's `storm`
            # block; the page does not decide what counts as a storm.
            _storm_block = auth_state.get("storm", {}) or {}
            _storm = _storm_block.get("intensity")
            st.metric("Controller mode", str(_mode))
            st.metric(
                "Storm intensity",
                f"{float(_storm):.2f}" if isinstance(_storm, (int, float)) else "--",
            )
            st.caption(f"storm level (backend verdict): `{_storm_block.get('level')}`")
            st.caption(
                f"physics clock `network_timestep` = {_ident.get('network_timestep')} · "
                f"operator steps = {_ident.get('sim_step_index')} · "
                f"speed ×{_sim.get('speed')}"
            )

            # STAGE 5/6 — DEMONSTRATION provenance banner.
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

            # STAGE 7/8/10 — authoritative controller + safety + capacity
            # provenance (read-only).
            _ctl = auth_state.get("control", {}) or {}
            st.markdown("---")
            st.caption("AUTHORITATIVE CONTROL PATH")
            st.metric("Controller", f"{_ctl.get('controller_type', 'UNKNOWN')} · "
                                    f"{_ctl.get('controller_status', 'UNKNOWN')}")
            st.metric("Forecast eligible", "YES" if _ctl.get("forecast_control_eligible") else "NO")
            st.metric("Safety layer", str(_ctl.get("safety_layer_status", "UNKNOWN")))
            if _ctl.get("safety_modified"):
                st.caption("Safety layer MODIFIED the MPC proposal before it was applied.")
            st.metric("Downstream guard", str(_ctl.get("downstream_status", "UNKNOWN")))
            st.caption(
                f"final action source: `{_ctl.get('final_safe_control_action_source', 'UNKNOWN')}`"
            )

            # STAGE 11 — live mass-balance integrity (read-only audit result).
            _mb = auth_state.get("mass_balance", {}) or {}
            st.markdown("---")
            st.caption("MASS-BALANCE INTEGRITY (Stage 11)")
            st.metric("Mass balance", str(_mb.get("status", "NOT_CHECKED")))
            _residual = _mb.get("residual")
            st.caption(
                f"residual {residual_text(_residual)} {_mb.get('unit', 'MCM')} "
                f"/ tolerance {_mb.get('tolerance')} · "
                f"{_mb.get('reservoirs_checked', 0)}/{_mb.get('reservoirs_expected', 0)} "
                f"reservoirs checked"
            )

            # STAGE 12 — downstream severity + limit are the backend's verdicts.
            _dwn = auth_state.get("downstream", {}) or {}
            st.markdown("---")
            st.caption("DOWNSTREAM CAPACITY (Stage 10)")
            st.metric(
                "Flow vs limit",
                f"{flow_text(_dwn.get('flow_m3_s'))} / {flow_text(_dwn.get('capacity_m3_s'))} m³/s",
            )
            st.metric("Severity", str(_dwn.get("status", "UNKNOWN")))
            _util = _dwn.get("utilisation")
            st.caption(
                "utilisation "
                + (f"{float(_util) * 100.0:.1f} %" if isinstance(_util, (int, float)) else "--")
            )

            # STAGE 12 — forecast aggregate + provenance (read-only).
            _fsum = auth_state.get("forecast_summary", {}) or {}
            _horizons = _fsum.get("horizons", {}) or {}
            st.markdown("---")
            st.caption("FORECAST (frozen LSTM V3)")
            st.metric("Forecast coverage", str(_fsum.get("status", "UNKNOWN")))
            st.caption(
                " · ".join(
                    f"{h}: {flow_text(_horizons.get(h))} m³/s"
                    for h in ("1d", "3d", "7d")
                )
            )
            st.caption(
                f"{_fsum.get('reservoirs_with_forecast', 0)}/"
                f"{_fsum.get('reservoirs_total', 0)} reservoirs with a forecast · "
                f"validated metrics apply: "
                f"{'YES' if _fsum.get('validated_metrics_apply') else 'NO'}"
            )

            # Hardware readiness (Stage 10/13 — nothing is faked).
            _hw = auth_state.get("hardware_status", {}) or {}
            st.markdown("---")
            st.caption("HARDWARE")
            st.metric(
                "Hardware",
                "CONNECTED" if _mb.get("hardware_connected") else "NOT CONNECTED",
            )
            st.caption(
                " · ".join(f"{k}: {v}" for k, v in sorted(_hw.items())) or "no hardware block"
            )

            # STAGE 14 — GNN ADVISORY (spatial dependency representation).
            # Read-only. The page renders the backend's own advisory block; it
            # runs no GNN, derives no embedding and computes no similarity. The
            # advisory is not, and cannot become, a command source.
            _gnn = auth_state.get("gnn_advisory", {}) or {}
            st.markdown("---")
            st.caption("GNN ADVISORY — SPATIAL DEPENDENCY (advisory only)")
            st.metric("GNN advisory", str(_gnn.get("status", "UNAVAILABLE")))
            st.caption(
                f"{_gnn.get('model_name', '--')} · {_gnn.get('model_version', '--')} · "
                f"affects control: "
                f"{'YES' if _gnn.get('affects_control') else 'NO'}"
            )
            _gnn_graph = _gnn.get("graph_provenance", {}) or {}
            st.caption(
                f"graph {_gnn.get('graph_nodes')} nodes / "
                f"{_gnn.get('graph_undirected_edges')} undirected edges "
                f"({_gnn.get('graph_directed_edges')} directed) · "
                f"components {_gnn_graph.get('connected_components')} · "
                f"isolated {_gnn_graph.get('isolated_reservoirs')}"
            )
            st.caption(
                f"construction: `{_gnn_graph.get('construction_method', '--')}` · "
                f"rule: {_gnn_graph.get('edge_rule') or '--'}"
            )
            _gnn_sim = _gnn.get("embedding_similarity") or {}
            _gnn_top = (_gnn_sim.get("top_relationships") or [None])[0]
            if _gnn_top:
                st.caption(
                    f"strongest **embedding similarity**: {_gnn_top.get('source')} ↔ "
                    f"{_gnn_top.get('target')} = {_gnn_top.get('embedding_similarity')}"
                )
            st.caption(str(_gnn_sim.get("label") or "--"))
            if _gnn.get("reason"):
                st.caption(f"reason: `{_gnn.get('reason')}`")

        # ------------------------------------------------------------------
        # STAGE 13 — COMMAND PROXY
        # ------------------------------------------------------------------
        # These controls are proxies, not an engine: each one POSTs a bounded
        # command to the authoritative FastAPI simulation and then re-reads the
        # state it produced. No physics, no clock and no state live here.
        st.markdown("---")
        with st.expander("🎛️ COMMAND PROXY — control the authoritative backend", expanded=False):
            st.caption(
                "Every control below POSTs a bounded command to the ONE "
                "authoritative backend (`MPC → SafetyLayer → "
                "DownstreamCapacityGuard → ReservoirNetwork`) and then re-reads "
                "the state it produced. This page runs no physics, holds no "
                "simulation object and cannot bypass those layers — it has no "
                "local simulation to apply a command to."
            )
            if auth_error:
                st.info("Start the authoritative backend (`python app.py`) to send commands.")
            else:
                render_command_proxy(auth_state)

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
    # AUTHORITATIVE FOUR-RESERVOIR STATE (Stage 9/13)
    # ---------------------------------------------------------------
    # Rendered only when the authoritative backend answered — never from a
    # local default, so the page can never present a plausible-looking state
    # that the backend did not produce.
    if not auth_error:
        st.markdown("---")
        render_authoritative_reservoirs(auth_state)

    # ---------------------------------------------------------------
    # BOTTOM AREA (Telemetry & Forecast)
    # ---------------------------------------------------------------
    st.markdown("---")
    st.markdown(f"### 📊 Telemetry & Forecasts: **{selected_real}**")
    st.caption(
        "NOTE — this section is a SEPARATE read-only analytics subsystem over the "
        "16 real dams (raw telemetry JSON + frozen V3 artifacts + the risk "
        "engine). It is not the authoritative cascade simulation: it owns no "
        "simulation, no clock and no gates, it cannot write anywhere, and it "
        "cannot influence the four-reservoir state above."
    )
    
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
