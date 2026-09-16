"""
Stage 12 — Authoritative Digital Twin State tests.

Proves that the Three.js Digital Twin is a pure display/interaction client of ONE
authoritative backend simulation state:

    authoritative backend simulation
        -> MassBalanceMonitor
        -> StateAdapter
        -> FastAPI / WebSocket
        -> Three.js Digital Twin   (DISPLAY ONLY)

Covered: single simulation owner, no browser state injection, WebSocket/REST
semantic agreement, no frontend physics, command flow (PLAY/PAUSE/STEP/RESET/
SET_SPEED/gate/storm/mode), all four reservoirs incl. Idukki/D, and the survival
of mass-balance / forecast-provenance / controller statuses across the whole path.
"""

import ast
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.dashboard.api import state_manager  # noqa: E402
from src.dashboard.api.app import app  # noqa: E402
from src.dashboard.api import routes  # noqa: E402
from src.network_env.reservoir_network import ReservoirNetwork  # noqa: E402

client = TestClient(app)

PROJECT_ROOT = str(_PROJECT_ROOT)
WEB_DIR = _PROJECT_ROOT / "src" / "dashboard" / "web"
TWIN_INDEX_PATH = WEB_DIR / "index.html"
API_JS_PATH = WEB_DIR / "api.js"
MIRROR_PATH = _PROJECT_ROOT / "src" / "dashboard" / "twin_component" / "reservoir_twin.html"
ADAPTER_PATH = _PROJECT_ROOT / "src" / "dashboard" / "twin_component" / "state_adapter.py"
STATE_MANAGER_PATH = _PROJECT_ROOT / "src" / "dashboard" / "api" / "state_manager.py"
ROUTES_PATH = _PROJECT_ROOT / "src" / "dashboard" / "api" / "routes.py"
APP_PATH = _PROJECT_ROOT / "src" / "dashboard" / "api" / "app.py"
NETWORK_PATH = _PROJECT_ROOT / "src" / "network_env" / "reservoir_network.py"
SAFETY_PATH = _PROJECT_ROOT / "src" / "controller" / "safety.py"
MPC_PATH = _PROJECT_ROOT / "src" / "controller" / "mpc_controller.py"
GUARD_PATH = _PROJECT_ROOT / "src" / "controller" / "downstream_capacity_guard.py"
ORCH_PATH = _PROJECT_ROOT / "src" / "controller" / "live_mpc_orchestrator.py"
MASS_BALANCE_PATH = _PROJECT_ROOT / "src" / "network_env" / "mass_balance.py"
MANIFEST_PATH = _PROJECT_ROOT / "results" / "phase15_v3_validation" / "v3_integrity_check.json"

NODES = [
    "Virtual Reservoir A",
    "Virtual Reservoir B",
    "Virtual Reservoir C",
    "Virtual Reservoir D",
]
A, B, C, D = NODES
VALIDATED_NAMES = {A: "Anayirankal", B: "Ponmudi", C: "Idamalayar", D: "Idukki"}

FROZEN_ARTIFACTS = [
    _PROJECT_ROOT / "models" / "lstm_pytorch_v3_logtarget" / "best_model.pt",
    _PROJECT_ROOT / "models" / "lstm_pytorch_v3_logtarget" / "log_target_scaler.pkl",
    _PROJECT_ROOT / "results" / "lstm_pytorch_v3_logtarget" / "test_predictions_original_units.csv",
]

#: Payload keys that must survive the whole backend -> WebSocket -> twin path.
REQUIRED_TOP_LEVEL = (
    "reservoirs", "cascade", "control", "mass_balance", "forecast_provenance",
    "state_identity", "simulation", "downstream", "storm", "forecast_summary",
    "downstream_flow", "storm_intensity", "controller_mode", "simulation_time",
    "hardware_status", "metadata",
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _twin_reservoir(payload, key):
    return payload["reservoirs"][key]


def _identity(payload):
    return payload["state_identity"]


def _module_script(html: str) -> str:
    marker = '<script type="module">'
    idx = html.index(marker)
    end = html.index("</script>", idx)
    return html[idx + len(marker):end]


def _node_check(js: str, name: str):
    node = shutil.which("node")
    if node is None:  # pragma: no cover - node is present in this environment
        pytest.skip("node is not available to parse-check the twin page")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / name
        path.write_text(js, encoding="utf-8")
        return subprocess.run([node, "--check", str(path)], capture_output=True, text=True)


def _strip_js_comments(js: str) -> str:
    """Remove // and /* */ comments (string-aware) so assertions test CODE, not prose."""
    out = []
    i, n = 0, len(js)
    mode = None
    while i < n:
        c = js[i]
        nxt = js[i + 1] if i + 1 < n else ""
        if mode is None:
            if c == "/" and nxt == "/":
                mode = "//"; i += 2; continue
            if c == "/" and nxt == "*":
                mode = "/*"; i += 2; continue
            if c in ('"', "'", "`"):
                mode = c; out.append(c); i += 1; continue
            out.append(c); i += 1; continue
        if mode == "//":
            if c == "\n":
                mode = None; out.append(c)
            i += 1; continue
        if mode == "/*":
            if c == "*" and nxt == "/":
                mode = None; i += 2; continue
            i += 1; continue
        if c == "\\":
            out.append(c); out.append(nxt); i += 2; continue
        if c == mode:
            mode = None
        out.append(c); i += 1
    return "".join(out)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


@pytest.fixture
def live_sim():
    """Snapshot/restore the authoritative singleton around a test."""
    sim = state_manager.sim_state
    saved = {
        "mode": sim.mode,
        "running": sim.running,
        "speed": sim.sim_speed,
        "manual_gates": dict(sim.manual_gates),
        "manual_inflows": dict(sim.manual_inflows),
        "storm": sim.storm_intensity,
    }
    try:
        client.post("/api/simulation/pause")
        client.post("/api/simulation/speed", json={"speed": 1.0})
        client.post("/api/controller/mode", json={"mode": "MANUAL"})
        yield sim
    finally:
        client.post("/api/simulation/pause")
        sim.mode = saved["mode"]
        sim.running = saved["running"]
        sim.sim_speed = saved["speed"]
        sim.manual_gates = saved["manual_gates"]
        sim.manual_inflows = saved["manual_inflows"]
        sim.storm_intensity = saved["storm"]
        sim.bridge.init_cascade(50.0)


# ===========================================================================
# 1. ONE authoritative live simulation owner
# ===========================================================================

def test_one_authoritative_live_simulation_owner():
    assert state_manager.authoritative_instance_count() == 1
    assert state_manager.get_authoritative_state_manager() is state_manager.sim_state
    # both transports are wired to the SAME object
    assert routes.sim_state is state_manager.sim_state
    assert app.__dict__.get("sim_state", state_manager.sim_state) is state_manager.sim_state


def test_no_second_simulation_instance_is_created_for_the_browser():
    """
    Nothing in the frontend-serving path constructs a simulation. Only
    ``state_manager`` holds the module-level instance.
    """
    for path in (ROUTES_PATH, APP_PATH, ADAPTER_PATH):
        source = path.read_text(encoding="utf-8")
        assert "GlobalSimulationState(" not in source, path.name
        assert "ReservoirNetwork(" not in source, path.name
        assert "LiveCascadeAdapter(" not in source, path.name

    # the ONE instance exists in exactly one module
    hits = []
    for path in (_PROJECT_ROOT / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"^sim_state = GlobalSimulationState\(\)", text, re.M):
            hits.append(path)
    assert hits == [STATE_MANAGER_PATH], f"extra live simulation owners: {hits}"


def test_the_single_owner_is_what_the_websocket_and_rest_serve(live_sim):
    client.post("/api/simulation/step")
    rest = client.get("/api/state").json()
    with client.websocket_connect("/ws/state") as ws:
        ws_state = ws.receive_json()
    assert _identity(rest)["state_id"] == _identity(ws_state)["state_id"]
    assert _identity(rest)["network_timestep"] == live_sim.bridge.cascade.network.timestep


# ===========================================================================
# 2. The browser cannot inject authoritative state
# ===========================================================================

def test_browser_cannot_post_or_put_state():
    for verb, method in (("post", client.post), ("put", client.put), ("patch", client.patch)):
        response = method("/api/state", json={"reservoirs": {"reservoir_1": {"storage": 1.0}}})
        assert response.status_code == 405, f"{verb} /api/state must not exist"


def test_forged_payloads_on_command_endpoints_are_ignored(live_sim):
    client.post("/api/simulation/reset")
    for i in (1, 2, 3, 4):
        client.post(f"/api/gate/reservoir_{i}", json={"value": 0.0})
    client.post("/api/controller/mode", json={"mode": "MANUAL"})
    before = client.get("/api/state").json()

    forged = {
        "reservoirs": {"reservoir_1": {"storage": 0.01, "gate": 1.0}},
        "state_identity": {"state_id": "forged", "network_timestep": 9999},
        "mass_balance": {"status": "VIOLATION", "checked": True},
        "downstream": {"status": "CRITICAL", "capacity_m3_s": 1.0},
        "controller_mode": "AI",
    }
    client.post("/api/simulation/step", json=forged)
    # the step itself is legitimate; the FORGED FIELDS must have been ignored
    after = client.get("/api/state").json()
    assert after["state_identity"]["state_id"] != "forged"
    assert after["state_identity"]["network_timestep"] < 9999
    assert after["mass_balance"]["status"] == "PASS"
    assert after["downstream"]["status"] == "NORMAL"
    assert after["downstream"]["capacity_m3_s"] == before["downstream"]["capacity_m3_s"]
    assert after["controller_mode"] == "MANUAL"


def test_gate_commands_carry_only_a_bounded_value(live_sim):
    response = client.post("/api/gate/reservoir_1",
                           json={"value": 55.0, "reservoirs": {"reservoir_2": {"storage": 0.1}}})
    assert response.status_code in (200, 204)
    # only the bounded gate value landed; no reservoir state was written
    assert live_sim.manual_gates["Virtual Reservoir A"] == pytest.approx(55.0)
    state = client.get("/api/state").json()
    for key in ("reservoir_1", "reservoir_2", "reservoir_3", "reservoir_4"):
        assert "storage_mcm" not in state["reservoirs"][key]


def test_authoritative_twin_has_no_postmessage_injection_path():
    html = TWIN_INDEX_PATH.read_text(encoding="utf-8")
    assert "streamlit:render" not in html
    assert 'addEventListener("message"' not in html
    assert "addEventListener('message'" not in html
    # ...and the legs it does use are the WebSocket feed and the REST commands
    assert "/ws/state" in API_JS_PATH.read_text(encoding="utf-8")
    assert "new WebSocket" in API_JS_PATH.read_text(encoding="utf-8")


def test_no_console_state_push_handle_remains():
    for path in (TWIN_INDEX_PATH, MIRROR_PATH):
        html = path.read_text(encoding="utf-8")
        assert "window.__twin = " not in html, f"{path.name} still exposes a state-push handle"
        assert "__twin.updateState" not in html
        assert not re.search(r"\.updateState\(\{", html)
    # a read-only readiness flag replaced it
    assert "window.__twinReady = true" in TWIN_INDEX_PATH.read_text(encoding="utf-8")


def test_streamlit_mirror_only_accepts_authoritative_payloads():
    """
    STAGE 12 — the Streamlit-embedded viewer keeps Stage 4's read-only one-way
    mirror, but now REFUSES any object that is not an authoritative payload.
    """
    html = MIRROR_PATH.read_text(encoding="utf-8")
    assert "streamlit:render" in html                       # Stage 4 architecture kept
    assert "hasAuthoritativeMarker" in html, "the mirror must validate what it renders"
    assert "state.state_identity" in html
    assert "if (!hasAuthoritativeMarker) return;" in html
    # no write-back path of any kind
    for forbidden in ("fetch(", "XMLHttpRequest", "navigator.sendBeacon"):
        assert forbidden not in html, f"the mirror must not call {forbidden}"


# ===========================================================================
# 3 & 4. WebSocket and REST both expose the authoritative state
# ===========================================================================

def test_get_api_state_exposes_the_full_authoritative_state(live_sim):
    client.post("/api/simulation/step")
    payload = client.get("/api/state").json()
    for key in REQUIRED_TOP_LEVEL:
        assert key in payload, f"{key} missing from GET /api/state"


def test_websocket_exposes_the_full_authoritative_state(live_sim):
    client.post("/api/simulation/step")
    with client.websocket_connect("/ws/state") as ws:
        payload = ws.receive_json()
    for key in REQUIRED_TOP_LEVEL:
        assert key in payload, f"{key} missing from the WebSocket payload"


def test_rest_and_websocket_semantics_agree(live_sim):
    client.post("/api/simulation/step")
    rest = client.get("/api/state").json()
    with client.websocket_connect("/ws/state") as ws:
        wss = ws.receive_json()

    assert set(rest.keys()) == set(wss.keys())
    assert rest["reservoirs"] == wss["reservoirs"]
    assert rest["cascade"] == wss["cascade"]
    assert rest["mass_balance"] == wss["mass_balance"]
    assert rest["control"] == wss["control"]
    assert rest["state_identity"] == wss["state_identity"]
    assert rest["downstream"] == wss["downstream"]
    assert rest["forecast_summary"] == wss["forecast_summary"]
    assert rest["forecast_provenance"] == wss["forecast_provenance"]
    assert rest["hardware_status"] == wss["hardware_status"]
    # `simulation_time` is the wall-clock stamp of the payload, not state identity
    assert "simulation_time" in rest and "simulation_time" in wss


# ===========================================================================
# 5 & 6. The frontend is display-only
# ===========================================================================

def test_frontend_contains_no_fabricated_initial_state():
    html = TWIN_INDEX_PATH.read_text(encoding="utf-8")
    assert "const INITIAL_STATE = {" not in html
    assert "downstream_flow: 65" not in html
    assert "storm_intensity: 0.7" not in html
    # it starts in an explicit no-data state instead
    assert "NO_AUTHORITATIVE_STATE" in html
    assert "hasState: false" in html


def test_frontend_does_not_calculate_physical_state():
    html = TWIN_INDEX_PATH.read_text(encoding="utf-8")
    code = _strip_js_comments(_module_script(html))

    # no hardcoded downstream limit and no JS flow thresholds
    assert "150 m³/s" not in html
    assert "ds < 120" not in code
    assert "clamp01(ds / 150)" not in code
    # no JS trend derived from inflow vs release
    assert "d.inflow - d.release" not in code
    assert "d4.inflow - d4.release" not in code
    # no fabricated forecast shape
    assert "0.14 * Math.sin" not in code
    # no reservoir equation / routing / spill / mass-balance arithmetic in CODE
    # (the Stage 11/12 comments mentioning these words are stripped first)
    for forbidden in ("attenuation", "controlled_release", "new_storage",
                      "inflow_routed", "natural_inflow", "spill_mcm",
                      "storage_before", "storage_change"):
        assert forbidden not in code, f"JavaScript computes physics: {forbidden}"
    # the backend's audit values may only be DISPLAYED, never re-computed
    for forbidden in ("mb.residual +", "mb.residual -", "mb.residual *",
                      "mb.tolerance +", "mb.tolerance *", "Math.abs(mb"):
        assert forbidden not in code, f"JavaScript computes mass balance: {forbidden}"
    # the only arithmetic on physical readouts is unit formatting / clamping
    for forbidden in ("/ 86400", "* 86400", "* 1e6", "/ 1e6", "* 1000000"):
        assert forbidden not in code, f"JavaScript converts physics units: {forbidden}"


def test_frontend_displays_the_backend_verdicts_instead():
    html = TWIN_INDEX_PATH.read_text(encoding="utf-8")
    code = _module_script(html)
    # it consumes the backend's classifications...
    for key in ("state.downstream", "state.forecast_summary", "state.state_identity",
                "state.simulation", "state.mass_balance", "state.control"):
        assert key in code, f"{key} is not consumed by the twin"
    assert "src.trend" in code            # backend trend
    assert "d4.trend" in code             # backend trend (Reservoir D)
    assert "downstreamStatus" in code     # backend severity
    assert "downstreamUtil" in code       # backend-computed bar ratio
    assert "downstreamCapacity" in code   # authoritative safe limit
    assert "stormLevel" in code           # backend storm classification
    # ...and prints the strings verbatim
    assert "el.dsstat.textContent = st;" in code
    assert "el.wlevel.textContent = lvl || '--';" in code


def test_twin_js_parses():
    proc = _node_check(_module_script(TWIN_INDEX_PATH.read_text(encoding="utf-8")),
                       "twin_index.mjs")
    assert proc.returncode == 0, proc.stderr
    proc = _node_check(_module_script(MIRROR_PATH.read_text(encoding="utf-8")),
                       "twin_mirror.mjs")
    assert proc.returncode == 0, proc.stderr


def test_mirror_viewer_has_no_fabricated_state_either():
    html = MIRROR_PATH.read_text(encoding="utf-8")
    assert "const INITIAL_STATE = {" not in html
    assert "150 m³/s" not in html
    assert "downstream_flow: 65" not in html
    assert "0.14 * Math.sin" not in html
    assert "ds < 120" not in html


# ===========================================================================
# 7 & 8. Commands are backend-authoritative
# ===========================================================================

def test_frontend_command_surface_is_commands_only():
    js = API_JS_PATH.read_text(encoding="utf-8")
    for endpoint in ("/gate/", "/simulation/play", "/simulation/pause",
                     "/simulation/step", "/simulation/reset",
                     "/simulation/speed", "/storm", "/controller/mode"):
        assert endpoint in js, f"missing command endpoint {endpoint}"
    # no state push, no local physics
    assert "reservoirs" not in js
    assert "reservoir_1" not in js.split("setGate")[0]
    for forbidden in ("mass_balance", "controlled_release", "attenuation", "storage_mcm"):
        assert forbidden not in js


def test_frontend_controls_send_commands_and_do_not_mutate_state():
    html = TWIN_INDEX_PATH.read_text(encoding="utf-8")
    code = _module_script(html)
    binding = code[code.index("bindClick('#btn-play'"):]

    for call in ("api.play()", "api.pause()", "api.step()", "api.reset()",
                 "api.setSpeed(", "api.setStorm(", "api.setGate(", "api.setMode("):
        assert call in binding, f"{call} is not wired to the backend"

    # the control bindings never write authoritative fields locally
    for forbidden in ("this.R[", "this.data.downstream =", "this.data.massBalance =",
                      "manual_gates", "storages"):
        assert forbidden not in binding, f"the controls mutate local state: {forbidden}"


def test_play_pause_step_reset_speed_are_backend_authoritative(live_sim):
    sim = live_sim
    client.post("/api/simulation/reset")
    baseline = client.get("/api/state").json()
    t0 = baseline["state_identity"]["network_timestep"]
    # RESET restarts the PHYSICS clock; the cumulative step counter is not reset
    assert t0 == 0
    assert baseline["state_identity"]["sim_step_index"] >= 0

    # STEP advances the backend by exactly one authoritative timestep
    client.post("/api/simulation/step")
    after_step = client.get("/api/state").json()
    assert after_step["state_identity"]["network_timestep"] == t0 + 1
    assert sim.bridge.cascade.network.timestep == t0 + 1

    # PLAY / PAUSE are backend flags, and the payload reports them
    assert client.post("/api/simulation/play").status_code == 200
    assert sim.running is True
    assert client.get("/api/state").json()["simulation"]["running"] is True
    assert client.post("/api/simulation/pause").status_code == 200
    assert sim.running is False
    assert client.get("/api/state").json()["simulation"]["running"] is False

    # SET_SPEED is stored on the backend
    assert client.post("/api/simulation/speed", json={"speed": 2.0}).status_code == 200
    assert sim.sim_speed == pytest.approx(2.0)
    assert client.get("/api/state").json()["simulation"]["speed"] == pytest.approx(2.0)
    # RESET returns the backend to its configured initial conditions
    current = client.get("/api/state").json()["state_identity"]["network_timestep"]
    client.post("/api/simulation/step")
    assert client.get("/api/state").json()["state_identity"]["network_timestep"] == current + 1
    client.post("/api/simulation/reset")
    reset = client.get("/api/state").json()
    assert reset["state_identity"]["network_timestep"] == 0
    assert reset["mass_balance"]["status"] == "NOT_CHECKED"


def _post_raw_json(url: str, raw_body: str):
    """NaN / Infinity are not JSON-compliant for httpx, but a hostile client can
    still put them on the wire, so the API must reject them."""
    return client.post(url, content=raw_body,
                       headers={"Content-Type": "application/json"})


def test_command_validation_is_preserved(live_sim):
    """
    The Stage 4 contract is preserved exactly: NON-FINITE / non-numeric values
    are REJECTED (422), finite out-of-range values are CLAMPED into the
    documented command domain (never injected as-is), unknown reservoirs are 404
    and an unknown mode is 422.
    """
    assert client.post("/api/gate/reservoir_1", json={"value": 250}).status_code == 200
    assert live_sim.manual_gates["Virtual Reservoir A"] == pytest.approx(100.0)
    assert client.post("/api/gate/reservoir_1", json={"value": -40}).status_code == 200
    assert live_sim.manual_gates["Virtual Reservoir A"] == pytest.approx(0.0)
    assert _post_raw_json("/api/gate/reservoir_1", '{"value": NaN}').status_code == 422
    assert client.post("/api/gate/reservoir_1", json={"value": "50"}).status_code == 422

    assert client.post("/api/storm", json={"value": 5}).status_code == 200
    assert live_sim.storm_intensity == pytest.approx(1.0)
    assert _post_raw_json("/api/storm", '{"value": Infinity}').status_code == 422

    assert client.post("/api/simulation/speed", json={"speed": 0}).status_code == 200
    assert live_sim.sim_speed == pytest.approx(0.05)      # never 0 (loop would divide by it)
    assert _post_raw_json("/api/simulation/speed", '{"speed": NaN}').status_code == 422

    assert client.post("/api/controller/mode", json={"mode": "TURBO"}).status_code == 422
    # an unknown reservoir id changes nothing and is rejected (400, as in Stage 4)
    assert client.post("/api/gate/reservoir_9", json={"value": 50}).status_code == 400


# ===========================================================================
# 9, 13. All four reservoirs incl. Idukki / D, and D's control path
# ===========================================================================

def test_all_four_reservoirs_are_represented(live_sim):
    payload = client.get("/api/state").json()
    assert payload["cascade"]["count"] == 4
    assert payload["cascade"]["order"] == [f"reservoir_{i}" for i in (1, 2, 3, 4)]
    for index, nid in enumerate(NODES, start=1):
        res = _twin_reservoir(payload, f"reservoir_{index}")
        assert res["node_id"] == nid
        assert res["repository_name"] == VALIDATED_NAMES[nid]
    assert payload["cascade"]["terminal"]["name"] == "Idukki"
    assert _twin_reservoir(payload, "reservoir_4")["repository_name"] == "Idukki"


def test_idukki_control_path_remains_intact(live_sim):
    sim = live_sim
    client.post("/api/simulation/reset")
    # reservoir_4 (Idukki) is commandable through the validated boundary
    assert client.post("/api/gate/reservoir_4", json={"value": 37.0}).status_code in (200, 204)
    assert sim.manual_gates["Virtual Reservoir D"] == pytest.approx(37.0)
    assert '"Virtual Reservoir D": 100.0' not in STATE_MANAGER_PATH.read_text(encoding="utf-8")
    # ...and D carries a real, non-pinned gate in the payload
    payload = client.get("/api/state").json()
    assert 0.0 <= _twin_reservoir(payload, "reservoir_4")["gate"] <= 1.0


# ===========================================================================
# 10, 11, 12. Provenance survives the whole path
# ===========================================================================

def test_mass_balance_diagnostics_survive_the_path(live_sim):
    client.post("/api/simulation/reset")
    client.post("/api/simulation/step")
    rest = client.get("/api/state").json()["mass_balance"]
    with client.websocket_connect("/ws/state") as ws:
        wss = ws.receive_json()["mass_balance"]

    for block in (rest, wss):
        assert block["checked"] is True
        assert block["status"] == "PASS"
        assert block["unit"] == "MCM"
        assert block["tolerance"] == 1e-9
        assert block["reservoirs_checked"] == 4
        assert len(block["per_reservoir"]) == 4
        assert block["applied_action_source"] in ("MANUAL_OPERATOR_GATES", "UNKNOWN")
        assert block["fail_safe"] == "NONE_DEFINED_IN_EXISTING_ARCHITECTURE"
    assert rest == wss


def test_forecast_provenance_survives_the_path(live_sim):
    payload = client.get("/api/state").json()
    fp = payload["forecast_provenance"]
    assert fp["model"] == "LSTM_V3_LOGTARGET"
    assert fp["live_forecasts_are_validated"] is False
    assert "provenance_note" in fp
    # every reservoir carries its own provenance/status verbatim
    for index in (1, 2, 3, 4):
        res = _twin_reservoir(payload, f"reservoir_{index}")
        assert "forecast_status" in res
        assert "forecast_provenance" in res
        assert "forecast_validated_metrics_apply" in res
    # the backend-computed forecast summary is part of the payload
    fs = payload["forecast_summary"]
    assert fs["unit"] == "m3/s"
    assert fs["reservoirs_total"] == 4
    assert fs["status"] in ("COMPLETE", "PARTIAL", "UNAVAILABLE")
    assert isinstance(fs["missing_forecast"], list)


def test_controller_safety_and_downstream_statuses_survive_the_path(live_sim):
    payload = client.get("/api/state").json()
    control = payload["control"]
    for key in ("controller_status", "safety_layer_status", "downstream_status",
                "downstream_capacity_mcm_day", "downstream_capacity_achieved",
                "final_safe_control_action_source", "forecast_control_eligible"):
        assert key in control, f"{key} missing from the control block"
    # ...and they match the authoritative decision object
    decision = live_sim.mpc_orchestrator.status_dict()
    assert control["controller_status"] == decision["controller_status"]
    assert control["safety_layer_status"] == decision["safety_layer_status"]
    assert control["downstream_status"] == decision["downstream_status"]

    # the downstream block is derived by the BACKEND from the authoritative physics
    down = payload["downstream"]
    assert down["unit"] == "m3/s"
    assert down["capacity_m3_s"] == pytest.approx(50.0 * 1e6 / 86400, rel=1e-9)
    assert down["source"].startswith("ReservoirNetwork.terminal_outflow")


# ===========================================================================
# 14. No second live simulation clock
# ===========================================================================

def test_no_second_live_simulation_clock():
    html = TWIN_INDEX_PATH.read_text(encoding="utf-8")
    code = _module_script(html)
    # the browser never counts simulation steps
    assert "setInterval" not in code
    assert "sim_step_index" not in code
    assert "currentStep" not in code
    # the only counters are the render loop's own animation clocks
    assert "_time" in code and "_hudT" in code
    # freshness is expressed by the BACKEND's identity, displayed verbatim
    assert "state_identity" in code
    assert "this.data.stateId = ident && ident.state_id ? String(ident.state_id) : null;" in code
    assert "el.stateid.textContent = hasState ? (this.data.stateId || '--') : '--';" in code


def test_state_identity_is_derived_from_existing_backend_counters(live_sim):
    sim = live_sim
    client.post("/api/simulation/reset")
    before = client.get("/api/state").json()["state_identity"]
    steps = before["sim_step_index"]
    assert before["network_timestep"] == 0
    assert before["state_id"] == f"step{steps}-t0"

    client.post("/api/simulation/step")
    after = client.get("/api/state").json()["state_identity"]
    assert after["sim_step_index"] == steps + 1
    assert after["network_timestep"] == 1
    assert after["state_id"] == f"step{steps + 1}-t1"
    assert after["network_timestep"] == sim.bridge.cascade.network.timestep
    # the identity names its (existing) sources — no new clock is introduced
    assert "sim_step_index" in after["source"] and "ReservoirNetwork.timestep" in after["source"]

    # a RESET is distinguishable: the physics clock restarts, the step counter runs on
    client.post("/api/simulation/step")
    client.post("/api/simulation/reset")
    reset = client.get("/api/state").json()["state_identity"]
    assert reset["network_timestep"] == 0
    assert reset["sim_step_index"] > after["sim_step_index"]


def test_adapter_reports_missing_identity_honestly():
    from src.dashboard.twin_component.state_adapter import (
        adapt_state_for_twin, _state_identity_block, _simulation_block,
    )
    twin = adapt_state_for_twin({"reservoirs": {}, "downstream_flow": 0}, "MANUAL", 0.0)
    ident = twin["state_identity"]
    assert ident["state_id"] is None
    assert ident["network_timestep"] is None
    assert ident["sim_step_index"] is None
    assert ident["reason"] == "NO_STATE_IDENTITY_PROVENANCE"
    assert twin["simulation"]["running"] is None
    assert twin["simulation"]["reason"] == "NO_SIMULATION_PROVENANCE"
    assert _state_identity_block(None)["state_id"] is None
    assert _simulation_block(None)["running"] is None
    # a populated identity passes through verbatim
    ident = _state_identity_block({"sim_step_index": 7, "network_timestep": 7,
                                  "state_id": "step7-t7", "source": "s"})
    assert ident["state_id"] == "step7-t7"
    assert _simulation_block({"running": True, "speed": 2.0, "source": "s"})["running"] is True


# ===========================================================================
# 15. Successive WebSocket updates = successive authoritative states
# ===========================================================================

def test_play_pause_broadcast_the_resulting_authoritative_state(live_sim):
    """
    A command must come back to the client as the resulting state, otherwise the
    twin displays a stale run state. (Stage 12 found PAUSE stopping the backend
    without broadcasting at all.)
    """
    client.post("/api/simulation/pause")
    with client.websocket_connect("/ws/state") as ws:
        ws.receive_json()                                  # initial state
        client.post("/api/simulation/play")
        assert ws.receive_json()["simulation"]["running"] is True

        client.post("/api/simulation/pause")
        pushed = None
        for _ in range(3):                                 # tolerate loop pushes
            pushed = ws.receive_json()
            if pushed["simulation"]["running"] is False:
                break
        assert pushed is not None and pushed["simulation"]["running"] is False
    assert live_sim.running is False


def test_repeated_websocket_updates_follow_successive_backend_states(live_sim):
    client.post("/api/simulation/reset")
    client.post("/api/simulation/pause")

    with client.websocket_connect("/ws/state") as ws:
        first = ws.receive_json()
        assert _identity(first)["network_timestep"] == 0

        seen = [_identity(first)["network_timestep"]]
        for _ in range(2):
            client.post("/api/simulation/step")
            pushed = ws.receive_json()
            seen.append(_identity(pushed)["network_timestep"])

    assert seen == [0, 1, 2], f"WebSocket updates did not track backend steps: {seen}"
    # each pushed payload really is the authoritative state at that step
    final = client.get("/api/state").json()
    assert _identity(final)["network_timestep"] == 2


# ===========================================================================
# Display classifications are the backend's, and correct
# ===========================================================================

def test_downstream_classification_is_the_backends():
    from src.dashboard.twin_component.state_adapter import (
        classify_downstream_status, classify_trend, classify_storm_level,
        DOWNSTREAM_STATUS_NORMAL, DOWNSTREAM_STATUS_WARNING, DOWNSTREAM_STATUS_CRITICAL,
        DOWNSTREAM_STATUS_UNKNOWN, TREND_RISING, TREND_FALLING, TREND_STEADY,
    )
    assert classify_downstream_status(None, 10.0) == DOWNSTREAM_STATUS_UNKNOWN
    assert classify_downstream_status(1.0, 0.0) == DOWNSTREAM_STATUS_UNKNOWN
    assert classify_downstream_status(5.0, 10.0) == DOWNSTREAM_STATUS_NORMAL
    assert classify_downstream_status(8.0, 10.0) == DOWNSTREAM_STATUS_WARNING
    assert classify_downstream_status(11.0, 10.0) == DOWNSTREAM_STATUS_CRITICAL

    assert classify_trend(2.0) == TREND_RISING
    assert classify_trend(-2.0) == TREND_FALLING
    assert classify_trend(0.0) == TREND_STEADY
    assert classify_trend(None) is None

    assert classify_storm_level(0.1) == "LIGHT"
    assert classify_storm_level(0.7) == "HEAVY"
    assert classify_storm_level(0.95) == "SEVERE"
    assert classify_storm_level(None) is None


def test_every_reservoir_carries_a_backend_trend_and_net_flux(live_sim):
    client.post("/api/simulation/step")
    payload = client.get("/api/state").json()
    for index in (1, 2, 3, 4):
        res = _twin_reservoir(payload, f"reservoir_{index}")
        assert "net_flux_m3_s" in res
        assert res["trend"] in ("RISING", "FALLING", "STEADY", None)
        if res["trend"] is not None:
            assert res["net_flux_m3_s"] is not None


# ===========================================================================
# Protected runtime components and frozen artifacts
# ===========================================================================

def test_protected_runtime_components_are_unmodified():
    for path in (NETWORK_PATH, MPC_PATH, SAFETY_PATH, GUARD_PATH, ORCH_PATH,
                 MASS_BALANCE_PATH):
        source = path.read_text(encoding="utf-8")
        for forbidden in ("state_identity", "state_adapter", "twin_component",
                          "forecast_summary", "classify_downstream_status"):
            assert forbidden not in source, f"{path.name} must not know about Stage 12"


def test_frozen_artifacts_are_unchanged_by_stage12(live_sim):
    before = {p: _sha(p) for p in FROZEN_ARTIFACTS}
    client.post("/api/simulation/step")
    client.get("/api/state")
    assert {p: _sha(p) for p in FROZEN_ARTIFACTS} == before


def test_frozen_artifacts_still_match_the_phase15_3_manifest():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["all_match"] is True
    for rel_path, record in manifest["pre_validation"].items():
        path = _PROJECT_ROOT / rel_path
        assert path.exists(), f"frozen artifact missing: {rel_path}"
        assert record["sha256"] in {_sha(path), _sha_lf(path)}, f"MODIFIED: {rel_path}"


# ===========================================================================
# Performance impact
# ===========================================================================

def test_adapter_overhead_is_negligible(live_sim):
    from src.dashboard.twin_component.state_adapter import adapt_state_for_twin
    sim_state = live_sim.bridge.get_state({})
    sim_state["control"] = live_sim.mpc_orchestrator.status_dict()
    sim_state["mass_balance"] = live_sim.bridge.mass_balance_diagnostic()
    sim_state["state_identity"] = {"sim_step_index": 1, "network_timestep": 1,
                                   "state_id": "step1-t1", "source": "s"}
    sim_state["simulation"] = {"running": False, "speed": 1.0, "source": "s"}

    adapt_state_for_twin(sim_state, "MANUAL", 0.0)          # warm-up
    iterations = 300
    t0 = time.perf_counter()
    for _ in range(iterations):
        adapt_state_for_twin(sim_state, "MANUAL", 0.0)
    per_call_ms = (time.perf_counter() - t0) * 1000.0 / iterations

    # the adapter is a presentation transform; it must stay far below one MPC cycle
    assert per_call_ms < 5.0, f"adapter is too slow: {per_call_ms:.4f} ms/call"
    print(f"\n[Stage 12] adapt_state_for_twin: {per_call_ms:.4f} ms/call")
