"""
Stage 4 — SINGLE AUTHORITATIVE SIMULATION: integration tests.

Proves the Digital Twin has exactly ONE live simulation/state producer, that
commands reach it, that the WebSocket publishes from it, and that neither the
browser nor Streamlit can create or overwrite authoritative state.

Target architecture:

    FastAPI -> GlobalSimulationState -> LiveCascadeAdapter -> ReservoirNetwork
            -> authoritative state -> WebSocket -> Three.js Digital Twin
"""

import ast
import copy
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.dashboard.api import state_manager  # noqa: E402
from src.dashboard.api.app import app  # noqa: E402
from src.dashboard.api import routes as routes_module  # noqa: E402
from src.network_env.live_cascade_adapter import LiveCascadeAdapter  # noqa: E402
from src.network_env.reservoir_network import ReservoirNetwork  # noqa: E402

client = TestClient(app)

APP_PATH = _PROJECT_ROOT / "src" / "dashboard" / "app.py"
SIM_PAGE_PATH = _PROJECT_ROOT / "src" / "dashboard" / "simulation_page.py"
WEB_INDEX_PATH = _PROJECT_ROOT / "src" / "dashboard" / "web" / "index.html"
EMBEDDED_TWIN_PATH = _PROJECT_ROOT / "src" / "dashboard" / "twin_component" / "reservoir_twin.html"

TARGET_ORDER = [
    "Virtual Reservoir A",
    "Virtual Reservoir B",
    "Virtual Reservoir C",
    "Virtual Reservoir D",
]
TARGET_DELAYS = {("Virtual Reservoir A", "Virtual Reservoir B"): 2,
                 ("Virtual Reservoir B", "Virtual Reservoir C"): 1,
                 ("Virtual Reservoir C", "Virtual Reservoir D"): 1}
TARGET_ATTENUATION = {("Virtual Reservoir A", "Virtual Reservoir B"): 0.90,
                      ("Virtual Reservoir B", "Virtual Reservoir C"): 0.85,
                      ("Virtual Reservoir C", "Virtual Reservoir D"): 0.80}

# A payload that tries to look like simulation STATE rather than a command.
FORGED_STATE = {
    "reservoirs": {"reservoir_1": {"storage": 9999.0, "water_level": 1.0, "gate": 1.0}},
    "storage": 9999.0,
    "spill": 4242.0,
    "downstream_flow": 1e9,
    "controller_mode": "HACKED",
    "storm_intensity": 99.0,
}


def _physical_state() -> dict:
    """Snapshot the authoritative physical state (storages)."""
    cascade = state_manager.sim_state.bridge.cascade
    return {n: cascade.reservoirs[n].state.storage_mcm for n in cascade.cascade_order}


def _sim_control_state() -> dict:
    s = state_manager.sim_state
    return {
        "mode": s.mode,
        "storm": s.storm_intensity,
        "speed": s.sim_speed,
        "running": s.running,
        "gates": copy.deepcopy(s.manual_gates),
    }


# ===========================================================================
# 1. Exactly ONE authoritative live simulation
# ===========================================================================

def test_exactly_one_live_simulation_instance_exists():
    """Only one GlobalSimulationState may exist in the FastAPI process."""
    assert state_manager.authoritative_instance_count() == 1, (
        "Expected exactly one live simulation instance, found "
        f"{state_manager.authoritative_instance_count()}"
    )


def test_all_entry_points_share_the_same_instance():
    """Routes, the WebSocket app and the accessor must all use one object."""
    assert routes_module.sim_state is state_manager.sim_state
    assert app.state is not None  # sanity: app imported
    assert state_manager.get_authoritative_state_manager() is state_manager.sim_state


def test_authoritative_instance_runs_the_stage3_physics():
    """The single live simulation is the validated Stage 3 stack."""
    cascade = state_manager.sim_state.bridge.cascade
    assert isinstance(cascade, LiveCascadeAdapter)
    assert isinstance(cascade.network, ReservoirNetwork)


def test_only_the_authoritative_module_constructs_a_live_bridge():
    """No other module in src/ may construct a SimBridge."""
    hits = []
    for path in (_PROJECT_ROOT / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "SimBridge(" in text:
            hits.append(str(path.relative_to(_PROJECT_ROOT)).replace("\\", "/"))
    assert hits == ["src/dashboard/api/state_manager.py"], (
        f"Unexpected live bridge construction in: {hits}"
    )


def test_no_module_in_src_constructs_a_second_simulation_state():
    hits = []
    for path in (_PROJECT_ROOT / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "GlobalSimulationState()" in text:
            hits.append(str(path.relative_to(_PROJECT_ROOT)).replace("\\", "/"))
    assert hits == ["src/dashboard/api/state_manager.py"], (
        f"Unexpected extra simulation construction in: {hits}"
    )


# ===========================================================================
# 2. Commands affect the AUTHORITATIVE simulation
# ===========================================================================

def test_gate_command_reaches_authoritative_simulation():
    before = state_manager.sim_state.manual_gates["Virtual Reservoir A"]
    response = client.post("/api/gate/reservoir_1", json={"value": 42.0})
    assert response.status_code == 200
    assert state_manager.sim_state.manual_gates["Virtual Reservoir A"] == 42.0
    assert state_manager.sim_state.manual_gates["Virtual Reservoir A"] != before or before == 42.0


def test_storm_command_reaches_authoritative_simulation():
    response = client.post("/api/storm", json={"value": 0.5})
    assert response.status_code == 200
    assert state_manager.sim_state.storm_intensity == pytest.approx(0.5)


def test_mode_command_reaches_authoritative_simulation():
    response = client.post("/api/controller/mode", json={"mode": "AI"})
    assert response.status_code == 200
    assert state_manager.sim_state.mode == "AI"
    client.post("/api/controller/mode", json={"mode": "MANUAL"})
    assert state_manager.sim_state.mode == "MANUAL"


def test_step_command_advances_authoritative_physics():
    before = _physical_state()
    response = client.post("/api/simulation/step")
    assert response.status_code == 200
    after = _physical_state()
    assert after != before, "STEP did not advance the authoritative simulation"


def test_reset_command_restores_authoritative_initial_conditions():
    client.post("/api/simulation/step")
    response = client.post("/api/simulation/reset")
    assert response.status_code == 200
    cascade = state_manager.sim_state.bridge.cascade
    for name in cascade.cascade_order:
        res = cascade.reservoirs[name]
        assert res.state.storage_mcm == pytest.approx(res.capacity_mcm * 0.5, abs=1e-6)


# ===========================================================================
# 3. WebSocket state comes from the SAME authoritative instance
# ===========================================================================

def test_websocket_publishes_authoritative_state():
    """The WS payload must reflect the authoritative simulation's own state."""
    client.post("/api/controller/mode", json={"mode": "MANUAL"})
    client.post("/api/gate/reservoir_1", json={"value": 37.0})

    with client.websocket_connect("/ws/state") as ws:
        data = ws.receive_json()

    assert data["reservoirs"]["reservoir_1"]["gate"] == pytest.approx(0.37, abs=1e-9)
    assert data["controller_mode"] == "MANUAL"


def test_websocket_and_rest_state_agree():
    """WS and REST must publish the identical authoritative snapshot."""
    client.post("/api/controller/mode", json={"mode": "MANUAL"})
    client.post("/api/gate/reservoir_2", json={"value": 12.0})

    rest = client.get("/api/state").json()
    with client.websocket_connect("/ws/state") as ws:
        ws_state = ws.receive_json()

    for key in ("reservoir_1", "reservoir_2", "reservoir_3"):
        assert ws_state["reservoirs"][key]["gate"] == pytest.approx(
            rest["reservoirs"][key]["gate"], abs=1e-9
        )
        assert ws_state["reservoirs"][key]["storage"] == pytest.approx(
            rest["reservoirs"][key]["storage"], abs=1e-9
        )


def test_websocket_state_changes_when_authoritative_simulation_changes():
    """A command that moves the simulation must be visible over the WS feed."""
    client.post("/api/controller/mode", json={"mode": "MANUAL"})
    client.post("/api/gate/reservoir_3", json={"value": 5.0})
    with client.websocket_connect("/ws/state") as ws:
        before = ws.receive_json()

    client.post("/api/gate/reservoir_3", json={"value": 65.0})

    with client.websocket_connect("/ws/state") as ws:
        after = ws.receive_json()

    assert before["reservoirs"]["reservoir_3"]["gate"] == pytest.approx(0.05, abs=1e-9)
    assert after["reservoirs"]["reservoir_3"]["gate"] == pytest.approx(0.65, abs=1e-9)


def test_websocket_uses_the_authoritative_singleton_object():
    """
    Structural proof: the WS endpoint reads the module-level singleton, which
    the command routes also import.
    """
    api_source = (_PROJECT_ROOT / "src" / "dashboard" / "api" / "app.py").read_text(encoding="utf-8")
    assert "sim_state.get_adapted_state()" in api_source
    assert routes_module.sim_state is state_manager.sim_state


# ===========================================================================
# 4. Frontend cannot overwrite backend state
# ===========================================================================

def test_no_state_write_endpoint_exists():
    """Ambiguity guard: there is NO endpoint that accepts simulation state."""
    assert client.post("/api/state", json=FORGED_STATE).status_code == 405
    assert client.put("/api/state", json=FORGED_STATE).status_code == 405
    assert client.delete("/api/state").status_code == 405


def test_forged_state_payload_is_ignored_by_gate_endpoint():
    client.post("/api/controller/mode", json={"mode": "MANUAL"})
    client.post("/api/gate/reservoir_1", json={"value": 50.0})
    before = _physical_state()

    response = client.post("/api/gate/reservoir_1", json={"value": 50.0, **FORGED_STATE})
    assert response.status_code == 200
    assert _physical_state() == before, "forged state mutated authoritative physics"
    # The forged physical fields were ignored; only the bounded command applied.
    assert state_manager.sim_state.manual_gates["Virtual Reservoir A"] == 50.0


def test_forged_state_payload_cannot_set_controller_mode():
    client.post("/api/controller/mode", json={"mode": "MANUAL"})
    client.post("/api/storm", json={"value": 0.0, **FORGED_STATE})
    assert state_manager.sim_state.mode == "MANUAL", "forged payload set the controller mode"
    assert state_manager.sim_state.storm_intensity == pytest.approx(0.0)


def test_unknown_reservoir_id_is_rejected_without_state_change():
    before_ctrl = _sim_control_state()
    before_phys = _physical_state()

    for bad in ("reservoir_5", "reservoir_99", "Virtual Reservoir D", "'; DROP TABLE"):
        response = client.post(f"/api/gate/{bad}", json={"value": 50.0})
        assert response.status_code == 400, f"{bad} should be rejected"

    assert _sim_control_state() == before_ctrl
    assert _physical_state() == before_phys


def test_invalid_controller_mode_is_rejected():
    client.post("/api/controller/mode", json={"mode": "MANUAL"})
    before = state_manager.sim_state.mode

    for bad in ("HACK", "manual", "AI ", "", "MANUAL;DROP"):
        response = client.post("/api/controller/mode", json={"mode": bad})
        assert response.status_code == 422, f"mode {bad!r} should be rejected"

    assert state_manager.sim_state.mode == before


def _post_raw_json(url: str, raw_body: str):
    """
    POST a raw JSON body.

    Needed because ``NaN`` / ``Infinity`` are not JSON-compliant for httpx's
    serializer, yet Python's ``json.loads`` (used server-side) *does* accept
    them. A hostile client can therefore send them on the wire, and the API
    must reject them — which is exactly what these tests verify.
    """
    return client.post(url, content=raw_body,
                       headers={"Content-Type": "application/json"})


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_storm_is_rejected(literal):
    client.post("/api/storm", json={"value": 0.25})
    before = state_manager.sim_state.storm_intensity

    response = _post_raw_json("/api/storm", f'{{"value": {literal}}}')
    assert response.status_code == 422
    assert state_manager.sim_state.storm_intensity == before


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_gate_is_rejected(literal):
    client.post("/api/gate/reservoir_1", json={"value": 10.0})
    before = state_manager.sim_state.manual_gates["Virtual Reservoir A"]

    response = _post_raw_json("/api/gate/reservoir_1", f'{{"value": {literal}}}')
    assert response.status_code == 422
    assert state_manager.sim_state.manual_gates["Virtual Reservoir A"] == before


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_speed_is_rejected(literal):
    client.post("/api/simulation/speed", json={"speed": 2.0})
    before = state_manager.sim_state.sim_speed

    response = _post_raw_json("/api/simulation/speed", f'{{"speed": {literal}}}')
    assert response.status_code == 422
    assert state_manager.sim_state.sim_speed == before


def test_non_numeric_command_values_are_rejected():
    assert client.post("/api/gate/reservoir_1", json={"value": "wide open"}).status_code == 422
    assert client.post("/api/storm", json={"value": None}).status_code == 422
    assert client.post("/api/simulation/speed", json={"speed": "fast"}).status_code == 422


def test_speed_can_never_be_zero():
    """A zero speed would raise ZeroDivisionError inside the authoritative loop."""
    for bad in (0.0, -1.0, -1e9, 1e-12):
        response = client.post("/api/simulation/speed", json={"speed": bad})
        assert response.status_code == 200
        assert state_manager.sim_state.sim_speed > 0.0
    assert 1.0 / state_manager.sim_state.sim_speed > 0.0


def test_out_of_range_commands_are_clamped_not_injected():
    """Finite out-of-domain values clamp (units contract); nothing explodes."""
    assert client.post("/api/gate/reservoir_1", json={"value": 1e9}).status_code == 200
    assert state_manager.sim_state.manual_gates["Virtual Reservoir A"] == 100.0

    assert client.post("/api/gate/reservoir_1", json={"value": -1e9}).status_code == 200
    assert state_manager.sim_state.manual_gates["Virtual Reservoir A"] == 0.0

    assert client.post("/api/storm", json={"value": 1e9}).status_code == 200
    assert state_manager.sim_state.storm_intensity == 1.0


def test_bootstrap_payload_cannot_inject_state():
    """Extra/unknown JSON fields are ignored, never written onto the simulation."""
    response = client.post("/api/simulation/play", json=FORGED_STATE)
    assert response.status_code == 200
    assert state_manager.sim_state.running is True
    client.post("/api/simulation/pause")
    # The forged keys did not become attributes on the simulation object.
    for key in ("storage", "spill", "downstream_flow"):
        assert not isinstance(getattr(state_manager.sim_state, key, None), float)


# ===========================================================================
# 5. Streamlit cannot create a competing live Digital Twin state
# ===========================================================================

def _forbidden_runtime_names(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    forbidden_modules = {"sim_bridge", "simulator.engine", "src.simulator.engine"}
    forbidden_names = {
        "SimBridge", "SimulationEngine", "VirtualCascade", "LiveCascadeAdapter",
        "GlobalSimulationState",
    }
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module in forbidden_modules:
                problems.append(f"imports {node.module}")
            for alias in node.names:
                if alias.name in forbidden_names:
                    problems.append(f"imports {alias.name}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in forbidden_modules:
                    problems.append(f"imports {alias.name}")
        elif isinstance(node, ast.Name) and node.id in forbidden_names:
            problems.append(f"references {node.id}")
    return problems


def test_streamlit_app_owns_no_simulation():
    problems = _forbidden_runtime_names(APP_PATH)
    assert problems == [], f"Streamlit app still owns simulation logic: {problems}"


def test_streamlit_app_does_not_import_the_research_page():
    source = APP_PATH.read_text(encoding="utf-8")
    assert "simulation_page" not in source, (
        "The offline research page must not be wired into the live dashboard"
    )
    assert "render_simulation_page" not in source


def test_streamlit_app_reads_only_the_authoritative_http_state():
    source = APP_PATH.read_text(encoding="utf-8")
    assert "/api/state" in source, "Streamlit must read the authoritative REST state"
    assert "TWIN_URL" in source


def test_offline_research_engine_is_isolated_and_dormant():
    """
    `SimulationEngine` may remain for offline experiments, but it must not be
    reachable from the live Digital Twin path.
    """
    import subprocess
    # Not imported by the live dashboard app.
    assert "SimulationEngine" not in APP_PATH.read_text(encoding="utf-8").split("STAGE 4")[0]
    # `render_simulation_page` has no callers anywhere in src/.
    result = subprocess.run(
        ["git", "grep", "-n", "render_simulation_page", "--", "src"],
        capture_output=True, text=True, cwd=str(_PROJECT_ROOT),
    )
    lines = [l for l in result.stdout.splitlines() if l.strip()]
    definition_lines = [l for l in lines if "def render_simulation_page" in l]
    assert len(definition_lines) == 1, f"unexpected definition sites: {lines}"
    assert len(lines) == 1, f"render_simulation_page is wired somewhere: {lines}"
    assert SIM_PAGE_PATH.exists(), "offline research page should be retained (Req. 7)"


# ===========================================================================
# 6. The authoritative twin has a single state source (frontend)
# ===========================================================================

def test_authoritative_twin_has_no_state_injection_path():
    """`web/index.html` must not accept pushed state — WebSocket only."""
    content = WEB_INDEX_PATH.read_text(encoding="utf-8")
    assert "streamlit:render" not in content, (
        "the authoritative twin must not accept postMessage state injection"
    )
    assert "addEventListener(\"message\"" not in content, (
        "the authoritative twin must not listen for injected state messages"
    )
    # Its legitimate source is the WebSocket feed.
    api_js = (_PROJECT_ROOT / "src" / "dashboard" / "web" / "api.js").read_text(encoding="utf-8")
    assert "/ws/state" in api_js, "the authoritative twin must consume the WS feed"


def test_frontend_command_surface_is_commands_only():
    """api.js may only POST bounded commands — no state push."""
    api_js = (_PROJECT_ROOT / "src" / "dashboard" / "web" / "api.js").read_text(encoding="utf-8")
    for endpoint in ("/gate/", "/simulation/play", "/simulation/pause",
                     "/simulation/step", "/simulation/reset",
                     "/simulation/speed", "/storm", "/controller/mode"):
        assert endpoint in api_js, f"missing command endpoint {endpoint}"
    assert "reservoirs" not in api_js, "the frontend must not push reservoir state"


def test_embedded_twin_is_documented_as_a_non_authoritative_mirror():
    content = EMBEDDED_TWIN_PATH.read_text(encoding="utf-8")
    assert "NON-AUTHORITATIVE MIRROR" in content
    assert "web/index.html" in content


# ===========================================================================
# 7. Stage 3 authoritative topology preserved (Req. 9)
# ===========================================================================

def test_stage3_topology_delays_and_attenuation_unchanged():
    cascade = state_manager.sim_state.bridge.cascade
    assert cascade.cascade_order == TARGET_ORDER
    for conn in cascade.network.connections:
        key = (conn.source, conn.destination)
        assert conn.delay == TARGET_DELAYS[key]
        assert conn.attenuation == pytest.approx(TARGET_ATTENUATION[key])


def test_authoritative_runtime_topology_survives_a_reset():
    client.post("/api/simulation/reset")
    cascade = state_manager.sim_state.bridge.cascade
    edges = [(c.source, c.destination) for c in cascade.network.connections]
    assert edges == list(zip(TARGET_ORDER[:-1], TARGET_ORDER[1:]))
    assert cascade.current_downstream_flow >= 0.0
    assert cascade.downstream_capacity == pytest.approx(50.0)
