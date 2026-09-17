"""
Stage 13 — STREAMLIT READ-ONLY / COMMAND-PROXY: tests.

Proves that the Streamlit dashboard is a VIEWER + COMMAND PROXY of the ONE
authoritative simulation and never a second runtime:

  * it owns no simulation object, no gate/reservoir state and no clock;
  * it has no way to execute reservoir physics;
  * the only thing it can do to the simulation is POST a bounded command to the
    authoritative FastAPI backend, which validates it and runs the full
    validated path (MPC -> SafetyLayer -> DownstreamCapacityGuard -> physics);
  * everything it displays is the backend's own verdict;
  * all four reservoirs — Idukki / Reservoir D included — are first class;
  * the Three.js Digital Twin and Streamlit are BOTH consumers.

The Streamlit page is analysed statically (AST / source). Streamlit itself is
never imported here, so these tests run in any environment. The behavioural
proof that the real script drives the real backend lives in
``scripts/run_stage13_streamlit_runtime_audit.py`` (Phase 12 evidence).
"""

import ast
import math
import re
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.dashboard.api import routes as routes_module  # noqa: E402
from src.dashboard.api import state_manager  # noqa: E402
from src.dashboard.api.app import app  # noqa: E402

client = TestClient(app)

DASHBOARD_DIR = _PROJECT_ROOT / "src" / "dashboard"
APP_PATH = DASHBOARD_DIR / "app.py"
WEB_DIR = DASHBOARD_DIR / "web"
TWIN_DIR = DASHBOARD_DIR / "twin_component"

#: The Streamlit page and the other browser-facing artifacts. `sim_bridge.py`
#: and `api/` are deliberately excluded: they belong to the AUTHORITATIVE
#: backend, which is allowed to own the simulation.
FRONTEND_FILES = (
    APP_PATH,
    DASHBOARD_DIR / "data_bridge.py",
    DASHBOARD_DIR / "simulation_page.py",
    DASHBOARD_DIR / "cascade_3d.py",
    DASHBOARD_DIR / "reservoir_3d.py",
    TWIN_DIR / "reservoir_twin.html",
    WEB_DIR / "index.html",
    WEB_DIR / "api.js",
)

#: Names that only the authoritative simulation may construct or touch.
SIMULATION_NAMES = {
    "SimBridge", "SimulationEngine", "VirtualCascade", "LiveCascadeAdapter",
    "ReservoirNetwork", "GlobalSimulationState", "MassBalanceMonitor",
}

#: Model names that must never be imported by a frontend.
CONTROL_LAYER_TOKENS = (
    "mpc_controller", "live_mpc_orchestrator", "downstream_capacity_guard",
    "from src.controller", "import safety",
)

#: PLAY / PAUSE / STEP / RESET / SET_SPEED / storm / mode / gate x4.
EXPECTED_PROXY_ENDPOINTS = {
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
}

ALL_RESERVOIRS = ("reservoir_1", "reservoir_2", "reservoir_3", "reservoir_4")
GATE_PATH = "/api/gate/reservoir_{}"


def app_source() -> str:
    return APP_PATH.read_text(encoding="utf-8")


def app_tree() -> ast.Module:
    return ast.parse(app_source(), filename=str(APP_PATH))


def literal_strings(tree: ast.Module) -> set:
    return {
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def source_of(func_name: str) -> str:
    tree = app_tree()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return ast.get_source_segment(app_source(), node) or ""
    raise AssertionError(f"{func_name} not found in {APP_PATH}")


def load_pure_function(func_name: str):
    """
    Compile ONE function out of the Streamlit script in isolation.

    Only used for helpers that touch no Streamlit and no global page state
    (`flow_text`, `residual_text`), so their real code can be unit-tested
    without executing the page.
    """
    tree = app_tree()
    target = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            target = node
            break
    assert target is not None, f"{func_name} is not a module-level function"
    module = ast.Module(body=[target], type_ignores=[])
    # The page's module-level stdlib imports the helper relies on.
    namespace: dict = {"math": math}
    exec(compile(module, str(APP_PATH), "exec"), namespace)  # noqa: S102
    return namespace[func_name]


def calls_to(tree: ast.Module, func_name: str):
    """Every call node of ``func_name`` with its positional arguments."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == func_name:
            out.append(node)
    return out


# ===========================================================================
# 1. The Streamlit page owns no simulation
# ===========================================================================

def test_streamlit_entrypoint_imports_no_simulation_layer():
    """No import in the page may reach a simulation/controller implementation."""
    forbidden_roots = {
        "sim_bridge", "simulator", "network_env", "controller", "modeling",
        "src", "state_manager",
    }
    problems = []
    for node in ast.walk(app_tree()):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in forbidden_roots:
                    problems.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            root = module.split(".")[0]
            if root in forbidden_roots or module in forbidden_roots:
                problems.append(module)
            for alias in node.names:
                if alias.name in SIMULATION_NAMES:
                    problems.append(f"{module}.{alias.name}")
    assert problems == [], f"Streamlit imports simulation code: {problems}"


def test_streamlit_entrypoint_constructs_or_references_no_simulation_object():
    tree = app_tree()
    referenced = {
        node.id for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id in SIMULATION_NAMES
    }
    assert referenced == set(), f"Streamlit references simulation objects: {referenced}"
    for name in SIMULATION_NAMES:
        assert f"{name}(" not in app_source(), f"Streamlit constructs {name}"


def test_streamlit_session_state_holds_only_the_transient_flash():
    """
    ``st.session_state`` may carry UI-only values. A physical value stored there
    would make the page a state owner, so only the documented flash key is
    allowed — no gates, storage, timestep, forecasts or controller state.
    """
    written = set()
    for node in ast.walk(app_tree()):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Attribute) \
                        and target.value.attr == "session_state":
                    written.add(ast.literal_eval(target.slice))
    assert written == {"_cmd_flash"}, f"Streamlit stores extra state: {written}"


def test_streamlit_has_no_simulation_clock_constructs():
    """No thread, task, timer, while-loop or simulation-advancing call may exist."""
    tree = app_tree()
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.While)], (
        "a while loop could become a simulation clock"
    )

    clock_modules = {"threading", "asyncio", "multiprocessing", "concurrent", "sched"}
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert not (imported & clock_modules), f"clock modules imported: {imported & clock_modules}"

    advancing = {"step", "init_cascade", "Thread", "Timer", "create_task",
                 "run_forever", "start", "tick", "advance"}
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.attr if isinstance(node.func, ast.Attribute) else (
            node.func.id if isinstance(node.func, ast.Name) else None
        )
        if name in advancing:
            offenders.append(name)
    assert offenders == [], f"Streamlit advances/owns a clock: {offenders}"


def test_streamlit_sleep_is_a_read_only_refresh_not_a_tick():
    """
    One ``time.sleep`` exists (the operator's read-only "Live follow"). It must
    sit in a branch whose only effect is re-running the script to re-READ the
    backend — never advancing anything.
    """
    tree = app_tree()
    sleeps = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "sleep"
    ]
    assert len(sleeps) == 1, f"expected exactly one read-only sleep, found {len(sleeps)}"

    refresh_if = None
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) \
                and node.test.id == "auto_refresh":
            refresh_if = node
    assert refresh_if is not None, "the read-only refresh branch is missing"

    body_calls = {
        node.func.attr for node in ast.walk(refresh_if)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert body_calls <= {"sleep", "rerun"}, (
        f"the refresh branch does more than re-read: {body_calls}"
    )
    # Nothing in the page advances the simulation.
    assert "dispatch_command" not in ast.get_source_segment(app_source(), refresh_if)


def test_streamlit_has_no_local_play_pause_reset_step_logic():
    tree = app_tree()
    names = {
        node.id for node in ast.walk(tree)
        if isinstance(node, ast.Name)
    }
    assert not (names & {"sim_running", "sim_tick", "manual_gates", "manual_inflows"}), (
        f"Streamlit keeps local simulation state: {names & {'sim_running', 'sim_tick', 'manual_gates', 'manual_inflows'}}"
    )
    # And no attribute on the page's own state ever names a physical quantity.
    physical = {"storage", "storage_mcm", "release", "spill", "timestep",
                "queue", "inflow"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in physical:
            value = node.value
            if isinstance(value, ast.Attribute) and value.attr == "session_state":
                raise AssertionError(f"physical value kept in session_state: {node.attr}")


# ===========================================================================
# 2. Commands are proxies to the authoritative backend
# ===========================================================================

def test_command_proxy_endpoints_are_the_only_api_routes_the_page_calls():
    declared = None
    for node in ast.walk(app_tree()):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id == "COMMAND_PROXY_ENDPOINTS":
            declared = {ast.literal_eval(e) for e in node.value.elts}
    assert declared == EXPECTED_PROXY_ENDPOINTS, f"unexpected proxy endpoints: {declared}"

    # Every "/api/..." string literal in the page is either the read-only state
    # route, the fixed prefix of the payload-driven gate route, or a declared
    # command route — no undocumented call can exist.
    literals = {s for s in literal_strings(app_tree()) if s.startswith("/api/")}
    assert literals - {"/api/state", "/api/gate/"} == EXPECTED_PROXY_ENDPOINTS, (
        f"page references undocumented routes: {sorted(literals)}"
    )
    # The gate route is built at runtime, so the allow-list — not the literal —
    # is what keeps it inside the command surface.
    assert "/api/gate/{key}" in source_of("render_command_proxy"), (
        "gate commands must be built from the payload's own cascade keys"
    )
    assert "if path not in COMMAND_PROXY_ENDPOINTS" in source_of("post_command"), (
        "the command helper must refuse any route outside the declared surface"
    )


def test_streamlit_dispatches_every_command_over_http():
    tree = app_tree()
    dispatched = {
        ast.literal_eval(call.args[1]) for call in calls_to(tree, "dispatch_command")
        if len(call.args) >= 2 and isinstance(call.args[1], ast.Constant)
    }
    dispatched |= {
        call.args[0].value for call in calls_to(tree, "post_command")
        if call.args and isinstance(call.args[0], ast.Constant)
    }
    # The four gate routes are built from the payload's own cascade inventory.
    for endpoint in dispatched:
        assert endpoint in EXPECTED_PROXY_ENDPOINTS
    # PLAY/PAUSE/STEP/RESET/SET_SPEED/mode/storm are all present verbatim.
    assert EXPECTED_PROXY_ENDPOINTS - dispatched == {
        "/api/gate/reservoir_1", "/api/gate/reservoir_2",
        "/api/gate/reservoir_3", "/api/gate/reservoir_4",
    }, "a parameterless command is not dispatched from the page"

    # And the gate routes come from the authoritative cascade inventory.
    gates = source_of("render_command_proxy")
    assert "cascade" in gates and "/api/gate/{key}" in gates, (
        "gate commands must be driven by the authoritative cascade inventory"
    )


def _route_table() -> dict:
    """POST route paths, from the app's own OpenAPI schema (version-agnostic)."""
    return {
        path: sorted(method.upper() for method in operations)
        for path, operations in app.openapi()["paths"].items()
        if any(method.lower() == "post" for method in operations)
    }


def _matches(template: str, concrete: str) -> bool:
    pattern = "^" + re.sub(r"\{[^}]+\}", r"[^/]+", template) + "$"
    return re.match(pattern, concrete) is not None


def test_every_proxied_route_exists_in_the_backend():
    table = _route_table()
    for endpoint in EXPECTED_PROXY_ENDPOINTS:
        assert any(_matches(t, endpoint) for t in table), (
            f"the page calls {endpoint}, which the backend does not expose"
        )


def test_proxied_routes_are_exactly_the_backend_command_surface():
    """Completeness both ways: no dead proxy call, no unreachable command."""
    table = _route_table()
    for template in table:
        assert any(_matches(template, e) for e in EXPECTED_PROXY_ENDPOINTS), (
            f"backend exposes POST {template} that the page cannot reach"
        )
    # And the backend exposes no state-writing route at all.
    assert client.post("/api/state", json={"storage": 123.0}).status_code == 405
    assert client.put("/api/state", json={"storage": 123.0}).status_code == 405


@pytest.mark.parametrize("endpoint,payload", [
    ("/api/simulation/pause", None),
    ("/api/simulation/play", None),
    ("/api/simulation/speed", {"speed": 2.5}),
    ("/api/storm", {"value": 0.4}),
    ("/api/controller/mode", {"mode": "MANUAL"}),
    ("/api/gate/reservoir_1", {"value": 33.0}),
])
def test_streamlit_commands_reach_the_authoritative_simulation(endpoint, payload):
    """
    The exact routes and payload shapes the page sends must move the ONE
    authoritative simulation object.
    """
    response = client.post(endpoint, json=payload) if payload is not None \
        else client.post(endpoint)
    assert response.status_code == 200, response.text

    sim = state_manager.sim_state
    if endpoint == "/api/simulation/speed":
        assert sim.sim_speed == pytest.approx(payload["speed"])
    elif endpoint == "/api/storm":
        assert sim.storm_intensity == pytest.approx(payload["value"])
    elif endpoint == "/api/controller/mode":
        assert sim.mode == payload["mode"]
    elif endpoint.startswith("/api/gate/"):
        assert sim.manual_gates["Virtual Reservoir A"] == pytest.approx(payload["value"])
    client.post("/api/simulation/pause")


def test_all_four_reservoirs_are_commandable_including_idukki():
    """Requirement 11 — Idukki (Reservoir D) must be an ordinary command."""
    client.post("/api/simulation/pause")
    for index, node in enumerate(ALL_RESERVOIRS, start=1):
        response = client.post(GATE_PATH.format(index), json={"value": 10.0 * index})
        assert response.status_code == 200, response.text

    assert state_manager.sim_state.manual_gates["Virtual Reservoir D"] == pytest.approx(40.0)
    assert state_manager.sim_state.manual_gates["Virtual Reservoir A"] == pytest.approx(10.0)

    # The page's own gate controls cover all four.
    gates = source_of("render_command_proxy")
    assert "cmd_gate_{key}" in gates
    assert EXPECTED_PROXY_ENDPOINTS >= {GATE_PATH.format(i) for i in range(1, 5)}


def test_backend_validation_remains_authoritative_for_proxy_payloads():
    """
    Requirement 6 / Phase 6 — the page must not weaken or duplicate validation.
    The backend still rejects what it always rejected.
    """
    client.post("/api/simulation/pause")
    assert client.post("/api/gate/reservoir_1", json={"value": "wide open"}).status_code == 422
    assert client.post("/api/gate/reservoir_5", json={"value": 10.0}).status_code == 400
    assert client.post("/api/controller/mode", json={"mode": "HACK"}).status_code == 422
    assert client.post("/api/storm", json={"value": None}).status_code == 422
    assert client.post("/api/simulation/speed", json={"speed": "fast"}).status_code == 422

    # Non-finite values are still refused on the wire.
    status = client.post(
        "/api/gate/reservoir_1", content='{"value": NaN}',
        headers={"Content-Type": "application/json"},
    ).status_code
    assert status == 422


def test_frontends_cannot_bypass_the_control_layers():
    """
    Requirements 17 / 18 — neither frontend imports the MPC, the SafetyLayer or
    the capacity guard, and the API cannot skip them either: ``routes.py``
    touches only bounded command fields on the authoritative object.
    """
    for path in FRONTEND_FILES:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for token in CONTROL_LAYER_TOKENS:
            assert token not in text, f"{path.name} reaches the control layer via {token!r}"

    routes_source = routes_module.__file__
    routes_text = Path(routes_source).read_text(encoding="utf-8")
    for token in ("mpc_controller", "downstream_capacity_guard", "live_mpc_orchestrator",
                  "import safety"):
        assert token not in routes_text, f"routes.py can bypass the control chain via {token}"

    # The guard/safety layers are applied by the authoritative controller chain.
    orchestrator = (_PROJECT_ROOT / "src" / "controller" / "live_mpc_orchestrator.py")
    orchestrator_text = orchestrator.read_text(encoding="utf-8")
    assert "SafetyLayer" in orchestrator_text
    assert "safety" in orchestrator_text.lower()


def test_only_the_authoritative_backend_owns_the_live_simulation():
    """Requirement 20 — one owner; both frontends are consumers."""
    bridge_owners = []
    state_owners = []
    for path in (_PROJECT_ROOT / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        rel = str(path.relative_to(_PROJECT_ROOT)).replace("\\", "/")
        if "SimBridge(" in text:
            bridge_owners.append(rel)
        if "GlobalSimulationState()" in text:
            state_owners.append(rel)

    assert bridge_owners == ["src/dashboard/api/state_manager.py"], bridge_owners
    assert state_owners == ["src/dashboard/api/state_manager.py"], state_owners
    assert state_manager.authoritative_instance_count() == 1

    # Both frontends read the same authoritative state.
    assert "/api/state" in app_source()
    assert "/ws/state" in (WEB_DIR / "api.js").read_text(encoding="utf-8")


# ===========================================================================
# 3. The page displays the backend's verdicts (and nothing it invents)
# ===========================================================================

@pytest.mark.parametrize("block", [
    "state_identity", "simulation", "downstream", "storm", "forecast_summary",
    "forecast_provenance", "mass_balance", "control", "hardware_status",
    "cascade", "reservoirs",
])
def test_streamlit_displays_the_backend_blocks(block):
    assert f'"{block}"' in app_source(), f"the page does not read the {block} block"


@pytest.mark.parametrize("key", [
    "status", "residual", "reservoirs_checked",       # mass balance
    "live_forecasts_are_validated",                    # forecast provenance
    "safety_layer_status", "downstream_status",        # control provenance
    "controller_type", "controller_status",
    "capacity_m3_s", "utilisation",                    # downstream capacity
    "state_id", "network_timestep", "sim_step_index",  # state identity
])
def test_streamlit_displays_the_required_verdicts(key):
    assert key in app_source(), f"the page does not display {key}"


def test_four_reservoir_inventory_is_payload_driven_not_hardcoded():
    """
    The cards must iterate the authoritative ``cascade`` inventory. A hardcoded
    reservoir list in the page would be a second, competing inventory.
    """
    renderer = source_of("render_authoritative_reservoirs")
    assert "cascade" in renderer
    assert "for entry in cascade" in renderer, (
        "the reservoir inventory must be iterated from the authoritative cascade block"
    )
    assert "for column, key in zip(columns, order)" in renderer, (
        "the render loop must follow the payload's own cascade order"
    )
    for hardcoded in ("Virtual Reservoir A", "Anayirankal", "Idukki", "reservoir_1"):
        assert hardcoded not in renderer, f"hardcoded reservoir name/key: {hardcoded}"
    assert "repository_name" in renderer or "node_id" in renderer


def test_authoritative_sections_render_only_when_the_backend_answered():
    """With the backend offline the page must show a notice, never values."""
    tree = app_tree()
    guarded = False
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and isinstance(node.test, ast.UnaryOp) \
                and isinstance(node.test.op, ast.Not) \
                and isinstance(node.test.operand, ast.Name) \
                and node.test.operand.id == "auth_error":
            if any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                   and c.func.id == "render_authoritative_reservoirs"
                   for c in ast.walk(node)):
                guarded = True
    assert guarded, "the authoritative section is not guarded by `not auth_error`"


def test_formatting_helpers_never_invent_a_value():
    flow_text = load_pure_function("flow_text")
    residual_text = load_pure_function("residual_text")

    for missing in (None, "n/a", [], {}, float("nan"), float("inf")):
        assert flow_text(missing) == "--", f"flow_text invented a value for {missing!r}"
        assert residual_text(missing) == "--", f"residual_text invented a value for {missing!r}"

    assert flow_text(578.7037037) == "578.7"
    assert residual_text(2.842170943040401e-14).startswith("2.84")


# ===========================================================================
# 4. The dormant research page stays dormant
# ===========================================================================

def test_research_simulation_page_is_not_wired_into_streamlit():
    """
    ``simulation_page.py`` contains a Phase 14.4 offline research engine. Stage 4
    retained it and Stage 13 must keep it UNREACHABLE: it must not be imported
    by the entrypoint, and Streamlit's multipage auto-discovery (a ``pages/``
    directory) must not exist.
    """
    assert "simulation_page" not in app_source()
    assert "render_simulation_page" not in app_source()

    callers = []
    for path in (_PROJECT_ROOT / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "render_simulation_page" in text or "import simulation_page" in text:
            callers.append(str(path.relative_to(_PROJECT_ROOT)).replace("\\", "/"))
    assert callers == ["src/dashboard/simulation_page.py"], (
        f"the research page is wired somewhere: {callers}"
    )

    assert not (DASHBOARD_DIR / "pages").exists(), (
        "a pages/ directory would let Streamlit auto-discover extra pages"
    )
