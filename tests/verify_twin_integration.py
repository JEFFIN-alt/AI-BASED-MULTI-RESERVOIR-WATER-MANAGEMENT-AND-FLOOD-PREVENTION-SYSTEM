"""
Phase 13 — Comprehensive Validation Tests

tests/verify_twin_integration.py

Verifies:
1. Python state adapter produces valid JSON
2. All 3 reservoirs exist
3. Required fields exist
4. Values are numeric where expected
5. Hardware status remains NOT_CONNECTED
6. Digital Twin HTML exists
7. TestDriver is absent from production integration
8. updateState(state) is present in the HTML
9. No JavaScript fake simulation controls production state
"""
import sys
import json
import re
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

TWIN_HTML_PATH = _PROJECT_ROOT / "src" / "dashboard" / "twin_component" / "reservoir_twin.html"
STATE_ADAPTER_PATH = _PROJECT_ROOT / "src" / "dashboard" / "twin_component" / "state_adapter.py"
APP_PATH = _PROJECT_ROOT / "src" / "dashboard" / "app.py"


def test_1_adapter_produces_valid_json():
    """State adapter must produce valid JSON."""
    from src.dashboard.twin_component.state_adapter import adapt_state_for_twin

    mock_sim_state = {
        "downstream_flow": 42.5,
        "reservoirs": {
            "Virtual Reservoir A": {
                "storage_pct": 65.0, "inflow": 20.0, "routed_inflow": 5.0,
                "outflow": 15.0, "gate_position_pct": 40.0, "risk_status": "NORMAL"
            },
            "Virtual Reservoir B": {
                "storage_pct": 58.0, "inflow": 30.0, "routed_inflow": 10.0,
                "outflow": 25.0, "gate_position_pct": 35.0, "risk_status": "WATCH"
            },
            "Virtual Reservoir C": {
                "storage_pct": 80.0, "inflow": 45.0, "routed_inflow": 15.0,
                "outflow": 30.0, "gate_position_pct": 50.0, "risk_status": "ALERT"
            },
        }
    }

    twin = adapt_state_for_twin(mock_sim_state, "AI", 0.7)
    json_str = json.dumps(twin)
    parsed = json.loads(json_str)
    assert isinstance(parsed, dict), "Must produce valid JSON dict"
    print("[PASS] Test 1: State adapter produces valid JSON")


def test_2_all_3_reservoirs_exist():
    """All 3 reservoirs must exist in output."""
    from src.dashboard.twin_component.state_adapter import adapt_state_for_twin

    twin = adapt_state_for_twin({"reservoirs": {}, "downstream_flow": 0}, "MANUAL", 0.0)
    for i in range(1, 4):
        key = f"reservoir_{i}"
        assert key in twin["reservoirs"], f"Missing {key}"
    print("[PASS] Test 2: All 3 reservoirs exist")


def test_3_required_fields_exist():
    """Each reservoir must have all required fields."""
    from src.dashboard.twin_component.state_adapter import adapt_state_for_twin

    twin = adapt_state_for_twin({
        "reservoirs": {
            "Virtual Reservoir A": {"storage_pct": 50, "inflow": 10, "routed_inflow": 0,
                                     "outflow": 5, "gate_position_pct": 20, "risk_status": "NORMAL"},
            "Virtual Reservoir B": {"storage_pct": 50, "inflow": 10, "routed_inflow": 0,
                                     "outflow": 5, "gate_position_pct": 20, "risk_status": "NORMAL"},
            "Virtual Reservoir C": {"storage_pct": 50, "inflow": 10, "routed_inflow": 0,
                                     "outflow": 5, "gate_position_pct": 20, "risk_status": "NORMAL"},
        },
        "downstream_flow": 10
    }, "MANUAL", 0.0)

    required = ["water_level", "storage", "inflow", "release", "gate", "risk"]
    for i in range(1, 4):
        rk = f"reservoir_{i}"
        for field in required:
            assert field in twin["reservoirs"][rk], f"Missing {field} in {rk}"
    print("[PASS] Test 3: Required fields exist")


def test_4_values_are_numeric():
    """Numeric fields must be numeric."""
    from src.dashboard.twin_component.state_adapter import adapt_state_for_twin

    twin = adapt_state_for_twin({
        "reservoirs": {
            "Virtual Reservoir A": {"storage_pct": 65, "inflow": 20, "routed_inflow": 5,
                                     "outflow": 15, "gate_position_pct": 40, "risk_status": "NORMAL"},
            "Virtual Reservoir B": {"storage_pct": 58, "inflow": 30, "routed_inflow": 10,
                                     "outflow": 25, "gate_position_pct": 35, "risk_status": "NORMAL"},
            "Virtual Reservoir C": {"storage_pct": 80, "inflow": 45, "routed_inflow": 15,
                                     "outflow": 30, "gate_position_pct": 50, "risk_status": "NORMAL"},
        },
        "downstream_flow": 42.5
    }, "AI", 0.5)

    numeric_fields = ["water_level", "storage", "inflow", "release", "gate"]
    for i in range(1, 4):
        rk = f"reservoir_{i}"
        for field in numeric_fields:
            val = twin["reservoirs"][rk][field]
            assert isinstance(val, (int, float)), f"{rk}.{field} is {type(val)}, expected numeric"

    assert isinstance(twin["storm_intensity"], (int, float))
    assert isinstance(twin["downstream_flow"], (int, float))
    print("[PASS] Test 4: Values are numeric where expected")


def test_5_hardware_not_connected():
    """Hardware status must remain NOT_CONNECTED."""
    from src.dashboard.twin_component.state_adapter import adapt_state_for_twin

    twin = adapt_state_for_twin({"reservoirs": {}, "downstream_flow": 0}, "MANUAL", 0.0)
    hw = twin["hardware_status"]
    for key in ["esp32", "water_level_sensor", "flow_sensor", "gate_actuator"]:
        assert hw[key] == "NOT_CONNECTED", f"{key} should be NOT_CONNECTED, got {hw[key]}"
    print("[PASS] Test 5: Hardware status remains NOT_CONNECTED")


def test_6_twin_html_exists():
    """Digital Twin HTML must exist at the expected path."""
    assert TWIN_HTML_PATH.exists(), f"HTML not found at {TWIN_HTML_PATH}"
    content = TWIN_HTML_PATH.read_text(encoding="utf-8")
    assert len(content) > 1000, "HTML file seems too small"
    assert "Three.js" in content or "three" in content.lower(), "HTML must reference Three.js"
    print("[PASS] Test 6: Digital Twin HTML exists")


def test_7_test_driver_present_but_labeled():
    """
    TestDriver is kept temporarily in the HTML (as stated in the plan).
    It must be clearly labeled as synthetic / NOT YOUR SIM.
    When removed, this test should verify absence.
    """
    content = TWIN_HTML_PATH.read_text(encoding="utf-8")
    # TestDriver is still present (Phase 6 removal happens after live verification)
    if "class TestDriver" in content:
        assert "SYNTHETIC" in content.upper() or "NOT YOUR SIM" in content.upper(), \
            "TestDriver must be clearly labeled as synthetic"
        print("[PASS] Test 7: TestDriver present but clearly labeled as synthetic (pre-Phase 6)")
    else:
        print("[PASS] Test 7: TestDriver has been removed (post-Phase 6)")


def test_8_update_state_present():
    """updateState(state) must be present in the HTML."""
    content = TWIN_HTML_PATH.read_text(encoding="utf-8")
    assert "updateState" in content, "updateState function must exist in HTML"
    # Verify it's a method on the renderer class
    assert "updateState(state)" in content, "updateState(state) signature must exist"
    print("[PASS] Test 8: updateState(state) is present")


def test_9_no_js_fake_simulation_in_production():
    """
    STAGE 4 — SINGLE AUTHORITATIVE SIMULATION.

    ``src/dashboard/app.py`` (Streamlit) must be a READ-ONLY viewer:
      * it must NOT import the live bridge (``sim_bridge.SimBridge``),
      * it must NOT import the offline research engine (``simulator.engine``),
      * it must NOT contain JavaScript that fabricates reservoir telemetry.

    The state adapter (``adapt_state_for_twin``) moved to the AUTHORITATIVE
    side (``src/dashboard/api/state_manager.py``) — it is asserted there, since
    Streamlit now forwards the backend payload verbatim instead of adapting it.

    The import/usage checks are AST-based so that documentation may still
    *name* the components without creating a runtime dependency.
    """
    import ast

    app_content = APP_PATH.read_text(encoding="utf-8")
    tree = ast.parse(app_content, filename=str(APP_PATH))

    forbidden_modules = {"sim_bridge", "simulator.engine", "src.simulator.engine"}
    forbidden_names = {"SimBridge", "SimulationEngine", "VirtualCascade"}

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module not in forbidden_modules, (
                f"app.py must not import {node.module} — Streamlit owns no simulation"
            )
            assert not any(a.name in forbidden_names for a in node.names), \
                "app.py must not import a live simulation class"
        elif isinstance(node, ast.Import):
            assert not any(a.name in forbidden_modules for a in node.names), \
                "app.py must not import a live simulation module"
        elif isinstance(node, ast.Name):
            assert node.id not in forbidden_names, (
                f"app.py constructs/uses {node.id} — that would be a competing simulation"
            )
        elif isinstance(node, ast.Attribute):
            assert node.attr not in forbidden_names, \
                f"app.py references {node.attr} at runtime"

    # Still no fabricated telemetry in the production path.
    assert "Math.random" not in app_content, \
        "app.py must not contain JS Math.random for fake telemetry"
    assert "Math.sin" not in app_content, \
        "app.py must not contain JS Math.sin for fake telemetry"

    # The adapter now lives where the AUTHORITATIVE state is produced.
    state_manager_path = _PROJECT_ROOT / "src" / "dashboard" / "api" / "state_manager.py"
    state_manager_content = state_manager_path.read_text(encoding="utf-8")
    assert "adapt_state_for_twin" in state_manager_content, \
        "the authoritative backend must produce twin state through the state adapter"

    print("[PASS] Test 9: Streamlit is read-only; adapter lives in the authoritative backend")


if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 13 — VERIFY TWIN INTEGRATION")
    print("=" * 60)

    tests = [
        test_1_adapter_produces_valid_json,
        test_2_all_3_reservoirs_exist,
        test_3_required_fields_exist,
        test_4_values_are_numeric,
        test_5_hardware_not_connected,
        test_6_twin_html_exists,
        test_7_test_driver_present_but_labeled,
        test_8_update_state_present,
        test_9_no_js_fake_simulation_in_production,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"[ERROR] {test.__name__}: {e}")
            failed += 1

    print()
    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(tests)}")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)
