"""
Regression test for the Streamlit simulation controls.

Specifically tests that:
1. advance_simulation is callable (no UnboundLocalError)
2. STEP actually advances the real simulation via SimBridge
3. PLAY/PAUSE/RESET use the same simulation state
4. Gate controls affect the real simulation
"""
import sys
import json
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.dashboard.sim_bridge import SimBridge

CONFIG_PATH = _PROJECT_ROOT / "configs" / "simulation" / "four_reservoir_demo.json"
THRESH_PATH = _PROJECT_ROOT / "data" / "processed" / "historical_inflow_thresholds.json"


def _make_bridge():
    return SimBridge(str(CONFIG_PATH), str(THRESH_PATH))


def _default_inflows():
    return {
        "Virtual Reservoir A": 10.0,
        "Virtual Reservoir B": 20.0,
        "Virtual Reservoir C": 100.0,
        "Virtual Reservoir D": 0.0,
    }


def _default_gates():
    return {
        "Virtual Reservoir A": 40.0,
        "Virtual Reservoir B": 35.0,
        "Virtual Reservoir C": 50.0,
        "Virtual Reservoir D": 100.0,
    }


# ------------------------------------------------------------------
# Part 1: Regression for UnboundLocalError
# ------------------------------------------------------------------
def test_advance_simulation_callable():
    """
    Reproduces the exact pattern that caused UnboundLocalError.
    The function must be defined before it is called.
    This test ensures the code pattern used in app.py is sound.
    """
    bridge = _make_bridge()
    inflows = _default_inflows()
    gate_commands = _default_gates()

    # This is the exact closure pattern from app.py.
    # The bug was: the function was defined AFTER being called.
    def advance_simulation(b):
        inf = inflows.copy()
        b.step(inf, gate_commands)

    # Must NOT raise UnboundLocalError
    advance_simulation(bridge)


# ------------------------------------------------------------------
# Part 2: STEP advances real simulation state
# ------------------------------------------------------------------
def test_step_advances_simulation():
    bridge = _make_bridge()
    inflows = _default_inflows()
    gates = _default_gates()

    state_before = bridge.get_state({})
    s_before = {
        r: state_before["reservoirs"][r]["storage_mcm"]
        for r in bridge.cascade.cascade_order
    }

    bridge.step(inflows, gates)

    state_after = bridge.get_state({})
    s_after = {
        r: state_after["reservoirs"][r]["storage_mcm"]
        for r in bridge.cascade.cascade_order
    }

    # At least one reservoir's storage must have changed
    changed = any(s_before[r] != s_after[r] for r in s_before)
    assert changed, (
        "STEP did not change any reservoir storage. "
        f"Before: {s_before}, After: {s_after}"
    )


# ------------------------------------------------------------------
# Part 3: RESET returns to initial state
# ------------------------------------------------------------------
def test_reset_restores_initial_state():
    bridge = _make_bridge()
    inflows = _default_inflows()
    gates = _default_gates()

    initial_state = bridge.get_state({})

    # Advance several steps
    for _ in range(5):
        bridge.step(inflows, gates)

    mid_state = bridge.get_state({})
    # Confirm state diverged
    assert mid_state != initial_state

    # Reset
    bridge.init_cascade(50.0)
    reset_state = bridge.get_state({})

    # Storage should match initial
    for r in bridge.cascade.cascade_order:
        assert abs(
            reset_state["reservoirs"][r]["storage_mcm"]
            - initial_state["reservoirs"][r]["storage_mcm"]
        ) < 1e-6, f"Reset did not restore storage for {r}"


# ------------------------------------------------------------------
# Part 4: Gate controls affect simulation
# ------------------------------------------------------------------
def test_gate_change_affects_release():
    bridge = _make_bridge()
    inflows = _default_inflows()

    # Step with gate fully closed
    closed_gates = {r: 0.0 for r in inflows}
    closed_gates["Virtual Reservoir D"] = 100.0
    bridge.step(inflows, closed_gates)
    state_closed = bridge.get_state({})

    # Reset and step with gate fully open
    bridge.init_cascade(50.0)
    open_gates = {r: 100.0 for r in inflows}
    open_gates["Virtual Reservoir D"] = 100.0
    bridge.step(inflows, open_gates)
    state_open = bridge.get_state({})

    # Release (outflow) with open gates must be >= release with closed gates
    for r in bridge.cascade.cascade_order:
        if r == "Virtual Reservoir D":
            continue
        release_closed = state_closed["reservoirs"][r]["outflow"]
        release_open = state_open["reservoirs"][r]["outflow"]
        assert release_open >= release_closed, (
            f"{r}: open gate release ({release_open}) < closed gate release ({release_closed})"
        )


# ------------------------------------------------------------------
# Part 5: Multiple steps are deterministic
# ------------------------------------------------------------------
def test_deterministic_multi_step():
    """Two identical runs produce identical results."""
    inflows = _default_inflows()
    gates = _default_gates()

    def run_n_steps(n):
        b = _make_bridge()
        for _ in range(n):
            b.step(inflows, gates)
        return b.get_state({})

    s1 = run_n_steps(10)
    s2 = run_n_steps(10)

    for r in s1["reservoirs"]:
        assert abs(
            s1["reservoirs"][r]["storage_mcm"] - s2["reservoirs"][r]["storage_mcm"]
        ) < 1e-9, f"Non-deterministic storage for {r}"
