"""
Regression tests — Gate unit contract (percentage 0-100)
=========================================================

BUG FIXED (read-only audit finding):
  src/dashboard/web/api.js setGate() previously sent `value / 100.0`,
  while the backend contract is PERCENTAGE units:
    - routes.py  POST /api/gate/{id} stores cmd.value into manual_gates
    - environment.py VirtualReservoir.step clamps gate_command_pct to [0, 100]
  So a UI slider value of 75 was transmitted as 0.75 (~1% gate).

These tests pin the contract at every layer:

  1. API layer: 0 / 40 / 75 / 100 percent round-trip to manual_gates
  2. Invalid reservoir IDs still return 400
  3. Backend clamps gate commands to [0, 100] (environment.py)
  4. Simulator mass balance is unchanged by gate commands
  5. api.js source no longer divides the gate value by 100
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.dashboard.api.app import app  # noqa: E402
from src.dashboard.api.state_manager import sim_state  # noqa: E402
from src.simulator.environment import VirtualReservoir  # noqa: E402

client = TestClient(app)

API_JS_PATH = _PROJECT_ROOT / "src" / "dashboard" / "web" / "api.js"


# ── 1. Gate percentage round-trip (the fixed bug) ────────────────────────────

@pytest.mark.parametrize("gate_pct", [0.0, 40.0, 75.0, 100.0])
def test_gate_command_sends_percentage(gate_pct):
    """UI slider value 0-100 must arrive at the backend as the SAME percentage.

    This is the regression test for the api.js unit bug: the frontend
    previously divided by 100, so 75 became 0.75.
    """
    response = client.post("/api/gate/reservoir_1", json={"value": gate_pct})
    assert response.status_code == 200
    assert response.json()["gate"] == gate_pct
    # Backend receives the exact percentage, not a fraction of it
    assert sim_state.manual_gates["Virtual Reservoir A"] == gate_pct


def test_gate_command_75_not_fraction():
    """Explicit guard: 75% must NOT arrive as 0.75 (the old bug)."""
    client.post("/api/gate/reservoir_2", json={"value": 75.0})
    assert sim_state.manual_gates["Virtual Reservoir B"] == 75.0
    assert sim_state.manual_gates["Virtual Reservoir B"] != 0.75


def test_gate_command_all_three_reservoirs():
    """Each UI reservoir id maps to its virtual reservoir with exact percent."""
    mapping = {
        "reservoir_1": "Virtual Reservoir A",
        "reservoir_2": "Virtual Reservoir B",
        "reservoir_3": "Virtual Reservoir C",
    }
    for rid, vname in mapping.items():
        response = client.post(f"/api/gate/{rid}", json={"value": 40.0})
        assert response.status_code == 200
        assert sim_state.manual_gates[vname] == 40.0


# ── 2. Invalid reservoir IDs still rejected ─────────────────────────────────

@pytest.mark.parametrize("bad_id", ["invalid_res", "reservoir_5", "reservoir_99"])
def test_invalid_reservoir_id_returns_400(bad_id):
    """Unknown reservoir IDs must still return 400 and not touch state.

    NOTE (Stage 9): ``reservoir_4`` used to appear in this list. Reservoir D
    (Idukki) is now a first-class, commandable reservoir, so the guard is
    exercised with ids that are genuinely unknown instead. The intent — an
    unknown id is rejected and mutates nothing — is unchanged.
    """
    before = dict(sim_state.manual_gates)
    response = client.post(f"/api/gate/{bad_id}", json={"value": 50.0})
    assert response.status_code == 400
    assert "Invalid reservoir_id" in response.json()["detail"]
    # No state mutation on failure
    assert sim_state.manual_gates == before


# ── 3. Backend clamps gate values to [0, 100] ────────────────────────────────

def _make_reservoir():
    config = {
        "capacity_mcm": {"value": 100.0},
        "initial_storage_mcm": {"value": 50.0},
        "max_release_capacity_mcm_day": {"value": 50.0},
    }
    return VirtualReservoir("TestRes", config)


@pytest.mark.parametrize(
    "raw_gate, expected",
    [
        (-50.0, 0.0),     # negative clamped to 0
        (-0.1, 0.0),
        (0.0, 0.0),
        (40.0, 40.0),     # in-range passes through
        (75.0, 75.0),
        (100.0, 100.0),
        (100.1, 100.0),   # above-range clamped to 100
        (250.0, 100.0),
    ],
)
def test_environment_clamps_gate(raw_gate, expected):
    """environment.py must clamp any gate command into [0, 100] percent."""
    res = _make_reservoir()
    res.step(local_inflow_mcm_day=10.0, routed_inflow_mcm_day=0.0,
             gate_command_pct=raw_gate)
    # Effective gate position is derived from the clamped release
    assert res.state.gate_position_pct == pytest.approx(
        (expected / 100.0) * 100.0
    )
    # Release follows the clamped gate: gate% * max_release_capacity
    assert res.state.release_mcm_day == pytest.approx(
        (expected / 100.0) * 50.0
    )


def test_gate_fraction_would_under_release():
    """Sanity: the OLD buggy fraction (0.75 for 75%) would under-release.

    Documents why the bug mattered: 0.75% of 50 MCM/day = 0.375 vs the
    intended 75% = 37.5.
    """
    res = _make_reservoir()
    res.step(10.0, 0.0, gate_command_pct=75.0)   # correct: percent
    correct_release = res.state.release_mcm_day

    res2 = _make_reservoir()
    res2.step(10.0, 0.0, gate_command_pct=0.75)  # old bug: fraction
    buggy_release = res2.state.release_mcm_day

    assert correct_release == pytest.approx(37.5)
    assert buggy_release == pytest.approx(0.375)
    assert correct_release > buggy_release * 10


# ── 4. Simulator mass balance unchanged ──────────────────────────────────────

def test_mass_balance_with_gate_commands():
    """storage_new = storage_old + inflows - release, regardless of gate value.

    Verifies the gate fix did not alter simulator physics.
    """
    for gate in [0.0, 40.0, 75.0, 100.0]:
        res = _make_reservoir()
        storage_before = res.state.storage_mcm
        inflow, routed = 10.0, 5.0
        res.step(inflow, routed, gate_command_pct=gate)

        expected_release = (max(0.0, min(100.0, gate)) / 100.0) * 50.0
        expected_storage = storage_before + inflow + routed - expected_release
        assert res.state.storage_mcm == pytest.approx(expected_storage)


def test_mass_balance_overflow_spill_unchanged():
    """Overflow still forces spill at capacity — physics untouched."""
    res = _make_reservoir()
    # 100% gate, tiny inflow relative to storage near capacity
    res.state.storage_mcm = 99.0
    res.step(local_inflow_mcm_day=50.0, routed_inflow_mcm_day=0.0,
             gate_command_pct=0.0)  # closed gate → must spill
    assert res.state.storage_mcm == pytest.approx(100.0)  # capped at capacity
    assert res.state.overflow_events == 1
    # Spill + storage cap conserve water: release includes the spill
    assert res.state.release_mcm_day == pytest.approx(49.0)


def test_mass_balance_negative_storage_prevention_unchanged():
    """Release is still capped by available water — no negative storage."""
    res = _make_reservoir()
    res.state.storage_mcm = 5.0
    res.step(local_inflow_mcm_day=0.0, routed_inflow_mcm_day=0.0,
             gate_command_pct=100.0)  # try to drain more than available
    assert res.state.storage_mcm == 0.0
    assert res.state.release_mcm_day == pytest.approx(5.0)


# ── 5. api.js source contract (the actual fix) ──────────────────────────────

def test_api_js_does_not_divide_gate_by_100():
    """api.js setGate must send the slider value directly (percent units)."""
    src = API_JS_PATH.read_text(encoding="utf-8")
    # Locate the setGate method body
    assert "async setGate" in src, "setGate method must exist"
    start = src.index("async setGate")
    # Method body ends at the next 'async' or end of class
    end = src.find("async", start + 1)
    body = src[start:end if end != -1 else len(src)]
    assert "value / 100" not in body and "value/100" not in body, (
        "setGate must NOT divide by 100 — backend expects percentage 0-100"
    )
    assert "{ value: value }" in body, (
        "setGate must post the raw slider percentage"
    )


def test_api_js_storm_still_fraction():
    """setStorm intentionally keeps /100 — storm_intensity is a 0-1 fraction."""
    src = API_JS_PATH.read_text(encoding="utf-8")
    start = src.index("async setStorm")
    end = src.find("async", start + 1)
    body = src[start:end if end != -1 else len(src)]
    assert "value / 100.0" in body, (
        "setStorm must keep dividing by 100 — backend storm_intensity is 0-1"
    )
