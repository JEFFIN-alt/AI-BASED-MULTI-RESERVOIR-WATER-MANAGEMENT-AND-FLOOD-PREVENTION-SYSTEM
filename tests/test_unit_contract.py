"""Stage 2 — Canonical unit / state contract regression tests.

These tests pin the contract defined in ``src/common/units.py``:

  * INTERNAL representation (MPCController, ReservoirNetwork):
        gate position = FRACTION in [0.0, 1.0]
  * EXTERNAL representation (HTTP API, operator UI, VirtualCascade):
        gate position = PERCENT in [0, 100]

They also assert that the conversion happens in exactly one place, so the
"percent value interpreted as a fraction" class of defect cannot silently
return.

Related existing suites (unchanged, and intentionally not duplicated here):
  * tests/test_gate_unit_regression.py  -- HTTP/UI percent round-trip
  * tests/test_controller.py            -- validated MPC / SafetyLayer behaviour
"""

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.common import units  # noqa: E402
from src.simulator.environment import VirtualReservoir  # noqa: E402


# ---------------------------------------------------------------------------
# Gate: fraction -> percent
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "fraction, expected_percent",
    [
        (0.0, 0.0),      # fully closed
        (0.05, 5.0),     # minimum environmental flow
        (0.25, 25.0),
        (0.5, 50.0),     # half open
        (0.75, 75.0),
        (1.0, 100.0),    # fully open
    ],
)
def test_gate_fraction_to_percent(fraction, expected_percent):
    assert units.gate_fraction_to_percent(fraction) == pytest.approx(expected_percent)


# ---------------------------------------------------------------------------
# Gate: percent -> fraction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "percent, expected_fraction",
    [
        (0.0, 0.0),
        (5.0, 0.05),
        (25.0, 0.25),
        (50.0, 0.5),
        (75.0, 0.75),
        (100.0, 1.0),
    ],
)
def test_gate_percent_to_fraction(percent, expected_fraction):
    assert units.gate_percent_to_fraction(percent) == pytest.approx(expected_fraction)


# ---------------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fraction", [0.0, 0.01, 0.15, 0.3, 0.5, 0.7, 1.0])
def test_roundtrip_fraction_percent_fraction(fraction):
    assert units.gate_percent_to_fraction(
        units.gate_fraction_to_percent(fraction)
    ) == pytest.approx(fraction)


@pytest.mark.parametrize("percent", [0.0, 1.0, 37.5, 75.0, 99.9, 100.0])
def test_roundtrip_percent_fraction_percent(percent):
    assert units.gate_fraction_to_percent(
        units.gate_percent_to_fraction(percent)
    ) == pytest.approx(percent)


# ---------------------------------------------------------------------------
# Clamping of finite out-of-range input
# ---------------------------------------------------------------------------

def test_gate_conversion_clamps_finite_out_of_range():
    assert units.gate_percent_to_fraction(150.0) == pytest.approx(1.0)
    assert units.gate_percent_to_fraction(-20.0) == pytest.approx(0.0)
    assert units.gate_fraction_to_percent(1.5) == pytest.approx(100.0)
    assert units.gate_fraction_to_percent(-0.5) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# NaN / Inf rejection (never silently fail open)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_gate_conversion_rejects_non_finite(bad):
    with pytest.raises(units.UnitContractError):
        units.gate_percent_to_fraction(bad)
    with pytest.raises(units.UnitContractError):
        units.gate_fraction_to_percent(bad)


@pytest.mark.parametrize("bad", [None, "50", [50.0]])
def test_gate_conversion_rejects_non_numeric(bad):
    with pytest.raises(units.UnitContractError):
        units.gate_percent_to_fraction(bad)


def test_strict_validators_reject_out_of_range():
    with pytest.raises(units.UnitContractError):
        units.validate_gate_fraction(1.01)
    with pytest.raises(units.UnitContractError):
        units.validate_gate_percent(101.0)
    assert units.validate_gate_fraction(0.5) == pytest.approx(0.5)
    assert units.validate_gate_percent(50.0) == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# Storage & flow conversions
# ---------------------------------------------------------------------------

def test_storage_conversions():
    assert units.storage_percent_to_fraction(0.0) == pytest.approx(0.0)
    assert units.storage_percent_to_fraction(50.0) == pytest.approx(0.5)
    assert units.storage_percent_to_fraction(100.0) == pytest.approx(1.0)
    assert units.storage_fraction_to_percent(0.5) == pytest.approx(50.0)
    assert units.storage_fraction_to_percent(1.0) == pytest.approx(100.0)
    # out-of-range finite input clamps
    assert units.storage_percent_to_fraction(120.0) == pytest.approx(1.0)


def test_flow_conversion_matches_definition():
    # 1 MCM/day == 1e6 m^3 / 86400 s
    assert units.mcm_per_day_to_cubic_metres_per_second(1.0) == pytest.approx(
        1_000_000.0 / 86_400.0
    )
    assert units.cubic_metres_per_second_to_mcm_per_day(1.0) == pytest.approx(
        86_400.0 / 1_000_000.0
    )


def test_flow_roundtrip_preserves_validated_total_release():
    """The Phase 15.3 MPC total release must survive the presentation boundary."""
    value = 2031.00
    m3s = units.mcm_per_day_to_cubic_metres_per_second(value)
    assert units.cubic_metres_per_second_to_mcm_per_day(m3s) == pytest.approx(value)


# ---------------------------------------------------------------------------
# State contract completeness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "field",
    [
        "storage",
        "capacity",
        "inflow_local",
        "inflow_routed",
        "controlled_release",
        "spill",
        "total_outflow",
        "gate_position",
        "gate_position_pct",
        "water_level",
        "rainfall",
        "forecast_1d",
        "forecast_3d",
        "forecast_7d",
    ],
)
def test_state_contract_declares_field_units(field):
    assert field in units.CANONICAL_STATE_UNITS
    assert units.CANONICAL_STATE_UNITS[field]


def test_state_contract_unit_assertions():
    """Guard the two representations that caused the original defect."""
    assert units.CANONICAL_STATE_UNITS["gate_position"] == "fraction [0.0, 1.0]"
    assert units.CANONICAL_STATE_UNITS["gate_position_pct"] == "percent [0, 100]"
    assert units.CANONICAL_STATE_UNITS["water_level"] == "metres"


# ---------------------------------------------------------------------------
# Simulator consumes PERCENT (external contract preserved)
# ---------------------------------------------------------------------------

def _make_reservoir():
    cfg = {
        "capacity_mcm": {"value": 100.0},
        "max_release_capacity_mcm_day": {"value": 50.0},
        "initial_storage_mcm": {"value": 50.0},
    }
    return VirtualReservoir("Test", cfg)


@pytest.mark.parametrize(
    "gate_pct, expected_release_mcm_day",
    [
        (0.0, 0.0),      # closed
        (50.0, 25.0),    # half of max release
        (100.0, 50.0),   # fully open
        (150.0, 50.0),   # clamped to fully open
        (-10.0, 0.0),    # clamped to closed
    ],
)
def test_simulator_gate_command_contract_is_percent(gate_pct, expected_release_mcm_day):
    res = _make_reservoir()
    res.step(local_inflow_mcm_day=0.0, routed_inflow_mcm_day=0.0,
             gate_command_pct=gate_pct)
    assert res.state.release_mcm_day == pytest.approx(expected_release_mcm_day)


def test_fraction_valued_command_under_releases_by_100x():
    """Guard against re-introducing the fraction/percent confusion.

    Passing 1.0 where a percent is expected means "1% open", NOT "fully open".
    This is the exact signature of the historical api.js bug.
    """
    res = _make_reservoir()
    res.step(0.0, 0.0, gate_command_pct=1.0)
    assert res.state.release_mcm_day == pytest.approx(0.5)  # 1% of 50, not 50


def test_reported_effective_gate_is_percent():
    res = _make_reservoir()
    res.step(0.0, 0.0, gate_command_pct=50.0)
    assert res.state.gate_position_pct == pytest.approx(50.0)


def test_water_level_proxy_is_storage_percentage():
    res = _make_reservoir()
    assert res.get_simulated_water_level_proxy() == pytest.approx(50.0)


def test_simulator_rejects_nan_gate():
    """A NaN command must raise, not silently open every gate."""
    res = _make_reservoir()
    with pytest.raises(units.UnitContractError):
        res.step(0.0, 0.0, gate_command_pct=float("nan"))


# ---------------------------------------------------------------------------
# Both representations exist and must stay on their own side of the boundary
# ---------------------------------------------------------------------------

def test_mpc_gate_levels_are_fractions():
    """The validated MPC emits FRACTIONS and must not be changed."""
    from src.controller.mpc_controller import MPCConfig

    levels = MPCConfig().gate_levels
    assert levels, "MPC candidate gate levels must not be empty"
    for level in levels:
        assert 0.0 <= level <= 1.0, f"MPC gate level is not a fraction: {level}"


def test_live_gate_store_is_percent():
    """The live command store holds PERCENT values (external contract)."""
    from src.dashboard.api.state_manager import sim_state

    assert sim_state.manual_gates, "live gate store must be populated"
    for name, value in sim_state.manual_gates.items():
        assert 0.0 <= value <= 100.0, f"{name} gate is not a percent: {value}"


def test_mpc_fraction_maps_onto_live_percent_contract():
    """Stage 7 will feed MPC output into the live percent contract.

    This documents the intended boundary mapping without wiring it yet.
    """
    for level in (0.0, 0.15, 0.3, 0.5, 0.7, 1.0):
        percent = units.gate_fraction_to_percent(level)
        assert 0.0 <= percent <= 100.0
        # and the live simulator accepts that percent directly
        res = _make_reservoir()
        res.step(0.0, 0.0, gate_command_pct=percent)
        assert res.state.release_mcm_day == pytest.approx(level * 50.0)


# ---------------------------------------------------------------------------
# Source-level guard: exactly one conversion site
# ---------------------------------------------------------------------------

def test_environment_uses_the_canonical_boundary():
    source = (
        _PROJECT_ROOT / "src" / "simulator" / "environment.py"
    ).read_text(encoding="utf-8")

    assert "units.gate_percent_to_fraction" in source
    assert "units.gate_fraction_to_percent" in source
    assert "units.storage_fraction_to_percent" in source
    # the old inline arithmetic must be gone
    assert "/ 100.0" not in source
    assert "gate_clamped" not in source


def test_state_adapter_uses_the_canonical_boundary():
    source = (
        _PROJECT_ROOT / "src" / "dashboard" / "twin_component" / "state_adapter.py"
    ).read_text(encoding="utf-8")

    assert "units.gate_percent_to_fraction" in source
    assert "units.storage_percent_to_fraction" in source
    assert "units.mcm_per_day_to_cubic_metres_per_second" in source
    assert "/ 100.0" not in source
