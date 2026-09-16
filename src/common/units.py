"""AquaFlow — Canonical Units & State Contract (SINGLE SOURCE OF TRUTH)
=====================================================================

This module defines the canonical unit conventions for the whole project and
the ONE explicit conversion boundary between the *external* (operator-facing)
representation and the *internal* (control / physics) representation of a gate
position.

It contains no algorithm, no model and no simulation logic. It is deliberately
dependency-free apart from the standard library so that every layer -- the
validated controller stack, the live simulator, the FastAPI backend and the
tests -- can import it without creating a cycle.

--------------------------------------------------------------------------------
CANONICAL INTERNAL UNITS
--------------------------------------------------------------------------------
Used by the validated algorithmic stack (``MPCController`` and
``ReservoirNetwork``) and by any new code that joins it:

    Gate position : FRACTION   [0.0, 1.0]    0.0 = closed, 1.0 = fully open
    Flow          : MCM/day
    Storage       : MCM
    Water level   : metres
    Rainfall      : millimetres
    Timestep      : 1 day

--------------------------------------------------------------------------------
EXTERNAL / PRESENTATION UNITS
--------------------------------------------------------------------------------
Used by the operator HTTP API, the UI slider and the live ``VirtualCascade``
simulator:

    Gate position : PERCENT    [0, 100]

This split is not arbitrary: the percent contract is pinned at the API layer by
``tests/test_gate_unit_regression.py`` and must not change.

--------------------------------------------------------------------------------
THE ONE CONVERSION BOUNDARY  (read this before dividing by 100)
--------------------------------------------------------------------------------
Every percent <-> fraction gate conversion MUST go through this module:

    external PERCENT  --gate_percent_to_fraction()-->  internal FRACTION
    internal FRACTION --gate_fraction_to_percent()-->  external PERCENT

Do NOT write ``/ 100.0`` or ``* 100.0`` for gate values anywhere else. Scattering
that arithmetic is exactly how the previously documented defect arose, where a
UI value of 75 was transmitted as 0.75 and produced a ~1% gate opening.

Storage uses the analogous ``storage_percent_to_fraction`` /
``storage_fraction_to_percent`` pair, and flow uses
``mcm_per_day_to_cubic_metres_per_second`` /
``cubic_metres_per_second_to_mcm_per_day`` for the presentation-layer (m3/s)
rendering boundary.

--------------------------------------------------------------------------------
CONVERSION SEMANTICS
--------------------------------------------------------------------------------
* Finite values outside the contract range are CLAMPED into range. This matches
  the pre-existing ``VirtualCascade`` clamping behaviour and keeps the
  conversions total for well-formed numbers.
* Non-finite values (NaN, +Inf, -Inf) and non-numeric values RAISE
  ``UnitContractError``. They are never silently coerced.

  Rationale: ``max(0.0, min(100.0, nan))`` evaluates to ``100.0`` in CPython,
  so an unguarded NaN gate command fails OPEN (fully open) rather than closed.
  Raising here is strictly safer than silently opening every gate. Callers that
  need tolerant behaviour must handle the exception explicitly.
"""

from __future__ import annotations

import math
from typing import Union

__all__ = [
    "UnitContractError",
    "GATE_FRACTION_MIN",
    "GATE_FRACTION_MAX",
    "GATE_PERCENT_MIN",
    "GATE_PERCENT_MAX",
    "SECONDS_PER_DAY",
    "CUBIC_METRES_PER_MCM",
    "CANONICAL_STATE_UNITS",
    "validate_gate_fraction",
    "validate_gate_percent",
    "clamp_gate_fraction",
    "clamp_gate_percent",
    "gate_fraction_to_percent",
    "gate_percent_to_fraction",
    "storage_fraction_to_percent",
    "storage_percent_to_fraction",
    "mcm_per_day_to_cubic_metres_per_second",
    "cubic_metres_per_second_to_mcm_per_day",
]

Number = Union[int, float]


class UnitContractError(ValueError):
    """Raised when a value cannot be interpreted under the canonical contract."""


# ---------------------------------------------------------------------------
# Contract constants
# ---------------------------------------------------------------------------

#: Canonical internal gate range (fraction, dimensionless).
GATE_FRACTION_MIN: float = 0.0
GATE_FRACTION_MAX: float = 1.0

#: External gate range (percent).
GATE_PERCENT_MIN: float = 0.0
GATE_PERCENT_MAX: float = 100.0

#: Time base for the flow-unit boundary. The simulation timestep is one day.
SECONDS_PER_DAY: float = 86_400.0

#: 1 MCM = 1e6 cubic metres.
CUBIC_METRES_PER_MCM: float = 1_000_000.0


#: Canonical units for the fields of a reservoir state snapshot / forecast.
#: This is the machine-readable half of the state contract: any new field added
#: to a state or forecast payload must be declared here with its canonical unit.
CANONICAL_STATE_UNITS: dict = {
    # storage & geometry
    "storage": "MCM",
    "capacity": "MCM",
    "initial_storage": "MCM",
    # flows
    "inflow_local": "MCM/day",
    "inflow_routed": "MCM/day",
    "controlled_release": "MCM/day",
    "spill": "MCM/day",
    "total_outflow": "MCM/day",
    "downstream_capacity": "MCM/day",
    "max_release": "MCM/day",
    # control
    "gate_position": "fraction [0.0, 1.0]",
    "gate_position_pct": "percent [0, 100]",
    # hydrometeorology
    "water_level": "metres",
    "rainfall": "millimetres",
    # forecasts (point forecasts, NOT cumulative volumes)
    "forecast_1d": "MCM/day",
    "forecast_3d": "MCM/day",
    "forecast_7d": "MCM/day",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _as_finite_float(value: Number, *, label: str) -> float:
    """Coerce ``value`` to a finite float or raise ``UnitContractError``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UnitContractError(
            f"{label} must be a real number, got {type(value).__name__}: {value!r}"
        )
    result = float(value)
    if not math.isfinite(result):
        raise UnitContractError(
            f"{label} must be finite, got {value!r}. "
            "NaN/Inf gate values are rejected rather than clamped, because "
            "clamping a NaN would silently fail OPEN."
        )
    return result


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


# ---------------------------------------------------------------------------
# Validation (strict)
# ---------------------------------------------------------------------------

def validate_gate_fraction(value: Number) -> float:
    """Return ``value`` as a float if it is a valid internal gate FRACTION.

    Raises ``UnitContractError`` for non-numeric, non-finite, or out-of-range
    input. Use this when an out-of-range value indicates a caller bug rather
    than something to be quietly clamped.
    """
    result = _as_finite_float(value, label="gate fraction")
    if not (GATE_FRACTION_MIN <= result <= GATE_FRACTION_MAX):
        raise UnitContractError(
            f"gate fraction out of range [{GATE_FRACTION_MIN}, {GATE_FRACTION_MAX}]: "
            f"{result!r}"
        )
    return result


def validate_gate_percent(value: Number) -> float:
    """Return ``value`` as a float if it is a valid external gate PERCENT.

    Raises ``UnitContractError`` for non-numeric, non-finite, or out-of-range
    input.
    """
    result = _as_finite_float(value, label="gate percent")
    if not (GATE_PERCENT_MIN <= result <= GATE_PERCENT_MAX):
        raise UnitContractError(
            f"gate percent out of range [{GATE_PERCENT_MIN}, {GATE_PERCENT_MAX}]: "
            f"{result!r}"
        )
    return result


# ---------------------------------------------------------------------------
# Clamping (lenient, still finite-only)
# ---------------------------------------------------------------------------

def clamp_gate_fraction(value: Number) -> float:
    """Clamp a gate value to the internal FRACTION range ``[0.0, 1.0]``.

    Non-finite / non-numeric input still raises ``UnitContractError``.
    """
    return _clamp(
        _as_finite_float(value, label="gate fraction"),
        GATE_FRACTION_MIN,
        GATE_FRACTION_MAX,
    )


def clamp_gate_percent(value: Number) -> float:
    """Clamp a gate value to the external PERCENT range ``[0, 100]``.

    Non-finite / non-numeric input still raises ``UnitContractError``.
    """
    return _clamp(
        _as_finite_float(value, label="gate percent"),
        GATE_PERCENT_MIN,
        GATE_PERCENT_MAX,
    )


# ---------------------------------------------------------------------------
# THE CONVERSION BOUNDARY
# ---------------------------------------------------------------------------

def gate_fraction_to_percent(fraction: Number) -> float:
    """Internal FRACTION ``[0.0, 1.0]`` -> external PERCENT ``[0, 100]``.

    Out-of-range finite input is clamped; non-finite input raises.
    """
    return clamp_gate_fraction(fraction) * GATE_PERCENT_MAX


def gate_percent_to_fraction(percent: Number) -> float:
    """External PERCENT ``[0, 100]`` -> internal FRACTION ``[0.0, 1.0]``.

    Out-of-range finite input is clamped; non-finite input raises.
    """
    return clamp_gate_percent(percent) / GATE_PERCENT_MAX


def storage_fraction_to_percent(fraction: Number) -> float:
    """Storage as a fraction of capacity ``[0.0, 1.0]`` -> percent ``[0, 100]``.

    Used for the ``storage_pct`` / water-level *proxy* representation. The proxy
    is NOT a physical water level in metres -- see the note below.
    """
    return clamp_gate_fraction(fraction) * GATE_PERCENT_MAX


def storage_percent_to_fraction(percent: Number) -> float:
    """Storage percent ``[0, 100]`` -> fraction of capacity ``[0.0, 1.0]``."""
    return clamp_gate_percent(percent) / GATE_PERCENT_MAX


# ---------------------------------------------------------------------------
# Flow-unit boundary (presentation layer renders m3/s; the project uses MCM/day)
# ---------------------------------------------------------------------------

def mcm_per_day_to_cubic_metres_per_second(mcm_per_day: Number) -> float:
    """MCM/day -> m3/s. Presentation boundary used by the Digital Twin payload."""
    return float(mcm_per_day) * (CUBIC_METRES_PER_MCM / SECONDS_PER_DAY)


def cubic_metres_per_second_to_mcm_per_day(cubic_metres_per_second: Number) -> float:
    """m3/s -> MCM/day. Inverse of :func:`mcm_per_day_to_cubic_metres_per_second`."""
    return float(cubic_metres_per_second) * (SECONDS_PER_DAY / CUBIC_METRES_PER_MCM)


# ---------------------------------------------------------------------------
# Contract notes (documentation only -- intentionally not executable)
# ---------------------------------------------------------------------------
#
# WATER LEVEL IS NOT DERIVED FROM THE STORAGE PROXY.
#
# ``VirtualReservoir.get_simulated_water_level_proxy()`` returns a 0-100 storage
# *percentage*, not a water level in metres. The frozen LSTM V3 model was trained
# with ``water_level`` measured in metres (training mean ~789 m). The repository
# contains no elevation-storage curve, so a physically meaningful live water
# level cannot be produced from the simulator state. No conversion function for
# that transform exists here by design: inventing one would fabricate data. Live
# simulator-driven forecasts are therefore SIMULATION / DEMONSTRATION inputs and
# must not be presented as equivalent to the validated real-data V3 evaluation.
