"""
Stage 5 — Frozen LSTM V3 feature contract and live forecast provenance.
=======================================================================

This module is the SINGLE SOURCE OF TRUTH for:

  1. the input contract of the frozen LSTM V3 model (feature names, ORDER,
     7-timestep sequence, units, horizons, output units), and
  2. the prose about what the LIVE Digital Twin can and cannot legitimately
     feed into that model.

It contains no model code and no simulation logic, and it **never** modifies,
rescales or re-fits anything. It only *describes* and *verifies*.

--------------------------------------------------------------------------------
THE FROZEN CONTRACT (authoritative — do not change without retraining)
--------------------------------------------------------------------------------
Model        : ``models/lstm_pytorch_v3_logtarget/best_model.pt``  (FROZEN)
Scaler       : ``data/processed/scaled/feature_scaler.pkl``        (FROZEN)
Target scaler: ``models/lstm_pytorch_v3_logtarget/log_target_scaler.pkl`` (FROZEN)

    Input tensor : (N, 7, 5)  — 7 daily timesteps x 5 dynamic features
    Feature order: inflow, water_level, live_storage, rainfall, total_outflow
    Scaler input : 51 columns = 35 dynamic (5 features x 7 days, in the order
                   above) + 8 static + 8 static-availability flags.
                   The static/availability columns are scaled and then
                   DISCARDED — the LSTM consumes only the 35 dynamic values.
    Target       : log1p(target) -> StandardScaler (train-only).
                   Inverse = log_target_scaler.inverse_transform -> expm1.
    Horizons     : target_1d, target_3d, target_7d (POINT forecasts).
    Output units : MCM/day.

--------------------------------------------------------------------------------
UNITS ARE NOT INTERCHANGEABLE
--------------------------------------------------------------------------------
Measured on the frozen TRAIN split (n = 16 662 windows):

    feature        unit         train min / median / max      scaler mean / scale
    inflow         MCM/day      0.00 / 13.30 / 2374.06         56.22 / 186.08
    water_level    METRES       41.90 / 758.30 / 1758.65     789.30 / 449.05
    live_storage   MCM          0.00 / 21.80 / 1845.36       149.15 / 300.35
    rainfall       mm           0.00 / 0.00 / 264.60           9.53 /  20.17
    total_outflow  MCM/day      0.00 / 12.50 / 2800.93        22.94 /  39.30

``water_level`` is a real reservoir STAGE in METRES (per-reservoir ranges match
published Kerala dam levels: Idamalayar ~132-166 m, Idukki ~703-732 m,
Ponmudi ~684-708 m, Anayirankal ~1188-1207 m).

It is **NOT** a storage percentage, and it is **NOT** derivable from storage
without an elevation-storage (rating) curve, which does not exist in this
repository. See ``UNAVAILABLE_IN_LIVE_SIMULATION`` below.

--------------------------------------------------------------------------------
WHAT THE AUTHORITATIVE ReservoirNetwork EXPOSES
--------------------------------------------------------------------------------
Per node (``src/network_env/reservoir_network.py::ReservoirState``):
``storage`` (MCM), ``inflow_local`` (MCM/day), ``inflow_routed`` (MCM/day),
``controlled_release`` (MCM/day), ``spill`` (MCM/day),
``total_outflow`` (MCM/day), ``gate_position`` (fraction), plus derived
``storage_fraction``.

It exposes NO water level in metres and NO rainfall.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# 1. FROZEN MODEL CONTRACT
# ---------------------------------------------------------------------------

HISTORY_DAYS: int = 7

#: Dynamic feature order — VERBATIM from the frozen training pipeline
#: (``src/modeling/prepare_lstm_sequences.py`` and
#: ``src/modeling/train_lstm_pytorch_v3_logtarget.py``).
DYNAMIC_FEATURES: List[str] = [
    "inflow",
    "water_level",
    "live_storage",
    "rainfall",
    "total_outflow",
]

#: Canonical units of the frozen features (see ``src/common/units.py``).
FEATURE_UNITS: Dict[str, str] = {
    "inflow": "MCM/day",
    "water_level": "metres",
    "live_storage": "MCM",
    "rainfall": "millimetres",
    "total_outflow": "MCM/day",
}

STATIC_FEATURES: List[str] = [
    "static_latitude",
    "static_longitude",
    "static_FRL",
    "static_MWL",
    "static_blue_level",
    "static_orange_level",
    "static_red_level",
    "static_rule_level",
]

STATIC_AVAILABILITY_FEATURES: List[str] = [
    "static_latitude_available",
    "static_longitude_available",
    "static_FRL_available",
    "static_MWL_available",
    "static_blue_level_available",
    "static_orange_level_available",
    "static_red_level_available",
    "static_rule_level_available",
]

#: Forecast horizons, in the order the frozen model emits them.
HORIZONS: List[str] = ["forecast_1d", "forecast_3d", "forecast_7d"]
TARGET_COLUMNS: List[str] = ["target_1d", "target_3d", "target_7d"]

#: Forecast output unit (point forecasts, NOT cumulative volumes).
FORECAST_UNITS: str = "MCM/day"

MODEL_VERSION: str = "LSTM_V3_LOGTARGET"

#: Ordered scaler columns: 35 dynamic, then 8 static, then 8 availability flags.
SCALER_DYNAMIC_COLUMNS: List[str] = [
    f"{feat}_day_{day}" for feat in DYNAMIC_FEATURES for day in range(1, HISTORY_DAYS + 1)
]
EXPECTED_SCALER_COLUMNS: List[str] = (
    SCALER_DYNAMIC_COLUMNS + STATIC_FEATURES + STATIC_AVAILABILITY_FEATURES
)


# ---------------------------------------------------------------------------
# 2. PROVENANCE VOCABULARY
# ---------------------------------------------------------------------------

class FeatureProvenance(str, Enum):
    """Where a feature VALUE came from. Never ambiguous, never implied."""
    #: A real observation (used only by the offline validated evaluation).
    MEASURED_HISTORICAL = "MEASURED_HISTORICAL"
    #: Produced by the authoritative ReservoirNetwork simulation.
    SIMULATED = "SIMULATED"
    #: A documented synthetic placeholder used for DEMONSTRATION only.
    #: Not a measurement, and not derived from any other physical quantity.
    SYNTHETIC_DEMO = "SYNTHETIC_DEMO"
    #: The physical quantity is not available to the live simulation.
    UNAVAILABLE = "UNAVAILABLE"


class ForecastStatus(str, Enum):
    """Which pipeline produced a forecast. Never conflate these."""
    #: historical held-out data -> frozen LSTM V3 -> reported metrics.
    VALIDATED = "VALIDATED"
    #: authoritative simulation state -> simulation/demo features -> frozen LSTM V3.
    DEMONSTRATION_ONLY = "DEMONSTRATION_ONLY"
    #: fewer than ``HISTORY_DAYS`` genuine simulation steps recorded yet.
    WARMUP = "WARMUP_INSUFFICIENT_HISTORY"
    #: a required feature could not be produced legitimately.
    UNAVAILABLE = "FORECAST_UNAVAILABLE"


#: Physical quantities the live simulation CANNOT legitimately produce.
UNAVAILABLE_IN_LIVE_SIMULATION: Dict[str, str] = {
    "water_level": (
        "ReservoirNetwork exposes storage (MCM) only. No elevation-storage "
        "(rating) curve exists in this repository, so metres CANNOT be derived "
        "without fabricating physics. A storage percentage is NOT a water level."
    ),
    "rainfall": (
        "The reservoir network has no rainfall state. Rainfall (mm) is an "
        "external meteorological input and is not modelled."
    ),
}

#: Features the live simulation CAN produce, and their source field.
LIVE_SIMULATED_SOURCES: Dict[str, str] = {
    "inflow": "ReservoirState.inflow_local  (MCM/day)",
    "live_storage": "ReservoirState.storage  (MCM)",
    "total_outflow": "ReservoirState.total_outflow  (MCM/day)",
}


# ---------------------------------------------------------------------------
# 3. SYNTHETIC DEMONSTRATION PLACEHOLDERS
# ---------------------------------------------------------------------------
# These exist ONLY so the demonstration twin can render a forecast without
# inventing physics. They are:
#   * documented statistics of the FROZEN TRAIN split (real historical data),
#   * constant for the session,
#   * labelled SYNTHETIC_DEMO in every payload,
#   * never presented as a live measurement,
#   * never produced by converting storage.
#
# They are NOT part of the validated evaluation and MUST NOT be used to claim
# real-world forecast skill.

DEMO_PLACEHOLDER_STATISTIC: str = "mean of *_day_1 over the frozen TRAIN split"
DEMO_PLACEHOLDER_SOURCE_FILE: str = "data/processed/splits/train.csv"

DEMO_PLACEHOLDERS: Dict[str, Dict[str, float]] = {
    "Anayirankal": {"water_level": 1201.8123, "rainfall": 6.4048},
    "Ponmudi": {"water_level": 701.0955, "rainfall": 7.8939},
    "Idamalayar": {"water_level": 153.3690, "rainfall": 10.7806},
    "Idukki": {"water_level": 720.5721, "rainfall": 9.2736},
}


# ---------------------------------------------------------------------------
# 4. FEATURE INPUT RECORD
# ---------------------------------------------------------------------------

@dataclass
class FeatureInput:
    """
    One value fed to the frozen model, with its unit and its provenance.

    ``is_simulated`` is True for anything that is not a real measurement.
    """
    name: str
    unit: str
    value: Optional[float]
    provenance: FeatureProvenance
    note: str = ""

    @property
    def is_simulated(self) -> bool:
        return self.provenance in (
            FeatureProvenance.SIMULATED,
            FeatureProvenance.SYNTHETIC_DEMO,
        )

    @property
    def is_available(self) -> bool:
        return self.value is not None and self.provenance is not FeatureProvenance.UNAVAILABLE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "unit": self.unit,
            "value": None if self.value is None else float(self.value),
            "provenance": self.provenance.value,
            "is_simulated": self.is_simulated,
            "is_available": self.is_available,
            "note": self.note,
        }


# ---------------------------------------------------------------------------
# 5. CONTRACT VERIFICATION
# ---------------------------------------------------------------------------

class ScalerContractError(RuntimeError):
    """Raised when a scaler does not match the frozen V3 feature contract."""


def verify_scaler_contract(scaler) -> Dict[str, Any]:
    """
    Assert a feature scaler matches the frozen V3 contract EXACTLY.

    Checks the column count (51) and the exact ordering of the 35 dynamic
    columns, since a reordering would silently corrupt every prediction while
    still producing plausible-looking numbers.

    Returns a small report dict. Raises ``ScalerContractError`` on mismatch.
    """
    names = getattr(scaler, "feature_names_in_", None)
    if names is None:
        raise ScalerContractError("Feature scaler has no feature_names_in_.")
    # NOTE: feature_names_in_ is a numpy array. Never use `array or default`
    # here — that raises "truth value of an array is ambiguous".
    cols = [str(c) for c in names]

    if len(cols) != len(EXPECTED_SCALER_COLUMNS):
        raise ScalerContractError(
            f"Frozen V3 scaler expects {len(EXPECTED_SCALER_COLUMNS)} columns, "
            f"found {len(cols)}."
        )

    if cols != EXPECTED_SCALER_COLUMNS:
        # Locate the first divergence for a useful message.
        for i, (got, want) in enumerate(zip(cols, EXPECTED_SCALER_COLUMNS)):
            if got != want:
                raise ScalerContractError(
                    f"Frozen V3 scaler column order mismatch at index {i}: "
                    f"expected {want!r}, found {got!r}."
                )
        raise ScalerContractError("Frozen V3 scaler column order mismatch.")

    for idx in range(3):
        if f"target_{['1d','3d','7d'][idx]}" != TARGET_COLUMNS[idx]:
            raise ScalerContractError("Target column ordering mismatch.")

    return {
        "n_columns": len(cols),
        "n_dynamic_columns": len(SCALER_DYNAMIC_COLUMNS),
        "dynamic_feature_order": list(DYNAMIC_FEATURES),
        "history_days": HISTORY_DAYS,
        "scaler_contract": "VERIFIED",
    }


def build_live_feature_inputs(
    *,
    inflow_local: Optional[float],
    storage: Optional[float],
    total_outflow: Optional[float],
    mapped_reservoir: Optional[str] = None,
    allow_synthetic_demo: bool = False,
) -> Dict[str, FeatureInput]:
    """
    Map authoritative ``ReservoirNetwork`` state onto the 5 frozen V3 features.

    Exactly three features are legitimately derivable and are marked
    ``SIMULATED``. ``water_level`` and ``rainfall`` are marked ``UNAVAILABLE``,
    or ``SYNTHETIC_DEMO`` when the caller explicitly opts into the demonstration
    placeholder path.

    NO storage -> water-level conversion is performed anywhere.
    """
    inputs: Dict[str, FeatureInput] = {}

    sim_values = {
        "inflow": inflow_local,
        "live_storage": storage,
        "total_outflow": total_outflow,
    }
    for name, value in sim_values.items():
        if value is None:
            inputs[name] = FeatureInput(
                name=name,
                unit=FEATURE_UNITS[name],
                value=None,
                provenance=FeatureProvenance.UNAVAILABLE,
                note=f"No authoritative simulation value for {name}.",
            )
        else:
            inputs[name] = FeatureInput(
                name=name,
                unit=FEATURE_UNITS[name],
                value=float(value),
                provenance=FeatureProvenance.SIMULATED,
                note=f"From {LIVE_SIMULATED_SOURCES[name]}.",
            )

    placeholder = (DEMO_PLACEHOLDERS.get(mapped_reservoir or "", {}) or {})
    for name in ("water_level", "rainfall"):
        if allow_synthetic_demo and name in placeholder:
            inputs[name] = FeatureInput(
                name=name,
                unit=FEATURE_UNITS[name],
                value=float(placeholder[name]),
                provenance=FeatureProvenance.SYNTHETIC_DEMO,
                note=(
                    f"DEMONSTRATION PLACEHOLDER — constant {DEMO_PLACEHOLDER_STATISTIC} "
                    f"of {DEMO_PLACEHOLDER_SOURCE_FILE} for {mapped_reservoir}. "
                    "NOT a live measurement and NOT derived from simulation state."
                ),
            )
        else:
            inputs[name] = FeatureInput(
                name=name,
                unit=FEATURE_UNITS[name],
                value=None,
                provenance=FeatureProvenance.UNAVAILABLE,
                note=UNAVAILABLE_IN_LIVE_SIMULATION[name],
            )

    # Return in the frozen feature order.
    return {name: inputs[name] for name in DYNAMIC_FEATURES}


def feature_provenance_summary(inputs: Dict[str, FeatureInput]) -> Dict[str, Any]:
    """Compact, machine-readable summary of a feature-input set."""
    return {
        "features": {name: rec.to_dict() for name, rec in inputs.items()},
        "feature_order": list(DYNAMIC_FEATURES),
        "unavailable": [
            name for name, rec in inputs.items()
            if rec.provenance is FeatureProvenance.UNAVAILABLE
        ],
        "synthetic_demo": [
            name for name, rec in inputs.items()
            if rec.provenance is FeatureProvenance.SYNTHETIC_DEMO
        ],
        "simulated": [
            name for name, rec in inputs.items()
            if rec.provenance is FeatureProvenance.SIMULATED
        ],
        "measured_historical": [
            name for name, rec in inputs.items()
            if rec.provenance is FeatureProvenance.MEASURED_HISTORICAL
        ],
        "all_real_measurements": all(
            rec.provenance is FeatureProvenance.MEASURED_HISTORICAL
            for rec in inputs.values()
        ),
    }
