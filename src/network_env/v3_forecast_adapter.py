"""
Phase 15.2 — Frozen LSTM V3 Forecast Adapter

Loads EXISTING, FROZEN V3 prediction artifacts and exposes them
to the network environment as structured forecast metadata.

CRITICAL SEMANTICS:
  target_1d = point forecast of inflow on day +1
  target_3d = point forecast of inflow on day +3
  target_7d = point forecast of inflow on day +7

  These are NOT cumulative volumes.
  These are NOT daily averages over a horizon.
  This adapter NEVER performs forecast × horizon.
  This adapter NEVER interpolates missing intermediate days.
  This adapter NEVER fabricates forecasts.

V3 ARTIFACT:
  results/lstm_pytorch_v3_logtarget/test_predictions_original_units.csv

  Columns:
    date, reservoir,
    target_1d_actual, target_1d_prediction,
    target_3d_actual, target_3d_prediction,
    target_7d_actual, target_7d_prediction

  Units: MCM/day (original physical units, already inverse-transformed)
  Reservoirs: 16 real Kerala reservoir names
  Dates: 2025-01-01 to 2025-08-22 (test set, 74 unique dates)

RESERVOIR MAPPING:
  Network node → V3 reservoir name (from topology_config.yaml repository_reference)
  This mapping is classified as ASSUMED_FOR_PROTOTYPE since the
  network topology itself is hypothetical.
"""

import pandas as pd
import numpy as np
import math
from pathlib import Path
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field
from enum import Enum

from .provenance import Provenance, ProvenanceLevel


# ---------------------------------------------------------------------------
# Forecast status
# ---------------------------------------------------------------------------

class ForecastStatus(Enum):
    AVAILABLE = "FORECAST_AVAILABLE"
    UNAVAILABLE = "FORECAST_UNAVAILABLE"
    INVALID = "FORECAST_INVALID"


# ---------------------------------------------------------------------------
# Single-reservoir forecast record
# ---------------------------------------------------------------------------

@dataclass
class ReservoirForecast:
    """
    Forecast metadata for a single reservoir at a single forecast date.

    All values are POINT FORECASTS for the specific future day, not
    cumulative volumes or daily averages.
    """
    reservoir_id: str               # network node ID (e.g. "Reservoir_A")
    v3_reservoir_name: str          # V3 artifact name (e.g. "Anayirankal")
    forecast_date: str              # date the forecast was issued (YYYY-MM-DD)

    # Point forecasts (MCM/day) — PREDICTIONS, not observed
    target_1d: Optional[float] = None    # predicted inflow on forecast_date + 1 day
    target_3d: Optional[float] = None    # predicted inflow on forecast_date + 3 days
    target_7d: Optional[float] = None    # predicted inflow on forecast_date + 7 days

    # Corresponding actuals (MCM/day) — OBSERVED (from the same artifact)
    actual_1d: Optional[float] = None
    actual_3d: Optional[float] = None
    actual_7d: Optional[float] = None

    # Status for each horizon
    status_1d: ForecastStatus = ForecastStatus.UNAVAILABLE
    status_3d: ForecastStatus = ForecastStatus.UNAVAILABLE
    status_7d: ForecastStatus = ForecastStatus.UNAVAILABLE

    # Provenance
    model: str = "LSTM_V3"
    provenance: str = "MODEL_PREDICTION"
    artifact_path: str = ""

    def is_available(self, horizon: str) -> bool:
        """Check if a specific horizon forecast is available and valid."""
        status = getattr(self, f"status_{horizon}", ForecastStatus.UNAVAILABLE)
        return status == ForecastStatus.AVAILABLE

    def get_prediction(self, horizon: str) -> Optional[float]:
        """Get prediction for a horizon, returning None if unavailable."""
        if not self.is_available(horizon):
            return None
        return getattr(self, f"target_{horizon}", None)


# ---------------------------------------------------------------------------
# Network-wide forecast snapshot
# ---------------------------------------------------------------------------

@dataclass
class NetworkForecastSnapshot:
    """
    Complete forecast snapshot for all reservoirs at a given date.
    """
    forecast_date: str
    forecasts: Dict[str, ReservoirForecast] = field(default_factory=dict)

    def get(self, reservoir_id: str) -> Optional[ReservoirForecast]:
        return self.forecasts.get(reservoir_id)

    def all_available(self, horizon: str) -> bool:
        """True if all reservoirs have the specified horizon forecast."""
        return all(f.is_available(horizon) for f in self.forecasts.values())

    def available_count(self, horizon: str) -> int:
        return sum(1 for f in self.forecasts.values() if f.is_available(horizon))


# ---------------------------------------------------------------------------
# V3 Forecast Adapter
# ---------------------------------------------------------------------------

class V3ForecastAdapter:
    """
    Adapter that loads the FROZEN V3 prediction artifact and provides
    structured forecast metadata to the network environment.

    IMPORTANT SAFETY PROPERTIES:
    1. The V3 artifact file is opened READ-ONLY.
    2. Predictions are never modified, scaled, or interpolated.
    3. Missing data is explicitly reported as FORECAST_UNAVAILABLE.
    4. NaN/Inf values are reported as FORECAST_INVALID.
    5. The source artifact path is recorded for provenance.
    6. This adapter NEVER writes to the V3 artifact directory.
    """

    # Default V3 artifact path (relative to project root)
    DEFAULT_ARTIFACT = "results/lstm_pytorch_v3_logtarget/test_predictions_original_units.csv"

    def __init__(self, project_root: str,
                 reservoir_mapping: Optional[Dict[str, str]] = None,
                 artifact_path: Optional[str] = None):
        """
        Parameters
        ----------
        project_root : str
            Absolute path to the project root directory.
        reservoir_mapping : dict, optional
            Maps network node IDs to V3 reservoir names.
            e.g. {"Reservoir_A": "Anayirankal", ...}
            If None, reads from topology_config.yaml.
        artifact_path : str, optional
            Override path to V3 predictions CSV (relative to project_root).
        """
        self._project_root = Path(project_root)
        self._artifact_rel = artifact_path or self.DEFAULT_ARTIFACT
        self._artifact_path = self._project_root / self._artifact_rel

        # Load reservoir mapping
        if reservoir_mapping:
            self._mapping = dict(reservoir_mapping)
            self._mapping_provenance = "USER_PROVIDED"
        else:
            self._mapping = self._load_mapping_from_config()
            self._mapping_provenance = "ASSUMED_FOR_PROTOTYPE (from topology_config.yaml)"

        # Reverse mapping: V3 name → network node ID
        self._reverse_mapping = {v: k for k, v in self._mapping.items()}

        # Load and validate V3 predictions (READ-ONLY)
        self._df = self._load_artifact()

        # Build lookup index: (date, v3_reservoir_name) → row
        self._index: Dict[tuple, pd.Series] = {}
        for _, row in self._df.iterrows():
            key = (str(row["date"]), str(row["reservoir"]))
            self._index[key] = row

    def _load_mapping_from_config(self) -> Dict[str, str]:
        """Load the network-node → V3-reservoir mapping from topology_config.yaml."""
        import yaml
        config_path = self._project_root / "src" / "network_env" / "topology_config.yaml"
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)

        mapping = {}
        for res in cfg["reservoirs"]:
            node_id = res["id"]
            ref = res.get("repository_reference")
            if ref:
                mapping[node_id] = ref
        return mapping

    def _load_artifact(self) -> pd.DataFrame:
        """Load V3 predictions READ-ONLY with schema validation."""
        if not self._artifact_path.exists():
            raise FileNotFoundError(
                f"V3 prediction artifact not found: {self._artifact_path}"
            )

        df = pd.read_csv(self._artifact_path)

        # Schema validation
        required_cols = [
            "date", "reservoir",
            "target_1d_actual", "target_1d_prediction",
            "target_3d_actual", "target_3d_prediction",
            "target_7d_actual", "target_7d_prediction",
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"V3 artifact schema mismatch. Missing columns: {missing}"
            )

        # Ensure date is string for consistent lookup
        df["date"] = df["date"].astype(str)

        return df

    def _validate_value(self, val) -> ForecastStatus:
        """Check if a prediction value is valid."""
        if val is None:
            return ForecastStatus.UNAVAILABLE
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return ForecastStatus.INVALID
        return ForecastStatus.AVAILABLE

    # ---- public API ----

    @property
    def artifact_path(self) -> str:
        return str(self._artifact_path)

    @property
    def reservoir_mapping(self) -> Dict[str, str]:
        return dict(self._mapping)

    @property
    def available_dates(self) -> List[str]:
        return sorted(self._df["date"].unique().tolist())

    @property
    def available_reservoirs(self) -> List[str]:
        return sorted(self._df["reservoir"].unique().tolist())

    def get_forecast(self, reservoir_id: str, date: str) -> ReservoirForecast:
        """
        Get V3 forecast for a specific network node on a specific date.

        Parameters
        ----------
        reservoir_id : str
            Network node ID (e.g. "Reservoir_A")
        date : str
            Forecast date in YYYY-MM-DD format.

        Returns
        -------
        ReservoirForecast with appropriate status flags.
        """
        # Map network node to V3 reservoir name
        v3_name = self._mapping.get(reservoir_id)
        if v3_name is None:
            return ReservoirForecast(
                reservoir_id=reservoir_id,
                v3_reservoir_name="UNKNOWN",
                forecast_date=date,
                artifact_path=str(self._artifact_path),
            )

        # Lookup in index
        key = (date, v3_name)
        row = self._index.get(key)

        if row is None:
            return ReservoirForecast(
                reservoir_id=reservoir_id,
                v3_reservoir_name=v3_name,
                forecast_date=date,
                artifact_path=str(self._artifact_path),
            )

        # Extract predictions and actuals
        pred_1d = row["target_1d_prediction"]
        pred_3d = row["target_3d_prediction"]
        pred_7d = row["target_7d_prediction"]
        act_1d = row["target_1d_actual"]
        act_3d = row["target_3d_actual"]
        act_7d = row["target_7d_actual"]

        return ReservoirForecast(
            reservoir_id=reservoir_id,
            v3_reservoir_name=v3_name,
            forecast_date=date,
            target_1d=pred_1d,
            target_3d=pred_3d,
            target_7d=pred_7d,
            actual_1d=act_1d,
            actual_3d=act_3d,
            actual_7d=act_7d,
            status_1d=self._validate_value(pred_1d),
            status_3d=self._validate_value(pred_3d),
            status_7d=self._validate_value(pred_7d),
            artifact_path=str(self._artifact_path),
        )

    def get_network_snapshot(self, date: str,
                             node_ids: Optional[List[str]] = None
                             ) -> NetworkForecastSnapshot:
        """
        Get forecasts for all (or specified) network nodes on a date.

        Parameters
        ----------
        date : str
            Forecast date in YYYY-MM-DD format.
        node_ids : list of str, optional
            Specific node IDs to query. Defaults to all mapped nodes.

        Returns
        -------
        NetworkForecastSnapshot
        """
        if node_ids is None:
            node_ids = list(self._mapping.keys())

        snapshot = NetworkForecastSnapshot(forecast_date=date)
        for nid in node_ids:
            snapshot.forecasts[nid] = self.get_forecast(nid, date)
        return snapshot

    def get_provenance_info(self) -> Dict[str, Any]:
        """Return provenance metadata for this adapter."""
        return {
            "model": "LSTM_V3",
            "artifact": str(self._artifact_path),
            "artifact_exists": self._artifact_path.exists(),
            "total_records": len(self._df),
            "unique_dates": len(self._df["date"].unique()),
            "unique_reservoirs": len(self._df["reservoir"].unique()),
            "reservoir_mapping": self._mapping,
            "mapping_provenance": self._mapping_provenance,
            "forecast_semantics": {
                "target_1d": "POINT FORECAST: predicted inflow (MCM/day) on forecast_date + 1 day",
                "target_3d": "POINT FORECAST: predicted inflow (MCM/day) on forecast_date + 3 days",
                "target_7d": "POINT FORECAST: predicted inflow (MCM/day) on forecast_date + 7 days",
            },
            "units": "MCM/day (original physical units)",
            "provenance_level": "MODEL_PREDICTION",
            "warnings": [
                "These are POINT FORECASTS, NOT cumulative volumes.",
                "Do NOT multiply by horizon length.",
                "Do NOT interpolate between horizons.",
                "Reservoir mapping is ASSUMED_FOR_PROTOTYPE.",
            ],
        }
