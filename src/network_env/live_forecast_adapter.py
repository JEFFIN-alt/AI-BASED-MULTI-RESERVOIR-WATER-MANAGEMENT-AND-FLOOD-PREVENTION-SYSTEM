"""
Stage 6 — Live Forecast Adapter
===============================

Creates the clean boundary between the LIVE forecast representation and the
VALIDATED MPC representation.

    Frozen LSTM V3  (UNMODIFIED)
          │
          ▼
    live forecast payload               (Stage 5: per-reservoir dicts carrying
          │                              target values + provenance)
          ▼
    LiveForecastAdapter                 <-- THIS MODULE (new, self-contained)
          │
          ▼
    NetworkForecastSnapshot             <-- EXACTLY the Phase 15.2 contract the
          │                                  validated MPCController.decide()
          │                                  already consumes
          ▼
    (future Stage 7) MPC

WHAT THIS MODULE DOES NOT DO
----------------------------
* It does NOT modify the MPC, the SafetyLayer, the ReservoirNetwork, the frozen
  LSTM artifacts, or the Phase 15.2 adapter that defines the snapshot contract.
* It does NOT scale, smooth, interpolate, extrapolate, clamp or round any
  forecast value. Values pass through byte-identical (``float`` copy only).
* It does NOT fabricate a forecast. Anything unavailable is represented AS
  unavailable, with a machine-readable reason.

WHY A SEPARATE MODULE
---------------------
The live pipeline speaks "virtual reservoir names + provenance dictionaries";
the validated MPC speaks ``NetworkForecastSnapshot`` keyed by network node id.
Keeping the translation in one auditable place means the MPC contract can stay
frozen while the *source* of the forecast changes underneath it — from
simulation/demo inputs today to real hardware telemetry later (Stage 6 Req. 17).

HARDWARE READINESS
------------------
The adapter is source-agnostic. It consumes a *forecast payload* and asks only:
the declared status, the declared unit, the declared horizons and the three
point values. A future real-telemetry pipeline that emits the same payload with
``forecast_status = VALIDATED`` and ``MEASURED_HISTORICAL`` provenance flows
through this adapter with **no change to the downstream MPC contract** and no
change to the frozen model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .v3_forecast_adapter import (
    ForecastStatus,
    NetworkForecastSnapshot,
    ReservoirForecast,
)

# ---------------------------------------------------------------------------
# Frozen contract constants (mirror of src/modeling/v3_feature_contract.py)
# ---------------------------------------------------------------------------

#: Forecast output unit. Preserved end-to-end; never converted by this adapter.
FORECAST_UNIT = "MCM/day"

#: The three horizons the frozen model emits, in contract order.
HORIZON_KEYS: Tuple[str, ...] = ("forecast_1d", "forecast_3d", "forecast_7d")
HORIZON_LABELS: Tuple[str, ...] = ("1d", "3d", "7d")

#: Statuses that indicate the upstream pipeline deliberately withheld a forecast.
_WITHHELD_STATUSES = {
    "WARMUP_INSUFFICIENT_HISTORY",
    "FORECAST_UNAVAILABLE",
}

DEFAULT_TOPOLOGY_PATH = Path(__file__).resolve().parent / "topology_config.yaml"

#: Requirement 5 mapping, restated verbatim as a hard-coded expectation so a
#: silent re-mapping cannot pass unnoticed. Verified against topology_config.yaml.
REQUIRED_RESERVOIR_MAPPING: Dict[str, str] = {
    "Virtual Reservoir A": "Anayirankal",
    "Virtual Reservoir B": "Ponmudi",
    "Virtual Reservoir C": "Idamalayar",
    "Virtual Reservoir D": "Idukki",
}


# ---------------------------------------------------------------------------
# Result bundle
# ---------------------------------------------------------------------------

@dataclass
class LiveForecastBundle:
    """
    The adapter's full output.

    ``snapshot`` is the EXACT ``NetworkForecastSnapshot`` the validated MPC
    consumes. ``provenance`` carries the full structured provenance record that
    the narrow Phase 15.2 contract cannot express, keyed by network node id.
    """
    snapshot: NetworkForecastSnapshot
    provenance: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    forecast_date: str = ""
    status: str = "ADAPTED"
    notes: List[str] = field(default_factory=list)

    # -- convenience passthroughs so callers can use the bundle like the snapshot
    @property
    def forecasts(self) -> Dict[str, ReservoirForecast]:
        return self.snapshot.forecasts

    def get(self, node_id: str) -> Optional[ReservoirForecast]:
        return self.snapshot.get(node_id)

    def get_provenance(self, node_id: str) -> Optional[Dict[str, Any]]:
        return self.provenance.get(node_id)

    def provenance_summary(self) -> Dict[str, Any]:
        return {
            "forecast_date": self.forecast_date,
            "status": self.status,
            "unit": FORECAST_UNIT,
            "nodes": {
                nid: {
                    "live_reservoir": p.get("live_reservoir"),
                    "v3_reservoir_name": p.get("v3_reservoir_name"),
                    "forecast_status": p.get("declared_status"),
                    "forecast_provenance": p.get("declared_provenance"),
                    "is_simulated": p.get("is_simulated"),
                    "validated_metrics_apply": p.get("validated_metrics_apply"),
                    "horizons_available": [
                        h for h in HORIZON_LABELS if p.get(f"status_{h}") == ForecastStatus.AVAILABLE.value
                    ],
                    "issue": p.get("issue"),
                }
                for nid, p in self.provenance.items()
            },
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class LiveForecastAdapter:
    """
    Translate a live forecast payload into the validated MPC snapshot contract.

    Parameters
    ----------
    project_root : str | Path | None
        Used to locate ``topology_config.yaml`` when no explicit mapping is given.
    topology_path : str | Path | None
        Override the canonical topology configuration.
    reservoir_mapping : dict | None
        Optional explicit ``{live_reservoir_name: v3_reservoir_name}`` override.
    """

    def __init__(
        self,
        project_root: Optional[str] = None,
        topology_path: Optional[str] = None,
        reservoir_mapping: Optional[Dict[str, str]] = None,
        node_id_map: Optional[Dict[str, str]] = None,
    ):
        self._project_root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]

        if topology_path:
            self._topology_path = Path(topology_path)
        else:
            candidate = self._project_root / "src" / "network_env" / "topology_config.yaml"
            self._topology_path = candidate if candidate.exists() else DEFAULT_TOPOLOGY_PATH

        self._canonical = self._load_topology()
        self._mapping = self._build_mapping(reservoir_mapping, node_id_map)
        self._node_to_live = {v["node_id"]: k for k, v in self._mapping.items()}

    @classmethod
    def for_network(cls, network, **kwargs) -> "LiveForecastAdapter":
        """
        Build an adapter whose snapshot node ids match ``network.processing_order``.

        The live ``ReservoirNetwork`` built by ``LiveCascadeAdapter`` uses the
        live reservoir names as node ids, while the canonical validated network
        uses ``Reservoir_A``…``Reservoir_D``. The MPC looks forecasts up by the
        node id of the network it is asked to control, so the caller selects the
        convention here rather than re-keying the snapshot afterwards.
        """
        node_ids = [str(n) for n in network.processing_order]
        return cls(node_id_map={nid: nid for nid in node_ids}, **kwargs)

    # ---- construction helpers ----

    def _load_topology(self) -> Dict[str, Any]:
        with open(self._topology_path, "r") as fh:
            return yaml.safe_load(fh)

    def _build_mapping(self, override: Optional[Dict[str, str]],
                       node_id_map: Optional[Dict[str, str]] = None,
                       ) -> Dict[str, Dict[str, str]]:
        """
        Build ``{live_name: {"node_id": ..., "v3_name": ...}}``.

        Node ids come from the canonical topology (``Reservoir_A`` … ``Reservoir_D``)
        unless ``node_id_map`` overrides them for a specific target network.
        The V3 reservoir name comes from ``repository_reference`` unless an
        explicit override is supplied. Mapping failure RAISES rather than guessing.
        """
        node_id_map = node_id_map or {}
        by_display: Dict[str, str] = {
            r["display_name"]: r["id"]
            for r in self._canonical["reservoirs"]
            if r.get("display_name")
        }
        by_node_v3: Dict[str, str] = {
            r["id"]: r.get("repository_reference", "")
            for r in self._canonical["reservoirs"]
        }

        mapping: Dict[str, Dict[str, str]] = {}
        for live_name, canonical_id in by_display.items():
            v3_name = (override or {}).get(live_name, by_node_v3.get(canonical_id, ""))
            if not v3_name:
                raise ValueError(
                    f"LiveForecastAdapter cannot resolve the V3 reservoir name for "
                    f"{live_name!r} ({canonical_id}). Provide an explicit mapping."
                )
            node_id = node_id_map.get(live_name, canonical_id)
            mapping[live_name] = {"node_id": node_id, "v3_name": v3_name}

        if not mapping:
            raise ValueError(
                f"No live reservoirs could be mapped from {self._topology_path}."
            )

        # When targeting a specific network, every live reservoir must exist in it.
        if node_id_map:
            missing = sorted(set(node_id_map) - set(by_display))
            if missing:
                raise ValueError(
                    f"node_id_map references reservoirs absent from the canonical "
                    f"topology: {missing}"
                )

        # Verify the Stage 6 requirement-5 mapping explicitly.
        for live_name, expected_v3 in REQUIRED_RESERVOIR_MAPPING.items():
            if live_name in mapping and mapping[live_name]["v3_name"] != expected_v3:
                raise ValueError(
                    f"Reservoir mapping mismatch for {live_name!r}: "
                    f"expected {expected_v3!r}, got {mapping[live_name]['v3_name']!r}."
                )
        return mapping

    # ---- public properties ----

    @property
    def reservoir_mapping(self) -> Dict[str, str]:
        """``{live_name: v3_name}`` — the four-reservoir mapping."""
        return {k: v["v3_name"] for k, v in self._mapping.items()}

    @property
    def node_mapping(self) -> Dict[str, str]:
        """``{live_name: network_node_id}``."""
        return {k: v["node_id"] for k, v in self._mapping.items()}

    @property
    def network_node_ids(self) -> List[str]:
        return [v["node_id"] for v in self._mapping.values()]

    # ---- value handling ----

    @staticmethod
    def _validate_value(value: Any) -> Tuple[Optional[float], ForecastStatus, str]:
        """
        Classify one forecast value. NEVER coerces a bad value into a good one.

        Returns ``(float_value_or_None, status, reason)``.
        """
        if value is None:
            return None, ForecastStatus.UNAVAILABLE, "value_missing"
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None, ForecastStatus.INVALID, f"non_numeric:{type(value).__name__}"
        number = float(value)
        if not math.isfinite(number):
            return None, ForecastStatus.INVALID, "non_finite"
        if number < 0.0:
            # Inflow is physically non-negative. A negative forecast is not a
            # number the MPC may act on; it is reported, not clamped.
            return None, ForecastStatus.INVALID, f"negative_inflow:{number!r}"
        return number, ForecastStatus.AVAILABLE, ""

    @staticmethod
    def _compact_provenance_string(
        declared_status: Optional[str],
        declared_provenance: Optional[str],
        is_simulated: Optional[bool],
        validated: Optional[bool],
        synthetic: List[str],
        unavailable: List[str],
    ) -> str:
        """
        Deterministic, parseable encoding of the live provenance, carried in the
        single free-text ``ReservoirForecast.provenance`` slot of the frozen
        Phase 15.2 contract (which must not be modified).
        """
        return (
            f"status={declared_status or 'UNSPECIFIED'}"
            f";source={declared_provenance or 'UNSPECIFIED'}"
            f";is_simulated={is_simulated if is_simulated is not None else 'UNKNOWN'}"
            f";validated_metrics_apply={validated if validated is not None else 'UNKNOWN'}"
            f";unit={FORECAST_UNIT}"
            f";synthetic={'+'.join(synthetic) if synthetic else 'none'}"
            f";unavailable={'+'.join(unavailable) if unavailable else 'none'}"
        )

    # ---- main entry points ----

    def build_bundle(self, live_forecasts: Dict[str, Any],
                     forecast_date: str) -> LiveForecastBundle:
        """
        Adapt ``{live_reservoir_name: forecast_payload}`` into the MPC contract.

        ``forecast_payload`` is the Stage 5 per-reservoir dict, e.g.::

            {"forecast_1d": 2.43, "forecast_3d": 2.54, "forecast_7d": 2.73,
             "forecast_status": "DEMONSTRATION_ONLY",
             "forecast_provenance": "SIMULATION_OR_SYNTHETIC_INPUTS_FROZEN_MODEL",
             "is_simulated": True, "validated_metrics_apply": False,
             "forecast_unit": "MCM/day",
             "horizons": ["forecast_1d", "forecast_3d", "forecast_7d"],
             "input_provenance": {...}}

        Every mapped reservoir ALWAYS appears in the returned snapshot — a
        reservoir with no payload is present and explicitly UNAVAILABLE.
        """
        live_forecasts = live_forecasts or {}
        snapshot = NetworkForecastSnapshot(forecast_date=str(forecast_date))
        provenance: Dict[str, Dict[str, Any]] = {}
        notes: List[str] = []

        for live_name, spec in self._mapping.items():
            node_id = spec["node_id"]
            v3_name = spec["v3_name"]
            payload = live_forecasts.get(live_name)

            record, prov = self._adapt_one(
                node_id=node_id,
                v3_name=v3_name,
                live_name=live_name,
                forecast_date=str(forecast_date),
                payload=payload,
            )
            snapshot.forecasts[node_id] = record
            provenance[node_id] = prov

        # Report any payload key that could not be mapped, rather than dropping
        # it silently.
        unmapped = sorted(k for k in live_forecasts.keys() if k not in self._mapping)
        if unmapped:
            notes.append(f"unmapped live forecast keys ignored: {unmapped}")

        status = "ADAPTED"
        if all(
            provenance[nid].get("issue") == "MISSING_FORECAST"
            for nid in provenance
        ):
            status = "NO_FORECASTS"

        return LiveForecastBundle(
            snapshot=snapshot,
            provenance=provenance,
            forecast_date=str(forecast_date),
            status=status,
            notes=notes,
        )

    def build_snapshot(self, live_forecasts: Dict[str, Any],
                       forecast_date: str) -> NetworkForecastSnapshot:
        """Return ONLY the ``NetworkForecastSnapshot`` the validated MPC consumes."""
        return self.build_bundle(live_forecasts, forecast_date).snapshot

    # ---- per-reservoir adaptation ----

    def _adapt_one(
        self,
        node_id: str,
        v3_name: str,
        live_name: str,
        forecast_date: str,
        payload: Any,
    ) -> Tuple[ReservoirForecast, Dict[str, Any]]:
        prov: Dict[str, Any] = {
            "network_node_id": node_id,
            "v3_reservoir_name": v3_name,
            "live_reservoir": live_name,
            "forecast_date": forecast_date,
            "unit": FORECAST_UNIT,
            "present": payload is not None,
            "issue": None,
            "declared_status": None,
            "declared_provenance": None,
            "is_simulated": None,
            "validated_metrics_apply": None,
            "input_provenance": None,
            "synthetic_features": [],
            "unavailable_features": [],
            "raw_values": {},
        }

        def _unavailable(issue: str, status: ForecastStatus = ForecastStatus.UNAVAILABLE):
            prov["issue"] = issue
            record = ReservoirForecast(
                reservoir_id=node_id,
                v3_reservoir_name=v3_name,
                forecast_date=forecast_date,
                target_1d=None, target_3d=None, target_7d=None,
                status_1d=status, status_3d=status, status_7d=status,
                model="LSTM_V3_LOGTARGET",
                provenance=self._compact_provenance_string(
                    prov["declared_status"], prov["declared_provenance"],
                    prov["is_simulated"], prov["validated_metrics_apply"],
                    prov["synthetic_features"], prov["unavailable_features"],
                ),
            )
            prov["status_1d"] = status.value
            prov["status_3d"] = status.value
            prov["status_7d"] = status.value
            return record, prov

        # --- 1. missing payload -------------------------------------------
        if payload is None:
            return _unavailable("MISSING_FORECAST")

        if not isinstance(payload, dict):
            return _unavailable("MALFORMED_PAYLOAD", ForecastStatus.INVALID)

        # --- 2. declared metadata ------------------------------------------
        declared_status = payload.get("forecast_status")
        declared_provenance = payload.get("forecast_provenance")
        is_simulated = payload.get("is_simulated")
        validated = payload.get("validated_metrics_apply")
        input_prov = payload.get("input_provenance")

        prov.update({
            "declared_status": declared_status,
            "declared_provenance": declared_provenance,
            "is_simulated": is_simulated,
            "validated_metrics_apply": validated,
            "input_provenance": input_prov,
        })
        if isinstance(input_prov, dict):
            prov["synthetic_features"] = list(input_prov.get("synthetic_demo", []) or [])
            prov["unavailable_features"] = list(input_prov.get("unavailable", []) or [])

        # --- 3. unit guard --------------------------------------------------
        declared_unit = payload.get("forecast_unit")
        if declared_unit is not None and declared_unit != FORECAST_UNIT:
            prov["unit_mismatch"] = declared_unit
            return _unavailable("UNIT_MISMATCH", ForecastStatus.INVALID)

        # --- 4. horizon guard -----------------------------------------------
        declared_horizons = payload.get("horizons")
        if declared_horizons is not None and list(declared_horizons) != list(HORIZON_KEYS):
            prov["horizon_mismatch"] = list(declared_horizons)
            return _unavailable("HORIZON_MISMATCH", ForecastStatus.INVALID)

        # --- 5. upstream withheld the forecast -------------------------------
        if declared_status in _WITHHELD_STATUSES:
            return _unavailable(str(declared_status))

        if declared_status == "FORECAST_UNAVAILABLE" or payload.get("unavailable_features"):
            return _unavailable("REQUIRED_FEATURE_UNAVAILABLE")

        # --- 6. per-horizon values -------------------------------------------
        values: Dict[str, Optional[float]] = {}
        statuses: Dict[str, ForecastStatus] = {}
        reasons: Dict[str, str] = {}
        for key, label in zip(HORIZON_KEYS, HORIZON_LABELS):
            raw = payload.get(key)
            prov["raw_values"][key] = raw
            value, status, reason = self._validate_value(raw)
            values[label] = value
            statuses[label] = status
            reasons[label] = reason

        prov["status_1d"] = statuses["1d"].value
        prov["status_3d"] = statuses["3d"].value
        prov["status_7d"] = statuses["7d"].value
        if any(reasons.values()):
            prov["value_issues"] = {k: v for k, v in reasons.items() if v}
            prov["issue"] = "INVALID_VALUE"

        # --- 7. build the record (values copied verbatim) --------------------
        record = ReservoirForecast(
            reservoir_id=node_id,
            v3_reservoir_name=v3_name,
            forecast_date=str(payload.get("forecast_date", forecast_date)),
            target_1d=values["1d"],
            target_3d=values["3d"],
            target_7d=values["7d"],
            status_1d=statuses["1d"],
            status_3d=statuses["3d"],
            status_7d=statuses["7d"],
            model="LSTM_V3_LOGTARGET",
            provenance=self._compact_provenance_string(
                declared_status, declared_provenance, is_simulated, validated,
                prov["synthetic_features"], prov["unavailable_features"],
            ),
        )
        return record, prov

    # ---- introspection ----

    def contract_info(self) -> Dict[str, Any]:
        """Describe the boundary this adapter implements."""
        # (documented in the returned dict below)
        return {
            "stage": "Stage 6 — Forecast Adapter",
            "input": "live forecast payload {live_reservoir_name: dict}",
            "output": "NetworkForecastSnapshot (Phase 15.2 contract, unmodified)",
            "unit": FORECAST_UNIT,
            "horizons": list(HORIZON_KEYS),
            "reservoir_mapping": self.reservoir_mapping,
            "node_mapping": self.node_mapping,
            "topology_config": str(self._topology_path),
            "alteration": "NONE — values are copied verbatim (float only)",
            "provenance": "carried in ReservoirForecast.provenance + bundle.provenance",
            "mpc_integrated": False,
            "safety_layer_integrated": False,
        }


# ---------------------------------------------------------------------------
# Provenance parsing / extraction helpers (Stage 7 consumes these)
# ---------------------------------------------------------------------------

def parse_provenance_string(provenance: str) -> Dict[str, Any]:
    """
    Parse the compact provenance encoding written into
    ``ReservoirForecast.provenance`` by :meth:`LiveForecastAdapter._adapt_one`.

    Returns a dict of raw strings. Missing keys are absent (never defaulted to a
    permissive value).
    """
    out: Dict[str, Any] = {}
    if not isinstance(provenance, str):
        return out
    for chunk in provenance.split(";"):
        if "=" not in chunk:
            continue
        key, _, value = chunk.partition("=")
        out[key.strip()] = value.strip()
    return out


def provenance_from_snapshot(snapshot: NetworkForecastSnapshot) -> Dict[str, Dict[str, Any]]:
    """
    Recover per-node provenance from a bare ``NetworkForecastSnapshot``.

    Allows a caller holding only the MPC contract (no bundle) to still apply a
    provenance gate, instead of having to trust the values blindly.
    """
    result: Dict[str, Dict[str, Any]] = {}
    for node_id, forecast in snapshot.forecasts.items():
        parsed = parse_provenance_string(getattr(forecast, "provenance", ""))
        result[node_id] = {
            "declared_status": parsed.get("status"),
            "declared_provenance": parsed.get("source"),
            "validated_metrics_apply": _as_bool(parsed.get("validated_metrics_apply")),
            "is_simulated": _as_bool(parsed.get("is_simulated")),
            "unit": parsed.get("unit"),
        }
    return result


def _as_bool(value: Any) -> Optional[bool]:
    """Strict bool parsing: never coerces an unknown value into True."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.strip() == "True":
            return True
        if value.strip() == "False":
            return False
    return None
