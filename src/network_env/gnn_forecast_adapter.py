"""
GNN Forecast Adapter — Bridges GNN inference to the application layer
=====================================================================

Provides structured forecast metadata from the Gated GCN-LSTM V1 model,
keeping GNN and LSTM forecasts strictly separate.

SCIENTIFIC POSITION (Stage 14)
------------------------------
The Gated GCN-LSTM V1 is an EXPERIMENTAL spatial-dependency model, not a
production forecaster. A controlled experiment showed it underperforming the
temporal LSTM V3 baseline, and its learned fusion gate collapsed to ~0.015
(i.e. it learned to ignore most of the graph signal). It is retained and
exposed as ADVISORY context only:

  * it does not discover causal relationships;
  * it does not prove hydraulic connectivity or routing;
  * it never controls gates and is not a safety mechanism.

DESIGN PRINCIPLES
-----------------
1. GNN forecasts are NEVER mixed with LSTM V3 forecasts.
2. The adapter caches the latest GNN inference result to avoid redundant
   computation (inference runs once per simulation tick, not per frame).
3. Missing/invalid forecasts are explicitly flagged (never silently fabricated).
4. All values are in MCM/day (native training units).
5. This adapter NEVER modifies the LSTM V3 forecast adapter or its data.

USAGE
-----
The application layer (FastAPI ``state_manager``) calls this adapter once per
simulation step to obtain the latest GNN forecasts, which are cached and
published as part of the Stage 14 DISPLAY-ONLY advisory block.

STAGE 14 — ADVISORY SCOPE
-------------------------
GNN forecasts do **not** reach the controller or any risk evaluation in the
live system. The methods below can BUILD alternative control forecasts
(``gnn_primary``, ``risk_envelope``), but those policies are offline research
affordances that have never been closed-loop validated; the live control path
uses ``lstm_primary`` only, and ``GlobalSimulationState`` additionally refuses
the cycle fail-closed if a control forecast ever differs from the validated
frozen-LSTM value. The GNN is a spatial-dependency ADVISORY, not a controller
and not a safety mechanism.
"""

import math
from pathlib import Path
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field
from enum import Enum


class GNNForecastStatus(Enum):
    """Status of a GNN forecast for a single reservoir-horizon."""
    AVAILABLE = "GNN_FORECAST_AVAILABLE"
    UNAVAILABLE = "GNN_FORECAST_UNAVAILABLE"
    INVALID = "GNN_FORECAST_INVALID"
    NODE_MISSING = "GNN_NODE_MISSING"


@dataclass
class GNNReservoirForecast:
    """
    GNN forecast for a single reservoir at a single timestep.
    
    All values are POINT FORECASTS of inflow in MCM/day.
    target_1d = predicted inflow on day+1
    target_3d = predicted inflow on day+3
    target_7d = predicted inflow on day+7
    """
    reservoir_name: str         # Canonical reservoir name (e.g. "Anayirankal")
    network_node_id: str        # Network topology node ID (e.g. "Reservoir_A")
    
    # Point forecasts (MCM/day)
    target_1d: Optional[float] = None
    target_3d: Optional[float] = None
    target_7d: Optional[float] = None
    
    # Status per horizon
    status_1d: GNNForecastStatus = GNNForecastStatus.UNAVAILABLE
    status_3d: GNNForecastStatus = GNNForecastStatus.UNAVAILABLE
    status_7d: GNNForecastStatus = GNNForecastStatus.UNAVAILABLE
    
    # Metadata
    model: str = "Gated_GCN_LSTM_V1"
    graph: str = "Graph_D_correlation_v1_2"
    units: str = "MCM/day"
    
    def is_available(self, horizon: str) -> bool:
        status = getattr(self, f"status_{horizon}", GNNForecastStatus.UNAVAILABLE)
        return status == GNNForecastStatus.AVAILABLE
    
    def get_prediction(self, horizon: str) -> Optional[float]:
        if not self.is_available(horizon):
            return None
        return getattr(self, f"target_{horizon}", None)


@dataclass
class GNNNetworkForecastSnapshot:
    """
    Complete GNN forecast snapshot for all reservoirs.
    """
    forecasts: Dict[str, GNNReservoirForecast] = field(default_factory=dict)
    inference_time_ms: float = 0.0
    gate_value: float = 0.0
    
    def get(self, reservoir_name: str) -> Optional[GNNReservoirForecast]:
        return self.forecasts.get(reservoir_name)
    
    def available_count(self, horizon: str) -> int:
        return sum(
            1 for f in self.forecasts.values() if f.is_available(horizon)
        )


class GNNForecastAdapter:
    """
    Adapter that converts raw GNN inference outputs into structured
    forecast metadata for the application layer.
    
    IMPORTANT:
    - This adapter does NOT perform inference itself.
    - It receives pre-computed inference results from LiveGNNForecaster
      and wraps them into the application schema.
    - LSTM and GNN forecasts are kept strictly separate.
    """
    
    def __init__(self, reservoir_mapping: Dict[str, str]):
        """
        Parameters
        ----------
        reservoir_mapping : dict
            Maps network node IDs → canonical reservoir names.
            e.g. {"Reservoir_A": "Anayirankal", ...}
        """
        self._mapping = dict(reservoir_mapping)
        self._reverse_mapping = {v: k for k, v in self._mapping.items()}
        self._latest_snapshot: Optional[GNNNetworkForecastSnapshot] = None
    
    def _validate_value(self, val: Optional[float]) -> GNNForecastStatus:
        """Check if a prediction value is valid."""
        if val is None:
            return GNNForecastStatus.UNAVAILABLE
        if not isinstance(val, (int, float)):
            return GNNForecastStatus.INVALID
        if math.isnan(val) or math.isinf(val):
            return GNNForecastStatus.INVALID
        return GNNForecastStatus.AVAILABLE
    
    def update_from_inference(
        self,
        gnn_result: Dict[str, Dict[str, float]],
        inference_time_ms: float = 0.0,
        gate_value: float = 0.0,
    ) -> GNNNetworkForecastSnapshot:
        """
        Convert raw GNN inference output into a structured forecast snapshot.
        
        Parameters
        ----------
        gnn_result : dict
            Output from LiveGNNForecaster.predict_all():
            {reservoir_name: {"target_1d": float, "target_3d": float, "target_7d": float, "available": bool}}
        inference_time_ms : float
            Inference latency in milliseconds.
        gate_value : float
            Current learned gate value sigmoid(alpha).
            
        Returns
        -------
        GNNNetworkForecastSnapshot
        """
        snapshot = GNNNetworkForecastSnapshot(
            inference_time_ms=inference_time_ms,
            gate_value=gate_value,
        )
        
        for res_name, preds in gnn_result.items():
            node_id = self._reverse_mapping.get(res_name, res_name)
            is_available = preds.get("available", True)
            
            t1d = preds.get("target_1d")
            t3d = preds.get("target_3d")
            t7d = preds.get("target_7d")
            
            if not is_available:
                status_1d = GNNForecastStatus.NODE_MISSING
                status_3d = GNNForecastStatus.NODE_MISSING
                status_7d = GNNForecastStatus.NODE_MISSING
            else:
                status_1d = self._validate_value(t1d)
                status_3d = self._validate_value(t3d)
                status_7d = self._validate_value(t7d)
            
            forecast = GNNReservoirForecast(
                reservoir_name=res_name,
                network_node_id=node_id,
                target_1d=t1d,
                target_3d=t3d,
                target_7d=t7d,
                status_1d=status_1d,
                status_3d=status_3d,
                status_7d=status_7d,
            )
            snapshot.forecasts[res_name] = forecast
        
        self._latest_snapshot = snapshot
        return snapshot
    
    @property
    def latest_snapshot(self) -> Optional[GNNNetworkForecastSnapshot]:
        """Return the most recent cached forecast snapshot."""
        return self._latest_snapshot
    
    def get_combined_forecast_dict(
        self,
        lstm_forecasts: Dict[str, Dict[str, Optional[float]]],
    ) -> Dict[str, Dict[str, Dict[str, Optional[float]]]]:
        """
        Merge LSTM and GNN forecasts into a combined dict.
        
        LSTM and GNN predictions are kept in SEPARATE sub-keys.
        Neither overwrites the other.
        
        Parameters
        ----------
        lstm_forecasts : dict
            Maps reservoir/node name → {"forecast_1d": float, "forecast_3d": float, "forecast_7d": float}
            (existing LSTM V3 forecasts from the current pipeline).
            
        Returns
        -------
        dict
            {
                "Reservoir_A": {
                    "lstm": {"1d": 12.0, "3d": 14.5, "7d": 10.0},
                    "gnn":  {"1d": 11.8, "3d": 13.9, "7d": 11.2},
                }
            }
        """
        combined = {}
        
        # Add LSTM forecasts
        for name, f_data in lstm_forecasts.items():
            combined[name] = {
                "lstm": {
                    "1d": f_data.get("forecast_1d"),
                    "3d": f_data.get("forecast_3d"),
                    "7d": f_data.get("forecast_7d"),
                },
                "gnn": {"1d": None, "3d": None, "7d": None},
            }
        
        # Add GNN forecasts
        if self._latest_snapshot:
            for res_name, gnn_f in self._latest_snapshot.forecasts.items():
                # Map canonical name to the node ID used in lstm_forecasts
                node_id = gnn_f.network_node_id
                
                # Try both the node_id and the canonical name
                target_key = None
                if node_id in combined:
                    target_key = node_id
                elif res_name in combined:
                    target_key = res_name
                
                if target_key:
                    combined[target_key]["gnn"] = {
                        "1d": gnn_f.target_1d if gnn_f.is_available("1d") else None,
                        "3d": gnn_f.target_3d if gnn_f.is_available("3d") else None,
                        "7d": gnn_f.target_7d if gnn_f.is_available("7d") else None,
                    }
        
        return combined
    
    def get_risk_envelope_forecasts(
        self,
        lstm_forecasts: Dict[str, Dict[str, Optional[float]]],
    ) -> Dict[str, Dict[str, Optional[float]]]:
        """
        Generate risk-envelope forecasts: for each horizon, use the MAXIMUM
        predicted inflow between LSTM and GNN.
        
        This is the pessimistic (conservative) policy for RISK EVALUATION only.
        NOT for control — see ``get_control_forecasts()`` for control policy.
        
        Parameters
        ----------
        lstm_forecasts : dict
            Standard forecast dict from the existing pipeline.
            
        Returns
        -------
        dict
            Same format as lstm_forecasts, but with max-envelope values.
        """
        envelope = {}
        
        for name, f_data in lstm_forecasts.items():
            lstm_1d = f_data.get("forecast_1d")
            lstm_3d = f_data.get("forecast_3d")
            lstm_7d = f_data.get("forecast_7d")
            
            gnn_1d, gnn_3d, gnn_7d = None, None, None
            if self._latest_snapshot:
                # Try to find GNN forecast by node_id or canonical name
                gnn_f = None
                for res_name, gf in self._latest_snapshot.forecasts.items():
                    if gf.network_node_id == name or res_name == name:
                        gnn_f = gf
                        break
                
                if gnn_f:
                    gnn_1d = gnn_f.target_1d if gnn_f.is_available("1d") else None
                    gnn_3d = gnn_f.target_3d if gnn_f.is_available("3d") else None
                    gnn_7d = gnn_f.target_7d if gnn_f.is_available("7d") else None
            
            def _max_or_fallback(a, b):
                if a is not None and b is not None:
                    return max(a, b)
                return a if a is not None else b
            
            envelope[name] = {
                "forecast_1d": _max_or_fallback(lstm_1d, gnn_1d),
                "forecast_3d": _max_or_fallback(lstm_3d, gnn_3d),
                "forecast_7d": _max_or_fallback(lstm_7d, gnn_7d),
                "source": "risk_envelope(LSTM_V3, GCN_LSTM_Gated_V1)",
            }
        
        return envelope
    
    def get_control_forecasts(
        self,
        lstm_forecasts: Dict[str, Dict[str, Optional[float]]],
        policy: str = "lstm_primary",
    ) -> Dict[str, Dict[str, Optional[float]]]:
        """
        Generate forecasts for the AI/MPC controller.
        
        The control policy is separate from the risk policy and must be
        validated through closed-loop simulation before use.
        
        Supported policies:
        - "lstm_primary": Use LSTM V3 only (default, validated baseline).
        - "gnn_primary": Use GNN only (requires closed-loop validation).
        - "risk_envelope": Use max(LSTM, GNN) — conservative but may cause
          over-release.
          
        Parameters
        ----------
        lstm_forecasts : dict
            Standard forecast dict from existing pipeline.
        policy : str
            One of "lstm_primary", "gnn_primary", "risk_envelope".
            
        Returns
        -------
        dict
            Forecast dict in standard format for ForecastAwareController.
        """
        if policy == "lstm_primary":
            # Default: LSTM V3 remains the control forecast
            result = {}
            for name, f_data in lstm_forecasts.items():
                result[name] = {
                    "forecast_1d": f_data.get("forecast_1d"),
                    "forecast_3d": f_data.get("forecast_3d"),
                    "forecast_7d": f_data.get("forecast_7d"),
                    "source": "LSTM_V3",
                }
            return result
        
        elif policy == "gnn_primary":
            result = {}
            for name, f_data in lstm_forecasts.items():
                gnn_1d, gnn_3d, gnn_7d = None, None, None
                if self._latest_snapshot:
                    for res_name, gf in self._latest_snapshot.forecasts.items():
                        if gf.network_node_id == name or res_name == name:
                            gnn_1d = gf.target_1d if gf.is_available("1d") else None
                            gnn_3d = gf.target_3d if gf.is_available("3d") else None
                            gnn_7d = gf.target_7d if gf.is_available("7d") else None
                            break
                
                result[name] = {
                    "forecast_1d": gnn_1d if gnn_1d is not None else f_data.get("forecast_1d"),
                    "forecast_3d": gnn_3d if gnn_3d is not None else f_data.get("forecast_3d"),
                    "forecast_7d": gnn_7d if gnn_7d is not None else f_data.get("forecast_7d"),
                    "source": "GCN_LSTM_Gated_V1" if gnn_1d is not None else "LSTM_V3_fallback",
                }
            return result
        
        elif policy == "risk_envelope":
            return self.get_risk_envelope_forecasts(lstm_forecasts)
        
        else:
            raise ValueError(
                f"Unknown control policy: {policy}. "
                f"Supported: lstm_primary, gnn_primary, risk_envelope"
            )
