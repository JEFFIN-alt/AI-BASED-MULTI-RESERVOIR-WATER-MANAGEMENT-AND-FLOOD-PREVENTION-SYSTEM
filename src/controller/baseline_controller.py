"""
Phase 15.3 — Baseline Threshold Controller

A simple, deterministic, state-based controller that uses ONLY
current reservoir storage levels to set gate positions.

It does NOT use V3 forecasts.
It does NOT consider other reservoirs (no coordination).
It makes purely local, threshold-based decisions.

This serves as the comparison baseline for the forecast-aware
MPC controller.

Thresholds are derived from the existing Risk Engine's blue/orange/red
warning levels (reused from src/management/risk_engine.py semantics).
"""

from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class BaselineConfig:
    """
    Threshold-based release policy parameters.
    All ASSUMED_FOR_PROTOTYPE.
    """
    # Storage fraction thresholds (from topology_config.yaml warning_thresholds)
    low_threshold: float = 0.3    # below: close gates (conserve water)
    blue_threshold: float = 0.75  # above: begin moderate release
    orange_threshold: float = 0.85  # above: increase release
    red_threshold: float = 0.95   # above: emergency maximum release

    # Gate positions for each regime
    gate_low: float = 0.0          # storage < low → closed
    gate_normal: float = 0.05      # low ≤ storage < blue → trickle
    gate_blue: float = 0.30        # blue ≤ storage < orange → moderate
    gate_orange: float = 0.60      # orange ≤ storage < red → high
    gate_red: float = 1.0          # storage ≥ red → fully open


class BaselineController:
    """
    Simple threshold-based controller.

    Decision rule per reservoir (INDEPENDENT — no coordination):
      if storage/capacity < low      → gate = 0.0  (closed)
      if low ≤ fraction < blue       → gate = 0.05 (trickle)
      if blue ≤ fraction < orange    → gate = 0.30 (moderate)
      if orange ≤ fraction < red     → gate = 0.60 (high)
      if fraction ≥ red              → gate = 1.0  (maximum)
    """

    def __init__(self, config: Optional[BaselineConfig] = None):
        self.config = config or BaselineConfig()

    def decide(
        self,
        storages: Dict[str, float],
        capacities: Dict[str, float],
    ) -> Dict[str, dict]:
        """
        Compute gate positions based on current storage only.

        Parameters
        ----------
        storages : {node_id: current_storage_mcm}
        capacities : {node_id: capacity_mcm}

        Returns
        -------
        {node_id: {"gate_position": float, "reason": str}}
        """
        c = self.config
        decisions = {}

        for nid in storages:
            cap = capacities.get(nid, 1.0)
            frac = storages[nid] / cap if cap > 0 else 0.0

            if frac >= c.red_threshold:
                gate = c.gate_red
                reason = f"storage {frac*100:.1f}% ≥ RED ({c.red_threshold*100}%) → max release"
            elif frac >= c.orange_threshold:
                gate = c.gate_orange
                reason = f"storage {frac*100:.1f}% ≥ ORANGE ({c.orange_threshold*100}%) → high release"
            elif frac >= c.blue_threshold:
                gate = c.gate_blue
                reason = f"storage {frac*100:.1f}% ≥ BLUE ({c.blue_threshold*100}%) → moderate release"
            elif frac >= c.low_threshold:
                gate = c.gate_normal
                reason = f"storage {frac*100:.1f}% normal → trickle release"
            else:
                gate = c.gate_low
                reason = f"storage {frac*100:.1f}% < LOW ({c.low_threshold*100}%) → gates closed"

            decisions[nid] = {
                "gate_position": gate,
                "reason": reason,
                "storage_fraction": frac,
                "controller": "BASELINE_THRESHOLD",
            }

        return decisions
