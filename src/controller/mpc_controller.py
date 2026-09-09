"""
Phase 15.3 — Deterministic MPC-Style Coordinated Reservoir Controller

Receding-horizon controller that uses:
  - Current network state
  - V3 forecast metadata (1d/3d/7d point forecasts)
  - Network dynamics model (Phase 15.1 reservoir_network)
  - Explicit objective function (objective.py)
  - Safety validation layer (safety.py)

to compute coordinated gate actions across all reservoirs.

FORECAST HANDLING — CRITICAL:
  V3 provides point forecasts for days +1, +3, +7 ONLY.
  Days +2, +4, +5, +6 are NOT predicted by V3.
  This controller does NOT interpolate missing horizons.
  This controller does NOT fabricate intermediate forecasts.

  Strategy: The MPC uses a 3-step lookahead aligned with the
  available V3 horizons:
    Step 0 (current day): use actual/observed current inflow
    Step 1 (day +1):      use target_1d if available, else hold current
    Step 2 (day +3):      use target_3d if available, else hold current

  Day +7 (target_7d) is used as an informational signal to adjust
  the objective's storage target but is NOT used as an explicit
  simulation step (since days +4-6 would require fabricated inflows).

  This is a scientifically defensible choice because:
  1. We only simulate through points with forecast support
  2. We do not invent intermediate-day forecasts
  3. The 7-day signal provides situational awareness without
     pretending we have daily resolution

OPTIMIZATION METHOD:
  Grid search over discrete candidate gate positions for each
  reservoir at each step. With 4 reservoirs and ~5 gate levels,
  the search space is manageable (5^4 = 625 candidates per step,
  evaluated via rollout on a cloned network).

  This is transparent, deterministic, and reproducible.
"""

import copy
import itertools
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from ..network_env.reservoir_network import ReservoirNetwork, ReservoirState
from ..network_env.v3_forecast_adapter import (
    V3ForecastAdapter, NetworkForecastSnapshot, ForecastStatus
)
from .objective import ObjectiveFunction, ObjectiveWeights
from .safety import SafetyLayer, SafetyCheckResult

logger = logging.getLogger(__name__)


@dataclass
class ControlDecision:
    """
    Structured output from the MPC controller.
    """
    gate_positions: Dict[str, float]        # {node_id: position [0,1]}
    objective_score: float
    status: str                             # OPTIMAL, FALLBACK, EMERGENCY
    reasons: List[str] = field(default_factory=list)
    per_node: Dict[str, dict] = field(default_factory=dict)
    forecast_used: bool = False
    forecast_status: str = ""
    safety_status: str = ""
    candidates_evaluated: int = 0


@dataclass
class MPCConfig:
    """
    MPC controller configuration. All non-physical parameters are
    ASSUMED_FOR_PROTOTYPE.
    """
    # Gate candidate levels for grid search
    gate_levels: List[float] = field(
        default_factory=lambda: [0.0, 0.15, 0.30, 0.50, 0.70, 1.0]
    )

    # MPC lookahead steps (aligned with V3 forecast availability)
    # Step 0 = current day, Step 1 = day+1, Step 2 = day+3
    lookahead_steps: int = 3

    # Fallback inflow when forecast unavailable (hold current)
    use_current_inflow_as_fallback: bool = True

    # Informational: if 7-day forecast shows high inflow, bias
    # storage target lower to create headroom
    day7_headroom_factor: float = 0.05  # reduce target_high by this if 7d is high

    # Maximum gate change per step
    max_gate_change: float = 0.5


class MPCController:
    """
    Receding-horizon coordinated controller for the reservoir network.

    Algorithm:
    1. Clone current network state
    2. Read V3 forecasts for current date
    3. Build inflow scenarios for lookahead steps
    4. Grid-search over candidate gate action sequences
    5. For each candidate, simulate the trajectory on the clone
    6. Score with objective function
    7. Select lowest-cost action
    8. Validate through safety layer
    9. Return ONLY the first-step gate positions
    """

    def __init__(
        self,
        config: Optional[MPCConfig] = None,
        weights: Optional[ObjectiveWeights] = None,
    ):
        self.config = config or MPCConfig()
        self.objective = ObjectiveFunction(weights)
        self.safety = SafetyLayer(max_gate_change_per_step=self.config.max_gate_change)

    def decide(
        self,
        network: ReservoirNetwork,
        forecast_snapshot: Optional[NetworkForecastSnapshot] = None,
        current_inflows: Optional[Dict[str, float]] = None,
    ) -> ControlDecision:
        """
        Compute optimal coordinated gate actions.

        Parameters
        ----------
        network : ReservoirNetwork
            The REAL network (will NOT be mutated).
        forecast_snapshot : NetworkForecastSnapshot, optional
            V3 forecast metadata for current date.
        current_inflows : dict, optional
            Current observed external inflows {node_id: mcm/day}.

        Returns
        -------
        ControlDecision
        """
        node_ids = network.processing_order
        capacities = {nid: network.nodes[nid].capacity for nid in node_ids}

        # Current state
        current_gates = {nid: network.nodes[nid].state.gate_position for nid in node_ids}
        if current_inflows is None:
            current_inflows = {nid: network.nodes[nid].state.inflow_local for nid in node_ids}

        # --- Build inflow scenarios for lookahead ---
        forecast_used = False
        forecast_status = "NO_FORECAST"
        inflow_scenarios = self._build_inflow_scenarios(
            node_ids, current_inflows, forecast_snapshot
        )
        if forecast_snapshot:
            any_available = any(
                forecast_snapshot.get(nid) and
                forecast_snapshot.get(nid).is_available("1d")
                for nid in node_ids
            )
            if any_available:
                forecast_used = True
                forecast_status = "FORECAST_USED"
            else:
                forecast_status = "FORECAST_UNAVAILABLE"

        # --- Grid search over candidate gate actions ---
        best_cost = float('inf')
        best_gates = None
        best_result = None
        candidates_evaluated = 0

        # For the first step, evaluate all gate combinations
        gate_combos = list(itertools.product(self.config.gate_levels, repeat=len(node_ids)))

        for combo in gate_combos:
            candidate_gates_step0 = {nid: g for nid, g in zip(node_ids, combo)}

            # Simulate trajectory on a CLONE
            trajectory = self._simulate_trajectory(
                network, candidate_gates_step0, inflow_scenarios, node_ids
            )
            candidates_evaluated += 1

            if trajectory is None:
                continue

            # Score
            result = self.objective.evaluate_trajectory(
                trajectory, capacities,
                network.downstream_capacity,
                network._terminal_node_id,
                previous_gates=dict(current_gates),
            )

            if result["total_cost"] < best_cost:
                best_cost = result["total_cost"]
                best_gates = candidate_gates_step0
                best_result = result

        # --- Handle case where no candidate was found ---
        if best_gates is None:
            return ControlDecision(
                gate_positions=SafetyLayer.emergency_fallback(node_ids),
                objective_score=float('inf'),
                status="EMERGENCY",
                reasons=["No feasible candidate action found"],
                forecast_used=forecast_used,
                forecast_status=forecast_status,
                safety_status="EMERGENCY",
            )

        # --- Safety validation ---
        safety_result = self.safety.validate(best_gates, current_gates, node_ids)

        # --- Build per-node explanation ---
        per_node = {}
        for nid in node_ids:
            node = network.nodes[nid]
            gate = safety_result.validated_gates[nid]
            release = gate * node.max_release
            frac = node.storage_fraction

            reason_parts = [f"storage={frac*100:.1f}%"]
            if forecast_snapshot and forecast_snapshot.get(nid):
                fc = forecast_snapshot.get(nid)
                if fc.is_available("1d"):
                    reason_parts.append(f"V3_1d={fc.target_1d:.2f}")
                if fc.is_available("3d"):
                    reason_parts.append(f"V3_3d={fc.target_3d:.2f}")

            per_node[nid] = {
                "gate_position": gate,
                "estimated_release": release,
                "storage_fraction": frac,
                "reason": ", ".join(reason_parts),
                "controller": "MPC_FORECAST_AWARE" if forecast_used else "MPC_NO_FORECAST",
            }

        return ControlDecision(
            gate_positions=safety_result.validated_gates,
            objective_score=best_cost,
            status="OPTIMAL" if safety_result.is_safe else safety_result.status,
            reasons=best_result.get("reasons", []) + safety_result.violations,
            per_node=per_node,
            forecast_used=forecast_used,
            forecast_status=forecast_status,
            safety_status=safety_result.status,
            candidates_evaluated=candidates_evaluated,
        )

    def _build_inflow_scenarios(
        self,
        node_ids: List[str],
        current_inflows: Dict[str, float],
        forecast_snapshot: Optional[NetworkForecastSnapshot],
    ) -> List[Dict[str, float]]:
        """
        Build inflow values for each MPC lookahead step.

        Steps aligned with V3 forecast availability:
          Step 0 (current): current observed inflow
          Step 1 (day+1):   V3 target_1d if available, else current
          Step 2 (day+3):   V3 target_3d if available, else current

        NO interpolation. NO fabrication.
        """
        steps = []

        for step_idx in range(self.config.lookahead_steps):
            step_inflows = {}

            for nid in node_ids:
                # Default: hold current inflow
                inflow = current_inflows.get(nid, 0.0)

                if forecast_snapshot and forecast_snapshot.get(nid):
                    fc = forecast_snapshot.get(nid)
                    if step_idx == 0:
                        # Current day: use actual current inflow
                        inflow = current_inflows.get(nid, 0.0)
                    elif step_idx == 1:
                        # Day +1: use target_1d if available
                        if fc.is_available("1d"):
                            inflow = fc.target_1d
                    elif step_idx == 2:
                        # Day +3: use target_3d if available
                        if fc.is_available("3d"):
                            inflow = fc.target_3d

                step_inflows[nid] = inflow

            steps.append(step_inflows)

        return steps

    def _simulate_trajectory(
        self,
        real_network: ReservoirNetwork,
        candidate_gates: Dict[str, float],
        inflow_scenarios: List[Dict[str, float]],
        node_ids: List[str],
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Simulate a multi-step trajectory on a CLONE of the network.

        The real network is NEVER mutated.

        Returns list of state dicts, one per step.
        """
        # Deep clone the network config and rebuild
        clone = ReservoirNetwork(config_dict=copy.deepcopy(real_network._raw_config))

        # Copy current storage state from the real network
        for nid in node_ids:
            real_storage = real_network.nodes[nid].state.storage
            clone.nodes[nid].state.storage = real_storage

        # Copy routing queue state
        for i, conn in enumerate(real_network.connections):
            clone.connections[i].queue = copy.deepcopy(conn.queue)

        trajectory = []

        for step_idx, step_inflows in enumerate(inflow_scenarios):
            # For simplicity, hold gates constant across lookahead
            # (the first-step action is what we select)
            gates = candidate_gates

            try:
                states = clone.step(step_inflows, gates)
                trajectory.append(states)
            except Exception as e:
                logger.warning(f"Simulation error at step {step_idx}: {e}")
                return None

        return trajectory
