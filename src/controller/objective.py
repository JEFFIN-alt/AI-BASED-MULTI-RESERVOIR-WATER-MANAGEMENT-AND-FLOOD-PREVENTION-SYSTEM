"""
Phase 15.3 — Objective Function for Coordinated Reservoir Control

Defines the cost function J that scores candidate gate action sequences.

J = W_overflow × overflow_penalty
  + W_downstream × downstream_violation_penalty
  + W_storage × storage_deviation_penalty
  + W_release × unnecessary_release_penalty
  + W_gate × gate_movement_penalty

Design rationale:
  - Overflow and downstream violations have 10-100× higher weight than
    ordinary storage deviations, ensuring the controller NEVER trades
    downstream safety for marginal storage improvements.
  - Storage deviation is measured as distance from a preferred operating
    band [target_low, target_high], not a single point.
  - Unnecessary release is penalized to prevent the controller from
    draining reservoirs when not under pressure.
  - Gate movement penalty smooths control actions across timesteps.

All weights are classified as ASSUMED_FOR_PROTOTYPE.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ObjectiveWeights:
    """
    Tunable weights for the objective function.
    All classified as ASSUMED_FOR_PROTOTYPE.
    """
    # Safety-critical penalties (high weight)
    overflow: float = 100.0        # per MCM of overflow
    downstream_violation: float = 200.0  # per MCM/day above limit

    # Operational penalties (moderate weight)
    storage_deviation: float = 1.0  # per unit deviation from target band

    # Smoothness penalties (low weight)
    unnecessary_release: float = 0.5  # per MCM of release when storage < target_high
    gate_movement: float = 0.2       # per unit of gate position change

    # Preferred operating band (fraction of capacity)
    target_storage_low: float = 0.3   # below this: penalize (too low)
    target_storage_high: float = 0.85  # above this: penalize (too high, risk)


class ObjectiveFunction:
    """
    Evaluates the cost of a simulated trajectory for a candidate
    gate action sequence.

    Lower cost = better.
    """

    def __init__(self, weights: Optional[ObjectiveWeights] = None):
        self.weights = weights or ObjectiveWeights()

    def evaluate_trajectory(
        self,
        trajectory: List[Dict[str, dict]],
        capacities: Dict[str, float],
        downstream_capacity: float,
        terminal_node_id: str,
        previous_gates: Optional[Dict[str, float]] = None,
    ) -> dict:
        """
        Score a multi-timestep trajectory.

        Parameters
        ----------
        trajectory : list of dicts
            Each entry is {node_id: ReservoirState} for one timestep.
        capacities : dict
            {node_id: capacity_mcm}
        downstream_capacity : float
            Maximum safe downstream flow (MCM/day).
        terminal_node_id : str
            ID of the terminal reservoir.
        previous_gates : dict, optional
            Gate positions from the timestep before the trajectory.

        Returns
        -------
        dict with total_cost, component breakdown, and reasons.
        """
        w = self.weights
        total_overflow = 0.0
        total_ds_violation = 0.0
        total_storage_dev = 0.0
        total_unnecessary_release = 0.0
        total_gate_movement = 0.0
        reasons = []

        prev_gates = previous_gates or {}

        for t, step_states in enumerate(trajectory):
            step_overflow = 0.0
            step_ds_violation = 0.0

            for nid, state in step_states.items():
                cap = capacities[nid]

                # --- Overflow penalty ---
                if state.spill > 0:
                    step_overflow += state.spill
                    reasons.append(f"t={t+1}: {nid} spill={state.spill:.2f} MCM")

                # --- Storage deviation from preferred band ---
                frac = state.storage / cap if cap > 0 else 0
                if frac < w.target_storage_low:
                    dev = w.target_storage_low - frac
                    total_storage_dev += dev * cap  # scale by capacity
                elif frac > w.target_storage_high:
                    dev = frac - w.target_storage_high
                    total_storage_dev += dev * cap * 3.0  # higher penalty for being too full

                # --- Unnecessary release penalty ---
                if frac < w.target_storage_high and state.controlled_release > 0:
                    total_unnecessary_release += state.controlled_release * 0.1

                # --- Gate movement penalty ---
                prev_g = prev_gates.get(nid, state.gate_position)
                gate_delta = abs(state.gate_position - prev_g)
                total_gate_movement += gate_delta

                prev_gates[nid] = state.gate_position

            total_overflow += step_overflow

            # --- Downstream violation ---
            terminal_state = step_states.get(terminal_node_id)
            if terminal_state:
                ds_flow = terminal_state.total_outflow
                if ds_flow > downstream_capacity:
                    violation = ds_flow - downstream_capacity
                    step_ds_violation += violation
                    reasons.append(
                        f"t={t+1}: downstream flow={ds_flow:.2f} > limit={downstream_capacity:.2f}"
                    )
            total_ds_violation += step_ds_violation

        # Compute weighted total
        cost = (
            w.overflow * total_overflow
            + w.downstream_violation * total_ds_violation
            + w.storage_deviation * total_storage_dev
            + w.unnecessary_release * total_unnecessary_release
            + w.gate_movement * total_gate_movement
        )

        return {
            "total_cost": cost,
            "overflow_cost": w.overflow * total_overflow,
            "downstream_cost": w.downstream_violation * total_ds_violation,
            "storage_cost": w.storage_deviation * total_storage_dev,
            "release_cost": w.unnecessary_release * total_unnecessary_release,
            "gate_cost": w.gate_movement * total_gate_movement,
            "raw_overflow_mcm": total_overflow,
            "raw_ds_violation_mcm_day": total_ds_violation,
            "reasons": reasons,
        }
