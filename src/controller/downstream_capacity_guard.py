"""Finite-search downstream-capacity guard using the authoritative daily physics.

The terminal flow includes controlled release and terminal spill. Upstream spill
is an explicitly separate outlet in this model. Routing queues are copied.
Candidates obey gate bounds and per-day movement limits. Lower gates are NOT a
proof of lower future peaks: early releases can create headroom before arrivals.
A failed finite search is NOT proof of global infeasibility. FAILED_CLOSED is a
legacy status name: it means capacity protection was not established, not that
flooding has been prevented. The least-peak tested admissible action is returned.
Predictions assume held local inflows and held gates; they are not flood guarantees.
"""

from __future__ import annotations

import copy
import itertools
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..network_env.reservoir_network import ReservoirNetwork

# ---------------------------------------------------------------------------
# Statuses — deliberately SEPARATE from the MPC and SafetyLayer statuses
# ---------------------------------------------------------------------------

#: The check ran and the applied action keeps the predicted downstream flow
#: within capacity at every step of the horizon, without modification.
DOWNSTREAM_STATUS_PROTECTED = "PROTECTED"
#: The check ran, the proposal was unsafe, and a safe alternative was applied.
DOWNSTREAM_STATUS_CORRECTED = "CORRECTED"
#: No safe candidate was found, or prediction failed. No global impossibility proof.
DOWNSTREAM_STATUS_FAILED_CLOSED = "FAILED_CLOSED"
#: No check was needed/possible because the MPC did not produce an action.
DOWNSTREAM_STATUS_NOT_APPLIED_MPC_BLOCKED = "NOT_APPLIED_MPC_BLOCKED"
DOWNSTREAM_STATUS_NOT_APPLIED_MPC_ERROR = "NOT_APPLIED_MPC_ERROR"
DOWNSTREAM_STATUS_NOT_APPLIED_ADAPTER_ERROR = "NOT_APPLIED_ADAPTER_ERROR"
#: The authoritative network declares no downstream capacity, so nothing can be
#: guaranteed. Reported instead of claiming protection.
DOWNSTREAM_STATUS_NOT_APPLIED_NO_CAPACITY = "NOT_APPLIED_NO_CAPACITY"

#: Numerical tolerance for the capacity comparison (MCM/day).
#: A predicted flow is "within capacity" when
#: ``flow <= capacity + TOLERANCE_MCM_DAY``. Chosen far below any physically
#: meaningful flow: one gate level changes the flow by ~0.4 MCM/day and the
#: network's own mass-balance residual is ~1e-13 MCM.
TOLERANCE_MCM_DAY = 1e-9

#: The project's existing conservative gate value (the validated
#: ``SafetyLayer.emergency_fallback``). Used as a SEARCH LEVEL only, so the guard
#: can always reproduce the value the rest of the project treats as conservative.
FALLBACK_GATE = 0.1

#: The flow quantities are reported in the authoritative network's own unit.
FLOW_UNIT = "MCM/day"


@dataclass
class DownstreamCapacityResult:
    """Outcome of the downstream-capacity boundary for one decision."""

    status: str = DOWNSTREAM_STATUS_NOT_APPLIED_MPC_BLOCKED
    capacity_mcm_day: float = 0.0
    horizon_steps: int = 0
    #: True only when the boundary ran AND the applied action is within capacity.
    is_protected: bool = False
    #: True when the applied action is within capacity at every horizon step.
    capacity_achieved: bool = False
    #: True when the boundary changed the action it was given.
    modified: bool = False

    #: Worst predicted downstream flow over the horizon, per action.
    proposed_predicted_flow_mcm_day: Optional[float] = None
    predicted_flow_mcm_day: Optional[float] = None
    #: Full predicted traces (MCM/day), one value per horizon step.
    proposed_trajectory_mcm_day: List[float] = field(default_factory=list)
    trajectory_mcm_day: List[float] = field(default_factory=list)
    #: Legacy field names: best peak TESTED, never a continuous-domain lower bound.
    min_achievable_flow_mcm_day: Optional[float] = None
    min_achievable_action_fraction: Dict[str, float] = field(default_factory=dict)

    #: The action the boundary was given (the SafetyLayer's output).
    proposed_action_fraction: Dict[str, float] = field(default_factory=dict)
    #: The action this boundary returns — FINAL_SAFE_CONTROL_ACTION.
    action_fraction: Dict[str, float] = field(default_factory=dict)
    candidates_evaluated: int = 0
    reason: str = ""
    tolerance_mcm_day: float = TOLERANCE_MCM_DAY
    unit: str = FLOW_UNIT
    physics: str = "ReservoirNetwork (authoritative)"

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe provenance for the API / WebSocket / Digital Twin."""
        return {
            "status": self.status,
            "active": self.status in (
                DOWNSTREAM_STATUS_PROTECTED,
                DOWNSTREAM_STATUS_CORRECTED,
                DOWNSTREAM_STATUS_FAILED_CLOSED,
            ),
            "is_protected": self.is_protected,
            "capacity_mcm_day": self.capacity_mcm_day,
            "horizon_steps": self.horizon_steps,
            "proposed_predicted_flow_mcm_day": self.proposed_predicted_flow_mcm_day,
            "predicted_flow_mcm_day": self.predicted_flow_mcm_day,
            "min_achievable_flow_mcm_day": self.min_achievable_flow_mcm_day,
            "feasibility_scope": "finite tested candidates; not a global infeasibility proof",
            "capacity_achieved": self.capacity_achieved,
            "modified": self.modified,
            "candidates_evaluated": self.candidates_evaluated,
            "reason": self.reason,
            "tolerance_mcm_day": self.tolerance_mcm_day,
            "unit": self.unit,
            "physics": self.physics,
        }


class DownstreamCapacityGuard:
    """
    Deterministic downstream-capacity safety boundary.

    Parameters
    ----------
    gate_levels : sequence of float, optional
        The validated MPC's gate lattice. ``LiveMPCOrchestrator`` passes its own
        ``MPCConfig.gate_levels``, so the boundary searches the same action space
        the controller searched and never invents a new one.
    tolerance : float
        Capacity comparison tolerance in MCM/day.
    fallback_gate : float
        The project's conservative gate value, added as a search level.
    """

    def __init__(
        self,
        gate_levels: Optional[Sequence[float]] = None,
        tolerance: float = TOLERANCE_MCM_DAY,
        fallback_gate: float = FALLBACK_GATE,
    ):
        self.mpc_gate_levels: List[float] = [
            float(g) for g in (gate_levels if gate_levels is not None
                               else [0.0, 0.15, 0.30, 0.50, 0.70, 1.0])
        ]
        self.tolerance = float(tolerance)
        self.fallback_gate = float(fallback_gate)
        #: Measured latency of the last evaluation (milliseconds).
        self.last_latency_ms: float = 0.0

    # ------------------------------------------------------------------
    # horizon
    # ------------------------------------------------------------------

    @staticmethod
    def horizon_for(network: ReservoirNetwork) -> int:
        """
        Prediction horizon in steps: ``1 + sum(routing delays)``.

        One step for the action about to be applied plus one step per unit of
        cumulative routing delay, so the window covers the full propagation of
        that action from the top of the cascade (A) to the river below the
        terminal reservoir (D). Derived from the authoritative topology.
        """
        return 1 + sum(int(conn.delay) for conn in network.connections)

    # ------------------------------------------------------------------
    # prediction — authoritative physics only
    # ------------------------------------------------------------------

    def predict_flows(
        self,
        network: ReservoirNetwork,
        action: Dict[str, float],
        inflows: Dict[str, float],
        node_ids: Sequence[str],
        horizon: int,
    ) -> List[float]:
        """
        Predict the downstream flow (MCM/day) for ``horizon`` steps.

        A **clone** of the live ``ReservoirNetwork`` is stepped, so the validated
        routing delays, attenuation factors, spill rules and mass balance are the
        ones that will actually be applied. The live network is never mutated.
        """
        clone = ReservoirNetwork(config_dict=network._raw_config, emit_warnings=False)
        for nid in node_ids:
            clone.nodes[nid].state.storage = float(network.nodes[nid].state.storage)
        # REAL water already in transit — this is what makes the check respect
        # the cascade rather than today's gate positions alone.
        for index, conn in enumerate(network.connections):
            clone.connections[index].queue = copy.deepcopy(conn.queue)

        terminal_id = clone._terminal_node_id
        gates = {nid: float(action[nid]) for nid in node_ids}
        scenarios = inflows if isinstance(inflows, list) else [inflows]
        flows: List[float] = []
        for day in range(max(1, horizon)):
            local = scenarios[min(day, len(scenarios) - 1)]
            states = clone.step({nid: float(local.get(nid, 0.0)) for nid in node_ids}, dict(gates))
            flows.append(float(states[terminal_id].total_outflow))
        return flows

    # ------------------------------------------------------------------
    # feasibility w.r.t. the validated SafetyLayer
    # ------------------------------------------------------------------

    def safety_layer_feasible(
        self,
        action: Dict[str, float],
        node_ids: Sequence[str],
        current: Dict[str, float],
        max_gate_change: float,
    ) -> bool:
        """
        True when the validated SafetyLayer would pass ``action`` UNCHANGED.

        The SafetyLayer's two rules are gate bounds ``[0, 1]`` and the per-step
        rate limit. Constraining this boundary's output to actions that satisfy
        both is what lets it sit *after* the SafetyLayer without invalidating the
        guarantee that precedes it: the final action is admissible under BOTH, so
        the safety properties compose instead of the later layer silently
        overriding the earlier one.
        """
        for nid in node_ids:
            try:
                gate = float(action[nid])
            except (KeyError, TypeError, ValueError):
                return False
            if not math.isfinite(gate) or gate < 0.0 or gate > 1.0:
                return False
            if abs(gate - float(current[nid])) > max_gate_change + self.tolerance:
                return False
        return True

    # ------------------------------------------------------------------
    # candidate generation
    # ------------------------------------------------------------------

    def candidate_levels_for(
        self,
        network: ReservoirNetwork,
        node_id: str,
        proposal_gate: float,
        current_gate: float,
    ) -> List[float]:
        """
        Gate values tried for ONE reservoir, in ascending order.

        * the validated MPC's own lattice (never a new action space),
        * ``0.0`` and the project's conservative value (``FALLBACK_GATE``),
        * the gate at which this reservoir's release equals the downstream
          capacity (``capacity / max_release``) — the analytic threshold below
          which this reservoir cannot by itself exceed the capacity,
        * **the proposal and the current gate**, so the guard is always able to
          leave a reservoir exactly as the SafetyLayer produced it (or exactly as
          it is) instead of gratuitously moving reservoirs that are not part of
          the problem.
        """
        capacity = float(network.downstream_capacity)
        node = network.nodes[node_id]
        levels = {0.0, float(self.fallback_gate)}
        levels.update(self.mpc_gate_levels)
        if node.max_release > 0:
            levels.add(max(0.0, min(1.0, capacity / float(node.max_release))))
        for value in (proposal_gate, current_gate):
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                levels.add(max(0.0, min(1.0, value)))
        return sorted(levels)

    # ------------------------------------------------------------------
    # evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        network: ReservoirNetwork,
        *,
        action_fraction: Dict[str, float],
        current_fraction: Dict[str, float],
        node_ids: Sequence[str],
        max_gate_change: float,
        inflows: Optional[Dict[str, float]] = None,
        gate_upper_bounds: Optional[Dict[str, float]] = None,
    ) -> DownstreamCapacityResult:
        """
        Run the boundary.

        ``action_fraction`` is the action proposed by the MPC **after** the
        validated SafetyLayer. The returned ``action_fraction`` is the action that
        may be applied to ``ReservoirNetwork``: ``FINAL_SAFE_CONTROL_ACTION``.
        """
        started = time.perf_counter()
        node_ids = list(node_ids)
        capacity = float(network.downstream_capacity)
        horizon = max(self.horizon_for(network), len(inflows) if isinstance(inflows, list) else 0)

        if inflows is None:
            inflows = {nid: float(network.nodes[nid].state.inflow_local) for nid in node_ids}

        proposal = {nid: float(action_fraction[nid]) for nid in node_ids}
        # AUTO reserve ceilings must survive every downstream replacement.
        if gate_upper_bounds is not None:
            for nid in node_ids:
                limit = gate_upper_bounds[nid]
                if not math.isfinite(limit) or not 0 <= limit <= 1:
                    raise ValueError('Invalid reserve gate ceiling')
                if proposal[nid] > limit + 1e-11:
                    raise RuntimeError('Downstream proposal exceeds AUTO reserve ceiling')
                if max(0., current_fraction[nid] - max_gate_change) > limit + 1e-11:
                    raise RuntimeError('AUTO reserve conflicts with gate movement limit')

        result = DownstreamCapacityResult(
            capacity_mcm_day=capacity,
            horizon_steps=horizon,
            proposed_action_fraction=dict(proposal),
            action_fraction=dict(proposal),
        )

        if not math.isfinite(capacity) or capacity <= 0.0:
            result.status = DOWNSTREAM_STATUS_NOT_APPLIED_NO_CAPACITY
            result.reason = (
                "the authoritative network declares no downstream capacity; no "
                "downstream-capacity guarantee is claimed"
            )
            self.last_latency_ms = (time.perf_counter() - started) * 1000.0
            return result

        # ---- 1. predict the action the MPC + SafetyLayer proposed ----
        try:
            proposed_trace = self.predict_flows(
                network, proposal, inflows, node_ids, horizon
            )
        except Exception as exc:  # pragma: no cover - defensive
            result.status = DOWNSTREAM_STATUS_FAILED_CLOSED
            result.action_fraction = self.minimum_admissible_action(
                node_ids, current_fraction, max_gate_change
            )
            result.modified = True
            result.reason = (
                f"PREDICTION_ERROR:{type(exc).__name__} — the downstream flow could "
                f"not be predicted, so the minimum admissible action was applied and "
                f"no capacity guarantee is claimed"
            )
            self.last_latency_ms = (time.perf_counter() - started) * 1000.0
            return result

        if not proposed_trace or not all(math.isfinite(v) and v >= 0 for v in proposed_trace):
            raise ValueError("Invalid downstream prediction; simulation action must not advance")
        result.proposed_trajectory_mcm_day = list(proposed_trace)
        result.proposed_predicted_flow_mcm_day = max(proposed_trace)
        result.trajectory_mcm_day = list(proposed_trace)
        result.predicted_flow_mcm_day = max(proposed_trace)

        if result.proposed_predicted_flow_mcm_day <= capacity + self.tolerance:
            # ---- 2. already safe: apply UNCHANGED ----
            result.status = DOWNSTREAM_STATUS_PROTECTED
            result.is_protected = True
            result.capacity_achieved = True
            result.modified = False
            result.reason = (
                f"predicted downstream flow {result.predicted_flow_mcm_day:.6f} "
                f"{FLOW_UNIT} <= capacity {capacity:.6f} {FLOW_UNIT} over "
                f"{horizon} step(s); action applied unchanged"
            )
            self.last_latency_ms = (time.perf_counter() - started) * 1000.0
            return result

        # ---- 3. Evaluate the lowest-gate corner as one candidate, not a proof. ----
        minimum_action = self.minimum_admissible_action(
            node_ids, current_fraction, max_gate_change
        )
        min_trace: Optional[List[float]]
        try:
            min_trace = self.predict_flows(
                network, minimum_action, inflows, node_ids, horizon
            )
        except Exception:  # pragma: no cover - defensive
            min_trace = None

        if min_trace is not None:
            result.min_achievable_flow_mcm_day = max(min_trace)
            result.min_achievable_action_fraction = dict(minimum_action)

        # Storage headroom makes multi-day peaks non-monotonic. Search even
        # when the all-minimum corner spills above capacity.
        safe = self._nearest_safe_action(
            network, proposal, current_fraction, node_ids, max_gate_change,
            inflows, horizon, capacity,
            gate_upper_bounds,
        )
        if safe is not None:
            action, trace, evaluated = safe
            result.candidates_evaluated = evaluated
            if max(trace) > capacity + self.tolerance:
                minimum_action, min_trace = action, trace
                result.min_achievable_flow_mcm_day = max(trace)
                result.min_achievable_action_fraction = dict(action)
                safe = None
        if safe is not None:
            action, trace, evaluated = safe
            result.status = DOWNSTREAM_STATUS_CORRECTED
            if result.min_achievable_flow_mcm_day is None or max(trace) < result.min_achievable_flow_mcm_day:
                result.min_achievable_flow_mcm_day = max(trace)
                result.min_achievable_action_fraction = dict(action)
            result.action_fraction = action
            result.trajectory_mcm_day = list(trace)
            result.predicted_flow_mcm_day = max(trace)
            result.capacity_achieved = True
            result.is_protected = True
            result.modified = True
            result.candidates_evaluated = evaluated
            result.reason = (
                f"proposed action predicted "
                f"{result.proposed_predicted_flow_mcm_day:.6f} {FLOW_UNIT} > capacity "
                f"{capacity:.6f} {FLOW_UNIT}; replaced with the nearest admissible "
                f"downstream-safe action (predicted "
                f"{result.predicted_flow_mcm_day:.6f} {FLOW_UNIT})"
            )
            self.last_latency_ms = (time.perf_counter() - started) * 1000.0
            return result

        # No safe finite-grid candidate was found. Report the shortfall honestly.
        result.status = DOWNSTREAM_STATUS_FAILED_CLOSED
        result.action_fraction = dict(minimum_action)
        result.trajectory_mcm_day = list(min_trace) if min_trace is not None else []
        result.predicted_flow_mcm_day = (
            max(min_trace) if min_trace is not None else None
        )
        result.modified = True
        result.capacity_achieved = False
        result.is_protected = False
        result.reason = (
            "the downstream-safe action search did not find an admissible action; "
            "the least-peak tested admissible action was applied; this finite search "
            "does not prove global infeasibility and no capacity guarantee "
            "is claimed"
        )
        self.last_latency_ms = (time.perf_counter() - started) * 1000.0
        return result

    # ------------------------------------------------------------------
    # search + minimum action
    # ------------------------------------------------------------------

    def _nearest_safe_action(
        self,
        network: ReservoirNetwork,
        proposal: Dict[str, float],
        current: Dict[str, float],
        node_ids: List[str],
        max_gate_change: float,
        inflows: Dict[str, float],
        horizon: int,
        capacity: float,
        gate_upper_bounds: Optional[Dict[str, float]] = None,
    ) -> Optional[Tuple[Dict[str, float], List[float], int]]:
        """
        Deterministically find the admissible action closest to the proposal.

        The candidate set is the validated MPC's lattice augmented per reservoir
        (see ``candidate_levels_for``), filtered to actions the SafetyLayer would
        accept. Candidates are tried in order of increasing L1 distance from the
        proposal, ties broken by enumeration order.
        """
        order = list(node_ids)
        per_node_levels = [
            sorted(set(self.candidate_levels_for(network, nid, proposal[nid], current[nid]))
                   | {max(0.0, current[nid] - max_gate_change),
                      min(1.0, current[nid] + max_gate_change)})
            for nid in order
        ]
        if gate_upper_bounds is not None:
            per_node_levels = [sorted(g for g in set(levels) | {gate_upper_bounds[nid]}
                                      if g <= gate_upper_bounds[nid] + 1e-11)
                               for nid, levels in zip(order, per_node_levels)]
        candidates = [
            combo for combo in itertools.product(*per_node_levels)
            if self.safety_layer_feasible(dict(zip(order, combo)), order,
                                          current, max_gate_change)
        ]
        scored = sorted(
            range(len(candidates)),
            key=lambda i: (
                sum(abs(candidates[i][j] - float(proposal[nid]))
                    for j, nid in enumerate(order)),
                i,
            ),
        )

        evaluated = 0
        best = None
        for index in scored:
            action = {nid: float(candidates[index][j]) for j, nid in enumerate(order)}
            evaluated += 1
            try:
                trace = self.predict_flows(network, action, inflows, order, horizon)
            except Exception:  # pragma: no cover - defensive
                continue
            if best is None or max(trace) < max(best[1]):
                best = (action, trace)
            if max(trace) <= capacity + self.tolerance:
                return action, trace, evaluated
        return (*best, evaluated) if best is not None else None

    @staticmethod
    def minimum_admissible_action(
        node_ids: Sequence[str],
        current: Dict[str, float],
        max_gate_change: float,
    ) -> Dict[str, float]:
        """Lowest reachable gates; NOT a bound on future downstream peaks."""
        return {
            nid: max(0.0, min(1.0, float(current.get(nid, 0.0)) - max_gate_change))
            for nid in node_ids
        }
