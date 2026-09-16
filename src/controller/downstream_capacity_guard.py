"""
Stage 10 — Downstream Capacity Safety Boundary
==============================================

The **final software safety boundary** on the live control path:

    MPCController.decide()
        ↓  raw proposal
    SafetyLayer.validate()            (validated Phase 15.3 — UNMODIFIED)
        ↓  gate bounds + per-step rate limit
    DownstreamCapacityGuard.evaluate()      ← THIS MODULE (Stage 10)
        ↓  FINAL_SAFE_CONTROL_ACTION
    ReservoirNetwork.step()

WHY THIS MODULE EXISTS
----------------------
Stage 8 established, by reading ``src/controller/safety.py``, that the validated
SafetyLayer does **NOT** check downstream capacity — its class docstring claims a
"downstream capacity (estimated from proposed releases)" check that has no
backing code. The validated MPC does not enforce it either: in
``src/controller/objective.py`` a downstream excursion is a **soft penalty**
(``ObjectiveWeights.downstream_violation = 200.0`` per MCM/day above the limit),
so a proposal that would flood the river below the terminal reservoir can still be
the lowest-cost candidate and can still be returned by ``MPCController.decide()``.

Neither of those components may be modified (both are frozen Phase 15.3
artifacts). The boundary is therefore added **downstream of them, inside the one
authoritative live controller path** (``LiveMPCOrchestrator``), which is the only
place that holds (a) the live network state, (b) the proposed action and (c) the
authority to change what is applied.

WHAT "DOWNSTREAM FLOW" MEANS HERE — the authoritative definition
----------------------------------------------------------------
``ReservoirNetwork`` computes the flow below the cascade as the **terminal
reservoir's total outflow** (controlled release + spill) and warns when it exceeds
``network.downstream_capacity``. ``ObjectiveFunction.evaluate_trajectory`` scores
violations against exactly the same quantity. This module uses that same
definition, so the boundary, the physics and the controller's own cost function
all agree on what "downstream flow" is. Water spilled from non-terminal
reservoirs leaves the network and is *not* part of the downstream flow (verified
in Stage 9).

THE PREDICTION USES THE AUTHORITATIVE PHYSICS
---------------------------------------------
No routing equation is duplicated here. The prediction clones the live
``ReservoirNetwork`` — the validated delays, attenuation factors, spill rules and
mass balance — copies the REAL storage and the REAL routing queues, and steps it.
Water already in transit is therefore part of the prediction, and the cascade
coupling of the candidate action over the horizon is produced by the validated
model itself.

THE HORIZON
-----------
``horizon = 1 + sum(routing delays)`` steps — one step for the action being
applied plus one step per unit of cumulative routing delay (2 + 1 + 1 = 4 in the
validated topology → 5 steps). That covers the full propagation of the applied
action from Reservoir A to the river below Reservoir D. It is derived from the
topology, never hardcoded.

Gate positions and exogenous inflows are held constant across the horizon — the
*same* convention the validated MPC uses for its own lookahead
(``MPCController._simulate_trajectory``: "hold gates constant across lookahead"),
and conservative for a safety boundary because it assumes the action persists.
Only the first step is ever applied, and this guard re-runs on every decision.

INFLOWS
-------
Exogenous inflows are held at the values the caller supplies — the live controller
passes the inflows that will actually be applied in the step — and default, when
absent, to the network's currently observed local inflows. Inflows are
deliberately NOT taken from the forecast: a safety boundary must not depend on the
quality of a forecast whose validity the Stage 7 provenance gate exists to police.
Water already committed to the routing queues is real state and IS included, so
arriving flood waves are seen.

UNITS
-----
Gate positions are canonical FRACTIONS in [0, 1]. Downstream capacity and
predicted flows are MCM/day — the authoritative network's own unit. Nothing is
converted here and no implicit conversion is introduced.

ACTION POLICY (deterministic, documented)
-----------------------------------------
1. **Safe proposal** → applied UNCHANGED (``PROTECTED``, ``modified = False``).
2. **Unsafe proposal, safe alternative exists** → the admissible action *closest*
   to the proposal is applied (``CORRECTED``). Candidates are the validated MPC's
   own gate lattice, augmented per reservoir with values that let the guard
   *keep* what it was given (the proposal and the current gate), the project's
   conservative value, and the analytically-derived gate at which a reservoir's
   release equals the capacity. Candidates are tried in order of increasing L1
   distance from the proposal, ties broken by enumeration order, so the
   intervention is minimal and the result deterministic.
3. **No admissible action can satisfy the capacity** → ``FAILED_CLOSED``. The
   minimum admissible action is applied and the shortfall is REPORTED, never
   hidden.

WHY THE MINIMUM ACTION IS A PROOF, NOT A GUESS
----------------------------------------------
Per reservoir, the terminal reservoir's ``total_outflow`` is
``max(release, available - capacity)``. That quantity is **non-decreasing** in the
terminal gate (a larger gate means a larger release) and non-decreasing in every
upstream gate (a larger upstream release means more water arriving at the terminal
reservoir, hence a larger ``available``). Downstream flow is therefore monotone
non-decreasing in all four gates, so the flow-minimising action over the
admissible box ``[current - max_gate_change, current + max_gate_change] ∩ [0, 1]``
is the all-minimum corner. If that action already predicts a flow above the
capacity, **no admissible action can avoid it** and the excess is forced
(unavoidable spill at the terminal reservoir). The guard therefore evaluates the
minimum action *before* searching, which both bounds the search (a safe action is
then guaranteed to exist) and makes the ``FAILED_CLOSED`` verdict provable.

Note on the minimum environmental flow: the demo config carries
``minimum_environmental_flow_percent = 0.05`` as a SIMULATION ASSUMPTION that is
**not enforced anywhere** in the validated code. The fail-closed action may close
the gates more than that; flood safety is this boundary's objective and the state
is reported loudly (``FAILED_CLOSED``, ``capacity_achieved = False``) rather than
traded away silently.
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
#: The check ran and PROVED that no admissible action satisfies the capacity; the
#: minimum admissible (flow-minimising) action was applied and the shortfall is
#: reported.
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
    #: The lowest flow any admissible action could reach, and the action that
    #: reaches it (computed whenever the proposal is unsafe).
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
        clone = ReservoirNetwork(config_dict=copy.deepcopy(network._raw_config))
        for nid in node_ids:
            clone.nodes[nid].state.storage = float(network.nodes[nid].state.storage)
        # REAL water already in transit — this is what makes the check respect
        # the cascade rather than today's gate positions alone.
        for index, conn in enumerate(network.connections):
            clone.connections[index].queue = copy.deepcopy(conn.queue)

        terminal_id = clone._terminal_node_id
        gates = {nid: float(action[nid]) for nid in node_ids}
        constant_inflows = {nid: float(inflows.get(nid, 0.0)) for nid in node_ids}

        flows: List[float] = []
        for _ in range(max(1, horizon)):
            states = clone.step(dict(constant_inflows), dict(gates))
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
        horizon = self.horizon_for(network)

        if inflows is None:
            inflows = {nid: float(network.nodes[nid].state.inflow_local) for nid in node_ids}

        proposal = {nid: float(action_fraction[nid]) for nid in node_ids}

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

        # ---- 3. flow-minimising action: is the capacity achievable AT ALL? ----
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

        if min_trace is not None and result.min_achievable_flow_mcm_day > capacity + self.tolerance:
            # PROVEN: downstream flow is monotone non-decreasing in every gate, so
            # no admissible action can do better than the flow-minimising corner.
            result.status = DOWNSTREAM_STATUS_FAILED_CLOSED
            result.action_fraction = dict(minimum_action)
            result.trajectory_mcm_day = list(min_trace)
            result.predicted_flow_mcm_day = result.min_achievable_flow_mcm_day
            result.capacity_achieved = False
            result.is_protected = False
            result.modified = True
            result.reason = (
                f"NO ADMISSIBLE ACTION CAN LIMIT THE DOWNSTREAM FLOW: the "
                f"flow-minimising admissible action (every gate at its lowest "
                f"reachable value) still predicts "
                f"{result.min_achievable_flow_mcm_day:.6f} {FLOW_UNIT} > capacity "
                f"{capacity:.6f} {FLOW_UNIT}. Downstream flow is non-decreasing in "
                f"every gate, so this is the minimum achievable; the excess is forced "
                f"spill at the terminal reservoir. That action was applied and no "
                f"capacity guarantee is claimed."
            )
            self.last_latency_ms = (time.perf_counter() - started) * 1000.0
            return result

        # ---- 4. a safe action is now KNOWN to exist: find the nearest one ----
        safe = self._nearest_safe_action(
            network, proposal, current_fraction, node_ids, max_gate_change,
            inflows, horizon, capacity,
        )
        if safe is not None:
            action, trace, evaluated = safe
            result.status = DOWNSTREAM_STATUS_CORRECTED
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

        # Unreachable in practice (step 3 guarantees a safe action exists when the
        # minimum action is safe). Kept so the boundary can never fail OPEN.
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
            "the minimum admissible action was applied and no capacity guarantee "
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
            self.candidate_levels_for(network, nid, proposal[nid], current[nid])
            for nid in order
        ]
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
        for index in scored:
            action = {nid: float(candidates[index][j]) for j, nid in enumerate(order)}
            evaluated += 1
            try:
                trace = self.predict_flows(network, action, inflows, order, horizon)
            except Exception:  # pragma: no cover - defensive
                continue
            if max(trace) <= capacity + self.tolerance:
                return action, trace, evaluated
        return None

    @staticmethod
    def minimum_admissible_action(
        node_ids: Sequence[str],
        current: Dict[str, float],
        max_gate_change: float,
    ) -> Dict[str, float]:
        """
        The flow-minimising admissible action: every gate at its lowest reachable
        value, ``clip(current - max_gate_change, 0, 1)``.

        Because downstream flow is non-decreasing in every gate, this action
        minimises the predicted downstream flow over the whole admissible box —
        so its predicted flow is a *lower bound* on what any admissible action can
        achieve. That is what makes the ``FAILED_CLOSED`` verdict provable.
        """
        return {
            nid: max(0.0, min(1.0, float(current.get(nid, 0.0)) - max_gate_change))
            for nid in node_ids
        }
