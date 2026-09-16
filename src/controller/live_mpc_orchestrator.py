"""
Stage 7 — Live MPC Orchestration Boundary
=========================================

Integrates the **validated Phase 15.3 MPC** into the live authoritative
simulation path, behind an explicit provenance gate.

    Frozen LSTM V3
        ↓
    V3 Forecast Adapter / LiveForecastAdapter   (Stage 6)
        ↓
    NetworkForecastSnapshot
        ↓
    PROVENANCE GATE                             <-- THIS MODULE
        ↓
    MPCController                               (validated, UNMODIFIED)
        ↓
    SafetyLayer                                 (validated, UNMODIFIED)  [STAGE 8]
        ↓
    DownstreamCapacityGuard                     (NEW, Stage 10)
        ↓  FINAL_SAFE_CONTROL_ACTION
    ReservoirNetwork
        ↓
    FastAPI / WebSocket
        ↓
    Three.js Digital Twin

THE CRITICAL RULE
-----------------
A forecast may drive the MPC **only** when its provenance says
``validated_metrics_apply == True``.

Demonstration forecasts (``DEMONSTRATION_ONLY``, synthetic placeholder inputs,
warm-up, unavailable, non-finite, wrong units, missing reservoirs) are
**BLOCKED**. When blocked this module:

  * does NOT call the MPC,
  * does NOT run the SafetyLayer to "rescue" the situation — the SafetyLayer
    must never manufacture a control decision,
  * does NOT fabricate, clamp, carry-forward or substitute a forecast value,
  * returns an explicit ``controller_status = BLOCKED`` decision with the exact
    reasons, and
  * holds the CURRENT gate positions (read from the authoritative network) and
    marks ``control_applied = False``, so nothing pretends an MPC decision was
    made.

THE SAFETY BOUNDARY (STAGE 8)
-----------------------------
When the MPC IS eligible, its **raw proposal** is passed through the existing
validated ``SafetyLayer`` (``src/controller/safety.py``) and the layer's OUTPUT —
never the raw proposal — is what reaches ``ReservoirNetwork``.

For exactly what that layer does and does not guarantee, see
``docs``-style notes in ``SAFETY_LAYER_GUARANTEES`` below. In short it
validates: node presence, finite numeric values, gate bounds [0,1] and
per-step gate rate limiting. It does NOT implement release bounds, downstream
capacity limits, storage-headroom constraints or minimum-flow rules — those are
reported as gaps, not silently added.

THE DOWNSTREAM CAPACITY BOUNDARY (STAGE 10)
-------------------------------------------
The validated SafetyLayer does **not** enforce downstream capacity (Stage 8
established this by reading its source), and the validated MPC prices a
downstream excursion only as a soft penalty in its objective. Neither may be
modified. So the action that survived the SafetyLayer is passed through
``DownstreamCapacityGuard`` (``src/controller/downstream_capacity_guard.py``),
which predicts the flow below the terminal reservoir on a clone of the
authoritative ``ReservoirNetwork`` and either

  * applies the action unchanged (``PROTECTED``),
  * replaces it with the nearest SafetyLayer-feasible action whose predicted flow
    stays within capacity (``CORRECTED``), or
  * fails closed with the conservative minimum-release action and REPORTS that the
    capacity could not be achieved (``FAILED_CLOSED``).

The three statuses are exposed SEPARATELY (``controller_status``,
``safety_layer_status``, ``downstream_status``) so the live system can never
claim "SafetyLayer ACTIVE = downstream capacity protected".

WHAT THIS MODULE DOES NOT DO
----------------------------
* It does NOT rewrite, tune or re-implement the MPC — it calls
  ``MPCController.decide()`` exactly as Phase 15.3 does.
* It does NOT re-implement the SafetyLayer — it calls
  ``SafetyLayer.validate()`` exactly as Phase 15.3 does.
* It does NOT let the GNN, the browser, or a rule-based controller make
  authoritative gate decisions.

SAFETY_LAYER_GUARANTEES — what the validated layer ACTUALLY checks
------------------------------------------------------------------
Verified by reading ``src/controller/safety.py`` (not assumed):

  1. ALL REQUIRED NODES PRESENT — a node missing from the proposal is replaced
     with its CURRENT gate (``current_gates.get(nid, 0.1)``).
  2. FINITE NUMERIC VALUES — a non-numeric / NaN / Inf gate is replaced by 0.1.
  3. GATE BOUNDS — finite gates are clamped into [0.0, 1.0].
  4. RATE LIMIT — a change larger than ``max_gate_change_per_step`` is
     truncated to the limit, then re-clamped into [0.0, 1.0].

NOT IMPLEMENTED in the validated layer (reported as gaps, deliberately NOT
added here): release bounds, downstream capacity constraints, storage-headroom
or reservoir safety constraints, minimum environmental flow, prior-state
validation, and detection of an impossible state. The module docstring's claim
of a "downstream capacity" check is not backed by code.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from src.common import units

from ..network_env.live_forecast_adapter import (
    FORECAST_UNIT,
    HORIZON_LABELS,
    LiveForecastBundle,
    parse_provenance_string,
    provenance_from_snapshot,
)
from ..network_env.v3_forecast_adapter import (
    ForecastStatus,
    NetworkForecastSnapshot,
)
from .mpc_controller import MPCController
from .safety import SafetyLayer
from .downstream_capacity_guard import (
    DOWNSTREAM_STATUS_FAILED_CLOSED,
    DOWNSTREAM_STATUS_NOT_APPLIED_ADAPTER_ERROR,
    DOWNSTREAM_STATUS_NOT_APPLIED_MPC_BLOCKED,
    DOWNSTREAM_STATUS_NOT_APPLIED_MPC_ERROR,
    DownstreamCapacityGuard,
)


class ControllerStatus(str, Enum):
    """Whether the MPC produced the authoritative decision."""
    ACTIVE = "ACTIVE"          # MPC ran and its gates are authoritative
    BLOCKED = "BLOCKED"        # provenance/validity gate refused the forecast
    UNAVAILABLE = "UNAVAILABLE"  # MPC could not be constructed or run


#: The SafetyLayer implementation used here is the validated Phase 15.3 layer
#: (``src/controller/safety.py``), NOT a re-implementation.
SAFETY_LAYER_VERSION = "PHASE_15_3_VALIDATED"

#: Statuses reported for the SafetyLayer boundary.
SAFETY_STATUS_SAFE = "SAFE"                    # ran, no violations
SAFETY_STATUS_CORRECTED = "CORRECTED"          # ran, modified the proposal
SAFETY_STATUS_EMERGENCY = "EMERGENCY"          # layer's own emergency status
SAFETY_STATUS_FALLBACK = "FALLBACK"
SAFETY_STATUS_NOT_APPLIED_MPC_BLOCKED = "NOT_APPLIED_MPC_BLOCKED"
SAFETY_STATUS_NOT_APPLIED_MPC_ERROR = "NOT_APPLIED_MPC_ERROR"
SAFETY_STATUS_NOT_APPLIED_ADAPTER_ERROR = "NOT_APPLIED_ADAPTER_ERROR"

#: Backwards-compatible alias (Stage 7 used this name).
SAFETY_LAYER_STATUS = SAFETY_STATUS_NOT_APPLIED_MPC_BLOCKED


@dataclass
class LiveControlDecision:
    """
    The single authoritative live control record.

    Carries both the decision and enough provenance for the API/UI to state
    exactly what happened, without ambiguity.
    """
    controller_type: str = "MPC"
    controller_status: str = ControllerStatus.BLOCKED.value
    forecast_control_eligible: bool = False
    forecast_provenance: Dict[str, Any] = field(default_factory=dict)

    # ── STAGE 8 — the SafetyLayer boundary ────────────────────────────────
    #: The validated Phase 15.3 SafetyLayer is now IN the live path.
    safety_layer_integrated: bool = True
    safety_layer_version: str = SAFETY_LAYER_VERSION
    safety_layer_status: str = SAFETY_STATUS_NOT_APPLIED_MPC_BLOCKED
    safety_is_safe: bool = False
    safety_violations: List[str] = field(default_factory=list)
    #: True when the SafetyLayer changed the MPC's proposal.
    safety_modified: bool = False

    #: External representation (percent) — what the live cascade consumes.
    #: When ACTIVE these are the SAFETY-VALIDATED gates, never the raw proposal.
    gate_positions_pct: Dict[str, float] = field(default_factory=dict)
    #: Canonical internal representation (fraction), as returned by the MPC.
    gate_positions_fraction: Dict[str, float] = field(default_factory=dict)

    #: The MPC's raw proposal BEFORE the SafetyLayer (audit / UI comparison).
    #: Finitely-valued entries only; an entry the SafetyLayer had to reject
    #: (NaN/Inf/non-numeric) is recorded as ``None`` here — its exact repr is
    #: preserved in ``safety_violations``. This copy is deliberately NOT routed
    #: through the unit boundary, because a non-gate value has no meaning as a
    #: percentage (and the boundary would correctly raise on NaN).
    proposed_gate_positions_fraction: Dict[str, Optional[float]] = field(default_factory=dict)

    #: ── STAGE 9 — COORDINATION PROVENANCE (read-only) ─────────────────────
    #: Describes the MPC's candidate/action space as OBSERVED from the validated
    #: controller's own configuration: its dimensionality, which reservoirs make
    #: up each candidate vector, and how many complete vectors were scored.
    #: Purely descriptive — the MPC algorithm is not modified, and this block
    #: never influences the decision.
    action_space: Dict[str, Any] = field(default_factory=dict)
    #: The SafetyLayer's OUTPUT (Stage 8), i.e. what the downstream-capacity
    #: boundary was given. Completes the audit chain:
    #: ``proposed_gate_positions_fraction`` (raw MPC) -> this -> applied gates.
    safety_layer_gate_positions_fraction: Dict[str, float] = field(default_factory=dict)
    safety_layer_gate_positions_pct: Dict[str, float] = field(default_factory=dict)
    #: ── STAGE 10 — DOWNSTREAM CAPACITY BOUNDARY ─────────────────────────────-
    #: Reported SEPARATELY from the MPC and the SafetyLayer. ``downstream_status``
    #: is one of PROTECTED / CORRECTED / FAILED_CLOSED / NOT_APPLIED_*.
    downstream_status: str = DOWNSTREAM_STATUS_NOT_APPLIED_MPC_BLOCKED
    downstream_capacity_mcm_day: float = 0.0
    downstream_predicted_flow_mcm_day: Optional[float] = None
    downstream_proposed_predicted_flow_mcm_day: Optional[float] = None
    downstream_min_achievable_flow_mcm_day: Optional[float] = None
    downstream_capacity_achieved: bool = False
    downstream_protection_modified: bool = False
    downstream_horizon_steps: int = 0
    downstream_candidates_evaluated: int = 0
    downstream_reason: str = ""
    #: Full provenance block (see ``DownstreamCapacityResult.to_dict()``).
    downstream_capacity_protection: Dict[str, Any] = field(default_factory=dict)
    #: True when the final action survived a re-validation by the SafetyLayer, i.e.
    #: the boundary did not invalidate any guarantee that precedes it.
    applied_action_safety_layer_clean: bool = True
    #: Measured latency of the downstream boundary (milliseconds).
    downstream_latency_ms: float = 0.0

    #: ── THE ONE ACTION ELIGIBLE TO CROSS A FUTURE HARDWARE BOUNDARY ─────────
    #: ``FINAL_SAFE_CONTROL_ACTION``: the four gate positions that have passed the
    #: provenance gate, the validated SafetyLayer AND the DownstreamCapacityGuard.
    #: Identical to ``gate_positions_pct`` / ``gate_positions_fraction`` (pinned by
    #: test) and set even when the MPC is blocked, in which case it is the held
    #: current gates and ``control_applied`` is False.
    final_safe_control_action_fraction: Dict[str, float] = field(default_factory=dict)
    final_safe_control_action_pct: Dict[str, float] = field(default_factory=dict)
    final_safe_control_action_source: str = "NOT_APPLIED"

    #: True only when the MPC actually determined these gates.
    control_applied: bool = False
    #: Set when the gate refused the forecast.
    blocked_reason: str = ""
    reasons: List[str] = field(default_factory=list)

    # MPC internals (for auditability — never used to bypass the gate)
    mpc_status: str = ""
    mpc_objective_score: Optional[float] = None
    mpc_forecast_used: bool = False
    mpc_forecast_status: str = ""
    mpc_safety_status: str = ""
    candidates_evaluated: int = 0
    per_node: Dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialisable control provenance for the API / WebSocket / UI."""
        return {
            "controller_type": self.controller_type,
            "controller_status": self.controller_status,
            "forecast_control_eligible": self.forecast_control_eligible,
            "forecast_provenance": self.forecast_provenance,
            "safety_layer_integrated": self.safety_layer_integrated,
            "safety_layer_version": self.safety_layer_version,
            "safety_layer_status": self.safety_layer_status,
            "safety_is_safe": self.safety_is_safe,
            "safety_violations": list(self.safety_violations),
            "safety_modified": self.safety_modified,
            "control_applied": self.control_applied,
            "blocked_reason": self.blocked_reason,
            "reasons": list(self.reasons),
            "gate_positions_pct": {k: float(v) for k, v in self.gate_positions_pct.items()},
            "gate_positions_fraction": {k: float(v) for k, v in self.gate_positions_fraction.items()},
            "proposed_gate_positions_fraction": {
                k: (None if v is None else float(v))
                for k, v in self.proposed_gate_positions_fraction.items()
            },
            "safety_layer_gate_positions_fraction": {
                k: float(v) for k, v in self.safety_layer_gate_positions_fraction.items()
            },
            "safety_layer_gate_positions_pct": {
                k: float(v) for k, v in self.safety_layer_gate_positions_pct.items()
            },
            "mpc_status": self.mpc_status,
            "mpc_objective_score": self.mpc_objective_score,
            "mpc_forecast_used": self.mpc_forecast_used,
            "mpc_forecast_status": self.mpc_forecast_status,
            "mpc_safety_status": self.mpc_safety_status,
            "candidates_evaluated": self.candidates_evaluated,
            "action_space": {
                "dimension": self.action_space.get("dimension"),
                "node_ids": list(self.action_space.get("node_ids", [])),
                "gate_levels": list(self.action_space.get("gate_levels", [])),
                "candidate_vectors": self.action_space.get("candidate_vectors"),
                "coordinated": self.action_space.get("coordinated"),
            },
            # ── Stage 10 — downstream capacity protection, reported SEPARATELY
            # from the MPC status and the SafetyLayer status.
            "downstream_status": self.downstream_status,
            "downstream_capacity_mcm_day": self.downstream_capacity_mcm_day,
            "downstream_predicted_flow_mcm_day": self.downstream_predicted_flow_mcm_day,
            "downstream_proposed_predicted_flow_mcm_day": self.downstream_proposed_predicted_flow_mcm_day,
            "downstream_min_achievable_flow_mcm_day": self.downstream_min_achievable_flow_mcm_day,
            "downstream_capacity_achieved": self.downstream_capacity_achieved,
            "downstream_protection_modified": self.downstream_protection_modified,
            "downstream_horizon_steps": self.downstream_horizon_steps,
            "downstream_candidates_evaluated": self.downstream_candidates_evaluated,
            "downstream_reason": self.downstream_reason,
            "downstream_capacity_protection": dict(self.downstream_capacity_protection),
            "applied_action_safety_layer_clean": self.applied_action_safety_layer_clean,
            "downstream_latency_ms": self.downstream_latency_ms,
            # ── Stage 10 — the ONE action eligible for a future hardware boundary.
            "final_safe_control_action_fraction": {
                k: float(v) for k, v in self.final_safe_control_action_fraction.items()
            },
            "final_safe_control_action_pct": {
                k: float(v) for k, v in self.final_safe_control_action_pct.items()
            },
            "final_safe_control_action_source": self.final_safe_control_action_source,
            "per_node": {k: dict(v) for k, v in self.per_node.items()},
            "gate_unit": "percent [0, 100] (external); fraction [0.0, 1.0] (internal)",
        }


class LiveMPCOrchestrator:
    """
    The ONE authoritative live controller path.

    Parameters
    ----------
    mpc : MPCController, optional
        The validated Phase 15.3 controller. Constructed once and reused; never
        modified.
    """

    def __init__(self, mpc: Optional[MPCController] = None,
                 safety: Optional[SafetyLayer] = None):
        self.mpc = mpc if mpc is not None else MPCController()
        # STAGE 8 — the validated SafetyLayer (src/controller/safety.py) as an
        # explicit boundary on the live path. It is the SAME class with the
        # SAME configuration the MPC uses internally, instantiated separately
        # so the live boundary is independent of the controller — not a second
        # safety system, and never a replacement for the controller's own.
        self.safety = safety if safety is not None else SafetyLayer(
            max_gate_change_per_step=self.mpc.config.max_gate_change
        )
        # STAGE 10 — the downstream-capacity boundary. It searches the SAME gate
        # lattice the validated MPC searched (never a new action space) and sits
        # AFTER the SafetyLayer, so its output is the FINAL_SAFE_CONTROL_ACTION.
        self.downstream_guard = DownstreamCapacityGuard(
            gate_levels=self.mpc.config.gate_levels
        )
        self.last_decision: Optional[LiveControlDecision] = None
        #: Measured latency of the SafetyLayer boundary (milliseconds).
        self.last_safety_latency_ms: float = 0.0
        #: Measured latency of the downstream-capacity boundary (milliseconds).
        self.last_downstream_latency_ms: float = 0.0

    # ------------------------------------------------------------------
    # STAGE 9 — coordination provenance (READ-ONLY introspection)
    # ------------------------------------------------------------------

    def action_space_for(self, node_ids: List[str]) -> Dict[str, Any]:
        """
        Describe the MPC's candidate/action space for ``node_ids``.

        READ-ONLY. It reads the validated controller's own configuration
        (``mpc.config.gate_levels``) and reports the dimensionality of the action
        vector the controller actually searches. It does not modify the MPC, its
        grid, its objective or its optimization logic.

        ``candidate_vectors`` is the size of the Cartesian product the validated
        MPC enumerates — i.e. the complete JOINT action vector, not a
        per-reservoir independent choice. That is what makes the control
        genuinely coordinated.
        """
        levels = list(getattr(self.mpc.config, "gate_levels", []) or [])
        dimension = len(node_ids)
        return {
            "dimension": dimension,
            "node_ids": list(node_ids),
            "gate_levels": levels,
            "candidate_vectors": len(levels) ** dimension if levels else 0,
            "coordinated": dimension > 1,
            "goal": "jointly optimise ONE gate position per reservoir",
            "source": "MPCController.config.gate_levels (validated Phase 15.3)",
        }

    # ------------------------------------------------------------------
    # eligibility — THE PROVENANCE GATE
    # ------------------------------------------------------------------

    @staticmethod
    def _provenance_for(
        snapshot: NetworkForecastSnapshot,
        bundle: Optional[LiveForecastBundle],
        provenance: Optional[Dict[str, Dict[str, Any]]],
    ) -> Dict[str, Dict[str, Any]]:
        """Prefer the structured bundle provenance; else parse the snapshot."""
        if bundle is not None and getattr(bundle, "provenance", None):
            return bundle.provenance
        if provenance:
            return provenance
        if snapshot is None:
            return {}
        return provenance_from_snapshot(snapshot)

    @classmethod
    def evaluate_eligibility(
        cls,
        node_ids: List[str],
        snapshot: Optional[NetworkForecastSnapshot],
        provenance: Dict[str, Dict[str, Any]],
    ) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Decide whether a forecast may drive the MPC.

        Returns ``(eligible, reasons, summary)``. Any reason ⇒ NOT eligible.
        The check is deliberately strict and fails closed.
        """
        reasons: List[str] = []
        summary: Dict[str, Any] = {}

        if snapshot is None:
            return False, ["NO_FORECAST_SNAPSHOT"], summary

        for node_id in node_ids:
            node_reasons: List[str] = []
            forecast = snapshot.get(node_id)
            prov = provenance.get(node_id) or {}

            if forecast is None:
                node_reasons.append("MISSING_FORECAST")
            else:
                # 0. Surface the adapter's precise issue code (MISSING_FORECAST,
                #    UNIT_MISMATCH, HORIZON_MISMATCH, MALFORMED_PAYLOAD, ...).
                issue = prov.get("issue")
                if issue:
                    node_reasons.append(str(issue))

                # 1. THE RULE: validated_metrics_apply must be exactly True.
                if prov.get("validated_metrics_apply") is not True:
                    node_reasons.append("NOT_VALIDATED_METRICS")

                # 2. Declared status must not be a demonstration / withheld state.
                declared = prov.get("declared_status")
                if declared is not None and declared != "VALIDATED":
                    node_reasons.append(f"STATUS_NOT_VALIDATED:{declared}")

                # 3. Unit must be canonical when declared.
                declared_unit = prov.get("unit")
                if declared_unit is not None and declared_unit != FORECAST_UNIT:
                    node_reasons.append(f"UNIT_MISMATCH:{declared_unit}")

                # 4. Every horizon the MPC consumes must be available + finite.
                for label in HORIZON_LABELS:
                    if not forecast.is_available(label):
                        node_reasons.append(f"HORIZON_UNAVAILABLE:{label}")
                        continue
                    value = forecast.get_prediction(label)
                    if value is None or not math.isfinite(value) or value < 0.0:
                        node_reasons.append(f"INVALID_VALUE:{label}")

            summary[node_id] = {
                "eligible": not node_reasons,
                "reasons": node_reasons,
                "issue": prov.get("issue"),
                "declared_status": prov.get("declared_status"),
                "validated_metrics_apply": prov.get("validated_metrics_apply"),
                "is_simulated": prov.get("is_simulated"),
            }
            reasons.extend(f"{node_id}:{r}" for r in node_reasons)

        return (len(reasons) == 0), reasons, summary

    # ------------------------------------------------------------------
    # decision
    # ------------------------------------------------------------------

    def decide(
        self,
        network,
        *,
        bundle: Optional[LiveForecastBundle] = None,
        snapshot: Optional[NetworkForecastSnapshot] = None,
        provenance: Optional[Dict[str, Dict[str, Any]]] = None,
        current_inflows: Optional[Dict[str, float]] = None,
    ) -> LiveControlDecision:
        """
        Produce the authoritative live control decision for ``network``.

        ``network`` is the live ``ReservoirNetwork`` (never mutated by the MPC).
        Provide EITHER a ``bundle`` (preferred) or a ``snapshot`` (+ optional
        ``provenance``).
        """
        if bundle is not None and snapshot is None:
            snapshot = bundle.snapshot

        node_ids: List[str] = list(network.processing_order)
        provenance = self._provenance_for(snapshot, bundle, provenance)

        # Current gates are REAL authoritative state — read, never invented.
        current_fraction = {nid: float(network.nodes[nid].state.gate_position) for nid in node_ids}
        current_pct = {nid: units.gate_fraction_to_percent(g) for nid, g in current_fraction.items()}

        eligible, reasons, summary = self.evaluate_eligibility(node_ids, snapshot, provenance)

        # STAGE 9 — the MPC's action space for THIS node set (read-only).
        action_space = self.action_space_for(node_ids)

        forecast_provenance = {
            "forecast_date": getattr(snapshot, "forecast_date", None),
            "nodes": summary,
            "reason_strings": reasons,
            "rule": "validated_metrics_apply == true for EVERY controlled reservoir",
        }

        if not eligible:
            decision = LiveControlDecision(
                controller_type="MPC",
                controller_status=ControllerStatus.BLOCKED.value,
                forecast_control_eligible=False,
                forecast_provenance=forecast_provenance,
                # The SafetyLayer must NOT manufacture a decision for a blocked
                # MPC: it is not applied at all, and says so.
                safety_layer_integrated=True,
                safety_layer_status=SAFETY_STATUS_NOT_APPLIED_MPC_BLOCKED,
                safety_is_safe=False,
                safety_violations=[],
                safety_modified=False,
                gate_positions_pct=current_pct,
                gate_positions_fraction=current_fraction,
                action_space=action_space,
                # Stage 10 — no action was produced, so the downstream boundary
                # did not run and does not fabricate one.
                downstream_status=DOWNSTREAM_STATUS_NOT_APPLIED_MPC_BLOCKED,
                downstream_capacity_mcm_day=float(network.downstream_capacity),
                downstream_reason=(
                    "MPC blocked by the forecast-provenance gate; no downstream "
                    "action was evaluated and no action was fabricated"
                ),
                final_safe_control_action_fraction=dict(current_fraction),
                final_safe_control_action_pct=dict(current_pct),
                final_safe_control_action_source="HELD_CURRENT_GATES",
                control_applied=False,
                blocked_reason="FORECAST_NOT_ELIGIBLE_FOR_CONTROL",
                reasons=reasons,
                mpc_status="NOT_INVOKED",
            )
            self.last_decision = decision
            return decision

        # ---- eligible: the MPC makes the decision ----
        try:
            mpc_decision = self.mpc.decide(
                network, forecast_snapshot=snapshot, current_inflows=current_inflows
            )
        except Exception as exc:  # pragma: no cover - defensive
            decision = LiveControlDecision(
                controller_type="MPC",
                controller_status=ControllerStatus.UNAVAILABLE.value,
                forecast_control_eligible=True,
                forecast_provenance=forecast_provenance,
                safety_layer_integrated=True,
                safety_layer_status=SAFETY_STATUS_NOT_APPLIED_MPC_ERROR,
                safety_is_safe=False,
                safety_violations=[],
                safety_modified=False,
                gate_positions_pct=current_pct,
                gate_positions_fraction=current_fraction,
                action_space=action_space,
                # Stage 10 — no action was produced, so the downstream boundary
                # did not run and does not fabricate one.
                downstream_status=DOWNSTREAM_STATUS_NOT_APPLIED_MPC_ERROR,
                downstream_capacity_mcm_day=float(network.downstream_capacity),
                downstream_reason=(
                    "the MPC raised; no downstream action was evaluated and no "
                    "action was fabricated"
                ),
                final_safe_control_action_fraction=dict(current_fraction),
                final_safe_control_action_pct=dict(current_pct),
                final_safe_control_action_source="HELD_CURRENT_GATES",
                control_applied=False,
                blocked_reason=f"MPC_ERROR:{type(exc).__name__}",
                reasons=[str(exc)],
                mpc_status="ERROR",
            )
            self.last_decision = decision
            return decision

        # ---- STAGE 8: the SafetyLayer boundary ----
        # The MPC's RAW proposal is passed verbatim (including any missing key)
        # so the validated layer's own missing-node handling runs. The layer's
        # output — never the raw proposal — is what reaches ReservoirNetwork.
        proposed_fraction = {
            str(k): v for k, v in (mpc_decision.gate_positions or {}).items()
        }
        t0 = time.perf_counter()
        safety_result = self.safety.validate(proposed_fraction, current_fraction, node_ids)
        self.last_safety_latency_ms = (time.perf_counter() - t0) * 1000.0

        safety_action_fraction = {
            nid: float(safety_result.validated_gates[nid]) for nid in node_ids
        }
        safety_action_pct = {
            nid: units.gate_fraction_to_percent(g) for nid, g in safety_action_fraction.items()
        }

        # ---- STAGE 10: the downstream-capacity boundary ----
        # The SafetyLayer's output is checked against the authoritative
        # downstream capacity. What comes out of here is the
        # FINAL_SAFE_CONTROL_ACTION. The downstream boundary never rescues a
        # missing/invalid forecast: this code only runs when the MPC produced an
        # action (the provenance gate above already decided that).
        t1 = time.perf_counter()
        downstream = self.downstream_guard.evaluate(
            network,
            action_fraction=safety_action_fraction,
            current_fraction=current_fraction,
            node_ids=node_ids,
            max_gate_change=self.mpc.config.max_gate_change,
            inflows=current_inflows,
        )
        self.last_downstream_latency_ms = (time.perf_counter() - t1) * 1000.0

        applied_fraction = {
            nid: float(downstream.action_fraction[nid]) for nid in node_ids
        }

        # POST-CHECK (verification, not a second safety layer): the boundary sits
        # after the SafetyLayer, so its output must still be admissible under the
        # SafetyLayer's own rules (bounds + per-step rate limit). It always is, by
        # construction — every candidate the boundary searches is filtered through
        # exactly that rule. This check recomputes the rule rather than invoking
        # the layer again, so the SafetyLayer is still called exactly ONCE per
        # decision (the Stage 8 invariant). If it ever fires, the check has found
        # a real defect and the downstream guarantee is NOT claimed.
        applied_action_safety_layer_clean = self.downstream_guard.safety_layer_feasible(
            applied_fraction,
            node_ids,
            current_fraction,
            self.mpc.config.max_gate_change,
        )
        if not applied_action_safety_layer_clean:  # pragma: no cover - defensive
            applied_fraction = dict(safety_action_fraction)
            downstream.status = DOWNSTREAM_STATUS_FAILED_CLOSED
            downstream.capacity_achieved = False
            downstream.is_protected = False
            downstream.modified = True
            downstream.reason = (
                "POST_CHECK_SAFETY_LAYER_DIVERGENCE: the downstream-safe action was "
                "not admissible under the validated SafetyLayer's rules; the "
                "SafetyLayer output was kept and NO downstream-capacity guarantee "
                "is claimed"
            )

        applied_pct = {nid: units.gate_fraction_to_percent(g) for nid, g in applied_fraction.items()}

        downstream_dict = downstream.to_dict()

        # Audit copy of the RAW proposal. Kept transport-safe (finite floats or
        # None) so a malformed proposal can never break the WebSocket JSON, and
        # never routed through the unit boundary (which rightly rejects NaN).
        proposed_audit: Dict[str, Optional[float]] = {}
        for nid in node_ids:
            if nid not in proposed_fraction:
                proposed_audit[nid] = None
                continue
            try:
                value = float(proposed_fraction[nid])
            except (TypeError, ValueError):
                proposed_audit[nid] = None
                continue
            proposed_audit[nid] = value if math.isfinite(value) else None

        safety_modified = any(
            proposed_audit[nid] is None or proposed_audit[nid] != safety_action_fraction[nid]
            for nid in node_ids
        )
        # The action the SafetyLayer produced, before the downstream boundary.
        safety_layer_modified_by_downstream = any(
            safety_action_fraction[nid] != applied_fraction[nid] for nid in node_ids
        )

        decision = LiveControlDecision(
            controller_type="MPC",
            controller_status=ControllerStatus.ACTIVE.value,
            forecast_control_eligible=True,
            forecast_provenance=forecast_provenance,
            safety_layer_integrated=True,
            safety_layer_status=str(safety_result.status),
            safety_is_safe=bool(safety_result.is_safe),
            safety_violations=list(safety_result.violations),
            safety_modified=safety_modified,
            gate_positions_pct=applied_pct,
            gate_positions_fraction=applied_fraction,
            proposed_gate_positions_fraction=proposed_audit,
            safety_layer_gate_positions_fraction=dict(safety_action_fraction),
            safety_layer_gate_positions_pct=dict(safety_action_pct),
            action_space=action_space,
            # ── Stage 10 ─────────────────────────────────────────────────
            downstream_status=downstream.status,
            downstream_capacity_mcm_day=float(downstream.capacity_mcm_day),
            downstream_predicted_flow_mcm_day=downstream.predicted_flow_mcm_day,
            downstream_proposed_predicted_flow_mcm_day=downstream.proposed_predicted_flow_mcm_day,
            downstream_min_achievable_flow_mcm_day=downstream.min_achievable_flow_mcm_day,
            downstream_capacity_achieved=bool(downstream.capacity_achieved),
            downstream_protection_modified=bool(downstream.modified),
            downstream_horizon_steps=int(downstream.horizon_steps),
            downstream_candidates_evaluated=int(downstream.candidates_evaluated),
            downstream_reason=str(downstream.reason),
            downstream_capacity_protection=downstream_dict,
            applied_action_safety_layer_clean=bool(applied_action_safety_layer_clean),
            downstream_latency_ms=float(self.last_downstream_latency_ms),
            final_safe_control_action_fraction=dict(applied_fraction),
            final_safe_control_action_pct=dict(applied_pct),
            final_safe_control_action_source="DOWNSTREAM_CAPACITY_GUARD",
            control_applied=True,
            reasons=list(mpc_decision.reasons) + (
                ["DOWNSTREAM_PROTECTION_MODIFIED_ACTION"]
                if safety_layer_modified_by_downstream else []
            ),
            mpc_status=mpc_decision.status,
            mpc_objective_score=float(mpc_decision.objective_score)
            if mpc_decision.objective_score is not None else None,
            mpc_forecast_used=bool(mpc_decision.forecast_used),
            mpc_forecast_status=str(mpc_decision.forecast_status),
            mpc_safety_status=str(mpc_decision.safety_status),
            candidates_evaluated=int(mpc_decision.candidates_evaluated),
            per_node={k: dict(v) for k, v in (mpc_decision.per_node or {}).items()},
        )
        self.last_decision = decision
        return decision

    # ------------------------------------------------------------------

    def status_dict(self) -> Dict[str, Any]:
        """Last decision, or an explicit 'never decided' record."""
        if self.last_decision is None:
            return {
                "controller_type": "MPC",
                "controller_status": ControllerStatus.UNAVAILABLE.value,
                "forecast_control_eligible": False,
                "forecast_provenance": {},
                "safety_layer_integrated": True,
                "safety_layer_version": SAFETY_LAYER_VERSION,
                "safety_layer_status": SAFETY_STATUS_NOT_APPLIED_MPC_BLOCKED,
                "safety_is_safe": False,
                "safety_violations": [],
                "safety_modified": False,
                "proposed_gate_positions_fraction": {},
                "action_space": {},
                # Stage 10 — no decision yet, so no downstream check has run.
                "downstream_status": DOWNSTREAM_STATUS_NOT_APPLIED_MPC_BLOCKED,
                "downstream_capacity_mcm_day": 0.0,
                "downstream_predicted_flow_mcm_day": None,
                "downstream_proposed_predicted_flow_mcm_day": None,
                "downstream_min_achievable_flow_mcm_day": None,
                "downstream_capacity_achieved": False,
                "downstream_protection_modified": False,
                "downstream_horizon_steps": 0,
                "downstream_candidates_evaluated": 0,
                "downstream_reason": "NO_DECISION_YET",
                "downstream_capacity_protection": {},
                "applied_action_safety_layer_clean": True,
                "downstream_latency_ms": 0.0,
                "final_safe_control_action_fraction": {},
                "final_safe_control_action_pct": {},
                "final_safe_control_action_source": "NOT_APPLIED",
                "control_applied": False,
                "blocked_reason": "NO_DECISION_YET",
                "reasons": [],
            }
        return self.last_decision.to_dict()
