"""
Stage 11 — Live Mass-Balance Integrity
======================================

Verifies, after **every** authoritative simulation step, that the state produced
by the validated physics is physically consistent — i.e. that the LIVE Digital
Twin is not merely emitting plausible-looking numbers.

WHAT THIS MODULE IS NOT
-----------------------
* It is NOT a simulator. It never advances storage, release, spill or routing.
* It does NOT rewrite, tune or re-derive a single equation of
  :class:`~src.network_env.reservoir_network.ReservoirNetwork`.
* It does NOT repair anything. A detected violation is REPORTED; storage,
  inflow, outflow and spill are never silently altered.

It is a **read-only auditor** of one step of the validated model.

THE AUTHORITATIVE PHYSICS (read out of ``reservoir_network.py``, not assumed)
----------------------------------------------------------------------------
``ReservoirNode.step`` implements, per reservoir, per step::

    gate_clamped       = clamp(gate_position, 0, 1)
    requested_release  = gate_clamped * max_release
    total_inflow       = local_inflow + routed_inflow
    available          = storage + total_inflow
    controlled_release = min(requested_release, available)
    preliminary        = available - controlled_release
    spill              = max(0, preliminary - capacity)
    storage            = min(preliminary, capacity)      # == preliminary - spill
    total_outflow      = controlled_release + spill

so the per-reservoir conservation law this module checks is exactly::

    new_storage - old_storage
        = inflow_local + inflow_routed - controlled_release - spill

``ReservoirNetwork.step`` implements, per connection, per step::

    raw_arriving       = queue.popleft()          # the OLDEST queued volume
    attenuated_arrival = raw_arriving * attenuation
    transmission_loss  = raw_arriving * (1 - attenuation)
    routed_arriving[destination] += attenuated_arrival
    ... (after the node pass) queue.append(today's controlled release of source)

so the connection law is::

    raw_arriving = attenuated_arrival + transmission_loss
    queue_after  == queue_before[1:] + [released_this_step]

and the network-level law (all quantities for THIS step) is::

    total_external_inflow
        = storage_change
        + terminal_outflow          (terminal controlled release + terminal spill)
        + nonterminal_spill         (spill from upstream nodes — leaves the network)
        + routing_loss
        + change_in_transit         (water that entered/left the routing queues)

The terminal node's spill is already inside ``terminal_outflow`` and must NOT
also appear in the spill term — doing so would double-count it. This form is
exactly the incremental form of the network's own cumulative
``ReservoirNetwork.mass_balance_check()``.

TIME-STEP CONVENTION (why MCM/day and MCM may be compared here)
--------------------------------------------------------------
One ``step()`` advances the model by **exactly one day** (``ReservoirNetwork.step``
increments ``timestep`` by 1 and the model is documented as "Discrete-time daily
mass balance"). A flow of ``X MCM/day`` therefore moves exactly ``X MCM`` of water
during one step. The validated implementation applies **no** ``dt`` scaling
anywhere — there is no ``* dt`` / ``/ dt`` factor in ``reservoir_network.py``.

This module therefore performs each per-step check in MCM-per-step and reports
``timestep_days = 1.0`` alongside it, so the daily-flow ↔ per-step-volume
relationship is explicit rather than implicit. Nothing is scaled, and no unit is
converted: it compares the model's own numbers, in the model's own units.

SPILL AND ROUTING TREATMENT
---------------------------
* ``spill`` is the forced overflow booked separately from the controlled
  release; it is never lumped into the release.
* Spill from a **non-terminal** node leaves the network (it is NOT routed to any
  downstream reservoir) and is reported as ``nonterminal_spill``. Spill from the
  **terminal** node leaves the network too, but it is already part of
  ``terminal_outflow``, so it is reported separately (``terminal_spill``) and is
  deliberately NOT added again — the network's own cumulative balance separates
  them the same way, and adding both would double-count the terminal overflow.
* ``routing_loss`` is the attenuation loss ``(1 - α) * raw_arriving`` on each
  connection. It is water that genuinely leaves the network and is reported as
  such — it is never "silently destroyed" (Stage 3 established this).
* Water still inside the routing queues is real state and is carried in
  ``change_in_transit``. Routed inflow is therefore **never** treated as
  instantaneous: what enters a node this step is what was released by its
  upstream neighbour ``delay`` steps earlier (verified explicitly per
  connection).

FAILURE HANDLING
----------------
A violation is reported (``status = VIOLATION``, ``state_valid = False``, the
offending residuals and a human-readable ``violations`` list). Nothing is
repaired, clamped or hidden, and the diagnostic still reaches the API, the
WebSocket and the Digital Twin.

The validated architecture defines **no** fail-safe response to a conservation
violation. Rather than invent one, this module reports the gap explicitly
(``fail_safe = "NONE_DEFINED_IN_EXISTING_ARCHITECTURE"`` +
``fail_safe_gap``). The monitor deliberately does not raise into the control
loop: doing so would let a numerical failure silently stop the twin instead of
being displayed by it. ``ever_violated`` / ``violation_count`` are sticky, so a
violation cannot be hidden by a later passing step.

HARDWARE PREPARATION (telemetry is NOT implemented here)
-------------------------------------------------------
The same integrity framework can later compare commanded versus actual plant
behaviour: :meth:`MassBalanceMonitor.verify` takes the *applied* action and the
*applied* inflows as explicit arguments and reads the resulting state, so a
future telemetry record can supply the same inputs. Every diagnostic carries

    timestamp, per-reservoir storage/level, commanded gate, applied gate,
    measured inflow, measured outflow, spill, and a sensor-quality slot
    (``None`` today — ``hardware_connected = False``).

No telemetry hardware, no sensor driver and no serial/ESP32 code exists here.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .reservoir_network import ReservoirNetwork

# ---------------------------------------------------------------------------
# Statuses — "PASS" may only ever be reported AFTER a check has actually run.
# ---------------------------------------------------------------------------

#: Every invariant was checked for this step and every residual is within
#: tolerance.
MASS_BALANCE_STATUS_PASS = "PASS"
#: A check ran and at least one invariant was broken (or the state was not
#: finite / not fully diagnosable). Never repaired, never hidden.
MASS_BALANCE_STATUS_VIOLATION = "VIOLATION"
#: No check has run yet (no authoritative step has been audited), or the
#: provenance needed for a check is absent. Never rendered as PASS.
MASS_BALANCE_STATUS_NOT_CHECKED = "NOT_CHECKED"

#: Numerical tolerance for every mass-balance residual, in MCM.
#:
#: The validated model's residuals are pure IEEE-754 accumulation noise —
#: measured 2.84e-14 MCM (worst case, per reservoir and network level) over a
#: 12-step live run. This tolerance sits ~4.5 orders of magnitude above that
#: noise floor and ~9.5 orders of magnitude below any physically meaningful
#: volume (the smallest reservoir's max release is 5 MCM/day), so it cannot
#: mask a real conservation error while never firing on floating-point noise.
MASS_BALANCE_TOLERANCE_MCM = 1e-9

#: Tolerance for the commanded-gate vs applied-gate comparison (gate FRACTION).
GATE_TOLERANCE = 1e-9

#: Every step of the validated model is exactly one day (no dt scaling exists).
TIMESTEP_DAYS = 1.0

#: The per-reservoir conservation law, verbatim from ``ReservoirNode.step``.
RESERVOIR_EQUATION = (
    "new_storage = old_storage + inflow_local + inflow_routed "
    "- controlled_release - spill"
)

#: The per-connection routing law, verbatim from ``ReservoirNetwork.step``.
ROUTING_EQUATION = (
    "raw_arriving = attenuated_arrival + transmission_loss ; "
    "queue_after = queue_before[1:] + [released_this_step]"
)

#: The network-level conservation law implemented by ``ReservoirNetwork``.
NETWORK_EQUATION = (
    "total_external_inflow = storage_change + terminal_outflow + nonterminal_spill "
    "+ routing_loss + change_in_transit"
)

#: The existing architecture defines no fail-safe response to a conservation
#: violation. This is REPORTED (the gap), not invented.
FAIL_SAFE_NONE_DEFINED = "NONE_DEFINED_IN_EXISTING_ARCHITECTURE"
FAIL_SAFE_GAP = (
    "The validated architecture (ReservoirNetwork / SafetyLayer / MPC / "
    "DownstreamCapacityGuard) defines no response to a mass-balance violation: "
    "the SafetyLayer validates gates only, the MPC prices downstream flow only, "
    "and the downstream guard constrains flow only. Stage 11 therefore REPORTS "
    "the violation, marks the state invalid and never repairs it. It does not "
    "invent a fail-safe, does not halt the loop and does not suppress the "
    "diagnostic."
)

#: Unit labels — the model's own units, no conversion anywhere in this module.
STORAGE_UNIT = "MCM"
FLOW_UNIT = "MCM/day"
GATE_UNIT = "fraction"

#: Fields a future hardware telemetry record would supply to the SAME framework.
#: NOT implemented — documented only (no ESP32 / sensor / serial code exists).
FUTURE_TELEMETRY_FIELDS = (
    "timestamp",
    "reservoir_state",
    "actual_gate_position",
    "measured_inflow_mcm_day",
    "measured_outflow_mcm_day",
    "storage_or_level",
    "sensor_quality",
)


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

@dataclass
class StepSnapshot:
    """Everything the audit needs from the network BEFORE it steps."""

    timestep: int
    storages: Dict[str, float] = field(default_factory=dict)
    gates: Dict[str, float] = field(default_factory=dict)
    queues: Dict[str, List[float]] = field(default_factory=dict)
    water_in_transit: float = 0.0


def connection_key(source: str, destination: str) -> str:
    return f"{source}->{destination}"


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _within(value: Any, tolerance: float) -> bool:
    """Finite and |value| <= tolerance. A non-finite value is NEVER within."""
    return _is_finite(value) and abs(float(value)) <= tolerance


def _json_safe(value: Any) -> Any:
    """
    Make the diagnostic transport-safe without hiding anything.

    A non-finite float is rendered in its string form (``"nan"`` / ``"inf"``).
    This matters: a NaN/Inf state would otherwise make ``json.dumps`` raise while
    the WebSocket payload is being built, i.e. the violation would take the
    broadcast down instead of being REPORTED by it. The violation status, the
    ``non_finite`` list and the human-readable reasons are unchanged.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class MassBalanceResult:
    """Structured outcome of one authoritative-step audit."""

    status: str = MASS_BALANCE_STATUS_NOT_CHECKED
    checked: bool = False
    #: Worst absolute residual over every invariant checked this step (MCM).
    residual: Optional[float] = None
    tolerance: float = MASS_BALANCE_TOLERANCE_MCM
    unit: str = STORAGE_UNIT
    flow_unit: str = FLOW_UNIT
    timestep_days: float = TIMESTEP_DAYS
    timestep: Optional[int] = None
    step_index: Optional[int] = None
    timestamp: Optional[str] = None

    per_reservoir: List[Dict[str, Any]] = field(default_factory=list)
    routing: List[Dict[str, Any]] = field(default_factory=list)
    network: Dict[str, Any] = field(default_factory=dict)

    reservoirs_checked: int = 0
    reservoirs_expected: int = 0

    applied_action_fraction: Dict[str, float] = field(default_factory=dict)
    applied_action_percent: Dict[str, float] = field(default_factory=dict)
    applied_action_source: Optional[str] = None
    applied_inflows_mcm_day: Dict[str, float] = field(default_factory=dict)
    action_fully_applied: bool = False
    gate_clamped: List[str] = field(default_factory=list)

    non_finite: List[str] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)
    reason: str = ""

    state_valid: bool = False
    ever_violated: bool = False
    violation_count: int = 0
    checks: int = 0

    fail_safe: str = FAIL_SAFE_NONE_DEFINED
    fail_safe_gap: str = FAIL_SAFE_GAP

    equation: str = RESERVOIR_EQUATION
    routing_equation: str = ROUTING_EQUATION
    network_equation: str = NETWORK_EQUATION
    physics: str = "ReservoirNetwork (src/network_env/reservoir_network.py)"

    latency_ms: float = 0.0
    telemetry: Dict[str, Any] = field(default_factory=dict)

    # -- convenience -----------------------------------------------------

    @property
    def is_pass(self) -> bool:
        return self.status == MASS_BALANCE_STATUS_PASS and self.checked

    def to_dict(self) -> Dict[str, Any]:
        """
        Transport form of the audit (JSON-safe).

        The dataclass keeps the raw numbers; this view renders non-finite floats
        as strings so the diagnostic can ALWAYS reach the API / WebSocket /
        Digital Twin — including when the thing it has to report is a NaN state.
        """
        return _json_safe({
            "status": self.status,
            "checked": self.checked,
            "residual": self.residual,
            "tolerance": self.tolerance,
            "unit": self.unit,
            "flow_unit": self.flow_unit,
            "timestep_days": self.timestep_days,
            "timestep": self.timestep,
            "step_index": self.step_index,
            "timestamp": self.timestamp,
            "per_reservoir": [dict(r) for r in self.per_reservoir],
            "routing": [dict(r) for r in self.routing],
            "network": dict(self.network),
            "reservoirs_checked": self.reservoirs_checked,
            "reservoirs_expected": self.reservoirs_expected,
            "applied_action_fraction": dict(self.applied_action_fraction),
            "applied_action_percent": dict(self.applied_action_percent),
            "applied_action_source": self.applied_action_source,
            "applied_inflows_mcm_day": dict(self.applied_inflows_mcm_day),
            "action_fully_applied": self.action_fully_applied,
            "gate_clamped": list(self.gate_clamped),
            "non_finite": list(self.non_finite),
            "violations": list(self.violations),
            "reason": self.reason,
            "state_valid": self.state_valid,
            "ever_violated": self.ever_violated,
            "violation_count": self.violation_count,
            "checks": self.checks,
            "fail_safe": self.fail_safe,
            "fail_safe_gap": self.fail_safe_gap,
            "equation": self.equation,
            "routing_equation": self.routing_equation,
            "network_equation": self.network_equation,
            "physics": self.physics,
            "latency_ms": self.latency_ms,
            "telemetry": dict(self.telemetry),
        })


def not_checked_result(reason: str = "NO_AUTHORITATIVE_STEP_AUDITED") -> Dict[str, Any]:
    """
    The honest record for "no check has happened".

    Used by every consumer (API, WebSocket, Digital Twin) so that a missing
    diagnostic can never be rendered as PASS.
    """
    return MassBalanceResult(reason=reason).to_dict()


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------

class MassBalanceMonitor:
    """
    Read-only auditor of the authoritative ``ReservoirNetwork``.

    Two-phase API (what a future hardware telemetry path can reuse):

      ``snapshot(network)``  -> capture the pre-step state
      ``verify(network, snapshot, applied_inflows=…, applied_gates_fraction=…)``
                             -> audit the state the step produced

    ``step_and_check(network, inflows, gates)`` is the live path's one-call
    form: it snapshots, calls ``network.step(...)`` **with exactly those
    arguments**, and audits the result. Because the audit inputs are the very
    objects handed to ``ReservoirNetwork.step``, the check can only ever be
    about the action that was actually applied.
    """

    def __init__(self, tolerance: float = MASS_BALANCE_TOLERANCE_MCM):
        self.tolerance = float(tolerance)
        self.last_result: Optional[MassBalanceResult] = None
        #: Per-connection history of controlled releases (most recent last).
        #: Used to prove the delay: what arrives now was released `delay` steps ago.
        self._release_history: Dict[str, deque] = {}
        self.checks = 0
        self.violation_count = 0
        self.ever_violated = False

    # -- phase 1 ---------------------------------------------------------

    def snapshot(self, network: ReservoirNetwork) -> StepSnapshot:
        """Capture the pre-step state of the authoritative network (read-only)."""
        return StepSnapshot(
            timestep=int(network.timestep),
            storages={nid: float(network.nodes[nid].state.storage) for nid in network.nodes},
            gates={nid: float(network.nodes[nid].state.gate_position) for nid in network.nodes},
            queues={
                connection_key(c.source, c.destination): [float(v) for v in c.queue]
                for c in network.connections
            },
            water_in_transit=self._water_in_transit(network),
        )

    @staticmethod
    def _water_in_transit(network: ReservoirNetwork) -> float:
        """Water currently inside the routing queues (real state, MCM)."""
        return float(sum(sum(float(v) for v in c.queue) for c in network.connections))

    # -- phase 2 ---------------------------------------------------------

    def verify(
        self,
        network: ReservoirNetwork,
        snapshot: StepSnapshot,
        *,
        applied_inflows: Dict[str, float],
        applied_gates_fraction: Dict[str, float],
        action_source: Optional[str] = None,
        action_percent: Optional[Dict[str, float]] = None,
        step_index: Optional[int] = None,
        timestamp: Optional[str] = None,
    ) -> MassBalanceResult:
        """
        Audit the state the network now holds against ``snapshot`` and the
        action/inflows that were applied. Never mutates the network.
        """
        t0 = time.perf_counter()

        node_ids = list(network.processing_order)
        tol = self.tolerance
        result = MassBalanceResult(
            status=MASS_BALANCE_STATUS_NOT_CHECKED,
            checked=False,
            tolerance=tol,
            timestep=int(network.timestep),
            step_index=step_index,
            timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
            reservoirs_expected=len(node_ids),
            applied_action_fraction={k: float(v) for k, v in applied_gates_fraction.items()},
            applied_action_percent={k: float(v) for k, v in (action_percent or {}).items()},
            applied_action_source=action_source,
            applied_inflows_mcm_day={k: float(v) for k, v in applied_inflows.items()},
            fail_safe=FAIL_SAFE_NONE_DEFINED,
            fail_safe_gap=FAIL_SAFE_GAP,
            telemetry={
                "source": "SIMULATION",
                "hardware_connected": False,
                "sensor_quality": None,
                "note": (
                    "No hardware is connected. These values are produced by the "
                    "authoritative simulation, not by plant telemetry."
                ),
                "future_fields": list(FUTURE_TELEMETRY_FIELDS),
            },
        )

        non_finite: List[str] = []

        # ---- 0. inputs actually applied to ReservoirNetwork.step() --------
        for nid in node_ids:
            gate = applied_gates_fraction.get(nid, None)
            if gate is None:
                non_finite.append(f"applied_gate[{nid}]=MISSING")
            elif not _is_finite(gate):
                non_finite.append(f"applied_gate[{nid}]={gate!r}")
            inflow = applied_inflows.get(nid, None)
            if inflow is None:
                non_finite.append(f"applied_inflow[{nid}]=MISSING")
            elif not _is_finite(inflow):
                non_finite.append(f"applied_inflow[{nid}]={inflow!r}")

        # ---- 1. per-connection routing -----------------------------------
        routed_expected: Dict[str, float] = {nid: 0.0 for nid in node_ids}
        step_routing_loss = 0.0
        routing_rows: List[Dict[str, Any]] = []
        worst = 0.0
        routing_ok = True

        for conn in network.connections:
            key = connection_key(conn.source, conn.destination)
            before = list(snapshot.queues.get(key, []))
            after = [float(v) for v in conn.queue]
            raw_arriving = before[0] if before else 0.0
            attenuated = raw_arriving * float(conn.attenuation)
            loss = raw_arriving * (1.0 - float(conn.attenuation))
            released_this_step = float(network.nodes[conn.source].state.controlled_release)

            residual = raw_arriving - attenuated - loss
            step_routing_loss += loss
            routed_expected[conn.destination] = (
                routed_expected.get(conn.destination, 0.0) + attenuated
            )

            expected_queue = before[1:] + [released_this_step]
            queue_ok = len(expected_queue) == len(after) and all(
                _is_finite(a) and _is_finite(b) and abs(a - b) <= tol
                for a, b in zip(expected_queue, after)
            )

            # The delay, proven from the monitor's own release history: what
            # arrives now is what the source released `delay` steps ago.
            delay = int(conn.delay)
            history = self._release_history.get(key)
            if history is not None and len(history) >= delay >= 1:
                expected_raw = float(history[-delay])
                delay_ok: Optional[bool] = _is_finite(raw_arriving) and abs(
                    raw_arriving - expected_raw
                ) <= tol
                delay_note = (
                    f"raw_arriving equals the release from {delay} step(s) ago"
                    if delay_ok else
                    f"raw_arriving {raw_arriving!r} != release from {delay} step(s) "
                    f"ago ({expected_raw!r})"
                )
            else:
                delay_ok = None
                delay_note = (
                    "no release history available for this connection yet; the "
                    "routing delay is instead verified by the FIFO queue identity"
                )

            if not _is_finite(raw_arriving) or not _is_finite(attenuated) or not _is_finite(loss):
                non_finite.append(f"routing[{key}] raw/attenuated/loss")
            if not _within(residual, tol):
                routing_ok = False
            if not queue_ok:
                routing_ok = False
            if delay_ok is False:
                routing_ok = False

            if _is_finite(residual):
                worst = max(worst, abs(residual))

            routing_rows.append({
                "connection": key,
                "source": conn.source,
                "destination": conn.destination,
                "delay_days": delay,
                "attenuation": float(conn.attenuation),
                "released_this_step_mcm_day": released_this_step,
                "raw_arriving_mcm_day": raw_arriving,
                "attenuated_arrival_mcm_day": attenuated,
                "transmission_loss_mcm": loss,
                "residual_mcm": residual,
                "queue_len_before": len(before),
                "queue_len_after": len(after),
                "queue_evolution_ok": bool(queue_ok),
                "delay_history_ok": delay_ok,
                "delay_note": delay_note,
                "within_tolerance": bool(_within(residual, tol) and queue_ok and delay_ok is not False),
            })

        # ---- 2. per-reservoir conservation ------------------------------
        per_reservoir: List[Dict[str, Any]] = []
        reservoirs_ok = True
        storage_change_total = 0.0
        nonterminal_spill_total = 0.0
        terminal_spill = 0.0
        spill_all_nodes = 0.0
        external_inflow_total = 0.0
        terminal_outflow = 0.0
        terminal_controlled_release = 0.0

        for nid in node_ids:
            node = network.nodes[nid]
            state = node.state
            storage_before = snapshot.storages.get(nid)
            storage_after = float(state.storage)
            inflow_local = float(state.inflow_local)
            inflow_routed = float(state.inflow_routed)
            controlled_release = float(state.controlled_release)
            spill = float(state.spill)
            total_outflow = float(state.total_outflow)
            gate_position = float(state.gate_position)
            commanded_gate = applied_gates_fraction.get(nid)

            values = (storage_before, storage_after, inflow_local, inflow_routed,
                      controlled_release, spill, total_outflow, gate_position)
            if not all(_is_finite(v) for v in values):
                non_finite.append(f"reservoir[{nid}] storage/inflow/outflow/spill")
            if commanded_gate is not None and _is_finite(commanded_gate):
                if abs(gate_position - float(commanded_gate)) > GATE_TOLERANCE:
                    result.gate_clamped.append(nid)

            storage_change = (
                storage_after - float(storage_before)
                if _is_finite(storage_before) and _is_finite(storage_after) else float("nan")
            )
            residual = (
                storage_change
                - (inflow_local + inflow_routed - controlled_release - spill)
            )

            # The routed inflow a node reports must be exactly what the
            # connections delivered — never the upstream release itself.
            routed_residual = inflow_routed - routed_expected.get(nid, 0.0)

            row_ok = _within(residual, tol) and _within(routed_residual, tol)
            if not row_ok:
                reservoirs_ok = False
            # Only genuine RESIDUALS contribute to the reported worst residual —
            # a storage change is a real volume, not an error term.
            for candidate in (residual, routed_residual):
                if _is_finite(candidate):
                    worst = max(worst, abs(candidate))

            if _is_finite(storage_change):
                storage_change_total += storage_change
            if _is_finite(spill):
                spill_all_nodes += spill
                if nid == network._terminal_node_id:
                    terminal_spill += spill
                else:
                    nonterminal_spill_total += spill
            if _is_finite(inflow_local):
                external_inflow_total += inflow_local
            if nid == network._terminal_node_id:
                if _is_finite(total_outflow):
                    terminal_outflow = total_outflow
                if _is_finite(controlled_release):
                    terminal_controlled_release = controlled_release

            per_reservoir.append({
                "node_id": nid,
                "description": (
                    f"{nid}: storage {storage_before!r} -> {storage_after!r} MCM, "
                    f"in {inflow_local!r} (local) + {inflow_routed!r} (routed) MCM/day, "
                    f"out {controlled_release!r} (release) + {spill!r} (spill) MCM/day"
                ),
                "storage_before_mcm": storage_before,
                "storage_after_mcm": storage_after,
                "storage_change_mcm": storage_change,
                "inflow_local_mcm_day": inflow_local,
                "inflow_routed_mcm_day": inflow_routed,
                "routed_inflow_residual_mcm": routed_residual,
                "controlled_release_mcm_day": controlled_release,
                "spill_mcm": spill,
                "total_outflow_mcm_day": total_outflow,
                "commanded_gate_fraction": (
                    float(commanded_gate) if commanded_gate is not None else None
                ),
                "applied_gate_position": gate_position,
                "residual_mcm": residual,
                "within_tolerance": bool(row_ok),
                "terminal": nid == network._terminal_node_id,
            })

        # ---- 3. network-level conservation ------------------------------
        transit_after = self._water_in_transit(network)
        change_in_transit = transit_after - float(snapshot.water_in_transit)
        accounted = (
            storage_change_total + terminal_outflow + nonterminal_spill_total
            + step_routing_loss + change_in_transit
        )
        network_residual = external_inflow_total - accounted
        if _is_finite(network_residual):
            worst = max(worst, abs(network_residual))
        else:
            non_finite.append("network_residual=NON_FINITE")
        network_ok = _within(network_residual, tol)

        result.network = {
            "total_external_inflow_mcm_day": external_inflow_total,
            "storage_change_mcm": storage_change_total,
            "terminal_outflow_mcm_day": terminal_outflow,
            "terminal_controlled_release_mcm_day": terminal_controlled_release,
            "terminal_spill_mcm": terminal_spill,
            "nonterminal_spill_mcm": nonterminal_spill_total,
            "spill_all_nodes_mcm": spill_all_nodes,
            "routing_loss_mcm": step_routing_loss,
            "water_in_transit_before_mcm": float(snapshot.water_in_transit),
            "water_in_transit_after_mcm": transit_after,
            "change_in_transit_mcm": change_in_transit,
            "accounted_mcm": accounted,
            "residual_mcm": network_residual,
            "within_tolerance": bool(network_ok),
            "terminal_node_id": network._terminal_node_id,
            "routing_delays_days": {
                connection_key(c.source, c.destination): int(c.delay)
                for c in network.connections
            },
            "attenuation_factors": {
                connection_key(c.source, c.destination): float(c.attenuation)
                for c in network.connections
            },
            "double_count_note": (
                "terminal_spill is already inside terminal_outflow and is NOT "
                "added again; only non-terminal spill appears as its own term."
            ),
            # Informational ONLY (not part of the verdict): the network's own
            # cumulative check, so the incremental audit above can be compared
            # against the model's own accounting rather than taken on trust.
            "cumulative_residual_mcm": (
                network.mass_balance_check().get("residual_error")
            ),
        }

        # ---- 4. verdict --------------------------------------------------
        result.per_reservoir = per_reservoir
        result.routing = routing_rows
        result.reservoirs_checked = len(per_reservoir)
        result.non_finite = list(non_finite)
        result.residual = worst if math.isfinite(worst) else float("nan")
        result.action_fully_applied = not result.gate_clamped

        all_reservoirs_present = result.reservoirs_checked == result.reservoirs_expected > 0
        result.checked = True

        if non_finite:
            result.violations.append(
                "NON_FINITE_STATE_DETECTED: " + "; ".join(non_finite[:8])
            )
        if not all_reservoirs_present:
            result.violations.append(
                f"INCOMPLETE_DIAGNOSTIC: {result.reservoirs_checked} of "
                f"{result.reservoirs_expected} reservoirs were diagnosable"
            )
        for row in per_reservoir:
            if not row["within_tolerance"]:
                result.violations.append(
                    f"RESERVOIR_MASS_BALANCE_VIOLATION: {row['node_id']} residual "
                    f"{row['residual_mcm']!r} MCM (tolerance {tol} MCM) "
                    f"[{row['description']}]"
                )
        for row in routing_rows:
            if not row["within_tolerance"]:
                result.violations.append(
                    f"ROUTING_MASS_BALANCE_VIOLATION: {row['connection']} residual "
                    f"{row['residual_mcm']!r} MCM, queue_evolution_ok="
                    f"{row['queue_evolution_ok']}, delay_history_ok={row['delay_history_ok']}"
                )
        if not network_ok:
            result.violations.append(
                f"NETWORK_MASS_BALANCE_VIOLATION: residual {network_residual!r} MCM "
                f"(tolerance {tol} MCM)"
            )

        result.status = (
            MASS_BALANCE_STATUS_PASS if not result.violations
            else MASS_BALANCE_STATUS_VIOLATION
        )
        result.state_valid = result.status == MASS_BALANCE_STATUS_PASS
        if result.status == MASS_BALANCE_STATUS_PASS:
            result.reason = (
                "every reservoir and every connection conserved mass within "
                f"{tol} MCM over one {TIMESTEP_DAYS:g}-day step"
            )
        else:
            result.reason = "; ".join(result.violations)

        self.checks += 1
        if not result.state_valid:
            self.violation_count += 1
            self.ever_violated = True
        result.checks = self.checks
        result.violation_count = self.violation_count
        result.ever_violated = self.ever_violated

        # ---- 5. remember this step's releases for the delay proof --------
        for conn in network.connections:
            key = connection_key(conn.source, conn.destination)
            history = self._release_history.setdefault(key, deque(maxlen=64))
            history.append(float(network.nodes[conn.source].state.controlled_release))

        result.latency_ms = (time.perf_counter() - t0) * 1000.0
        self.last_result = result
        return result

    # -- one-call live form ----------------------------------------------

    def step_and_check(
        self,
        network: ReservoirNetwork,
        external_inflows: Dict[str, float],
        gate_fractions: Dict[str, float],
        *,
        action_source: Optional[str] = None,
        action_percent: Optional[Dict[str, float]] = None,
        step_index: Optional[int] = None,
        timestamp: Optional[str] = None,
    ) -> MassBalanceResult:
        """
        Advance the authoritative network by one step **and** audit it.

        ``external_inflows`` and ``gate_fractions`` are passed to
        ``ReservoirNetwork.step()`` unchanged, so the recorded applied action is
        by construction the action the physics consumed.
        """
        snapshot = self.snapshot(network)
        network.step(external_inflows, gate_fractions)
        return self.verify(
            network,
            snapshot,
            applied_inflows=external_inflows,
            applied_gates_fraction=gate_fractions,
            action_source=action_source,
            action_percent=action_percent,
            step_index=step_index,
            timestamp=timestamp,
        )

    # -- lifecycle -------------------------------------------------------

    def reset(self) -> None:
        """Forget all history (called when the live cascade is re-initialised)."""
        self._release_history.clear()
        self.last_result = None
        self.checks = 0
        self.violation_count = 0
        self.ever_violated = False

    def diagnostic(self) -> Dict[str, Any]:
        """Last audit, or the honest NOT_CHECKED record."""
        if self.last_result is None:
            return not_checked_result()
        return self.last_result.to_dict()

    def performance_overhead(self, network: ReservoirNetwork, iterations: int = 200,
                             inflows: Optional[Dict[str, float]] = None,
                             gates: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Measure the diagnostic overhead against a bare ``ReservoirNetwork.step``.

        The comparison network is a clone: the live network is never advanced by
        a measurement.
        """
        import copy as _copy

        flows = dict(inflows or {nid: 0.0 for nid in network.processing_order})
        action = dict(gates or {nid: 0.0 for nid in network.processing_order})

        def _clone() -> ReservoirNetwork:
            clone = ReservoirNetwork(config_dict=_copy.deepcopy(network._raw_config))
            for nid in network.nodes:
                clone.nodes[nid].state.storage = network.nodes[nid].state.storage
            for index, conn in enumerate(network.connections):
                clone.connections[index].queue = _copy.deepcopy(conn.queue)
            return clone

        bare = _clone()
        t0 = time.perf_counter()
        for _ in range(iterations):
            bare.step(dict(flows), dict(action))
        bare_ms = (time.perf_counter() - t0) * 1000.0 / iterations

        probe = _clone()
        monitor = MassBalanceMonitor(tolerance=self.tolerance)
        t0 = time.perf_counter()
        for _ in range(iterations):
            monitor.step_and_check(probe, dict(flows), dict(action))
        audited_ms = (time.perf_counter() - t0) * 1000.0 / iterations

        return {
            "iterations": iterations,
            "bare_step_ms": bare_ms,
            "step_with_audit_ms": audited_ms,
            "overhead_ms": audited_ms - bare_ms,
            "overhead_ratio": (audited_ms / bare_ms) if bare_ms > 0 else None,
            "last_audit_latency_ms": monitor.last_result.latency_ms if monitor.last_result else None,
            "unit": "ms",
        }
