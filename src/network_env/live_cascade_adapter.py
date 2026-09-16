"""
Stage 3 — Live Digital Twin Physics Adapter
===========================================

Makes the validated :class:`~src.network_env.reservoir_network.ReservoirNetwork`
the ONE authoritative physics model for the **live** Digital Twin, without
modifying ``ReservoirNetwork`` and without deleting ``VirtualCascade``.

WHY AN ADAPTER (and not a rewrite)
----------------------------------
The live Digital Twin (FastAPI ``state_manager`` / Streamlit ``app.py``) talks to
``src.dashboard.sim_bridge.SimBridge``, which historically wrapped the Phase 14.4
``src.simulator.environment.VirtualCascade``.  That legacy class:

  * hardcodes exactly 4 reservoirs with names baked into Python source,
  * has NO routing attenuation (implicit factor 1.0),
  * LUMPS forced spill into ``release_mcm_day`` (no separate spill accounting),
  * ROUTES spill downstream together with the controlled release,
  * reads its routing delays from the live JSON config (1/1/1 days).

The validated ``ReservoirNetwork`` instead:

  * is topology-agnostic (graph read from YAML),
  * applies an explicit attenuation factor per connection and books the
    transmission loss as ``total_routing_loss``,
  * separates ``controlled_release`` from ``spill``,
  * does NOT route spill from non-terminal nodes (it leaves the network),
  * reads routing delays/attenuation from ``topology_config.yaml`` (2/1/1 days,
    0.90/0.85/0.80).

Rather than editing either implementation, this module **composes** the
validated network behind a facade that speaks the legacy live interface
(external PERCENT gates, MCM storages, per-reservoir objects with the same
attribute names).  ``ReservoirNetwork`` is used completely unmodified.

UNIT CONTRACT
-------------
This is the single live boundary.  Gate values arrive in the EXTERNAL
representation (PERCENT, 0-100) — exactly as the HTTP API, the UI slider and
``VirtualCascade`` require — and are converted to the canonical INTERNAL
FRACTION (0.0-1.0) exactly once, through :mod:`src.common.units`.

SEMANTIC DIFFERENCES
--------------------
Every behavioural difference between the legacy cascade and the authoritative
network is documented and measured in
``results/phase15_stage3_topology_reconciliation/PHASE_15_STAGE3_REPORT.md``
and pinned by ``tests/test_stage3_live_network_authority.py``.
"""

from __future__ import annotations

import copy
import logging
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.common import units

from .reservoir_network import ReservoirNetwork, ReservoirNode
from .mass_balance import MassBalanceMonitor, not_checked_result

logger = logging.getLogger(__name__)

#: Canonical, validated physics configuration (the authority).
DEFAULT_TOPOLOGY_PATH = Path(__file__).resolve().parent / "topology_config.yaml"


# ---------------------------------------------------------------------------
# Reservoir facade
# ---------------------------------------------------------------------------

class _LiveReservoirState:
    """
    Mutable, attribute-compatible view of a ``ReservoirState``.

    Presents the legacy ``VirtualReservoir.state`` field names while reading
    and writing straight through to the authoritative network node, so callers
    (and existing tests) can keep mutating ``res.state.storage_mcm`` etc.

    Units at this boundary:
        storage_*        : MCM
        *_mcm_day        : MCM/day
        gate_position_pct: PERCENT [0, 100]
    """

    __slots__ = ("_node",)

    def __init__(self, node: ReservoirNode):
        self._node = node

    # -- storage (MCM) ------------------------------------------------------
    @property
    def storage_mcm(self) -> float:
        return self._node.state.storage

    @storage_mcm.setter
    def storage_mcm(self, value: float) -> None:
        self._node.state.storage = float(value)

    # -- local catchment inflow (MCM/day) -----------------------------------
    @property
    def inflow_mcm_day(self) -> float:
        return self._node.state.inflow_local

    # -- routed inflow from upstream (MCM/day) ------------------------------
    @property
    def upstream_routed_inflow(self) -> float:
        return self._node.state.inflow_routed

    # -- TOTAL outflow = controlled release + spill (MCM/day) ---------------
    # Legacy ``VirtualCascade`` semantics: ``release_mcm_day`` was the total
    # water leaving the reservoir (spill lumped in). Preserved here so the
    # live twin's ``outflow`` / ``release`` keeps its established meaning.
    @property
    def release_mcm_day(self) -> float:
        return self._node.state.total_outflow

    # -- explicit, non-lumped components (Stage 3 transparency) -------------
    @property
    def controlled_release_mcm_day(self) -> float:
        """Gate-controlled release only (excludes spill)."""
        return self._node.state.controlled_release

    @property
    def spill_mcm(self) -> float:
        """Forced overflow only (excludes controlled release)."""
        return self._node.state.spill

    @property
    def transmission_loss_mcm(self) -> float:
        """Water lost during routing INTO this node (attenuation loss)."""
        return self._node.state.transmission_loss

    # -- gate (EXTERNAL percent representation) -----------------------------
    @property
    def gate_position_pct(self) -> float:
        return units.gate_fraction_to_percent(self._node.state.gate_position)

    @gate_position_pct.setter
    def gate_position_pct(self, value: float) -> None:
        self._node.state.gate_position = units.gate_percent_to_fraction(value)

    # -- cumulative overflow event counter ----------------------------------
    @property
    def overflow_events(self) -> int:
        return self._node._overflow_count


class LiveReservoirView:
    """
    Reservoir facade exposing the legacy ``VirtualReservoir`` public surface,
    backed by an authoritative ``ReservoirNode``.
    """

    def __init__(self, node: ReservoirNode):
        self._node = node
        # Stable object identity: callers may hold a reference and mutate it.
        self.state = _LiveReservoirState(node)

    @property
    def name(self) -> str:
        return self._node.node_id

    @property
    def capacity_mcm(self) -> float:
        return self._node.capacity

    @property
    def max_release_capacity_mcm_day(self) -> float:
        return self._node.max_release

    def get_simulated_water_level_proxy(self) -> float:
        """
        Storage as a PERCENT of capacity (0-100).

        IMPORTANT: this is a STORAGE PERCENTAGE, not a water level in metres.
        It must never be passed to the frozen LSTM as the ``water_level``
        feature. See ``src/common/units.py``.
        """
        return units.storage_fraction_to_percent(self._node.storage_fraction)


# ---------------------------------------------------------------------------
# Cascade adapter
# ---------------------------------------------------------------------------

class LiveCascadeAdapter:
    """
    Live Digital Twin facade whose physics is the validated ``ReservoirNetwork``.

    Parameters
    ----------
    config : dict
        The live cascade configuration (``configs/simulation/four_reservoir_demo.json``
        shape). Used for the reservoir inventory (capacity, initial storage,
        max release) and the live reservoir names.
    topology_path : str | Path | None
        Canonical validated physics configuration. Defaults to
        ``src/network_env/topology_config.yaml`` — the authority for topology,
        routing delays and attenuation.
    name_mapping : dict | None
        Explicit ``{live_name: canonical_node_id}`` override. When omitted the
        mapping is resolved from ``display_name`` and then
        ``repository_reference`` ↔ ``repository_derived_source``. Resolution
        failure raises rather than guessing.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        topology_path: Optional[str] = None,
        name_mapping: Optional[Dict[str, str]] = None,
    ):
        self.config = copy.deepcopy(config)
        self._topology_path = Path(topology_path) if topology_path else DEFAULT_TOPOLOGY_PATH
        with open(self._topology_path, "r") as fh:
            self._canonical = yaml.safe_load(fh)

        configured_order = list(self.config["topology"]["cascade_order"])
        self.name_mapping = (
            dict(name_mapping) if name_mapping
            else self._resolve_name_mapping(configured_order)
        )

        # ── The authoritative physics model ────────────────────────────────
        self.network = ReservoirNetwork(config_dict=self._build_network_config())
        self._finish_init()

    @classmethod
    def from_network_config(
        cls,
        network_config: Dict[str, Any],
        topology_path: Optional[str] = None,
    ) -> "LiveCascadeAdapter":
        """
        Build the facade over an explicitly composed ``ReservoirNetwork`` config.

        Used by the Stage 3 parity harness to run the network with
        legacy-equivalent parameters (delays 1/1/1, attenuation 1.0) so that the
        implementation can be compared against ``VirtualCascade`` in isolation
        from the validated parameter targets.
        """
        self = cls.__new__(cls)
        self.config = {
            "topology": {"cascade_order": [r["id"] for r in network_config["reservoirs"]]}
        }
        self._topology_path = Path(topology_path) if topology_path else DEFAULT_TOPOLOGY_PATH
        self._canonical = {"reservoirs": [], "connections": [], "topology": {}}
        self.name_mapping: Dict[str, str] = {}
        self.network = ReservoirNetwork(config_dict=network_config)
        self._finish_init()
        return self

    def _finish_init(self) -> None:
        """Expose the legacy-compatible public surface over ``self.network``."""
        self.cascade_order: List[str] = list(self.network.processing_order)
        self.reservoirs: Dict[str, LiveReservoirView] = {
            nid: LiveReservoirView(self.network.nodes[nid])
            for nid in self.cascade_order
        }
        self.downstream_capacity: float = self.network.downstream_capacity
        self.current_downstream_flow: float = 0.0

        # ── STAGE 11 — LIVE MASS-BALANCE INTEGRITY ──────────────────────
        # Every authoritative step of this adapter is audited. The monitor is
        # read-only (it never repairs storage, inflow, outflow or spill) and it
        # is invoked with exactly the arguments handed to
        # ``ReservoirNetwork.step()``, so the audit can only ever be about the
        # action that was actually applied.
        self.mass_balance_monitor = MassBalanceMonitor()
        self.last_mass_balance_result = None

        # Legacy ``routing_queues`` / ``routing_delays`` surface, projected
        # from the authoritative connections (read-only introspection).
        self.routing_delays: Dict[str, Dict[str, Any]] = {}
        self.routing_queues: Dict[str, deque] = {}
        for conn in self.network.connections:
            key = f"{self._short(conn.source)}_to_{self._short(conn.destination)}"
            self.routing_delays[key] = {"value": conn.delay, "unit": "days"}
            self.routing_queues[key] = conn.queue

    # ---- construction helpers ----

    @staticmethod
    def _short(name: str) -> str:
        """'Virtual Reservoir A' -> 'A'; falls back to the last token."""
        token = name.strip().split()[-1] if name.strip() else name
        return token

    def _resolve_name_mapping(self, configured_order: List[str]) -> Dict[str, str]:
        """Map live reservoir names onto canonical node ids, or raise."""
        canon = self._canonical["reservoirs"]
        by_display = {r["display_name"]: r["id"] for r in canon if r.get("display_name")}
        by_repo = {r["repository_reference"]: r["id"] for r in canon if r.get("repository_reference")}
        live = self.config.get("reservoirs", {})

        mapping: Dict[str, str] = {}
        unresolved: List[str] = []
        for name in configured_order:
            nid = by_display.get(name)
            if nid is None:
                live_repo = (live.get(name) or {}).get("repository_derived_source")
                nid = by_repo.get(live_repo) if live_repo else None
            if nid is None:
                unresolved.append(name)
            else:
                mapping[name] = nid

        if unresolved:
            raise ValueError(
                "LiveCascadeAdapter cannot map live reservoir(s) to the canonical "
                f"topology {self._topology_path}: {unresolved}. Provide an explicit "
                "name_mapping; refusing to guess a physical identity."
            )
        return mapping

    def _canonical_record(self, node_id: str) -> Dict[str, Any]:
        for rec in self._canonical["reservoirs"]:
            if rec["id"] == node_id:
                return rec
        raise ValueError(f"Canonical node '{node_id}' not present in {self._topology_path}")

    def _build_network_config(self) -> Dict[str, Any]:
        """
        Compose a ``ReservoirNetwork`` config dict:

          * reservoir inventory  — from the LIVE config (capacity, initial
            storage, max release) so live parameterisation is preserved exactly;
          * connections          — from the CANONICAL validated topology
            (delays 2/1/1, attenuation 0.90/0.85/0.80);
          * downstream capacity  — from the canonical topology.

        Node ids are the live virtual names so downstream consumers keep working
        without a second mapping layer.
        """
        live = self.config.get("reservoirs", {})
        inverse = {v: k for k, v in self.name_mapping.items()}

        reservoirs: List[Dict[str, Any]] = []
        for name in self.config["topology"]["cascade_order"]:
            canon_id = self.name_mapping[name]
            canon_rec = self._canonical_record(canon_id)
            lr = live.get(name, {})

            cap_rec = canon_rec["capacity_mcm"]
            init_rec = canon_rec["initial_storage_mcm"]
            maxrel_rec = canon_rec["max_release_mcm_day"]

            live_cap = (lr.get("capacity_mcm") or {}).get("value")
            live_init = (lr.get("initial_storage_mcm") or {}).get("value")
            live_maxrel = (lr.get("max_release_capacity_mcm_day") or {}).get("value")

            if live_cap is not None and abs(float(live_cap) - float(cap_rec["value"])) > 1e-9:
                logger.warning(
                    "[LiveCascadeAdapter] capacity drift for %s: live=%s canonical=%s "
                    "(using live value; provenance=%s)",
                    name, live_cap, cap_rec["value"], cap_rec.get("provenance"),
                )

            reservoirs.append({
                "id": name,
                "display_name": name,
                "canonical_node_id": canon_id,
                "capacity_mcm": {
                    "value": float(live_cap) if live_cap is not None else float(cap_rec["value"]),
                    "provenance": cap_rec.get("provenance", "ASSUMED_FOR_PROTOTYPE"),
                    "source": cap_rec.get("source", "topology_config.yaml"),
                },
                "initial_storage_mcm": {
                    "value": float(live_init) if live_init is not None else float(init_rec["value"]),
                    "provenance": "ASSUMED_FOR_PROTOTYPE",
                    "source": "live cascade config (SimBridge.init_cascade applies the start percentage)",
                },
                "max_release_mcm_day": {
                    "value": float(live_maxrel) if live_maxrel is not None else float(maxrel_rec["value"]),
                    "provenance": maxrel_rec.get("provenance", "ASSUMED_FOR_PROTOTYPE"),
                    "source": maxrel_rec.get("source", "topology_config.yaml"),
                },
                "warning_thresholds": canon_rec.get("warning_thresholds", {}),
            })

        connections: List[Dict[str, Any]] = []
        for conn in self._canonical["connections"]:
            src = inverse.get(conn["source"])
            dst = inverse.get(conn["destination"])
            if src is None or dst is None:
                # A canonical connection touches a node absent from the live
                # cascade; skip it explicitly rather than inventing endpoints.
                logger.info(
                    "[LiveCascadeAdapter] skipping canonical connection %s->%s "
                    "(endpoint not present in the live cascade)",
                    conn["source"], conn["destination"],
                )
                continue
            connections.append({
                "source": src,
                "destination": dst,
                "routing_delay_days": dict(conn["routing_delay_days"]),
                "attenuation_factor": dict(conn["attenuation_factor"]),
            })

        topo = self._canonical.get("topology", {})
        return {
            "reservoirs": reservoirs,
            "connections": connections,
            "topology": {
                "downstream_capacity_mcm_day": dict(topo.get("downstream_capacity_mcm_day", {"value": 50.0})),
                "topology_provenance": dict(topo.get("topology_provenance", {})),
            },
        }

    # ---- simulation ----

    def step(self, inflows: Dict[str, float], gate_commands: Dict[str, float],
             action_source: Optional[str] = None) -> None:
        """
        Advance the cascade by one day.

        Parameters
        ----------
        inflows : dict {live_name: MCM/day}
        gate_commands : dict {live_name: PERCENT in [0, 100]}

        Missing entries default to 0.0 (matching ``ReservoirNetwork`` semantics).
        Non-finite gate values raise ``UnitContractError`` at the boundary —
        never silently failing open.

        STAGE 11 — this is the ONLY place the live ``ReservoirNetwork`` is stepped,
        and the step is performed by ``MassBalanceMonitor.step_and_check``, so the
        per-step conservation audit cannot be bypassed. ``action_source`` records
        WHERE the applied action came from (e.g. ``DOWNSTREAM_CAPACITY_GUARD``,
        ``HELD_CURRENT_GATES``, ``MANUAL_OPERATOR_GATES``) for the diagnostic; it
        never influences the physics.
        """
        external_inflows: Dict[str, float] = {}
        gate_fractions: Dict[str, float] = {}

        for name in self.cascade_order:
            raw_inflow = inflows.get(name, 0.0)
            external_inflows[name] = 0.0 if raw_inflow is None else float(raw_inflow)
            # EXTERNAL percent -> INTERNAL fraction, once, at the boundary.
            gate_fractions[name] = units.gate_percent_to_fraction(gate_commands.get(name, 0.0))

        # STAGE 11 — step the authoritative physics AND audit the step it produced.
        self.last_mass_balance_result = self.mass_balance_monitor.step_and_check(
            self.network,
            external_inflows,
            gate_fractions,
            action_source=action_source,
            action_percent=dict(gate_commands),
        )
        self.current_downstream_flow = self.network.terminal_outflow

        if self.current_downstream_flow > self.downstream_capacity:
            logging.warning(
                "DOWNSTREAM CAPACITY EXCEEDED: Flow=%.2f, Limit=%.2f",
                self.current_downstream_flow, self.downstream_capacity,
            )

    def reset(self) -> None:
        """Return the cascade to its configured initial conditions."""
        self.network.reset()
        self.mass_balance_monitor.reset()
        self.last_mass_balance_result = None
        self.current_downstream_flow = 0.0

    # ---- introspection ----

    @property
    def terminal_node_id(self) -> str:
        return self.network._terminal_node_id

    def mass_balance_check(self) -> Dict[str, float]:
        """Global mass balance of the authoritative network."""
        return self.network.mass_balance_check()

    def mass_balance_diagnostic(self) -> Dict[str, Any]:
        """
        STAGE 11 — the last audited step, or the honest NOT_CHECKED record.

        Never reports PASS before a check has run.
        """
        if self.last_mass_balance_result is None:
            return not_checked_result("NO_AUTHORITATIVE_STEP_AUDITED")
        return self.last_mass_balance_result.to_dict()

    def physics_provenance(self) -> Dict[str, Any]:
        """Provenance registry of the authoritative physics parameters."""
        return {
            "topology_path": str(self._topology_path),
            "node_id_mapping": dict(self.name_mapping),
            "summary": self.network.provenance.summary(),
            "entries": {k: v.to_dict() for k, v in self.network.provenance.all_entries().items()},
        }
