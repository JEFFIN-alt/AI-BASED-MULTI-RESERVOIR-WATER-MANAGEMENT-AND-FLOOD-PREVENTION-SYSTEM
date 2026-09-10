"""
Deterministic Interconnected Reservoir Network Environment.

Implements a generic N-reservoir network with:
  - Configurable topology (directed graph of connections)
  - Discrete-time daily mass balance
  - Routing delays (FIFO queues)
  - Routing attenuation with explicit transmission-loss accounting
  - Gate-controlled release with physical limits
  - Overflow/spill when capacity is exceeded
  - Negative-storage prevention
  - Full provenance tracking

Design Notes vs. src/simulator/environment.py (Phase 14.4):
  - Phase 14.4 VirtualCascade is hardcoded to exactly 4 reservoirs
    with rigid names (A, B, C, D) baked into the Python source.
  - This environment is topology-agnostic: the graph structure,
    number of nodes, and all parameters are read from a YAML config.
  - Routing attenuation is added (Phase 14.4 assumed factor=1.0).
  - Spill overflow is separated from controlled release in accounting.
  - The Phase 14.4 simulator is NOT modified or replaced. It remains
    available for the existing Streamlit dashboard/simulation page.
"""

import logging
import copy
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

import yaml
from pathlib import Path

from .provenance import Provenance, ProvenanceLevel, ProvenanceRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ReservoirState:
    """
    Complete observable state of a single reservoir at a given timestep.
    """
    storage: float          # current storage (MCM)
    inflow_local: float = 0.0        # external (catchment) inflow this step
    inflow_routed: float = 0.0       # routed inflow from upstream reservoirs
    controlled_release: float = 0.0  # gate-controlled release volume
    spill: float = 0.0               # uncontrolled overflow when capacity exceeded
    total_outflow: float = 0.0       # controlled_release + spill
    gate_position: float = 0.0       # normalised [0.0, 1.0]
    transmission_loss: float = 0.0   # water lost during routing TO this node


@dataclass
class ConnectionState:
    """
    State of a single directed connection between two reservoirs.
    Maintains a FIFO queue implementing the routing delay.
    """
    source: str
    destination: str
    delay: int                # timesteps (days)
    attenuation: float        # fraction [0, 1] of release that arrives
    queue: deque = field(default_factory=deque)

    def __post_init__(self):
        # Initialise queue to correct length filled with zeros
        if len(self.queue) == 0:
            self.queue = deque([0.0] * self.delay, maxlen=max(self.delay, 1))


# ---------------------------------------------------------------------------
# Reservoir Node
# ---------------------------------------------------------------------------

class ReservoirNode:
    """
    A single reservoir within the network.

    Mass balance (discrete daily):
        new_storage = old_storage
                    + local_inflow
                    + routed_inflow
                    - controlled_release
                    - spill

    Constraints:
        storage >= 0                  (enforced; release capped)
        storage <= capacity           (excess becomes spill)
        0 <= gate_position <= 1.0
        0 <= controlled_release <= max_release
        controlled_release <= available_water
    """

    def __init__(self, node_id: str, capacity: float, initial_storage: float,
                 max_release: float, warning_thresholds: Optional[Dict[str, float]] = None):
        self.node_id = node_id
        self.capacity = capacity
        self.max_release = max_release
        self.warning_thresholds = warning_thresholds or {}

        # Validate initial conditions
        if initial_storage < 0:
            raise ValueError(f"[{node_id}] initial_storage cannot be negative: {initial_storage}")
        if initial_storage > capacity:
            raise ValueError(f"[{node_id}] initial_storage ({initial_storage}) > capacity ({capacity})")

        self.state = ReservoirState(storage=initial_storage)
        self._cumulative_spill = 0.0
        self._overflow_count = 0

    # ---- properties ----

    @property
    def storage_fraction(self) -> float:
        """Storage as fraction of capacity [0.0, 1.0]."""
        if self.capacity <= 0:
            return 0.0
        return self.state.storage / self.capacity

    # ---- step ----

    def step(self, local_inflow: float, routed_inflow: float,
             gate_position: float) -> Tuple[float, float]:
        """
        Advance the reservoir by one timestep.

        Parameters
        ----------
        local_inflow : float
            External catchment inflow (MCM/day).
        routed_inflow : float
            Water arriving from upstream reservoir(s) after delay/attenuation.
        gate_position : float
            Normalised gate command [0.0 = closed, 1.0 = fully open].

        Returns
        -------
        (controlled_release, spill) : Tuple[float, float]
            The volumes of water leaving the reservoir.
        """
        # 1. Clamp gate position
        gate_clamped = max(0.0, min(1.0, gate_position))

        # 2. Requested release
        requested_release = gate_clamped * self.max_release

        # 3. Total inflow
        total_inflow = local_inflow + routed_inflow

        # 4. Available water for release
        available = self.state.storage + total_inflow

        # 5. Actual controlled release (cannot exceed available water)
        controlled_release = min(requested_release, available)

        # 6. Preliminary storage after release
        preliminary = available - controlled_release

        # 7. Spill check
        spill = 0.0
        if preliminary > self.capacity:
            spill = preliminary - self.capacity
            preliminary = self.capacity
            self._overflow_count += 1
            self._cumulative_spill += spill
            logger.warning(
                f"[{self.node_id}] OVERFLOW: {spill:.4f} MCM spilled "
                f"(storage capped at capacity {self.capacity:.4f} MCM)."
            )

        # 8. Final storage (should never be negative, but guard anyway)
        assert preliminary >= -1e-12, f"Negative storage {preliminary} in {self.node_id}"
        preliminary = max(0.0, preliminary)

        # 9. Update state
        self.state = ReservoirState(
            storage=preliminary,
            inflow_local=local_inflow,
            inflow_routed=routed_inflow,
            controlled_release=controlled_release,
            spill=spill,
            total_outflow=controlled_release + spill,
            gate_position=gate_clamped,
            transmission_loss=0.0,   # filled by network routing
        )

        return controlled_release, spill

    def reset(self, storage: float) -> None:
        """Reset reservoir to a given storage."""
        self.state = ReservoirState(storage=storage)
        self._cumulative_spill = 0.0
        self._overflow_count = 0


# ---------------------------------------------------------------------------
# Reservoir Network
# ---------------------------------------------------------------------------

class ReservoirNetwork:
    """
    A generic N-reservoir interconnected network environment.

    The network is defined by a YAML configuration file specifying:
      - reservoir nodes (capacity, initial storage, max release, …)
      - directed connections (source → dest, delay, attenuation)
      - downstream capacity limit

    The processing order is determined by topological sort of the
    connection graph so that upstream nodes are always processed
    before downstream nodes.

    Routing model:
      When reservoir A releases water and A→B has delay D and
      attenuation α:
        - The release enters a FIFO queue of length D.
        - After D timesteps the oldest entry leaves the queue.
        - B receives: α × queued_amount.
        - Transmission loss = (1 − α) × queued_amount.
        - Transmission losses are logged and accounted for in the
          global mass balance as "routing losses".
    """

    def __init__(self, config_path: Optional[str] = None,
                 config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialise from a YAML file path or from an in-memory dict.
        Exactly one of config_path / config_dict must be provided.
        """
        if config_path and config_dict:
            raise ValueError("Provide config_path OR config_dict, not both.")
        if config_path:
            with open(config_path, 'r') as f:
                self._raw_config = yaml.safe_load(f)
        elif config_dict:
            self._raw_config = copy.deepcopy(config_dict)
        else:
            raise ValueError("Must provide config_path or config_dict.")

        self.provenance = ProvenanceRegistry()
        self.nodes: Dict[str, ReservoirNode] = {}
        self.connections: List[ConnectionState] = []
        self._processing_order: List[str] = []
        self._downstream_capacity: float = 0.0
        self._terminal_node_id: str = ""
        self.timestep: int = 0

        # Cumulative mass-balance trackers
        self._total_external_inflow = 0.0
        self._total_routing_loss = 0.0
        self._total_terminal_outflow = 0.0
        # Spill from non-terminal nodes that is NOT routed anywhere
        # (it leaves the network as uncontrolled overflow)
        self._total_nonterminal_spill = 0.0

        self._build_from_config()

    # ---- construction ----

    def _build_from_config(self) -> None:
        cfg = self._raw_config

        # --- Reservoirs ---
        for node_cfg in cfg["reservoirs"]:
            nid = node_cfg["id"]
            cap = node_cfg["capacity_mcm"]
            init = node_cfg["initial_storage_mcm"]
            max_rel = node_cfg["max_release_mcm_day"]
            warnings = node_cfg.get("warning_thresholds", {})

            node = ReservoirNode(
                node_id=nid,
                capacity=cap["value"],
                initial_storage=init["value"],
                max_release=max_rel["value"],
                warning_thresholds=warnings,
            )
            self.nodes[nid] = node

            # Register provenance
            self.provenance.register(
                f"{nid}.capacity",
                Provenance(f"{nid} capacity", cap["value"], "MCM",
                           ProvenanceLevel(cap.get("provenance", "ASSUMED_FOR_PROTOTYPE")),
                           cap.get("source", "unknown"))
            )
            self.provenance.register(
                f"{nid}.initial_storage",
                Provenance(f"{nid} initial storage", init["value"], "MCM",
                           ProvenanceLevel(init.get("provenance", "ASSUMED_FOR_PROTOTYPE")),
                           init.get("source", "unknown"))
            )
            self.provenance.register(
                f"{nid}.max_release",
                Provenance(f"{nid} max release", max_rel["value"], "MCM/day",
                           ProvenanceLevel(max_rel.get("provenance", "ASSUMED_FOR_PROTOTYPE")),
                           max_rel.get("source", "unknown"))
            )

        # --- Connections ---
        for conn_cfg in cfg["connections"]:
            src = conn_cfg["source"]
            dst = conn_cfg["destination"]
            delay_val = conn_cfg["routing_delay_days"]
            atten_val = conn_cfg["attenuation_factor"]

            if src not in self.nodes:
                raise ValueError(f"Connection source '{src}' not in nodes.")
            if dst not in self.nodes:
                raise ValueError(f"Connection destination '{dst}' not in nodes.")

            conn = ConnectionState(
                source=src,
                destination=dst,
                delay=delay_val["value"],
                attenuation=atten_val["value"],
            )
            self.connections.append(conn)

            # Provenance
            conn_key = f"{src}->{dst}"
            self.provenance.register(
                f"{conn_key}.routing_delay",
                Provenance(f"{conn_key} routing delay", delay_val["value"], "days",
                           ProvenanceLevel(delay_val.get("provenance", "ASSUMED_FOR_PROTOTYPE")),
                           delay_val.get("source", "unknown"))
            )
            self.provenance.register(
                f"{conn_key}.attenuation",
                Provenance(f"{conn_key} attenuation factor", atten_val["value"], "fraction",
                           ProvenanceLevel(atten_val.get("provenance", "ASSUMED_FOR_PROTOTYPE")),
                           atten_val.get("source", "unknown"))
            )

        # --- Topology metadata ---
        topo = cfg.get("topology", {})
        ds_cap = topo.get("downstream_capacity_mcm_day", {})
        self._downstream_capacity = ds_cap.get("value", 50.0)
        self.provenance.register(
            "downstream_capacity",
            Provenance("downstream river capacity", self._downstream_capacity,
                       "MCM/day",
                       ProvenanceLevel(ds_cap.get("provenance", "ASSUMED_FOR_PROTOTYPE")),
                       ds_cap.get("source", "unknown"))
        )

        # Connection topology provenance
        topo_prov = topo.get("topology_provenance", {})
        self.provenance.register(
            "network_topology",
            Provenance("network topology A->B->C->D",
                       "linear cascade",
                       "graph",
                       ProvenanceLevel(topo_prov.get("provenance", "ASSUMED_FOR_PROTOTYPE")),
                       topo_prov.get("source", "unknown"))
        )

        # --- Determine processing order (topological sort) ---
        self._processing_order = self._topological_sort()
        if self._processing_order:
            self._terminal_node_id = self._processing_order[-1]

    def _topological_sort(self) -> List[str]:
        """
        Kahn's algorithm for topological ordering of the reservoir DAG.
        """
        in_degree = {nid: 0 for nid in self.nodes}
        adjacency: Dict[str, List[str]] = {nid: [] for nid in self.nodes}

        for conn in self.connections:
            adjacency[conn.source].append(conn.destination)
            in_degree[conn.destination] += 1

        queue = deque([nid for nid, deg in in_degree.items() if deg == 0])
        order = []

        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbour in adjacency[node]:
                in_degree[neighbour] -= 1
                if in_degree[neighbour] == 0:
                    queue.append(neighbour)

        if len(order) != len(self.nodes):
            raise ValueError("Network contains a cycle — not a valid DAG.")
        return order

    # ---- simulation step ----

    def step(self, external_inflows: Dict[str, float],
             gate_positions: Dict[str, float]) -> Dict[str, ReservoirState]:
        """
        Advance the entire network by one discrete daily timestep.

        Parameters
        ----------
        external_inflows : dict  {node_id: inflow_mcm_day}
        gate_positions   : dict  {node_id: position [0.0, 1.0]}

        Returns
        -------
        dict of node_id → ReservoirState (after this step).
        """
        self.timestep += 1

        # 1. Determine routed inflows arriving at each node this step
        routed_arriving: Dict[str, float] = {nid: 0.0 for nid in self.nodes}
        step_routing_loss = 0.0

        for conn in self.connections:
            # Pop the oldest entry from the FIFO queue
            if len(conn.queue) > 0:
                raw_arriving = conn.queue.popleft()
            else:
                raw_arriving = 0.0

            attenuated = raw_arriving * conn.attenuation
            loss = raw_arriving * (1.0 - conn.attenuation)

            routed_arriving[conn.destination] += attenuated
            step_routing_loss += loss

        self._total_routing_loss += step_routing_loss

        # 2. Process each node in topological order
        releases: Dict[str, float] = {}  # controlled release per node

        # Determine which nodes have outgoing connections (non-terminal for spill)
        nodes_with_outgoing = {conn.source for conn in self.connections}

        for nid in self._processing_order:
            node = self.nodes[nid]
            ext_inflow = external_inflows.get(nid, 0.0)
            routed_in = routed_arriving[nid]
            gate_pos = gate_positions.get(nid, 0.0)

            self._total_external_inflow += ext_inflow

            controlled, spill_vol = node.step(ext_inflow, routed_in, gate_pos)
            releases[nid] = controlled  # only controlled release is routed downstream

            # Spill from non-terminal nodes leaves the network entirely
            # (it is NOT routed to any downstream reservoir).
            # Spill from the terminal node is captured in terminal_outflow.
            if nid != self._terminal_node_id:
                self._total_nonterminal_spill += spill_vol

        # 3. Enqueue today's releases into routing queues for future arrival
        for conn in self.connections:
            release_vol = releases.get(conn.source, 0.0)
            conn.queue.append(release_vol)

        # 4. Track terminal outflow
        terminal_outflow = 0.0
        if self._terminal_node_id:
            terminal_node = self.nodes[self._terminal_node_id]
            terminal_outflow = terminal_node.state.total_outflow
            self._total_terminal_outflow += terminal_outflow

            if terminal_outflow > self._downstream_capacity:
                logger.warning(
                    f"DOWNSTREAM CAPACITY EXCEEDED at t={self.timestep}: "
                    f"flow={terminal_outflow:.4f} MCM/day, "
                    f"limit={self._downstream_capacity:.4f} MCM/day"
                )

        # 5. Return snapshot of all node states
        return {nid: copy.copy(self.nodes[nid].state) for nid in self.nodes}

    # ---- mass balance verification ----

    def mass_balance_check(self) -> Dict[str, float]:
        """
        Compute the global mass balance for the simulation so far.

        Conservation law:
            total_external_inflow =
                storage_change
              + total_terminal_outflow  (controlled + spill from terminal node)
              + total_nonterminal_spill (overflow from upstream nodes leaving network)
              + total_routing_loss      (attenuation losses)
              + water_in_transit        (water still in routing queues)

        Returns a dict with the components and the residual error.
        """
        current_total_storage = sum(n.state.storage for n in self.nodes.values())
        initial_total_storage = sum(
            self._raw_config["reservoirs"][i]["initial_storage_mcm"]["value"]
            for i in range(len(self._raw_config["reservoirs"]))
        )
        storage_change = current_total_storage - initial_total_storage

        water_in_transit = sum(sum(conn.queue) for conn in self.connections)

        accounted = (
            storage_change
            + self._total_terminal_outflow
            + self._total_nonterminal_spill
            + self._total_routing_loss
            + water_in_transit
        )

        residual = self._total_external_inflow - accounted

        return {
            "total_external_inflow": self._total_external_inflow,
            "storage_change": storage_change,
            "total_terminal_outflow": self._total_terminal_outflow,
            "total_nonterminal_spill": self._total_nonterminal_spill,
            "total_routing_loss": self._total_routing_loss,
            "water_in_transit": water_in_transit,
            "accounted": accounted,
            "residual_error": residual,
        }

    # ---- reset ----

    def reset(self) -> None:
        """Reset the entire network to initial conditions."""
        cfg = self._raw_config
        for node_cfg in cfg["reservoirs"]:
            nid = node_cfg["id"]
            self.nodes[nid].reset(node_cfg["initial_storage_mcm"]["value"])

        for conn in self.connections:
            conn.queue = deque([0.0] * conn.delay, maxlen=max(conn.delay, 1))

        self.timestep = 0
        self._total_external_inflow = 0.0
        self._total_routing_loss = 0.0
        self._total_terminal_outflow = 0.0
        self._total_nonterminal_spill = 0.0

    # ---- accessors ----

    @property
    def terminal_outflow(self) -> float:
        """The most recent outflow from the terminal reservoir."""
        if self._terminal_node_id:
            return self.nodes[self._terminal_node_id].state.total_outflow
        return 0.0

    @property
    def downstream_capacity(self) -> float:
        return self._downstream_capacity

    @property
    def processing_order(self) -> List[str]:
        return list(self._processing_order)
