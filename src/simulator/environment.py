import json
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field

@dataclass
class ReservoirState:
    storage_mcm: float
    inflow_mcm_day: float = 0.0
    release_mcm_day: float = 0.0
    gate_position_pct: float = 0.0
    overflow_events: int = 0
    upstream_routed_inflow: float = 0.0

class VirtualReservoir:
    def __init__(self, name: str, config: Dict):
        self.name = name
        self.capacity_mcm = config["capacity_mcm"]["value"]
        self.max_release_capacity_mcm_day = config["max_release_capacity_mcm_day"]["value"]
        self.state = ReservoirState(storage_mcm=config["initial_storage_mcm"]["value"])
        # For simplicity, let's treat the Risk Engine Blue level equivalent to 75% capacity, Orange to 85%, Red to 95%
        # since we lack actual elevation-storage curves. This is a SIMULATION ASSUMPTION.
        self.blue_storage_mcm = self.capacity_mcm * 0.75
        self.orange_storage_mcm = self.capacity_mcm * 0.85
        self.red_storage_mcm = self.capacity_mcm * 0.95
        
    def get_simulated_water_level_proxy(self) -> float:
        """
        Returns a generic 0-100% full metric. We use this to feed the risk engine
        instead of physical meters, since we lack elevation curves.
        SIMULATION ASSUMPTION.
        """
        return (self.state.storage_mcm / self.capacity_mcm) * 100.0

    def step(self, local_inflow_mcm_day: float, routed_inflow_mcm_day: float, gate_command_pct: float):
        """
        Advances the reservoir state by one discrete daily timestep.
        new_storage = current_storage + inflow + upstream_routed_inflow - release
        """
        # 1. Enforce gate bounds
        gate_clamped = max(0.0, min(100.0, gate_command_pct))
        
        # 2. Calculate requested release (linear assumption)
        requested_release = (gate_clamped / 100.0) * self.max_release_capacity_mcm_day
        
        # 3. Calculate preliminary new storage
        total_inflow = local_inflow_mcm_day + routed_inflow_mcm_day
        preliminary_storage = self.state.storage_mcm + total_inflow - requested_release
        
        actual_release = requested_release
        overflow = 0
        
        # 4. Enforce physical constraints
        if preliminary_storage < 0:
            # Cannot release more than we have + inflow
            actual_release = self.state.storage_mcm + total_inflow
            preliminary_storage = 0.0
            
        elif preliminary_storage > self.capacity_mcm:
            # Overflow condition (forced spill)
            overflow_vol = preliminary_storage - self.capacity_mcm
            actual_release += overflow_vol
            preliminary_storage = self.capacity_mcm
            self.state.overflow_events += 1
            logging.warning(f"[{self.name}] OVERFLOW EVENT: {overflow_vol:.2f} MCM forcibly spilled.")
            
        # 5. Update state
        self.state.storage_mcm = preliminary_storage
        self.state.inflow_mcm_day = local_inflow_mcm_day
        self.state.upstream_routed_inflow = routed_inflow_mcm_day
        self.state.release_mcm_day = actual_release
        # Recalculate actual effective gate position 
        if self.max_release_capacity_mcm_day > 0:
             self.state.gate_position_pct = min(100.0, (actual_release / self.max_release_capacity_mcm_day) * 100.0)
        else:
             self.state.gate_position_pct = 0.0

class VirtualCascade:
    def __init__(self, config: Dict):
        self.config = config
            
        self.reservoirs = {}
        for name in self.config["topology"]["cascade_order"]:
            self.reservoirs[name] = VirtualReservoir(name, self.config["reservoirs"][name])
            
        self.cascade_order = self.config["topology"]["cascade_order"]
        self.routing_delays = self.config["topology"]["routing_delays_days"]
        self.downstream_capacity = self.config["topology"]["downstream_capacity"]["value"]
        
        # Routing buffer: queues of water traveling between reservoirs
        self.routing_queues = {
            "A_to_B": [0.0] * self.routing_delays["A_to_B"]["value"],
            "B_to_C": [0.0] * self.routing_delays["B_to_C"]["value"],
            "C_to_D": [0.0] * self.routing_delays["C_to_D"]["value"]
        }
        
        self.current_downstream_flow = 0.0

    def step(self, inflows: Dict[str, float], gate_commands: Dict[str, float]):
        """
        Advances the entire cascade by one day.
        inflows: Dict of local catchment inflow for each reservoir.
        gate_commands: Dict of gate percentage commands for each reservoir.
        """
        # Process from top to bottom
        
        # 1. Reservoir A
        res_A = self.reservoirs["Virtual Reservoir A"]
        res_A.step(inflows["Virtual Reservoir A"], 0.0, gate_commands["Virtual Reservoir A"])
        
        # Route A -> B
        routed_to_B = self.routing_queues["A_to_B"].pop(0) if self.routing_queues["A_to_B"] else 0.0
        self.routing_queues["A_to_B"].append(res_A.state.release_mcm_day)
        
        # 2. Reservoir B
        res_B = self.reservoirs["Virtual Reservoir B"]
        res_B.step(inflows["Virtual Reservoir B"], routed_to_B, gate_commands["Virtual Reservoir B"])
        
        # Route B -> C
        routed_to_C = self.routing_queues["B_to_C"].pop(0) if self.routing_queues["B_to_C"] else 0.0
        self.routing_queues["B_to_C"].append(res_B.state.release_mcm_day)
        
        # 3. Reservoir C
        res_C = self.reservoirs["Virtual Reservoir C"]
        res_C.step(inflows["Virtual Reservoir C"], routed_to_C, gate_commands["Virtual Reservoir C"])
        
        # Route C -> D
        routed_to_D = self.routing_queues["C_to_D"].pop(0) if self.routing_queues["C_to_D"] else 0.0
        self.routing_queues["C_to_D"].append(res_C.state.release_mcm_day)
        
        # 4. Reservoir D
        res_D = self.reservoirs["Virtual Reservoir D"]
        res_D.step(inflows["Virtual Reservoir D"], routed_to_D, gate_commands["Virtual Reservoir D"])
        
        # Terminal outflow to downstream river
        self.current_downstream_flow = res_D.state.release_mcm_day
        
        if self.current_downstream_flow > self.downstream_capacity:
            logging.warning(f"DOWNSTREAM CAPACITY EXCEEDED: Flow={self.current_downstream_flow:.2f}, Limit={self.downstream_capacity:.2f}")

