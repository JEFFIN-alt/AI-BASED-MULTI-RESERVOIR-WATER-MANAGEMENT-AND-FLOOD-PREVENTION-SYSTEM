"""Daily AUTO reserve bounds from actual storage, local inflow and due queues.

30% is a prototype operating assumption, not a surveyed dam rule. Forecast
replenishment cannot authorize spending this reserve. If already below the
target, prevent further depletion rather than inventing stored water.
"""
import math


def reserve_bounds(network, inflows, fraction):
    if fraction is None:
        return None
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError('Invalid AUTO reserve fraction')
    arriving = dict.fromkeys(network.processing_order, 0.)
    for connection in network.connections:
        arriving[connection.destination] += (
            float(connection.queue[0]) * connection.attenuation if connection.queue else 0.)
    bounds, floors, targets = {}, {}, {}
    for node_id in network.processing_order:
        node = network.nodes[node_id]
        local = inflows.get(node_id, 0.)
        if isinstance(local, bool) or not math.isfinite(local) or local < 0:
            raise ValueError(f'Invalid reserve inflow for {node_id}')
        target = fraction * node.capacity
        floor = min(node.state.storage, target)
        available_release = node.state.storage + local + arriving[node_id] - floor
        bounds[node_id] = min(1., max(0., available_release / node.max_release)) if node.max_release > 0 else 1.
        floors[node_id], targets[node_id] = floor, target
    return {'enabled': True, 'fraction': fraction, 'scope': 'next daily step; actual local inflows and due routed arrivals',
            'gate_upper_bounds': bounds, 'protected_floor_mcm': floors, 'target_storage_mcm': targets,
            'assumption': 'prototype reserve; below-target states cannot be depleted further'}


def verify_reserve_action(action, bounds):
    if bounds is None:
        return
    for node, maximum in bounds['gate_upper_bounds'].items():
        if node not in action or not math.isfinite(action[node]) or action[node] > maximum + 1e-11:
            raise RuntimeError(f'AUTO storage reserve violated for {node}; action refused')
