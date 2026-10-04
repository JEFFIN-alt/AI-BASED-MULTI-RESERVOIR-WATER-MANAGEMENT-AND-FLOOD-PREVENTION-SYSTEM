"""AUTO reserve regressions: forecast mismatch and final safety composition."""
import ast
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

from src.controller.downstream_capacity_guard import DownstreamCapacityGuard
from src.controller.live_mpc_orchestrator import LiveMPCOrchestrator
from src.controller.storage_reserve import reserve_bounds, verify_reserve_action
from src.network_env.reservoir_network import ReservoirNetwork
from src.network_env.v3_forecast_adapter import V3ForecastAdapter

ROOT = Path(__file__).resolve().parents[1]


def demo_network():
    demo = json.loads((ROOT / 'configs/simulation/four_reservoir_demo.json').read_text())
    ids = list(demo['reservoirs'])
    cfg = {'reservoirs': [dict(id=n, capacity_mcm=r['capacity_mcm'],
        initial_storage_mcm={'value': r['capacity_mcm']['value'] / 2},
        max_release_mcm_day=r['max_release_capacity_mcm_day']) for n, r in demo['reservoirs'].items()],
        'connections': [dict(source=ids[i], destination=ids[i+1],
            routing_delay_days={'value': [2, 1, 1][i]},
            attenuation_factor={'value': [.9, .85, .8][i]}) for i in range(3)],
        'topology': {'downstream_capacity_mcm_day': {'value': 50}}}
    return ReservoirNetwork(config_dict=cfg, emit_warnings=False)


def final_boundary():
    # Exercise the exact production method without loading unrelated model/UI
    # dependencies or instantiating their global FastAPI simulation.
    tree = ast.parse((ROOT / 'src/dashboard/api/state_manager.py').read_text(encoding='utf-8-sig'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'GlobalSimulationState')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_final_action_boundary')
    namespace = dict(math=math, reserve_bounds=reserve_bounds, verify_reserve_action=verify_reserve_action)
    exec(compile(ast.Module(body=[method], type_ignores=[]), 'production_final_boundary', 'exec'), namespace)
    return namespace['_final_action_boundary']


class StorageReserveTests(unittest.TestCase):
    def test_only_due_routed_water_authorizes_release(self):
        net = demo_network()
        a, b, c, d = net.processing_order
        net.nodes[b].state.storage = .3 * net.nodes[b].capacity
        net.connections[0].queue[0] = 2
        net.connections[0].queue[1] = 100
        before = copy.deepcopy(list(net.connections[0].queue))
        bounds = reserve_bounds(net, {b: .5}, .3)
        self.assertAlmostEqual(bounds['gate_upper_bounds'][b], (.5 + 2 * .9) / net.nodes[b].max_release)
        self.assertEqual(list(net.connections[0].queue), before)

    def test_already_below_target_does_not_invent_water(self):
        net = demo_network()
        b = net.processing_order[1]
        net.nodes[b].state.storage = 1
        bounds = reserve_bounds(net, {b: .5}, .3)
        self.assertEqual(bounds['protected_floor_mcm'][b], 1)
        gates = dict.fromkeys(net.processing_order, 0.)
        gates[b] = bounds['gate_upper_bounds'][b]
        net.step({b: .5}, gates)
        self.assertAlmostEqual(net.nodes[b].state.storage, 1)

    def test_conflicting_rate_and_reserve_refuses_before_mutation(self):
        net = demo_network()
        b = net.processing_order[1]
        net.nodes[b].state.storage = .3 * net.nodes[b].capacity
        net.nodes[b].state.gate_position = 1.
        before = net.nodes[b].state.storage
        with self.assertRaisesRegex(RuntimeError, 'reserve and movement'):
            LiveMPCOrchestrator().mpc.decide(net, current_inflows={b: .5})
        self.assertEqual(net.timestep, 0)
        self.assertEqual(net.nodes[b].state.storage, before)

    def test_invalid_actual_inflow_cannot_authorize_release(self):
        net = demo_network()
        for invalid in (True, math.nan, math.inf, -1):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                reserve_bounds(net, {net.processing_order[1]: invalid}, .3)

    def test_downstream_replacement_retains_reserve_ceiling(self):
        net = demo_network()
        ids = net.processing_order
        a, b, c, d = ids
        net.nodes[c].state.storage = net.nodes[c].capacity
        net.nodes[d].state.storage = net.nodes[d].capacity - 10
        net.connections[-1].queue[0] = 75
        current = dict(zip(ids, [0., 0., .5, .25]))
        limits = dict(zip(ids, [1., 1., .1, .15]))
        proposal = dict(zip(ids, [0., 0., .1, .15]))
        result = DownstreamCapacityGuard().evaluate(net, action_fraction=proposal,
            current_fraction=current, node_ids=ids, max_gate_change=.5,
            inflows=dict(zip(ids, [3., 4., 90., 0.])), gate_upper_bounds=limits)
        verify_reserve_action(result.action_fraction, {'gate_upper_bounds': limits})
        self.assertEqual(net.timestep, 0)
        self.assertTrue(DownstreamCapacityGuard().safety_layer_feasible(result.action_fraction, ids, current, .5))
        self.assertEqual(result.capacity_achieved, max(result.trajectory_mcm_day) <= 50 + 1e-9)

    def test_final_boundary_rejects_checked_but_depleting_action(self):
        net = demo_network()
        ids = net.processing_order
        b = ids[1]
        net.nodes[b].state.storage = .3 * net.nodes[b].capacity
        action = dict.fromkeys(ids, 0.)
        action[b] = .5
        fake = SimpleNamespace(mode='AI', manual_inflows=dict(zip(ids, [1., .5, 8., 0.])),
            bridge=SimpleNamespace(cascade=SimpleNamespace(network=net)), mpc_orchestrator=LiveMPCOrchestrator(),
            last_control_decision=SimpleNamespace(control_applied=True,
                final_safe_control_action_fraction=action,
                downstream_capacity_protection={'predicted_flow_mcm_day': 0}))
        with self.assertRaisesRegex(RuntimeError, 'reserve violated'):
            final_boundary()(fake, {n: g * 100 for n, g in action.items()})
        self.assertEqual(net.timestep, 0)

    def test_blocked_auto_hold_still_protects_actual_reserve(self):
        net = demo_network()
        ids = net.processing_order
        b = ids[1]
        net.nodes[b].state.storage = .3 * net.nodes[b].capacity
        net.nodes[b].state.gate_position = .5
        inflows = dict(zip(ids, [1., .5, 8., 0.]))
        fake = SimpleNamespace(mode='AI', manual_inflows=inflows,
            bridge=SimpleNamespace(cascade=SimpleNamespace(network=net)),
            mpc_orchestrator=LiveMPCOrchestrator(), last_control_decision=None)
        applied = final_boundary()(fake, {n: net.nodes[n].state.gate_position * 100 for n in ids})
        self.assertAlmostEqual(applied[b], 5.)
        self.assertTrue(fake.final_safety['storage_reserve']['verified'])
        net.step(inflows, {n: g / 100 for n, g in applied.items()})
        self.assertAlmostEqual(net.nodes[b].storage_fraction, .3)

    def test_historical_forecasts_cannot_empty_ponmudi_over_12_days(self):
        import pandas as pd
        net = demo_network()
        ids = net.processing_order
        demo = json.loads((ROOT / 'configs/simulation/four_reservoir_demo.json').read_text())
        adapter = V3ForecastAdapter(str(ROOT), reservoir_mapping={n: r['repository_derived_source'] for n, r in demo['reservoirs'].items()})
        first_date = pd.Timestamp(sorted(adapter.available_dates)[0])
        controller = LiveMPCOrchestrator()
        inflows = dict(zip(ids, [1., .5, 8., 0.]))
        provenance = {n: dict(validated_metrics_apply=True, declared_status='VALIDATED', unit='MCM/day', is_simulated=False) for n in ids}
        for day in range(12):
            snapshot = adapter.get_network_snapshot((first_date + pd.Timedelta(days=day)).strftime('%Y-%m-%d'))
            # The regression retains the historical mismatch that caused depletion.
            self.assertGreater(snapshot.get(ids[1]).target_1d, inflows[ids[1]])
            before_storage = {n: net.nodes[n].state.storage for n in ids}
            decision = controller.decide(net, snapshot=snapshot, provenance=provenance, current_inflows=inflows)
            self.assertTrue(decision.control_applied)
            self.assertEqual(net.timestep, day)
            self.assertEqual(before_storage, {n: net.nodes[n].state.storage for n in ids})
            fake = SimpleNamespace(mode='AI', manual_inflows=inflows,
                bridge=SimpleNamespace(cascade=SimpleNamespace(network=net)),
                mpc_orchestrator=controller, last_control_decision=decision)
            applied = final_boundary()(fake, decision.final_safe_control_action_pct)
            net.step(inflows, {n: g / 100 for n, g in applied.items()})
            for n in ids:
                self.assertGreaterEqual(net.nodes[n].storage_fraction, .3 - 1e-10)
                self.assertAlmostEqual(net.nodes[n].state.spill, 0.)
            self.assertLessEqual(net.terminal_outflow, net.downstream_capacity + 1e-9)
            self.assertAlmostEqual(net.mass_balance_check()['residual_error'], 0., places=7)


if __name__ == '__main__':
    unittest.main()
