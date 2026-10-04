"""Regression coverage for presentation reliability and honest status reporting."""
import asyncio
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.management.risk_engine import assess_risk
from src.dashboard.twin_component.state_adapter import _forecast_source_block, adapt_state_for_twin


@pytest.mark.parametrize('invalid', [None, math.nan, math.inf, True, 'missing'])
def test_missing_or_invalid_measurements_cannot_be_green(invalid):
    result = assess_risk({'waterLevel': invalid},
                         {'redLevel': 95, 'historical_95th_inflow': 10},
                         invalid, invalid, invalid)
    assert result['overall_status'] == 'UNKNOWN'
    assert result['water_level_status'] == 'UNKNOWN'
    assert result['inflow_forecast_status'] == 'INSUFFICIENT_DATA'


def test_partial_missing_data_does_not_hide_a_known_high_risk():
    result = assess_risk({'waterLevel': 100}, {'redLevel': 95}, None, None, None)
    assert result['overall_status'] == 'HIGH RISK'


def test_selecting_replay_without_eligible_data_does_not_authorize_control():
    block = _forecast_source_block({'forecast_source': 'VALIDATED_REPLAY',
                                   'auto_control': {'forecast_control_eligible': False}})
    assert block['replay_is_historical_not_live'] is True
    assert block['control_forecast_validated'] is False


def test_current_replay_records_determine_readiness_even_after_an_eligible_decision():
    rows = {f'Virtual Reservoir {letter}': {
        'forecast_status': 'VALIDATED', 'validated_metrics_apply': True,
        'is_simulated': False, 'forecast_1d': 1., 'forecast_3d': 2., 'forecast_7d': 3.
    } for letter in 'ABCD'}
    state = {'forecast_source': 'VALIDATED_REPLAY', 'reservoirs': rows,
             'auto_control': {'forecast_control_eligible': True}}
    assert _forecast_source_block(state)['control_forecast_validated'] is True
    rows['Virtual Reservoir D']['forecast_7d'] = None
    assert _forecast_source_block(state)['control_forecast_validated'] is False


def test_simulation_failure_is_visible_in_the_adapted_payload():
    state = adapt_state_for_twin({'simulation': {'running': False, 'error': 'Gate check failed'}})
    assert state['simulation']['error'] == 'Gate check failed'


def test_fractional_checked_action_survives_percent_roundtrip(monkeypatch):
    from src.dashboard.api import state_manager
    monkeypatch.setattr(state_manager, '_LIVE_INSTANCES', list(state_manager._LIVE_INSTANCES))
    sim = state_manager.GlobalSimulationState()
    try:
        sim.load_classroom_demo()
        network = sim.bridge.cascade.network
        ids = network.processing_order
        current = dict(zip(ids, [0, 0, .5, .25]))
        for node in ids:
            network.nodes[node].state.gate_position = current[node]
        network.nodes[ids[2]].state.storage = network.nodes[ids[2]].capacity
        network.nodes[ids[3]].state.storage = network.nodes[ids[3]].capacity - 10
        network.connections[-1].queue[0] = 75
        sim.manual_inflows = dict(zip(ids, [3, 4, 90, 0]))
        result = sim.mpc_orchestrator.downstream_guard.evaluate(
            network, action_fraction=current, current_fraction=current, node_ids=ids,
            max_gate_change=.5, inflows=sim.manual_inflows)
        assert result.capacity_achieved
        assert result.action_fraction[ids[2]] == pytest.approx(1 / 3)
        sim.mode = 'AI'
        sim.last_control_decision = SimpleNamespace(control_applied=True,
            final_safe_control_action_fraction=result.action_fraction,
            downstream_capacity_protection=result.to_dict())
        requested = {node: value * 100 for node, value in result.action_fraction.items()}
        assert sim._final_action_boundary(requested) == requested
        assert network.timestep == 0
        requested[ids[2]] = 0
        with pytest.raises(RuntimeError, match='diverged'):
            sim._final_action_boundary(requested)
    finally:
        sim.notification_manager.discord.stop()
        sim.notification_manager.telegram.stop()
        sim.notification_manager.executor.shutdown(wait=True)


def test_step_failure_pauses_without_killing_the_playback_task():
    from src.dashboard.api.state_manager import GlobalSimulationState
    async def probe():
        broadcast = asyncio.Event()
        def fail():
            raise RuntimeError('Deliberate test failure')
        dummy = SimpleNamespace(running=True, sim_speed=50, step=fail,
                                log_event=lambda *a: None, _record_loop_overrun=lambda *a: None)
        dummy.record_simulation_error = lambda exc: GlobalSimulationState.record_simulation_error(dummy, exc)
        async def push():
            broadcast.set()
        dummy.broadcast_state = push
        task = asyncio.create_task(GlobalSimulationState.simulation_loop(dummy))
        try:
            await asyncio.wait_for(broadcast.wait(), 1)
            assert dummy.running is False
            assert dummy.simulation_error == 'Deliberate test failure'
            assert not task.done()
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    asyncio.run(probe())
