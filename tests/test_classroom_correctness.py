"""Regression tests for the classroom audit's physical/control defects."""
import math
from pathlib import Path

import pytest

from src.controller.downstream_capacity_guard import DownstreamCapacityGuard
from src.controller.safety import SafetyLayer
from src.network_env.reservoir_network import ReservoirNetwork

ROOT = Path(__file__).resolve().parents[1]


def network():
    from src.dashboard.sim_bridge import SimBridge
    return SimBridge(str(ROOT / 'configs/simulation/four_reservoir_demo.json'),
                     str(ROOT / 'data/processed/historical_inflow_thresholds.json')).cascade.network


def test_lower_gates_are_not_a_multiday_feasibility_proof():
    net = network()
    a, b, c, d = ids = net.processing_order
    net.nodes[c].state.storage = net.nodes[c].capacity
    net.nodes[d].state.storage = net.nodes[d].capacity
    current = dict(zip(ids, [0, 0, 1, .5]))
    proposal = dict(zip(ids, [0, 0, .5, .5]))
    inflows = dict(zip(ids, [3, 4, 90, 0]))
    guard = DownstreamCapacityGuard()
    minimum = guard.minimum_admissible_action(ids, current, .5)
    assert max(guard.predict_flows(net, minimum, inflows, ids, 5)) == 60
    feasible = dict(zip(ids, [0, 0, .5, .25]))
    assert max(guard.predict_flows(net, feasible, inflows, ids, 5)) == 50
    result = guard.evaluate(net, action_fraction=proposal, current_fraction=current,
                            node_ids=ids, max_gate_change=.5, inflows=inflows)
    assert result.capacity_achieved
    assert max(result.trajectory_mcm_day) <= 50 + 1e-9
    assert guard.safety_layer_feasible(result.action_fraction, ids, current, .5)


@pytest.mark.parametrize('invalid', [math.nan, math.inf, 'bad', True])
def test_invalid_gate_fallback_still_obeys_movement_limit(invalid):
    result = SafetyLayer(.5).validate({'A': invalid}, {'A': 1.0}, ['A'])
    assert .5 <= result.validated_gates['A'] <= 1
    assert result.status == 'CORRECTED'


def test_live_mpc_places_point_forecasts_on_their_actual_days():
    from src.controller.live_mpc_orchestrator import LiveMPCOrchestrator
    from src.network_env.v3_forecast_adapter import NetworkForecastSnapshot, ReservoirForecast, ForecastStatus
    fc = ReservoirForecast('A', 'A', '2025-01-01', target_1d=10, target_3d=30,
        target_7d=70, status_1d=ForecastStatus.AVAILABLE,
        status_3d=ForecastStatus.AVAILABLE, status_7d=ForecastStatus.AVAILABLE)
    snapshot = NetworkForecastSnapshot('2025-01-01', forecasts={'A': fc})
    mpc = LiveMPCOrchestrator().mpc
    assert [s['A'] for s in mpc._build_inflow_scenarios(['A'], {'A': 0}, snapshot)] == list(range(0, 71, 10))


def test_live_mpc_decision_responds_to_the_seven_day_forecast():
    from src.controller.live_mpc_orchestrator import LiveMPCOrchestrator
    from src.network_env.v3_forecast_adapter import NetworkForecastSnapshot, ReservoirForecast, ForecastStatus
    net = network()
    mpc = LiveMPCOrchestrator().mpc
    decisions = []
    for future in (0, 20):
        forecasts = {n: ReservoirForecast(n, n, '2025-01-01', target_1d=0,
            target_3d=0, target_7d=future if n == net.processing_order[0] else 0,
            status_1d=ForecastStatus.AVAILABLE, status_3d=ForecastStatus.AVAILABLE,
            status_7d=ForecastStatus.AVAILABLE) for n in net.processing_order}
        decisions.append(mpc.decide(net, NetworkForecastSnapshot('2025-01-01', forecasts=forecasts),
                                    {n: 0 for n in net.processing_order}))
    assert decisions[0].gate_positions != decisions[1].gate_positions
    assert net.timestep == 0  # planning never advances the authoritative network


@pytest.fixture
def live(monkeypatch):
    from src.dashboard.api import state_manager
    # Isolated test servers are not additional production owners; restore registry.
    monkeypatch.setattr(state_manager, '_LIVE_INSTANCES', list(state_manager._LIVE_INSTANCES))
    sim = state_manager.GlobalSimulationState()
    # Deliveries are outside the physical demonstration, never contact providers.
    monkeypatch.setattr(sim.notification_manager, 'observe', lambda state: None)
    sim.load_classroom_demo()
    yield sim
    sim.notification_manager.discord.stop()
    sim.notification_manager.telegram.stop()
    sim.notification_manager.executor.shutdown(wait=True)


def test_manual_release_storage_routing_and_mass_balance(live):
    net = live.bridge.cascade.network
    a, b, c, d = net.processing_order
    live.step()
    assert net.nodes[a].state.storage == pytest.approx(6.41)
    live.manual_gates[a] = 50
    live.step()
    assert net.nodes[a].state.controlled_release == 2.5
    assert net.nodes[a].state.storage == pytest.approx(4.91)
    assert net.nodes[b].state.inflow_routed == 0
    live.step()
    assert net.nodes[b].state.inflow_routed == 0
    before = net.nodes[b].state.storage
    state = live.step()
    assert net.nodes[b].state.inflow_routed == 2.25
    assert net.nodes[b].state.storage == pytest.approx(before + .5 + 2.25)
    assert state['final_safety']['checked']
    assert state['mass_balance']['checked']
    assert all(0 <= n.state.storage <= n.capacity for n in net.nodes.values())


@pytest.mark.parametrize('mode', ['MANUAL', 'AI', 'ADAPTER_ERROR'])
def test_every_live_path_runs_final_safety(live, monkeypatch, mode):
    net = live.bridge.cascade.network
    d = net.processing_order[-1]
    # A held wide-open gate also requires correction, even without forecasts.
    net.nodes[d].state.gate_position = .5
    live.manual_gates[d] = 100
    live.mode = 'MANUAL' if mode == 'MANUAL' else 'AI'
    if mode == 'ADAPTER_ERROR':
        def fail(*args):
            raise ValueError('unavailable adapter')
        monkeypatch.setattr(live, '_build_live_forecast_bundle', fail)
    state = live.step()
    assert state['final_safety']['checked']
    assert net.nodes[d].state.gate_position <= .25
    assert net.nodes[d].state.total_outflow <= 50 + 1e-9
    if mode != 'MANUAL':
        assert live.last_control_decision.control_applied is False


def test_reset_discards_forecast_history_and_replay_advances(live):
    live.set_forecast_source('VALIDATED_REPLAY')
    adapter = live._replay_adapter()
    first = live._replay_date(adapter)
    live.step()
    from datetime import date, timedelta
    assert live._replay_date(adapter) == str(date.fromisoformat(first) + timedelta(days=1))
    assert any(live.history_buffers.values())
    live.load_classroom_demo()
    assert not any(live.history_buffers.values())
    assert live._replay_date(adapter) == first


def test_actual_live_mpc_action_and_final_score(live):
    live.set_forecast_source('VALIDATED_REPLAY')
    live.mode = 'AI'
    state = live.step()
    decision = live.last_control_decision
    assert decision.control_applied and decision.mpc_forecast_used
    assert decision.candidates_evaluated > 0
    assert decision.downstream_horizon_steps == 8
    assert math.isfinite(decision.final_action_objective_score)
    for n, pct in state['final_safety']['applied_gate_pct'].items():
        assert live.bridge.cascade.network.nodes[n].state.gate_position == pytest.approx(pct / 100)
    assert state['mass_balance']['matches_final_safe_control_action']
    for nid, explanation in decision.per_node.items():
        assert explanation['gate_position'] == live.bridge.cascade.network.nodes[nid].state.gate_position
        assert explanation['estimated_release'] == pytest.approx(live.bridge.cascade.network.nodes[nid].state.controlled_release)


def test_slow_websocket_cannot_block_other_clients(live, monkeypatch):
    import asyncio
    sent = []
    class Fast:
        async def send_text(self, payload):
            sent.append(payload)
    class Slow:
        closed = False
        async def send_text(self, payload):
            await asyncio.sleep(10)
        async def close(self, code):
            assert code == 1013
            self.closed = True
    slow = Slow()
    live.clients = {slow, Fast()}
    monkeypatch.setattr(live, 'get_adapted_state', lambda: {'state_identity': {'state_id': 'test'}})
    async def run():
        task = asyncio.create_task(live.broadcast_state())
        await asyncio.sleep(.05)
        assert sent and not task.done()
        await asyncio.wait_for(task, 1)
        assert slow not in live.clients
        assert slow.closed
    asyncio.run(run())


@pytest.mark.parametrize('bad', [-1.0, math.nan, math.inf])
def test_invalid_inflow_is_rejected_before_any_network_mutation(bad):
    import copy
    net = network()
    before = copy.deepcopy({n: node.state for n, node in net.nodes.items()})
    queues = [list(c.queue) for c in net.connections]
    with pytest.raises(ValueError):
        net.step({net.processing_order[-1]: bad}, {})
    assert {n: node.state for n, node in net.nodes.items()} == before
    assert [list(c.queue) for c in net.connections] == queues
    assert net.timestep == 0


def test_classroom_browser_causal_chain(live, monkeypatch, tmp_path):
    """Real HTTP, WebSocket, daily physics, frozen forecast replay and Three.js."""
    import importlib
    import json
    import socket
    import time
    from threading import Thread
    import uvicorn
    from playwright.sync_api import sync_playwright, expect

    api_module = importlib.import_module('src.dashboard.api.app')
    for module in (api_module, importlib.import_module('src.dashboard.api.routes'),
                   importlib.import_module('src.dashboard.api.state_manager')):
        monkeypatch.setattr(module, 'sim_state', live)
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    listener.listen(128)
    base = f'http://127.0.0.1:{listener.getsockname()[1]}'
    server = uvicorn.Server(uvicorn.Config(api_module.app, log_level='warning', access_log=False))
    thread = Thread(target=server.run, kwargs={'sockets': [listener]}, daemon=True)
    thread.start()
    frames, errors, evidence = [], [], {'transport': 'real local HTTP + WebSocket', 'steps': []}
    backend_times = []
    original_step = live.step
    def timed_step():
        start = time.perf_counter()
        result = original_step()
        backend_times.append((time.perf_counter() - start) * 1000)
        return result
    monkeypatch.setattr(live, 'step', timed_step)
    try:
        deadline = time.monotonic() + 15
        while not server.started and time.monotonic() < deadline:
            time.sleep(.02)
        assert server.started
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True, args=['--enable-unsafe-swiftshader'])
            try:
                page = browser.new_page(viewport={'width': 1440, 'height': 900})
                page.set_default_timeout(60000)
                # Use the existing presentation preset for software-rendered CI.
                # No physical state or controller output is mocked.
                page.add_init_script("localStorage.setItem('aquaflow-visual-quality', 'PERFORMANCE')")
                page.on('pageerror', lambda e: errors.append(str(e)))
                def received(payload):
                    value = json.loads(payload)
                    if 'state_identity' in value:
                        frames.append(value)
                page.on('websocket', lambda ws: ws.on('framereceived', received))
                page.goto(base, wait_until='domcontentloaded')
                page.wait_for_function("window.__twin_api?.lastStateAt > 0", timeout=90000)
                page.locator('[data-nav="simulation"]').click()
                with page.expect_response('**/api/simulation/classroom-demo'):
                    page.locator('[data-classroom-load]').click()

                def step():
                    day = live.bridge.cascade.network.timestep + 1
                    start = time.perf_counter()
                    with page.expect_response('**/api/simulation/step') as response:
                        page.locator('#btn-step').click()
                    assert response.value.status == 200
                    page.wait_for_function('(day) => window.__twin_api?.lastStateId?.endsWith("-t" + day)', arg=day)
                    state = page.request.get(base + '/api/state').json()
                    assert frames[-1]['state_identity'] == state['state_identity']
                    evidence['steps'].append({'click_to_observed_state_ms': (time.perf_counter()-start)*1000,
                        'backend_step_ms': backend_times[-1],
                        'state_id': state['state_identity']['state_id'],
                        'telemetry': state['classroom_demo'], 'safety': state['final_safety'],
                        'control': state['control'], 'mass_balance': state['mass_balance']})
                    return state

                first = step()
                assert first['classroom_demo']['reservoirs'][0]['storage'] == pytest.approx(6.41)
                before_visual = page.evaluate('window.__twinDiagnostics.reservoirs[0].gateY')
                with page.expect_response('**/api/gate/reservoir_1') as command:
                    slider = page.locator('[data-ctl="g1"]')
                    slider.evaluate("el => el.value = '50'")
                    slider.dispatch_event('input')
                assert command.value.json()['gate'] == 50
                evidence['gate_command'] = command.value.json()
                second = step()
                a = second['classroom_demo']['reservoirs'][0]
                assert a['gate_pct'] == 50 and a['controlled_release'] == 2.5
                assert a['storage'] == pytest.approx(4.91)
                page.wait_for_function('(old) => Math.abs(window.__twinDiagnostics.reservoirs[0].gateY-old) > .01', arg=before_visual)
                page.wait_for_function('Math.abs(window.__twinDiagnostics.reservoirs[0].gateY-window.__twinDiagnostics.reservoirs[0].gateTargetY) < .1')
                assert step()['classroom_demo']['reservoirs'][1]['routed_inflow'] == 0
                fourth = step()
                assert fourth['classroom_demo']['reservoirs'][1]['routed_inflow'] == 2.25
                expect(page.locator('[data-classroom-panel]')).to_contain_text('2.25')
                evidence['manual_visual'] = page.evaluate('window.__twinDiagnostics')
                assert evidence['manual_visual']['rendering']['contextLost'] is False
                assert evidence['manual_visual']['fps'] > 0
                page.screenshot(path=str(tmp_path / 'classroom_manual.png'), full_page=True)

                # Exercise the actual UI forecast-source and AUTO controls.
                with page.expect_response('**/api/controller/mode'):
                    page.locator('#ctl-source [data-source="VALIDATED_REPLAY"]').click()
                with page.expect_response('**/api/controller/mode'):
                    page.locator('#ctl-mode [data-mode="AUTO"]').click()
                ai = step()
                assert ai['control']['control_applied'] and ai['control']['mpc_forecast_used']
                assert ai['mass_balance']['matches_final_safe_control_action']
                assert ai['final_safety']['applied_gate_pct'] != fourth['final_safety']['applied_gate_pct']
                assert ai['control']['downstream_horizon_steps'] == 8
                evidence['ai_visual'] = page.evaluate('window.__twinDiagnostics')
                page.screenshot(path=str(tmp_path / 'classroom_ai.png'), full_page=True)

                with page.expect_response('**/api/controller/mode'):
                    page.locator('#ctl-mode [data-mode="MANUAL"]').click()
                with page.expect_response('**/api/gate/reservoir_4'):
                    page.locator('[data-ctl="g4"]').evaluate("el => el.value = '100'")
                    page.locator('[data-ctl="g4"]').dispatch_event('input')
                guarded = step()
                assert guarded['final_safety']['requested_gate_pct']['Virtual Reservoir D'] == 100
                assert guarded['final_safety']['applied_gate_pct']['Virtual Reservoir D'] <= 25
                assert not errors
                evidence.update(result='PASS', websocket_frames=len(frames), page_errors=errors)
                (tmp_path / 'classroom_evidence.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
                print('CLASSROOM_EVIDENCE=' + str(tmp_path))
            finally:
                browser.close()
    finally:
        server.should_exit = True
        server.force_exit = True
        thread.join(timeout=20)
        listener.close()
    # In heavily loaded Windows runners Uvicorn may take longer to finish its
    # lifespan worker shutdown; the listener is already closed and isolated.
