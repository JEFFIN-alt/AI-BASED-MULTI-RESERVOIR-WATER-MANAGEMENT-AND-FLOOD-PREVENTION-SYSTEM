"""Presentation must distinguish commands, physical gates, and flow components."""
import contextlib
import io
from pathlib import Path
import shutil
import subprocess

import pytest
from fastapi.testclient import TestClient

from src.dashboard.api.app import app
from src.dashboard.api.state_manager import sim_state
from src.dashboard.twin_component.state_adapter import adapt_state_for_twin


def test_manual_request_does_not_move_displayed_physical_gate(monkeypatch):
    client = TestClient(app)
    monkeypatch.setattr(sim_state, "running", False)
    monkeypatch.setattr(sim_state, "mode", "MANUAL")
    monkeypatch.setattr(sim_state, "manual_gates", dict(sim_state.manual_gates))
    network = sim_state.bridge.cascade.network
    node = network.nodes["Virtual Reservoir A"]
    physical = node.state.gate_position
    request = 73.0 if physical != 0.73 else 27.0
    with contextlib.redirect_stdout(io.StringIO()):
        assert client.post('/api/gate/reservoir_1', json={'value': request}).status_code == 200
        state = client.get('/api/state').json()
        with client.websocket_connect('/ws/state') as ws:
            pushed = ws.receive_json()
    assert node.state.gate_position == physical
    assert state['reservoirs']['reservoir_1']['gate'] == physical
    assert state['reservoirs']['reservoir_1']['requested_gate_pct'] == request
    assert pushed['reservoirs'] == state['reservoirs']


def test_flow_components_and_reason_are_authoritative_and_optional():
    raw = {'reservoirs': {'Virtual Reservoir A': {
        'storage_pct': 100, 'outflow': 12, 'controlled_release': 5,
        'spill_mcm': 7, 'gate_position_pct': 30,
        'risk_status': 'HIGH RISK', 'risk_reason': 'Backend reason',
    }}}
    row = adapt_state_for_twin(raw)['reservoirs']['reservoir_1']
    assert row['release'] == pytest.approx(12e6 / 86400)
    assert row['controlled_release'] == pytest.approx(5e6 / 86400)
    assert row['spill_mcm'] == 7
    assert row['risk_reason'] == 'Backend reason'
    del raw['reservoirs']['Virtual Reservoir A']['controlled_release']
    del raw['reservoirs']['Virtual Reservoir A']['spill_mcm']
    row = adapt_state_for_twin(raw)['reservoirs']['reservoir_1']
    assert row['controlled_release'] is None
    assert row['spill_mcm'] is None


def test_high_storm_r3_reports_real_risk_and_flow_components(monkeypatch):
    from src.dashboard.api import state_manager
    from src.dashboard.api.state_manager import GlobalSimulationState
    # The standalone scenario has a test-only owner. Isolate its registration
    # so subsequent tests still inspect the real application's singleton list.
    monkeypatch.setattr(state_manager, '_LIVE_INSTANCES', list(state_manager._LIVE_INSTANCES))
    with contextlib.redirect_stdout(io.StringIO()):
        sim = GlobalSimulationState()
        sim.storm_intensity = 0.7
        sim.set_forecast_source('VALIDATED_REPLAY')
        sim.set_mode('AI')
        for _ in range(6):
            state = sim.step()
        raw = sim.bridge.get_state(sim._display_forecasts())
    node = sim.bridge.cascade.network.nodes['Virtual Reservoir C']
    row = state['reservoirs']['reservoir_3']
    assert row['storage'] == pytest.approx(1.0)
    assert row['risk'] == 'high risk'
    assert row['risk_reason'] == raw['reservoirs']['Virtual Reservoir C']['risk_reason']
    assert row['gate'] == node.state.gate_position
    assert row['controlled_release'] == pytest.approx(node.state.controlled_release * 1e6 / 86400)
    assert row['spill_mcm'] == node.state.spill
    assert state['control']['controller_status'] == 'ACTIVE'
    assert state['mass_balance']['status'] == 'PASS'


def test_api_feedback_does_not_confuse_acceptance_with_application():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node.js unavailable')
    source = (Path(__file__).parents[1] / 'src/dashboard/web/api.js').read_text(encoding='utf-8')
    script = """
import assert from 'node:assert/strict';
globalThis.window = {location: {protocol:'http:',host:'localhost'}};
globalThis.WebSocket = class { static OPEN=1; constructor(){this.readyState=0;} };
globalThis.setTimeout = () => 0;
const connection = [], feedback = [], states = [], health = [];
globalThis.fetch = async () => ({ok:true,json:async()=>({state_identity:{state_id:'step0-t0'},simulation:{running:false}})});
const api = new TwinAPI(s => states.push(s), s => connection.push(s), s => feedback.push(s), h => health.push(h));
api.ws.readyState=WebSocket.OPEN;
api.ws.onopen();
assert.equal(connection.at(-1), 'CONNECTING');
api.ws.onmessage({data:JSON.stringify({state_identity:{state_id:'step0-t0'},reservoirs:{}})});
await new Promise(resolve => setImmediate(resolve));
assert.equal(connection.at(-1), 'CONNECTED');
assert.equal(health.at(-1).rest, 'LIVE');
assert.equal(health.at(-1).websocket, 'LIVE');
assert.equal(health.at(-1).state, 'LIVE');
globalThis.fetch = async () => ({ok:true,json:async()=>({status:'success'})});
await api.setGate('reservoir_1',73);
assert.equal(feedback.at(-2),'COMMAND SENT');
assert.match(feedback.at(-1),/^ACCEPTED/);
assert.equal(states.length,1); // the initial authoritative frame only; commands inject no state.
globalThis.fetch = async () => ({ok:false,status:422,json:async()=>({detail:'invalid'})});
await api.setGate('reservoir_1',73);
assert.match(feedback.at(-1),/REJECTED/);
globalThis.fetch = async () => { throw Error('offline'); };
await api.pause();
assert.match(feedback.at(-1),/FAILED/);
// REST outages and late REST responses must not freeze a healthy WS display.
await api.checkFreshness();
assert.equal(connection.at(-1),'CONNECTED');
api.ws.onmessage({data:JSON.stringify({state_identity:{state_id:'step8-t8',sim_step_index:8}})});
globalThis.fetch = async () => ({ok:true,json:async()=>({state_identity:{state_id:'step7-t7',sim_step_index:7},simulation:{running:true}})});
await api.checkFreshness();
assert.equal(connection.at(-1),'CONNECTED');
const received = states.length;
api.ws.onmessage({data:JSON.stringify({state_identity:{state_id:'step6-t6',sim_step_index:6}})});
assert.equal(states.length,received);
assert.equal(api.lastStateId,'step8-t8');
api.ws.onclose();
assert.equal(connection.at(-1),'STALE DATA');
"""
    result = subprocess.run([node, '--input-type=module'], input=source + script,
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
