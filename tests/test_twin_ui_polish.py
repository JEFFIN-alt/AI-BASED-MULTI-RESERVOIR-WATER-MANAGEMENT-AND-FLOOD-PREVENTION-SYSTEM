"""Exercise frontend presentation logic without substituting a physics engine."""
from pathlib import Path
import shutil
import subprocess
import tempfile

import pytest


def test_real_gate_transition_feedback_is_not_retriggered_by_duplicate_state():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node.js unavailable')
    page = (Path(__file__).parents[1] / 'src/dashboard/web/index.html').read_text(encoding='utf-8')
    helpers = page[page.index('const setText ='):page.index('/* STAGE 12', page.index('const setText ='))]
    start = page.index('  _syncPresentation(state) {')
    method = page[start:page.index('  /* ================= STAGE 18', start)]
    script = """
import assert from 'node:assert/strict';
globalThis.document = {activeElement: null};
function element() {
  const children = new Map();
  return {textContent:'',value:'',dataset:{},animations:0,
    classList:{add(name) { if(name==='gate-applied') this.owner.animations++; },remove(){}},
    querySelector(key) { if(!children.has(key)) children.set(key, element()); return children.get(key); }};
}
function el() { const e=element(); e.classList.owner=e; return e; }
""" + helpers + '\nconst ui = {' + method + """};
ui.el = {res:Array.from({length:4},()=>({panel:el()})),
  sliders:{g:Array.from({length:4},el)},sliderVals:{g:Array.from({length:4},el)},
  autoChain:el(),mpcStatus:el(),downstreamReason:el(),autoBalance:el(),autoEligible:el(),
  autoVals:{decision:el(),safety:el(),downstream:el()}};
const state = {controller_mode:'AI',state_identity:{state_id:'step1-t1'},
  reservoirs:{reservoir_1:{node_id:'A',gate:.5,risk_reason:'Backend reason',controlled_release:12,spill_mcm:3}},
  mass_balance:{status:'PASS'},control:{downstream_reason:'Backend guard reason'},
  auto_control:{control_mode:'AI',auto_enabled:true,control_applied:true,forecast_control_eligible:true,
    controller_status:'ACTIVE',safety_layer_status:'SAFE',downstream_status:'PROTECTED',
    actions:[{node:'A',mpc_proposal_pct:50,safety_pct:50,final_pct:50}],gate_transitions:[]}};
ui._syncPresentation(state);
state.state_identity.state_id='step2-t2';
state.auto_control.gate_transitions=[{node:'A',previous_pct:70,new_pct:50}];
const original=JSON.stringify(state);
ui._syncPresentation(state);
const card=ui.el.res[0].panel;
assert.equal(card.animations,1);
assert.match(card.querySelector('[data-k="gate-feedback"]').textContent,/70.0% → 50.0%/);
ui._syncPresentation(state);
assert.equal(card.animations,1);
assert.equal(JSON.stringify(state),original);
assert.equal(card.querySelector('[data-k="reason"]').textContent,'Backend reason');
assert.equal(ui.el.autoBalance.textContent,'PASS');
state.auto_control.control_applied=false;
state.auto_control.auto_enabled=false;
state.auto_control.control_mode='MANUAL';
ui._syncPresentation(state);
assert.equal(card.querySelector('[data-k="gate-feedback"]').textContent,'');
assert.equal(card.animations,1);
"""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'presentation.mjs'
        path.write_text(script, encoding='utf-8')
        result = subprocess.run([node, str(path)], text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr
