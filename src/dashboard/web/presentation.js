// Presentation UI consumes the same authoritative payload and TwinAPI as the 3D scene.
// It never advances physics, manufactures forecasts, or decides control eligibility.
export const COLORS = ['#278565', '#269ca7', '#537dcb', '#b88938'];
export const finite = value => typeof value === 'number' && Number.isFinite(value);
export const number = (value, digits = 2) => finite(value) ? value.toLocaleString('en-IN', {maximumFractionDigits: digits}) : '—';
export const escapeHTML = value => String(value ?? '—').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
export const mcmDay = value => finite(value) ? value * 86400 / 1000000 : null;
export function reserveMessage(state) {
  if (state.controller_mode !== 'AI') return 'Manual mode: AUTO storage reserve is inactive.';
  const checked = state.final_safety, reserve = checked?.storage_reserve;
  if (checked?.mode === 'AI' && reserve?.enabled && reserve.verified && finite(reserve.fraction))
    return `AUTO reserve ${number(reserve.fraction * 100, 0)}% · checked against actual daily inflow and due arrivals. Below-target storage is protected from further depletion.`;
  return 'AUTO storage reserve: awaiting the next step check.';
}
export function guideFor(state) {
  const day = state.classroom_demo?.day;
  if (!state.classroom_demo?.active) return ['START HERE', 'Initialize the standard scenario: 50% storage, closed gates and fixed local inflows.'];
  if (state.controller_mode === 'AI') return ['04 / COORDINATED CONTROL', 'STEP to apply MPC. Compare its proposal with the final gates and downstream check.'];
  if (day === 0) return ['01 / STORAGE BALANCE', 'STEP once with gates closed. A gains 1 MCM of local inflow.'];
  if (day === 1) return ['02 / RELEASE', 'Request A at 50%, then STEP. Watch A release 2.5 MCM/day and its storage fall.'];
  if (day === 2 || day === 3) return ['03 / ROUTING DELAY', 'STEP until day 4. The day-2 release reaches B after two days: 2.5 × 0.90 = 2.25 MCM.'];
  if (day === 4) return ['04 / COORDINATED CONTROL', 'Choose Historical replay, then AUTO and STEP. Stored LSTM forecasts now feed MPC.'];
  return ['05 / SAFETY BOUNDARY', 'Use MANUAL, request D at 100%, then STEP. Compare requested and applied gates.'];
}
export class PresentationUI {
  constructor(app, api) {
    this.app = app; this.api = api; this.state = {}; this.history = []; this.tab = 'demo'; this.connection = 'CONNECTING';
    this.active = true; this.busy = false; this.graph = null;
    this.root = document.createElement('div'); this.root.className = 'p-shell';
    this.root.innerHTML = `
      <header class="p-header">
        <div class="p-brand">Aqua <em>Flow</em><small>Reservoir coordination</small></div>
        <div class="p-claim"><strong>Coordinated reservoir control</strong><small>Daily simulation · forecast-informed release planning</small></div>
        <div class="p-header-state"><div class="p-day"><strong data-p="day">—</strong><small>simulation day</small></div><div class="p-connection"><span class="p-status" data-p="connection">Connecting</span><div data-p="running">Awaiting backend</div></div></div>
      </header>
      <aside class="p-sidebar">
        <h2>Scenario controls</h2><p class="p-muted">One step = one physical day.</p>
        <button class="p-load" data-action="load">Initialize scenario</button>
        <div class="p-guide"><small data-p="guide-title"></small><p data-p="guide-text"></p></div>
        <div class="p-playback"><button class="p-btn" data-action="play">▶ Play</button><button class="p-btn" data-action="pause">Ⅱ Pause</button><button class="p-btn primary" data-action="step">+1 day · Step</button></div>
        <div class="p-control-label">Gate control <span data-p="owner">—</span></div>
        <div class="p-segment"><button class="p-btn" data-mode="MANUAL">Manual</button><button class="p-btn" data-mode="AI">AUTO · MPC</button></div>
        <div class="p-control-label">Forecast input to MPC</div>
        <div class="p-segment"><button class="p-btn" data-source="SIMULATION">Simulation inputs</button><button class="p-btn" data-source="VALIDATED_REPLAY">Historical replay</button></div>
        <p class="p-source-note" data-p="source-note"></p>
        <p class="p-source-note" data-p="reserve-status"></p>
        <p class="p-gate-hint">Edit <strong>Requested %</strong> in the table below, then STEP.</p>
        <details class="p-decision" data-p="decision"><summary data-p="decision-title">Awaiting a safety check</summary><span data-p="decision-text">STEP to evaluate the applied action.</span></details>
        <div class="p-command" data-p="command" role="status" aria-live="polite"></div>
        <details class="p-extra"><summary>Playback &amp; extra controls</summary>
          <label>Days / second <input type="range" min="0.1" max="2" step="0.1" value="1" data-speed aria-label="Playback days per second"><output data-p="speed">1</output></label>
          <label title="Fixed local inflows override storm input in this scenario">Storm input <input type="range" min="0" max="100" value="0" data-storm aria-label="Storm intensity"></label>
          <p class="p-muted" data-p="storm-note"></p><button class="p-btn" data-action="reset">Exit preset / reset</button>
        </details>
      </aside>
      <nav class="p-toolbar" aria-label="Presentation views"><div class="p-tabs"><button class="p-btn on" data-tab="demo">Simulation</button><button class="p-btn" data-tab="models">Models &amp; control</button><button class="p-btn" data-tab="assumptions">Assumptions</button></div><div class="p-camera"><select data-camera aria-label="Scene camera"><option value="overview">Whole cascade</option><option value="reservoir1">A · Anayirankal</option><option value="reservoir2">B · Ponmudi</option><option value="reservoir3">C · Idamalayar</option><option value="reservoir4">D · Idukki arch dam</option><option value="downstream">Downstream</option></select><button class="p-btn" data-action="theme" title="Switch light / dark theme">◐</button><button class="p-btn" data-action="fullscreen" title="Fullscreen presentation">⛶</button><button class="p-btn" data-action="dashboard" title="Open the full engineering dashboard">More</button></div></nav>
      <div class="p-flow" data-p="routes"></div>
      <div class="p-scene-caption"><span>Schematic cascade · water height represents storage fraction</span><span>Drag to orbit · scroll to zoom</span></div>
      <section class="p-stage-panel" data-panel="models" hidden>
        <h2>Forecasts, planning &amp; safety</h2><p>The spatial analysis and the physical cascade have different roles.</p>
        <div class="p-chain"><div><strong>LSTM V3 →</strong><p>Frozen model forecasts inflow at +1, +3 and +7 days.</p><small data-p="lstm-status">Awaiting forecasts</small></div><div><strong>MPC →</strong><p>Plans joint gate actions across four reservoirs and an eight-step horizon.</p><small data-p="mpc-status">Awaiting decision</small></div><div><strong>Safety checks →</strong><p>Gate bounds, movement limits, AUTO storage reserve and predicted downstream capacity.</p><small data-p="safety-status">Not checked</small></div><div><strong>Daily simulation</strong><p>Applies final gates, updates storage and routes released water.</p><small data-p="balance-status">Not checked</small></div></div>
        <div class="p-model-grid"><article class="p-model-card"><h3>GCN-LSTM · spatial-temporal advisory</h3><p>The gated model analyses statistical dependencies and seven-day reservoir histories. Its output is advisory; it does not issue gate commands.</p><div class="p-status" data-p="gnn-status">Awaiting model status</div><svg class="p-graph" data-p="graph" viewBox="0 0 500 310" role="img" aria-label="Training correlation graph"></svg><p class="p-muted" data-p="graph-note">Loading the recorded training graph.</p></article><article class="p-model-card"><h3>LSTM forecast inputs · MCM/day</h3><table class="p-table"><thead><tr><th>Reservoir</th><th>+1 day</th><th>+3 days</th><th>+7 days</th></tr></thead><tbody data-p="forecasts"></tbody></table><h3>Controller explanation</h3><p data-p="controller-reason">Awaiting a backend decision.</p><h3>Storage protection</h3><p data-p="reserve-explanation"></p><h3>Forecast provenance</h3><p data-p="provenance"></p><h3>Most recent MPC decision · %</h3><p data-p="action-context"></p><table class="p-table"><thead><tr><th>Node</th><th>MPC</th><th>Gate safety</th><th>Final</th></tr></thead><tbody data-p="actions"></tbody></table></article></div>
      </section>
      <section class="p-stage-panel" data-panel="assumptions" hidden><h2>What this prototype represents</h2><p>A daily decision-support experiment on an assumed four-reservoir cascade.</p><div class="p-equation">S(next) = S + local inflow + routed inflow − release − spill</div><div class="p-assumption-grid">
        <article><h3>Time &amp; units</h3><p>One STEP is one day. Storage is MCM; inflow and release are MCM/day. Playback changes viewing cadence only.</p></article>
        <article><h3>Routing</h3><p>A → B: 2 days × 0.90. B → C: 1 day × 0.85. C → D: 1 day × 0.80. Water in transit and transmission losses enter the mass-balance audit.</p></article>
        <article><h3>Prototype geography &amp; graphics</h3><p>Reservoir names identify dataset nodes. This cascade is not verified Kerala connectivity. Reservoir sizes and Idukki's arch are schematic; they are not surveyed structures.</p></article>
        <article><h3>Forecast replay</h3><p>Historical replay uses stored forecasts from the frozen LSTM model. It is a historical what-if experiment on simulated states, not a prediction of the preset's current inflows.</p></article>
        <article><h3>Simplified hydraulics</h3><p>Release is proportional to gate opening and capped by available water. Water height is a storage proxy. Evaporation is omitted. Upstream spill exits through separate assumed lateral outlets.</p></article>
        <article><h3>Scope of the safety check</h3><p>Manual and AUTO share gate and downstream checks. AUTO additionally protects a 30% storage reserve using actual daily inflow and due arrivals. This is a prototype assumption, not a dam operating rule. Below-target storage cannot be depleted further. Conflicting movement limits stop the step. Downstream protection is conditional on the tested forecast scenario and finite action search. Hardware, RL and continual learning are future work.</p></article>
      </div></section>
      <div class="p-error" data-p="error" role="alert" hidden></div>
      <footer class="p-bottom"><section class="p-telemetry"><div class="p-section-title">Reservoir state <small>Storage: MCM · flows: MCM/day · gates: %</small></div><table class="p-table"><thead><tr><th>Reservoir</th><th>Storage</th><th>Local in</th><th>Routed in</th><th>Release</th><th>Spill</th><th data-p="request-heading">Requested %</th><th>Applied %</th></tr></thead><tbody data-p="telemetry">${COLORS.map((c,i)=>`<tr data-row="${i+1}" style="--node-color:${c}"><td><strong>${'ABCD'[i]}</strong><span data-cell="name"></span></td><td><span data-cell="storage">—</span><span class="p-percent" data-cell="percent">—</span></td><td data-cell="local">—</td><td data-cell="routed">—</td><td data-cell="release">—</td><td data-cell="spill">—</td><td><input type="number" min="0" max="100" step="0.1" data-gate="${i+1}" aria-label="Reservoir ${'ABCD'[i]} requested gate opening" placeholder="—"></td><td data-cell="applied">—</td></tr>`).join('')}</tbody></table></section><section class="p-trend"><div class="p-section-title">Storage over simulation days <small>%</small></div><svg data-p="trend" viewBox="0 0 280 112" role="img" aria-label="Actual storage history"></svg><div class="p-legend">${COLORS.map((c,i)=>`<span style="--node-color:${c}">● ${'ABCD'[i]}</span>`).join('')}</div></section></footer>`;
    app.container.appendChild(this.root); app.container.classList.add('presentation-mode');
    this.root.addEventListener('click', event => this.onClick(event));
    this.root.addEventListener('change', event => this.onChange(event));
    this.root.addEventListener('keydown', event => {if(event.key === 'Enter' && event.target.dataset.gate) event.target.blur();});
    app.fLabels?.forEach((label,i) => label.addEventListener('click', event => {
      if(!this.active || i > 3) return;
      event.stopImmediatePropagation();
      const camera='reservoir'+(i+1); app.setCameraPreset(camera); this.root.querySelector('[data-camera]').value=camera;
    },true));
    this.resize = () => { if (this.active) app._onResize(); };
    window.addEventListener('resize', this.resize);
    this.update(app.data.rawState || {}); this.resize();
    fetch('/presentation-graph.json').then(r => r.ok ? r.json() : Promise.reject()).then(graph => {this.graph = graph; this.renderGraph();}).catch(() => this.text('graph-note','Training graph unavailable. Advisory status remains backend-reported.'));
    document.querySelector('[data-nav="simulation"]')?.addEventListener('click', () => {this.show();});
  }
  el(key) { return this.root.querySelector(`[data-p="${key}"]`); }
  text(key, value) { const e = this.el(key); if(e && e.textContent !== String(value)) e.textContent = value; }
  viewport() {
    if (!this.active) return null;
    const styles = getComputedStyle(this.app.container), side = parseFloat(styles.getPropertyValue('--present-side'));
    return {left:side + (innerWidth <= 850 ? 24 : 40), top:158, right:innerWidth <= 850 ? 12 : 24, bottom:parseFloat(styles.getPropertyValue('--present-bottom'))};
  }
  setConnection(status) {
    if(this.state.preview_notice) status = 'RECORDED PREVIEW';
    this.connection = status; this.text('connection', status === 'RECORDED PREVIEW' ? 'Layout preview' : status === 'CONNECTED' ? 'Backend live' : status === 'CONNECTING' ? 'Connecting' : status);
    this.el('connection').className = 'p-status ' + (status === 'CONNECTED' ? 'ok' : ['CONNECTING','RECORDED PREVIEW'].includes(status) ? 'warn' : 'bad');
    this.syncControls();
  }
  setCommand(status) { this.text('command', status); }
  show() { this.active = true; this.root.hidden = false; this.app.container.classList.add('presentation-mode'); this.setTab('demo'); this.resize(); this.update(this.app.data.rawState || {}); }
  hide() { this.active = false; this.root.hidden = true; this.app.fLabels?.forEach(label => label.style.visibility = ''); this.app.container.classList.remove('presentation-mode'); this.app._onResize(); }
  setTab(tab) {
    this.tab = tab;
    this.root.classList.toggle('p-explanation', tab !== 'demo');
    this.root.querySelector('.p-bottom').hidden = tab !== 'demo';
    this.root.querySelector('.p-gate-hint').innerHTML = tab === 'demo'
      ? 'Edit <strong>Requested %</strong> in the table below, then STEP.'
      : 'Return to Simulation to inspect storage and edit manual gate requests.';
    this.root.querySelectorAll('[data-tab]').forEach(e => e.classList.toggle('on', e.dataset.tab === tab));
    this.root.querySelectorAll('[data-panel]').forEach(e => e.hidden = e.dataset.panel !== tab);
    this.app.fLabels?.forEach(label => label.style.visibility = tab === 'demo' ? '' : 'hidden');
    this.root.querySelector('.p-scene-caption').hidden = tab !== 'demo'; this.el('routes').hidden = tab !== 'demo';
  }
  async command(call) {
    if(this.busy) return;
    this.busy = true; this.syncControls();
    try { await call(); } catch(error) { this.setCommand('Command failed: ' + error.message); }
    finally {this.busy = false; this.syncControls();}
  }
  async onClick(event) {
    const button = event.target.closest('button'); if(!button) return;
    if(button.dataset.tab) { this.setTab(button.dataset.tab); return; }
    if(button.dataset.mode) { await this.command(() => this.api.setMode(button.dataset.mode)); return; }
    if(button.dataset.source) { await this.command(() => this.api.setMode(this.state.controller_mode || 'MANUAL', button.dataset.source)); return; }
    const action = button.dataset.action;
    if(action === 'dashboard') { this.hide(); document.querySelector('[data-nav="home"]')?.click(); return; }
    if(action === 'theme') { document.getElementById('theme-toggle')?.click(); return; }
    if(action === 'fullscreen') { try { if(document.fullscreenElement) await document.exitFullscreen(); else await this.app.container.requestFullscreen(); } catch(error) { this.setCommand('Fullscreen unavailable: '+error.message); } return; }
    const handlers = {load:()=>this.api.classroomDemo(), play:()=>this.api.play(), pause:()=>this.api.pause(), step:()=>this.api.step(), reset:()=>this.api.reset()};
    if(handlers[action]) await this.command(handlers[action]);
  }
  async onChange(event) {
    const e = event.target;
    if(e.matches('[data-camera]')) {this.app.setCameraPreset(e.value);return;}
    if(e.dataset.gate) {
      if(e.value === '' || !e.validity.valid) {this.setCommand('Enter a gate opening between 0% and 100%.');return;}
      await this.command(()=>this.api.setGate('reservoir_'+e.dataset.gate, Number(e.value)));
    }
    if(e.matches('[data-speed]')) await this.command(()=>this.api.setSpeed(Number(e.value)));
    if(e.matches('[data-storm]')) await this.command(()=>this.api.setStorm(Number(e.value)));
  }
  syncControls() {
    const connected = this.connection === 'CONNECTED';
    this.root.querySelectorAll('[data-action="load"],[data-action="play"],[data-action="pause"],[data-action="step"],[data-action="reset"],[data-mode],[data-source],[data-speed]').forEach(e=>e.disabled = this.busy || !connected);
    this.root.querySelectorAll('[data-gate]').forEach(e=>e.disabled = this.busy || !connected || this.state.controller_mode === 'AI');
    this.root.querySelector('[data-storm]').disabled = this.busy || !connected || this.state.classroom_demo?.active === true;
  }
  update(state) {
    this.state = state;
    const rs = state.reservoirs || {}, demo = state.classroom_demo || {}, sim = state.simulation || {}, ac = state.auto_control || {}, final = state.final_safety || {}, downstream = final.downstream || {};
    const rows = demo.reservoirs || [], day = demo.day ?? state.state_identity?.network_timestep;
    const auto = state.controller_mode === 'AI', source = state.forecast_source_selection?.selected, replay = source === 'VALIDATED_REPLAY';
    this.text('day', number(day, 0)); this.text('running', state.preview_notice ? 'Recorded · controls disabled' : sim.error ? 'Paused · step error' : sim.running === true ? 'Playing · '+number(sim.speed,1)+' days/s' : sim.running === false ? 'Paused · daily steps' : 'Awaiting backend');
    this.root.querySelector('[data-action="load"]').textContent = demo.active ? 'Restart scenario' : 'Initialize scenario';
    this.root.querySelector('[data-action="load"]').title = state.preview_notice ? 'Layout preview only. Start python app.py to initialize a live scenario.' : 'Reset to 50% storage, closed gates and fixed local inflows.';
    const guide = guideFor(state); this.text('guide-title', guide[0]); this.text('guide-text', guide[1]);
    this.text('owner', auto ? 'MPC proposes' : state.controller_mode === 'MANUAL' ? 'Operator requests' : 'Awaiting mode');
    this.root.querySelectorAll('[data-mode]').forEach(e=>e.classList.toggle('on',e.dataset.mode === state.controller_mode));
    this.root.querySelectorAll('[data-source]').forEach(e=>e.classList.toggle('on',e.dataset.source === source));
    this.text('source-note', replay ? 'Historical what-if: stored LSTM forecasts. AUTO eligibility is checked by the backend.' : 'Simulation-derived inputs. AUTO may hold when forecasts are not eligible.');
    const reserveText = reserveMessage(state);
    this.text('reserve-status', reserveText); this.text('reserve-explanation', reserveText);
    this.text('storm-note', demo.active ? 'Fixed preset inflows: storm control is disabled.' : 'Storm input changes the simulator inflow scenario.');
    this.text('speed', number(sim.speed, 1));
    const speed = this.root.querySelector('[data-speed]'); if(document.activeElement !== speed && finite(sim.speed)) speed.value = sim.speed;
    const storm = this.root.querySelector('[data-storm]'); if(document.activeElement !== storm && finite(state.storm_intensity)) storm.value = state.storm_intensity * 100;
    this.el('error').hidden = !sim.error; this.text('error', sim.error ? 'Simulation paused: '+sim.error+' · Restart the scenario to recover.' : '');
    this.text('request-heading', auto ? 'MPC proposal %' : 'Requested %');
    COLORS.forEach((color,i)=>{
      const r = rs['reservoir_'+(i+1)] || {}, physical = rows.find(row=>row.node === r.node_id) || {}, action = (ac.actions || []).find(a=>a.node === r.node_id) || {};
      const request = auto ? action.mpc_proposal_pct : r.requested_gate_pct;
      const input = this.root.querySelector(`[data-gate="${i+1}"]`);
      if(document.activeElement !== input) input.value = finite(request) ? Math.round(request*10)/10 : '';
      const row=this.root.querySelector(`[data-row="${i+1}"]`);
      const cells={name:r.repository_name || this.app.resNames[i],storage:number(physical.storage),percent:finite(r.storage)?number(r.storage*100,0)+'%':'—',local:number(physical.local_inflow),routed:number(physical.routed_inflow),release:number(physical.controlled_release),spill:number(physical.spill),applied:number(finite(r.gate)?r.gate*100:null,1)};
      Object.entries(cells).forEach(([key,value])=>{row.querySelector(`[data-cell="${key}"]`).textContent=value;});
    });
    this.el('routes').innerHTML = (demo.routes || []).map(r=>`<span>${escapeHTML(r.source.slice(-1))} → ${escapeHTML(r.destination.slice(-1))} · ${number(r.delay_days,0)}d · ×${number(r.attenuation)} · queued ${number(r.queued_mcm)} MCM</span>`).join('');
    const achieved = downstream.capacity_achieved;
    this.text('decision-title', !final.checked ? 'Awaiting a safety check' : achieved === true ? 'Downstream protected in checked scenario' : achieved === false ? 'Downstream protection not achieved' : 'Downstream verdict unavailable');
    this.el('decision').className = 'p-decision ' + (achieved === true ? 'ok' : achieved === false ? 'bad' : '');
    this.text('decision-text', final.checked ? `Gate check: ${final.gate_status || 'unknown'}. `+(downstream.reason || 'No downstream explanation supplied.') : 'STEP to evaluate the applied action.');
    this.text('lstm-status', replay ? 'Historical forecast replay' : state.forecast_summary?.status || 'Awaiting forecasts');
    this.text('mpc-status', ac.state || 'Awaiting decision'); this.text('safety-status', final.checked ? final.gate_status || 'Unknown' : 'Not checked');
    this.text('balance-status', state.mass_balance?.checked ? 'Mass balance: '+state.mass_balance.status : 'Mass balance not checked');
    this.text('controller-reason', ac.reason || 'Awaiting a backend decision.');
    this.text('provenance', replay ? 'Stored outputs from frozen LSTM V3 on historical data. These are not forecasts of the current scenario inflows.' : 'LSTM runs on simulation-derived features. Missing measured inputs remain labelled; model test metrics do not apply to this scenario.');
    this.text('gnn-status', (state.gnn_advisory?.status || 'UNAVAILABLE').replaceAll('_',' '));
    this.text('action-context', !(ac.actions || []).length ? 'No MPC action has been recorded.' : auto
      ? 'Proposal, gate safety and final action recorded for the most recent MPC decision.'
      : 'Recorded AUTO decision. Manual steps may change gates; see Simulation → Applied % for current positions.');
    this.el('forecasts').innerHTML = COLORS.map((c,i)=>{const r=rs['reservoir_'+(i+1)] || {};return `<tr><td>${'ABCD'[i]} · ${escapeHTML(r.repository_name || this.app.resNames[i])}</td>${['1d','3d','7d'].map(h=>`<td>${number(mcmDay(r['forecast_'+h]))}</td>`).join('')}</tr>`;}).join('');
    this.el('actions').innerHTML = (ac.actions || []).map(a=>`<tr><td>${escapeHTML(a.node?.slice(-1))}</td><td>${number(a.mpc_proposal_pct,1)}</td><td>${number(a.safety_pct,1)}</td><td>${number(a.final_pct,1)}</td></tr>`).join('') || '<tr><td colspan="4">No MPC decision yet.</td></tr>';
    if(finite(day) && state.state_identity) {
      if(this.history.length && day < this.history.at(-1).day) this.history = [];
      const point = {day, values:COLORS.map((c,i)=>rs['reservoir_'+(i+1)]?.storage)};
      if(this.history.at(-1)?.day === day) this.history[this.history.length-1] = point; else this.history.push(point);
      this.history = this.history.slice(-120); this.renderTrend();
    }
    this.syncControls();
  }
  renderTrend() {
    const history=this.history, min=history[0]?.day ?? 0, max=history.at(-1)?.day ?? min, x=day=>24+(day-min)/Math.max(1,max-min)*244, y=value=>87-value*72;
    let svg=`<path d="M24 15V87H268" stroke="var(--hairline)" fill="none"/><path d="M24 51H268" stroke="var(--hairline-soft)"/><text x="0" y="18" fill="var(--ink-soft)" font-size="9">100</text><text x="8" y="90" fill="var(--ink-soft)" font-size="9">0</text><text x="24" y="107" fill="var(--ink-soft)" font-size="9">Day ${min}</text><text x="268" y="107" text-anchor="end" fill="var(--ink-soft)" font-size="9">Day ${max}</text>`;
    COLORS.forEach((color,i)=>{let segment=false,path='';for(const point of history){if(finite(point.values[i])){path+=(segment?'L':'M')+x(point.day).toFixed(1)+','+y(point.values[i]).toFixed(1)+' ';segment=true;}else segment=false;}if(path)svg+=`<path d="${path}" fill="none" stroke="${color}" stroke-width="2"/>`;const last=history.at(-1);if(last&&finite(last.values[i]))svg+=`<circle cx="${x(last.day)}" cy="${y(last.values[i])}" r="2.5" fill="${color}"/>`;});
    if(history.length<2) svg+='<text x="140" y="38" text-anchor="middle" fill="var(--ink-soft)" font-size="10">STEP to build an actual history</text>';
    this.el('trend').innerHTML=svg;
  }
  renderGraph() {
    if(!this.graph) return;
    const nodes=this.graph.nodes, points=nodes.map((n,i)=>{const a=-Math.PI/2+2*Math.PI*i/nodes.length;return {x:250+112*Math.cos(a),y:153+112*Math.sin(a),a};});
    const edges=this.graph.edges.map(edge=>{const a=points[nodes.indexOf(edge[0])],b=points[nodes.indexOf(edge[1])];return a&&b?`<line x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}"/>`:'';}).join('');
    this.el('graph').innerHTML=edges+points.map((p,i)=>`<circle cx="${p.x}" cy="${p.y}" r="4"/><text x="${p.x+Math.cos(p.a)*13}" y="${p.y+Math.sin(p.a)*13+3}" text-anchor="${Math.cos(p.a)>.2?'start':Math.cos(p.a)<-.2?'end':'middle'}">${escapeHTML(nodes[i])}</text>`).join('');
    this.text('graph-note', `${nodes.length} dataset reservoirs · ${this.graph.edges.length} undirected training correlations. This graph is statistical; the water-routing cascade has four nodes.`);
  }
}

