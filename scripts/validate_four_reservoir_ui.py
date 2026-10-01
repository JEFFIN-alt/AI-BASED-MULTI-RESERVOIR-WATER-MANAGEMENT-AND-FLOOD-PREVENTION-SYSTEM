"""Real browser + single running backend validation. Start uvicorn before running.
Commands change only the local demo through its public API. No injected state.
"""
import json,time,statistics
from pathlib import Path
import requests
from playwright.sync_api import sync_playwright
OUT=Path('results/twin_four_reservoir_ui');OUT.mkdir(parents=True,exist_ok=True)
BASE='http://127.0.0.1:8000'
evidence={'checks':{},'resolutions':{},'errors':[]}
def check(name,condition,detail=None):
 evidence['checks'][name]={'pass':bool(condition),'detail':detail}
 print(name, bool(condition),flush=True)
 if not condition: raise AssertionError(name+': '+str(detail))
def post(path,body=None):
 r=requests.post(BASE+'/api'+path,json=body or {},timeout=120);r.raise_for_status();return r.json()
def state():return requests.get(BASE+'/api/state',timeout=30).json()
frames=[];heartbeats=[]
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True)
 page=browser.new_page(viewport={'width':1440,'height':900})
 page.on('pageerror',lambda e:evidence['errors'].append(str(e)))
 def record_frame(payload):
  frame=json.loads(payload)
  if frame.get('type')=='heartbeat': heartbeats.append(frame)
  else: frames.append(frame)
 page.on('websocket',lambda ws:ws.on('framereceived',record_frame))
 try:
  post('/simulation/pause');post('/controller/mode',{'mode':'MANUAL','source':'SIMULATION'});post('/simulation/reset');post('/storm',{'value':.2})
  page.goto(BASE);page.wait_for_function('window.__twinDiagnostics?.connection === "CONNECTED"');page.wait_for_timeout(1500)
  boot_id=page.evaluate('window.__twinDiagnostics.stateId');page.locator('[data-demo="normal"]').click()
  page.wait_for_function('(id)=>window.__twinDiagnostics.stateId!==id',arg=boot_id,timeout=10000);page.wait_for_timeout(700)
  initial=state();evidence['initial']=initial
  page.screenshot(path=str(OUT/'demo-normal.png'))
  check('REST and WebSocket visibly verified',page.locator('[data-ref="rest-health"]').inner_text()=='LIVE' and page.locator('[data-ref="ws-health"]').inner_text()=='LIVE' and page.locator('[data-ref="state-health"]').inner_text()=='LIVE')
  check('authoritative source disclosed','PYTHON AUTHORITATIVE SIMULATION' in page.locator('.transport-status').inner_text())
  check('REST four reservoirs',list(initial['reservoirs'])==['reservoir_'+str(i) for i in range(1,5)])
  check('normal scenario uses backend baseline inflows',initial['storm']['intensity']==0 and initial['state_identity']['network_timestep']==1 and initial['reservoirs']['reservoir_1']['inflow']>0)
  check('WebSocket matches REST',frames[-1]['reservoirs']==initial['reservoirs'])
  d=page.evaluate('window.__twinDiagnostics')
  check('four complete scene reservoirs',len(d['reservoirs'])==4 and all('DAM_0'+str(i) in d['objects'] and 'RESERVOIR_0'+str(i) in d['objects'] and 'SPILLWAY_JET_0'+str(i) in d['objects'] for i in range(1,5)),d)
  # Operator changes Idukki gate; backend decides physical result on STEP.
  gate_inputs=[40,35,50,10]
  for i,value in enumerate(gate_inputs,1):
   slider=page.locator(f'[data-ctl="g{i}"]');slider.fill(str(value));slider.dispatch_event('input')
  page.wait_for_timeout(500);requested=state()
  check('operator gate requests acknowledged before application',all(requested['reservoirs'][f'reservoir_{i}']['requested_gate_pct']==gate_inputs[i-1] and requested['reservoirs'][f'reservoir_{i}']['gate']==initial['reservoirs'][f'reservoir_{i}']['gate'] for i in range(1,5)),gate_inputs)
  page.locator('#btn-step').click();page.wait_for_function('(id)=>window.__twinDiagnostics.stateId!==id',arg=d['stateId']);page.wait_for_timeout(3500)
  after=state();d=page.evaluate('window.__twinDiagnostics');evidence['gate_step']=after
  check('all gate REST-WebSocket-renderer values synchronize',all(frames[-1]['reservoirs'][f'reservoir_{i}']['gate']==after['reservoirs'][f'reservoir_{i}']['gate']==d['reservoirs'][i-1]['gate'] and abs(d['reservoirs'][i-1]['gateY']-d['reservoirs'][i-1]['gateTargetY'])<.1 for i in range(1,5)),d['reservoirs'])
  # Real upstream storm command and real propagation steps.
  page.locator('[data-demo="rain"]').click();page.wait_for_timeout(500);check('storm command',state()['storm']['intensity']==.35)
  inflow_before=after['reservoirs']['reservoir_1']['inflow'];steps=[]
  for _ in range(4):
   post('/simulation/step');page.wait_for_timeout(600);steps.append(state())
  evidence['storm_steps']=steps
  page.wait_for_timeout(1000);page.screenshot(path=str(OUT/'demo-rising-inflow.png'))
  check('inflow increased in authoritative backend',steps[-1]['reservoirs']['reservoir_1']['inflow']>inflow_before)
  routed=[steps[-1]['reservoirs'][f'reservoir_{i}']['inflow'] for i in range(1,5)]
  base=[initial['reservoirs'][f'reservoir_{i}']['inflow'] for i in range(1,5)]
  check('storm response reaches inflow telemetry A through D',all(routed[i]>base[i] for i in range(4)) and routed[3]>0,{'baseline':base,'rising':routed})
  page.wait_for_timeout(2000);d=page.evaluate('window.__twinDiagnostics');last=steps[-1]
  check('four state mappings follow cascade',all(abs(d['reservoirs'][i]['level']-last['reservoirs']['reservoir_'+str(i+1)]['water_level'])<1e-8 for i in range(4)))
  check('terminal flow belongs to D',last['downstream']['flow_m3_s']==last['reservoirs']['reservoir_4']['release'] and d['channels'][3]['flow']==min(1,last['downstream']['flow_m3_s']/70),{'C':last['reservoirs']['reservoir_3']['release'],'D':last['reservoirs']['reservoir_4']['release'],'downstream':last['downstream']})
  page.locator('[data-demo="auto"]').click();page.wait_for_timeout(500);post('/simulation/step');page.wait_for_timeout(800)
  blocked=state();evidence['auto_simulation']=blocked
  check('simulation AUTO honestly blocked',blocked['auto_control']['state']=='AUTO_BLOCKED' and 'BLOCKED' in page.locator('[data-ref="auto-state"]').inner_text())
  page.locator('[data-source="VALIDATED_REPLAY"]').click();page.wait_for_timeout(500);post('/simulation/step');page.wait_for_timeout(1500)
  auto=state();evidence['auto_replay']=auto
  page.screenshot(path=str(OUT/'demo-auto-control.png'))
  check('historical replay MPC actual status',auto['control']['controller_status']=='ACTIVE' and page.locator('[data-ref="auto-decision"]').inner_text()==auto['auto_control']['controller_status'])
  check('safety status matches backend',page.locator('[data-ref="auto-safety"]').inner_text()==auto['auto_control']['safety_layer_status'])
  check('guard status matches backend',page.locator('[data-ref="auto-downstream"]').inner_text()==auto['auto_control']['downstream_status'])
  check('mass balance matches backend',page.locator('[data-ref="auto-balance"]').inner_text()==auto['mass_balance']['status'])
  before_gates=[blocked['reservoirs'][f'reservoir_{i}']['gate'] for i in range(1,5)]
  after_gates=[auto['reservoirs'][f'reservoir_{i}']['gate'] for i in range(1,5)]
  finals=[a['final_pct'] for a in auto['auto_control']['actions']]
  check('MPC final actions applied to physical gates',all(abs(after_gates[i]*100-finals[i])<0.11 for i in range(4)) and any(abs(a-b)>1e-6 for a,b in zip(before_gates,after_gates)),{'before':before_gates,'after':after_gates,'final_pct':finals})
  check('gate movement changes D release and downstream',auto['reservoirs']['reservoir_4']['release']!=blocked['reservoirs']['reservoir_4']['release'] and auto['downstream']['flow_m3_s']==auto['reservoirs']['reservoir_4']['release'])
  check('replay provenance disclosed','HISTORICAL FORECAST REPLAY' in page.locator('#demo-badge').inner_text())
  check('GNN remains advisory',auto['gnn_advisory']['advisory_only'] is True and auto['gnn_advisory']['affects_control'] is False)
  check('hardware not connected',all(v=='NOT_CONNECTED' for v in auto['hardware_status'].values()))
  page.locator('.secondary-panels > summary').click()
  check('GNN UI flags honest','Advisory only: YES' in page.locator('[data-ref="gnn-boundary"]').inner_text() and 'Affects control: NO' in page.locator('[data-ref="gnn-boundary"]').inner_text())
  check('hardware UI honest',page.locator('[data-ref="hw1"]').inner_text()=='NOT CONNECTED')
  page.locator('.secondary-panels > summary').click()
  # Camera moves; Idukki selection opens the same card in the drawer.
  before=page.evaluate('window.__twinDiagnostics.camera');page.locator('[data-cam="reservoir4"]').click();page.wait_for_timeout(2500)
  aftercam=page.evaluate('window.__twinDiagnostics.camera');check('Idukki camera actually moves',sum(abs(a-b) for a,b in zip(before,aftercam))>100,aftercam)
  page.locator('.flabel').nth(3).click();page.wait_for_timeout(700)
  check('Idukki drawer correct',page.locator('#reservoir-drawer #hud-r4').is_visible())
  check('Idukki forecast values displayed',all(page.locator('#hud-r4 [data-k="forecast-'+h+'"]').inner_text()==f"{auto['reservoirs']['reservoir_4']['forecast_'+h]:.1f}" for h in ['1d','3d','7d']))
  page.screenshot(path=str(OUT/'idukki.png'))
  page.locator('[data-close]').click();page.wait_for_timeout(2500)
  page.locator('#btn-focus').click();check('focus mode',page.locator('#twin-root').evaluate('(e)=>e.classList.contains("scene-focus")'));page.locator('#btn-focus').click()
  # Responsive viewport + quantitative browser performance, no injected state.
  for width,height in [(1920,1080),(1440,900),(1366,768),(1280,720)]:
   page.set_viewport_size({'width':width,'height':height});page.wait_for_timeout(2000)
   samples=[]
   for _ in range(8):
    page.wait_for_timeout(1000);samples.append(page.evaluate('({fps:window.__twinDiagnostics.fps,frameMs:window.__twinDiagnostics.frameMs,memory:window.__twinDiagnostics.memory,heap:performance.memory?.usedJSHeapSize})'))
   layout=page.evaluate('''() => {const rect=e=>{const r=e.getBoundingClientRect();return {x:r.x,y:r.y,w:r.width,h:r.height,right:r.right,bottom:r.bottom}};return {overflow:document.documentElement.scrollWidth>innerWidth,labels:[...document.querySelectorAll('.flabel')].map(rect),canvas:rect(document.querySelector('canvas')),controls:rect(document.querySelector('#hud-controls')),cards:rect(document.querySelector('#hud-bottom'))}}''')
   check(f'{width}x{height} scene clear of controls',layout['canvas']['bottom']<=layout['controls']['y']+2,layout)
   check(f'{width}x{height} layout bounds',not layout['overflow'] and layout['controls']['bottom']<=height and all(0<=r['x'] and r['right']<=width and 0<=r['y'] and r['bottom']<=height for r in layout['labels']),layout)
   evidence['resolutions'][f'{width}x{height}']={'layout':layout,'samples':samples,'median_fps':statistics.median(x['fps'] for x in samples)}
   page.screenshot(path=str(OUT/f'{width}x{height}.png'))
  # Live update stream stability.
  page.locator('#btn-play').click();page.wait_for_timeout(6500);page.locator('#btn-pause').click();page.wait_for_timeout(500)
  evidence['stream']={'frames':len(frames),'heartbeats':len(heartbeats),'identities':[f['state_identity']['state_id'] for f in frames[-12:]]}
  check('live WebSocket progress',len(set(evidence['stream']['identities']))>=3)
  check('paused WebSocket heartbeat observed',len(heartbeats)>=1)
  page.locator('#btn-reset').click();page.wait_for_timeout(1500);reset=state();evidence['reset']=reset
  check('reset returns backend to manual baseline',reset['state_identity']['network_timestep']==0 and all(r['storage']==.5 for r in reset['reservoirs'].values()) and reset['storm']['intensity']==0 and reset['controller_mode']=='MANUAL')
  page.screenshot(path=str(OUT/'demo-recovery.png'))
  page.evaluate('window.__twin_api.ws.close()')
  page.wait_for_function("document.querySelector('[data-ref=ws-health]').textContent==='DISCONNECTED' && document.querySelector('[data-ref=rest-health]').textContent==='LIVE'",timeout=5000)
  check('lost WebSocket is marked stale while REST remains live','STALE DATA' in page.locator('[data-ref="connection"]').inner_text())
  page.wait_for_function('window.__twinDiagnostics.connection === "CONNECTED"',timeout=10000)
  # RESET was already verified above against the same live backend and UI.
  # Avoid issuing another storm/reset sequence while the slow SwiftShader
  # renderer is collecting screenshots.
  page.set_viewport_size({'width':1440,'height':900});page.wait_for_timeout(2500);page.screenshot(path=str(OUT/'normal.png'))
  page.locator('#theme-toggle').click();page.wait_for_timeout(800);page.screenshot(path=str(OUT/'dark.png'));page.locator('#theme-toggle').click()
  # Break actual browser transport. No fabricated state is sent to the renderer.
  page.context.set_offline(True);page.evaluate('window.__twin_api.ws.close()')
  page.wait_for_function("document.querySelector('[data-ref=connection]').textContent.includes('BACKEND DISCONNECTED')",timeout=10000)
  check('disconnect honestly shown','BACKEND DISCONNECTED' in page.locator('[data-ref="connection"]').inner_text() and page.locator('[data-ref="auto-state"]').inner_text()=='LAST RECEIVED STATE')
  page.screenshot(path=str(OUT/'disconnected.png'))
  page.context.set_offline(False);page.wait_for_function('window.__twinDiagnostics.connection === "CONNECTED"',timeout=15000)
  # Health failure while socket stays open exercises stale state separately.
  page.route('**/api/state',lambda route:route.abort());page.evaluate('window.__twin_api.checkFreshness()');page.wait_for_timeout(700)
  check('stale data honestly shown','STALE DATA' in page.locator('[data-ref="connection"]').inner_text())
  page.unroute('**/api/state');page.evaluate('window.__twin_api.checkFreshness()');page.wait_for_timeout(700)
  check('no browser errors',not evidence['errors'],evidence['errors'])
 except Exception as e:
  evidence['failure']=str(e);page.screenshot(path=str(OUT/'failure.png'));raise
 finally:
  (OUT/'browser_evidence.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
  if all(k in evidence for k in ('initial','storm_steps','auto_simulation','auto_replay','reset')):
   def demo_snapshot(label,s):
    return {'stage':label,'state_identity':s.get('state_identity'),'storm':s.get('storm'),
     'reservoirs':{k:{field:v.get(field) for field in ('storage','inflow','release','gate','spill_mcm','risk')}
                   for k,v in s.get('reservoirs',{}).items()},
     'forecast_source_selection':s.get('forecast_source_selection'),
     'forecast_provenance':s.get('forecast_provenance'),
     'auto_control':s.get('auto_control'),'downstream':s.get('downstream'),
     'mass_balance':s.get('mass_balance')}
   stages=[demo_snapshot('NORMAL',evidence['initial']),
    demo_snapshot('RISING INFLOW',evidence['storm_steps'][-1]),
    demo_snapshot('SIMULATION FORECAST BLOCKED',evidence['auto_simulation']),
    demo_snapshot('VALIDATED REPLAY AUTO CONTROL',evidence['auto_replay']),
    demo_snapshot('RECOVERY RESET',evidence['reset'])]
   (OUT/'recorded_demo.json').write_text(json.dumps({'source':'captured public REST/WebSocket state; no injected frontend state','stages':stages},indent=2),encoding='utf-8')
  browser.close()
