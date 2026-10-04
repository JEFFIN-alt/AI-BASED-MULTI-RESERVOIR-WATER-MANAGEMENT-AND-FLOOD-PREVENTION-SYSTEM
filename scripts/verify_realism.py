"""Capture the actual local WebGL application; never inject hydraulic state."""
import argparse
import json
import time
import statistics
import subprocess
from pathlib import Path
from playwright.sync_api import sync_playwright

parser = argparse.ArgumentParser()
parser.add_argument('--stage', default='baseline')
parser.add_argument('--quick', action='store_true')
parser.add_argument('--hardware', action='store_true')
parser.add_argument('--normal', action='store_true')
parser.add_argument('--original', action='store_true')
parser.add_argument('--exercise', action='store_true')
parser.add_argument('--flows', action='store_true', help='Capture positive live flow without issuing simulation commands')
args = parser.parse_args()
out = Path('results/realism') / args.stage
out.mkdir(parents=True, exist_ok=True)
report = {'errors': [], 'cameras': {}}
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=['--use-angle=d3d11'] if args.hardware else [])
    page = browser.new_page(viewport={'width': 1600, 'height': 1000}, device_scale_factor=1)
    page.on('pageerror', lambda e: report['errors'].append(str(e)))
    page.on('console', lambda m: report['errors'].append(m.text) if m.type == 'error' else None)
    page.add_init_script("localStorage.setItem('aqua-theme','light')")
    if args.original:
        original = subprocess.check_output(['git','show','HEAD:src/dashboard/web/index.html']).decode('utf-8')
        page.route('http://127.0.0.1:8000/',lambda route: route.fulfill(body=original,content_type='text/html'))
    page.goto('http://127.0.0.1:8000/', wait_until='networkidle')
    page.wait_for_function('window.__twinDiagnostics?.connection === "CONNECTED"')
    if args.normal or args.exercise:
        report['state_before_scenario']=page.request.get('http://127.0.0.1:8000/api/state').json()
        page.locator('[data-demo="normal"]').click()
        page.wait_for_timeout(2500)
        # RESET preserves operator gate requests. Explicit public gate commands
        # are therefore needed after the zero-release test, followed by STEP.
        for i,value in enumerate([40,35,50,10],1):
            response=page.request.post(f'http://127.0.0.1:8000/api/gate/reservoir_{i}',data={'value':value})
            assert response.ok
        response=page.request.post('http://127.0.0.1:8000/api/simulation/step',data={})
        assert response.ok
        page.wait_for_timeout(3500)
    page.wait_for_timeout(2500)
    report['state'] = page.request.get('http://127.0.0.1:8000/api/state').json()
    if args.exercise:
        assert all(r['controlled_release']>0 for r in report['state']['reservoirs'].values()),'Flow evidence requires actual positive releases'
        assert report['state']['downstream']['flow_m3_s']>0
    report['gpu'] = page.evaluate('window.aquaFlowGpuDiagnostics()')
    for camera in (['overview'] if args.quick else ['overview','reservoir1','reservoir2','reservoir3','reservoir4','downstream']):
        page.locator(f'[data-cam="{camera}"]').click()
        page.wait_for_timeout(2200)
        page.screenshot(path=str(out / f'{camera}.png'))
        report['cameras'][camera] = page.evaluate('window.__twinDiagnostics')
    page.locator('[data-cam="overview"]').click()
    page.locator('#btn-focus').click()
    page.wait_for_timeout(2500)
    def sequence(name):
        frames=[]
        start=time.monotonic()
        for sec in [0,1,2,4]:
            page.wait_for_timeout(max(0,(start+sec-time.monotonic())*1000))
            stamp=time.monotonic()-start
            page.screenshot(path=str(out / f'{name}-t{sec}.png'))
            frames.append({'requested_s':sec,'capture_started_s':stamp,'diagnostics':page.evaluate('window.__twinDiagnostics')})
        return frames
    report['overview_sequence']=sequence('focus')
    report['focus'] = page.evaluate('window.__twinDiagnostics')
    samples=[]
    for _ in range(5):
        page.wait_for_timeout(1000)
        samples.append(page.evaluate('window.__twinDiagnostics.fps'))
    report['focus_fps_samples']=samples
    report['focus_median_fps']=statistics.median(samples)
    if args.flows:
        assert all(r['controlled_release']>0 for r in report['state']['reservoirs'].values()),'Current backend must supply positive releases; no state is injected'
        report['links']={}
        for i in range(1,5):
            page.locator(f'[data-cam="reservoir{i}"]').click();page.wait_for_timeout(2400)
            report['links'][f'R{i}']=sequence(f'link{i}')
    if args.exercise:
        report['links']={}
        for i in range(1,5):
            page.locator(f'[data-cam="reservoir{i}"]').click()
            page.wait_for_timeout(2400)
            report['links'][f'R{i}']=sequence(f'link{i}')
        page.locator('[data-cam="overview"]').click();page.wait_for_timeout(2300)
        before=page.evaluate('window.__twinDiagnostics')
        canvas=page.locator('canvas').first.bounding_box()
        x=canvas['x']+canvas['width']*.45;y=canvas['y']+canvas['height']*.6
        page.mouse.move(x,y);page.mouse.down();page.mouse.move(x+85,y+30,steps=12);page.mouse.up()
        page.wait_for_timeout(600)
        orbit=page.evaluate('window.__twinDiagnostics')
        page.mouse.move(x,y);page.mouse.wheel(0,-180);page.wait_for_timeout(700)
        zoom=page.evaluate('window.__twinDiagnostics')
        page.mouse.down(button='right');page.mouse.move(x+60,y+25,steps=10);page.mouse.up(button='right');page.wait_for_timeout(700)
        pan=page.evaluate('window.__twinDiagnostics')
        report['interaction']={'orbit':before['camera']!=orbit['camera'],'zoom':orbit['camera']!=zoom['camera'],'pan':pan['target']!=zoom['target']}
        assert all(report['interaction'].values())
        page.locator('[data-cam="overview"]').click();page.wait_for_timeout(2300)
        page.locator('.flabel').nth(3).click();page.wait_for_timeout(2200)
        report['selection']=page.locator('#reservoir-drawer #hud-r4').is_visible()
        page.screenshot(path=str(out/'selected-idukki.png'))
        assert report['selection']
        page.locator('[data-close]').click();page.wait_for_timeout(2200)
        page.locator('[data-world="site"]').click()
        report['site']={}
        for mode in ['map','satellite','street']:
            page.locator(f'.site-context [data-mode="{mode}"]').click()
            report['site'][mode]=page.locator('.site-content strong').inner_text()
            assert report['site'][mode]=='SITE IMAGERY NOT CONFIGURED'
        page.screenshot(path=str(out/'site-unconfigured.png'))
        page.locator('[data-world="twin"]').click()
        assert not page.locator('.site-context').is_visible()
        # Exercise the real command path. Never override WebSocket/REST values.
        def post(path,body=None):
            response=page.request.post('http://127.0.0.1:8000/api'+path,data=body or {})
            assert response.ok, response.text()
        post('/simulation/pause')
        post('/controller/mode',{'mode':'MANUAL'})
        for i in range(1,5):post(f'/gate/reservoir_{i}',{'value':0})
        post('/simulation/step');page.wait_for_timeout(3500)
        report['zero_state']=page.request.get('http://127.0.0.1:8000/api/state').json()
        report['zero_scene']=page.evaluate('window.__twinDiagnostics')
        for i,r in enumerate(report['zero_scene']['reservoirs'],1):
            actual=report['zero_state']['reservoirs'][f'reservoir_{i}']
            assert r['jetVisible']==(actual['controlled_release']>0)
            assert r['spillVisible']==(actual['spill_mcm']>0)
        page.screenshot(path=str(out/'zero-controlled-release.png'))
        post('/storm',{'value':1})
        for _ in range(4):post('/simulation/step')
        page.wait_for_timeout(3500)
        report['spill_state']=page.request.get('http://127.0.0.1:8000/api/state').json()
        report['spill_scene']=page.evaluate('window.__twinDiagnostics')
        for i,r in enumerate(report['spill_scene']['reservoirs'],1):
            actual=report['spill_state']['reservoirs'][f'reservoir_{i}']
            assert r['spillVisible']==(actual['spill_mcm']>0)
        page.locator('[data-cam="reservoir1"]').click();page.wait_for_timeout(2300)
        page.screenshot(path=str(out/'separate-spill.png'))
        # Leave the local demo in its normal, paused scenario with visible releases.
        page.locator('#btn-focus').click()
        page.locator('[data-demo="normal"]').click();page.wait_for_timeout(3500)
        for i,value in enumerate([40,35,50,10],1):post(f'/gate/reservoir_{i}',{'value':value})
        post('/simulation/step');page.wait_for_timeout(3500)
        page.locator('[data-cam="overview"]').click();page.wait_for_timeout(2300)
        report['restored_state']=page.request.get('http://127.0.0.1:8000/api/state').json()
        assert report['restored_state']['simulation']['running'] is False
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'gpu':report['gpu'],'fps':report['focus_median_fps'],'errors':report['errors']}))
    browser.close()
