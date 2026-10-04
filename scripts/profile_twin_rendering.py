"""Real Chromium rendering benchmark. Overrides presentation only, never state."""
import argparse,json,statistics
from pathlib import Path
from playwright.sync_api import sync_playwright

p=argparse.ArgumentParser()
p.add_argument('--stage',default='before')
p.add_argument('--ablate',action='store_true')
p.add_argument('--matrix',action='store_true')
p.add_argument('--snapshot')
p.add_argument('--headless',action='store_true')
args=p.parse_args()
out=Path('results/performance')/args.stage;out.mkdir(parents=True,exist_ok=True)
source=Path('src/dashboard/web/index.html').read_text(encoding='utf-8')
module=Path('src/dashboard/web/realism.js').read_text(encoding='utf-8')
if args.snapshot:
 source=(Path(args.snapshot)/'source.html').read_text(encoding='utf-8')
 module=(Path(args.snapshot)/'realism.js').read_text(encoding='utf-8')
(out/'realism.js').write_text(module,encoding='utf-8')
(out/'source.html').write_text(source,encoding='utf-8')
probe='''
window.__renderProbe=()=>{
 const r=app.renderer,g=r.getContext(), ext=g.getExtension('WEBGL_debug_renderer_info');
 const shadows=[];app.scene.traverse(o=>{if(o.isLight&&o.castShadow)shadows.push({name:o.type,size:o.shadow.mapSize.toArray(),autoUpdate:r.shadowMap.autoUpdate});});
 return {fps:app._fps,frameMs:1000/app._fps,calls:r.info.render.calls,triangles:r.info.render.triangles,
 quality:app.visualQuality || 'PREVIOUS',
 memory:{...r.info.memory},shadowMaps:shadows,canvas:[r.domElement.width,r.domElement.height],
 css:[r.domElement.clientWidth,r.domElement.clientHeight],dpr:devicePixelRatio,pixelRatio:r.getPixelRatio(),
 renderer:ext?g.getParameter(ext.UNMASKED_RENDERER_WEBGL):g.getParameter(g.RENDERER),
 vendor:ext?g.getParameter(ext.UNMASKED_VENDOR_WEBGL):g.getParameter(g.VENDOR),webgl:g.getParameter(g.VERSION),
 requested:g.getContextAttributes().powerPreference};
};
window.__renderAblate=(kind)=>{
 if(kind==='vegetation')app.scene.getObjectByName('VEGETATION').visible=false;
 if(kind==='terrain')app.scene.getObjectByName('TERRAIN_WESTERN_GHATS').visible=false;
 if(kind==='water')app.R.forEach(r=>r.waterMesh.visible=false);
 if(kind==='mist')app.mistPlanes.forEach(m=>m.mesh.visible=false);
 if(kind==='shadows'){app.renderer.shadowMap.enabled=false;app.scene.traverse(o=>{if(o.material)o.material.needsUpdate=true;});}
 if(kind==='half_pixels')app.renderer.setPixelRatio(.7);
 if(kind==='labels')app._updateConnectors=()=>{};
 if(kind==='ground_shader')app.scene.getObjectByName('TERRAIN_WESTERN_GHATS').material=new THREE.MeshStandardMaterial({vertexColors:true,roughness:.94});
};
'''
source=source.replace('  /* Bind Simulation Controls */',probe+'\n  /* Bind Simulation Controls */')
report={'errors':[],'runs':{},'headless':args.headless}
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=args.headless,args=['--enable-gpu','--ignore-gpu-blocklist','--use-angle=d3d11',
  '--disable-backgrounding-occluded-windows','--disable-renderer-backgrounding','--disable-background-timer-throttling',
  '--disable-features=CalculateNativeWinOcclusion'])
 page=browser.new_page(viewport={'width':1600,'height':1000},device_scale_factor=1)
 page.add_init_script("localStorage.setItem('aqua-theme','light')")
 page.route('http://127.0.0.1:8000/',lambda route:route.fulfill(body=source,content_type='text/html'))
 page.route('**/realism.js',lambda route:route.fulfill(body=module,content_type='text/javascript'))
 page.on('pageerror',lambda e:report['errors'].append(str(e)))
 page.on('console',lambda m:(report['errors'].append(m.text),print(m.text[:500],flush=True)) if m.type=='error' else None)
 def sample(key):
  page.wait_for_timeout(6500)
  samples=[]
  for _ in range(5):
   page.wait_for_timeout(1050);samples.append(page.evaluate('window.__renderProbe()'))
  report['runs'][key]={'median_fps':statistics.median(x['fps'] for x in samples),'samples':samples}
  page.screenshot(path=str(out/(key+'.png')))
  (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
  print(key,report['runs'][key]['median_fps'],flush=True)
 cases=['baseline','vegetation','terrain','ground_shader','water','mist','shadows','half_pixels','labels'] if args.ablate else ['baseline']
 for case in cases:
  page.goto('http://127.0.0.1:8000/',wait_until='domcontentloaded',timeout=120000);page.wait_for_function('window.__twinDiagnostics?.connection === "CONNECTED"',timeout=120000)
  if case!='baseline':page.evaluate('(kind)=>window.__renderAblate(kind)',case)
  if args.matrix:
   for quality in ['ULTRA','HIGH','BALANCED','PERFORMANCE']:
    page.locator('#visual-quality').select_option(quality)
    for camera in ['overview','reservoir1','reservoir2','reservoir3','reservoir4','downstream']:
     page.locator(f'[data-cam="{camera}"]').click();sample(quality+'-'+camera)
    page.locator('[data-cam="overview"]').click();page.locator('#btn-focus').click();sample(quality+'-focus');page.locator('#btn-focus').click()
  else:
   sample(case+'-overview')
   page.locator('#btn-focus').click();sample(case+'-focus')
 browser.close()
