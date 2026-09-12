import sys

with open('src/dashboard/twin_component/reservoir_twin.html', 'r', encoding='utf-8') as f:
    content = f.read()

# Update importmap
importmap_start = content.find('<script type="importmap">')
importmap_end = content.find('</script>', importmap_start)
if importmap_start != -1 and importmap_end != -1:
    new_importmap = '''<script type="importmap">
{
  "imports": {
    "three": "/vendor/three/three.module.js",
    "three/addons/": "/vendor/three/addons/"
  }
}
'''
    content = content[:importmap_start] + new_importmap + content[importmap_end+9:]

# Insert controls into bar.innerHTML
bar_idx = content.find('bar.innerHTML = `')
if bar_idx != -1:
    ds_btn_idx = content.find('DOWNSTREAM</button>', bar_idx)
    insert_idx = content.find('</div>', ds_btn_idx)
    if insert_idx != -1:
        controls_html = '''
      <div class="ctl-group harness"><span class="ctl-cap">SIMULATION CONTROLS</span>
        <button class="cbtn" id="btn-play">PLAY</button>
        <button class="cbtn" id="btn-pause">PAUSE</button>
        <button class="cbtn" id="btn-step">STEP</button>
        <button class="cbtn" id="btn-reset">RESET</button>
        <label class="sld"><span>SPEED</span><input type="range" id="ctl-speed" min="25" max="200" value="100"><b id="v-speed">100</b></label>
        <label class="sld"><span>STORM</span><input type="range" data-ctl="storm" min="0" max="100" value="70"><b data-v="storm">70</b></label>
        <label class="sld"><span>G1</span><input type="range" data-ctl="g1" min="0" max="100" value="40"><b data-v="g1">40</b></label>
        <label class="sld"><span>G2</span><input type="range" data-ctl="g2" min="0" max="100" value="35"><b data-v="g2">35</b></label>
        <label class="sld"><span>G3</span><input type="range" data-ctl="g3" min="0" max="100" value="50"><b data-v="g3">50</b></label>
        <label class="chk"><input type="checkbox" data-ctl="auto" checked>AUTO</label>
      </div>'''
        content = content[:insert_idx + 6] + controls_html + content[insert_idx + 6:]

# Add slider selectors to this.el 
el_idx = content.find('this.el = {')
if el_idx != -1:
    el_end = content.find('};', el_idx)
    if el_end != -1:
        sliders_js = ''',
      sliders: {
        storm: q(bar, '[data-ctl="storm"]'),
        g: [q(bar, '[data-ctl="g1"]'), q(bar, '[data-ctl="g2"]'), q(bar, '[data-ctl="g3"]')],
        auto: q(bar, '[data-ctl="auto"]')
      },
      sliderVals: {
        storm: q(bar, '[data-v="storm"]'),
        g: [q(bar, '[data-v="g1"]'), q(bar, '[data-v="g2"]'), q(bar, '[data-v="g3"]')]
      }'''
        content = content[:el_end] + sliders_js + content[el_end:]


# Update bootstrap section
bootstrap_idx = content.find('/* ================= bootstrap')
if bootstrap_idx != -1:
    before = content[:bootstrap_idx]
    after = content[content.find('</script>', bootstrap_idx):]
    
    new_script = '''/* ================= bootstrap — one init per page load ================= */
import { TwinAPI } from './api.js';

const root = document.getElementById('twin-root');
let app = null;
try {
  app = new ReservoirTwinRenderer(root, { testControls: true });
} catch (e) {
  /* boot overlay already shows the failure reason; nothing further to do */
}
if (app) {
  const api = new TwinAPI((state) => {
    app.updateState(state);
    
    // Sync UI with state
    if (app.el && app.el.sliders) {
      if (state.controller_mode === 'AI') {
        app.el.sliders.auto.checked = true;
      } else {
        app.el.sliders.auto.checked = false;
      }
      
      const s = state.storm_intensity * 100;
      if (app.el.sliders.storm) app.el.sliders.storm.value = s;
      if (app.el.sliderVals.storm) app.el.sliderVals.storm.textContent = s.toFixed(0);
      
      // Update sliders only if in AI mode (in MANUAL user controls it)
      if (state.controller_mode === 'AI') {
         const g1 = state.reservoirs.reservoir_1.gate * 100;
         const g2 = state.reservoirs.reservoir_2.gate * 100;
         const g3 = state.reservoirs.reservoir_3.gate * 100;
         
         if (app.el.sliders.g[0]) {
             app.el.sliders.g[0].value = g1;
             app.el.sliderVals.g[0].textContent = g1.toFixed(0);
         }
         if (app.el.sliders.g[1]) {
             app.el.sliders.g[1].value = g2;
             app.el.sliderVals.g[1].textContent = g2.toFixed(0);
         }
         if (app.el.sliders.g[2]) {
             app.el.sliders.g[2].value = g3;
             app.el.sliderVals.g[2].textContent = g3.toFixed(0);
         }
      }
    }
  });

  const el = app.el;
  if (el) {
    if (el.camBtns) {
        el.camBtns.forEach(btn => btn.addEventListener('click', () => {
          el.camBtns.forEach(b => b.classList.remove('on'));
          btn.classList.add('on');
          app.setCameraPreset(btn.dataset.cam);
        }));
    }

    const wire = (input, valOut, fn) => {
        if (!input) return;
        input.addEventListener('input', () => {
          if (valOut) valOut.textContent = input.value;
          fn(input.value);
        });
    };

    if (el.sliders && el.sliders.storm) {
        wire(el.sliders.storm, el.sliderVals.storm, v => api.setStorm(v));
        wire(el.sliders.g[0], el.sliderVals.g[0], v => { api.setGate('reservoir_1', v); if(el.sliders.auto.checked) api.setMode('MANUAL'); });
        wire(el.sliders.g[1], el.sliderVals.g[1], v => { api.setGate('reservoir_2', v); if(el.sliders.auto.checked) api.setMode('MANUAL'); });
        wire(el.sliders.g[2], el.sliderVals.g[2], v => { api.setGate('reservoir_3', v); if(el.sliders.auto.checked) api.setMode('MANUAL'); });

        if (el.sliders.auto) {
            el.sliders.auto.addEventListener('change', () => {
              api.setMode(el.sliders.auto.checked ? 'AI' : 'MANUAL');
            });
        }
    }
    
    // Playback controls
    const btnPlay = document.getElementById('btn-play');
    if (btnPlay) btnPlay.addEventListener('click', () => api.play());
    const btnPause = document.getElementById('btn-pause');
    if (btnPause) btnPause.addEventListener('click', () => api.pause());
    const btnStep = document.getElementById('btn-step');
    if (btnStep) btnStep.addEventListener('click', () => api.step());
    const btnReset = document.getElementById('btn-reset');
    if (btnReset) btnReset.addEventListener('click', () => api.reset());
    
    const spdSlider = document.getElementById('ctl-speed');
    const spdVal = document.getElementById('v-speed');
    if (spdSlider) {
      spdSlider.addEventListener('input', () => {
        spdVal.textContent = spdSlider.value;
        api.setSpeed(spdSlider.value / 100);
      });
    }
  }

  window.__twin = app;
  window.ReservoirTwinRenderer = ReservoirTwinRenderer;
  window.api = api;
}
'''
    content = before + new_script + after

with open('src/dashboard/web/index.html', 'w', encoding='utf-8') as out:
    out.write(content)
print('index.html fully regenerated')
