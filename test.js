
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';

/* ================================================================
   MULTI-RESERVOIR DIGITAL TWIN — STANDALONE 3D RENDERER (v2)
   Public API:  updateState(state) · setCameraPreset(name) · dispose()
   ONE requestAnimationFrame loop drives everything.
   No backend, no ML, no controller logic lives in this file.
   ================================================================ */

/* ---------------- math / noise / rng ---------------- */
const clamp01 = v => v < 0 ? 0 : v > 1 ? 1 : v;
const lerp = (a, b, t) => a + (b - a) * t;
const smooth01 = t => { t = clamp01(t); return t * t * (3 - 2 * t); };
const numv = v => (typeof v === 'number' && isFinite(v)) ? v : 0;
const fracv = v => { v = numv(v); return v > 1.001 ? v / 100 : v; };
const RISKS = ['normal', 'warning', 'danger', 'critical'];

function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rng = mulberry32(20240715);

function hash2(x, y) {
  const n = Math.sin(x * 127.1 + y * 311.7) * 43758.5453;
  return n - Math.floor(n);
}
function vnoise(x, y) {
  const xi = Math.floor(x), yi = Math.floor(y);
  const xf = x - xi, yf = y - yi;
  const u = xf * xf * (3 - 2 * xf), v = yf * yf * (3 - 2 * yf);
  const a = hash2(xi, yi), b = hash2(xi + 1, yi);
  const c = hash2(xi, yi + 1), d = hash2(xi + 1, yi + 1);
  return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
}
function fbm(x, y, oct = 5) {
  let v = 0, amp = 0.5, f = 1;
  for (let i = 0; i < oct; i++) { v += amp * vnoise(x * f, y * f); f *= 2; amp *= 0.5; }
  return v;
}

/* ---------------- world layout ---------------- */
const VALLEY_Z0 = -450, VALLEY_Z1 = 450;
const valleyFloor = z => 150 - ((z - VALLEY_Z0) / (VALLEY_Z1 - VALLEY_Z0)) * 170;
const valleyCenterX = z => 70 * Math.sin(z * 0.0045) + 25 * Math.sin(z * 0.011 + 1.7);

const BASINS = [
  { id: 1, cz: -290, up: 135, dn: 112, rx: 88, depth: 44 },
  { id: 2, cz:    0, up: 135, dn: 112, rx: 92, depth: 44 },
  { id: 3, cz:  290, up: 135, dn: 112, rx: 88, depth: 44 },
];
const DAMSITES = BASINS.map(B => ({ B, zd: B.cz + B.dn, xc: valleyCenterX(B.cz + B.dn) }));

/* Bathymetry: carve is deepest AT the dam and shoals upstream (real reservoirs) */
function basinCarve(x, z) {
  let c = 0;
  for (const B of BASINS) {
    const dz = z - B.cz;
    if (dz < -B.up || dz > B.dn) continue;
    const t = (dz + B.up) / (B.up + B.dn);
    const carveZ = Math.pow(smooth01(t), 1.15) * B.depth;
    const w = B.rx * (1.0 - 0.42 * (1.0 - t));
    const dxn = Math.abs(x - valleyCenterX(z)) / w;
    if (dxn > 1) continue;
    c = Math.max(c, carveZ * smooth01(1 - dxn * dxn));
  }
  return c;
}
function channelCarve(x, z) {
  const d = Math.abs(x - valleyCenterX(z));
  const w = 16;
  if (d < w) { const t = 1 - d / w; return t * t * (3 - 2 * t) * 7; }
  return 0;
}
/* Scour / plunge pool just downstream of each dam */
function plungeCarve(x, z) {
  let c = 0;
  for (const D of DAMSITES) {
    const dz = z - (D.zd + 2);
    if (dz < 0 || dz > 30) continue;
    const lat = Math.abs(x - D.xc);
    if (lat > 26) continue;
    c = Math.max(c, smooth01(1 - dz / 30) * smooth01(1 - lat / 26) * 14);
  }
  return c;
}
/* Promontories flanking each dam — natural dam-site narrowing, contains full pool */
function abutmentRaise(x, z) {
  let r = 0;
  for (const D of DAMSITES) {
    const lat = Math.abs(x - D.xc);
    if (lat < 55 || lat > 200) continue;
    if (Math.abs(z - D.zd) > 60) continue;
    const latT = clamp01(1 - Math.abs(lat - 130) / 75);
    const zT = clamp01(1 - Math.abs(z - D.zd) / 60);
    r = Math.max(r, latT * smooth01(zT) * 28);
  }
  return r;
}
function terrainHeight(x, z) {
  const vf = valleyFloor(z);
  const dist = Math.abs(x - valleyCenterX(z));
  const m = smooth01((dist - 110) / 260);
  const mountains = (fbm(x * 0.004, z * 0.004) * 130 + fbm(x * 0.015, z * 0.015) * 32) * m;
  const undul = (fbm(x * 0.008 + 7.3, z * 0.008 + 2.1) - 0.5) * 15 * (1 - m);
  let h = vf + mountains + undul + abutmentRaise(x, z);
  h -= basinCarve(x, z);
  h -= channelCarve(x, z);
  h -= plungeCarve(x, z);
  return h;
}
function terrainSlope(x, z, h) {
  return (Math.abs(terrainHeight(x + 3, z) - h) + Math.abs(terrainHeight(x, z + 3) - h)) / 3;
}

/* ---------------- canvas textures ---------------- */
function canvasTex(w, h, draw, repeat) {
  const c = document.createElement('canvas'); c.width = w; c.height = h;
  draw(c.getContext('2d'), w, h);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  if (repeat) { t.wrapS = t.wrapT = THREE.RepeatWrapping; t.repeat.set(repeat[0], repeat[1]); }
  return t;
}
const skyTex = canvasTex(16, 512, (ctx, w, h) => {
  const g = ctx.createLinearGradient(0, 0, 0, h);
  g.addColorStop(0.0, '#57687a'); g.addColorStop(0.45, '#8e9aa6');
  g.addColorStop(0.78, '#b9c3ca'); g.addColorStop(1.0, '#c3ccd2');
  ctx.fillStyle = g; ctx.fillRect(0, 0, w, h);
});
const mistTex = canvasTex(128, 128, (ctx, w, h) => {
  ctx.clearRect(0, 0, w, h);
  for (let i = 0; i < 9; i++) {
    const cx = 20 + Math.random() * 88, cy = 20 + Math.random() * 88, r = 18 + Math.random() * 34;
    const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, r);
    g.addColorStop(0, 'rgba(255,255,255,0.20)'); g.addColorStop(1, 'rgba(255,255,255,0)');
    ctx.fillStyle = g; ctx.fillRect(0, 0, w, h);
  }
});
const rainTex = canvasTex(32, 64, (ctx, w, h) => {
  ctx.clearRect(0, 0, w, h);
  const g = ctx.createLinearGradient(0, 4, 0, h - 4);
  g.addColorStop(0, 'rgba(207,226,238,0)'); g.addColorStop(0.5, 'rgba(207,226,238,0.9)');
  g.addColorStop(1, 'rgba(207,226,238,0)');
  ctx.strokeStyle = g; ctx.lineWidth = 1.6;
  ctx.beginPath(); ctx.moveTo(16, 4); ctx.lineTo(16, h - 4); ctx.stroke();
});
const concreteTex = canvasTex(128, 128, (ctx, w, h) => {
  ctx.fillStyle = '#8f9496'; ctx.fillRect(0, 0, w, h);
  for (let i = 0; i < 2600; i++) {
    ctx.fillStyle = `rgba(${Math.random() > 0.5 ? '0,0,0' : '255,255,255'},${Math.random() * 0.09})`;
    ctx.fillRect(Math.random() * w, Math.random() * h, 1.5, 1.5);
  }
  for (let i = 0; i < 26; i++) {
    const x = Math.random() * w;
    const g = ctx.createLinearGradient(0, 0, 0, h);
    g.addColorStop(0, 'rgba(40,42,40,0.16)'); g.addColorStop(1, 'rgba(40,42,40,0)');
    ctx.fillStyle = g; ctx.fillRect(x, 0, 2 + Math.random() * 4, h);
  }
  const moss = ctx.createLinearGradient(0, h * 0.72, 0, h);
  moss.addColorStop(0, 'rgba(46,74,38,0)'); moss.addColorStop(1, 'rgba(46,74,38,0.35)');
  ctx.fillStyle = moss; ctx.fillRect(0, 0, w, h);
}, [10, 2]);

/* ---------------- ACTUAL water / flow shaders (complete GLSL) ---------------- */
const WATER_VERT = /* glsl */`
uniform float uTime;
uniform float uWaveAmp;
uniform float uLevelY;
attribute float aTerrain;
varying vec3 vWorldPos;
varying vec3 vNrm;
varying vec2 vUv2;
varying float vDepth;
varying float vWave;

float waveH(vec2 p, float t){
  float h = 0.0;
  h += sin(p.x * 0.16 + t * 1.05) * 0.42;
  h += sin(p.y * 0.12 - t * 0.85) * 0.34;
  h += sin((p.x + p.y) * 0.075 + t * 0.6) * 0.45;
  h += sin(p.x * 0.42 - t * 2.1) * 0.10;
  h += sin(p.y * 0.36 + t * 1.6) * 0.09;
  return h;
}
void main(){
  vUv2 = uv;
  vec3 pos = position;
  float h = waveH(pos.xy, uTime) * uWaveAmp;
  pos.z += h;
  vWave = h;
  float e = 2.0;
  float hX = waveH(pos.xy + vec2(e, 0.0), uTime) * uWaveAmp;
  float hY = waveH(pos.xy + vec2(0.0, e), uTime) * uWaveAmp;
  vec3 n = normalize(vec3(-(hX - h) / e, -(hY - h) / e, 1.0));
  vNrm = normalize(mat3(modelMatrix) * n);   /* WORLD-space normal (frag works in world space) */
  vec4 wp = modelMatrix * vec4(pos, 1.0);
  vWorldPos = wp.xyz;
  vDepth = uLevelY - aTerrain;               /* real terrain-relative water depth */
  gl_Position = projectionMatrix * viewMatrix * wp;
}`;

const WATER_FRAG = /* glsl */`
uniform float uTime;
uniform float uRain;
uniform vec3 uDeep;
uniform vec3 uShallow;
uniform vec3 uSky;
uniform vec3 uSunDir;
uniform vec3 uFogColor;
uniform float uFogDensity;
varying vec3 vWorldPos;
varying vec3 vNrm;
varying vec2 vUv2;
varying float vDepth;
varying float vWave;

float hashg(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float vnoiseg(vec2 p){
  vec2 i = floor(p), f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hashg(i), hashg(i + vec2(1.0, 0.0)), u.x),
             mix(hashg(i + vec2(0.0, 1.0)), hashg(i + vec2(1.0, 1.0)), u.x), u.y);
}
void main(){
  float depth = max(vDepth, 0.0);
  float shoreMask = smoothstep(-0.6, 1.6, vDepth);
  if (shoreMask < 0.02) discard;             /* water exists only where terrain is below surface */

  vec3 N = normalize(vNrm);
  vec3 V = normalize(cameraPosition - vWorldPos);
  float fres = pow(1.0 - max(dot(N, V), 0.0), 3.0);
  fres = 0.05 + 0.95 * fres;

  vec3 body = mix(uShallow, uDeep, smoothstep(1.0, 16.0, depth));
  body *= 0.94 + 0.06 * vnoiseg(vUv2 * 40.0 + uTime * 0.3);   /* turbulence patchiness */

  vec3 R = reflect(-V, N);
  float upness = smoothstep(-0.05, 0.45, R.y);
  vec3 skyRef = mix(uSky * 0.55, uSky, upness);
  float spec = pow(max(dot(R, normalize(uSunDir)), 0.0), 120.0);

  float r1 = vnoiseg(vUv2 * 220.0 + vec2(uTime * 2.6, -uTime * 1.9));
  float r2 = vnoiseg(vUv2 * 220.0 - vec2(uTime * 3.4, uTime * 1.3));
  float ripples = smoothstep(0.78, 0.96, r1 * 0.5 + r2 * 0.5) * uRain;

  float crest = smoothstep(0.30, 0.70, vWave);
  float shoreFoam = 1.0 - smoothstep(0.2, 1.8, depth);
  float foam = clamp(crest * 0.4 + shoreFoam * 0.85 + ripples * 0.35, 0.0, 1.0);

  vec3 col = body * (1.0 - fres * 0.6);
  col = mix(col, skyRef, fres * 0.6);
  col += spec * vec3(1.0, 0.99, 0.92) * 1.1;
  col = mix(col, vec3(0.94, 0.98, 1.0), foam);
  col += vec3(0.05, 0.08, 0.09) * ripples;

  float alpha = (0.88 + fres * 0.12) * shoreMask;
  float dist = length(cameraPosition - vWorldPos);
  float fogF = 1.0 - exp(-pow(dist * uFogDensity, 2.0));
  col = mix(col, uFogColor, fogF);
  gl_FragColor = vec4(col, alpha);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}`;

const FLOW_VERT = /* glsl */`
varying vec2 vUv2;
varying vec3 vWorldPos;
void main(){
  vUv2 = uv;
  vec4 wp = modelMatrix * vec4(position, 1.0);
  vWorldPos = wp.xyz;
  gl_Position = projectionMatrix * viewMatrix * wp;
}`;

const FLOW_FRAG = /* glsl */`
uniform float uTime;
uniform float uFlow;
uniform float uWhite;
uniform vec3 uFogColor;
uniform float uFogDensity;
varying vec2 vUv2;
varying vec3 vWorldPos;

float hashg(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float vnoiseg(vec2 p){
  vec2 i = floor(p), f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hashg(i), hashg(i + vec2(1.0, 0.0)), u.x),
             mix(hashg(i + vec2(0.0, 1.0)), hashg(i + vec2(1.0, 1.0)), u.x), u.y);
}
void main(){
  float a = clamp(uFlow * 1.8, 0.0, 1.0);
  float speed = 0.5 + uFlow * 2.6;
  float n1 = vnoiseg(vec2(vUv2.x * 5.0,  vUv2.y * 42.0 - uTime * speed * 3.0));
  float n2 = vnoiseg(vec2(vUv2.x * 11.0 + 4.7, vUv2.y * 80.0 - uTime * speed * 5.2));
  float streak = smoothstep(0.36, 0.74, n1 * 0.6 + n2 * 0.4);
  float edge = smoothstep(0.0, 0.09, vUv2.x) * (1.0 - smoothstep(0.91, 1.0, vUv2.x));
  float ends = smoothstep(0.0, 0.05, vUv2.y) * (1.0 - smoothstep(0.88, 1.0, vUv2.y));
  vec3 base = mix(vec3(0.05, 0.16, 0.17), vec3(0.62, 0.72, 0.76), uWhite);
  vec3 col = mix(base, vec3(0.95, 0.98, 1.0), streak * (0.3 + 0.5 * a));
  float alpha = (0.5 + streak * 0.4) * edge * ends * a;
  if (alpha < 0.01) discard;
  float dist = length(cameraPosition - vWorldPos);
  float fogF = 1.0 - exp(-pow(dist * uFogDensity, 2.0));
  col = mix(col, uFogColor, fogF);
  gl_FragColor = vec4(col, alpha);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}`;

/* ---------------- geometry helpers ---------------- */
const UP = new THREE.Vector3(0, 1, 0);

function ribbonGeometry(pts, width) {   /* pts: Vector3[] in world space; v runs 0→1 along path */
  const pos = [], uv = [], idx = [];
  const dir = new THREE.Vector3(), side = new THREE.Vector3();
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i];
    const prev = pts[Math.max(0, i - 1)], next = pts[Math.min(pts.length - 1, i + 1)];
    dir.subVectors(next, prev).normalize();
    side.crossVectors(dir, UP).normalize().multiplyScalar(width / 2);
    const l = p.clone().sub(side), r = p.clone().add(side);
    pos.push(l.x, l.y, l.z, r.x, r.y, r.z);
    const v = i / (pts.length - 1);
    uv.push(0, v, 1, v);
    if (i > 0) { const a = (i - 1) * 2; idx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2); }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
  g.setIndex(idx);
  return g;
}
function terrainRibbon(zStart, zEnd, step, width) {
  const pts = [];
  for (let z = zStart; z <= zEnd; z += step) {
    const x = valleyCenterX(z);
    pts.push(new THREE.Vector3(x, terrainHeight(x, z) + 1.1, z));
  }
  return ribbonGeometry(pts, width);
}
function waypointRibbon(waypoints, width) {
  const pts = [];
  for (let i = 0; i < waypoints.length - 1; i++) {
    const a = waypoints[i], b = waypoints[i + 1];
    const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
    const n = Math.max(2, Math.ceil(len / 5));
    for (let s = 0; s <= n; s++) {
      const x = lerp(a[0], b[0], s / n), z = lerp(a[1], b[1], s / n);
      pts.push(new THREE.Vector3(x, terrainHeight(x, z) + 0.9, z));
    }
  }
  return ribbonGeometry(pts, width);
}
/* organic blob jitter — position-based, so duplicated vertices stay welded */
function jitterGeometry(geo, amp, yScale, yLift, seed) {
  const p = geo.attributes.position;
  const v = new THREE.Vector3();
  for (let i = 0; i < p.count; i++) {
    v.fromBufferAttribute(p, i);
    const k = 1 - amp * 0.5 + amp * vnoise(v.x * 1.1 + seed, v.y * 1.1 + v.z * 1.1 + seed);
    v.multiplyScalar(k);
    v.y = v.y * yScale + yLift;
    p.setXYZ(i, v.x, v.y, v.z);
  }
  geo.computeVertexNormals();
  return geo;
}

/* ---------------- camera presets ---------------- */
const PRESETS = {
  overview:   { p: [ 480, 360,  620], t: [  0,  40,   30] },
  reservoir1: { p: [-300, 240,  -40], t: [-92, 100, -290] },
  reservoir2: { p: [ 260, 200,  260], t: [ 25,  52,    0] },
  reservoir3: { p: [-220, 140,  540], t: [ 43,   4,  290] },
  downstream: { p: [ 140,  90,  760], t: [ 40, -25,  520] },
};

const INITIAL_STATE = {
  reservoirs: {
    reservoir_1: { water_level: 0.65, storage: 0.72, inflow: 45, release: 30, gate: 0.40, risk: 'normal'  },
    reservoir_2: { water_level: 0.58, storage: 0.65, inflow: 35, release: 25, gate: 0.35, risk: 'normal'  },
    reservoir_3: { water_level: 0.75, storage: 0.80, inflow: 40, release: 20, gate: 0.50, risk: 'warning' },
  },
  storm_intensity: 0.7,
  downstream_flow: 65,
  controller_mode: 'AI',
  simulation_time: new Date().toISOString(),
  hardware_status: {
    esp32: 'NOT_CONNECTED', water_level_sensor: 'NOT_CONNECTED',
    flow_sensor: 'NOT_CONNECTED', gate_actuator: 'NOT_CONNECTED',
  },
};

/* ================================================================
   THE RENDERER
   ================================================================ */
class ReservoirTwinRenderer {

  constructor(container, opts = {}) {
    this.container = container;
    this.opts = opts;
    this._disposed = false;
    this._raf = 0;
    this._last = performance.now();
    this._time = 0;
    this._hudT = 0;
    this._v3 = new THREE.Vector3();          /* hoisted — no per-frame allocation */
    this.onTick = null;                      /* single external hook, called by THE loop */

    /* per-reservoir runtime */
    this.R = BASINS.map(B => {
      const D = DAMSITES[B.id - 1];
      const full = valleyFloor(B.cz) - 4;
      const floorTrue = terrainHeight(D.xc, D.zd);
      return {
        B, D, full,
        base: floorTrue - 3,
        minY: floorTrue + 9,
        sill: full - 16,                     /* mid-high outlet: real hydraulic drop, flows above ~73% */
        crest: full + 2.4,
        levelCur: 0.6, levelTgt: 0.6,
        gateCur: 0.4, gateTgt: 0.4,
        jetFlow: 0, jetTgt: 0, waterY: 0,
        data: { level: 0.6, storage: 0.6, inflow: 0, release: 0, gate: 0.4, risk: 'normal' },
      };
    });
    this.storm = { cur: 0.5, tgt: 0.7 };
    this.data = {
      downstream: 65, mode: 'AI',
      simTime: INITIAL_STATE.simulation_time,
      hardware: INITIAL_STATE.hardware_status,
    };

    this._initCore();
    this._initTerrain();
    this._initVegetation();
    this._initDams();
    this._initWater();
    this._initChannels();
    this._initRain();
    this._initMist();
    this._initControls();
    if (opts.testControls !== false) this._initHUD();

    this.updateState(INITIAL_STATE);
    for (const r of this.R) { r.levelCur = r.levelTgt; r.gateCur = r.gateTgt; }

    this._onResize = () => {
      const w = this.container.clientWidth || window.innerWidth;
      const h = this.container.clientHeight || window.innerHeight;
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(w, h);
      this._cachePanelAnchors();
    };
    window.addEventListener('resize', this._onResize);
    this._onResize();

    const boot = this.container.querySelector('#boot');
    if (boot) boot.remove();

    this._tick = this._tick.bind(this);
    this._raf = requestAnimationFrame(this._tick);      /* the ONLY loop start */
    
    window.addEventListener("message", (event) => {
      if (event.data.type === "streamlit:render") {
        if (event.data.args && event.data.args.state) {
          this.updateState(event.data.args.state);
        }
      }
    });
  }

  /* ================= public API ================= */
  updateState(state) {
    if (!state || this._disposed) return;
    const rs = state.reservoirs || {};
    for (let i = 0; i < 3; i++) {
      const r = this.R[i];
      const src = rs['reservoir_' + (i + 1)] || {};
      const lvlRaw = src.water_level !== undefined ? src.water_level
                   : (src.storage !== undefined ? src.storage : r.levelTgt);
      r.levelTgt = clamp01(fracv(lvlRaw));
      r.gateTgt = clamp01(fracv(src.gate !== undefined ? src.gate : r.gateTgt));
      r.data = {
        level: r.levelTgt,
        storage: clamp01(fracv(src.storage !== undefined ? src.storage : r.data.storage)),
        inflow: numv(src.inflow), release: numv(src.release),
        gate: r.gateTgt,
        risk: RISKS.includes(src.risk) ? src.risk : 'normal',
      };
    }
    this.storm.tgt = clamp01(fracv(state.storm_intensity !== undefined ? state.storm_intensity : this.storm.tgt));
    this.data.downstream = numv(state.downstream_flow);
    if (typeof state.controller_mode === 'string') this.data.mode = state.controller_mode;
    if (state.simulation_time) this.data.simTime = String(state.simulation_time);
    if (state.hardware_status) this.data.hardware = state.hardware_status;
  }

  setCameraPreset(name) {
    const p = PRESETS[name];
    if (!p || this._disposed) return;
    this._tween = {
      t: 0,
      fromP: this.camera.position.clone(),
      fromT: this.controls.target.clone(),
      toP: new THREE.Vector3(...p.p),
      toT: new THREE.Vector3(...p.t),
    };
    this.controls.enabled = false;
  }

  dispose() {
    if (this._disposed) return;
    this._disposed = true;
    cancelAnimationFrame(this._raf);
    window.removeEventListener('resize', this._onResize);
    this.controls.dispose();

    this.scene.traverse(obj => {
      if (obj.geometry) obj.geometry.dispose();
      if (obj.isInstancedMesh && typeof obj.dispose === 'function') obj.dispose();
      const mats = Array.isArray(obj.material) ? obj.material : (obj.material ? [obj.material] : []);
      for (const m of mats) {
        if (m.uniforms) {
          for (const key in m.uniforms) {
            const v = m.uniforms[key].value;
            if (v && v.isTexture) v.dispose();
          }
        }
        for (const key in m) {
          const v = m[key];
          if (v && v.isTexture) v.dispose();
        }
        m.dispose();
      }
    });
    this.renderer.dispose();
    if (this.renderer.forceContextLoss) this.renderer.forceContextLoss();
    if (this.renderer.domElement && this.renderer.domElement.parentNode) {
      this.renderer.domElement.parentNode.removeChild(this.renderer.domElement);
    }
    if (this.hudRoot && this.hudRoot.parentNode) this.hudRoot.parentNode.removeChild(this.hudRoot);
    if (this.barRoot && this.barRoot.parentNode) this.barRoot.parentNode.removeChild(this.barRoot);
  }

  /* ================= THE central animation loop ================= */
  _tick(now) {
    if (this._disposed) return;
    const dt = Math.min(0.05, Math.max(0.001, (now - this._last) / 1000));
    this._last = now;
    this._time += dt;

    if (this.onTick) this.onTick(dt);        /* test driver / future Python bridge */
    this._updateCameraTween(dt);
    this._updateStorm(dt);
    this._updateWater(dt);
    this._updateGatesAndJets(dt);
    this._updateRain(dt);
    this._updateMist(dt);
    this._updateConnectors();
    this._updateHUD(dt);
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    this._raf = requestAnimationFrame(this._tick);
  }

  _fail(msg) {
    const boot = this.container.querySelector('#boot');
    if (boot) boot.textContent = msg;
    this._disposed = true;
  }

  /* ================= core: renderer / scene / lights / sky ================= */
  _initCore() {
    try {
      this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    } catch (e) {
      this._fail('WEBGL UNAVAILABLE — CANNOT RENDER DIGITAL TWIN');
      throw e;
    }
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.domElement.addEventListener('contextmenu', e => e.preventDefault());
    this.container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(55, 1, 1, 6000);
    this.camera.position.set(...PRESETS.overview.p);

    this.scene.fog = new THREE.FogExp2(0xb9c3ca, 0.0016);
    this.hemi = new THREE.HemisphereLight(0xc6d3de, 0x33502c, 0.85);
    this.scene.add(this.hemi);
    this.sun = new THREE.DirectionalLight(0xf2f6fa, 1.5);
    this.sun.position.set(250, 400, 150);
    this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(2048, 2048);
    const sc = this.sun.shadow.camera;
    sc.left = -460; sc.right = 460; sc.top = 460; sc.bottom = -460; sc.near = 50; sc.far = 1400;
    this.sun.shadow.bias = -0.0004;
    this.sun.shadow.normalBias = 2;
    this.scene.add(this.sun);

    const sky = new THREE.Mesh(
      new THREE.SphereGeometry(2400, 24, 14),
      new THREE.MeshBasicMaterial({ map: skyTex, side: THREE.BackSide, fog: false })
    );
    sky.name = 'SKY_OVERCAST';
    this.scene.add(sky);
  }

  /* ================= terrain (two-pass: heights grid, then slope+color) ================= */
  _initTerrain() {
    const SIZE = 1400, SEG = 280, N = SEG + 1, cell = SIZE / SEG;
    const geo = new THREE.PlaneGeometry(SIZE, SIZE, SEG, SEG);
    geo.rotateX(-Math.PI / 2);
    const pos = geo.attributes.position;
    const heights = new Float32Array(pos.count);
    for (let i = 0; i < pos.count; i++) {
      const h = terrainHeight(pos.getX(i), pos.getZ(i));
      heights[i] = h;
      pos.setY(i, h);
    }
    const colors = new Float32Array(pos.count * 3);
    const col = new THREE.Color();
    for (let i = 0; i < pos.count; i++) {
      const wx = pos.getX(i), wz = pos.getZ(i), h = heights[i];
      const row = Math.floor(i / N), ci = i % N;
      const hL = heights[row * N + Math.max(0, ci - 1)];
      const hR = heights[row * N + Math.min(SEG, ci + 1)];
      const hU = heights[Math.max(0, row - 1) * N + ci];
      const hD = heights[Math.min(SEG, row + 1) * N + ci];
      const slope = (Math.abs(hR - hL) + Math.abs(hD - hU)) / (2 * cell);

      const n1 = fbm(wx * 0.02 + 11.3, wz * 0.02 + 5.7);
      const n2 = fbm(wx * 0.09 + 3.1, wz * 0.09 + 8.2);
      let r = 0.12 + n1 * 0.15 + n2 * 0.05;
      let g = 0.31 + n1 * 0.14 + n2 * 0.06;
      let b = 0.09 + n1 * 0.07 + n2 * 0.03;
      const dry = clamp01((h - 120) / 80);
      r += dry * 0.10; g += dry * 0.02;
      const rockT = clamp01((slope - 0.55) / 0.45) * (0.6 + n2 * 0.4);
      r = lerp(r, 0.33 + n2 * 0.10, rockT);
      g = lerp(g, 0.34 + n2 * 0.08, rockT);
      b = lerp(b, 0.32 + n2 * 0.08, rockT);
      let wet = 0;                                    /* soaked dark ground near all water lines */
      for (const B of BASINS) {
        const dz = wz - B.cz;
        if (dz > -B.up - 12 && dz < B.dn + 16 && Math.abs(wx - valleyCenterX(wz)) < B.rx + 25) {
          wet = Math.max(wet, clamp01((valleyFloor(B.cz) - 4 + 5 - h) / 10));
        }
      }
      wet = Math.max(wet, clamp01(channelCarve(wx, wz) / 7) * 0.7);
      wet = Math.max(wet, clamp01(plungeCarve(wx, wz) / 14) * 0.8);
      r = lerp(r, 0.09, wet * 0.55); g = lerp(g, 0.15, wet * 0.5); b = lerp(b, 0.09, wet * 0.5);
      col.setRGB(r, g, b, THREE.SRGBColorSpace);
      colors[i * 3] = col.r; colors[i * 3 + 1] = col.g; colors[i * 3 + 2] = col.b;
    }
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geo.computeVertexNormals();
    const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({
      vertexColors: true, roughness: 0.95, metalness: 0,
    }));
    mesh.name = 'TERRAIN_WESTERN_GHATS';
    mesh.receiveShadow = true;
    this.scene.add(mesh);
  }

  /* ================= vegetation & rocks — InstancedMesh(BufferGeometry) only ================= */
  _initVegetation() {
    const group = new THREE.Group();
    group.name = 'VEGETATION';

    const scatter = (count, cfg) => {
      const pts = []; let tries = 0;
      while (pts.length < count && tries < count * 40) {
        tries++;
        const x = (rng() * 2 - 1) * 640, z = (rng() * 2 - 1) * 640;
        const h = terrainHeight(x, z);
        if (h > cfg.maxH) continue;
        if (terrainSlope(x, z, h) > cfg.maxSlope) continue;
        let skip = false;
        for (const B of BASINS) {
          const dz = z - B.cz;
          if (dz > -B.up - 12 && dz < B.dn + 16 && Math.abs(x - valleyCenterX(z)) < B.rx + 22) {
            if (h < valleyFloor(B.cz) - 4 + (cfg.shore ? -1 : 2.5)) { skip = true; break; }
          }
        }
        if (skip) continue;
        if (cfg.avoidChannel && channelCarve(x, z) > 2.5) continue;
        if (cfg.avoidPlunge && plungeCarve(x, z) > 2) continue;
        if (cfg.gate && fbm(x * 0.012 + 40, z * 0.012 + 9) < cfg.gate) continue;
        pts.push({ x, z, h });
      }
      return pts;
    };
    const transforms = (pts, sMin, sMax) => pts.map(p => {
      const s = sMin + rng() * (sMax - sMin);
      return { x: p.x, y: p.h - 0.12, z: p.z, ry: rng() * Math.PI * 2,
               sx: s * (0.85 + rng() * 0.3), sy: s, sz: s * (0.85 + rng() * 0.3) };
    });
    /* hue = [baseH, hueRange, saturation, baseL, lightnessRange] */
    const instanced = (geo, mat, tf, hue, shadow) => {
      const mesh = new THREE.InstancedMesh(geo, mat, tf.length);   /* geometry is a BufferGeometry */
      const d = new THREE.Object3D(), c = new THREE.Color();
      for (let i = 0; i < tf.length; i++) {
        const t = tf[i];
        d.position.set(t.x, t.y, t.z);
        d.rotation.set(0, t.ry, 0);
        d.scale.set(t.sx, t.sy, t.sz);
        d.updateMatrix();
        mesh.setMatrixAt(i, d.matrix);
        if (hue) c.setHSL(hue[0] + rng() * hue[1], hue[2], hue[3] + rng() * hue[4], THREE.SRGBColorSpace);
        if (hue) mesh.setColorAt(i, c);
      }
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
      mesh.frustumCulled = false;               /* instance spread exceeds geometry bounds */
      mesh.castShadow = !!shadow;
      group.add(mesh);
      return mesh;
    };

    /* trees: shared transform list drives trunk + canopy InstancedMesh pair */
    const treePts = scatter(3000, { maxSlope: 0.5, maxH: 175, avoidChannel: true, avoidPlunge: true, gate: 0.34 });
    const treeTf = transforms(treePts, 1.1, 2.9);
    const trunkGeo = new THREE.CylinderGeometry(0.13, 0.24, 2.6, 7).translate(0, 1.3, 0);
    const canopyGeo = jitterGeometry(new THREE.SphereGeometry(1.5, 18, 12), 0.5, 0.85, 2.4, 3.7);
    instanced(trunkGeo, new THREE.MeshStandardMaterial({ color: 0x4e3a2a, roughness: 0.95 }), treeTf, null, false);
    const canopy = instanced(canopyGeo, new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.88 }),
      treeTf, [0.30, 0.07, 0.55, 0.22, 0.10], true);
    canopy.name = 'FOREST_CANOPY';

    const shrubPts = scatter(2400, { maxSlope: 0.8, maxH: 200, avoidChannel: true, avoidPlunge: true, gate: 0.30 });
    instanced(jitterGeometry(new THREE.SphereGeometry(0.9, 12, 9), 0.4, 0.65, 0.45, 9.2),
      new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.9 }),
      transforms(shrubPts, 0.8, 2.0), [0.27, 0.08, 0.50, 0.24, 0.10], false);

    const grassPts = scatter(3200, { maxSlope: 0.95, maxH: 190, avoidChannel: true, avoidPlunge: true, gate: 0.25 });
    instanced(jitterGeometry(new THREE.SphereGeometry(0.5, 8, 6), 0.45, 0.80, 0.22, 15.8),
      new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 1.0 }),
      transforms(grassPts, 0.5, 1.2), [0.24, 0.08, 0.60, 0.28, 0.12], false);

    const rockPts = scatter(850, { maxSlope: 2.0, maxH: 320, shore: true, avoidChannel: false, avoidPlunge: false });
    const rock = instanced(jitterGeometry(new THREE.DodecahedronGeometry(1.1, 0), 0.28, 1, 0, 21.4),
      new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.6, metalness: 0.05 }),
      transforms(rockPts, 0.6, 2.6), [0.58, 0.02, 0.05, 0.26, 0.22], true);
    rock.name = 'ROCKS';

    this.scene.add(group);
  }

  /* ================= dams, gates, spillways ================= */
  _initDams() {
    this.concreteMat = new THREE.MeshStandardMaterial({ map: concreteTex, roughness: 0.85, metalness: 0.08 });
    this.metalMat = new THREE.MeshStandardMaterial({ color: 0x5a6367, roughness: 0.4, metalness: 0.7 });
    this.darkMat = new THREE.MeshStandardMaterial({ color: 0x14181b, roughness: 0.9 });

    for (const r of this.R) {
      const { D, base, crest, sill } = r;
      const H = crest - base;
      const parts = [];
      const segs = 9, span = 300, radius = 900, segW = span / segs;

      /* arched gravity wall (convex upstream) + crest road + railings */
      for (let s = 0; s < segs; s++) {
        const frac = (s + 0.5) / segs - 0.5;
        const ang = frac * (span / radius);
        const x = D.xc + Math.sin(ang) * radius;
        const z = D.zd + (Math.cos(ang) - 1) * radius;
        const yaw = -ang;
        const wall = new THREE.BoxGeometry(segW + 2.2, H, 12);
        wall.rotateY(yaw); wall.translate(x, base + H / 2, z); parts.push(wall);
        const road = new THREE.BoxGeometry(segW + 2.2, 1.2, 14.5);
        road.rotateY(yaw); road.translate(x, crest + 0.6, z); parts.push(road);
        for (const zo of [-6.4, 6.4]) {
          const rail = new THREE.BoxGeometry(segW + 2.2, 0.9, 0.25);
          rail.rotateY(yaw); rail.translate(x, crest + 1.3, z + zo); parts.push(rail);
        }
      }
      /* spillway piers */
      for (const xo of [-15, 15]) {
        const pier = new THREE.BoxGeometry(4, crest - sill + 5, 9);
        pier.translate(D.xc + xo, (sill - 2 + crest + 3) / 2, D.zd - 2); parts.push(pier);
      }
      /* ski-jump chute: from sill down to the plunge pool, terrain-derived endpoint */
      const cTop = new THREE.Vector3(D.xc, sill, D.zd + 4);
      const cBot = new THREE.Vector3(D.xc, terrainHeight(D.xc, D.zd + 24) + 0.6, D.zd + 24);
      const cDir = new THREE.Vector3().subVectors(cBot, cTop).normalize();
      const cLen = cTop.distanceTo(cBot) + 2;
      const chute = new THREE.BoxGeometry(26, cLen, 5);
      chute.applyMatrix4(new THREE.Matrix4().compose(
        new THREE.Vector3().addVectors(cTop, cBot).multiplyScalar(0.5),
        new THREE.Quaternion().setFromUnitVectors(UP, cDir),
        new THREE.Vector3(1, 1, 1)));
      parts.push(chute);
      /* control tower on right abutment */
      const tX = D.xc + 62, tZ = D.zd - 10, tY = terrainHeight(tX, tZ) - 2;
      parts.push(new THREE.CylinderGeometry(2.6, 3.2, 16, 10).translate(tX, tY + 8, tZ));
      parts.push(new THREE.BoxGeometry(8, 3.5, 8).translate(tX, tY + 17, tZ));

      const dam = new THREE.Mesh(mergeGeometries(parts, false), this.concreteMat);
      dam.castShadow = true; dam.receiveShadow = true;
      dam.name = 'DAM_0' + r.B.id;
      this.scene.add(dam);

      /* hoist beam (metal, static) */
      const hoist = new THREE.Mesh(new THREE.BoxGeometry(34, 1.5, 3), this.metalMat);
      hoist.position.set(D.xc, crest + 3.2, D.zd - 2);
      hoist.castShadow = true;
      dam.add(hoist);

      /* dark gate opening on upstream face — revealed as the gate lifts */
      const opening = new THREE.Mesh(new THREE.BoxGeometry(24, 8.5, 0.6), this.darkMat);
      opening.name = 'GATE_OPENING_0' + r.B.id;
      opening.position.set(D.xc, sill + 3.5, D.zd - 6.4);
      dam.add(opening);

      /* GATE — the only animated dam part; slide = gate opening */
      const gate = new THREE.Mesh(new THREE.BoxGeometry(22, 9, 1.4), this.metalMat);
      gate.name = 'GATE_0' + r.B.id;
      gate.position.set(D.xc, sill + 3.8, D.zd - 7.5);
      gate.castShadow = true;
      dam.add(gate);
      r.gateMesh = gate;
      r.gateClosedY = sill + 3.8;
      r.gateTravel = 8;

      /* spillway jet (flow-driven) + plunge spray */
      const jTop = new THREE.Vector3(D.xc, sill + 1, D.zd + 4);
      const jBot = new THREE.Vector3(D.xc, terrainHeight(D.xc, D.zd + 22) + 1.2, D.zd + 22);
      const jCtrl = new THREE.Vector3(D.xc, sill + 5, D.zd + 11);
      r.jetMat = new THREE.ShaderMaterial({
        vertexShader: FLOW_VERT, fragmentShader: FLOW_FRAG, transparent: true,
        depthWrite: false, side: THREE.DoubleSide,
        uniforms: {
          uTime: { value: 0 }, uFlow: { value: 0 }, uWhite: { value: 0.85 },
          uFogColor: { value: new THREE.Color(0xb9c3ca) }, uFogDensity: { value: 0.0016 },
        },
      });
      const jet = new THREE.Mesh(
        ribbonGeometry(new THREE.QuadraticBezierCurve3(jTop, jCtrl, jBot).getPoints(24), 17),
        r.jetMat);
      jet.name = 'SPILLWAY_JET_0' + r.B.id;
      jet.renderOrder = 1;
      this.scene.add(jet);
      r.plunge = jBot;

      const sprayN = 220;
      const sprayPos = new Float32Array(sprayN * 3);
      r.sprayVel = new Float32Array(sprayN * 3);
      for (let i = 0; i < sprayN; i++) {
        sprayPos[i * 3] = jBot.x; sprayPos[i * 3 + 1] = jBot.y; sprayPos[i * 3 + 2] = jBot.z;
        this._respawnSpray(r, i);
      }
      const sprayGeo = new THREE.BufferGeometry();
      sprayGeo.setAttribute('position', new THREE.BufferAttribute(sprayPos, 3));
      r.sprayMat = new THREE.PointsMaterial({
        map: mistTex, color: 0xeaf4fa, size: 2.4, transparent: true, opacity: 0,
        depthWrite: false, sizeAttenuation: true,
      });
      const spray = new THREE.Points(sprayGeo, r.sprayMat);
      spray.name = 'PLUNGE_SPRAY_0' + r.B.id;
      spray.renderOrder = 4;
      spray.frustumCulled = false;
      this.scene.add(spray);
      r.spray = spray;
    }
  }
  _respawnSpray(r, i) {
    r.sprayVel[i * 3] = (rng() - 0.5) * 10;
    r.sprayVel[i * 3 + 1] = 8 + rng() * 10;
    r.sprayVel[i * 3 + 2] = (rng() - 0.5) * 10 + 4;
  }

  /* ================= reservoir water — terrain-aware shader surface ================= */
  _initWater() {
    for (const r of this.R) {
      const B = r.B;
      const cx = valleyCenterX(B.cz);
      const zC = B.cz - 16;                       /* asymmetric: upstream margin, ends inside dam wall */
      const L = B.up + B.dn + 15;
      const geo = new THREE.PlaneGeometry(380, L, 60, 88);
      const pos = geo.attributes.position;
      const aTerrain = new Float32Array(pos.count);
      for (let i = 0; i < pos.count; i++) {
        aTerrain[i] = terrainHeight(pos.getX(i) + cx, -pos.getY(i) + zC);
      }
      geo.setAttribute('aTerrain', new THREE.BufferAttribute(aTerrain, 1));
      r.waterMat = new THREE.ShaderMaterial({
        vertexShader: WATER_VERT, fragmentShader: WATER_FRAG,
        transparent: true, depthWrite: true, side: THREE.FrontSide,
        uniforms: {
          uTime: { value: 0 }, uWaveAmp: { value: 0.8 }, uLevelY: { value: r.minY },
          uRain: { value: 0.5 },
          uDeep: { value: new THREE.Color(0x0a3d46) },
          uShallow: { value: new THREE.Color(0x1f6e64) },
          uSky: { value: new THREE.Color(0xb9c8d2) },
          uSunDir: { value: new THREE.Vector3(250, 400, 150).normalize() },
          uFogColor: { value: new THREE.Color(0xb9c3ca) },
          uFogDensity: { value: 0.0016 },
        },
      });
      const mesh = new THREE.Mesh(geo, r.waterMat);
      mesh.name = 'WATER_0' + B.id;
      mesh.rotation.x = -Math.PI / 2;
      mesh.position.set(cx, r.minY, zC);
      mesh.renderOrder = 2;                       /* after jets/channels → correct submergence */
      const resGroup = new THREE.Group();
      resGroup.name = 'RESERVOIR_0' + B.id;
      resGroup.userData = { basin: B, full: r.full, minLevel: r.minY };
      resGroup.add(mesh);
      this.scene.add(resGroup);
      r.waterMesh = mesh;
    }
  }

  /* ================= channels + inflows (release/inflow-driven) ================= */
  _initChannels() {
    const flowMat = white => new THREE.ShaderMaterial({
      vertexShader: FLOW_VERT, fragmentShader: FLOW_FRAG,
      transparent: true, depthWrite: false, side: THREE.DoubleSide,
      uniforms: {
        uTime: { value: 0 }, uFlow: { value: 0.5 }, uWhite: { value: white },
        uFogColor: { value: new THREE.Color(0xb9c3ca) }, uFogDensity: { value: 0.0016 },
      },
    });
    this.flowMeshes = [];
    const addFlow = (geo, name, white, key) => {
      const mat = flowMat(white);
      const mesh = new THREE.Mesh(geo, mat);
      mesh.name = name;
      mesh.renderOrder = 1;
      this.scene.add(mesh);
      this.flowMeshes.push({ mat, key });
    };
    /* dam → next reservoir / final downstream */
    addFlow(terrainRibbon(-150, -40, 6, 13), 'CHANNEL_01', 0.18, 'ch1');
    addFlow(terrainRibbon( 140, 250, 6, 13), 'CHANNEL_02', 0.18, 'ch2');
    addFlow(terrainRibbon( 430, 580, 6, 14), 'CHANNEL_03', 0.22, 'ch3');
    /* upstream river + tributaries */
    addFlow(terrainRibbon(-470, -408, 6, 12), 'INFLOW_01', 0.28, 'in1');
    addFlow(waypointRibbon([[150, 80], [110, 45], [70, 10]], 7), 'INFLOW_02', 0.30, 'in2');
    addFlow(waypointRibbon([[170, 330], [125, 312], [80, 295], [65, 285]], 7), 'INFLOW_03', 0.30, 'in3');
  }

  /* ================= rain — ONE particle system ================= */
  _initRain() {
    const N = 6000;
    this.rainVel = new Float32Array(N * 3);
    const posArr = new Float32Array(N * 3);
    for (let i = 0; i < N; i++) {
      posArr[i * 3] = (rng() - 0.5) * 560;
      posArr[i * 3 + 1] = rng() * 240;
      posArr[i * 3 + 2] = (rng() - 0.5) * 560;
      this.rainVel[i * 3] = 14 + rng() * 10;
      this.rainVel[i * 3 + 1] = -(80 + rng() * 30);
      this.rainVel[i * 3 + 2] = (rng() - 0.5) * 10;
    }
    this.rainGeo = new THREE.BufferGeometry();
    this.rainGeo.setAttribute('position', new THREE.BufferAttribute(posArr, 3));
    this.rainGeo.setDrawRange(0, 4000);
    this.rainMat = new THREE.PointsMaterial({
      map: rainTex, color: 0xcfe2ee, size: 2.0, transparent: true, opacity: 0.55,
      depthWrite: false, sizeAttenuation: true,
    });
    this.rain = new THREE.Points(this.rainGeo, this.rainMat);
    this.rain.name = 'RAIN_SYSTEM';
    this.rain.renderOrder = 5;
    this.rain.frustumCulled = false;
    this.scene.add(this.rain);
  }

  /* ================= mist + low clouds (horizontal layers) ================= */
  _initMist() {
    this.mistPlanes = [];
    const addMist = (y, w, h, op, spd, cloud) => {
      const mat = new THREE.MeshBasicMaterial({
        map: mistTex, transparent: true, opacity: op, depthWrite: false,
        side: THREE.DoubleSide, color: cloud ? 0xd8e0e6 : 0xc4d0d8,
        fog: !cloud,                               /* high clouds must not dissolve into fog */
      });
      const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), mat);
      m.position.set((rng() - 0.5) * 700, y, (rng() - 0.5) * 700);
      m.rotation.x = -Math.PI / 2 + (rng() - 0.5) * 0.25;
      m.rotation.y = rng() * Math.PI;
      m.renderOrder = 3;
      m.frustumCulled = false;
      this.scene.add(m);
      this.mistPlanes.push({ mesh: m, baseO: op, speed: spd, phase: rng() * 10, cloud });
    };
    for (let i = 0; i < 7; i++) addMist(25 + rng() * 85, 380 + rng() * 160, 110 + rng() * 60, 0.10 + rng() * 0.12, 2 + rng() * 3, false);
    for (let i = 0; i < 4; i++) addMist(260 + rng() * 80, 640 + rng() * 260, 220, 0.16 + rng() * 0.10, 1 + rng() * 1.5, true);
    const g = new THREE.Group(); g.name = 'MIST_LAYER'; this.scene.add(g);
  }

  /* ================= camera controls ================= */
  _initControls() {
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.06;
    this.controls.minDistance = 40;
    this.controls.maxDistance = 2000;
    this.controls.maxPolarAngle = 1.52;
    this.controls.target.set(...PRESETS.overview.t);
  }
  _updateCameraTween(dt) {
    if (!this._tween) return;
    const tw = this._tween;
    tw.t += dt / 1.6;
    const k = smooth01(tw.t);
    this.camera.position.lerpVectors(tw.fromP, tw.toP, k);
    this.controls.target.lerpVectors(tw.fromT, tw.toT, k);
    if (tw.t >= 1) { this._tween = null; this.controls.enabled = true; }
  }

  /* ================= per-frame subsystem updates ================= */
  _updateStorm(dt) {
    const s = this.storm;
    s.cur += (s.tgt - s.cur) * Math.min(1, dt * 0.6);
    const c = s.cur;
    this.scene.fog.density = 0.0014 + c * 0.0011;
    this.sun.intensity = 1.5 - c * 0.75;
    this.hemi.intensity = 0.85 - c * 0.30;
    this.rainMat.opacity = 0.28 + c * 0.42;
    this.rainGeo.setDrawRange(0, 900 + Math.floor(c * 5100));
    for (const r of this.R) {
      r.waterMat.uniforms.uRain.value = c;
      r.waterMat.uniforms.uWaveAmp.value = 0.55 + c * 0.55;
    }
  }

  _updateWater(dt) {
    for (const r of this.R) {
      r.levelCur += (r.levelTgt - r.levelCur) * Math.min(1, dt * 0.8);
      const y = lerp(r.minY, r.full, r.levelCur);
      r.waterMesh.position.y = y;                 /* REAL surface movement, not a label */
      r.waterMat.uniforms.uTime.value = this._time;
      r.waterMat.uniforms.uLevelY.value = y;
      r.waterY = y;
    }
    const flows = {
      ch1: clamp01(this.R[0].data.release / 70), ch2: clamp01(this.R[1].data.release / 70),
      ch3: clamp01(this.R[2].data.release / 70),
      in1: clamp01(this.R[0].data.inflow / 120), in2: clamp01(this.R[1].data.inflow / 120),
      in3: clamp01(this.R[2].data.inflow / 120),
    };
    for (const fm of this.flowMeshes) {
      fm.mat.uniforms.uFlow.value = flows[fm.key];
      fm.mat.uniforms.uTime.value = this._time;
    }
  }

  _updateGatesAndJets(dt) {
    for (const r of this.R) {
      r.gateCur += (r.gateTgt - r.gateCur) * Math.min(1, dt * 2.5);
      r.gateMesh.position.y = r.gateClosedY + r.gateCur * r.gateTravel;

      const head = clamp01((r.waterY - r.sill) / 5);          /* hydraulic head over the sill */
      r.jetTgt = Math.max(head * r.gateCur, r.levelTgt > 0.995 ? 1 : 0);
      r.jetFlow += (r.jetTgt - r.jetFlow) * Math.min(1, dt * 3);
      r.jetMat.uniforms.uFlow.value = r.jetFlow;
      r.jetMat.uniforms.uTime.value = this._time;

      const active = r.jetFlow > 0.06;
      r.spray.visible = active;
      r.sprayMat.opacity = r.jetFlow * 0.55;
      if (active) {
        const p = r.spray.geometry.attributes.position.array;
        for (let i = 0; i < p.length / 3; i++) {
          p[i * 3] += r.sprayVel[i * 3] * dt;
          p[i * 3 + 1] += r.sprayVel[i * 3 + 1] * dt;
          p[i * 3 + 2] += r.sprayVel[i * 3 + 2] * dt;
          r.sprayVel[i * 3 + 1] -= 30 * dt;
          if (p[i * 3 + 1] < r.plunge.y) {
            p[i * 3] = r.plunge.x; p[i * 3 + 1] = r.plunge.y; p[i * 3 + 2] = r.plunge.z;
            this._respawnSpray(r, i);
          }
        }
        r.spray.geometry.attributes.position.needsUpdate = true;
      }
    }
  }

  _updateRain(dt) {
    this.rain.position.x += (this.controls.target.x - this.rain.position.x) * Math.min(1, dt * 1.5);
    this.rain.position.z += (this.controls.target.z - this.rain.position.z) * Math.min(1, dt * 1.5);
    const p = this.rainGeo.attributes.position.array;
    const N = p.length / 3;
    for (let i = 0; i < N; i++) {
      const i3 = i * 3;
      p[i3] += this.rainVel[i3] * dt;
      p[i3 + 1] += this.rainVel[i3 + 1] * dt;
      p[i3 + 2] += this.rainVel[i3 + 2] * dt;
      if (p[i3 + 1] < 0) { p[i3 + 1] = 170 + rng() * 70; p[i3] = (rng() - 0.5) * 560; p[i3 + 2] = (rng() - 0.5) * 560; }
      if (p[i3] > 300) p[i3] -= 600; else if (p[i3] < -300) p[i3] += 600;
      if (p[i3 + 2] > 300) p[i3 + 2] -= 600; else if (p[i3 + 2] < -300) p[i3 + 2] += 600;
    }
    this.rainGeo.attributes.position.needsUpdate = true;
  }

  _updateMist(dt) {
    const s = this.storm.cur;
    for (const m of this.mistPlanes) {
      m.mesh.position.x += m.speed * dt * (m.cloud ? 0.5 : 1);
      if (m.mesh.position.x > 420) m.mesh.position.x = -420;
      m.mesh.material.opacity = m.baseO * (0.55 + s * 0.95) *
        (0.75 + 0.25 * Math.sin(this._time * 0.25 + m.phase));
    }
  }

  /* ================= HUD — built once, refs cached, zero null risk ================= */
  _initHUD() {
    const hud = document.createElement('div');
    hud.id = 'hud';
    const panel = (id, cls, html) => {
      const d = document.createElement('div');
      d.className = 'hp ' + (cls || '');
      d.id = id;
      d.innerHTML = html;
      hud.appendChild(d);
      return d;
    };

    panel('hud-title', '', `
      <div class="t1">AI-BASED MULTI-RESERVOIR SYSTEM</div>
      <div class="t2">DIGITAL TWIN · WATER MANAGEMENT &amp; FLOOD PREVENTION</div>`);

    panel('hud-live', '', `
      <div class="r1"><span class="dot ok pulse" style="display:inline-block;margin-right:5px"></span>LIVE SIMULATION</div>
      <div class="r2" id="live-time">--:--:--</div>
      <div class="r2" id="live-wx">RAIN — HEAVY · 24°C</div>
      <div class="r3">KERALA, INDIA · 9.5°N 76.9°E</div>`);

    panel('hud-system', '', `
      <div class="hp-title">SYSTEM STATUS</div>
      <div class="hp-row"><span class="dot ok pulse"></span><span class="lbl">SIMULATION</span><span class="val">RUNNING</span></div>
      <div class="hp-row"><span class="dot cy"></span><span class="lbl">CONTROLLER</span><span class="val" data-ref="mode">AI / MPC</span></div>
      <div class="hp-row"><span class="dot cy"></span><span class="lbl">LSTM V3</span><span class="val">LIVE</span></div>
      <div class="hp-row"><span class="dot cy"></span><span class="lbl">RISK ENGINE</span><span class="val">ACTIVE</span></div>
      <div class="hp-row"><span class="dot warn"></span><span class="lbl">HARDWARE I/F</span><span class="val" style="color:var(--warn)">NOT CONNECTED</span></div>`);

    const weatherP = panel('hud-weather', '', `
      <div class="hp-title">WEATHER / STORM</div>
      <div class="hp-row"><span class="lbl">RAINFALL INT</span><span class="val" data-ref="wint">70%</span></div>
      <div class="bar"><div class="fill" data-ref="wbar"></div></div>
      <div class="hp-row"><span class="lbl">STORM LEVEL</span><span class="val" style="color:var(--warn)" data-ref="wlevel">HEAVY</span></div>
      <div class="hp-sub" style="color:var(--lbl);font-size:8px;letter-spacing:.2em;margin-top:6px">FORECAST</div>
      <div class="fc" data-ref="fc"></div>
      <div class="fc-lbls"><span>1D</span><span>3D</span><span>7D</span></div>`);
    const fc = weatherP.querySelector('[data-ref="fc"]');
    this.fcBars = [];
    for (let i = 0; i < 7; i++) { const b = document.createElement('div'); fc.appendChild(b); this.fcBars.push(b); }

    panel('hud-downstream', '', `
      <div class="hp-title">DOWNSTREAM FLOW</div>
      <div class="hp-row"><span class="lbl">CURRENT</span><span class="val" data-ref="dsval">-- m³/s</span></div>
      <div class="hp-row"><span class="lbl">SAFE LIMIT</span><span class="val">150 m³/s</span></div>
      <div class="bar"><div class="fill" data-ref="dsbar"></div></div>
      <div class="hp-row"><span class="lbl">STATUS</span><span class="val" data-ref="dsstat" style="color:var(--ok)">NORMAL</span></div>`);

    panel('hud-topology', '', `
      <div class="hp-title">CASCADE TOPOLOGY</div>
      <div class="topo">
        <div class="tnode" data-ref="tn1">R1</div><div class="tarrow">→</div>
        <div class="tnode" data-ref="tn2">R2</div><div class="tarrow">→</div>
        <div class="tnode" data-ref="tn3">R3</div><div class="tarrow">→</div>
        <div class="tnode" data-ref="tnds">DS</div>
      </div>`);

    panel('hud-hardware', '', `
      <div class="hp-title">HARDWARE INTERFACE</div>
      <div class="hp-row"><span class="dot warn"></span><span class="lbl">ESP32</span><span class="val" style="color:var(--warn)" data-ref="hw1">NOT CONNECTED</span></div>
      <div class="hp-row"><span class="dot warn"></span><span class="lbl">WATER LVL SENSOR</span><span class="val" style="color:var(--warn)" data-ref="hw2">NOT CONNECTED</span></div>
      <div class="hp-row"><span class="dot warn"></span><span class="lbl">FLOW SENSOR</span><span class="val" style="color:var(--warn)" data-ref="hw3">NOT CONNECTED</span></div>
      <div class="hp-row"><span class="dot warn"></span><span class="lbl">GATE ACTUATOR</span><span class="val" style="color:var(--warn)" data-ref="hw4">NOT CONNECTED</span></div>
      <div class="hp-row"><span class="dot ok pulse"></span><span class="lbl">SOFTWARE SIM</span><span class="val" style="color:var(--ok)">ACTIVE</span></div>`);

    const resPanel = (id, n) => panel(id, 'res-panel', `
      <div class="hp-title">RESERVOIR 0${n}</div>
      <div class="hp-row"><span class="lbl">WATER LEVEL</span><span class="val" data-k="level">--</span></div>
      <div class="bar"><div class="fill" data-k="barL"></div></div>
      <div class="hp-row"><span class="lbl">STORAGE</span><span class="val" data-k="storage">--</span></div>
      <div class="bar"><div class="fill" data-k="barS"></div></div>
      <div class="hp-row"><span class="lbl">INFLOW</span><span class="val" data-k="inflow">--</span></div>
      <div class="hp-row"><span class="lbl">RELEASE</span><span class="val" data-k="release">--</span></div>
      <div class="hp-row"><span class="lbl">GATE OPEN</span><span class="val" data-k="gate">--</span></div>
      <div class="hp-row"><span class="lbl">RISK</span><span class="chip normal" data-k="risk">NORMAL</span></div>`);
    const rPanels = [resPanel('hud-r1', 1), resPanel('hud-r2', 2), resPanel('hud-r3', 3)];

    /* SVG connector lines: panel → actual reservoir position in 3D */
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.id = 'hud-connectors';
    this.connLines = []; this.connDots = [];
    for (let i = 0; i < 3; i++) {
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      const dot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      dot.setAttribute('r', '3.5');
      svg.appendChild(line); svg.appendChild(dot);
      this.connLines.push(line); this.connDots.push(dot);
    }

    const hint = document.createElement('div');
    hint.id = 'hud-hint';
    hint.textContent = 'LMB ORBIT · WHEEL ZOOM · RMB PAN';

    const bar = document.createElement('div');
    bar.id = 'hud-controls';
    bar.innerHTML = \`
      <div class="ctl-group"><span class="ctl-cap">CAMERA</span>
        <button class="cbtn on" data-cam="overview">OVERVIEW</button>
        <button class="cbtn" data-cam="reservoir1">R1</button>
        <button class="cbtn" data-cam="reservoir2">R2</button>
        <button class="cbtn" data-cam="reservoir3">R3</button>
        <button class="cbtn" data-cam="downstream">DOWNSTREAM</button>
      </div>
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
      </div>\`;

    hud.appendChild(svg);
    hud.appendChild(hint);
    this.container.appendChild(hud);
    this.container.appendChild(bar);
    this.hudRoot = hud;
    this.barRoot = bar;

    /* Cache every reference ONCE, from elements we just created. */
    const q = (root, sel) => root.querySelector(sel);
    this.el = {
      mode: q(hud, '[data-ref="mode"]'),
      liveTime: q(hud, '#live-time'), liveWx: q(hud, '#live-wx'),
      wint: q(hud, '[data-ref="wint"]'), wbar: q(hud, '[data-ref="wbar"]'),
      wlevel: q(hud, '[data-ref="wlevel"]'),
      dsval: q(hud, '[data-ref="dsval"]'), dsbar: q(hud, '[data-ref="dsbar"]'),
      dsstat: q(hud, '[data-ref="dsstat"]'),
      tn: [q(hud, '[data-ref="tn1"]'), q(hud, '[data-ref="tn2"]'), q(hud, '[data-ref="tn3"]'), q(hud, '[data-ref="tnds"]')],
      hw: [q(hud, '[data-ref="hw1"]'), q(hud, '[data-ref="hw2"]'), q(hud, '[data-ref="hw3"]'), q(hud, '[data-ref="hw4"]')],
      res: rPanels.map(p => {
        const g = k => q(p, \`[data-k="\${k}"]\`);
        return { panel: p, level: g('level'), storage: g('storage'), inflow: g('inflow'),
                 release: g('release'), gate: g('gate'), risk: g('risk'),
                 barL: g('barL'), barS: g('barS') };
      }),
      camBtns: Array.from(bar.querySelectorAll('[data-cam]')),
      sliders: {
        storm: q(bar, '[data-ctl="storm"]'),
        g: [q(bar, '[data-ctl="g1"]'), q(bar, '[data-ctl="g2"]'), q(bar, '[data-ctl="g3"]')],
        auto: q(bar, '[data-ctl="auto"]')
      },
      sliderVals: {
        storm: q(bar, '[data-v="storm"]'),
        g: [q(bar, '[data-v="g1"]'), q(bar, '[data-v="g2"]'), q(bar, '[data-v="g3"]')]
      }
    };
    /* Wire camera preset buttons */
    this.el.camBtns.forEach(btn => btn.addEventListener('click', () => {
      this.el.camBtns.forEach(b => b.classList.remove('on'));
      btn.classList.add('on');
      this.setCameraPreset(btn.dataset.cam);
    }));
    this._cachePanelAnchors();
  }

  _cachePanelAnchors() {
    if (!this.el) return;
    this.panelAnchors = this.el.res.map((re, i) => {
      const rect = re.panel.getBoundingClientRect();
      return { x: i === 0 ? rect.right : rect.left, y: rect.top + rect.height / 2 };
    });
  }

  _updateConnectors() {
    if (!this.connLines || !this.panelAnchors) return;
    const w = this.container.clientWidth, h = this.container.clientHeight;
    for (let i = 0; i < 3; i++) {
      const r = this.R[i];
      this._v3.set(valleyCenterX(r.B.cz), (r.waterY || r.minY) + 5, r.B.cz).project(this.camera);
      const line = this.connLines[i], dot = this.connDots[i];
      if (this._v3.z > 1 || this._v3.z < -1) { line.style.display = 'none'; dot.style.display = 'none'; continue; }
      line.style.display = ''; dot.style.display = '';
      const sx = (this._v3.x * 0.5 + 0.5) * w, sy = (-this._v3.y * 0.5 + 0.5) * h;
      const a = this.panelAnchors[i];
      line.setAttribute('x1', a.x); line.setAttribute('y1', a.y);
      line.setAttribute('x2', sx); line.setAttribute('y2', sy);
      dot.setAttribute('cx', sx); dot.setAttribute('cy', sy);
    }
  }

  _updateHUD(dt) {
    this._hudT += dt;
    if (this._hudT < 0.2) return;               /* DOM text at 5 Hz — never per-frame */
    this._hudT = 0;
    const el = this.el;
    if (!el) return;

    for (let i = 0; i < 3; i++) {
      const d = this.R[i].data, e = el.res[i];
      e.level.textContent = (d.level * 100).toFixed(1) + '%';
      e.storage.textContent = (d.storage * 100).toFixed(1) + '%';
      e.inflow.textContent = d.inflow.toFixed(1) + ' m³/s';
      e.release.textContent = d.release.toFixed(1) + ' m³/s';
      e.gate.textContent = (d.gate * 100).toFixed(0) + '%';
      e.barL.style.width = (d.level * 100) + '%';
      e.barS.style.width = (d.storage * 100) + '%';
      e.risk.textContent = d.risk.toUpperCase();
      e.risk.className = 'chip ' + d.risk;
    }
    const s = this.storm.cur;
    el.wint.textContent = (s * 100).toFixed(0) + '%';
    el.wbar.style.width = (s * 100) + '%';
    const lvl = s < 0.25 ? 'LIGHT' : s < 0.55 ? 'NORMAL' : s < 0.85 ? 'HEAVY' : 'SEVERE';
    el.wlevel.textContent = lvl;
    el.wlevel.style.color = s < 0.55 ? 'var(--ok)' : 'var(--warn)';
    for (let i = 0; i < this.fcBars.length; i++) {
      const t = clamp01(s + 0.14 * Math.sin(this._time * 0.2 + i * 1.9) - 0.04);
      this.fcBars[i].style.height = (3 + t * 23) + 'px';
      this.fcBars[i].style.opacity = 0.4 + t * 0.6;
    }
    const ds = this.data.downstream;
    el.dsval.textContent = ds.toFixed(1) + ' m³/s';
    el.dsbar.style.width = (clamp01(ds / 150) * 100) + '%';
    const st = ds < 120 ? 'NORMAL' : ds < 150 ? 'WARNING' : 'CRITICAL';
    el.dsstat.textContent = st;
    el.dsstat.style.color = ds < 120 ? 'var(--ok)' : ds < 150 ? 'var(--warn)' : 'var(--bad)';
    for (let i = 0; i < 3; i++) {
      el.tn[i].className = 'tnode' + (this.R[i].data.risk === 'normal' ? '' : ' ' + this.R[i].data.risk);
    }
    el.tn[3].className = 'tnode' + (st === 'NORMAL' ? '' : ' ' + (st === 'WARNING' ? 'warn' : 'bad'));
    el.mode.textContent = this.data.mode === 'MANUAL' ? 'MANUAL' : 'AI / MPC';
    const t = new Date(this.data.simTime);
    el.liveTime.textContent = isFinite(t.getTime())
      ? t.toLocaleTimeString('en-GB') + ' · ' + t.toLocaleDateString('en-GB')
      : String(this.data.simTime);
    el.liveWx.textContent = 'RAIN — ' + lvl + ' · 24°C';
    const hwKeys = ['esp32', 'water_level_sensor', 'flow_sensor', 'gate_actuator'];
    for (let i = 0; i < 4; i++) {
      el.hw[i].textContent = String(this.data.hardware[hwKeys[i]] || 'NOT_CONNECTED')
        .toUpperCase().replace(/_/g, ' ');
    }
  }
}

/* ================= bootstrap — one init per page load ================= */
/* TestDriver REMOVED — Python simulation state is the sole input via postMessage. */
const root = document.getElementById('twin-root');
let app = null;
try {
  app = new ReservoirTwinRenderer(root, { testControls: true });
} catch (e) {
  console.error("Renderer initialization failed:", e);
  /* boot overlay already shows the failure reason; nothing further to do */
}
if (app) {
  window.__twin = app;                       /* console: __twin.updateState({...}), __twin.dispose() */
  window.ReservoirTwinRenderer = ReservoirTwinRenderer;
}

