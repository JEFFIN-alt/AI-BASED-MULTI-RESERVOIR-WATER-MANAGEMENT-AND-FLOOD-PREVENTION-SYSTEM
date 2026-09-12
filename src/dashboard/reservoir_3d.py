"""
reservoir_3d.py — Three.js 3D Reservoir Digital-Twin Visualization

Generates a self-contained HTML/WebGL scene driven by real reservoir
telemetry. Embedded in Streamlit via st.components.v1.html().
"""

import json


def generate_3d_html(data: dict) -> str:
    """
    Generate a complete Three.js 3D reservoir visualization.

    Parameters
    ----------
    data : dict
        Must contain: name, water_level, frl, blue_level, orange_level,
        red_level, live_storage, storage_pct, inflow, outflow,
        forecast_1d/3d/7d, risk_status, source, forecast_date, telemetry_date
    """
    data_json = json.dumps(data, default=str)
    return _HTML_TEMPLATE.replace("__RESERVOIR_DATA__", data_json)


# ======================================================================
# COMPLETE HTML / THREE.JS TEMPLATE
# ======================================================================

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<style>
*{margin:0;padding:0;box-sizing:border-box}
html,body{width:100%;height:100%;overflow:hidden;background:#0a0e17;font-family:'Segoe UI',system-ui,sans-serif;color:#e0e6f0}
#viewport{width:100%;height:100%;display:block}
/* ---- HUD ---- */
.hud{position:absolute;pointer-events:none}
.hud-interact{pointer-events:auto}
#hud-top-left{top:16px;left:16px}
#hud-top-right{top:16px;right:16px;text-align:right}
#hud-bottom{bottom:0;left:0;right:0;padding:12px 20px;background:linear-gradient(to top,rgba(6,10,22,.92),rgba(6,10,22,.0));display:flex;gap:28px;align-items:flex-end;flex-wrap:wrap}
.res-name{font-size:22px;font-weight:700;letter-spacing:1px;text-shadow:0 2px 12px rgba(0,0,0,.7)}
.badge{display:inline-block;padding:3px 10px;border-radius:4px;font-size:11px;font-weight:600;letter-spacing:.5px;margin-top:4px}
.badge-live{background:#28a745;color:#fff}
.badge-test{background:#17a2b8;color:#fff}
.badge-unavail{background:#6c757d;color:#fff}
.badge-normal{background:rgba(40,167,69,.85);color:#fff}
.badge-watch{background:rgba(255,193,7,.9);color:#111}
.badge-alert{background:rgba(253,126,20,.9);color:#fff}
.badge-highrisk{background:rgba(220,53,69,.9);color:#fff;animation:pulse-risk 2s infinite}
@keyframes pulse-risk{0%,100%{box-shadow:0 0 6px rgba(220,53,69,.4)}50%{box-shadow:0 0 18px rgba(220,53,69,.8)}}
.data-cell{min-width:110px}
.data-cell .label{font-size:10px;color:#8899aa;text-transform:uppercase;letter-spacing:.8px}
.data-cell .value{font-size:16px;font-weight:600;color:#e0e6f0}
.data-cell .unit{font-size:11px;color:#6b7f94}
#cam-controls{position:absolute;bottom:80px;right:16px;display:flex;flex-direction:column;gap:6px;pointer-events:auto}
#cam-controls button{background:rgba(20,28,45,.8);border:1px solid rgba(100,140,200,.25);color:#a0b4cc;padding:6px 12px;border-radius:4px;cursor:pointer;font-size:11px;backdrop-filter:blur(6px)}
#cam-controls button:hover{background:rgba(40,60,100,.7);color:#fff}
#loading-overlay{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;background:#0a0e17;z-index:100;transition:opacity .8s}
#loading-overlay.fade{opacity:0;pointer-events:none}
.loader{width:48px;height:48px;border:3px solid rgba(100,160,255,.2);border-top-color:#4a9eff;border-radius:50%;animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
</style>
</head>
<body>
<canvas id="viewport"></canvas>

<div id="loading-overlay"><div class="loader"></div></div>

<!-- HUD: Top Left -->
<div class="hud" id="hud-top-left">
  <div class="res-name" id="h-name"></div>
  <div id="h-source-badge"></div>
  <div style="margin-top:4px" id="h-tele-date"></div>
</div>

<!-- HUD: Top Right -->
<div class="hud" id="hud-top-right">
  <div id="h-risk-badge"></div>
  <div style="margin-top:6px;font-size:11px;color:#8899aa" id="h-risk-reason"></div>
</div>

<!-- HUD: Bottom Data Bar -->
<div class="hud" id="hud-bottom">
  <div class="data-cell"><div class="label">Water Level</div><div class="value" id="h-wl">—</div></div>
  <div class="data-cell"><div class="label">Storage</div><div class="value" id="h-stor">—</div></div>
  <div class="data-cell"><div class="label">Inflow</div><div class="value" id="h-inflow">—</div></div>
  <div class="data-cell"><div class="label">Outflow</div><div class="value" id="h-outflow">—</div></div>
  <div style="width:1px;height:36px;background:rgba(100,140,200,.2)"></div>
  <div class="data-cell"><div class="label">Forecast 1D</div><div class="value" id="h-fc1">—</div></div>
  <div class="data-cell"><div class="label">Forecast 3D</div><div class="value" id="h-fc3">—</div></div>
  <div class="data-cell"><div class="label">Forecast 7D</div><div class="value" id="h-fc7">—</div></div>
  <div style="width:1px;height:36px;background:rgba(100,140,200,.2)"></div>
  <div class="data-cell"><div class="label">FRL</div><div class="value" id="h-frl">—</div></div>
  <div class="data-cell"><div class="label">Blue / Orange / Red</div><div class="value" id="h-thresholds">—</div></div>
</div>

<!-- Camera Controls -->
<div id="cam-controls">
  <button id="btn-reset">⟲ Reset</button>
  <button id="btn-top">⬆ Top</button>
  <button id="btn-dam">◉ Dam</button>
</div>

<script type="importmap">
{"imports":{"three":"https://unpkg.com/three@0.164.1/build/three.module.js","three/addons/":"https://unpkg.com/three@0.164.1/examples/jsm/"}}
</script>
<script type="module">
import*as THREE from'three';
import{OrbitControls}from'three/addons/controls/OrbitControls.js';

// ============================================================
// DATA
// ============================================================
const D = __RESERVOIR_DATA__;

// Derived
const waterLevel = D.water_level || 0;
const frl = D.frl || waterLevel * 1.1 || 100;
const storagePct = D.storage_pct != null ? D.storage_pct : (frl > 0 ? Math.min(100, Math.max(0, (waterLevel / frl) * 100)) : 50);
const inflow = D.inflow || 0;
const outflow = D.outflow || 0;
const riskStatus = (D.risk_status || 'NORMAL').toUpperCase();

// Map storage % to scene water Y   (basin floor = -8, FRL = 6)
const BASIN_FLOOR = -8;
const FRL_Y = 6;
const waterY = BASIN_FLOOR + (storagePct / 100) * (FRL_Y - BASIN_FLOOR);

// ============================================================
// NOISE
// ============================================================
const _P = new Uint8Array(512);
for(let i=0;i<256;i++){let v=i;v=((v>>1)^(-(v&1)&0xEDB88320))>>>0;v=((v>>1)^(-(v&1)&0xEDB88320))>>>0;_P[i]=_P[i+256]=v&255}
function _grad(h,x,y){const g=h&3;return(g===0?x+y:g===1?-x+y:g===2?x-y:-x-y)}
function noise2D(x,y){const X=Math.floor(x)&255,Y=Math.floor(y)&255;x-=Math.floor(x);y-=Math.floor(y);const u=x*x*(3-2*x),v=y*y*(3-2*y);const a=_P[X]+Y,b=_P[X+1]+Y;return .5+.5*((1-v)*((1-u)*_grad(_P[a],x,y)+u*_grad(_P[b],x-1,y))+v*((1-u)*_grad(_P[a+1],x,y-1)+u*_grad(_P[b+1],x-1,y-1)))}
function fbm(x,y,oct=6){let v=0,a=.5,f=1;for(let i=0;i<oct;i++){v+=a*noise2D(x*f,y*f);a*=.5;f*=2}return v}

// ============================================================
// RENDERER / SCENE / CAMERA
// ============================================================
const canvas = document.getElementById('viewport');
const renderer = new THREE.WebGLRenderer({canvas, antialias:true, alpha:false});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.1;

const scene = new THREE.Scene();

// Risk-dependent fog
const fogColors = {NORMAL:0x8faec8, WATCH:0xa09870, ALERT:0x8a7060, 'HIGH RISK':0x705050};
const fogColor = new THREE.Color(fogColors[riskStatus] || 0x8faec8);
scene.fog = new THREE.FogExp2(fogColor, 0.012);
scene.background = fogColor;

const camera = new THREE.PerspectiveCamera(50, window.innerWidth/window.innerHeight, 0.5, 500);
camera.position.set(38, 28, 42);
camera.lookAt(0, 0, 0);

const clock = new THREE.Clock();

// ============================================================
// LIGHTING
// ============================================================
const hemi = new THREE.HemisphereLight(0x87aed6, 0x3a5c30, 0.6);
scene.add(hemi);

const sun = new THREE.DirectionalLight(0xffeedd, 1.6);
sun.position.set(30, 45, 20);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.camera.left = -60; sun.shadow.camera.right = 60;
sun.shadow.camera.top = 60; sun.shadow.camera.bottom = -60;
sun.shadow.camera.near = 1; sun.shadow.camera.far = 150;
sun.shadow.bias = -0.001;
scene.add(sun);

const ambient = new THREE.AmbientLight(0x405570, 0.35);
scene.add(ambient);

// ============================================================
// TERRAIN
// ============================================================
function getHeight(x, z) {
    // Base noise terrain
    let h = fbm(x * 0.025 + 3.7, z * 0.025 + 1.2, 6) * 22 - 4;
    // Mountain ridges on sides
    const ridge = Math.max(0, Math.abs(x) - 18) * 0.7;
    h += ridge;
    // Valley deepening toward center
    const valleyHalfW = 16 + z * 0.08;
    const cx = Math.abs(x) / Math.max(valleyHalfW, 1);
    if (cx < 1) h -= (1 - cx * cx) * 10;
    // Reservoir basin bowl
    const bx = x / 28, bz = (z + 4) / 38;
    const bd = bx * bx + bz * bz;
    if (bd < 1) h -= (1 - bd) * 12;
    // Downstream channel (through dam)
    if (z < -26) {
        const ch = Math.exp(-x * x / 18);
        h -= ch * 6 * Math.min(1, (-26 - z) / 10);
    }
    return h;
}

const TSEG = 160;
const TSIZE = 120;
const terrainGeo = new THREE.PlaneGeometry(TSIZE, TSIZE, TSEG, TSEG);
terrainGeo.rotateX(-Math.PI / 2);
const tPos = terrainGeo.attributes.position;
const tCol = new Float32Array(tPos.count * 3);
for (let i = 0; i < tPos.count; i++) {
    const x = tPos.getX(i), z = tPos.getZ(i);
    const h = getHeight(x, z);
    tPos.setY(i, h);
    // Vertex color based on height
    let r, g, b;
    if (h > 18) { r = .88; g = .90; b = .92; }          // snow
    else if (h > 12) { r = .45; g = .42; b = .38; }      // rock
    else if (h > 3) {                                      // grass-forest
        const t = fbm(x * 0.08, z * 0.08, 3);
        r = .12 + t * .08; g = .28 + t * .12; b = .10 + t * .04;
    } else if (h > -2) {                                   // shore / mud
        r = .32; g = .28; b = .20;
    } else {                                               // basin floor
        r = .18; g = .16; b = .13;
    }
    tCol[i * 3] = r; tCol[i * 3 + 1] = g; tCol[i * 3 + 2] = b;
}
terrainGeo.setAttribute('color', new THREE.BufferAttribute(tCol, 3));
terrainGeo.computeVertexNormals();

const terrainMat = new THREE.MeshStandardMaterial({
    vertexColors: true, roughness: 0.88, metalness: 0.02, flatShading: false
});
const terrain = new THREE.Mesh(terrainGeo, terrainMat);
terrain.receiveShadow = true;
terrain.castShadow = true;
scene.add(terrain);

// ============================================================
// TREES  (instanced cones on terrain above water)
// ============================================================
const treeTrunkGeo = new THREE.CylinderGeometry(0.1, 0.15, 1.0, 5);
const treeTopGeo = new THREE.ConeGeometry(0.6, 2.0, 6);
const treeTopMat = new THREE.MeshStandardMaterial({color: 0x2d5a1e, roughness: 0.9});
const treeTrunkMat = new THREE.MeshStandardMaterial({color: 0x4a3520, roughness: 0.95});
const treeGroup = new THREE.Group();
const treeCount = 300;
for (let i = 0; i < treeCount; i++) {
    const tx = (Math.random() - 0.5) * TSIZE * 0.85;
    const tz = (Math.random() - 0.5) * TSIZE * 0.85;
    const th = getHeight(tx, tz);
    if (th < waterY + 0.5 || th > 16) continue;
    const scale = 0.6 + Math.random() * 0.8;
    const trunk = new THREE.Mesh(treeTrunkGeo, treeTrunkMat);
    trunk.position.set(tx, th + scale * 0.5, tz);
    trunk.scale.set(scale, scale, scale);
    const top = new THREE.Mesh(treeTopGeo, treeTopMat);
    top.position.set(tx, th + scale * 1.6, tz);
    top.scale.set(scale, scale, scale);
    trunk.castShadow = true;
    top.castShadow = true;
    treeGroup.add(trunk, top);
}
scene.add(treeGroup);

// ============================================================
// DAM
// ============================================================
const damW = 34, damH = 16, damD = 3;
const damGeo = new THREE.BoxGeometry(damW, damH, damD, 8, 4, 1);
const damMat = new THREE.MeshStandardMaterial({color: 0x8a8680, roughness: 0.75, metalness: 0.05});
const dam = new THREE.Mesh(damGeo, damMat);
dam.position.set(0, getHeight(0, -28) + damH / 2 - 2, -28);
dam.castShadow = true;
dam.receiveShadow = true;
scene.add(dam);

// Dam walkway
const walkGeo = new THREE.BoxGeometry(damW + 2, 0.3, 4);
const walkMat = new THREE.MeshStandardMaterial({color: 0x606058, roughness: 0.8});
const walkway = new THREE.Mesh(walkGeo, walkMat);
walkway.position.set(0, dam.position.y + damH / 2 + 0.15, -28);
walkway.castShadow = true;
scene.add(walkway);

// Spillway slot
const spillSlotGeo = new THREE.BoxGeometry(4, 5, damD + 1);
const spillSlotMat = new THREE.MeshStandardMaterial({color: 0x1a1a1a, roughness: 0.5});
const spillSlot = new THREE.Mesh(spillSlotGeo, spillSlotMat);
spillSlot.position.set(0, dam.position.y + 1, -28);
scene.add(spillSlot);

// ============================================================
// WATER — Custom Shader
// ============================================================
const waterVert = `
uniform float uTime;
uniform float uWaveStr;
varying vec3 vWorldPos;
varying vec3 vNormal;
varying vec2 vUv;

void main(){
    vec3 pos = position;
    float t = uTime;
    float ws = uWaveStr;
    // Multi-layer Gerstner-style waves
    float w = 0.0;
    w += sin(pos.x*0.4 + t*0.7)*0.18*ws;
    w += sin(pos.y*0.3 + t*0.55)*0.14*ws;
    w += sin((pos.x+pos.y)*0.6 + t*1.1)*0.09*ws;
    w += sin(pos.x*1.3 - t*0.85)*0.06*ws;
    w += sin(pos.y*1.7 + t*1.4)*0.04*ws;
    w += sin((pos.x*0.9 - pos.y*0.7)*1.1 + t*0.95)*0.05*ws;
    pos.z += w;

    // Analytical normal from partial derivatives
    float dx = cos(pos.x*0.4+t*0.7)*0.4*0.18*ws
             + cos((pos.x+pos.y)*0.6+t*1.1)*0.6*0.09*ws
             + cos(pos.x*1.3-t*0.85)*1.3*0.06*ws
             + cos((pos.x*0.9-pos.y*0.7)*1.1+t*0.95)*0.9*1.1*0.05*ws;
    float dy = cos(pos.y*0.3+t*0.55)*0.3*0.14*ws
             + cos((pos.x+pos.y)*0.6+t*1.1)*0.6*0.09*ws
             + cos(pos.y*1.7+t*1.4)*1.7*0.04*ws
             - cos((pos.x*0.9-pos.y*0.7)*1.1+t*0.95)*0.7*1.1*0.05*ws;
    vNormal = normalize(vec3(-dx, -dy, 1.0));

    vWorldPos = (modelMatrix * vec4(pos, 1.0)).xyz;
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
}`;

const waterFrag = `
uniform float uTime;
uniform vec3 uSunDir;
uniform vec3 uDeepColor;
uniform vec3 uShallowColor;
uniform vec3 uFogColor;
uniform float uFogDensity;
varying vec3 vWorldPos;
varying vec3 vNormal;
varying vec2 vUv;

void main(){
    vec3 viewDir = normalize(cameraPosition - vWorldPos);
    vec3 N = normalize(vNormal);

    // Fresnel (Schlick approx)
    float fresnel = pow(1.0 - max(dot(N, viewDir), 0.0), 4.0);
    fresnel = 0.04 + 0.96 * fresnel;

    // Water body color
    float depthFactor = 0.45 + 0.55 * sin(vUv.x * 3.14);
    vec3 bodyColor = mix(uShallowColor, uDeepColor, depthFactor);

    // Reflection (fake sky)
    vec3 R = reflect(-viewDir, N);
    float skyGrad = smoothstep(-0.1, 0.6, R.y);
    vec3 skyColor = mix(vec3(0.35, 0.42, 0.52), vec3(0.55, 0.70, 0.90), skyGrad);
    vec3 reflected = skyColor;

    // Specular sun
    vec3 halfDir = normalize(uSunDir + viewDir);
    float spec = pow(max(dot(N, halfDir), 0.0), 512.0);
    vec3 specular = vec3(1.0, 0.95, 0.85) * spec * 2.5;

    // Secondary specular (broader)
    float spec2 = pow(max(dot(N, halfDir), 0.0), 32.0);
    specular += vec3(0.8, 0.85, 0.9) * spec2 * 0.15;

    // Small ripple detail via procedural noise-ish pattern
    float ripple = sin(vWorldPos.x * 8.0 + uTime * 2.0) * sin(vWorldPos.z * 6.0 + uTime * 1.5) * 0.02;

    // Combine
    vec3 color = mix(bodyColor, reflected, fresnel) + specular;
    color += ripple;

    // Fog
    float dist = length(vWorldPos - cameraPosition);
    float fogFactor = 1.0 - exp(-uFogDensity * dist * dist);
    color = mix(color, uFogColor, fogFactor);

    gl_FragColor = vec4(color, 0.88);
}`;

// Risk-dependent wave strength
const waveStrengths = {NORMAL: 1.0, WATCH: 1.3, ALERT: 1.6, 'HIGH RISK': 2.2};
const waveStr = waveStrengths[riskStatus] || 1.0;

// Risk-dependent deep water color
const deepColors = {
    NORMAL: new THREE.Color(0.01, 0.06, 0.16),
    WATCH: new THREE.Color(0.04, 0.06, 0.14),
    ALERT: new THREE.Color(0.06, 0.04, 0.10),
    'HIGH RISK': new THREE.Color(0.08, 0.03, 0.08)
};

const waterUniforms = {
    uTime: {value: 0},
    uWaveStr: {value: waveStr},
    uSunDir: {value: new THREE.Vector3(0.5, 0.7, 0.3).normalize()},
    uDeepColor: {value: deepColors[riskStatus] || deepColors.NORMAL},
    uShallowColor: {value: new THREE.Color(0.04, 0.22, 0.38)},
    uFogColor: {value: fogColor},
    uFogDensity: {value: 0.00014}
};

const waterGeo = new THREE.PlaneGeometry(56, 52, 128, 128);
const waterMat = new THREE.ShaderMaterial({
    vertexShader: waterVert, fragmentShader: waterFrag,
    uniforms: waterUniforms, transparent: true, side: THREE.DoubleSide,
    depthWrite: false
});
const water = new THREE.Mesh(waterGeo, waterMat);
water.rotation.x = -Math.PI / 2;
water.position.set(0, waterY, -2);
scene.add(water);

// ============================================================
// INFLOW PARTICLES  (upstream → reservoir)
// ============================================================
const INFLOW_COUNT = 200;
const inflowGeo = new THREE.BufferGeometry();
const inflowPositions = new Float32Array(INFLOW_COUNT * 3);
const inflowSpeeds = new Float32Array(INFLOW_COUNT);
for (let i = 0; i < INFLOW_COUNT; i++) {
    inflowPositions[i*3]   = (Math.random()-.5) * 6 + 18;        // x: start upstream right
    inflowPositions[i*3+1] = waterY + Math.random() * 0.4;        // y
    inflowPositions[i*3+2] = (Math.random()-.5) * 8 + 20;         // z: upstream
    inflowSpeeds[i] = 0.5 + Math.random() * 1.5;
}
inflowGeo.setAttribute('position', new THREE.BufferAttribute(inflowPositions, 3));
const inflowMat = new THREE.PointsMaterial({
    color: 0x5599dd, size: 0.25, transparent: true, opacity: 0.7,
    blending: THREE.AdditiveBlending, depthWrite: false
});
const inflowPts = new THREE.Points(inflowGeo, inflowMat);
scene.add(inflowPts);

// Scale inflow particle speed by actual inflow value
const inflowIntensity = Math.min(3, Math.max(0.2, inflow / 15));

function updateInflow(dt) {
    const pos = inflowPts.geometry.attributes.position.array;
    for (let i = 0; i < INFLOW_COUNT; i++) {
        pos[i*3]   -= dt * inflowSpeeds[i] * inflowIntensity * 3;
        pos[i*3+2] -= dt * inflowSpeeds[i] * inflowIntensity * 5;
        // Reset when reaches reservoir center
        if (pos[i*3+2] < -5 || pos[i*3] < -5) {
            pos[i*3]   = (Math.random()-.5)*6 + 18;
            pos[i*3+1] = waterY + Math.random()*0.4;
            pos[i*3+2] = (Math.random()-.5)*8 + 20;
        }
    }
    inflowPts.geometry.attributes.position.needsUpdate = true;
}

// ============================================================
// OUTFLOW / SPILLWAY PARTICLES  (dam → downstream)
// ============================================================
const OUTFLOW_COUNT = 120;
const outflowGeo = new THREE.BufferGeometry();
const outflowPositions = new Float32Array(OUTFLOW_COUNT * 3);
const outflowSpeeds = new Float32Array(OUTFLOW_COUNT);
for (let i = 0; i < OUTFLOW_COUNT; i++) {
    outflowPositions[i*3]   = (Math.random()-.5) * 3;
    outflowPositions[i*3+1] = waterY - Math.random() * 3;
    outflowPositions[i*3+2] = -29 - Math.random() * 8;
    outflowSpeeds[i] = 0.4 + Math.random() * 1.2;
}
outflowGeo.setAttribute('position', new THREE.BufferAttribute(outflowPositions, 3));
const outflowMat = new THREE.PointsMaterial({
    color: 0x88bbee, size: 0.2, transparent: true, opacity: 0.6,
    blending: THREE.AdditiveBlending, depthWrite: false
});
const outflowPts = new THREE.Points(outflowGeo, outflowMat);
scene.add(outflowPts);

const outflowIntensity = Math.min(3, Math.max(0.1, outflow / 10));

function updateOutflow(dt) {
    const pos = outflowPts.geometry.attributes.position.array;
    for (let i = 0; i < OUTFLOW_COUNT; i++) {
        pos[i*3+1] -= dt * outflowSpeeds[i] * outflowIntensity * 2;
        pos[i*3+2] -= dt * outflowSpeeds[i] * outflowIntensity * 4;
        if (pos[i*3+2] < -50 || pos[i*3+1] < -15) {
            pos[i*3]   = (Math.random()-.5)*3;
            pos[i*3+1] = waterY - Math.random()*1;
            pos[i*3+2] = -29 - Math.random()*2;
        }
    }
    outflowPts.geometry.attributes.position.needsUpdate = true;
}

// ============================================================
// MIST PARTICLES (near spillway base)
// ============================================================
const MIST_COUNT = 80;
const mistGeo = new THREE.BufferGeometry();
const mistPos = new Float32Array(MIST_COUNT * 3);
for (let i = 0; i < MIST_COUNT; i++) {
    mistPos[i*3]   = (Math.random()-.5) * 8;
    mistPos[i*3+1] = dam.position.y - 4 + Math.random() * 6;
    mistPos[i*3+2] = -30 - Math.random() * 6;
}
mistGeo.setAttribute('position', new THREE.BufferAttribute(mistPos, 3));
const mistMat = new THREE.PointsMaterial({
    color: 0xc8d8e8, size: 1.2, transparent: true, opacity: 0.15 * outflowIntensity,
    blending: THREE.AdditiveBlending, depthWrite: false
});
const mistPts = new THREE.Points(mistGeo, mistMat);
scene.add(mistPts);

function updateMist(dt, elapsed) {
    const pos = mistPts.geometry.attributes.position.array;
    for (let i = 0; i < MIST_COUNT; i++) {
        pos[i*3+1] += dt * (0.3 + Math.sin(elapsed + i) * 0.2);
        pos[i*3]   += dt * Math.sin(elapsed * 0.7 + i * 0.3) * 0.4;
        if (pos[i*3+1] > dam.position.y + 4) {
            pos[i*3]   = (Math.random()-.5)*8;
            pos[i*3+1] = dam.position.y - 4;
            pos[i*3+2] = -30 - Math.random()*6;
        }
    }
    mistPts.geometry.attributes.position.needsUpdate = true;
}

// ============================================================
// AI DATA PULSE (forecast signal orbiting reservoir)
// ============================================================
const PULSE_COUNT = 40;
const pulseGeo = new THREE.BufferGeometry();
const pulsePos = new Float32Array(PULSE_COUNT * 3);
pulseGeo.setAttribute('position', new THREE.BufferAttribute(pulsePos, 3));
const pulseMat = new THREE.PointsMaterial({
    color: riskStatus === 'HIGH RISK' ? 0xff4466 : riskStatus === 'ALERT' ? 0xffaa33 : 0x44aaff,
    size: 0.35, transparent: true, opacity: 0.6,
    blending: THREE.AdditiveBlending, depthWrite: false
});
const pulsePts = new THREE.Points(pulseGeo, pulseMat);
scene.add(pulsePts);

function updatePulse(elapsed) {
    const pos = pulsePts.geometry.attributes.position.array;
    for (let i = 0; i < PULSE_COUNT; i++) {
        const angle = (i / PULSE_COUNT) * Math.PI * 2 + elapsed * 0.3;
        const r = 22 + Math.sin(elapsed * 0.5 + i) * 3;
        pos[i*3]   = Math.cos(angle) * r;
        pos[i*3+1] = waterY + 1.5 + Math.sin(elapsed * 1.2 + i * 0.5) * 0.8;
        pos[i*3+2] = Math.sin(angle) * r * 0.7 - 2;
    }
    pulsePts.geometry.attributes.position.needsUpdate = true;
}

// ============================================================
// RAIN PARTICLES (storm mode — active when inflow > threshold)
// ============================================================
const isStorm = inflow > 20 || riskStatus === 'ALERT' || riskStatus === 'HIGH RISK';
const RAIN_COUNT = isStorm ? 600 : 0;
let rainPts = null;
if (RAIN_COUNT > 0) {
    const rainGeo = new THREE.BufferGeometry();
    const rainPos = new Float32Array(RAIN_COUNT * 3);
    for (let i = 0; i < RAIN_COUNT; i++) {
        rainPos[i*3]   = (Math.random()-.5)*100;
        rainPos[i*3+1] = 20 + Math.random()*30;
        rainPos[i*3+2] = (Math.random()-.5)*100;
    }
    rainGeo.setAttribute('position', new THREE.BufferAttribute(rainPos, 3));
    const rainMat = new THREE.PointsMaterial({
        color: 0x99aabb, size: 0.08, transparent: true, opacity: 0.4
    });
    rainPts = new THREE.Points(rainGeo, rainMat);
    scene.add(rainPts);
}

function updateRain(dt) {
    if (!rainPts) return;
    const pos = rainPts.geometry.attributes.position.array;
    const speed = riskStatus === 'HIGH RISK' ? 40 : 25;
    for (let i = 0; i < RAIN_COUNT; i++) {
        pos[i*3+1] -= dt * speed;
        if (pos[i*3+1] < -5) {
            pos[i*3]   = (Math.random()-.5)*100;
            pos[i*3+1] = 25 + Math.random()*25;
            pos[i*3+2] = (Math.random()-.5)*100;
        }
    }
    rainPts.geometry.attributes.position.needsUpdate = true;
}

// ============================================================
// CAMERA CONTROLS
// ============================================================
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, waterY, -5);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.minDistance = 10;
controls.maxDistance = 120;
controls.maxPolarAngle = Math.PI / 2.05;
controls.update();

const defaultCamPos = camera.position.clone();
const defaultTarget = controls.target.clone();

function animateCamera(targetPos, targetLook, duration = 1.5) {
    const startPos = camera.position.clone();
    const startTarget = controls.target.clone();
    const startTime = clock.getElapsedTime();
    function step() {
        const t = Math.min(1, (clock.getElapsedTime() - startTime) / duration);
        const ease = t < 0.5 ? 4*t*t*t : 1 - Math.pow(-2*t+2,3)/2; // easeInOutCubic
        camera.position.lerpVectors(startPos, targetPos, ease);
        controls.target.lerpVectors(startTarget, targetLook, ease);
        controls.update();
        if (t < 1) requestAnimationFrame(step);
    }
    step();
}

document.getElementById('btn-reset').onclick = () =>
    animateCamera(defaultCamPos, defaultTarget);
document.getElementById('btn-top').onclick = () =>
    animateCamera(new THREE.Vector3(0, 55, 5), new THREE.Vector3(0, 0, -5));
document.getElementById('btn-dam').onclick = () =>
    animateCamera(new THREE.Vector3(5, 8, -20), new THREE.Vector3(0, waterY, -28));

// ============================================================
// HUD UPDATE
// ============================================================
function updateHUD() {
    const el = id => document.getElementById(id);
    el('h-name').textContent = D.name || 'RESERVOIR';

    // Source badge
    const src = D.source || 'UNAVAILABLE';
    const srcClass = src === 'LIVE_INFERENCE' ? 'badge-live' : src === 'VALIDATED_TEST_PREDICTION' ? 'badge-test' : 'badge-unavail';
    const srcLabel = src === 'LIVE_INFERENCE' ? 'LIVE INFERENCE' : src === 'VALIDATED_TEST_PREDICTION' ? 'VALIDATED TEST' : 'UNAVAILABLE';
    el('h-source-badge').innerHTML = `<span class="badge ${srcClass}">${srcLabel}</span>`;

    el('h-tele-date').innerHTML = `<span style="font-size:11px;color:#8899aa">Telemetry: ${D.telemetry_date || '—'}</span>`;

    // Risk badge
    const riskClass = riskStatus === 'NORMAL' ? 'badge-normal' : riskStatus === 'WATCH' ? 'badge-watch' : riskStatus === 'ALERT' ? 'badge-alert' : 'badge-highrisk';
    el('h-risk-badge').innerHTML = `<span class="badge ${riskClass}" style="font-size:14px;padding:5px 14px">${riskStatus}</span>`;
    el('h-risk-reason').textContent = (D.reason || '').substring(0, 120);

    // Bottom data
    const fmt = (v, u) => v != null ? `${typeof v === 'number' ? v.toFixed(2) : v} <span class="unit">${u}</span>` : '—';
    el('h-wl').innerHTML = fmt(D.water_level, 'm');
    el('h-stor').innerHTML = storagePct != null ? `${storagePct.toFixed(1)} <span class="unit">%</span>` : '—';
    el('h-inflow').innerHTML = fmt(D.inflow, 'MCM/d');
    el('h-outflow').innerHTML = fmt(D.outflow, 'MCM/d');
    el('h-fc1').innerHTML = fmt(D.forecast_1d, 'MCM/d');
    el('h-fc3').innerHTML = fmt(D.forecast_3d, 'MCM/d');
    el('h-fc7').innerHTML = fmt(D.forecast_7d, 'MCM/d');
    el('h-frl').innerHTML = fmt(D.frl, 'm');
    const b = D.blue_level, o = D.orange_level, r = D.red_level;
    el('h-thresholds').innerHTML = `${b||'—'} / ${o||'—'} / ${r||'—'} <span class="unit">m</span>`;
}
updateHUD();

// ============================================================
// RESIZE
// ============================================================
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

// ============================================================
// ANIMATION LOOP
// ============================================================
function animate() {
    requestAnimationFrame(animate);
    const dt = Math.min(clock.getDelta(), 0.05);
    const elapsed = clock.getElapsedTime();

    // Water animation
    waterUniforms.uTime.value = elapsed;

    // Particles
    updateInflow(dt);
    updateOutflow(dt);
    updateMist(dt, elapsed);
    updatePulse(elapsed);
    updateRain(dt);

    // Subtle mist opacity pulse
    mistMat.opacity = (0.1 + Math.sin(elapsed * 0.5) * 0.05) * outflowIntensity;

    controls.update();
    renderer.render(scene, camera);
}

// Remove loading overlay
setTimeout(() => document.getElementById('loading-overlay').classList.add('fade'), 600);

animate();
</script>
</body>
</html>"""
