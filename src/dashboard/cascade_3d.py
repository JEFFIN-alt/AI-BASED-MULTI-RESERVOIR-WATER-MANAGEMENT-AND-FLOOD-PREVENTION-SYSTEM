"""
Multi-Reservoir 3D Digital Twin — Premium Cinematic Scene Generator
"""

def generate_cascade_3d_html() -> str:
    return _HTML_TEMPLATE

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
html, body { width: 100%; height: 100%; overflow: hidden; background: #020408; font-family: 'Inter', 'Segoe UI', system-ui, sans-serif; color: #e0e6f0; }
#viewport { width: 100%; height: 100%; display: block; }

/* Futuristic Glass HUD */
.hud-label {
    background: rgba(5, 10, 15, 0.4);
    border-left: 2px solid rgba(100, 140, 200, 0.5);
    padding: 10px 14px;
    border-radius: 0 6px 6px 0;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    pointer-events: none;
    font-size: 11px;
    min-width: 170px;
    transition: border-left-color 0.3s;
    box-shadow: 0 4px 20px rgba(0,0,0,0.5);
}
.hud-label .title { font-size: 12px; font-weight: 700; color: #8ab4f8; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1.5px; border-bottom: 1px solid rgba(138, 180, 248, 0.2); padding-bottom: 4px; }
.hud-label .row { display: flex; justify-content: space-between; margin-bottom: 4px; }
.hud-label .lbl { color: #8899aa; font-weight: 500; }
.hud-label .val { font-weight: 600; text-shadow: 0 0 5px rgba(255,255,255,0.2); }
.hud-label .risk { display: inline-block; margin-top: 6px; padding: 3px 8px; border-radius: 2px; font-weight: 700; font-size: 9px; text-transform: uppercase; letter-spacing: 1px; border: 1px solid transparent; }

/* Line pointing to dam */
.hud-line {
    position: absolute;
    left: -30px; top: 15px;
    width: 30px; height: 1px;
    background: rgba(255,255,255,0.3);
}

/* Global HUD */
#hud-top { position: absolute; top: 20px; left: 24px; pointer-events: none; }
.sys-title { font-size: 18px; font-weight: 800; letter-spacing: 3px; color: #e8f0fe; text-shadow: 0 0 15px rgba(138, 180, 248, 0.5); }
.sys-sub { font-size: 10px; color: #8ab4f8; letter-spacing: 2px; text-transform: uppercase; margin-top: 4px; font-weight: 600; }

/* Downstream Label */
#downstream-hud { 
    position: absolute; bottom: 24px; left: 24px; 
    background: rgba(5, 10, 15, 0.6); backdrop-filter: blur(10px);
    padding: 12px 20px; border-radius: 4px; border-left: 3px solid #8ab4f8; 
    box-shadow: 0 4px 20px rgba(0,0,0,0.5);
}
#downstream-hud .title { font-size: 9px; font-weight: 700; color: #8ab4f8; text-transform: uppercase; letter-spacing: 2px; }
#downstream-hud .val { font-size: 24px; font-weight: 300; margin-top: 4px; color: #fff; }

/* Camera Controls */
#cam-controls { position: absolute; bottom: 24px; right: 24px; display: flex; gap: 6px; pointer-events: auto; }
#cam-controls button { 
    background: rgba(10, 15, 25, 0.6); backdrop-filter: blur(4px);
    border: 1px solid rgba(138, 180, 248, 0.2); color: #8ab4f8; 
    padding: 8px 14px; border-radius: 4px; cursor: pointer; font-size: 10px; 
    font-weight: 600; text-transform: uppercase; letter-spacing: 1px; transition: all 0.2s;
}
#cam-controls button:hover { background: rgba(138, 180, 248, 0.2); color: #fff; border-color: #8ab4f8; box-shadow: 0 0 10px rgba(138,180,248,0.3); }

#loading { position: absolute; inset: 0; background: #020408; display: flex; align-items: center; justify-content: center; z-index: 100; transition: opacity 1s ease-in-out; }
.loader { width: 50px; height: 50px; border: 2px solid rgba(138,180,248,0.1); border-top-color: #8ab4f8; border-radius: 50%; animation: spin 1s cubic-bezier(0.5, 0, 0.5, 1) infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
<canvas id="viewport"></canvas>

<div id="loading"><div class="loader"></div></div>

<div id="hud-top">
    <div class="sys-title">MULTI-RESERVOIR DIGITAL TWIN</div>
    <div class="sys-sub">LIVE SIMULATION ENGINE</div>
</div>

<div id="downstream-hud">
    <div class="title">Terminal Outflow</div>
    <div class="val"><span id="ds-flow">0.0</span> <span style="font-size:11px; color:#8899aa; font-weight:500">MCM/d</span></div>
    <div style="font-size:9px; font-weight:700; letter-spacing:1px; color:#ff5555; margin-top:4px; display:none" id="ds-warn">⚠ CAPACITY EXCEEDED</div>
</div>

<div id="cam-controls">
    <button id="btn-overview">Overview</button>
    <button id="btn-resa">Res A</button>
    <button id="btn-resb">Res B</button>
    <button id="btn-resc">Res C</button>
</div>

<script type="importmap">
{"imports":{
    "three":"https://unpkg.com/three@0.164.1/build/three.module.js",
    "three/addons/":"https://unpkg.com/three@0.164.1/examples/jsm/"
}}
</script>

<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';

// ============================================================
// NOISE UTILS (For procedural terrain)
// ============================================================
const _P = new Uint8Array(512);
for(let i=0;i<256;i++){let v=i;v=((v>>1)^(-(v&1)&0xEDB88320))>>>0;v=((v>>1)^(-(v&1)&0xEDB88320))>>>0;_P[i]=_P[i+256]=v&255}
function _grad(h,x,y){const g=h&3;return(g===0?x+y:g===1?-x+y:g===2?x-y:-x-y)}
function noise2D(x,y){const X=Math.floor(x)&255,Y=Math.floor(y)&255;x-=Math.floor(x);y-=Math.floor(y);const u=x*x*(3-2*x),v=y*y*(3-2*y);const a=_P[X]+Y,b=_P[X+1]+Y;return .5+.5*((1-v)*((1-u)*_grad(_P[a],x,y)+u*_grad(_P[b],x-1,y))+v*((1-u)*_grad(_P[a+1],x,y-1)+u*_grad(_P[b+1],x-1,y-1)))}
function fbm(x,y,oct=6){let v=0,a=.5,f=1;for(let i=0;i<oct;i++){v+=a*noise2D(x*f,y*f);a*=.4;f*=2.1}return v}

// ============================================================
// STATE & CONFIG
// ============================================================
let simState = {
    storm_intensity: 0,
    downstream_flow: 0,
    downstream_capacity: 50,
    reservoirs: {
        "Virtual Reservoir A": { storage_pct: 50, inflow: 0, outflow: 0, gate_position_pct: 0, risk_status: "NORMAL" },
        "Virtual Reservoir B": { storage_pct: 50, inflow: 0, outflow: 0, gate_position_pct: 0, risk_status: "NORMAL" },
        "Virtual Reservoir C": { storage_pct: 50, inflow: 0, outflow: 0, gate_position_pct: 0, risk_status: "NORMAL" }
    }
};

// Spatially separated to form a majestic winding valley
const RES_CONFIG = {
    A: { name: "Virtual Reservoir A", z: 70, x: -10, w: 30, h: 40, floor: 5, frl: 18, damZ: 50 },
    B: { name: "Virtual Reservoir B", z: 10, x: 15, w: 40, h: 45, floor: -5, frl: 8, damZ: -12 },
    C: { name: "Virtual Reservoir C", z: -55, x: -5, w: 50, h: 50, floor: -18, frl: -5, damZ: -80 }
};

const RISK_COLORS = {
    "NORMAL": { border: "rgba(138, 180, 248, 0.6)", text: "#8ab4f8", bg: "transparent" },
    "WATCH": { border: "rgba(251, 188, 4, 0.8)", text: "#fbbc04", bg: "rgba(251, 188, 4, 0.1)" },
    "ALERT": { border: "rgba(250, 123, 23, 0.9)", text: "#fa7b17", bg: "rgba(250, 123, 23, 0.15)" },
    "HIGH RISK": { border: "rgba(234, 67, 53, 1.0)", text: "#ea4335", bg: "rgba(234, 67, 53, 0.2)" }
};

// ============================================================
// THREE.JS CORE SETUP
// ============================================================
const canvas = document.getElementById('viewport');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false, powerPreference: "high-performance" });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;

const labelRenderer = new CSS2DRenderer();
labelRenderer.setSize(window.innerWidth, window.innerHeight);
labelRenderer.domElement.style.position = 'absolute';
labelRenderer.domElement.style.top = '0px';
labelRenderer.domElement.style.pointerEvents = 'none';
document.body.appendChild(labelRenderer.domElement);

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x020408, 0.007);
scene.background = new THREE.Color(0x020408);

// Cinematic Camera Angle
const camera = new THREE.PerspectiveCamera(40, window.innerWidth/window.innerHeight, 1, 1000);
camera.position.set(-90, 110, 120);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, -10, 0);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.maxPolarAngle = Math.PI / 2.05; // allow looking slightly up at mountains
controls.minDistance = 20;
controls.maxDistance = 300;

const clock = new THREE.Clock();

// ============================================================
// CINEMATIC LIGHTING
// ============================================================
// Deep ambient moon/starlight
const hemi = new THREE.HemisphereLight(0x111b2d, 0x050a12, 1.0);
scene.add(hemi);

// Strong directional moon/rim light
const moon = new THREE.DirectionalLight(0x8ab4f8, 3.5);
moon.position.set(100, 120, -50);
moon.castShadow = true;
moon.shadow.mapSize.set(2048, 2048);
moon.shadow.camera.left = -100; moon.shadow.camera.right = 100;
moon.shadow.camera.top = 100; moon.shadow.camera.bottom = -100;
moon.shadow.camera.near = 10; moon.shadow.camera.far = 400;
moon.shadow.bias = -0.0005;
scene.add(moon);

// Soft fill light
const fill = new THREE.DirectionalLight(0x4a6b8c, 0.8);
fill.position.set(-100, 40, 100);
scene.add(fill);

// ============================================================
// CINEMATIC PROCEDURAL TERRAIN
// ============================================================
function getTerrainHeight(x, z) {
    // Base mountainous noise
    let h = fbm(x*0.015, z*0.015, 6) * 45 - 5;
    
    // Add sharp peaks
    const ridge = fbm(x*0.04, z*0.04, 3);
    if(ridge > 0.6) h += (ridge-0.6)*40;

    // Main sweeping valley along Z
    const valleyDist = Math.abs(x - Math.sin(z*0.02)*20);
    h += Math.max(0, valleyDist - 15) * 0.7; // steep walls
    
    // Carve Reservoir Basins
    const carve = (cx, cz, cw, ch, floorY) => {
        const dx = (x - cx) / cw, dz = (z - cz) / ch;
        const dist = dx*dx + dz*dz;
        if (dist < 1) {
            // Smooth blend into basin
            const basinDepth = h - floorY;
            h -= (1 - Math.pow(dist, 1.5)) * basinDepth;
        }
    };
    carve(RES_CONFIG.A.x, RES_CONFIG.A.z, RES_CONFIG.A.w, RES_CONFIG.A.h, RES_CONFIG.A.floor);
    carve(RES_CONFIG.B.x, RES_CONFIG.B.z, RES_CONFIG.B.w, RES_CONFIG.B.h, RES_CONFIG.B.floor);
    carve(RES_CONFIG.C.x, RES_CONFIG.C.z, RES_CONFIG.C.w, RES_CONFIG.C.h, RES_CONFIG.C.floor);
    
    // Carve connecting river channel
    if (valleyDist < 8) {
        // Gradient slope down the valley
        const riverBedY = (z / 150) * 20 - 15; 
        if (h > riverBedY) {
            h -= (1 - (valleyDist/8)) * (h - riverBedY);
        }
    }
    
    return h;
}

const TSIZE = 240, TSEG = 240;
const terrainGeo = new THREE.PlaneGeometry(TSIZE, TSIZE*1.5, TSEG, parseInt(TSEG*1.5));
terrainGeo.rotateX(-Math.PI/2);
const tPos = terrainGeo.attributes.position;
const tCol = new Float32Array(tPos.count * 3);

for (let i = 0; i < tPos.count; i++) {
    const x = tPos.getX(i), z = tPos.getZ(i);
    const h = getTerrainHeight(x, z);
    tPos.setY(i, h);
    
    // Procedural texturing based on height and noise
    let r, g, b;
    const n = fbm(x*0.1, z*0.1, 3);
    
    if (h > 25) { 
        // Snow/Light rock peaks
        r = 0.6+n*0.2; g = 0.6+n*0.2; b = 0.65+n*0.2; 
    } else if (h > 5) { 
        // Dark steep rock
        r = 0.15+n*0.1; g = 0.16+n*0.1; b = 0.18+n*0.1; 
    } else { 
        // Valley soil / dark moss
        r = 0.08+n*0.05; g = 0.11+n*0.05; b = 0.10+n*0.05; 
    }
    
    tCol[i*3] = r; tCol[i*3+1] = g; tCol[i*3+2] = b;
}
terrainGeo.setAttribute('color', new THREE.BufferAttribute(tCol, 3));
terrainGeo.computeVertexNormals();

const terrainMat = new THREE.MeshStandardMaterial({
    vertexColors: true, 
    roughness: 0.8, 
    metalness: 0.1,
    flatShading: false
});
const terrain = new THREE.Mesh(terrainGeo, terrainMat);
terrain.receiveShadow = true;
terrain.castShadow = true;
scene.add(terrain);

// ============================================================
// PREMIUM WATER SHADER (GLSL)
// ============================================================
// Vertex shader: physically displaces the mesh to create waves
const waterVert = `
uniform float uTime;
uniform float uWaveStr;
varying vec3 vWorldPos;
varying vec3 vNormal;
void main() {
    vec3 pos = position;
    float t = uTime * 1.5;
    
    // Complex multi-directional waves
    float w1 = sin(pos.x * 0.8 + t) * 0.2;
    float w2 = sin(pos.y * 0.6 - t * 0.8) * 0.15;
    float w3 = sin((pos.x + pos.y) * 0.4 + t * 1.2) * 0.1;
    
    float totalWave = (w1 + w2 + w3) * uWaveStr;
    pos.z += totalWave; // Note: Plane is rotated, z is world Y
    
    vec4 worldPosition = modelMatrix * vec4(pos, 1.0);
    vWorldPos = worldPosition.xyz;
    
    // Compute analytical normals
    float dx = cos(pos.x * 0.8 + t) * 0.8 * 0.2 + cos((pos.x + pos.y) * 0.4 + t * 1.2) * 0.4 * 0.1;
    float dy = cos(pos.y * 0.6 - t * 0.8) * 0.6 * 0.15 + cos((pos.x + pos.y) * 0.4 + t * 1.2) * 0.4 * 0.1;
    vNormal = normalize(vec3(-dx * uWaveStr, -dy * uWaveStr, 1.0));
    
    gl_Position = projectionMatrix * viewMatrix * worldPosition;
}`;

// Fragment shader: handles deep water absorption, reflections, and foam
const waterFrag = `
uniform float uTime;
uniform float uTurbulence;
uniform vec3 uColorBase;
uniform vec3 uColorShallow;
varying vec3 vWorldPos;
varying vec3 vNormal;

void main() {
    vec3 viewDir = normalize(cameraPosition - vWorldPos);
    vec3 normal = normalize(vNormal);
    
    // Fresnel effect for reflections
    float f0 = 0.02;
    float fresnel = f0 + (1.0 - f0) * pow(1.0 - max(dot(normal, viewDir), 0.0), 5.0);
    
    // Sky reflection color (cinematic moon/night sky)
    vec3 skyColor = mix(vec3(0.05, 0.1, 0.2), vec3(0.5, 0.7, 0.9), fresnel);
    
    // Depth fake (gradient along world Y or center)
    float depthMix = smoothstep(-10.0, 10.0, vWorldPos.y);
    vec3 waterBase = mix(uColorBase, uColorShallow, depthMix);
    
    // Specular highlight from moonlight
    vec3 lightDir = normalize(vec3(100.0, 120.0, -50.0));
    vec3 halfVec = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfVec), 0.0), 256.0) * 2.0;
    
    // Foam for turbulence
    float foam = smoothstep(0.6, 1.0, normal.z) * uTurbulence * 0.5;
    vec3 foamColor = vec3(0.8, 0.9, 1.0) * foam;
    
    vec3 finalColor = mix(waterBase, skyColor, fresnel) + vec3(spec) + foamColor;
    
    gl_FragColor = vec4(finalColor, 0.92);
}`;

class ResVisual {
    constructor(key, config) {
        this.key = key;
        this.config = config;
        
        // Premium Water Mesh
        const geo = new THREE.PlaneGeometry(config.w * 1.5, config.h * 1.5, 128, 128);
        this.uniforms = {
            uTime: { value: 0 },
            uWaveStr: { value: 1.0 },
            uTurbulence: { value: 0.0 },
            uColorBase: { value: new THREE.Color(0x020a14) },
            uColorShallow: { value: new THREE.Color(0x0a1a30) }
        };
        const mat = new THREE.ShaderMaterial({
            vertexShader: waterVert, fragmentShader: waterFrag,
            uniforms: this.uniforms, transparent: true,
            side: THREE.DoubleSide, depthWrite: false
        });
        this.water = new THREE.Mesh(geo, mat);
        this.water.rotation.x = -Math.PI/2;
        this.water.position.set(config.x, config.floor + (config.frl - config.floor)*0.5, config.z);
        scene.add(this.water);
        
        // Realistic Engineering Dam Structure
        const damGroup = new THREE.Group();
        damGroup.position.set(config.x, config.floor, config.damZ);
        
        const damMat = new THREE.MeshStandardMaterial({color: 0x222222, roughness: 0.7, metalness: 0.2});
        
        // Main wall arc
        const damGeo = new THREE.CylinderGeometry(20, 20, 25, 32, 1, false, Math.PI*0.8, Math.PI*0.4);
        const wall = new THREE.Mesh(damGeo, damMat);
        wall.position.set(0, 12.5, 18);
        wall.castShadow = true; wall.receiveShadow = true;
        damGroup.add(wall);
        
        // Spillway channel
        const spillGeo = new THREE.BoxGeometry(6, 15, 6);
        const spillway = new THREE.Mesh(spillGeo, damMat);
        spillway.position.set(0, 15, 0);
        damGroup.add(spillway);
        
        scene.add(damGroup);
        
        // Moving Gate
        const gateGeo = new THREE.BoxGeometry(4.5, 8, 1.5);
        const gateMat = new THREE.MeshStandardMaterial({color: 0x4a6b8c, roughness: 0.4, metalness: 0.8});
        this.gate = new THREE.Mesh(gateGeo, gateMat);
        this.gate.position.set(config.x, config.floor + 15, config.damZ);
        scene.add(this.gate);
        
        // Minimalist Glass HUD
        const div = document.createElement('div');
        div.className = 'hud-label';
        div.innerHTML = `
            <div class="hud-line"></div>
            <div class="title">${config.name}</div>
            <div class="row"><span class="lbl">Storage</span><span class="val" id="hud-${key}-st">0%</span></div>
            <div class="row"><span class="lbl">Level</span><span class="val" id="hud-${key}-lvl">0m</span></div>
            <div class="row"><span class="lbl">Inflow</span><span class="val" id="hud-${key}-in">0</span></div>
            <div class="row"><span class="lbl">Release</span><span class="val" id="hud-${key}-out">0</span></div>
            <div class="row"><span class="lbl">Gate</span><span class="val" id="hud-${key}-gate">0%</span></div>
            <div class="risk" id="hud-${key}-risk">NORMAL</div>
        `;
        this.label = new CSS2DObject(div);
        // Position HUD offset to the right so it doesn't obscure the dam
        this.label.position.set(config.x + 20, config.frl + 10, config.z);
        scene.add(this.label);
        this.div = div;
        
        // Outflow/Turbulence Particles
        const pGeo = new THREE.BufferGeometry();
        const pPos = new Float32Array(150 * 3);
        pGeo.setAttribute('position', new THREE.BufferAttribute(pPos, 3));
        const pMat = new THREE.PointsMaterial({color: 0xffffff, size: 0.8, transparent: true, opacity: 0.5, blending: THREE.AdditiveBlending, depthWrite: false});
        this.outflowPts = new THREE.Points(pGeo, pMat);
        scene.add(this.outflowPts);
        this.outSpeed = new Float32Array(150);
        for(let i=0; i<150; i++) this.outSpeed[i] = 0.5 + Math.random();
    }
    
    update(state, dt, elapsed) {
        if (!state) return;
        
        // Data interpolation & HUD mapping
        const pct = state.storage_pct || 0;
        
        // 1. Water Elevation (Driven EXACTLY by simulation state)
        const targetY = this.config.floor + (pct/100)*(this.config.frl - this.config.floor);
        // Smoothly interpolate physical visual height
        this.water.position.y += (targetY - this.water.position.y) * 3 * dt;
        
        // 2. Water Fluid Dynamics (Driven by inflow/storm)
        this.uniforms.uTime.value = elapsed;
        const waveIntensity = state.inflow > 60 ? 2.5 : (state.inflow > 20 ? 1.5 : 0.8);
        this.uniforms.uWaveStr.value += (waveIntensity - this.uniforms.uWaveStr.value) * 2 * dt;
        
        const turb = state.outflow > 20 ? 1.0 : (state.outflow / 20.0);
        this.uniforms.uTurbulence.value += (turb - this.uniforms.uTurbulence.value) * 2 * dt;
        
        // 3. Gate Animation (Driven by simulation command)
        const gateOpenY = this.config.floor + 20;
        const gateClosedY = this.config.floor + 12;
        const targetGateY = gateClosedY + (state.gate_position_pct/100)*(gateOpenY - gateClosedY);
        this.gate.position.y += (targetGateY - this.gate.position.y) * 5 * dt;
        
        // 4. Outflow Spray Particles
        const flowInt = Math.min(2.0, state.outflow / 15.0);
        const pts = this.outflowPts.geometry.attributes.position.array;
        for(let i=0; i<150; i++) {
            pts[i*3+1] -= dt * 15 * this.outSpeed[i] * flowInt;
            pts[i*3+2] -= dt * 20 * this.outSpeed[i] * flowInt;
            
            // Reset particle at dam outlet
            if(pts[i*3+1] < this.config.floor - 5 || pts[i*3+2] < this.config.damZ - 20) {
                pts[i*3] = this.config.x + (Math.random()-0.5)*4;
                pts[i*3+1] = this.gate.position.y - 4;
                pts[i*3+2] = this.config.damZ - 2;
            }
        }
        this.outflowPts.geometry.attributes.position.needsUpdate = true;
        this.outflowPts.material.opacity = flowInt > 0.05 ? 0.6 : 0.0;
        
        // 5. Update UI Safely (no global document query)
        const el = this.label.element;
        el.querySelector(`#hud-${this.key}-st`).innerText = `${pct.toFixed(1)}%`;
        el.querySelector(`#hud-${this.key}-lvl`).innerText = `${targetY.toFixed(1)}m`;
        el.querySelector(`#hud-${this.key}-in`).innerText = `${state.inflow.toFixed(1)}`;
        el.querySelector(`#hud-${this.key}-out`).innerText = `${state.outflow.toFixed(1)}`;
        el.querySelector(`#hud-${this.key}-gate`).innerText = `${state.gate_position_pct.toFixed(0)}%`;
        
        // Risk visuals
        const rEl = el.querySelector(`#hud-${this.key}-risk`);
        const riskVis = RISK_COLORS[state.risk_status] || RISK_COLORS["NORMAL"];
        rEl.innerText = state.risk_status;
        rEl.style.color = riskVis.text;
        rEl.style.backgroundColor = riskVis.bg;
        rEl.style.borderColor = riskVis.text;
        this.div.style.borderLeftColor = riskVis.text;
    }
}

// Instantiate the cascade
const resA = new ResVisual("A", RES_CONFIG.A);
const resB = new ResVisual("B", RES_CONFIG.B);
const resC = new ResVisual("C", RES_CONFIG.C);

// ============================================================
// STORM / WEATHER FX
// ============================================================
const rainGeo = new THREE.BufferGeometry();
const rainPos = new Float32Array(3000 * 3);
for(let i=0; i<3000; i++) {
    rainPos[i*3] = (Math.random()-0.5)*200;
    rainPos[i*3+1] = Math.random()*100;
    rainPos[i*3+2] = (Math.random()-0.5)*300;
}
rainGeo.setAttribute('position', new THREE.BufferAttribute(rainPos, 3));
const rainMat = new THREE.PointsMaterial({color: 0x8ab4f8, size: 0.15, transparent: true, opacity: 0});
const rain = new THREE.Points(rainGeo, rainMat);
scene.add(rain);

// ============================================================
// SIMULATION DATA BRIDGE
// ============================================================
window.addEventListener("message", (event) => {
    if (event.data && event.data.type === "UPDATE_SIM") {
        simState = event.data.data;
    }
});

// ============================================================
// CINEMATIC CAMERA SYSTEM
// ============================================================
function tweenCam(pos, tgt) {
    const sP = camera.position.clone(), sT = controls.target.clone();
    let t = 0;
    function step() {
        t += 0.015; // smooth cinematic transition
        if(t > 1) t = 1;
        const e = t<.5 ? 2*t*t : -1+(4-2*t)*t; // easeInOut Quad
        camera.position.lerpVectors(sP, pos, e);
        controls.target.lerpVectors(sT, tgt, e);
        controls.update();
        if(t < 1) requestAnimationFrame(step);
    }
    step();
}

// Presets clearly showing topology and downsteam flow
document.getElementById('btn-overview').onclick = () => tweenCam(new THREE.Vector3(-100, 130, 100), new THREE.Vector3(0, -10, 0));
document.getElementById('btn-resa').onclick = () => tweenCam(new THREE.Vector3(-40, 50, 110), new THREE.Vector3(RES_CONFIG.A.x, 15, RES_CONFIG.A.z));
document.getElementById('btn-resb').onclick = () => tweenCam(new THREE.Vector3(-45, 40, 60), new THREE.Vector3(RES_CONFIG.B.x, 0, RES_CONFIG.B.z));
document.getElementById('btn-resc').onclick = () => tweenCam(new THREE.Vector3(-55, 30, 0), new THREE.Vector3(RES_CONFIG.C.x, -10, RES_CONFIG.C.z));

// ============================================================
// MAIN RENDER LOOP
// ============================================================
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    labelRenderer.setSize(window.innerWidth, window.innerHeight);
});

function animate() {
    requestAnimationFrame(animate);
    const dt = Math.min(clock.getDelta(), 0.1);
    const elapsed = clock.getElapsedTime();
    
    // Update individual reservoirs from simulation state
    if(simState.reservoirs[RES_CONFIG.A.name]) resA.update(simState.reservoirs[RES_CONFIG.A.name], dt, elapsed);
    if(simState.reservoirs[RES_CONFIG.B.name]) resB.update(simState.reservoirs[RES_CONFIG.B.name], dt, elapsed);
    if(simState.reservoirs[RES_CONFIG.C.name]) resC.update(simState.reservoirs[RES_CONFIG.C.name], dt, elapsed);
    
    // Process Storm Weather state
    const targetRainOp = simState.storm_intensity > 0 ? 0.6 : 0.0;
    rainMat.opacity += (targetRainOp - rainMat.opacity) * dt * 1.5;
    
    if (rainMat.opacity > 0.01) {
        const rPts = rain.geometry.attributes.position.array;
        for(let i=0; i<3000; i++) {
            // Wind and gravity
            rPts[i*3] -= dt * (10 + simState.storm_intensity * 10);
            rPts[i*3+1] -= dt * (40 + simState.storm_intensity * 30);
            if(rPts[i*3+1] < -30) {
                rPts[i*3] = (Math.random()-0.5)*200;
                rPts[i*3+1] = 80 + Math.random()*20;
                rPts[i*3+2] = (Math.random()-0.5)*300;
            }
        }
        rain.geometry.attributes.position.needsUpdate = true;
        
        // Make fog denser during storm
        scene.fog.density = 0.007 + (simState.storm_intensity * 0.01);
    } else {
        scene.fog.density = 0.007;
    }
    
    // Update Downstream Terminal Flow HUD
    document.getElementById('ds-flow').innerText = simState.downstream_flow.toFixed(1);
    const isExceeded = simState.downstream_flow > simState.downstream_capacity;
    document.getElementById('ds-warn').style.display = isExceeded ? "block" : "none";
    document.getElementById('downstream-hud').style.borderLeftColor = isExceeded ? "#ea4335" : "#8ab4f8";
    
    controls.update();
    renderer.render(scene, camera);
    labelRenderer.render(scene, camera);
}

// Dismiss loader gracefully
setTimeout(() => {
    const l = document.getElementById('loading');
    l.style.opacity = 0;
    setTimeout(() => l.style.display = 'none', 1000);
}, 800);

animate();
</script>
</body>
</html>
"""
