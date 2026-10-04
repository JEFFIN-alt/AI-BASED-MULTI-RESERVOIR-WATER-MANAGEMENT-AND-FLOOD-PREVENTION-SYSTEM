import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';

// All coordinates and detail here are art-directed scene units, not GIS data.
// Triplanar world-space detail avoids stretched UVs on steep rock faces.
export function terrainMaterial() {
  const material = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.94 });
  material.onBeforeCompile = shader => {
    shader.vertexShader = shader.vertexShader.replace('#include <common>', `#include <common>
      varying vec3 vGround; varying vec3 vGroundNormal;`)
      .replace('#include <begin_vertex>', `#include <begin_vertex>
      vGround = position; vGroundNormal = normal;`);
    shader.fragmentShader = shader.fragmentShader.replace('#include <common>', `#include <common>
      varying vec3 vGround; varying vec3 vGroundNormal;
      float groundHash(vec3 p) { return fract(sin(dot(p,vec3(127.1,311.7,74.7)))*43758.5453); }
      float groundNoise(vec3 p) {
        vec3 i=floor(p),f=fract(p); f=f*f*(3.-2.*f);
        return mix(mix(mix(groundHash(i),groundHash(i+vec3(1,0,0)),f.x),
          mix(groundHash(i+vec3(0,1,0)),groundHash(i+vec3(1,1,0)),f.x),f.y),
          mix(mix(groundHash(i+vec3(0,0,1)),groundHash(i+vec3(1,0,1)),f.x),
          mix(groundHash(i+vec3(0,1,1)),groundHash(i+vec3(1,1,1)),f.x),f.y),f.z);
      }`)
      .replace('#include <color_fragment>', `#include <color_fragment>
        float detail = groundNoise(vGround * 0.48);
        float grain = groundNoise(vGround * 2.1);
        float rock = smoothstep(0.30,0.72,1.-normalize(vGroundNormal).y);
        float strata = sin(vGround.y*0.22 + groundNoise(vGround*0.055)*14.)*.5+.5;
        diffuseColor.rgb *= (0.68 + detail*0.52 + grain*0.20);
        diffuseColor.rgb = mix(diffuseColor.rgb, vec3(.10,.115,.095)*(0.90+strata*.12+detail*.2),rock*.45);`)
      .replace('#include <roughnessmap_fragment>', `#include <roughnessmap_fragment>
        roughnessFactor = 0.82 + 0.16*groundNoise(vGround*0.3);`);
  };
  material.customProgramCacheKey = () => 'aquaflow-ground-v1';
  return material;
}

export function forestCrown(seed, slender = false, distant = false) {
  const parts = [];
  // Overlapping irregular crowns read as broadleaf canopy, with alternate tall trees.
  for (let k = 0; k < (distant ? 2 : 4); k++) {
    const angle = k * 2.399 + seed;
    const g = new THREE.SphereGeometry(k === 0 ? 2.1 : 1.45, distant ? 5 : 7, distant ? 4 : 5);
    const p = g.attributes.position;
    for (let i=0; i<p.count; i++) {
      const x=p.getX(i), y=p.getY(i), z=p.getZ(i);
      const n=1+Math.sin(x*3.1+y*2.7+z*4.3+seed)*.16;
      p.setXYZ(i,x*n,y*n,z*n);
    }
    g.scale(slender ? .67 : 1, slender ? 1.7 : .75, slender ? .67 : 1);
    g.translate(k ? Math.cos(angle)*1.35 : 0, 3.5+(k ? Math.sin(k*4+seed)*.8 : 1), k ? Math.sin(angle)*1.35 : 0);
    // Preserve the smooth radial normals; recomputing after jitter exaggerates facets.
    parts.push(g);
  }
  const result=mergeGeometries(parts,false);
  parts.forEach(g=>g.dispose());
  return result;
}

export function forestMaterial() {
  const material=new THREE.MeshStandardMaterial({color:0xffffff,roughness:0.96});
  material.onBeforeCompile=shader=>{
    shader.vertexShader=shader.vertexShader.replace('#include <common>',`#include <common>
      varying vec3 vLeaf;`).replace('#include <project_vertex>',`
      vLeaf=(modelMatrix*instanceMatrix*vec4(transformed,1.0)).xyz;
      #include <project_vertex>`);
    shader.fragmentShader=shader.fragmentShader.replace('#include <common>',`#include <common>
      varying vec3 vLeaf;
      float leafGrain(vec3 p){return fract(sin(dot(floor(p),vec3(12.9898,78.233,37.719)))*43758.5453);}`)
      .replace('#include <color_fragment>',`#include <color_fragment>
        float leaves=leafGrain(vLeaf*2.7);
        float clusters=leafGrain(vLeaf*.65);
        diffuseColor.rgb *= 0.72+leaves*.30+clusters*.35;`);
  };
  material.customProgramCacheKey=()=> 'aquaflow-foliage-v1';
  return material;
}

export function addDistantRidges(scene, height) {
  const geometry = new THREE.PlaneGeometry(4400, 4400, 128, 128);
  geometry.rotateX(-Math.PI/2);
  const p=geometry.attributes.position;
  for(let i=0;i<p.count;i++) {
    const x=p.getX(i),z=p.getZ(i);
    p.setY(i,height(x,z)-12);
  }
  geometry.computeVertexNormals();
  const indices=geometry.index.array, outer=[];
  for(let i=0;i<indices.length;i+=3) {
    const a=indices[i],b=indices[i+1],c=indices[i+2];
    const x=(p.getX(a)+p.getX(b)+p.getX(c))/3,z=(p.getZ(a)+p.getZ(b)+p.getZ(c))/3;
    if(Math.max(Math.abs(x),Math.abs(z))>1270)outer.push(a,b,c);
  }
  geometry.setIndex(outer);
  const mesh=new THREE.Mesh(geometry,new THREE.MeshStandardMaterial({color:0x354b3e,roughness:1}));
  mesh.name='DISTANT_MOUNTAIN_CONTEXT';
  // The fine central terrain covers this lower mesh. No image background.
  scene.add(mesh);
}

export function createSiteView(container, reservoirName, onSelect) {
  const toolbar=document.createElement('div');
  toolbar.className='scene-view-switch';
  toolbar.innerHTML='<button data-world="twin" aria-pressed="true">DIGITAL TWIN</button><button data-world="site" aria-pressed="false">SITE VIEW</button>';
  const panel=document.createElement('section');
  panel.className='site-context'; panel.hidden=true;
  panel.setAttribute('aria-label','Real-world site reference');
  panel.innerHTML='<div class="site-heading"><span>REAL-WORLD SITE REFERENCE</span><h2></h2></div><div class="site-reservoirs"></div><div class="site-modes"><button data-mode="map">MAP</button><button data-mode="satellite">SATELLITE</button><button data-mode="street">STREET VIEW</button></div><div class="site-content"></div>';
  let selected=0,mode='map';
  const render=()=>{
    panel.querySelector('h2').textContent=reservoirName(selected);
    const choices=panel.querySelector('.site-reservoirs'); choices.replaceChildren();
    for(let i=0;i<4;i++) {
      const b=document.createElement('button'); b.textContent=reservoirName(i);
      b.setAttribute('aria-pressed',String(i===selected));
      b.onclick=()=>{selected=i;onSelect(i);render();}; choices.append(b);
    }
    panel.querySelectorAll('[data-mode]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.mode===mode)));
    const content=panel.querySelector('.site-content');content.replaceChildren();
    // Deployment may configure legitimate embed URLs; no inferred coordinates or fake panoramas.
    const configured=window.AQUAFLOW_SITE_IMAGERY?.['reservoir_'+(selected+1)]?.[mode];
    let url=null;
    try { const parsed=new URL(configured); if(parsed.protocol==='https:') url=parsed.href; } catch (_) { /* unconfigured */ }
    if(url) {
      const frame=document.createElement('iframe');frame.src=url;
      frame.title=reservoirName(selected)+' '+mode+' provider imagery';
      frame.referrerPolicy='strict-origin-when-cross-origin';
      frame.setAttribute('allowfullscreen','');content.append(frame);
    } else {
      const title=document.createElement('strong');title.textContent='SITE IMAGERY NOT CONFIGURED';
      const note=document.createElement('p');note.textContent='No '+(mode==='street'?'Street View':mode)+' provider is configured for this reservoir. The digital twin uses simulation topology, not surveyed geographic terrain.';
      content.append(title,note);
    }
  };
  toolbar.onclick=e=>{
    const button=e.target.closest('[data-world]');if(!button)return;
    panel.hidden=button.dataset.world==='twin';
    toolbar.querySelectorAll('button').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));
    container.classList.toggle('site-open',!panel.hidden);if(!panel.hidden)render();
  };
  panel.querySelectorAll('[data-mode]').forEach(b=>b.onclick=()=>{mode=b.dataset.mode;render();});
  container.append(toolbar,panel);
  return {select(i){selected=i;if(!panel.hidden)render();}};
}
