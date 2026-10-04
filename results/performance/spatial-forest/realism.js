import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';

// All coordinates and detail here are art-directed scene units, not GIS data.
// Triplanar world-space detail avoids stretched UVs on steep rock faces.
export function terrainMaterial() {
  // Cache the scalar lattice once. Hardware bilinear interpolation replaces
  // eight transcendental hashes per 3D noise lookup; no terrain detail removed.
  const tile=66,size=tile*8,data=new Uint8Array(size*size);
  for(let z=0;z<64;z++)for(let y=0;y<tile;y++)for(let x=0;x<tile;x++) {
    const n=Math.sin((x%64)*127.1+(y%64)*311.7+z*74.7)*43758.5453;
    data[(Math.floor(z/8)*tile+y)*size+(z%8)*tile+x]=Math.round((n-Math.floor(n))*255);
  }
  const noise=new THREE.DataTexture(data,size,size,THREE.RedFormat);
  noise.minFilter=noise.magFilter=THREE.LinearFilter;noise.needsUpdate=true;
  const material = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.94 });
  material.userData.noiseTexture=noise;
  material.addEventListener('dispose',()=>noise.dispose());
  material.onBeforeCompile = shader => {
    shader.uniforms.uGroundNoise={value:noise};
    shader.vertexShader = shader.vertexShader.replace('#include <common>', `#include <common>
      varying vec3 vGround; varying vec3 vGroundNormal;`)
      .replace('#include <begin_vertex>', `#include <begin_vertex>
      vGround = position; vGroundNormal = normal;`);
    shader.fragmentShader = shader.fragmentShader.replace('#include <common>', `#include <common>
      varying vec3 vGround; varying vec3 vGroundNormal;
      uniform sampler2D uGroundNoise;
      float groundSlice(vec2 p,float z) {
        return texture2D(uGroundNoise,(vec2(mod(z,8.),floor(z/8.))*66.+p+.5)/528.).r;
      }
      float groundNoise(vec3 p) {
        vec3 i=floor(p),f=fract(p); f=f*f*(3.-2.*f);
        vec2 xy=mod(i.xy,64.)+f.xy; float z=mod(i.z,64.);
        return mix(groundSlice(xy,z),groundSlice(xy,mod(z+1.,64.)),f.z);
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
  material.customProgramCacheKey = () => 'aquaflow-ground-cached-lattice-v2';
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

// Spatial batches retain every instance and its original colour/transform.
// Bounding spheres now cover one forest patch instead of the entire valley.
export function partitionForest(group, lods) {
  const matrix=new THREE.Matrix4(),color=new THREE.Color();
  const result=[];
  for(const source of [...group.children]) {
    if(!source.isInstancedMesh || source.count<1000)continue;
    const cells=new Map(),lod=lods.find(l=>l.mesh===source);
    for(let i=0;i<source.count;i++) {
      source.getMatrixAt(i,matrix);
      const key=Math.floor(matrix.elements[12]/260)+','+Math.floor(matrix.elements[14]/260);
      if(!cells.has(key))cells.set(key,[]);cells.get(key).push(i);
    }
    for(const ids of cells.values()) {
      const mesh=new THREE.InstancedMesh(source.geometry,source.material,ids.length);
      mesh.name=source.name+'_PATCH';mesh.castShadow=source.castShadow;mesh.receiveShadow=source.receiveShadow;
      for(let j=0;j<ids.length;j++) {
        source.getMatrixAt(ids[j],matrix);mesh.setMatrixAt(j,matrix);
        if(source.instanceColor){source.getColorAt(ids[j],color);mesh.setColorAt(j,color);}
      }
      mesh.computeBoundingSphere();group.add(mesh);
      if(lod)result.push({mesh,near:lod.near,far:lod.far,center:mesh.boundingSphere.center.clone()});
    }
    group.remove(source);source.dispose();
  }
  return result;
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
