import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { Line2 } from 'three/examples/jsm/lines/Line2.js';
import { LineGeometry } from 'three/examples/jsm/lines/LineGeometry.js';
import { LineMaterial } from 'three/examples/jsm/lines/LineMaterial.js';
import { visualGeometry, visualMaterial } from './sceneGraphics';
import type { Geometry, Pose } from './types';

export type RoutePoint = { x: number; y: number; wall_s: number; segment: number; status: string; activity: string };
export type RecordedRoute = { points: RoutePoint[]; bounds_m: number[]; contacts: RoutePoint[]; sample_count: number;
  downsampled: boolean; contact_markers_truncated: boolean;
  scene?: { source: 'recorded_initial' | 'reconstructed_current'; geometry: Geometry[]; poses: Pose[]; simulated_s: number | null } | null };

export function RecordedScene({route, cursor, view, wholeRoom}: {
  route: RecordedRoute; cursor: number; view: 'top' | 'perspective'; wholeRoom: boolean;
}) {
  const host = useRef<HTMLDivElement>(null);
  const update = useRef<(time: number) => void>(()=>{});
  const latestTime = useRef(cursor);
  latestTime.current = cursor;
  const [error, setError] = useState('');
  const [hovered, setHovered] = useState('');
  useEffect(() => {
    if (!host.current || !route.scene) return;
    const element = host.current;
    setError(''); setHovered('');
    let renderer: THREE.WebGLRenderer;
    try { renderer = new THREE.WebGLRenderer({antialias:true, preserveDrawingBuffer:true}); }
    catch { setError('3D environment unavailable.'); return; }
    let disposed = false;
    const palette = getComputedStyle(document.documentElement);
    const color = (name: string)=>new THREE.Color(palette.getPropertyValue(`--cp-${name}`).trim());
    renderer.setPixelRatio(Math.min(devicePixelRatio,2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.domElement.setAttribute('aria-label','Recorded environment and robot route');
    renderer.domElement.tabIndex = 0;
    element.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    scene.background = color('surface-soft');
    scene.add(new THREE.HemisphereLight(color('surface'),color('border-strong'),2.6));
    const light = new THREE.DirectionalLight(color('surface'),2);
    light.position.set(-3,10,4); scene.add(light);
    const world = new THREE.Group(); world.rotation.x = -Math.PI/2; scene.add(world);
    const camera = new THREE.OrthographicCamera(-5,5,4,-4,.01,200);
    const controls = new OrbitControls(camera,renderer.domElement);
    controls.enableRotate = view==='perspective';
    controls.screenSpacePanning = true;
    controls.minZoom = .25; controls.maxZoom = 8;
    controls.maxPolarAngle = Math.PI/2-.04;
    const geometries: THREE.BufferGeometry[] = [];
    const materials: THREE.Material[] = [];
    const textures = new Map<string,THREE.Texture>();
    const meshes: THREE.Mesh[] = [];
    const loader = new THREE.TextureLoader();
    const poses = new Map(route.scene.poses.map(pose=>[pose.key,pose]));
    const render = ()=>{if(!disposed) renderer.render(scene,camera);};
    let pendingTextures = 0;
    const textureDone = ()=>{
      pendingTextures--; renderer.domElement.dataset.texturesPending=String(pendingTextures); render();
    };
    for (const asset of route.scene.geometry) {
      const pose=poses.get(asset.key);
      if (!pose) continue;
      let texture: THREE.Texture | undefined;
      if (asset.texture && /^\/api\/textures\/[a-z]+\.png$/.test(asset.texture)) {
        texture=textures.get(asset.texture);
        if (!texture) {
          pendingTextures++;
          texture=loader.load(asset.texture,textureDone,undefined,()=>{
            renderer.domElement.dataset.textureError='true'; textureDone();
          });
          texture.colorSpace=THREE.SRGBColorSpace;
          texture.anisotropy=Math.min(8,renderer.capabilities.getMaxAnisotropy());
          textures.set(asset.texture,texture);
        }
      }
      const geometry=visualGeometry(asset,'enhanced'), material=visualMaterial(asset,texture,'standard');
      const mesh=new THREE.Mesh(geometry,material);
      mesh.position.fromArray(asset.position); mesh.quaternion.fromArray(asset.quaternion);
      mesh.userData.name=(asset.name ?? 'Object').replaceAll('_',' ');
      const group=new THREE.Group();
      group.position.fromArray(pose.position); group.quaternion.fromArray(pose.quaternion);
      group.add(mesh); world.add(group);
      meshes.push(mesh); geometries.push(geometry); materials.push(material);
    }
    renderer.domElement.dataset.texturesPending=String(pendingTextures);
    renderer.domElement.dataset.objects=String(meshes.length);
    const roomBounds=new THREE.Box3().setFromObject(world);
    const [minimumX,minimumY,maximumX,maximumY]=route.bounds_m;
    const centerX=wholeRoom?(roomBounds.min.x+roomBounds.max.x)/2:(minimumX+maximumX)/2;
    const centerY=wholeRoom?-(roomBounds.min.z+roomBounds.max.z)/2:(minimumY+maximumY)/2;
    const spanX=wholeRoom?roomBounds.max.x-roomBounds.min.x+1:Math.max(4,maximumX-minimumX+2.5);
    const spanY=wholeRoom?roomBounds.max.z-roomBounds.min.z+1:Math.max(4,maximumY-minimumY+2.5);
    const span=Math.max(spanX,spanY);
    controls.target.set(centerX,0,-centerY);
    camera.position.set(centerX+(view==='perspective'?span*.55:0),span*1.4,-centerY+(view==='perspective'?span*.75:.001));
    camera.lookAt(controls.target); controls.update();
    const routeLines: {line: Line2; points: RoutePoint[]}[]=[];
    const lineMaterials: LineMaterial[]=[];
    const segments: RoutePoint[][]=[];
    for (const point of route.points) {
      if (segments.at(-1)?.at(-1)?.segment!==point.segment) segments.push([]);
      segments.at(-1)!.push(point);
    }
    for (const points of segments) {
      if(points.length<2) continue;
      for(const travelled of [false,true]) {
        const geometry=new LineGeometry(); geometry.setPositions(points.flatMap(point=>[point.x,point.y,.06]));
        const material=new LineMaterial({color:color(travelled?'accent':'border-strong'),linewidth:travelled?4:2,
          transparent:true,opacity:travelled?1:.65,depthTest:false,depthWrite:false});
        const line=new Line2(geometry,material); line.renderOrder=travelled?11:10;
        world.add(line); geometries.push(geometry); materials.push(material); lineMaterials.push(material);
        if(travelled) routeLines.push({line,points});
      }
    }
    function marker(point: RoutePoint, kind: string, size: number, square=false) {
      const geometry=square?new THREE.BoxGeometry(size,size,size):new THREE.SphereGeometry(size/2,20,12);
      const material=new THREE.MeshBasicMaterial({color:color(kind),depthTest:false,depthWrite:false});
      const mesh=new THREE.Mesh(geometry,material); mesh.position.set(point.x,point.y,.1); mesh.renderOrder=12;
      world.add(mesh); geometries.push(geometry); materials.push(material); return mesh;
    }
    marker(route.points[0],'success',.22);
    marker(route.points.at(-1)!,'accent',.17,true);
    for(const point of route.contacts) marker(point,'danger',.17);
    const selected=marker(route.points[0],'link',.18); selected.renderOrder=13;
    update.current=time=>{
      let current=route.points[0];
      for(const point of route.points) {if(point.wall_s>time)break; current=point;}
      selected.position.set(current.x,current.y,.14);
      for(const entry of routeLines) entry.line.geometry.instanceCount=Math.max(0,entry.points.filter(point=>point.wall_s<=time).length-1);
      renderer.domElement.dataset.time=String(current.wall_s);
      renderer.domElement.dataset.position=JSON.stringify([current.x,current.y]);
      render();
    };
    const raycaster=new THREE.Raycaster();
    const pointer=new THREE.Vector2();
    function inspect(event: PointerEvent) {
      const bounds=renderer.domElement.getBoundingClientRect();
      pointer.set((event.clientX-bounds.left)/bounds.width*2-1,-(event.clientY-bounds.top)/bounds.height*2+1);
      raycaster.setFromCamera(pointer,camera);
      const hit=raycaster.intersectObjects(meshes,false)[0];
      setHovered(hit?String(hit.object.userData.name):'');
    }
    function clearHover(){setHovered('');}
    function resize(){
      const width=element.clientWidth, height=element.clientHeight;
      if(!width || !height)return;
      const aspect=width/height;
      camera.updateMatrixWorld();
      const projected=roomBounds.clone().applyMatrix4(camera.matrixWorldInverse);
      const visibleHeight=wholeRoom?2.1*Math.max(Math.abs(projected.min.y),Math.abs(projected.max.y),
        Math.abs(projected.min.x)/aspect,Math.abs(projected.max.x)/aspect):Math.max(spanY,spanX/aspect)*(view==='perspective'?1.15:1);
      camera.left=-visibleHeight*aspect/2; camera.right=-camera.left;
      camera.top=visibleHeight/2; camera.bottom=-camera.top; camera.updateProjectionMatrix();
      renderer.setSize(width,height,false);
      for(const material of lineMaterials)material.resolution.set(width,height);
      render();
    }
    const observer=new ResizeObserver(resize); observer.observe(element);
    controls.addEventListener('change',render);
    renderer.domElement.addEventListener('pointermove',inspect);
    renderer.domElement.addEventListener('pointerup',inspect);
    renderer.domElement.addEventListener('pointerleave',clearHover);
    resize(); update.current(latestTime.current);
    return ()=>{
      disposed=true; update.current=()=>{}; observer.disconnect(); controls.dispose();
      renderer.domElement.removeEventListener('pointermove',inspect);
      renderer.domElement.removeEventListener('pointerup',inspect);
      renderer.domElement.removeEventListener('pointerleave',clearHover);
      for(const geometry of geometries)geometry.dispose();
      for(const material of materials)material.dispose();
      for(const texture of textures.values())texture.dispose();
      renderer.dispose(); renderer.forceContextLoss(); renderer.domElement.remove();
    };
  },[route,view,wholeRoom]);
  useEffect(()=>update.current(cursor),[cursor]);
  return <div className="results-scene" ref={host}>
    {error && <div className="results-scene-error" role="status">{error}</div>}
    {hovered && <output className="results-scene-object">{hovered}</output>}
  </div>;
}