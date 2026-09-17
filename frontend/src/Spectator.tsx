import { useEffect, useRef, useState } from 'react';
import { MousePointer2 } from 'lucide-react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { LineSegments2 } from 'three/examples/jsm/lines/LineSegments2.js';
import { LineSegmentsGeometry } from 'three/examples/jsm/lines/LineSegmentsGeometry.js';
import { LineMaterial } from 'three/examples/jsm/lines/LineMaterial.js';
import type { LiveState, ManualPlacement, SpatialTelemetry } from './types';
import { MovementZoneOverlay } from './movementZones';
import { TravelledPath } from './travelledPath';
import { detailRobotVisual, enhancedLighting, visualGeometry, visualMaterial, type GraphicsQuality } from './sceneGraphics';
import { usePreference } from './Preferences';

export function Spectator({ state, axes, enabled, onPlace, showZones = false, showTrail = true, telemetry, connected = true }: {
  state: LiveState; axes: boolean; enabled: boolean; onPlace: (placement: ManualPlacement) => Promise<void>;
  showZones?:boolean;showTrail?:boolean;telemetry?:SpatialTelemetry | null;connected?:boolean;
}) {
  const host = useRef<HTMLDivElement>(null);
  const current = useRef(state);
  const axesVisible = useRef(axes);
  const options = useRef({ enabled, onPlace });
  const zoneOptions=useRef({showZones,telemetry,connected});
  zoneOptions.current={showZones,telemetry,connected};
  const [zoneStatus,setZoneStatus]=useState('Unavailable');
  const [beamStatus,setBeamStatus]=useState('Unavailable');
  const [zoneHint,setZoneHint]=useState('');
  const travelledPath = useRef(new TravelledPath());
  const trailVisible = useRef(showTrail);
  trailVisible.current = showTrail;
  useEffect(() => {travelledPath.current.sample(state,connected);}, [state,connected]);
  const interaction = useRef({ select: () => {}, cancel: () => {} });
  const [selected, setSelected] = useState(false);
  const [preview, setPreview] = useState<[number, number] | null>(null);
  const [placing, setPlacing] = useState(false);
  const [error, setError] = useState('');
  const [savedQuality, saveQuality] = usePreference('graphics', 'standard');
  const [qualityOverride, setQualityOverride] = useState<GraphicsQuality | null>(() => {
    const option = new URLSearchParams(location.search).get('graphics');
    return option === 'standard' || option === 'enhanced' ? option : null;
  });
  const preferredQuality = qualityOverride ?? savedQuality;
  function setQuality(mode: GraphicsQuality) { setQualityOverride(null); saveQuality(mode); }
  const quality = state.rendering === 'enhanced' ? 'enhanced' : preferredQuality;
  const savedView = useRef<{ run: string; position: THREE.Vector3; target: THREE.Vector3 } | null>(null);
  current.current = state;
  axesVisible.current = axes;
  options.current = { enabled, onPlace };
  useEffect(() => { if (!enabled) interaction.current.cancel(); }, [enabled]);
  useEffect(() => {
    if (!host.current) return;
    const element = host.current;
    let active = true;
    let isSelected = false;
    setSelected(false);
    setPreview(null);
    setPlacing(false);
    setError('');
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(getComputedStyle(document.documentElement).getPropertyValue('--cp-surface-soft').trim());
    const camera = new THREE.PerspectiveCamera(43, 1, .01, 80);
    const parking = current.current.challenge?.id === 'park';
    const recharging = ['recharge', 'pedestrian_crossing'].includes(current.current.challenge?.id ?? '');
    const apartment = ['apartment', 'kitchen_bathroom'].includes(current.current.challenge?.id ?? '');
    const facility = current.current.challenge?.environment === 'shared_apartment_v1'
      || ['clinic_delivery', 'warehouse', 'inspection', 'flat_kitchen', 'furniture_circuit', 'movement_practice'].includes(current.current.challenge?.id ?? '');
    const workshop = current.current.challenge?.id === 'workshop';
    const floor = current.current.geometry.find(asset => asset.type === 3 && asset.dimensions[0] >= 4 && asset.dimensions[1] >= 4 && asset.dimensions[2] <= .11);
    const floorWidth = floor?.dimensions[0] ?? 6, floorDepth = floor?.dimensions[1] ?? 6;
    camera.position.set(...(facility ? [0, 12, 4.5] : workshop ? [2.5, 4.5, 4] : apartment ? [.45, 7.8, 2.8] : recharging ? [2, 2.6, 2.3] : parking ? [1.8, 1.5, 2] : [1.25, 1.05, 1.5]) as [number, number, number]);
    const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.domElement.setAttribute('aria-label', 'Live robot spectator viewport');
    renderer.domElement.tabIndex = 0;
    renderer.domElement.setAttribute('aria-keyshortcuts', 'ArrowUp ArrowDown ArrowLeft ArrowRight Enter Escape');
    renderer.domElement.title = 'Select the robot and drag to position. Arrow keys adjust a selected robot; Enter places it and Escape cancels.';
    element.appendChild(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(facility ? 0 : workshop ? .5 : apartment ? .45 : recharging ? .65 : parking ? .6 : .12, .25, 0);
    if (savedView.current?.run === state.run_id) {
      camera.position.copy(savedView.current.position);
      controls.target.copy(savedView.current.target);
    }
    controls.enableDamping = true;
    controls.minDistance = .45;
    controls.maxDistance = facility ? 45 : apartment ? 12 : 8;
    const initialCameraOffset = camera.position.clone().sub(controls.target);
    let cameraAdjusted = savedView.current?.run === state.run_id;
    controls.addEventListener('start', () => { cameraAdjusted = true; });
    controls.maxPolarAngle = Math.PI / 2 - .02;
    const hemisphere = new THREE.HemisphereLight(0xffffff, 0x777777, 2);
    scene.add(hemisphere);
    const sun = new THREE.DirectionalLight(0xffffff, 2.4);
    sun.position.set(2, 5, 3);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    const shadowExtent = Math.max(4, floorWidth / 2 + 1, floorDepth / 2 + 1);
    Object.assign(sun.shadow.camera, { left: -shadowExtent, right: shadowExtent, top: shadowExtent, bottom: -shadowExtent, near: .1, far: 30 });
    sun.shadow.normalBias = .015;
    scene.add(sun);
    let disposeLighting = () => {};
    if (quality === 'enhanced') {
      scene.remove(hemisphere, sun);
      disposeLighting = enhancedLighting(scene, renderer, shadowExtent);
    }
    const world = new THREE.Group();
    world.rotation.x = -Math.PI / 2;
    scene.add(world);
    const robotRoot = new THREE.Group();
    world.add(robotRoot);
    const movementZones=new MovementZoneOverlay();
    world.add(movementZones.group);
    const trailGeometry = new LineSegmentsGeometry();
    const trailPositions = new Float32Array(TravelledPath.maximumSegments*6);
    trailGeometry.setPositions(trailPositions);
    const trailBuffer = (trailGeometry.getAttribute('instanceStart') as THREE.InterleavedBufferAttribute).data;
    trailBuffer.setUsage(THREE.DynamicDrawUsage);
    trailGeometry.instanceCount = 0;
    const trailMaterial = new LineMaterial({color:new THREE.Color(getComputedStyle(document.documentElement).getPropertyValue('--cp-link').trim()).getHex(),
      linewidth:3,transparent:true,opacity:.9,depthWrite:false});
    const trail = new LineSegments2(trailGeometry,trailMaterial);
    trail.frustumCulled = false;
    trail.visible = false;
    world.add(trail);
    let trailRevision = -1;
    const axisHelper = new THREE.AxesHelper(.7);
    world.add(axisHelper);
    const links = new Map<string, THREE.Group>();
    const meshes: THREE.Mesh[] = [];
    const robotDetails: (() => void)[] = [];
    const textures = new Map<string, THREE.Texture>();
    const textureLoader = new THREE.TextureLoader();
    for (const asset of current.current.geometry) {
      const dimensions = asset.dimensions;
      const geometry = visualGeometry(asset, quality);
      let texture: THREE.Texture | undefined;
      if (asset.texture) {
        texture = textures.get(asset.texture);
        if (!texture) {
          texture = textureLoader.load(asset.texture);
          texture.colorSpace = THREE.SRGBColorSpace;
          texture.anisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy());
          textures.set(asset.texture, texture);
        }
      }
      const material = visualMaterial(asset, texture, quality);
      const mesh = new THREE.Mesh(geometry, material);
      mesh.castShadow = dimensions[2] > .01;
      mesh.receiveShadow = true;
      mesh.userData.robot = asset.key.startsWith(`${current.current.robot_body_id}:`);
      if (mesh.userData.robot) robotDetails.push(detailRobotVisual(mesh, asset));
      mesh.position.fromArray(asset.position);
      mesh.quaternion.fromArray(asset.quaternion);
      let link = links.get(asset.key);
      if (!link) {
        link = new THREE.Group();
        links.set(asset.key, link);
        (mesh.userData.robot ? robotRoot : world).add(link);
      }
      link.add(mesh);
      meshes.push(mesh);
    }
    const outline = new THREE.BoxHelper(robotRoot, new THREE.Color(getComputedStyle(document.documentElement).getPropertyValue('--cp-accent').trim()));
    outline.visible = false;
    scene.add(outline);
    const raycaster = new THREE.Raycaster();
    const ground = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
    let drag: { pointer: number | null; start: THREE.Vector3; screen: THREE.Vector2; source: ManualPlacement; moved: boolean } | null = null;
    function cast(event: PointerEvent) {
      const bounds = renderer.domElement.getBoundingClientRect();
      raycaster.setFromCamera(new THREE.Vector2((event.clientX - bounds.left) / bounds.width * 2 - 1, -(event.clientY - bounds.top) / bounds.height * 2 + 1), camera);
    }
    function floorPoint(event: PointerEvent) {
      cast(event);
      const point = raycaster.ray.intersectPlane(ground, new THREE.Vector3());
      return point ? world.worldToLocal(point) : null;
    }
    function select() {
      if (!options.current.enabled) return;
      isSelected = true;
      setSelected(true);
      setError('');
      renderer.domElement.focus({ preventScroll: true });
    }
    function releasePointer() {
      const pointer = drag?.pointer;
      if (drag) drag.pointer = null;
      if (pointer != null && renderer.domElement.hasPointerCapture(pointer)) renderer.domElement.releasePointerCapture(pointer);
      controls.enabled = true;
    }
    function cancel() {
      releasePointer();
      drag = null;
      robotRoot.position.set(0, 0, 0);
      isSelected = false;
      setSelected(false);
      setPreview(null);
      renderer.domElement.style.cursor = '';
    }
    interaction.current = { select, cancel };
    function begin(pointer: number | null, start = new THREE.Vector3(), screen = new THREE.Vector2()) {
      const latest = current.current;
      const base = latest.snapshot.poses.find(pose => pose.key === `${latest.robot_body_id}:-1`)!;
      drag = { pointer, start, screen, moved: false, source: { run_id: latest.run_id, episode_epoch: latest.episode_epoch,
        observation_seq: latest.observation.seq, xy_m: [base.position[0], base.position[1]] } };
      controls.enabled = false;
    }
    function updatePreview() {
      if (!drag) return;
      drag.moved = true;
      setPreview([drag.source.xy_m[0] + robotRoot.position.x, drag.source.xy_m[1] + robotRoot.position.y]);
    }
    async function commit() {
      if (!drag?.moved || !options.current.enabled) return;
      const placement: ManualPlacement = { ...drag.source, xy_m: [drag.source.xy_m[0] + robotRoot.position.x, drag.source.xy_m[1] + robotRoot.position.y] };
      releasePointer();
      drag = null;
      setPlacing(true);
      try {
        await options.current.onPlace(placement);
        if (active) setError('');
      } catch (failure) {
        if (active) setError(failure instanceof Error ? failure.message : 'Placement failed.');
      } finally {
        if (active) { cancel(); setPlacing(false); }
      }
    }
    function pointerDown(event: PointerEvent) {
      if (drag && !event.isPrimary) { cancel(); event.stopImmediatePropagation(); return; }
      if (!options.current.enabled || event.button !== 0 || !event.isPrimary) return;
      cast(event);
      if (!raycaster.intersectObjects(meshes)[0]?.object.userData.robot) return;
      const point = floorPoint(event);
      if (!point) return;
      event.preventDefault();
      event.stopImmediatePropagation();
      cancel();
      select();
      begin(event.pointerId, point, new THREE.Vector2(event.clientX, event.clientY));
      renderer.domElement.setPointerCapture(event.pointerId);
    }
    function pointerMove(event: PointerEvent) {
      if(!drag && zoneOptions.current.showZones && movementZones.group.visible) {
        cast(event);
        const hit=raycaster.intersectObjects([...movementZones.meshes.values()])[0];
        setZoneHint(hit?.object.userData.hint ?? '');
      }
      if (drag?.pointer === event.pointerId) {
        event.stopImmediatePropagation();
        if (new THREE.Vector2(event.clientX, event.clientY).distanceTo(drag.screen) < 4 && !drag.moved) return;
        const point = floorPoint(event);
        if (!point) return;
        robotRoot.position.copy(point.sub(drag.start));
        robotRoot.position.z = 0;
        renderer.domElement.style.cursor = 'grabbing';
        updatePreview();
      } else if (!drag && options.current.enabled) {
        cast(event);
        renderer.domElement.style.cursor = raycaster.intersectObjects(meshes)[0]?.object.userData.robot ? 'grab' : '';
      }
    }
    function pointerUp(event: PointerEvent) {
      if (drag?.pointer !== event.pointerId) return;
      event.stopImmediatePropagation();
      releasePointer();
      if (drag?.moved) void commit();
      else drag = null;
    }
    function pointerCancel(event: PointerEvent) { if (drag?.pointer === event.pointerId) cancel(); }
    function keyDown(event: KeyboardEvent) {
      if (event.key === 'Escape') { event.preventDefault(); cancel(); return; }
      if (!isSelected || !options.current.enabled) return;
      if (event.key === 'Enter') { event.preventDefault(); void commit(); return; }
      const offsets: Record<string, [number, number]> = { ArrowLeft: [-.1, 0], ArrowRight: [.1, 0], ArrowUp: [0, .1], ArrowDown: [0, -.1] };
      const offset = offsets[event.key];
      if (!offset) return;
      event.preventDefault();
      if (!drag) begin(null);
      robotRoot.position.x += offset[0];
      robotRoot.position.y += offset[1];
      updatePreview();
    }
    renderer.domElement.addEventListener('pointerdown', pointerDown, true);
    renderer.domElement.addEventListener('pointermove', pointerMove, true);
    renderer.domElement.addEventListener('pointerup', pointerUp, true);
    renderer.domElement.addEventListener('pointercancel', pointerCancel);
    renderer.domElement.addEventListener('lostpointercapture', pointerCancel);
    renderer.domElement.addEventListener('keydown', keyDown);
    const resize = new ResizeObserver(() => {
      const width = element.clientWidth, height = element.clientHeight;
      if (width && height) {
        camera.aspect = width / height;
        if (facility && !cameraAdjusted) {
          const tangent = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
          const distance = Math.max((floorDepth / 2 + 1) / tangent, (floorWidth / 2 + .35) / (tangent * camera.aspect)) + floorDepth * .2;
          camera.position.copy(controls.target).add(new THREE.Vector3(0, 1, .42).normalize().multiplyScalar(distance));
          camera.lookAt(controls.target);
        } else if (!cameraAdjusted) {
          camera.position.copy(controls.target).add(initialCameraOffset.clone().multiplyScalar(1 / Math.min(1, camera.aspect)));
          camera.lookAt(controls.target);
        }
        camera.updateProjectionMatrix();
        renderer.setSize(width, height);
        trailMaterial.resolution.set(width,height);
      }
    });
    resize.observe(element);
    let frameId = 0;
    let placements = current.current.manual_placements;
    let previousZoneState='';
    let previousBeamState='';
    const animate = () => {
      if (drag && (!options.current.enabled || drag.source.observation_seq !== current.current.observation.seq)) cancel();
      const positioned = placements !== current.current.manual_placements;
      placements = current.current.manual_placements;
      for (const pose of current.current.snapshot.poses) {
        const link = links.get(pose.key);
        if (link) {
          if (positioned) link.position.fromArray(pose.position);
          else link.position.lerp(new THREE.Vector3(...pose.position), .35);
          link.quaternion.slerp(new THREE.Quaternion(...pose.quaternion), .35);
        }
      }
      axisHelper.visible = axesVisible.current;
      const history = travelledPath.current;
      if (trailRevision !== history.revision) {
        trailRevision = history.revision;
        trailPositions.set(history.positions);
        trailBuffer.needsUpdate = true;
        trailGeometry.instanceCount = history.positions.length/6;
      }
      trail.visible = trailVisible.current && history.positions.length > 0;
      renderer.domElement.dataset.trailSegments = String(history.positions.length/6);
      renderer.domElement.dataset.trailVisible = String(trail.visible);
      renderer.domElement.dataset.trailEnd = JSON.stringify(history.positions.slice(-3));
      const zoneState=movementZones.update(current.current,zoneOptions.current.telemetry,zoneOptions.current.showZones && !drag,zoneOptions.current.connected);
      if(previousZoneState!==zoneState) {previousZoneState=zoneState;setZoneStatus(zoneState);}
      if(previousBeamState!==movementZones.beamStatus) {previousBeamState=movementZones.beamStatus;setBeamStatus(movementZones.beamStatus);}
      renderer.domElement.dataset.zoneState=zoneState;
      renderer.domElement.dataset.zoneCount=String(movementZones.group.visible && movementZones.sampledAreas.visible ? movementZones.meshes.size : 0);
      renderer.domElement.dataset.zoneOrigin=JSON.stringify([...movementZones.sampledAreas.position.toArray().slice(0,2),movementZones.sampledAreas.rotation.z]);
      renderer.domElement.dataset.beamState=movementZones.beamStatus;
      renderer.domElement.dataset.beamOrigin=JSON.stringify([...movementZones.liveSensors.position.toArray().slice(0,2),movementZones.liveSensors.rotation.z]);
      renderer.domElement.dataset.coloredZones=String(movementZones.group.visible && movementZones.sampledAreas.visible
        ? [...movementZones.meshes.values()].filter(mesh=>['clear','restricted'].includes(mesh.userData.level)).length : 0);
      outline.visible = isSelected;
      if (isSelected) outline.update();
      if (!drag) controls.update();
      renderer.render(scene, camera);
      frameId = requestAnimationFrame(animate);
    };
    animate();
    return () => {
      active = false;
      savedView.current = { run: state.run_id, position: camera.position.clone(), target: controls.target.clone() };
      interaction.current = { select: () => {}, cancel: () => {} };
      cancelAnimationFrame(frameId);
      releasePointer();
      renderer.domElement.removeEventListener('pointerdown', pointerDown, true);
      renderer.domElement.removeEventListener('pointermove', pointerMove, true);
      renderer.domElement.removeEventListener('pointerup', pointerUp, true);
      renderer.domElement.removeEventListener('pointercancel', pointerCancel);
      renderer.domElement.removeEventListener('lostpointercapture', pointerCancel);
      renderer.domElement.removeEventListener('keydown', keyDown);
      resize.disconnect();
      controls.dispose();
      movementZones.dispose();
      trailGeometry.dispose();
      trailMaterial.dispose();
      outline.geometry.dispose();
      (outline.material as THREE.Material).dispose();
      robotDetails.forEach(dispose => dispose());
      for (const mesh of meshes) {
        mesh.geometry.dispose();
        (mesh.material as THREE.Material).dispose();
      }
      for (const texture of textures.values()) texture.dispose();
      disposeLighting();
      sun.shadow.map?.dispose();
      axisHelper.dispose();
      renderer.dispose();
      element.replaceChildren();
    };
  }, [state.run_id, quality]);
  return <div className="spectator-shell">
    <div className="spectator-stage">
    <div className="spectator" ref={host} />
    {showZones && <div className="movement-zone-legend" aria-label="Movement zone legend">
      <div>{[['clear','Map-clear'],['restricted','Restricted'],['unknown','Unknown'],['unavailable','Unavailable']].map(([kind,label])=><span key={kind}><i data-zone={kind}/>{label}</span>)}</div>
      <strong>Areas: {zoneStatus} / geometry only</strong>
      <span>Beams: {beamStatus} / red hit, blue no return</span>
      {telemetry?.motion_zones?.display_pose && <span>Areas and stop ticks at sampled pose</span>}
      {telemetry?.motion_zones && <span>Map radius {(telemetry.motion_zones.planning_radius_m+(telemetry.motion_zones.map_margin_m ?? 0)).toFixed(2)} m / includes footprint margin</span>}
      <span>{telemetry?.motion_zones ? `Footprint ${telemetry.motion_zones.footprint.radius_m.toFixed(2)} m / beam stop ${telemetry.motion_zones.beam_stop_distance_m.toFixed(2)} m at ${telemetry.motion_zones.preview_speed_mps.toFixed(2)} m/s${telemetry.motion_zones.speed_basis==='idle_reference' ? ' reference' : ''}` : 'Zone telemetry unavailable'}</span>
      {state.navigation?.diagnostics?.stop && <span>Stop: {state.navigation.diagnostics.stop.reason.split(':')[0]}</span>}
      {zoneHint && <span className="movement-zone-hint">{zoneHint}</span>}
    </div>}
    <div className="placement-toolbar">
      <button className="icon-button" aria-label="Select robot" aria-pressed={selected} disabled={!enabled || placing}
        title={enabled ? selected ? 'Deselect robot' : 'Select robot to reposition' : 'Manual control is required to reposition'}
        onClick={() => selected ? interaction.current.cancel() : interaction.current.select()}><MousePointer2 size={18} /></button>
      <span role="status">{placing ? 'Applying position' : preview ? 'Placement preview' : selected ? 'Milo selected' : ''}</span>
      {preview && <span className="placement-coordinates">X {preview[0].toFixed(2)} / Y {preview[1].toFixed(2)} m</span>}
    </div>
    {state.rendering !== 'enhanced' && <div className="graphics-quality" role="group" aria-label="Spectator graphics">
      {(['standard', 'enhanced'] as const).map(mode => <button key={mode} type="button" aria-pressed={quality === mode}
        title={`${mode === 'standard' ? 'Standard' : 'Enhanced'} spectator graphics; robot camera unchanged`}
        onClick={() => setQuality(mode)}>
        {mode === 'standard' ? 'Standard' : 'Enhanced'}</button>)}
    </div>}
    {error && <div className="placement-error" role="alert">{error}</div>}
    </div>
    {showZones && telemetry?.motion_zones && <details className="movement-zone-readings"><summary>Zone readings / center travel</summary>
      <p>{telemetry.motion_zones.source} / {telemetry.motion_zones.reason}</p>
      <table><thead><tr><th>Direction</th><th>0.25 m</th><th>0.50 m</th><th>1.00 m</th></tr></thead><tbody>
        {Array.from({length:16},(_,sector)=><tr key={sector}><th>{(sector*22.5).toFixed(1)} deg</th>
          {telemetry.motion_zones!.sectors.filter(zone=>zone.sector===sector).map(zone=><td key={zone.outer_m} title={zone.reason}>{zoneStatus==='Observed' ? zone.status : zoneStatus}</td>)}
        </tr>)}
      </tbody></table></details>}
  </div>;
}