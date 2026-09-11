import { useEffect, useRef, useState } from 'react';
import { MousePointer2 } from 'lucide-react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import type { LiveState, ManualPlacement } from './types';

export function Spectator({ state, axes, enabled, onPlace }: { state: LiveState; axes: boolean; enabled: boolean; onPlace: (placement: ManualPlacement) => Promise<void> }) {
  const host = useRef<HTMLDivElement>(null);
  const current = useRef(state);
  const axesVisible = useRef(axes);
  const options = useRef({ enabled, onPlace });
  const interaction = useRef({ select: () => {}, cancel: () => {} });
  const [selected, setSelected] = useState(false);
  const [preview, setPreview] = useState<[number, number] | null>(null);
  const [placing, setPlacing] = useState(false);
  const [error, setError] = useState('');
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
    const recharging = current.current.challenge?.id === 'recharge';
    const apartment = ['apartment', 'kitchen_bathroom'].includes(current.current.challenge?.id ?? '');
    const facility = ['clinic_delivery', 'warehouse', 'inspection'].includes(current.current.challenge?.id ?? '');
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
    controls.enableDamping = true;
    controls.minDistance = .45;
    controls.maxDistance = facility ? 45 : apartment ? 12 : 8;
    let cameraAdjusted = false;
    controls.addEventListener('start', () => { cameraAdjusted = true; });
    controls.maxPolarAngle = Math.PI / 2 - .02;
    scene.add(new THREE.HemisphereLight(0xffffff, 0x777777, 2));
    const sun = new THREE.DirectionalLight(0xffffff, 2.4);
    sun.position.set(2, 5, 3);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    const shadowExtent = Math.max(4, floorWidth / 2 + 1, floorDepth / 2 + 1);
    Object.assign(sun.shadow.camera, { left: -shadowExtent, right: shadowExtent, top: shadowExtent, bottom: -shadowExtent, near: .1, far: 30 });
    sun.shadow.normalBias = .015;
    scene.add(sun);
    const world = new THREE.Group();
    world.rotation.x = -Math.PI / 2;
    scene.add(world);
    const robotRoot = new THREE.Group();
    world.add(robotRoot);
    const axisHelper = new THREE.AxesHelper(.7);
    world.add(axisHelper);
    const links = new Map<string, THREE.Group>();
    const meshes: THREE.Mesh[] = [];
    const textures = new Map<string, THREE.Texture>();
    const textureLoader = new THREE.TextureLoader();
    for (const asset of current.current.geometry) {
      const dimensions = asset.dimensions;
      const geometry = asset.type === 2 ? new THREE.SphereGeometry(dimensions[0], 24, 16)
        : asset.type === 4 ? new THREE.CylinderGeometry(dimensions[1], dimensions[1], dimensions[0], 32)
        : new THREE.BoxGeometry(dimensions[0], dimensions[1], dimensions[2]);
      if (asset.type === 4) geometry.rotateX(Math.PI / 2);
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
      const material = new THREE.MeshStandardMaterial({ color: new THREE.Color(...asset.color.slice(0, 3) as [number, number, number]),
        map: texture, roughness: texture ? .9 : .65, metalness: texture ? 0 : .12 });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.castShadow = dimensions[2] > .01;
      mesh.receiveShadow = true;
      mesh.userData.robot = asset.key.startsWith(`${current.current.robot_body_id}:`);
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
        }
        camera.updateProjectionMatrix();
        renderer.setSize(width, height);
      }
    });
    resize.observe(element);
    let frameId = 0;
    let placements = current.current.manual_placements;
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
      outline.visible = isSelected;
      if (isSelected) outline.update();
      if (!drag) controls.update();
      renderer.render(scene, camera);
      frameId = requestAnimationFrame(animate);
    };
    animate();
    return () => {
      active = false;
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
      outline.geometry.dispose();
      (outline.material as THREE.Material).dispose();
      for (const mesh of meshes) {
        mesh.geometry.dispose();
        (mesh.material as THREE.Material).dispose();
      }
      for (const texture of textures.values()) texture.dispose();
      axisHelper.dispose();
      renderer.dispose();
      element.replaceChildren();
    };
  }, [state.run_id]);
  return <div className="spectator-shell">
    <div className="spectator" ref={host} />
    <div className="placement-toolbar">
      <button className="icon-button" aria-label="Select robot" aria-pressed={selected} disabled={!enabled || placing}
        title={enabled ? selected ? 'Deselect robot' : 'Select robot to reposition' : 'Manual control is required to reposition'}
        onClick={() => selected ? interaction.current.cancel() : interaction.current.select()}><MousePointer2 size={18} /></button>
      <span role="status">{placing ? 'Applying position' : preview ? 'Placement preview' : selected ? 'Milo selected' : ''}</span>
      {preview && <span className="placement-coordinates">X {preview[0].toFixed(2)} / Y {preview[1].toFixed(2)} m</span>}
    </div>
    {error && <div className="placement-error" role="alert">{error}</div>}
  </div>;
}