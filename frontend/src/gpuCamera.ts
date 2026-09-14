import * as THREE from 'three';
import { enhancedLighting, visualGeometry, visualMaterial, type GraphicsQuality } from './sceneGraphics';
import type { Geometry, Pose, Vec3 } from './types';

export type RenderPacket = {
  schema: 'milo-render-v1'; run_id: string; episode_epoch: number; observation_seq: number; robot_body_id: number;
  geometry: Geometry[]; snapshot: { simulated_time_s: number; poses: Pose[] };
  camera: { eye: Vec3; target: Vec3; up: Vec3; vertical_fov_deg: number; near_m: number; far_m: number; width: number; height: number };
};

function validate(packet: RenderPacket) {
  const camera = packet.camera;
  if (packet.schema !== 'milo-render-v1' || !packet.run_id ||
    !Number.isInteger(packet.episode_epoch) || !Number.isInteger(packet.observation_seq) ||
    !Number.isFinite(packet.snapshot.simulated_time_s) ||
    ![camera.width, camera.height].every(value => Number.isInteger(value) && value > 0 && value <= 2048) ||
    ![...camera.eye, ...camera.target, ...camera.up, camera.near_m, camera.far_m, camera.vertical_fov_deg].every(Number.isFinite) ||
    camera.near_m <= 0 || camera.far_m <= camera.near_m || camera.far_m > 100 || camera.vertical_fov_deg <= 0 || camera.vertical_fov_deg >= 179 ||
    new THREE.Vector3(...camera.eye).distanceTo(new THREE.Vector3(...camera.target)) < 1e-6 || new THREE.Vector3(...camera.up).length() < 1e-6) {
    throw new Error('Invalid calibrated render packet');
  }
  const keys = new Set<string>();
  for (const pose of packet.snapshot.poses) {
    if (keys.has(pose.key) || ![...pose.position, ...pose.quaternion].every(Number.isFinite)) throw new Error('Invalid snapshot poses');
    keys.add(pose.key);
  }
  if (packet.geometry.some(asset => !keys.has(asset.key) || ![2, 3, 4].includes(asset.type) ||
    ![...asset.dimensions, ...asset.color, ...asset.position, ...asset.quaternion].every(Number.isFinite))) throw new Error('Invalid snapshot geometry');
}

function base64(bytes: Uint8Array) {
  let binary = '';
  for (let offset = 0; offset < bytes.length; offset += 8192) binary += String.fromCharCode(...bytes.subarray(offset, offset + 8192));
  return btoa(binary);
}

export function installGpuCamera() {
  const renderer = new THREE.WebGLRenderer({ antialias: false, preserveDrawingBuffer: true, alpha: false });
  renderer.setPixelRatio(1);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.domElement.setAttribute('aria-label', 'Snapshot robot head camera');
  document.getElementById('root')!.replaceChildren(renderer.domElement);
  const context = renderer.getContext();
  const debug = context.getExtension('WEBGL_debug_renderer_info');
  const device = String(context.getParameter(debug?.UNMASKED_RENDERER_WEBGL ?? context.RENDERER));
  const rotation = new THREE.Matrix4().makeRotationX(-Math.PI / 2);
  let scene = new THREE.Scene();
  let links = new Map<string, THREE.Group>();
  let meshes: { mesh: THREE.Mesh; rgb: THREE.Material; depth: THREE.ShaderMaterial }[] = [];
  let textures: THREE.Texture[] = [];
  let disposeLighting = () => {};
  let source: { run: string; epoch: number; robot: number } | null = null;
  let depthTarget: THREE.WebGLRenderTarget | null = null;
  let loading = false;
  let closed = false;

  function clear() {
    source = null;
    disposeLighting();
    disposeLighting = () => {};
    for (const item of meshes) { item.mesh.geometry.dispose(); item.rgb.dispose(); item.depth.dispose(); }
    for (const texture of textures) texture.dispose();
    meshes = [];
    textures = [];
    links = new Map();
    depthTarget?.dispose();
    depthTarget = null;
    scene = new THREE.Scene();
  }

  async function load(packet: RenderPacket, quality: GraphicsQuality = 'enhanced', lightingSeed = 0) {
    if (closed || loading) throw new Error('Camera worker unavailable');
    validate(packet);
    if (!['standard', 'enhanced'].includes(quality)) throw new Error('Unknown graphics quality');
    if (!Number.isInteger(lightingSeed) || lightingSeed < 0 || lightingSeed > 2147483647) throw new Error('Invalid lighting seed');
    loading = true;
    clear();
    try {
      const loader = new THREE.TextureLoader();
      const loaded = new Map<string, THREE.Texture>();
      for (const path of new Set(packet.geometry.map(asset => asset.texture).filter((path): path is string => !!path))) {
        if (!/^\/api\/textures\/[a-z]+\.png$/.test(path)) throw new Error('Only local scene textures are allowed');
        const texture = await new Promise<THREE.Texture>((resolve, reject) => {
          let expired = false;
          const deadline = window.setTimeout(() => { expired = true; reject(new Error(`Texture load timed out: ${path}`)); }, 10000);
          loader.load(path, value => {
            clearTimeout(deadline);
            if (expired || closed) { value.dispose(); reject(new Error('Camera worker closed')); }
            else resolve(value);
          }, undefined, error => { clearTimeout(deadline); reject(error); });
        });
        texture.colorSpace = THREE.SRGBColorSpace;
        texture.anisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy());
        textures.push(texture);
        loaded.set(path, texture);
      }
      const world = new THREE.Group();
      world.rotation.x = -Math.PI / 2;
      scene.add(world);
      scene.background = new THREE.Color(0xffffff);
      renderer.toneMapping = THREE.NoToneMapping;
      if (quality === 'enhanced') {
        const extent = Math.max(4, ...packet.geometry.filter(asset => asset.name === 'floor').flatMap(asset => asset.dimensions.slice(0, 2).map(value => value / 2 + 1)));
        disposeLighting = enhancedLighting(scene, renderer, extent, lightingSeed);
      } else {
        scene.add(new THREE.HemisphereLight(0xffffff, 0x777777, 2));
        const sun = new THREE.DirectionalLight(0xffffff, 2.4);
        sun.position.set(2, 5, 3);
        scene.add(sun);
      }
      for (const asset of packet.geometry) {
        const rgb = visualMaterial(asset, asset.texture ? loaded.get(asset.texture) : undefined, quality);
        const mesh = new THREE.Mesh(visualGeometry(asset, quality), rgb);
        mesh.position.fromArray(asset.position);
        mesh.quaternion.fromArray(asset.quaternion);
        mesh.castShadow = asset.dimensions[2] > .01;
        mesh.receiveShadow = true;
        const depth = new THREE.ShaderMaterial({
          uniforms: { self: { value: asset.key.startsWith(`${packet.robot_body_id}:`) ? 0 : 1 } },
          vertexShader: 'varying float axial; void main() { vec4 view = modelViewMatrix * vec4(position, 1.0); axial = -view.z; gl_Position = projectionMatrix * view; }',
          fragmentShader: 'varying float axial; uniform float self; void main() { float code = floor(axial * 100000.0 + 0.5); gl_FragColor = vec4(floor(code / 65536.0), floor(mod(code, 65536.0) / 256.0), mod(code, 256.0), self * 255.0) / 255.0; }',
          toneMapped: false,
        });
        let link = links.get(asset.key);
        if (!link) { link = new THREE.Group(); links.set(asset.key, link); world.add(link); }
        link.add(mesh);
        meshes.push({ mesh, rgb, depth });
      }
      source = { run: packet.run_id, epoch: packet.episode_epoch, robot: packet.robot_body_id };
      return { renderer: device, hardware: !/swiftshader|llvmpipe|software|basic render/i.test(device), quality,
        textures: textures.length, shapes: meshes.length, three_revision: THREE.REVISION, lighting_seed: lightingSeed };
    } catch (error) { clear(); throw error; }
    finally { loading = false; }
  }

  function capture(packet: RenderPacket, includeDepth = true) {
    if (closed || loading || !source) throw new Error('Camera worker is not ready');
    validate(packet);
    if (packet.run_id !== source.run || packet.episode_epoch !== source.epoch || packet.robot_body_id !== source.robot) throw new Error('Stale render episode');
    const started = performance.now();
    for (const pose of packet.snapshot.poses) {
      const link = links.get(pose.key);
      if (link) { link.position.fromArray(pose.position); link.quaternion.fromArray(pose.quaternion); }
    }
    const calibration = packet.camera;
    const { width, height } = calibration;
    const camera = new THREE.PerspectiveCamera(calibration.vertical_fov_deg, width / height, calibration.near_m, calibration.far_m);
    camera.position.fromArray(calibration.eye).applyMatrix4(rotation);
    camera.up.fromArray(calibration.up).transformDirection(rotation);
    camera.lookAt(new THREE.Vector3(...calibration.target).applyMatrix4(rotation));
    renderer.setSize(width, height, false);
    renderer.render(scene, camera);
    const rgb = renderer.domElement.toDataURL('image/png').split(',')[1];
    const result = { run_id: packet.run_id, episode_epoch: packet.episode_epoch, observation_seq: packet.observation_seq,
      simulated_time_s: packet.snapshot.simulated_time_s, width, height, rgb, depth_f32: '',
      depth_encoding: 'float32-le-axial-metres-nan-invalid', elapsed_ms: performance.now() - started };
    if (!includeDepth) return result;
    if (!depthTarget) depthTarget = new THREE.WebGLRenderTarget(width, height, { minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter });
    else depthTarget.setSize(width, height);
    const packed = new Uint8Array(width * height * 4);
    const background = scene.background;
    try {
      scene.background = new THREE.Color(0);
      for (const item of meshes) item.mesh.material = item.depth;
      renderer.setRenderTarget(depthTarget);
      renderer.render(scene, camera);
      renderer.readRenderTargetPixels(depthTarget, 0, 0, width, height, packed);
    } finally {
      renderer.setRenderTarget(null);
      scene.background = background;
      for (const item of meshes) item.mesh.material = item.rgb;
    }
    const depth = new Float32Array(width * height);
    const self = new Uint8Array(width * height);
    for (let row = 0; row < height; row++) for (let column = 0; column < width; column++) {
      const destination = row * width + column;
      const offset = ((height - 1 - row) * width + column) * 4;
      const value = (packed[offset] * 65536 + packed[offset + 1] * 256 + packed[offset + 2]) / 100000;
      self[destination] = packed[offset + 3] === 0 ? 1 : 0;
      depth[destination] = value > 0 && value < calibration.far_m ? value : NaN;
    }
    for (let row = 0; row < height; row++) for (let column = 0; column < width; column++) {
      if (!self[row * width + column]) continue;
      for (const [horizontal, vertical] of [[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]]) {
        const adjacentRow = row + vertical, adjacentColumn = column + horizontal;
        if (adjacentRow >= 0 && adjacentRow < height && adjacentColumn >= 0 && adjacentColumn < width) depth[adjacentRow * width + adjacentColumn] = NaN;
      }
    }
    return { ...result, depth_f32: base64(new Uint8Array(depth.buffer)), elapsed_ms: performance.now() - started };
  }

  function dispose() { closed = true; clear(); renderer.dispose(); renderer.domElement.remove(); }
  window.miloGpuCamera = { load, capture, dispose };
}

declare global {
  interface Window { miloGpuCamera: { load: (packet: RenderPacket, quality?: GraphicsQuality, lightingSeed?: number) => Promise<Record<string, unknown>>;
    capture: (packet: RenderPacket, includeDepth?: boolean) => { run_id: string; episode_epoch: number; observation_seq: number; simulated_time_s: number;
      width: number; height: number; rgb: string; depth_f32: string; depth_encoding: string; elapsed_ms: number }; dispose: () => void } }
}