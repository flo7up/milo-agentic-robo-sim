import * as THREE from 'three';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js';
import type { Geometry } from './types';

export type GraphicsQuality = 'standard' | 'enhanced';

export function visualGeometry(asset: Geometry, quality: GraphicsQuality) {
  const dimensions = asset.dimensions;
  if (asset.type === 2) return new THREE.SphereGeometry(dimensions[0], quality === 'enhanced' ? 32 : 24, quality === 'enhanced' ? 24 : 16);
  if (asset.type === 4) {
    const geometry = new THREE.CylinderGeometry(dimensions[1], dimensions[1], dimensions[0], quality === 'enhanced' ? 48 : 32);
    geometry.rotateX(Math.PI / 2);
    return geometry;
  }
  const rounded = quality === 'enhanced' && asset.name !== 'robot' &&
    /fridge|stove_body|counter|cabinet|cistern|toilet_foot|tub|vanity|rim|handle/.test(asset.name ?? '');
  return rounded ? new RoundedBoxGeometry(dimensions[0], dimensions[1], dimensions[2], 2, Math.min(.012, Math.min(...dimensions) * .16))
    : new THREE.BoxGeometry(dimensions[0], dimensions[1], dimensions[2]);
}

export function visualMaterial(asset: Geometry, texture: THREE.Texture | undefined, quality: GraphicsQuality) {
  const color = new THREE.Color(...asset.color.slice(0, 3) as [number, number, number]);
  if (quality === 'standard') return new THREE.MeshStandardMaterial({ color, map: texture,
    roughness: texture ? .9 : .65, metalness: texture ? 0 : .12 });
  const name = asset.name ?? '';
  const metal = /faucet|spout|tap_|handle|flush|sink_rim|burner/.test(name);
  const glass = /glass|hob|oven_window/.test(name);
  const ceramic = /toilet|bathtub|basin|sink|fridge|stove_body/.test(name) && !metal;
  color.convertSRGBToLinear();
  return new THREE.MeshPhysicalMaterial({ color, map: texture,
    roughness: metal ? .26 : glass ? .16 : ceramic ? .28 : texture ? .84 : .62,
    metalness: metal ? .82 : glass ? .28 : 0,
    clearcoat: ceramic || glass ? .35 : 0, clearcoatRoughness: .24,
    bumpMap: texture, bumpScale: texture ? .002 : 0,
    envMapIntensity: metal || glass ? 1 : .45 });
}

export function detailRobotVisual(mesh: THREE.Mesh, asset: Geometry) {
  const [width, depth, height] = asset.dimensions;
  const near = (value: number, expected: number) => Math.abs(value - expected) < .0001;
  const pearl = new THREE.MeshPhysicalMaterial({ color: 0xe8eeec, roughness: .3, metalness: .12, clearcoat: .4 });
  const graphite = new THREE.MeshStandardMaterial({ color: 0x293237, roughness: .6, metalness: .25 });
  const alloy = new THREE.MeshStandardMaterial({ color: 0xaebbc0, roughness: .3, metalness: .8 });
  const rose = new THREE.MeshPhysicalMaterial({ color: 0xb82e53, roughness: .34, metalness: .16, clearcoat: .35 });
  const optical = new THREE.MeshPhysicalMaterial({ color: 0x85dce3, roughness: .13, metalness: .4,
    emissive: 0x32818a, emissiveIntensity: .35, clearcoat: 1 });
  const glass = new THREE.MeshPhysicalMaterial({ color: 0x101d24, roughness: .18, metalness: .3, clearcoat: 1 });
  const materials = [pearl, graphite, alloy, rose, optical, glass];
  const add = (geometry: THREE.BufferGeometry, material: THREE.Material, position: [number, number, number], name: string) => {
    const detail = new THREE.Mesh(geometry, material);
    detail.position.set(...position);
    detail.name = `milo-${name}`;
    detail.castShadow = true;
    detail.receiveShadow = true;
    detail.userData.robot = true;
    mesh.add(detail);
    return detail;
  };
  const panel = (size: [number, number, number], position: [number, number, number], material: THREE.Material, name: string) =>
    add(new RoundedBoxGeometry(...size, 2, Math.min(...size) * .22), material, position, name);
  const disc = (radius: number, length: number, position: [number, number, number], material: THREE.Material, name: string, axis: 'x' | 'z' = 'x') => {
    const geometry = new THREE.CylinderGeometry(radius, radius, length, 24);
    if (axis === 'x') geometry.rotateZ(Math.PI / 2);
    else geometry.rotateX(Math.PI / 2);
    return add(geometry, material, position, name);
  };
  if (asset.type === 3) {
    mesh.geometry.dispose();
    mesh.geometry = new RoundedBoxGeometry(width, depth, height, 3, Math.min(.016, Math.min(width, depth, height) * .18));
  }
  (mesh.material as THREE.Material).dispose();
  mesh.material = asset.color[0] > .8 ? pearl : asset.color[0] > .5 ? rose : graphite;
  if (asset.type === 3 && near(width, .34) && near(depth, .31)) {
    mesh.name = 'milo-chassis';
    panel([.336, .306, .037], [0, 0, -.045], graphite, 'bumper');
    panel([.27, .252, .009], [-.012, 0, .079], alloy, 'deck');
    panel([.009, .178, .048], [.17, 0, .013], graphite, 'front-panel');
    for (const side of [-1, 1]) {
      panel([.009, .035, .009], [.176, side * .062, .022], optical, 'running-light');
      panel([.006, .042, .012], [-.17, side * .098, .015], rose, 'rear-reflector');
      for (const offset of [-.09, -.045, 0, .045, .09]) {
        panel([.018, .004, .025], [offset, side * .155, .021], graphite, 'chassis-vent');
      }
      for (const offset of [-.115, .105]) {
        disc(.006, .003, [offset, side * .105, .085], graphite, 'deck-fastener', 'z');
      }
    }
  } else if (asset.type === 3 && near(width, .14) && near(height, .24)) {
    mesh.name = 'milo-torso';
    panel([.007, .125, .172], [.069, 0, .008], pearl, 'chest-panel');
    panel([.009, .066, .046], [.074, 0, .045], graphite, 'chest-badge');
    for (const offset of [-.022, 0, .022]) {
      panel([.003, .009, .018], [.080, offset, .045], alloy, 'badge-bar');
    }
    panel([.009, .048, .006], [.075, 0, -.058], optical, 'chest-accent');
    panel([.009, .106, .148], [-.069, 0, 0], graphite, 'service-panel');
    for (const offset of [-.042, -.021, 0, .021, .042]) {
      panel([.003, .074, .004], [-.075, 0, offset], alloy, 'rear-vent');
    }
  } else if (asset.type === 3 && near(width, .16) && near(depth, .22)) {
    mesh.name = 'milo-head-shell';
    panel([.095, .152, .004], [-.014, 0, .064], alloy, 'head-crown');
    for (const side of [-1, 1]) {
      const pivot = disc(.028, .007, [0, side * .109, 0], graphite, 'head-pivot', 'z');
      pivot.rotation.x = Math.PI / 2;
      const cap = disc(.018, .009, [0, side * .111, 0], alloy, 'head-pivot-cap', 'z');
      cap.rotation.x = Math.PI / 2;
      panel([.035, .004, .005], [-.025, side * .11, -.042], rose, 'temple-accent');
    }
    panel([.004, .063, .004], [.079, 0, -.045], graphite, 'speaker-slot');
  } else if (asset.type === 3 && near(width, .012) && near(depth, .175)) {
    mesh.name = 'milo-visor';
    mesh.material = glass;
    for (const side of [-1, 1]) {
      disc(.024, .003, [.006, side * .048, .003], alloy, 'lens-bezel');
      disc(.020, .004, [.008, side * .048, .003], optical, 'lens');
      disc(.010, .004, [.0105, side * .048, .003], glass, 'lens-pupil');
      disc(.0035, .002, [.013, side * .048 - .005, .010], pearl, 'lens-glint');
    }
    panel([.002, .018, .003], [.007, 0, -.019], alloy, 'visor-bridge');
  } else if (asset.type === 4 && near(width, .045) && near(depth, .09)) {
    mesh.name = 'milo-wheel';
    mesh.material = graphite;
    for (const side of [-1, 1]) {
      disc(.058, .003, [0, 0, side * .023], alloy, 'wheel-rim', 'z');
      disc(.045, .004, [0, 0, side * .025], graphite, 'wheel-recess', 'z');
      disc(.018, .006, [0, 0, side * .027], rose, 'hub-cap', 'z');
      add(new THREE.TorusGeometry(.076, .0018, 6, 40), graphite, [0, 0, side * .022], 'sidewall-ring');
      for (let index = 0; index < 6; index++) {
        const angle = index * Math.PI / 3;
        const spoke = panel([.026, .008, .004], [Math.cos(angle) * .032, Math.sin(angle) * .032, side * .028], alloy, 'wheel-spoke');
        spoke.rotation.z = angle;
      }
    }
  } else if (asset.type === 3 && near(width, .065) && near(depth, .065)) {
    for (const offset of [-.03, -.01, .01, .03]) {
      panel([.066, .066, .006], [0, 0, offset], alloy, 'neck-collar');
    }
  } else if (asset.type === 3 && near(width, .24) && near(depth, .032)) {
    panel([.145, .034, .035], [0, 0, 0], pearl, 'arm-sleeve');
    for (const offset of [-.083, .083]) {
      panel([.012, .035, .036], [offset, 0, 0], alloy, 'arm-collar');
    }
    panel([.073, .006, .002], [0, 0, .018], rose, 'arm-inlay');
  } else if (asset.type === 2 && near(width, .018)) {
    mesh.material = graphite;
    const ring = add(new THREE.TorusGeometry(.016, .0025, 8, 24), alloy, [0, 0, 0], 'joint-ring');
    ring.rotation.x = Math.PI / 2;
  } else if (asset.type === 3 && near(width, .075) && near(depth, .012)) {
    panel([.042, .013, .027], [.004, 0, 0], graphite, 'gripper-pad');
  }
  const used = new Set<THREE.Material>();
  mesh.traverse(object => { if (object instanceof THREE.Mesh) used.add(object.material as THREE.Material); });
  materials.filter(material => !used.has(material)).forEach(material => material.dispose());
  return () => {
    mesh.traverse(object => { if (object instanceof THREE.Mesh && object !== mesh) object.geometry.dispose(); });
    used.forEach(material => { if (material !== mesh.material) material.dispose(); });
  };
}

export function enhancedLighting(scene: THREE.Scene, renderer: THREE.WebGLRenderer, extent: number, seed = 0) {
  const background = scene.background;
  scene.background = new THREE.Color(0xf5f5f5);
  const variation = seed ? THREE.MathUtils.seededRandom(seed) : .5;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.15 + (variation - .5) * .3;
  const generator = new THREE.PMREMGenerator(renderer);
  const room = new RoomEnvironment();
  const environment = generator.fromScene(room, .04);
  room.dispose();
  generator.dispose();
  scene.environment = environment.texture;
  const ambient = new THREE.HemisphereLight(0xe9f3ff, 0x8c8175, .85);
  const sun = new THREE.DirectionalLight(0xfff3df, 4);
  sun.position.set(-3 + (variation - .5) * 3, 6, -2);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  Object.assign(sun.shadow.camera, { left: -extent, right: extent, top: extent, bottom: -extent, near: .1, far: 40 });
  sun.shadow.normalBias = .004;
  sun.shadow.bias = -.00008;
  const fill = new THREE.DirectionalLight(0xe0eeff, 1.1);
  fill.position.set(3, 4, 4);
  scene.add(ambient, sun, fill);
  return () => {
    scene.background = background;
    scene.environment = null;
    scene.remove(ambient, sun, fill);
    sun.shadow.map?.dispose();
    environment.dispose();
  };
}
