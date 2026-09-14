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