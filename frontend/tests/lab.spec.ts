import { test, expect, type Page, type WebSocketRoute } from '@playwright/test';
import type { LiveState, MotionZones } from '../src/types';
import { execFileSync } from 'node:child_process';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import type { RenderPacket } from '../src/gpuCamera';

type VisualSample = { frame: string; time: number; camera: string; spectator: string };
type VisualProbe = Window & { visualSamples: VisualSample[]; visualTimer: number };

async function spectatorPixels(page: Page) {
  return page.locator('.spectator canvas').evaluate((element: HTMLCanvasElement) => {
    const context = element.getContext('webgl2')!;
    const pixels = new Uint8Array(element.width * element.height * 4);
    context.readPixels(0, 0, element.width, element.height, context.RGBA, context.UNSIGNED_BYTE, pixels);
    const samples: string[] = [];
    for (let offset = 0; offset < pixels.length; offset += 160) samples.push(`${pixels[offset]},${pixels[offset + 1]},${pixels[offset + 2]}`);
    return { colors: new Set(samples).size, signature: samples.join(';') };
  });
}

test.beforeEach(async ({ request }) => {
  expect((await request.post('/api/test/preferences/reset')).ok()).toBe(true);
  const response = await request.post('/api/challenges/load', { data: { challenge_id: 'bench' } });
  expect(response.ok()).toBeTruthy();
  await request.post('/api/agent/config', { data: { endpoint: '', models: [
    { id: 'luna', label: 'GPT-5.6 Luna', deployment: '' },
  ] } });
  await request.post('/api/voice/config', { data: { endpoint: '', deployment: '' } });
});

for (const imageFailure of ['expired', 'stalled']) {
  test(`spatial map telemetry stays independent of ${imageFailure} paired images`, async ({ page }) => {
    let sequence = 1;
    let stale = false;
    let unavailable = false;
    let releaseImages!: () => void;
    const imagesReleased = new Promise<void>(resolve => { releaseImages = resolve; });
    const cells = Array.from({ length: 160 * 160 }, (_, index) => index % 3 === 0 ? -1 : index % 3 === 1 ? 0 : 100);
    await page.route('**/api/spatial', route => route.fulfill({ status: unavailable ? 503 : 200, json: {
      enabled: true, paused: false, error: null,
      frame: { sequence, simulated_time_s: sequence, rgb_url: '/api/test-spatial/rgb', depth_url: '/api/test-spatial/depth' },
      map: { width: 160, height: 160, cells, resolution_m: .05, origin_m: [-4, -4], stale, age_s: stale ? 2 : 0,
        observed_floor_cells: 100, obstacle_cells: 100, robot_odometry_m_rad: [0, 0, 0] }, footprint: null,
    } }));
    await page.route('**/api/test-spatial/*', async route => {
      if (imageFailure === 'stalled') await imagesReleased;
      await route.fulfill({ status: 404 });
    });
    try {
      await page.goto('/');
      const map = page.getByLabel('Observed floor and obstacle map', { exact: true });
      await expect(map).toHaveAttribute('data-state', 'live');
      stale = true;
      sequence++;
      await expect(map).toHaveAttribute('data-state', 'stale');
      stale = false;
      sequence++;
      await expect(map).toHaveAttribute('data-state', 'live');
      unavailable = true;
      await expect(map).toHaveAttribute('data-state', 'unavailable');
      unavailable = false;
      sequence++;
      await expect(map).toHaveAttribute('data-state', 'live');
    } finally {
      releaseImages();
      await page.unrouteAll({ behavior: 'wait' });
    }
  });
}

test('movement zones render sampled clearance without changing camera or motion', async ({page,request}) => {
  const initial:LiveState=await(await request.get('/api/state')).json();
  const cameraBefore=await(await request.get(initial.camera.url)).body();
  await request.post('/api/spatial',{data:{run_id:initial.run_id,episode_epoch:initial.episode_epoch,enabled:true}});
  const headers={'X-Milo-Motion-Zones':'1'};
  await expect.poll(async()=>((await(await request.get('/api/spatial',{headers})).json()).motion_zones?.sectors.length ?? 0)).toBe(48);
  const observed=await(await request.get('/api/spatial',{headers})).json();
  const original:MotionZones=observed.motion_zones;
  let mode='fresh';
  const previewRequests:string[]=[];
  await page.route('**/api/spatial',async route=>{
    const preview=route.request().headers()['x-milo-motion-zones']==='1';
    previewRequests.push(preview ? 'zones' : 'plain');
    const zones:MotionZones={...original,source:'scripted_display_fixture',stale:mode==='stale',sensor_age_s:mode==='stale'?2:.05,valid_for_s:.95,
      odometry_m_rad:mode==='moved'?[initial.observation.odometry_m_rad[0]+1,0,0]:initial.observation.odometry_m_rad,
      run_id:mode==='other-episode'?'other-episode':initial.run_id,
      sectors:original.sectors.map((sector,index)=>({...sector,status:index%3===0?'clear':index%3===1?'restricted':'unknown',reason:'Scripted observed-mask region'}))};
    await route.fulfill({json:{...observed,power:{on:true,mode:'working'},paused:false,frame:null,
      error:mode==='error'?'Sensor unavailable':null,motion_zones:preview?zones:undefined}});
  });
  const commands:string[]=[];
  page.on('request',message=>{if(message.method()==='POST'&&!message.url().endsWith('/api/preferences'))commands.push(message.url());});
  const errors:string[]=[];
  page.on('pageerror',error=>errors.push(error.message));
  await page.goto('/');
  const canvas=page.locator('.spectator canvas');
  const toggle=page.getByRole('checkbox',{name:'Movement zones',exact:true});
  await expect(toggle).not.toBeChecked();
  await expect(canvas).toHaveAttribute('data-zone-count','0');
  const before=await spectatorPixels(page);
  await toggle.check();
  await expect(canvas).toHaveAttribute('data-zone-state','Observed');
  await expect(canvas).toHaveAttribute('data-zone-count','48');
  await expect.poll(async()=>(await spectatorPixels(page)).signature!==before.signature).toBe(true);
  await expect(page.getByLabel('Movement zone legend')).toContainText('geometry only');
  for(const width of [1440,1024,390,320]){
    await page.setViewportSize({width,height:1000});
    await page.locator('.world-viewport').scrollIntoViewIfNeeded();
    await expect.poll(async()=>(await spectatorPixels(page)).colors).toBeGreaterThan(30);
    expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
    await page.locator('.world-viewport').screenshot({path:`../.runtime/policy-clearance-v1/zones-${width}.png`});
  }
  await page.getByText('Zone readings / center travel',{exact:true}).click();
  await expect(page.locator('.movement-zone-readings tbody tr')).toHaveCount(16);
  await expect(page.locator('.movement-zone-readings tbody')).toContainText('restricted');
  mode='stale';
  await expect(canvas).toHaveAttribute('data-zone-state','Stale');
  await expect(page.locator('.movement-zone-readings tbody')).not.toContainText('clear');
  mode='moved';
  await expect(canvas).toHaveAttribute('data-zone-state','Pose changed');
  mode='error';
  await expect(canvas).toHaveAttribute('data-zone-state','Unavailable');
  mode='other-episode';
  await expect(canvas).toHaveAttribute('data-zone-count','0');
  mode='fresh';
  await expect(canvas).toHaveAttribute('data-zone-count','48');
  await toggle.uncheck();
  await expect(canvas).toHaveAttribute('data-zone-state','Hidden');
  await expect.poll(()=>previewRequests.at(-1)).toBe('plain');
  const after:LiveState=await(await request.get('/api/state')).json();
  expect(after.snapshot).toEqual(initial.snapshot);
  expect(after.agent.active).toBe(false);
  expect(await(await request.get(after.camera.url)).body()).toEqual(cameraBefore);
  expect(commands).toEqual([]);
  expect(errors).toEqual([]);
  expect((await request.post('/api/spatial',{data:{run_id:initial.run_id,episode_epoch:initial.episode_epoch,enabled:false}})).ok()).toBe(true);
  await page.close();
  await expect.poll(async()=>((await(await request.get('/api/state')).json()) as LiveState).stopped).toBe(true);
});

test('camera and spatial map stay in the right-side 3D HUD across viewports', async ({ page, request }) => {
  test.setTimeout(90000);
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.goto('/');
  const hud = page.getByRole('complementary', { name: 'Robot sensor HUD' });
  const map = page.getByRole('region', { name: 'Spatial map HUD' });
  const camera = hud.getByAltText('Authoritative robot head camera');
  const mapCanvas = page.getByLabel('Observed floor and obstacle map', { exact: true });
  const mapColors = () => mapCanvas.evaluate((element: HTMLCanvasElement) => {
    const pixels = element.getContext('2d')!.getImageData(0, 0, 320, 320).data;
    const colors = new Set<string>();
    for (let offset = 0; offset < pixels.length; offset += 32) colors.add(`${pixels[offset]},${pixels[offset + 1]},${pixels[offset + 2]}`);
    return colors.size;
  });
  await expect(camera).toHaveJSProperty('naturalWidth', 640);
  await expect(map).toContainText('Sensing off');
  await expect(page.locator('.spatial-section')).not.toHaveAttribute('open');
  await expect(mapCanvas).toBeVisible();
  await expect(map).toHaveAttribute('data-minimized', 'true');
  await expect(hud.getByRole('region', { name: 'Head camera HUD' })).toHaveAttribute('data-minimized', 'true');
  await hud.getByRole('button', { name: 'Restore head camera', exact: true }).click();
  await hud.getByRole('button', { name: 'Restore spatial map', exact: true }).click();
  const initial: LiveState = await (await request.get('/api/state')).json();
  expect((await request.post('/api/spatial', { data: { run_id: initial.run_id, episode_epoch: initial.episode_epoch, enabled: true } })).ok()).toBe(true);
  await expect(mapCanvas).toHaveAttribute('data-state', 'idle');
  await expect.poll(mapColors).toBeGreaterThan(2);
  for (const width of [1440, 1024, 390, 320]) {
    await page.setViewportSize({ width, height: 1000 });
    await page.locator('.world-viewport').scrollIntoViewIfNeeded();
    await expect.poll(async () => (await spectatorPixels(page)).colors).toBeGreaterThan(30);
    const layout = await hud.evaluate(element => {
      const viewport = element.closest('.world-viewport')!.getBoundingClientRect();
      const bounds = element.getBoundingClientRect();
      const widgets = [...element.querySelectorAll('.viewport-widget')].map(widget => widget.getBoundingClientRect());
      const clipped = [...element.querySelectorAll('h3, .camera-meta, .viewport-map-meta')]
        .filter(child => child.scrollWidth > child.clientWidth + 1).length;
      return { rightGap: viewport.right - bounds.right, topGap: bounds.top - viewport.top,
        bottomGap: viewport.bottom - bounds.bottom, separation: widgets[1].top - widgets[0].bottom,
        worldVisible: bounds.left - viewport.left, clipped, pageOverflow: document.documentElement.scrollWidth > innerWidth };
    });
    expect(layout.rightGap).toBeGreaterThanOrEqual(7);
    expect(layout.rightGap).toBeLessThanOrEqual(13);
    expect(layout.topGap).toBeGreaterThanOrEqual(7);
    expect(layout.bottomGap).toBeGreaterThanOrEqual(0);
    expect(layout.separation).toBeGreaterThanOrEqual(7);
    expect(layout.worldVisible).toBeGreaterThan(140);
    expect(layout.clipped).toBe(0);
    expect(layout.pageOverflow).toBe(false);
    await page.locator('.world-viewport').screenshot({ path: `../.runtime/viewport-micro-hud-v2/hud-${width}.png` });
    await page.getByRole('button', { name: 'Expand spatial map', exact: true }).click();
    const dialog = page.getByRole('dialog', { name: 'Spatial map', exact: true });
    await expect(dialog).toBeVisible();
    await expect(dialog.locator('canvas')).toBeVisible();
    await expect(mapCanvas).toHaveCount(1);
    expect(await dialog.evaluate(element => element.scrollWidth <= element.clientWidth)).toBe(true);
    await page.keyboard.press('Escape');
    await expect(dialog).not.toBeVisible();
    await expect(hud.locator('canvas')).toBeVisible();
    await expect.poll(mapColors).toBeGreaterThan(2);
    await expect(page.getByRole('button', { name: 'Expand spatial map', exact: true })).toBeFocused();
    await hud.getByRole('button', { name: 'Minimize head camera', exact: true }).click();
    await expect(camera).toBeVisible();
    await expect(mapCanvas).toBeVisible();
    await expect(map).toHaveAttribute('data-minimized', 'false');
    await hud.getByRole('button', { name: 'Minimize spatial map', exact: true }).click();
    await expect(mapCanvas).toBeVisible();
    await expect(hud.getByRole('button', { name: 'Restore head camera', exact: true })).toHaveAttribute('aria-expanded', 'false');
    await expect(hud.getByRole('button', { name: 'Restore spatial map', exact: true })).toHaveAttribute('aria-expanded', 'false');
    const microWidgets = await hud.locator('.viewport-widget').evaluateAll(elements => elements.map(element => {
      const bounds = element.getBoundingClientRect();
      return { width: bounds.width, height: bounds.height, text: (element as HTMLElement).innerText.trim() };
    }));
    for (const widget of microWidgets) {
      expect(widget.width).toBe(width <= 600 ? 72 : 96);
      expect(widget.text).toBe('');
    }
    expect(microWidgets[0].height).toBeCloseTo((microWidgets[0].width - 2) * .75 + 2, 0);
    expect(microWidgets[1].height).toBe(microWidgets[1].width);
    await expect.poll(mapColors).toBeGreaterThan(2);
    for (const name of ['Restore head camera', 'Restore spatial map']) {
      const restore = hud.getByRole('button', { name, exact: true });
      await restore.hover();
      expect(await restore.evaluate(element => {
        const swatch = document.createElement('canvas').getContext('2d')!;
        swatch.fillStyle = getComputedStyle(element).backgroundColor;
        swatch.fillRect(0, 0, 1, 1);
        return swatch.getImageData(0, 0, 1, 1).data[3];
      })).toBeLessThan(32);
    }
    await page.locator('.world-viewport').screenshot({ path: `../.runtime/viewport-micro-hud-v2/minimized-${width}.png` });
    await hud.getByRole('button', { name: 'Restore spatial map', exact: true }).focus();
    await page.keyboard.press('Enter');
    await expect(mapCanvas).toBeVisible();
    await expect(map).toHaveAttribute('data-minimized', 'false');
    await expect(hud.getByRole('region', { name: 'Head camera HUD' })).toHaveAttribute('data-minimized', 'true');
    await hud.getByRole('button', { name: 'Restore head camera', exact: true }).click();
    await expect(camera).toBeVisible();
  }
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.locator('.spectator canvas').hover({ position: { x: 100, y: 200 } });
  const beforeOrbit = await spectatorPixels(page);
  const world = (await page.locator('.spectator canvas').boundingBox())!;
  expect(await page.evaluate(({ left, top }) => document.elementFromPoint(left, top)?.getAttribute('aria-label'),
    { left: world.x + 100, top: world.y + 200 })).toBe('Live robot spectator viewport');
  await page.mouse.down();
  await page.mouse.move(world.x + 200, world.y + 240, { steps: 8 });
  await page.mouse.up();
  await expect.poll(async () => (await spectatorPixels(page)).signature !== beforeOrbit.signature).toBe(true);
  const before: LiveState = await (await request.get('/api/state')).json();
  await hud.getByRole('button', { name: 'Minimize head camera', exact: true }).click();
  const response = await request.post('/api/command', { data: { run_id: before.run_id, episode_epoch: before.episode_epoch,
    observation_seq: before.observation.seq, action_id: 'hud-head-motion', tool: 'set_head',
    arguments: { yaw_rad: .3, pitch_rad: .5, duration_s: 1 } } });
  expect((await response.json()).status).toBe('ok');
  await expect.poll(async () => Number(await camera.getAttribute('data-simulated-time'))).toBeGreaterThan(before.snapshot.simulated_time_s);
  await expect(camera).toBeVisible();
  await expect(hud.getByRole('region', { name: 'Head camera HUD' })).toHaveAttribute('data-minimized', 'true');
  await hud.getByRole('button', { name: 'Restore head camera', exact: true }).click();
  await expect(camera).toBeVisible();
  expect((await request.post('/api/spatial', { data: { run_id: before.run_id, episode_epoch: before.episode_epoch, enabled: false } })).ok()).toBe(true);
  await expect(map).toContainText('Sensing off');
  await expect(mapCanvas).toBeVisible();
  expect(errors).toEqual([]);
});

test('detailed robot appearance preserves sensors and follows head motion across viewports', async ({ page, request }) => {
  test.setTimeout(90000);
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  const output = fileURLToPath(new URL('../../.runtime/robot-appearance-v1/', import.meta.url));
  await mkdir(output, { recursive: true });
  await page.setViewportSize({ width: 1440, height: 1100 });
  await page.goto('/');
  const viewport = page.locator('.spectator canvas');
  await expect.poll(async () => (await spectatorPixels(page)).colors).toBeGreaterThan(100);
  const before: LiveState = await (await request.get('/api/state')).json();
  const cameraBefore = await (await request.get(before.camera.url)).body();
  const initialPixels = await spectatorPixels(page);
  for (const width of [1440, 390, 320]) {
    await page.setViewportSize({ width, height: 1100 });
    await viewport.scrollIntoViewIfNeeded();
    await expect.poll(async () => (await spectatorPixels(page)).colors).toBeGreaterThan(100);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await page.locator('.spectator-shell').screenshot({ path: `${output}/robot-${width}.png` });
  }
  const after: LiveState = await (await request.get('/api/state')).json();
  expect(after.snapshot).toEqual(before.snapshot);
  expect(after.geometry).toEqual(before.geometry);
  expect(after.observation).toEqual(before.observation);
  expect((await (await request.get(after.camera.url)).body()).equals(cameraBefore)).toBe(true);
  expect(after.agent.active).toBe(false);
  await page.setViewportSize({ width: 1440, height: 1100 });
  await viewport.scrollIntoViewIfNeeded();
  const motion = await request.post('/api/command', { data: { run_id: after.run_id, episode_epoch: after.episode_epoch,
    observation_seq: after.observation.seq, action_id: 'detailed-robot-head', tool: 'set_head',
    arguments: { yaw_rad: -.6, pitch_rad: .25, duration_s: 1 } } });
  expect((await motion.json()).status).toBe('ok');
  await expect.poll(async () => (await spectatorPixels(page)).signature).not.toBe(initialPixels.signature);
  await page.locator('.spectator-shell').screenshot({ path: `${output}/robot-head-turned.png` });
  await page.getByRole('button', { name: 'Stop', exact: true }).click();
  await expect(page.getByRole('status', { name: 'Robot status', exact: true })).toContainText('Stopped');
  await expect.poll(async () => (await (await request.get('/api/state')).json()).stopped).toBe(true);
  const stopped: LiveState = await (await request.get('/api/state')).json();
  await page.mouse.move((await viewport.boundingBox())!.x + 100, (await viewport.boundingBox())!.y + 100);
  await page.mouse.wheel(0, -180);
  const zoomed: LiveState = await (await request.get('/api/state')).json();
  expect(zoomed.snapshot.simulated_time_s).toBe(stopped.snapshot.simulated_time_s);
  expect(zoomed.run_id).toBe(before.run_id);
  expect(errors).toEqual([]);
});

test('enhanced backend shares graphics across live RGB depth and world view', async ({ page, request }) => {
  test.setTimeout(90000);
  const initial: LiveState = await (await request.get('/api/state')).json();
  test.skip(initial.rendering !== 'enhanced', 'Run with MILO_RENDERER=enhanced to exercise the live GPU backend');
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  const state: LiveState = await (await request.post('/api/challenges/load', { data: { challenge_id: 'flat_kitchen' } })).json();
  await page.addInitScript(() => localStorage.setItem('milo-spectator-graphics', 'standard'));
  await page.setViewportSize({ width: 1440, height: 1100 });
  await page.goto('/?graphics=standard');
  await expect(page.getByRole('group', { name: 'Spectator graphics', exact: true })).toHaveCount(0);
  await expect(page.getByAltText('Authoritative robot head camera')).toHaveJSProperty('naturalWidth', 640);
  await expect.poll(async () => (await spectatorPixels(page)).colors).toBeGreaterThan(50);
  const image = await (await request.get(state.camera.url)).body();
  const modelImage = await (await request.get(`/api/frames/${state.run_id}/${state.observation.frame_ref}`)).body();
  expect(image.equals(modelImage)).toBe(true);
  const spatial = await request.post('/api/spatial', { data: { run_id: state.run_id, episode_epoch: state.episode_epoch, enabled: true } });
  expect(spatial.ok()).toBe(true);
  await expect(page.getByAltText('Authoritative robot head camera')).toHaveJSProperty('naturalWidth', 160);
  const motion = await request.post('/api/command', { data: { run_id: state.run_id, episode_epoch: state.episode_epoch,
    observation_seq: state.observation.seq, action_id: 'enhanced-live-head', tool: 'set_head',
    arguments: { yaw_rad: .35, pitch_rad: .5, duration_s: 1 } } });
  expect((await motion.json()).status).toBe('ok');
  const moved: LiveState = await (await request.get('/api/state')).json();
  expect(moved.rendering).toBe('enhanced');
  expect(moved.snapshot.simulated_time_s).toBeGreaterThan(0);
  expect((await (await request.get(moved.camera.url)).body()).equals(image)).toBe(false);
  for (const width of [1440, 390, 320]) {
    await page.setViewportSize({ width, height: 1100 });
    await page.locator('.spectator').scrollIntoViewIfNeeded();
    await expect.poll(async () => (await spectatorPixels(page)).colors).toBeGreaterThan(50);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await page.locator('.observatory').screenshot({ path: `test-results/all-enhanced-${width}.png` });
  }
  await page.getByRole('button', { name: 'Stop', exact: true }).click();
  await expect(page.getByText('Robot stopped', { exact: true })).toBeVisible();
  const reset: LiveState = await (await request.post('/api/reset')).json();
  expect(reset.rendering).toBe('enhanced');
  expect(reset.run_id).not.toBe(state.run_id);
  expect(reset.snapshot.simulated_time_s).toBe(0);
  expect((await request.get(state.camera.url)).status()).toBe(404);
  expect((await request.post('/api/spatial', { data: { run_id: reset.run_id, episode_epoch: reset.episode_epoch, enabled: false } })).ok()).toBe(true);
  expect(errors).toEqual([]);
});

test('enhanced spectator graphics preserve sensor frames and work across viewports', async ({ page, request }) => {
  const state: LiveState = await (await request.get('/api/state')).json();
  test.skip(state.rendering === 'enhanced', 'The unified enhanced backend owns graphics; its separate test checks the hidden toggle.');
  test.setTimeout(90000);
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await request.post('/api/challenges/load', { data: { challenge_id: 'kitchen_bathroom' } });
  await page.setViewportSize({ width: 1440, height: 1100 });
  await page.goto('/?graphics=enhanced');
  const quality = page.getByRole('group', { name: 'Spectator graphics', exact: true });
  await expect(quality.getByRole('button', { name: 'Enhanced', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await expect.poll(async () => (await spectatorPixels(page)).colors).toBeGreaterThan(50);
  const before: LiveState = await (await request.get('/api/state')).json();
  const enhanced = await spectatorPixels(page);
  await quality.getByRole('button', { name: 'Standard', exact: true }).click();
  await expect(quality.getByRole('button', { name: 'Standard', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await expect.poll(async () => (await spectatorPixels(page)).signature).not.toBe(enhanced.signature);
  await quality.getByRole('button', { name: 'Enhanced', exact: true }).click();
  const after: LiveState = await (await request.get('/api/state')).json();
  expect(after.snapshot).toEqual(before.snapshot);
  expect(after.observation).toEqual(before.observation);
  expect(after.camera).toEqual(before.camera);
  expect(after.agent.active).toBe(false);
  for (const width of [1440, 390, 320]) {
    await page.setViewportSize({ width, height: 1100 });
    await page.locator('.spectator').scrollIntoViewIfNeeded();
    await expect.poll(async () => (await spectatorPixels(page)).colors).toBeGreaterThan(50);
    await expect(quality.getByRole('button', { name: 'Enhanced', exact: true })).toBeVisible();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await page.locator('.spectator-shell').screenshot({ path: `test-results/enhanced-kitchen-${width}.png` });
  }
  await expect(page.getByAltText('Authoritative robot head camera')).toHaveJSProperty('naturalWidth', 640);
  const motion = await request.post('/api/command', { data: { run_id: after.run_id, episode_epoch: after.episode_epoch,
    observation_seq: after.observation.seq, action_id: 'enhanced-head-motion', tool: 'set_head',
    arguments: { yaw_rad: -.4, pitch_rad: .4, duration_s: 1 } } });
  expect((await motion.json()).status).toBe('ok');
  const moved: LiveState = await (await request.get('/api/state')).json();
  expect(moved.snapshot.simulated_time_s).toBeGreaterThan(after.snapshot.simulated_time_s);
  expect(moved.camera.frame_ref).not.toEqual(after.camera.frame_ref);
  expect(errors).toEqual([]);
});

test('GPU camera renders snapshots without live control and rejects wrong episodes', async ({ page }, testInfo) => {
  test.setTimeout(90000);
  const root = fileURLToPath(new URL('../..', import.meta.url));
  const python = process.env.ROBOSIM_PYTHON ?? fileURLToPath(new URL('../../.runtime/env/python.exe', import.meta.url));
  const output = testInfo.outputPath('render-cases');
  execFileSync(python, ['-m', 'scripts.benchmark_rendering', '--output', output, '--export-only', '--samples', '1', '--widths', '160'], { cwd: root, timeout: 60000 });
  const packet: RenderPacket = JSON.parse(await readFile(`${output}/kitchen_bathroom-0-160/snapshot.json`, 'utf8'));
  const moved: RenderPacket = JSON.parse(await readFile(`${output}/kitchen_bathroom-1-160/snapshot.json`, 'utf8'));
  const sockets: string[] = [], errors: string[] = [];
  page.on('websocket', socket => sockets.push(socket.url()));
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('/?cameraWorker=1');
  await page.waitForFunction(() => !!window.miloGpuCamera);
  const loaded = await page.evaluate(packet => window.miloGpuCamera.load(packet, 'enhanced', 713), packet);
  expect(loaded.textures).toBeGreaterThan(0);
  const first = await page.evaluate(packet => window.miloGpuCamera.capture(packet), packet);
  const repeated = await page.evaluate(packet => window.miloGpuCamera.capture(packet), packet);
  expect(repeated.rgb).toEqual(first.rgb);
  expect(repeated.depth_f32).toEqual(first.depth_f32);
  expect(first.run_id).toBe(packet.run_id);
  expect(first.observation_seq).toBe(packet.observation_seq);
  const depthBytes = Buffer.from(first.depth_f32, 'base64');
  expect(depthBytes.byteLength).toBe(160 * 120 * 4);
  let valid = 0, invalidRange = 0;
  for (let offset = 0; offset < depthBytes.length; offset += 4) {
    const depth = depthBytes.readFloatLE(offset);
    if (Number.isFinite(depth)) { if (depth <= 0 || depth >= 12) invalidRange++; valid++; }
  }
  expect(invalidRange).toBe(0);
  expect(valid).toBeGreaterThan(160 * 120 * .2);
  const changed = await page.evaluate(packet => window.miloGpuCamera.capture(packet), moved);
  expect(changed.rgb).not.toBe(first.rgb);
  expect(changed.depth_f32).not.toBe(first.depth_f32);
  await page.evaluate(packet => window.miloGpuCamera.load(packet, 'enhanced', 714), packet);
  const varied = await page.evaluate(packet => window.miloGpuCamera.capture(packet), packet);
  expect(varied.rgb).not.toBe(first.rgb);
  expect(varied.depth_f32).toBe(first.depth_f32);
  await page.evaluate(packet => window.miloGpuCamera.load(packet, 'enhanced', 713), packet);
  expect((await page.evaluate(packet => window.miloGpuCamera.capture(packet), packet)).rgb).toBe(first.rgb);
  for (const invalid of [{ ...packet, episode_epoch: packet.episode_epoch + 1 }, { ...packet, run_id: 'other-run' }]) {
    await expect(page.evaluate(packet => window.miloGpuCamera.capture(packet), invalid)).rejects.toThrow('Stale render episode');
  }
  const large = await page.evaluate(packet => window.miloGpuCamera.capture({ ...packet, camera: { ...packet.camera, width: 640, height: 480 } }), packet);
  expect(Buffer.from(large.depth_f32, 'base64').byteLength).toBe(640 * 480 * 4);
  await writeFile(testInfo.outputPath('gpu-head.png'), Buffer.from(large.rgb, 'base64'));
  expect(sockets).toEqual([]);
  expect(errors).toEqual([]);
  await page.evaluate(() => window.miloGpuCamera.dispose());
  await expect(page.locator('canvas')).toHaveCount(0);
});

test('textures and live proximity readings render in the operator views', async ({ page, request }) => {
  const loadedTextures = new Set<string>();
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('response', response => { if (response.url().includes('/api/textures/') && response.ok()) loadedTextures.add(response.url().split('/').at(-1)!); });
  await page.setViewportSize({ width: 1440, height: 1100 });
  await page.goto('/');
  const sensors = page.getByRole('region', { name: 'Collision and distance sensors', exact: true });
  await expect(sensors.locator('.distance-reading')).toHaveCount(8);
  await expect(sensors).toContainText('No collision');
  await expect.poll(() => loadedTextures.size).toBe(2);
  const initial: LiveState = await (await request.get('/api/state')).json();
  const positioned = await request.post('/api/robot/placement', { data: {
    run_id: initial.run_id, episode_epoch: initial.episode_epoch, observation_seq: initial.observation.seq, xy_m: [1.6, 0],
  } });
  expect(positioned.status()).toBe(200);
  const placed: LiveState = await positioned.json();
  const front = sensors.locator('.distance-reading').filter({ has: page.getByText('front', { exact: true }) });
  await expect(front.locator('dd')).toHaveText('0.67 m');
  const motion = request.post('/api/command', { data: { run_id: placed.run_id, episode_epoch: placed.episode_epoch,
    observation_seq: placed.observation.seq, action_id: 'range-motion', tool: 'drive_base',
    arguments: { linear_mps: .3, angular_radps: 0, duration_s: 2 },
  } });
  await expect.poll(async () => Number(await sensors.getAttribute('data-simulated-time'))).toBeGreaterThan(0);
  await expect(front.locator('dd')).not.toHaveText('0.67 m');
  expect((await (await motion).json()).status).toBe('ok');
  await page.getByRole('combobox', { name: 'Predefined challenge', exact: true }).selectOption('apartment');
  await page.getByRole('button', { name: 'Load challenge', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Apartment Search', exact: true })).toBeVisible();
  await expect.poll(() => loadedTextures.size).toBe(6);
  await expect.poll(async () => (await spectatorPixels(page)).colors).toBeGreaterThan(50);
  await expect(page.getByAltText('Authoritative robot head camera')).toHaveJSProperty('naturalWidth', 640);
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: 'test-results/textured-apartment-desktop.png', fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await expect.poll(async () => (await spectatorPixels(page)).colors).toBeGreaterThan(50);
  await page.screenshot({ path: 'test-results/textured-apartment-mobile.png', fullPage: true });
  expect(errors).toEqual([]);
});

test('outcome feedback separates verified completion, agent reports, and unsuccessful results', async ({ page, request }) => {
  const initial: LiveState = await (await request.post('/api/challenges/load', { data: { challenge_id: 'park' } })).json();
  let socket: WebSocketRoute;
  await page.routeWebSocket('**/api/live', connection => { socket = connection; connection.send(JSON.stringify(initial)); });
  function publish(patch: Partial<LiveState> = {}, agent: Partial<LiveState['agent']> = {}) {
    socket.send(JSON.stringify({ ...initial, ...patch, agent: { ...initial.agent, ...agent } }));
  }
  await page.goto('/');
  const outcome = page.getByRole('region', { name: 'Robot activity', exact: true });
  await expect(outcome).toContainText('On / Idle');
  for (const [kind, title] of [['completed', 'Agent reports completion'], ['unachievable', 'Task cannot be completed'],
    ['limited', 'Task limit reached'], ['interrupted', 'Stopped'], ['ended', 'Response finished']]) {
    publish({stopped:kind === 'interrupted'}, { outcome: { kind, message: 'A clear reason for this outcome.', source: kind === 'completed' || kind === 'unachievable' ? 'agent' : 'controller', timestamp: 1 } });
    await expect(outcome.locator('.status')).toHaveText(title);
    await expect(outcome).not.toHaveAttribute('data-state', 'success');
    await expect(outcome).toContainText('A clear reason for this outcome.');
  }
  publish({}, { error: 'The model connection timed out.' });
  await expect(outcome).toContainText('Agent error');
  await expect(outcome).toContainText('timed out');
  publish({ result: { action_id: 'blocked', status: 'error', error: 'COLLISION_BLOCKED', message: 'Arm trajectory intersects a wall.', actual_duration_s: 0, observation: initial.observation } });
  await expect(outcome).toContainText('Action blocked');
  publish({ result: { action_id: 'new-action', status: 'ok', error: null, message: 'Manual head movement completed.', actual_duration_s: 1,
    observation: { ...initial.observation, wall_timestamp: 2 } } }, {
    outcome: { kind: 'interrupted', message: 'Previous run interrupted.', source: 'controller', timestamp: 1 },
  });
  await expect(outcome).toContainText('Action completed');
  await expect(outcome).toContainText('Manual head movement completed.');
  await expect(outcome).not.toContainText('Previous run interrupted');
  publish({ challenge: { ...initial.challenge!, status: 'failed', progress: [{ label: 'Goal', complete: false, detail: 'Battery empty' }] } });
  await expect(outcome).toContainText('Task failed');
  await expect(outcome).toContainText('Battery empty');
  publish({ manual_placements: 1, challenge: { ...initial.challenge!, status: 'completed', completed_objectives: 1 } }, { phase: 'sleeping' });
  await expect(outcome.locator('.status')).toHaveText('Task completed');
  await expect(outcome).toHaveAttribute('data-state', 'success');
  await expect(outcome).toContainText('verified by physics / Operator-assisted episode');
  for (const width of [1440, 390, 320]) {
    await page.setViewportSize({ width, height: 1000 });
    await page.locator('#controls').scrollIntoViewIfNeeded();
    await expect(outcome).toBeVisible();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await page.locator('.robot-activity').screenshot({ path: `test-results/task-completed-${width}.png` });
  }
  publish({});
  await expect(outcome).toContainText('On / Idle');
  socket!.close();
  await expect(outcome).toContainText('Connection lost');
  await expect(outcome).not.toHaveAttribute('data-outcome', 'success');
});

async function expectDefaultTurnLimit(page: Page) {
  await expect(page.locator('#panel-settings input[aria-label="Supervisor turn limit"]')).toHaveValue('80');
}

async function writeScenarioPreview(page: Page, identifier: string) {
  if (process.env.MILO_CAPTURE_PREVIEWS !== '1') return;
  await expect.poll(async () => (await spectatorPixels(page)).colors).toBeGreaterThan(20);
  const encoded = await page.locator('.spectator canvas').evaluate((source: HTMLCanvasElement) => {
    const original = document.createElement('canvas');
    original.width = source.width;
    original.height = source.height;
    const originalContext = original.getContext('2d')!;
    originalContext.drawImage(source, 0, 0);
    const pixels = originalContext.getImageData(0, 0, source.width, source.height).data;
    let left = source.width, right = 0, top = source.height, bottom = 0;
    for (let row = 0; row < source.height; row += 2) {
      for (let column = 0; column < source.width; column += 2) {
        const offset = (row * source.width + column) * 4;
        if (Math.max(...[0, 1, 2].map(channel => Math.abs(pixels[offset + channel] - pixels[channel]))) <= 8) continue;
        left = Math.min(left, column); right = Math.max(right, column);
        top = Math.min(top, row); bottom = Math.max(bottom, row);
      }
    }
    const margin = 20;
    left = Math.max(0, left - margin); top = Math.max(0, top - margin);
    right = Math.min(source.width, right + margin); bottom = Math.min(source.height, bottom + margin);
    const cropWidth = Math.max(1, right - left), cropHeight = Math.max(1, bottom - top);
    const image = document.createElement('canvas');
    image.width = 480;
    image.height = 300;
    const context = image.getContext('2d')!;
    context.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--cp-surface-soft').trim();
    context.fillRect(0, 0, image.width, image.height);
    const scale = Math.min(image.width / cropWidth, image.height / cropHeight);
    context.drawImage(original, left, top, cropWidth, cropHeight,
      (image.width - cropWidth * scale) / 2, (image.height - cropHeight * scale) / 2, cropWidth * scale, cropHeight * scale);
    return image.toDataURL('image/webp', .88).split(',')[1];
  });
  const directory = new URL('../public/scenario-previews/', import.meta.url);
  await mkdir(directory, { recursive: true });
  await writeFile(new URL(`${identifier}.webp`, directory), Buffer.from(encoded, 'base64'));
}

test('shared apartment selection preserves the common world and renders across viewports', async ({ page, request }) => {
  test.setTimeout(120000);
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.setViewportSize({ width: 1440, height: 1100 });
  await page.goto('/');
  const environment = page.getByRole('combobox', { name: 'Training environment', exact: true });
  const selector = page.getByRole('combobox', { name: 'Predefined challenge', exact: true });
  const before: LiveState = await (await request.get('/api/state')).json();
  await environment.selectOption('shared_apartment_v1');
  await expect(selector.locator('option')).toHaveCount(4);
  await expect(selector).toHaveValue('furniture_circuit');
  await expect(page.getByRole('combobox', { name: 'Object to circle', exact: true }).locator('option')).toHaveCount(1);
  expect((await (await request.get('/api/state')).json()).run_id).toBe(before.run_id);
  for (const identifier of ['furniture_circuit', 'apartment', 'flat_kitchen', 'recharge']) {
    await selector.selectOption(identifier);
    const response = page.waitForResponse(reply => reply.url().endsWith('/api/challenges/load') && reply.request().method() === 'POST');
    await page.getByRole('button', { name: 'Load challenge', exact: true }).click();
    expect((await response).ok()).toBe(true);
    await expect(page.locator('.scenario-selection-state')).toHaveText('Loaded');
    const state: LiveState = await (await request.get('/api/state')).json();
    expect(state.challenge?.environment).toBe('shared_apartment_v1');
    expect(state.challenge?.id).toBe(identifier);
    expect(state.agent.active).toBe(false);
    expect(state.snapshot.simulated_time_s).toBe(0);
    await expect(page.getByRole('textbox', { name: 'Robot goal', exact: true })).toHaveValue(state.challenge!.goal);
    await expect.poll(async () => (await spectatorPixels(page)).colors).toBeGreaterThan(40);
    await expect(page.getByAltText('Authoritative robot head camera')).toHaveJSProperty('naturalWidth', 640);
    if (identifier === 'furniture_circuit') await writeScenarioPreview(page, 'shared_apartment_v1');
  }
  const resetResponse = page.waitForResponse(reply => reply.url().endsWith('/api/reset'));
  if (await page.locator('.robot-options').getAttribute('open') === null) await page.locator('.robot-options > summary').click();
  await page.getByRole('button', { name: 'Reset episode', exact: true }).click();
  const reset: LiveState = await (await resetResponse).json();
  await expect(page.locator('.viewport-footer')).toContainText(`Epoch ${reset.episode_epoch}`);
  await expect(environment).toHaveValue('shared_apartment_v1');
  await expect(selector).toHaveValue('recharge');
  await page.reload();
  await expect(environment).toHaveValue('shared_apartment_v1');
  await expect(selector.locator('option')).toHaveCount(4);
  await expect(page.getByAltText('Scene preview: Shared Apartment V1')).toHaveJSProperty('naturalWidth', 480);
  for (const width of [1440, 768, 390, 320]) {
    await page.setViewportSize({ width, height: 1100 });
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    for (const control of [environment, selector]) expect((await control.boundingBox())!.width).toBeGreaterThan(140);
    await expect.poll(async () => (await spectatorPixels(page)).colors).toBeGreaterThan(40);
    await page.locator('#setup').screenshot({ path: `test-results/shared-apartment-setup-${width}.png` });
    await page.locator('.spectator-shell').screenshot({ path: `test-results/shared-apartment-world-${width}.png` });
  }
  expect((await request.post('/api/resume')).ok()).toBe(true);
  const state: LiveState = await (await request.get('/api/state')).json();
  const pixels = (await spectatorPixels(page)).signature;
  const motion = await request.post('/api/command', { data: { run_id: state.run_id, episode_epoch: state.episode_epoch,
    observation_seq: state.observation.seq, action_id: 'shared-apartment-drive', tool: 'drive_base',
    arguments: { linear_mps: .15, angular_radps: 0, duration_s: 1 } } });
  expect((await motion.json()).status).toBe('ok');
  await expect.poll(async () => (await spectatorPixels(page)).signature).not.toBe(pixels);
  await page.getByRole('button', { name: 'Stop', exact: true }).click();
  await expect(page.getByText('Robot stopped', { exact: true })).toBeVisible();
  await environment.selectOption('standalone');
  await expect(selector.locator('option')).toHaveCount(15);
  await expect(page.locator('.scenario-selection-state')).toHaveText('Not loaded');
  await expect(page.getByAltText('Scene preview: Remember and Recharge')).toHaveJSProperty('naturalWidth', 480);
  expect(errors).toEqual([]);
});


test('predefined challenges load distinct scenes and goals and reset in place', async ({ page, request }) => {
  test.setTimeout(120000);
  const presets = await (await request.get('/api/challenges')).json();
  await page.setViewportSize({ width: 1440, height: 1100 });
  await page.goto('/');
  const selector = page.getByRole('combobox', { name: 'Predefined challenge', exact: true });
  await expect(selector.locator('option')).toHaveCount(presets.length + 1);
  const signatures = new Set();
  for (const preset of presets) {
    await selector.selectOption(preset.id);
    await expect(page.locator('.challenge-goal')).toHaveText(preset.goal);
    await page.getByRole('button', { name: 'Load challenge', exact: true }).click();
    await expect(page.getByRole('heading', { name: preset.title, exact: true })).toBeVisible();
    await expect(page.getByRole('textbox', { name: 'Robot goal', exact: true })).toHaveValue(preset.goal);
    await expectDefaultTurnLimit(page);
    await expect(page.locator('.challenge-objectives li')).toHaveCount(preset.objectives.length);
    await expect(page.locator('.challenge-status')).toHaveText(`0 / ${preset.objectives.length} goals complete`);
    await expect.poll(async () => (await spectatorPixels(page)).colors).toBeGreaterThan(20);
    await expect(page.getByAltText('Authoritative robot head camera')).toHaveJSProperty('naturalWidth', 640);
    signatures.add((await spectatorPixels(page)).signature);
    await writeScenarioPreview(page, preset.id);
    await page.screenshot({ path: `test-results/challenge-${preset.id}.png`, fullPage: true });
    const loaded = await (await request.get('/api/state')).json();
    const resetResponse = page.waitForResponse(response => response.url().endsWith('/api/reset') && response.request().method() === 'POST');
    if (await page.locator('.robot-options').getAttribute('open') === null) await page.locator('.robot-options > summary').click();
    await page.getByRole('button', { name: 'Reset episode', exact: true }).click();
    expect((await resetResponse).ok()).toBe(true);
    await expect.poll(async () => (await (await request.get('/api/state')).json()).run_id).not.toBe(loaded.run_id);
    const reset = await (await request.get('/api/state')).json();
    await expect(page.locator('.viewport-footer')).toContainText(`Epoch ${reset.episode_epoch}`);
    await expect(selector).toHaveValue(preset.id);
    await expect(page.getByRole('textbox', { name: 'Robot goal', exact: true })).toHaveValue(preset.goal);
  }
  expect(signatures.size).toBe(presets.length);
  await selector.selectOption('bench');
  await page.getByRole('button', { name: 'Load challenge', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Practice bench', exact: true })).toBeVisible();
  await expect(page.locator('.challenge-status')).toHaveCount(0);
  await writeScenarioPreview(page, 'bench');
});

for (const width of [1440, 390]) {
  test(`advanced training grounds render and move at ${width}px`, async ({ page, request }) => {
    test.setTimeout(90000);
    await page.setViewportSize({ width, height: 1000 });
    await page.goto('/');
    const selector = page.getByRole('combobox', { name: 'Predefined challenge', exact: true });
    await expect(selector.locator('optgroup')).toHaveCount(3);
    for (const [identifier, title] of [['clinic_delivery', 'Clinic Supply Delivery'], ['warehouse', 'Warehouse Dispatch Circuit'],
      ['inspection', 'Service Gallery Inspection'], ['workshop', 'Cluttered Assembly Workshop'], ['pedestrian_crossing', 'Pedestrian Crossing'],
      ['flat_kitchen', 'Find the Kitchen']]) {
      await selector.selectOption(identifier);
      await page.getByRole('button', { name: 'Load challenge', exact: true }).click();
      await expect(page.getByRole('heading', { name: title, exact: true })).toBeVisible();
      await expect(page.locator('.challenge-toolbar')).toContainText('Advanced');
      await expect.poll(async () => (await spectatorPixels(page)).colors).toBeGreaterThan(30);
      await expect(page.getByAltText('Authoritative robot head camera')).toHaveJSProperty('naturalWidth', 640);
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
      await page.locator('.spectator-shell').screenshot({ path: `test-results/ground-${identifier}-${width}.png` });
      await page.locator('.camera-frame').screenshot({ path: `test-results/ground-camera-${identifier}-${width}.png` });
      if (identifier === 'warehouse' || identifier === 'flat_kitchen') {
        const before: LiveState = await (await request.get('/api/state')).json();
        const pixelsBefore = (await spectatorPixels(page)).signature;
        const response = await request.post('/api/command', { data: {
          run_id: before.run_id, episode_epoch: before.episode_epoch, observation_seq: before.observation.seq,
          action_id: `${identifier}-drive-${width}`, tool: 'drive_base', arguments: { linear_mps: .15, angular_radps: 0, duration_s: 1 },
        } });
        const result = await response.json();
        expect(result.status).toBe('ok');
        expect(result.observation.odometry_m_rad[0]).toBeGreaterThan(.05);
        await expect(page.getByAltText('Authoritative robot head camera')).toHaveAttribute('data-simulated-time', '1');
        await expect.poll(async () => (await spectatorPixels(page)).signature).not.toBe(pixelsBefore);
        if (await page.locator('.robot-options').getAttribute('open') === null) await page.locator('.robot-options > summary').click();
        await page.getByRole('button', { name: 'Reset episode', exact: true }).click();
        await expect(page.getByAltText('Authoritative robot head camera')).toHaveAttribute('data-simulated-time', '0');
        await expect(page.locator('.challenge-status')).toHaveText(`0 / ${identifier === 'warehouse' ? 3 : 1} goals complete`);
      }
    }
    await page.screenshot({ path: `test-results/grounds-layout-${width}.png`, fullPage: true });
  });
}

test('kitchen to bathroom shows recognizable camera views and resettable arrival', async ({ page, request }) => {
  await page.setViewportSize({ width: 1440, height: 1100 });
  await page.goto('/');
  const selector = page.getByRole('combobox', { name: 'Predefined challenge', exact: true });
  await selector.selectOption('kitchen_bathroom');
  await page.getByRole('button', { name: 'Load challenge', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Kitchen to Bathroom', exact: true })).toBeVisible();
  await expect(page.getByRole('textbox', { name: 'Robot goal', exact: true })).toContainText('identify your current room');
  await expectDefaultTurnLimit(page);
  await expect(page.locator('.challenge-status')).toHaveText('0 / 1 goals complete');
  const camera = page.getByAltText('Authoritative robot head camera');
  await expect(camera).toHaveJSProperty('naturalWidth', 640);
  await expect.poll(async () => (await spectatorPixels(page)).colors).toBeGreaterThan(50);
  await page.locator('.camera-frame').screenshot({ path: 'test-results/kitchen-start-camera.png' });
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: 'test-results/kitchen-bathroom-desktop.png', fullPage: true });
  const initial: LiveState = await (await request.get('/api/state')).json();
  expect(initial.snapshot.poses.find(pose => pose.key === `${initial.robot_body_id}:-1`)!.position[1]).toBeLessThan(-1.4);
  const placement = await request.post('/api/robot/placement', { data: {
    run_id: initial.run_id, episode_epoch: initial.episode_epoch, observation_seq: initial.observation.seq, xy_m: [1.2, 1.4],
  } });
  expect(placement.ok()).toBe(true);
  const placed: LiveState = await placement.json();
  expect((await request.post('/api/command', { data: {
    run_id: placed.run_id, episode_epoch: placed.episode_epoch, observation_seq: placed.observation.seq,
    action_id: 'bathroom-settle', tool: 'wait', arguments: { duration_s: .5 },
  } })).ok()).toBe(true);
  await expect(page.locator('.challenge-status')).toHaveText('Completed');
  await expect(page.getByRole('region', { name: 'Robot activity', exact: true })).toContainText('Task completed');
  await expect(page.getByRole('region', { name: 'Robot activity', exact: true })).toContainText('Operator-assisted');
  await expect(camera).toHaveAttribute('data-simulated-time', '0.5');
  await page.locator('.camera-frame').screenshot({ path: 'test-results/bathroom-arrival-camera.png' });
  await page.setViewportSize({ width: 390, height: 844 });
  await expect.poll(async () => (await spectatorPixels(page)).colors).toBeGreaterThan(50);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: 'test-results/kitchen-bathroom-mobile.png', fullPage: true });
  if (await page.locator('.robot-options').getAttribute('open') === null) await page.locator('.robot-options > summary').click();
  await page.getByRole('button', { name: 'Reset episode', exact: true }).click();
  await expect(page.locator('.challenge-status')).toHaveText('0 / 1 goals complete');
  await expect(selector).toHaveValue('kitchen_bathroom');
  await expect(camera).toHaveAttribute('data-simulated-time', '0');
  const reset: LiveState = await (await request.get('/api/state')).json();
  expect(reset.snapshot.poses.find(pose => pose.key === `${reset.robot_body_id}:-1`)!.position[1]).toBeLessThan(-1.4);
});

test('apartment search loads on mobile and shows inspected-target completion and reset', async ({ page, request }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/');
  await page.getByRole('combobox', { name: 'Predefined challenge', exact: true }).selectOption('apartment');
  await page.getByRole('button', { name: 'Load challenge', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Apartment Search', exact: true })).toBeVisible();
  await expect(page.getByRole('textbox', { name: 'Robot goal', exact: true })).toContainText('yellow cube');
  await expectDefaultTurnLimit(page);
  await expect(page.locator('.challenge-status')).toHaveText('0 / 1 goals complete');
  await expect.poll(async () => (await spectatorPixels(page)).colors).toBeGreaterThan(20);
  await expect(page.getByAltText('Authoritative robot head camera')).toHaveJSProperty('naturalWidth', 640);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({ path: 'test-results/challenge-apartment-mobile.png', fullPage: true });
  const initial: LiveState = await (await request.get('/api/state')).json();
  const positioned = await request.post('/api/robot/placement', { data: {
    run_id: initial.run_id, episode_epoch: initial.episode_epoch, observation_seq: initial.observation.seq, xy_m: [1.2, 1.5],
  } });
  expect(positioned.ok()).toBe(true);
  const placed: LiveState = await positioned.json();
  expect(placed.challenge?.status).toBe('in_progress');
  const inspected = await request.post('/api/command', { data: {
    run_id: placed.run_id, episode_epoch: placed.episode_epoch, observation_seq: placed.observation.seq,
    action_id: 'inspect-apartment-target', tool: 'wait', arguments: { duration_s: 1.5 },
  } });
  expect((await inspected.json()).status).toBe('ok');
  await expect(page.locator('.challenge-status')).toHaveText('Completed');
  await expect(page.locator('.viewport-footer')).toContainText('Manual placements: 1');
  await expect(page.getByRole('region', { name: 'Robot activity', exact: true })).toContainText('Task completed');
  await expect(page.getByRole('region', { name: 'Robot activity', exact: true })).toContainText('Operator-assisted episode');
  await expect(page.getByAltText('Authoritative robot head camera')).toHaveAttribute('data-simulated-time', '1.5');
  await page.locator('.camera-frame').screenshot({ path: 'test-results/apartment-target-camera.png' });
  if (await page.locator('.robot-options').getAttribute('open') === null) await page.locator('.robot-options > summary').click();
  await page.getByRole('button', { name: 'Reset episode', exact: true }).click();
  await expect(page.locator('.challenge-status')).toHaveText('0 / 1 goals complete');
  await expect(page.getByRole('combobox', { name: 'Predefined challenge', exact: true })).toHaveValue('apartment');
  await expect(page.getByRole('region', { name: 'Robot activity', exact: true })).not.toContainText('Task completed');
});

test('parking challenge shows measured completion and mobile goals fit', async ({ page, request }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/');
  await page.getByRole('combobox', { name: 'Predefined challenge', exact: true }).selectOption('park');
  await page.getByRole('button', { name: 'Load challenge', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Park in the Bay', exact: true })).toBeVisible();
  for (const speed of [.3, .3, 0]) {
    const state = await (await request.get('/api/state')).json();
    const result = await request.post('/api/command', { data: {
      run_id: state.run_id, episode_epoch: state.episode_epoch, observation_seq: state.observation.seq,
      action_id: `parking-${state.observation.seq}`, tool: 'drive_base',
      arguments: { linear_mps: speed, angular_radps: 0, duration_s: speed ? 2 : .5 },
    } });
    expect((await result.json()).status).toBe('ok');
  }
  await expect(page.locator('.challenge-status')).toHaveText('Completed');
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({ path: 'test-results/challenge-mobile.png', fullPage: true });
  if (await page.locator('.robot-options').getAttribute('open') === null) await page.locator('.robot-options > summary').click();
  await page.getByRole('button', { name: 'Reset episode', exact: true }).click();
  await expect(page.locator('.challenge-status')).toHaveText('0 / 1 goals complete');
  await page.getByRole('combobox', { name: 'Predefined challenge', exact: true }).selectOption('sort');
  await page.getByRole('button', { name: 'Load challenge', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Color Sort', exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Load challenge', exact: true })).toBeEnabled();
  await expect(page.locator('.challenge-objectives li')).toHaveCount(2);
  await expect(page.locator('.challenge-status')).toHaveText('0 / 2 goals complete');
  await expect.poll(async () => (await spectatorPixels(page)).colors).toBeGreaterThan(20);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({ path: 'test-results/challenge-sort-mobile.png', fullPage: true });
});

test('remember and recharge exposes battery, wait, and a resettable mission', async ({ page, request }) => {
  await page.setViewportSize({ width: 1440, height: 1100 });
  await page.goto('/');
  await page.getByRole('combobox', { name: 'Predefined challenge', exact: true }).selectOption('recharge');
  await page.getByRole('button', { name: 'Load challenge', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Remember and Recharge', exact: true })).toBeVisible();
  await expect(page.getByRole('textbox', { name: 'Robot goal', exact: true })).toContainText('find your way back');
  await expectDefaultTurnLimit(page);
  await expect(page.locator('.challenge-status')).toHaveText('0 / 3 goals complete');
  const battery = page.getByRole('meter', { name: 'Battery level', exact: true });
  await expect(battery).toHaveAttribute('value', '100');
  const waitCompleted = page.waitForResponse(response => response.url().endsWith('/api/command')
    && response.request().postDataJSON()?.tool === 'wait', { timeout: 20000 });
  await page.getByRole('link', {name:'Controls',exact:true}).click();
  await page.getByRole('button', { name: 'Wait', exact: true }).click();
  expect((await (await waitCompleted).json()).status).toBe('ok');
  await expect(page.locator('.event-list')).toContainText('wait');
  await expect(battery).toHaveAttribute('value', '100');
  await page.getByRole('button', { name: 'Drive backward', exact: true }).click();
  await expect.poll(async () => Number(await battery.getAttribute('value')), { timeout: 20000 }).toBeLessThan(100);
  const state = await (await request.get('/api/state')).json();
  expect(Object.keys(state.observation.battery).sort()).toEqual(['charge_pct', 'charging', 'low']);
  expect(JSON.stringify(state.observation)).not.toContain('charger_beacon');
  if (await page.locator('.robot-options').getAttribute('open') === null) await page.locator('.robot-options > summary').click();
  await page.getByRole('button', { name: 'Reset episode', exact: true }).click();
  await expect(battery).toHaveAttribute('value', '100');
  await expect(page.locator('.challenge-status')).toHaveText('0 / 3 goals complete');
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await expect.poll(async () => (await spectatorPixels(page)).colors).toBeGreaterThan(20);
  await page.screenshot({ path: 'test-results/challenge-recharge-mobile.png', fullPage: true });
});

async function robotPickPoint(page: Page) {
  const canvas = page.locator('.spectator canvas');
  const point = await canvas.evaluate((element: HTMLCanvasElement) => {
    const context = element.getContext('webgl2')!;
    const pixels = new Uint8Array(element.width * element.height * 4);
    context.readPixels(0, 0, element.width, element.height, context.RGBA, context.UNSIGNED_BYTE, pixels);
    let target = { horizontal: 0, vertical: 0, span: 0 };
    for (let vertical = 0; vertical < element.height; vertical += 2) {
      let start = -1;
      for (let horizontal = 0; horizontal < element.width; horizontal += 2) {
        const offset = ((element.height - vertical - 1) * element.width + horizontal) * 4;
        const [red, green, blue] = pixels.slice(offset, offset + 3);
        const painted = red > green * 1.25 && blue > green * 1.1 && green > 40;
        if (painted && start < 0) start = horizontal;
        if (painted && horizontal - start > target.span) {
          target = { horizontal: (start + horizontal) / 2, vertical, span: horizontal - start };
        }
        if (!painted) start = -1;
      }
    }
    if (target.span < 6) throw new Error('No continuous robot body paint is visible');
    return { horizontal: target.horizontal / element.width, vertical: target.vertical / element.height };
  });
  const bounds = (await canvas.boundingBox())!;
  return { x: bounds.x + point.horizontal * bounds.width, y: bounds.y + point.vertical * bounds.height };
}

test.describe('manual robot placement', () => {
  test.use({ hasTouch: true });
  for (const touch of [false, true]) {
    test(`${touch ? 'touch' : 'mouse'} selection and drag commit the robot pose only on drop`, async ({ page, request }) => {
      await page.setViewportSize(touch ? { width: 390, height: 844 } : { width: 1440, height: 1000 });
      await page.goto('/');
      const select = page.getByRole('button', { name: 'Select robot', exact: true });
      const camera = page.getByAltText('Authoritative robot head camera');
      await expect(select).toBeEnabled();
      await expect(camera).toHaveJSProperty('naturalWidth', 640);
      await expect.poll(async () => (await spectatorPixels(page)).colors).toBeGreaterThan(20);
      await page.locator('.spectator canvas').scrollIntoViewIfNeeded();
      const initial: LiveState = await (await request.get('/api/state')).json();
      const initialFrame = await camera.getAttribute('src');
      const pick = await robotPickPoint(page);
      if (touch) await page.touchscreen.tap(pick.x, pick.y);
      else await page.mouse.click(pick.x, pick.y);
      await expect(select).toHaveAttribute('aria-pressed', 'true');
      await expect(page.locator('.placement-toolbar')).toContainText('Milo selected');
      const session = touch ? await page.context().newCDPSession(page) : null;
      if (session) await session.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [pick] });
      else { await page.mouse.move(pick.x, pick.y); await page.mouse.down(); }
      for (let step = 1; step <= 8; step++) {
        const target = { x: pick.x - (touch ? 42 : 90) * step / 8, y: pick.y + 15 * step / 8 };
        if (session) await session.send('Input.dispatchTouchEvent', { type: 'touchMove', touchPoints: [target] });
        else await page.mouse.move(target.x, target.y);
      }
      await expect(page.locator('.placement-toolbar')).toContainText('Placement preview');
      const beforeDrop: LiveState = await (await request.get('/api/state')).json();
      expect(beforeDrop.snapshot).toEqual(initial.snapshot);
      expect(await camera.getAttribute('src')).toBe(initialFrame);
      await page.locator('.spectator-shell').screenshot({ path: `test-results/placement-preview-${touch ? 'touch' : 'mouse'}.png` });
      const placementResponse = page.waitForResponse(response => response.url().endsWith('/api/robot/placement'));
      if (session) await session.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] });
      else await page.mouse.up();
      const response = await placementResponse;
      expect(response.status()).toBe(200);
      const placed: LiveState = await response.json();
      const originalBase = initial.snapshot.poses.find(pose => pose.key === `${initial.robot_body_id}:-1`)!;
      const movedBase = placed.snapshot.poses.find(pose => pose.key === `${placed.robot_body_id}:-1`)!;
      expect(Math.hypot(movedBase.position[0] - originalBase.position[0], movedBase.position[1] - originalBase.position[1])).toBeGreaterThan(.1);
      expect(movedBase.quaternion).toEqual(originalBase.quaternion);
      expect(placed.observation.seq).toBe(initial.observation.seq + 1);
      expect(placed.snapshot.simulated_time_s).toBe(initial.snapshot.simulated_time_s);
      await expect(page.locator('.viewport-footer')).toContainText('Manual placements: 1');
      await expect(camera).not.toHaveAttribute('src', initialFrame!);
      await expect(camera).toHaveJSProperty('naturalWidth', 640);
      await expect(select).toHaveAttribute('aria-pressed', 'false');
      await expect.poll(async () => (await spectatorPixels(page)).colors).toBeGreaterThan(20);
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
      await page.screenshot({ path: `test-results/placement-applied-${touch ? 'touch' : 'mouse'}.png`, fullPage: true });
      await session?.detach();
    });
  }

  test('keyboard placement supports cancel, collision rejection, stop, and reset', async ({ page, request }) => {
    await page.goto('/');
    const select = page.getByRole('button', { name: 'Select robot', exact: true });
    await expect(select).toBeEnabled();
    const original: LiveState = await (await request.get('/api/state')).json();
    await select.click();
    await page.keyboard.press('ArrowLeft');
    await expect(page.locator('.placement-toolbar')).toContainText('Placement preview');
    await page.keyboard.press('Escape');
    await expect(select).toHaveAttribute('aria-pressed', 'false');
    expect((await (await request.get('/api/state')).json()).snapshot).toEqual(original.snapshot);
    await select.click();
    for (let step = 0; step < 25; step++) await page.keyboard.press('ArrowRight');
    const blockedResponse = page.waitForResponse(response => response.url().endsWith('/api/robot/placement'));
    await page.keyboard.press('Enter');
    expect((await blockedResponse).status()).toBe(422);
    await expect(page.locator('.placement-error')).toContainText('overlaps an obstacle');
    expect((await (await request.get('/api/state')).json()).snapshot).toEqual(original.snapshot);
    await select.click();
    await page.keyboard.press('ArrowLeft');
    await page.keyboard.press('Enter');
    await expect(page.locator('.viewport-footer')).toContainText('Manual placements: 1');
    await select.click();
    await page.keyboard.press('ArrowLeft');
    await request.post('/api/stop');
    await expect(select).toBeDisabled();
    await expect(select).toHaveAttribute('aria-pressed', 'false');
    await page.keyboard.press('Enter');
    expect((await (await request.get('/api/state')).json()).manual_placements).toBe(1);
    await request.post('/api/reset');
    await expect(select).toBeEnabled();
    await expect(page.locator('.viewport-footer')).toContainText('MILO-01');
  });
});

test('token tracker shows exact run totals and retains stale readings until reconnect', async ({ page, request }) => {
  const initial: LiveState = await (await request.get('/api/state')).json();
  let current = initial;
  let liveSocket: WebSocketRoute | null = null;
  let connections = 0;
  await page.routeWebSocket('**/api/live', socket => {
    liveSocket = socket;
    if (++connections === 1) socket.send(JSON.stringify(current));
  });
  await page.route('**/api/agent/trace?*', route => route.fulfill({ json: {
    session_id: current.agent.session_id, revision: 0, first_id: 1, capacity: 200, events: [],
  } }));
  function publish(agent: Partial<LiveState['agent']>) {
    current = { ...initial, agent: { ...initial.agent, ...agent } };
    liveSocket!.send(JSON.stringify(current));
  }
  await page.goto('/');
  const tracker = page.getByRole('group', { name: 'Token usage', exact: true });
  await expect(tracker).toHaveCount(0);
  expect(await page.locator('.activity-summary .token-tracker').count()).toBe(0);
  const usage = { active: true, session_id: 'usage-run', phase: 'thinking', input_tokens: 1234, output_tokens: 567 };
  publish(usage);
  await expect(tracker).toHaveAttribute('title', /interrupted or unreported requests may be missing/);
  await expect(tracker.locator('.token-total dd')).toHaveText('1,801');
  await expect(tracker.locator('.token-input dd')).toHaveText('1,234');
  await expect(tracker.locator('.token-output dd')).toHaveText('567');
  await expect(tracker.locator('.token-scope')).toHaveText('Current run');
  publish({ ...usage, input_tokens: 1234567, output_tokens: 234567 });
  await expect(tracker.locator('.token-total dd')).toHaveText('1,469,134');
  for (const width of [1440, 390, 320]) {
    await page.setViewportSize({ width, height: width === 1440 ? 1000 : 844 });
    await tracker.scrollIntoViewIfNeeded();
    await expect(tracker).toBeVisible();
    await expect(page.getByRole('button', {name:'Stop',exact:true})).toBeInViewport();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    expect(await tracker.locator('dd').evaluateAll(elements => elements.every(element => element.scrollWidth <= element.clientWidth))).toBe(true);
    await tracker.screenshot({ path: `../.runtime/unified-controls-v1/token-tracker-${width}.png` });
  }
  liveSocket!.close();
  await expect(tracker.locator('.token-scope')).toHaveText('Last received');
  await expect(tracker.locator('.token-total dd')).toHaveText('1,469,134');
  await expect.poll(() => connections).toBe(2);
  await expect(tracker.locator('.token-scope')).toHaveText('Last received');
  publish({ ...usage, active: false, phase: 'completed' });
  await expect(tracker.locator('.token-scope')).toHaveText('Last run');
  await expect(tracker.locator('.token-total dd')).toHaveText('1,801');
  publish({ active: true, session_id: 'next-run', mode: 'voice', phase: 'listening' });
  await expect(tracker.locator('.token-total dd')).toHaveText('0');
  await expect(tracker.locator('.token-scope')).toHaveText('Current run');
  publish({});
  await expect(tracker).toHaveCount(0);
});

test('activity strip distinguishes execution, held phases, stop, and lost connection', async ({ page, request }) => {
  const initial: LiveState = await (await request.get('/api/state')).json();
  let liveSocket: WebSocketRoute | null = null;
  let connections = 0;
  await page.routeWebSocket('**/api/live', socket => {
    liveSocket = socket;
    connections++;
    if (connections === 1) socket.send(JSON.stringify(initial));
  });
  function publish(patch: Partial<LiveState> = {}, agent: Partial<LiveState['agent']> = {}) {
    liveSocket!.send(JSON.stringify({ ...initial, ...patch, agent: { ...initial.agent, ...agent } }));
  }
  await page.goto('/');
  const activity = page.getByRole('region', { name: 'Robot activity', exact: true });
  const title = activity.locator('.status');
  await expect(title).toHaveText('On / Idle');
  await expect(activity.getByRole('status')).toHaveAttribute('aria-atomic', 'true');
  expect(await activity.getByRole('status').locator('.activity-clock').count()).toBe(0);
  for (const width of [1440, 390, 320]) {
    await page.setViewportSize({ width, height: width === 1440 ? 1000 : 844 });
    publish({ busy: true }, { active: true, phase: 'acting' });
    await expect(title).toHaveText('Robot running');
    await expect(activity).toContainText('Executing command / LLM control');
    await expect(activity.locator('.activity-icon')).toHaveCSS('animation-name', 'activity-pulse');
    await page.locator('#controls').scrollIntoViewIfNeeded();
    await expect(page.getByRole('button', {name:'Stop',exact:true})).toBeInViewport();
    await activity.scrollIntoViewIfNeeded();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    expect(await activity.evaluate(element => {
      const banner = element.getBoundingClientRect();
      const summary = element.querySelector('.activity-summary')!.getBoundingClientRect();
      const clock = element.querySelector('.activity-clock')!.getBoundingClientRect();
      return summary.right <= clock.left && summary.bottom <= banner.bottom && clock.right <= banner.right;
    })).toBe(true);
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.screenshot({ path: `test-results/activity-running-${width}.png`, fullPage: true });
    publish({}, { active: true, phase: 'thinking' });
    await expect(title).toHaveText('Model thinking');
    await expect(activity).toContainText('Robot holding position');
    await expect(activity).toHaveAttribute('data-state', 'thinking');
    await expect(activity.locator('.activity-icon')).toHaveCSS('animation-name', 'activity-spin');
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  }
  for (const [phase, expected] of [
    ['waiting', 'Waiting for feedback'], ['starting', 'Starting controller'], ['acting', 'Preparing command'],
    ['voice_ready', 'Voice ready'], ['listening', 'Listening'], ['speaking', 'Milo speaking'],
  ]) {
    publish({}, { active: true, phase });
    await expect(title).toHaveText(expected);
    await expect(activity).toContainText('Robot holding position');
    await expect(activity).not.toHaveAttribute('data-state', 'running');
  }
  publish({}, { phase: 'completed' });
  await expect(title).toHaveText('On / Idle');
  publish({}, { phase: 'error' });
  await expect(title).toHaveText('Control error');
  publish({ stopped: true, busy: true }, { active: true, phase: 'acting' });
  await expect(title).toHaveText('Stopping robot');
  publish({ stopped: true }, { active: true, phase: 'thinking' });
  await expect(title).toHaveText('Robot stopped');
  await expect(activity).toContainText('Motion disabled');
  await page.emulateMedia({ reducedMotion: 'reduce' });
  publish({ busy: true });
  await expect(title).toHaveText('Robot running');
  await expect(activity.locator('.activity-icon')).toHaveCSS('animation-name', 'none');
  liveSocket!.close();
  await expect(title).toHaveText('Connection lost');
  await expect(activity).toContainText('Robot state unavailable');
  await expect(activity.locator('.activity-clock strong')).toHaveText('--');
  await expect.poll(() => connections).toBe(2);
  await expect(title).toHaveText('Connection lost');
  publish();
  await expect(title).toHaveText('On / Idle');
});

for (const viewport of [{ name: 'desktop', width: 1440, height: 1000 }, { name: 'mobile', width: 390, height: 844 }]) {
  test(`LLM actions update camera and 3D pixels before completion on ${viewport.name}`, async ({ page, request }) => {
    const observationSequences = new Set<number>();
    page.on('websocket', socket => {
      if (socket.url().endsWith('/api/live')) socket.on('framereceived', event => {
        const state = JSON.parse(String(event.payload));
        if (state.busy && state.agent.active && state.agent.phase === 'acting') observationSequences.add(state.observation.seq);
      });
    });
    let requestsInFlight = 0;
    let maximumRequests = 0;
    if (viewport.name === 'mobile') await page.route('**/api/camera/**', async route => {
      requestsInFlight++;
      maximumRequests = Math.max(maximumRequests, requestsInFlight);
      try {
        const response = await route.fetch();
        await new Promise(resolve => setTimeout(resolve, 180));
        await route.fulfill({ response });
      } finally { requestsInFlight--; }
    });
    await request.post('/api/agent/config', { data: { endpoint: 'https://test.openai.azure.com', models: [
      { id: 'luna', label: 'Scripted motion fixture', deployment: 'test-only' },
    ] } });
    await page.setViewportSize({ width: viewport.width, height: viewport.height });
    await page.goto('/');
    await expect(page.getByAltText('Authoritative robot head camera')).toHaveJSProperty('naturalWidth', 640);
    await expect.poll(async () => (await spectatorPixels(page)).colors).toBeGreaterThan(20);
    await page.locator('.view-grid').scrollIntoViewIfNeeded();
    await page.evaluate(() => {
      const probe = window as VisualProbe;
      probe.visualSamples = [];
      const imageCanvas = document.createElement('canvas');
      imageCanvas.width = 64;
      imageCanvas.height = 48;
      const imageContext = imageCanvas.getContext('2d')!;
      let lastFrame = '';
      probe.visualTimer = window.setInterval(() => {
        const image = document.querySelector<HTMLImageElement>('.camera-frame img');
        const canvas = document.querySelector<HTMLCanvasElement>('.spectator canvas');
        if (!image?.complete || !image.naturalWidth || !canvas || document.querySelector('.robot-activity')?.getAttribute('data-state') !== 'running') return;
        const frame = image.dataset.frame!;
        const time = Number(image.dataset.simulatedTime);
        if (frame === lastFrame || time <= 0 || time >= 2) return;
        lastFrame = frame;
        imageContext.drawImage(image, 0, 0, 64, 48);
        const camera = Array.from(imageContext.getImageData(0, 0, 64, 48).data).join(',');
        const context = canvas.getContext('webgl2')!;
        const pixels = new Uint8Array(canvas.width * canvas.height * 4);
        context.readPixels(0, 0, canvas.width, canvas.height, context.RGBA, context.UNSIGNED_BYTE, pixels);
        const colors = [];
        for (let offset = 0; offset < pixels.length; offset += 160) colors.push(`${pixels[offset]},${pixels[offset + 1]},${pixels[offset + 2]}`);
        probe.visualSamples.push({ frame, time, camera, spectator: colors.join(';') });
      }, 40);
    });
    const initial = await (await request.get('/api/state')).json();
    expect((await request.post('/api/agent/start', { data: {
      run_id: initial.run_id, episode_epoch: initial.episode_epoch, goal: 'Move forward and turn gently.',
      feedback_interval_s: 10,
    } })).ok()).toBe(true);
    await expect.poll(() => page.evaluate(() => (window as VisualProbe).visualSamples.length)).toBeGreaterThanOrEqual(3);
    const during = await (await request.get('/api/state')).json();
    expect(during.busy).toBe(true);
    expect(during.agent.phase).toBe('acting');
    await expect(page.locator('.status')).toHaveText('Robot running');
    await expect(page.locator('.activity-detail')).toHaveText('Executing command / LLM control');
    const samples = await page.evaluate(() => (window as VisualProbe).visualSamples);
    expect(new Set(samples.map(sample => sample.camera)).size).toBeGreaterThan(1);
    expect(new Set(samples.map(sample => sample.spectator)).size).toBeGreaterThan(1);
    expect(samples.every((sample, index) => index === 0 || sample.time > samples[index - 1].time)).toBe(true);
    await page.locator('.view-grid').screenshot({ path: `test-results/live-motion-${viewport.name}.png` });
    await expect.poll(async () => (await (await request.get('/api/state')).json()).busy).toBe(false);
    await expect(page.locator('.status')).toHaveText('Waiting for feedback');
    await expect(page.locator('.activity-detail')).toHaveText('Robot holding position');
    await expect(page.getByAltText('Authoritative robot head camera')).toHaveAttribute('data-simulated-time', '2');
    expect(observationSequences.size).toBe(1);
    expect((await (await request.get('/api/agent')).json()).turns).toBe(1);
    if (viewport.name === 'mobile') expect(maximumRequests).toBe(1);
    await page.evaluate(() => clearInterval((window as VisualProbe).visualTimer));
    await request.post('/api/stop');
    await request.post('/api/reset');
    await expect(page.getByAltText('Authoritative robot head camera')).toHaveAttribute('data-simulated-time', '0');
    await expect(page.getByAltText('Authoritative robot head camera')).toHaveJSProperty('naturalWidth', 640);
  });
}

test('manual control, authoritative camera isolation, and immediate stop', async ({ page, request }) => {
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.goto('/');
  await expect(page.getByRole('button', { name: 'Drive forward', exact: true })).toBeHidden();
  const camera = page.getByAltText('Authoritative robot head camera');
  await expect(camera).toBeVisible();
  await expect(camera).toHaveJSProperty('naturalWidth', 640);
  const initialFrame = await camera.getAttribute('src');
  const canvas = page.locator('.spectator canvas');
  const bounds = await canvas.boundingBox();
  expect(bounds?.width).toBeGreaterThan(500);
  await expect.poll(async () => (await spectatorPixels(page)).colors).toBeGreaterThan(20);
  const initialPixels = await spectatorPixels(page);
  await page.mouse.move(bounds!.x + 250, bounds!.y + 160);
  await page.mouse.down();
  await page.mouse.move(bounds!.x + 330, bounds!.y + 185, { steps: 10 });
  await page.mouse.up();
  await expect.poll(async () => (await spectatorPixels(page)).signature).not.toBe(initialPixels.signature);
  expect(await camera.getAttribute('src')).toBe(initialFrame);
  const beforeDrive = await (await request.get('/api/state')).json();
  await page.getByRole('link', {name:'Controls',exact:true}).click();
  await page.getByRole('button', { name: 'Drive forward', exact: true }).click();
  await expect(page.locator('.status')).toHaveText('Robot running');
  await expect(page.locator('.activity-detail')).toHaveText('Executing command / Manual control');
  await expect(page.locator('.event-list')).toContainText('drive_base');
  await expect(camera).not.toHaveAttribute('src', initialFrame!);
  const afterDrive = await (await request.get('/api/state')).json();
  expect(afterDrive.observation.odometry_m_rad[0] - beforeDrive.observation.odometry_m_rad[0]).toBeGreaterThan(.1);
  await page.getByRole('tab', { name: 'Head', exact: true }).click();
  await page.getByRole('button', { name: 'Set head', exact: true }).click();
  await expect(page.locator('.event-list')).toContainText('set_head');
  await page.getByRole('button', { name: 'Stop', exact: true }).click();
  await expect(page.locator('.status')).toHaveText('Robot stopped');
  if (await page.locator('.robot-options').getAttribute('open') === null) await page.locator('.robot-options > summary').click();
  await page.getByRole('button', { name: 'Enable manual control', exact: true }).click();
  await expect(page.locator('.status')).toHaveText('On / Idle');
  await page.screenshot({ path: 'test-results/desktop.png', fullPage: true });
});

test('mobile layout has no horizontal overflow and renders nonblank pixels', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/');
  await expect(page.locator('.spectator canvas')).toBeVisible();
  await page.getByRole('link', {name:'Controls',exact:true}).click();
  await page.getByRole('tab', { name: 'Arms & grippers', exact: true }).click();
  const dimensions = await page.evaluate(() => ({ viewport: innerWidth, page: document.documentElement.scrollWidth }));
  expect(dimensions.page).toBeLessThanOrEqual(dimensions.viewport);
  await expect(page.getByAltText('Authoritative robot head camera')).toHaveJSProperty('naturalWidth', 640);
  await expect.poll(async () => (await spectatorPixels(page)).colors).toBeGreaterThan(20);
  await page.screenshot({ path: 'test-results/mobile.png', fullPage: true });
});