import { test, expect, type Page, type WebSocketRoute } from '@playwright/test';
import type { LiveState } from '../src/types';

type VisualSample = { frame: string; time: number; camera: string; spectator: string };
type VisualProbe = Window & { visualSamples: VisualSample[]; visualTimer: number };
type ClipboardProbe = Window & { copiedExchange: string; rejectCopy: boolean };

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
  const response = await request.post('/api/challenges/load', { data: { challenge_id: 'bench' } });
  expect(response.ok()).toBeTruthy();
  await request.post('/api/agent/config', { data: { endpoint: '', models: [
    { id: 'luna', label: 'GPT-5.6 Luna', deployment: '' },
  ] } });
  await request.post('/api/voice/config', { data: { endpoint: '', deployment: '' } });
});

test('request context indicator distinguishes history estimates from actual request tokens', async ({ page, request }) => {
  const initial: LiveState = await (await request.get('/api/state')).json();
  let socket: WebSocketRoute;
  await page.routeWebSocket('**/api/live', connection => { socket = connection; connection.send(JSON.stringify(initial)); });
  await page.setViewportSize({ width: 1440, height: 1100 });
  await page.goto('/');
  const panel = page.getByRole('region', { name: 'Request context', exact: true });
  await expect(panel).toContainText('No request measurements available');
  const usage = { turn: 7, observation_seq: 12, retained_tokens_estimate: 6144, retained_budget: 8192,
    retained_turns: 2, images_sent: 2, image_limit: 3, input_tokens: 15321, response_received: true,
    active_command: 'Pick up the red cube. Keep the base parked and avoid the blue object.', instructions_repeated: true };
  const publish = (context: typeof usage | null, connected = true) => {
    socket.send(JSON.stringify({ ...initial, agent: { ...initial.agent, input_tokens: 900000,
      context_usage: context } }));
    if (!connected) socket.close();
  };
  publish(usage);
  await expect(panel).toContainText('6,144 / 8,192 tokens');
  await expect(panel).toContainText('15,321 tokens');
  await expect(panel).not.toContainText('900,000');
  await expect(panel.getByRole('meter', { name: 'Retained history budget', exact: true })).toHaveAttribute('value', '6144');
  await expect(panel.getByRole('meter')).toHaveAttribute('max', '8192');
  await expect(panel).toContainText('Full model window: not configured');
  await page.getByRole('spinbutton', { name: 'Retained context (tokens)', exact: true }).fill('12000');
  await expect(panel.getByRole('meter')).toHaveAttribute('max', '8192');
  await panel.locator('summary').click();
  await expect(panel).toContainText(usage.active_command);
  await panel.screenshot({ path: 'test-results/context-indicator-desktop.png' });
  for (const width of [390, 320]) {
    await page.setViewportSize({ width, height: 844 });
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await panel.screenshot({ path: `test-results/context-indicator-${width}.png` });
  }
  socket.send(JSON.stringify({ ...initial, agent: { ...initial.agent,
    context_usage: { ...usage, turn: 8, input_tokens: null, response_received: false } } }));
  await expect(panel).toContainText('Not yet reported');
  socket.send(JSON.stringify({ ...initial, agent: { ...initial.agent,
    context_usage: { ...usage, input_tokens: null, response_received: true } } }));
  await expect(panel).toContainText('Not reported');
  publish({ ...usage, retained_budget: 0, retained_tokens_estimate: 0, retained_turns: 0 });
  await expect(panel).toContainText('History disabled');
  await expect(panel.getByRole('meter')).toHaveCount(0);
  publish(null);
  await expect(panel).toContainText('No request measurements available');
  publish(usage, false);
  await expect(panel).toContainText('Last received');
});

test('request context telemetry follows real scripted inputs and resets with the episode', async ({ page, request }) => {
  await request.post('/api/agent/config', { data: { endpoint: 'https://test.openai.azure.com', models: [
    { id: 'luna', label: 'GPT-5.6 Luna', deployment: 'scripted-supervisor' },
  ] } });
  await page.goto('/');
  await page.getByRole('spinbutton', { name: 'Images per request', exact: true }).fill('2');
  await page.getByRole('spinbutton', { name: 'Retained context (tokens)', exact: true }).fill('8192');
  const command = 'Retain this original task during the scripted movement';
  await page.getByRole('textbox', { name: 'Robot goal', exact: true }).fill(command);
  await page.getByRole('button', { name: 'Start LLM control', exact: true }).click();
  const panel = page.getByRole('region', { name: 'Request context', exact: true });
  await expect.poll(async () => ((await (await request.get('/api/state')).json()) as LiveState).agent.context_usage?.turn).toBe(2);
  const state: LiveState = await (await request.get('/api/state')).json();
  expect(state.agent.context_usage?.retained_budget).toBe(8192);
  expect(state.agent.context_usage?.images_sent).toBe(2);
  expect(state.agent.context_usage?.retained_tokens_estimate).toBeGreaterThan(0);
  expect(state.agent.context_usage?.input_tokens).toBeNull();
  expect(state.agent.context_usage?.active_command).toBe(command);
  await expect(panel).toContainText('Not yet reported');
  await page.getByRole('button', { name: 'Take manual control', exact: true }).click();
  await expect(panel).toContainText('Request 2');
  await page.getByRole('button', { name: 'Reset episode', exact: true }).click();
  await expect(panel).toContainText('No request measurements available');
});

test('offline policy readiness blocks start and chat and preserves single-step control', async ({ page, request }) => {
  await request.post('/api/agent/config', { data: { endpoint: 'https://test.openai.azure.com', models: [
    { id: 'luna', label: 'GPT-5.6 Luna', deployment: 'scripted-supervisor' },
  ] } });
  let ready = false;
  let checks = 0;
  await page.route('**/api/policy/check', route => {
    checks += 1;
    return route.fulfill({ json: { ready, status: ready ? 'ready' : 'unavailable',
      message: ready ? 'Compatible policy server connected; task success is not verified.' : 'SmolVLA server is offline. A local server and Milo-trained checkpoint are required.',
      metadata: ready ? { checkpoint: 'scripted-test-only' } : null } });
  });
  await page.goto('/');
  await page.getByRole('button', { name: 'Luna + SmolVLA', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Left-arm policy connection', exact: true })).toBeVisible();
  await expect(page.locator('.policy-readiness')).toContainText('navigation checkpoints are currently CLI-only');
  const status = page.getByRole('status', { name: 'Policy connection status', exact: true });
  await expect(status).toContainText('offline');
  await expect(page.getByRole('button', { name: 'Start LLM control', exact: true })).toBeDisabled();
  await page.getByRole('textbox', { name: 'Chat message', exact: true }).fill('Pick up the cube');
  await expect(page.getByRole('button', { name: 'Send chat message', exact: true })).toBeDisabled();
  const state: LiveState = await (await request.get('/api/state')).json();
  expect(state.agent.turns).toBe(0);
  expect(state.agent.session_id).toBeNull();
  expect(state.agent.error).toBeNull();
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.locator('.policy-readiness').screenshot({ path: 'test-results/policy-offline-mobile.png' });
  await page.getByRole('button', { name: 'Single step', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Start LLM control', exact: true })).toBeEnabled();
  await expect(page.getByRole('button', { name: 'Send chat message', exact: true })).toBeEnabled();
  await page.getByRole('button', { name: 'Luna + SmolVLA', exact: true }).click();
  await expect(status).toContainText('offline');
  ready = true;
  await page.getByRole('button', { name: 'Check policy connection', exact: true }).click();
  await expect(status).toContainText('Compatible policy');
  await expect(page.getByRole('button', { name: 'Start LLM control', exact: true })).toBeEnabled();
  ready = false;
  await page.getByRole('textbox', { name: 'SmolVLA endpoint', exact: true }).fill('http://127.0.0.1:8086');
  await expect(status).toContainText('offline');
  await expect(page.getByRole('button', { name: 'Start LLM control', exact: true })).toBeDisabled();
  expect(checks).toBeGreaterThanOrEqual(4);
});

test('supervised policy mode runs scripted motion and cancels both models on desktop and mobile', async ({ page, request }) => {
  await request.post('/api/agent/config', { data: { endpoint: 'https://test.openai.azure.com', models: [
    { id: 'luna', label: 'GPT-5.6 Luna', deployment: 'scripted-supervisor', reasoning_efforts: ['none', 'low'] },
  ] } });
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.setViewportSize({ width: 1440, height: 1100 });
  await page.goto('/');
  await page.getByRole('button', { name: 'Luna + SmolVLA', exact: true }).click();
  await expect(page.getByRole('textbox', { name: 'SmolVLA endpoint', exact: true })).toHaveValue('http://127.0.0.1:8085');
  await expect(page.getByRole('status', { name: 'Policy skill status', exact: true })).toHaveText('Milo-trained checkpoint required');
  await page.getByRole('textbox', { name: 'Robot goal', exact: true }).fill('Scripted policy transport test');
  await page.getByRole('button', { name: 'Start LLM control', exact: true }).click();
  await expect.poll(async () => ((await (await request.get('/api/state')).json()) as LiveState).skill?.policy_requests, { timeout: 15000 }).toBeGreaterThan(0);
  await expect.poll(async () => ((await (await request.get('/api/state')).json()) as LiveState).snapshot.simulated_time_s).toBeGreaterThan(.2);
  await expect(page.getByRole('textbox', { name: 'SmolVLA endpoint', exact: true })).toBeDisabled();
  await expect(page.getByRole('button', { name: 'Single step', exact: true })).toBeDisabled();
  await page.getByRole('button', { name: 'Policy', exact: true }).click();
  await expect(page.locator('.exchange-policy').first()).toBeVisible();
  await expect.poll(async () => (await spectatorPixels(page)).colors).toBeGreaterThan(20);
  await expect(page.getByAltText('Authoritative robot head camera')).toHaveJSProperty('naturalWidth', 320);
  await expect(page.locator('.camera-meta')).toContainText('320 x 240');
  await page.screenshot({ path: 'test-results/supervised-policy-desktop.png', fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await expect.poll(async () => (await spectatorPixels(page)).colors).toBeGreaterThan(20);
  await page.screenshot({ path: 'test-results/supervised-policy-mobile.png', fullPage: true });
  await page.getByRole('button', { name: 'Cancel policy skill', exact: true }).click();
  await expect.poll(async () => ((await (await request.get('/api/state')).json()) as LiveState).agent.active).toBe(false);
  const stopped: LiveState = await (await request.get('/api/state')).json();
  expect(stopped.skill?.remaining_s).toBe(0);
  expect(stopped.skill?.status).toBe('cancelled');
  await page.getByRole('button', { name: 'Reset episode', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Single step', exact: true })).toHaveAttribute('aria-pressed', 'true');
  expect(errors).toEqual([]);
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
  const outcome = page.getByRole('status', { name: 'Task outcome', exact: true });
  await expect(outcome).toHaveCount(0);
  for (const [kind, title] of [['completed', 'Agent reports completion'], ['unachievable', 'Task cannot be completed'],
    ['limited', 'Turn limit reached'], ['interrupted', 'Run interrupted'], ['ended', 'Response finished']]) {
    publish({}, { outcome: { kind, message: 'A clear reason for this outcome.', source: kind === 'completed' || kind === 'unachievable' ? 'agent' : 'controller', timestamp: 1 } });
    await expect(outcome.getByRole('heading')).toHaveText(title);
    await expect(outcome).not.toHaveAttribute('data-outcome', 'success');
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
  await expect(outcome.getByRole('heading')).toHaveText('Task completed');
  await expect(outcome).toHaveAttribute('data-outcome', 'success');
  await expect(outcome).toContainText('verified by physics / Operator-assisted episode');
  for (const width of [1440, 390, 320]) {
    await page.setViewportSize({ width, height: 1000 });
    await page.locator('.controls-section').scrollIntoViewIfNeeded();
    await expect(outcome).toBeVisible();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await page.locator('.robot-activity').screenshot({ path: `test-results/task-completed-${width}.png` });
  }
  publish({});
  await expect(outcome).toHaveCount(0);
  socket!.close();
  await expect(outcome).toContainText('Connection lost');
  await expect(outcome).not.toHaveAttribute('data-outcome', 'success');
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
    await expect(page.getByRole('spinbutton', { name: 'LLM turn limit', exact: true })).toHaveValue(String(preset.suggested_turn_limit));
    await expect(page.locator('.challenge-objectives li')).toHaveCount(preset.objectives.length);
    await expect(page.locator('.challenge-status')).toHaveText(`0 / ${preset.objectives.length} goals complete`);
    await expect.poll(async () => (await spectatorPixels(page)).colors).toBeGreaterThan(20);
    await expect(page.getByAltText('Authoritative robot head camera')).toHaveJSProperty('naturalWidth', 640);
    signatures.add((await spectatorPixels(page)).signature);
    await page.screenshot({ path: `test-results/challenge-${preset.id}.png`, fullPage: true });
    const loaded = await (await request.get('/api/state')).json();
    await page.getByRole('button', { name: 'Reset episode', exact: true }).click();
    await expect.poll(async () => (await (await request.get('/api/state')).json()).run_id).not.toBe(loaded.run_id);
    await expect(selector).toHaveValue(preset.id);
    await expect(page.getByRole('textbox', { name: 'Robot goal', exact: true })).toHaveValue(preset.goal);
  }
  expect(signatures.size).toBe(presets.length);
  await selector.selectOption('bench');
  await page.getByRole('button', { name: 'Load challenge', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Floor pickup & release', exact: true })).toBeVisible();
  await expect(page.locator('.challenge-status')).toHaveCount(0);
});

for (const width of [1440, 390]) {
  test(`advanced training grounds render and move at ${width}px`, async ({ page, request }) => {
    test.setTimeout(90000);
    await page.setViewportSize({ width, height: 1000 });
    await page.goto('/');
    const selector = page.getByRole('combobox', { name: 'Predefined challenge', exact: true });
    await expect(selector.locator('optgroup')).toHaveCount(3);
    for (const [identifier, title] of [['clinic_delivery', 'Clinic Supply Delivery'], ['warehouse', 'Warehouse Dispatch Circuit'],
      ['inspection', 'Service Gallery Inspection'], ['workshop', 'Cluttered Assembly Workshop']]) {
      await selector.selectOption(identifier);
      await page.getByRole('button', { name: 'Load challenge', exact: true }).click();
      await expect(page.getByRole('heading', { name: title, exact: true })).toBeVisible();
      await expect(page.locator('.challenge-toolbar')).toContainText('Advanced');
      await expect.poll(async () => (await spectatorPixels(page)).colors).toBeGreaterThan(30);
      await expect(page.getByAltText('Authoritative robot head camera')).toHaveJSProperty('naturalWidth', 640);
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
      await page.locator('.spectator-shell').screenshot({ path: `test-results/ground-${identifier}-${width}.png` });
      await page.locator('.camera-frame').screenshot({ path: `test-results/ground-camera-${identifier}-${width}.png` });
      if (identifier === 'warehouse') {
        const before: LiveState = await (await request.get('/api/state')).json();
        const pixelsBefore = (await spectatorPixels(page)).signature;
        const response = await request.post('/api/command', { data: {
          run_id: before.run_id, episode_epoch: before.episode_epoch, observation_seq: before.observation.seq,
          action_id: `warehouse-drive-${width}`, tool: 'drive_base', arguments: { linear_mps: .15, angular_radps: 0, duration_s: 1 },
        } });
        const result = await response.json();
        expect(result.status).toBe('ok');
        expect(result.observation.odometry_m_rad[0]).toBeGreaterThan(.05);
        await expect(page.getByAltText('Authoritative robot head camera')).toHaveAttribute('data-simulated-time', '1');
        await expect.poll(async () => (await spectatorPixels(page)).signature).not.toBe(pixelsBefore);
        await page.getByRole('button', { name: 'Reset episode', exact: true }).click();
        await expect(page.getByAltText('Authoritative robot head camera')).toHaveAttribute('data-simulated-time', '0');
        await expect(page.locator('.challenge-status')).toHaveText('0 / 3 goals complete');
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
  await expect(page.getByRole('spinbutton', { name: 'LLM turn limit', exact: true })).toHaveValue('100');
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
  await expect(page.getByRole('status', { name: 'Task outcome', exact: true })).toContainText('Task completed');
  await expect(page.getByRole('status', { name: 'Task outcome', exact: true })).toContainText('Operator-assisted');
  await expect(camera).toHaveAttribute('data-simulated-time', '0.5');
  await page.locator('.camera-frame').screenshot({ path: 'test-results/bathroom-arrival-camera.png' });
  await page.setViewportSize({ width: 390, height: 844 });
  await expect.poll(async () => (await spectatorPixels(page)).colors).toBeGreaterThan(50);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: 'test-results/kitchen-bathroom-mobile.png', fullPage: true });
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
  await expect(page.getByRole('spinbutton', { name: 'LLM turn limit', exact: true })).toHaveValue('100');
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
  await expect(page.getByRole('status', { name: 'Task outcome', exact: true })).toContainText('Task completed');
  await expect(page.getByRole('status', { name: 'Task outcome', exact: true })).toContainText('Operator-assisted episode');
  await expect(page.getByAltText('Authoritative robot head camera')).toHaveAttribute('data-simulated-time', '1.5');
  await page.locator('.camera-frame').screenshot({ path: 'test-results/apartment-target-camera.png' });
  await page.getByRole('button', { name: 'Reset episode', exact: true }).click();
  await expect(page.locator('.challenge-status')).toHaveText('0 / 1 goals complete');
  await expect(page.getByRole('combobox', { name: 'Predefined challenge', exact: true })).toHaveValue('apartment');
  await expect(page.getByRole('status', { name: 'Task outcome', exact: true })).toHaveCount(0);
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
  await expect(page.getByRole('spinbutton', { name: 'LLM turn limit', exact: true })).toHaveValue('100');
  await expect(page.locator('.challenge-status')).toHaveText('0 / 3 goals complete');
  const battery = page.getByRole('meter', { name: 'Battery level', exact: true });
  await expect(battery).toHaveAttribute('value', '100');
  await page.getByRole('button', { name: 'Wait', exact: true }).click();
  await expect(page.locator('.event-list')).toContainText('wait');
  await expect(battery).toHaveAttribute('value', '100');
  await page.getByRole('button', { name: 'Drive backward', exact: true }).click();
  await expect.poll(async () => Number(await battery.getAttribute('value'))).toBeLessThan(100);
  const state = await (await request.get('/api/state')).json();
  expect(Object.keys(state.observation.battery).sort()).toEqual(['charge_pct', 'charging', 'low']);
  expect(JSON.stringify(state.observation)).not.toContain('charger_beacon');
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
    const matches: { horizontal: number; vertical: number }[] = [];
    for (let vertical = 0; vertical < element.height; vertical += 2) {
      for (let horizontal = 0; horizontal < element.width; horizontal += 2) {
        const offset = ((element.height - vertical - 1) * element.width + horizontal) * 4;
        const [red, green, blue] = pixels.slice(offset, offset + 3);
        if (red > green * 1.25 && blue > green * 1.1 && green > 40) matches.push({ horizontal, vertical });
      }
    }
    const band = matches.filter(candidate => candidate.vertical >= matches[0].vertical + 4 && candidate.vertical <= matches[0].vertical + 12);
    const target = band[Math.floor(band.length / 2)];
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
  await expect(tracker.locator('.token-total dd')).toHaveText('0');
  await expect(tracker.locator('.token-scope')).toHaveText('No run yet');
  await expect(tracker).toHaveAttribute('title', /interrupted or unreported requests may be missing/);
  expect(await page.locator('.activity-summary .token-tracker').count()).toBe(0);
  const usage = { active: true, session_id: 'usage-run', phase: 'thinking', input_tokens: 1234, output_tokens: 567 };
  publish(usage);
  await expect(tracker.locator('.token-total dd')).toHaveText('1,801');
  await expect(tracker.locator('.token-input dd')).toHaveText('1,234');
  await expect(tracker.locator('.token-output dd')).toHaveText('567');
  await expect(tracker.locator('.token-scope')).toHaveText('Current run');
  publish({ ...usage, input_tokens: 1234567, output_tokens: 234567 });
  await expect(tracker.locator('.token-total dd')).toHaveText('1,469,134');
  for (const width of [1440, 390, 320]) {
    await page.setViewportSize({ width, height: width === 1440 ? 1000 : 844 });
    await page.locator('.controls-section').scrollIntoViewIfNeeded();
    await expect(tracker).toBeVisible();
    expect((await page.locator('.robot-activity').boundingBox())!.y).toBeCloseTo(0, 0);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    expect(await tracker.locator('dd').evaluateAll(elements => elements.every(element => element.scrollWidth <= element.clientWidth))).toBe(true);
    await page.locator('.robot-activity').screenshot({ path: `test-results/token-tracker-${width}.png` });
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
  await expect(tracker.locator('.token-total dd')).toHaveText('0');
  await expect(tracker.locator('.token-scope')).toHaveText('No run yet');
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
  await expect(title).toHaveText('Robot idle');
  await expect(activity.getByRole('status')).toHaveAttribute('aria-atomic', 'true');
  expect(await activity.getByRole('status').locator('.activity-clock').count()).toBe(0);
  for (const width of [1440, 390, 320]) {
    await page.setViewportSize({ width, height: width === 1440 ? 1000 : 844 });
    publish({ busy: true }, { active: true, phase: 'acting' });
    await expect(title).toHaveText('Robot running');
    await expect(activity).toContainText('Executing command / LLM control');
    await expect(activity.locator('.activity-icon')).toHaveCSS('animation-name', 'activity-pulse');
    await page.locator('.controls-section').scrollIntoViewIfNeeded();
    expect((await activity.boundingBox())!.y).toBeCloseTo(0, 0);
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
  await expect(title).toHaveText('Run finished');
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
  await expect(title).toHaveText('Robot idle');
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
  await expect(page.getByRole('button', { name: 'Drive forward', exact: true })).toBeEnabled();
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
  await page.getByRole('button', { name: 'Resume manual control', exact: true }).click();
  await expect(page.locator('.status')).toHaveText('Robot idle');
  await page.screenshot({ path: 'test-results/desktop.png', fullPage: true });
});

test('mobile layout has no horizontal overflow and renders nonblank pixels', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/');
  await expect(page.locator('.spectator canvas')).toBeVisible();
  await page.getByRole('tab', { name: 'Arms & grippers', exact: true }).click();
  const dimensions = await page.evaluate(() => ({ viewport: innerWidth, page: document.documentElement.scrollWidth }));
  expect(dimensions.page).toBeLessThanOrEqual(dimensions.viewport);
  await expect(page.getByAltText('Authoritative robot head camera')).toHaveJSProperty('naturalWidth', 640);
  await expect.poll(async () => (await spectatorPixels(page)).colors).toBeGreaterThan(20);
  await page.screenshot({ path: 'test-results/mobile.png', fullPage: true });
});

test('LLM selection, paced feedback, real tools, and manual takeover with scripted inference', async ({ page, request }) => {
  await page.setViewportSize({ width: 1440, height: 1100 });
  await page.goto('/');
  await expect(page.getByRole('combobox', { name: 'LLM model', exact: true })).toHaveValue('luna');
  await expect(page.getByRole('combobox', { name: 'Reasoning effort', exact: true })).toHaveValue('low');
  await expect(page.getByRole('spinbutton', { name: 'Feedback interval (s)', exact: true })).toHaveValue('2');
  await expect(page.getByRole('spinbutton', { name: 'Images per request', exact: true })).toHaveValue('1');
  await expect(page.getByRole('spinbutton', { name: 'Retained context (tokens)', exact: true })).toHaveValue('4096');
  await expect(page.getByRole('button', { name: 'Start LLM control', exact: true })).toBeDisabled();
  await page.locator('.agent-connection summary').click();
  await page.getByRole('textbox', { name: 'Foundry endpoint', exact: true }).fill('https://test.services.ai.azure.com/api/projects/test');
  await page.getByRole('textbox', { name: 'Foundry deployment name', exact: true }).fill('test-luna');
  await page.getByRole('button', { name: 'Apply connection', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Start LLM control', exact: true })).toBeEnabled();
  await page.getByRole('button', { name: 'New model', exact: true }).click();
  await page.getByRole('textbox', { name: 'Model label', exact: true }).fill('Alternate vision model');
  await page.getByRole('textbox', { name: 'Foundry deployment name', exact: true }).fill('test-alternate');
  await page.getByRole('button', { name: 'Add model', exact: true }).click();
  await expect(page.getByRole('combobox', { name: 'LLM model', exact: true }).locator('option:checked')).toHaveText('Alternate vision model');
  await page.getByRole('combobox', { name: 'LLM model', exact: true }).selectOption('luna');
  await page.getByRole('textbox', { name: 'Robot goal', exact: true }).fill('Move forward once, then observe.');
  await page.getByRole('spinbutton', { name: 'Images per request', exact: true }).fill('2');
  await page.getByRole('spinbutton', { name: 'Retained context (tokens)', exact: true }).fill('12000');
  await page.getByRole('spinbutton', { name: 'Feedback interval (s)', exact: true }).fill('10');
  const startRequest = page.waitForRequest(request => request.url().endsWith('/api/agent/start'));
  await page.getByRole('button', { name: 'Start LLM control', exact: true }).click();
  expect((await startRequest).postDataJSON()).toMatchObject({ model_id: 'luna', reasoning: 'low', feedback_interval_s: 10, goal: 'Move forward once, then observe.', images_per_request: 2, context_tokens: 12000 });
  await expect(page.getByRole('spinbutton', { name: 'Images per request', exact: true })).toBeDisabled();
  await expect(page.getByRole('spinbutton', { name: 'Retained context (tokens)', exact: true })).toBeDisabled();
  await expect(page.getByRole('button', { name: 'Drive forward', exact: true })).toBeDisabled();
  await expect(page.getByRole('button', { name: 'Observe', exact: true })).toBeDisabled();
  await expect(page.getByRole('button', { name: 'Select robot', exact: true })).toBeDisabled();
  await expect(page.getByRole('combobox', { name: 'LLM model', exact: true })).toBeDisabled();
  const feed = page.getByRole('log', { name: 'Robot and LLM exchanges' });
  await expect(feed.locator('[data-kind="result"]')).toContainText('drive_base');
  const tracker = page.getByRole('group', { name: 'Token usage', exact: true });
  await expect(tracker.locator('.token-total dd')).toHaveText('15');
  await expect(tracker.locator('.token-input dd')).toHaveText('10');
  await expect(tracker.locator('.token-output dd')).toHaveText('5');
  await expect(tracker.locator('.token-scope')).toHaveText('Current run');
  expect(await feed.locator('article').evaluateAll(entries => entries.map(entry => entry.getAttribute('data-kind')))).toEqual(['session', 'feedback', 'response', 'tool', 'result']);
  await expect(feed.locator('[data-kind="response"]')).toContainText('Taking a short step');
  await page.getByRole('checkbox', { name: 'Follow latest', exact: true }).uncheck();
  await feed.evaluate(element => { element.scrollTop = 0; });
  const moved = await (await request.get('/api/state')).json();
  expect(moved.observation.odometry_m_rad[0]).toBeGreaterThan(.02);
  await page.getByRole('spinbutton', { name: 'Feedback interval (s)', exact: true }).fill('.5');
  await page.getByRole('button', { name: 'Apply rate', exact: true }).click();
  await expect(page.locator('.agent-phase')).toHaveText('Awaiting model');
  await expect(page.locator('.status')).toHaveText('Model thinking');
  await expect(page.locator('.activity-detail')).toHaveText('Robot holding position');
  expect((await (await request.get('/api/agent')).json()).feedback_interval_s).toBe(.5);
  await expect(tracker.locator('.token-total dd')).toHaveText('15');
  await expect(feed.locator('[data-kind="feedback"]')).toHaveCount(2);
  const latestInput = feed.locator('[data-kind="feedback"]').last();
  await expect(latestInput).toContainText('2 image(s) in request');
  await expect(latestInput).toContainText('/ 12000 tokens (est.)');
  await expect(latestInput.getByRole('img')).toHaveCount(2);
  await latestInput.getByRole('img', { name: /Historical input camera/ }).scrollIntoViewIfNeeded();
  await expect(latestInput.getByRole('img', { name: /Historical input camera/ })).toHaveJSProperty('naturalWidth', 640);
  await latestInput.screenshot({ path: 'test-results/camera-batch-desktop.png' });
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await latestInput.screenshot({ path: 'test-results/camera-batch-mobile.png' });
  await page.setViewportSize({ width: 1440, height: 1100 });
  await feed.evaluate(element => { element.scrollTop = 0; });
  expect(await feed.evaluate(element => element.scrollTop)).toBe(0);
  await page.getByRole('button', { name: 'Inputs', exact: true }).click();
  await expect(feed.locator('article')).toHaveCount(2);
  const firstInput = feed.locator('[data-kind="feedback"]').first();
  const inputImage = firstInput.getByRole('img');
  await inputImage.scrollIntoViewIfNeeded();
  await expect(inputImage).toHaveJSProperty('naturalWidth', 640);
  await firstInput.getByText('Sensor payload', { exact: true }).click();
  await expect(firstInput.locator('.exchange-payload').first().locator('pre')).toContainText('"grippers"');
  await expect(firstInput.locator('.exchange-payload').first().locator('pre')).not.toContainText('robot_position');
  expect(await firstInput.locator('.exchange-payload').first().locator('pre').textContent()).not.toMatch(/\d+\.\d{4}/);
  await firstInput.getByText('Sensor payload', { exact: true }).click();
  await expect(feed.locator('[data-kind="feedback"]').last()).toContainText('call-1');
  await page.getByRole('button', { name: 'LLM', exact: true }).click();
  await expect(feed.locator('article')).toHaveCount(1);
  await expect(feed).toContainText('10 in / 5 out tokens');
  await page.getByRole('button', { name: 'Tools', exact: true }).click();
  await expect(feed.locator('article')).toHaveCount(2);
  await feed.getByText('Result payload', { exact: true }).click();
  const resultEntry = feed.locator('[data-kind="result"]');
  await expect(resultEntry.locator('details').filter({ hasText: 'Result payload' }).locator('pre')).toContainText('"odometry_m_rad"');
  await resultEntry.getByText('Reply sent to model', { exact: true }).click();
  const modelReply = resultEntry.locator('details').filter({ hasText: 'Reply sent to model' }).locator('pre');
  await expect(modelReply).toContainText('"observation_seq"');
  await expect(modelReply).not.toContainText('"observation":');
  expect(await modelReply.textContent()).not.toMatch(/\d+\.\d{4}/);
  await page.getByRole('button', { name: 'All', exact: true }).click();
  await feed.locator('[data-kind="feedback"]').first().scrollIntoViewIfNeeded();
  await page.locator('.exchange-section').screenshot({ path: 'test-results/exchange-desktop.png' });
  await page.getByRole('checkbox', { name: 'Follow latest', exact: true }).check();
  await expect.poll(async () => feed.evaluate(element => element.scrollHeight - element.clientHeight - element.scrollTop)).toBeLessThan(2);
  await page.locator('.agent-section').screenshot({ path: 'test-results/llm-desktop.png' });
  await page.getByRole('button', { name: 'Take manual control', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Drive forward', exact: true })).toBeEnabled();
  expect((await (await request.get('/api/agent')).json()).active).toBe(false);
  await expect(feed).toContainText('Manual takeover');
  await expect(tracker.locator('.token-total dd')).toHaveText('15');
  await expect(tracker.locator('.token-scope')).toHaveText('Last run');
  await page.getByRole('button', { name: 'Start LLM control', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Take manual control', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Stop', exact: true }).click();
  await expect(page.locator('.status')).toHaveText('Robot stopped');
  await expect(page.getByRole('button', { name: 'Start LLM control', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Reset episode', exact: true }).click();
  await expect(feed).toContainText('No exchanges in this session.');
  await expect(feed.locator('article')).toHaveCount(0);
  await expect(tracker.locator('.token-total dd')).toHaveText('0');
  await expect(tracker.locator('.token-scope')).toHaveText('No run yet');
});

test('mobile LLM connection and controls stay within the viewport', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/');
  await page.locator('.agent-connection summary').click();
  const dimensions = await page.evaluate(() => ({ viewport: innerWidth, page: document.documentElement.scrollWidth }));
  expect(dimensions.page).toBeLessThanOrEqual(dimensions.viewport);
  for (const control of await page.locator('.agent-section input:visible, .agent-section select:visible, .agent-section textarea:visible, .agent-section button:visible').all()) {
    const bounds = await control.boundingBox();
    expect(bounds!.x).toBeGreaterThanOrEqual(0);
    expect(bounds!.x + bounds!.width).toBeLessThanOrEqual(390);
  }
  await page.locator('.agent-section').screenshot({ path: 'test-results/llm-mobile.png' });
});

test('completed Chat idles with stable tokens and wakes on camera changes or user messages', async ({ page, request }) => {
  await request.post('/api/agent/config', { data: { endpoint: 'https://test.openai.azure.com', models: [
    { id: 'luna', label: 'Scripted idle fixture', deployment: 'test-idle' },
  ] } });
  await page.setViewportSize({ width: 1440, height: 1100 });
  await page.goto('/');
  const message = page.getByRole('textbox', { name: 'Chat message', exact: true });
  const send = page.getByRole('button', { name: 'Send chat message', exact: true });
  const tracker = page.getByRole('group', { name: 'Token usage', exact: true });
  await message.fill('What do you see?');
  await send.click();
  await expect(page.getByRole('log', { name: 'Chat conversation', exact: true })).toContainText('I can see the room');
  await expect(page.locator('.status')).toHaveText('Agent idle', { timeout: 9000 });
  await expect(tracker.locator('.token-total dd')).toHaveText('15');
  await expect(tracker.locator('.token-scope')).toHaveText('Idle run');
  const idle: LiveState = await (await request.get('/api/state')).json();
  expect(idle.agent.active).toBe(false);
  expect(idle.agent.auto_wake).toBe(true);
  expect(idle.agent.camera_unchanged_s).toBeGreaterThanOrEqual(5);
  expect(idle.snapshot.simulated_time_s).toBe(0);
  const idleObservation = idle.observation.seq;
  const axes = page.getByRole('checkbox', { name: 'Axes', exact: true });
  await axes.check();
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: 'test-results/agent-idle-desktop.png', fullPage: true });
  const stillIdle: LiveState = await (await request.get('/api/state')).json();
  expect(stillIdle.agent.turns).toBe(1);
  expect(stillIdle.observation.seq).toBe(idleObservation);
  const changed = await request.post('/api/command', { data: {
    run_id: idle.run_id, episode_epoch: idle.episode_epoch, observation_seq: idle.observation.seq,
    action_id: 'idle-camera-change', tool: 'set_head', arguments: { yaw_rad: .7, pitch_rad: .7, duration_s: .5 },
  } });
  expect((await changed.json()).status).toBe('ok');
  await expect(tracker.locator('.token-total dd')).toHaveText('30');
  const awakened: LiveState = await (await request.get('/api/state')).json();
  expect(awakened.agent.session_id).toBe(idle.agent.session_id);
  expect(awakened.agent.turns).toBe(2);
  await expect(page.getByRole('log', { name: 'Robot and LLM exchanges' })).toContainText('Agent waking');
  await expect(page.locator('.status')).toHaveText('Agent idle', { timeout: 9000 });
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: 'test-results/agent-idle-mobile.png', fullPage: true });
  await message.fill('What changed?');
  await expect(send).toBeEnabled();
  await send.click();
  await expect(page.getByRole('log', { name: 'Chat conversation', exact: true })).toContainText('What changed?');
  await expect.poll(async () => (await (await request.get('/api/state')).json()).agent.session_id).not.toBe(idle.agent.session_id);
  await request.post('/api/stop');
  await expect(page.locator('.status')).toHaveText('Robot stopped');
  expect((await (await request.get('/api/state')).json()).agent.auto_wake).toBe(false);
});

test('Chat and Voice are separate modes with text followups and safe switching', async ({ page, request }) => {
  await request.post('/api/agent/config', { data: { endpoint: 'https://test.openai.azure.com', models: [
    { id: 'luna', label: 'Scripted chat', deployment: 'test-chat' },
  ] } });
  await page.setViewportSize({ width: 1440, height: 1100 });
  await page.goto('/');
  const chat = page.getByRole('tab', { name: 'Chat', exact: true });
  const voice = page.getByRole('tab', { name: 'Voice', exact: true });
  const composer = page.getByRole('textbox', { name: 'Chat message', exact: true });
  const send = page.getByRole('button', { name: 'Send chat message', exact: true });
  const transcript = page.getByRole('log', { name: 'Chat conversation', exact: true });
  await expect(chat).toHaveAttribute('aria-selected', 'true');
  await expect(page.getByRole('button', { name: 'Start microphone', exact: true })).toHaveCount(0);
  await expect(send).toBeDisabled();
  await composer.fill('   ');
  await expect(send).toBeDisabled();
  await page.getByRole('spinbutton', { name: 'Feedback interval (s)', exact: true }).fill('.25');
  await composer.fill('What do you see?');
  await send.click();
  await expect(transcript).toContainText('What do you see?');
  await expect(transcript).toContainText('I can see the room through my head camera.');
  await expect(composer).toHaveValue('');
  const first = await (await request.get('/api/state')).json();
  expect(first.agent.mode).toBe('chat');
  expect(first.snapshot.simulated_time_s).toBe(0);
  await composer.fill('Move forward a little');
  await expect(send).toBeEnabled();
  const followupRequest = page.waitForRequest(value => value.url().endsWith('/api/agent/chat'));
  await page.getByRole('spinbutton', { name: 'Images per request', exact: true }).fill('3');
  await page.getByRole('spinbutton', { name: 'Retained context (tokens)', exact: true }).fill('12000');
  await send.click();
  expect((await followupRequest).postDataJSON()).toMatchObject({ conversation_id: first.agent.session_id,
    images_per_request: 3, context_tokens: 12000 });
  await expect(transcript).toContainText('Movement complete.');
  await expect(transcript.locator('article')).toHaveCount(5);
  await expect.poll(() => transcript.evaluate(element => element.scrollHeight - element.scrollTop - element.clientHeight)).toBeLessThan(2);
  expect((await (await request.get('/api/state')).json()).observation.odometry_m_rad[0]).toBeGreaterThan(.02);
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: 'test-results/chat-desktop.png', fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.locator('.chat-section').screenshot({ path: 'test-results/chat-mobile.png' });
  await composer.fill('Hold your reply');
  await send.click();
  await expect(page.locator('.chat-progress')).toHaveText('Awaiting model');
  await expect(send).toBeDisabled();
  await voice.click();
  await expect(voice).toHaveAttribute('aria-selected', 'true');
  await expect(composer).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Start microphone', exact: true })).toBeVisible();
  await expect(page.getByRole('combobox', { name: 'LLM model', exact: true })).toHaveCount(0);
  const switched = await (await request.get('/api/state')).json();
  expect(switched.agent.active).toBe(false);
  expect(switched.stopped).toBe(true);
  await expect(page.locator('.status')).toHaveText('Robot stopped');
  await expect(page.locator('.model-connection-status')).toHaveText('Realtime resource endpoint missing');
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: 'test-results/voice-mode-mobile.png', fullPage: true });
  await voice.focus();
  await page.keyboard.press('ArrowLeft');
  await expect(chat).toHaveAttribute('aria-selected', 'true');
  await expect(chat).toBeFocused();
  await expect(page.getByRole('button', { name: 'Start microphone', exact: true })).toHaveCount(0);
  await expect(transcript).toContainText('What do you see?');
  await request.post('/api/reset');
  await expect(transcript.locator('article')).toHaveCount(0);
});

test('built-in Luna supports no reasoning and Nano can run with scripted inference', async ({ page, request }) => {
  await request.post('/api/agent/config', { data: { endpoint: 'https://test.openai.azure.com' } });
  await page.goto('/');
  const selector = page.getByRole('combobox', { name: 'LLM model', exact: true });
  await expect(selector).toHaveValue('luna');
  await expect(selector.locator('option')).toHaveText(['GPT-5.6 Luna', 'GPT-5.4 Nano', 'Gemma 4 E2B (Ollama)']);
  await expect(page.locator('.model-connection-status')).toHaveText('Configured / gpt-5.6-luna');
  const reasoning = page.getByRole('combobox', { name: 'Reasoning effort', exact: true });
  await expect(reasoning).toHaveValue('low');
  await expect(reasoning.locator('option')).toHaveText(['Low', 'None', 'Medium', 'High']);
  await reasoning.selectOption('none');
  await expect(page.getByRole('button', { name: 'Start LLM control', exact: true })).toBeEnabled();
  await selector.selectOption('nano');
  await expect(reasoning).toHaveValue('low');
  await expect(reasoning.locator('option[value="none"]')).toHaveCount(0);
  await expect(page.locator('.model-connection-status')).toHaveText('Configured / gpt-5.4-nano');
  await page.getByRole('spinbutton', { name: 'LLM turn limit', exact: true }).fill('1');
  const started = page.waitForRequest(value => value.url().endsWith('/api/agent/start'));
  await page.getByRole('button', { name: 'Start LLM control', exact: true }).click();
  expect((await started).postDataJSON()).toMatchObject({ model_id: 'nano', reasoning: 'low' });
  const feed = page.getByRole('log', { name: 'Robot and LLM exchanges' });
  await expect(feed.locator('[data-kind="session"]').first()).toContainText('GPT-5.4 Nano');
  const trace = await (await request.get('/api/agent/trace')).json();
  expect(trace.events[0].payload.deployment).toBe('gpt-5.4-nano');
  await expect(feed.locator('[data-kind="result"]')).toContainText('drive_base');
  await page.getByRole('button', { name: 'Reset episode', exact: true }).click();
  await expect(selector).toHaveValue('luna');
  await expect(page.locator('.model-connection-status')).toHaveText('Configured / gpt-5.6-luna');
  await expect(reasoning).toHaveValue('low');
  await reasoning.selectOption('none');
  await page.getByRole('spinbutton', { name: 'LLM turn limit', exact: true }).fill('1');
  const lunaStarted = page.waitForRequest(value => value.url().endsWith('/api/agent/start'));
  await page.getByRole('button', { name: 'Start LLM control', exact: true }).click();
  expect((await lunaStarted).postDataJSON()).toMatchObject({ model_id: 'luna', reasoning: 'none' });
  await expect(feed.locator('[data-kind="session"]').first()).toContainText('GPT-5.6 Luna');
  await expect(feed.locator('[data-kind="result"]')).toContainText('drive_base');
  const lunaTrace = await (await request.get('/api/agent/trace')).json();
  expect(lunaTrace.events[0].payload).toMatchObject({ deployment: 'gpt-5.6-luna', reasoning: 'none' });
  expect((await (await request.get('/api/state')).json()).agent.reasoning).toBe('none');
});

test('Ollama Gemma uses local settings and scripted robot control while preserving Foundry', async ({ page, request }) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await request.post('/api/agent/config', { data: { endpoint: 'https://test.openai.azure.com' } });
  await page.setViewportSize({ width: 1440, height: 1100 });
  await page.goto('/');
  const model = page.getByRole('combobox', { name: 'LLM model', exact: true });
  const reasoning = page.getByRole('combobox', { name: 'Reasoning effort', exact: true });
  await expect(model).toHaveValue('luna');
  await model.selectOption('gemma');
  await expect(reasoning).toHaveValue('none');
  await expect(reasoning.locator('option')).toHaveText(['None']);
  await expect(page.locator('.model-connection-status')).toHaveText('Configured locally / gemma4:e2b-it-qat');
  await page.locator('.agent-connection summary').click();
  await expect(page.getByRole('combobox', { name: 'Model provider', exact: true })).toHaveValue('ollama');
  await expect(page.getByRole('textbox', { name: 'Ollama model tag', exact: true })).toHaveValue('gemma4:e2b-it-qat');
  await page.getByRole('textbox', { name: 'Ollama endpoint', exact: true }).fill('http://localhost:11434');
  await page.getByRole('button', { name: 'Apply connection', exact: true }).click();
  await expect.poll(async () => (await (await request.get('/api/state')).json()).agent.configuration.ollama_endpoint).toBe('http://localhost:11434');
  const configuration = (await (await request.get('/api/state')).json()).agent.configuration;
  expect(configuration.endpoint).toBe('https://test.openai.azure.com/openai/v1/');
  expect(configuration.models.map((entry: { id: string }) => entry.id)).toEqual(['luna', 'nano', 'gemma']);
  await page.getByRole('spinbutton', { name: 'LLM turn limit', exact: true }).fill('2');
  const started = page.waitForRequest(value => value.url().endsWith('/api/agent/start'));
  await page.getByRole('button', { name: 'Start LLM control', exact: true }).click();
  expect((await started).postDataJSON()).toMatchObject({ model_id: 'gemma', reasoning: 'none' });
  const feed = page.getByRole('log', { name: 'Robot and LLM exchanges' });
  await expect(feed.locator('[data-kind="result"]')).toContainText('drive_base');
  const trace = await (await request.get('/api/agent/trace')).json();
  expect(trace.events[0].payload).toMatchObject({ provider: 'ollama', deployment: 'gemma4:e2b-it-qat', reasoning: 'none' });
  await page.getByRole('button', { name: 'Stop', exact: true }).click();
  await expect.poll(async () => (await (await request.get('/api/state')).json()).agent.active).toBe(false);
  await model.selectOption('luna');
  await expect(reasoning).toHaveValue('low');
  await expect(page.getByRole('textbox', { name: 'Foundry endpoint', exact: true })).toHaveValue('https://test.openai.azure.com/openai/v1/');
  await model.selectOption('gemma');
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: 'test-results/ollama-controls-desktop.png', fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: 'test-results/ollama-controls-mobile.png', fullPage: true });
  expect(errors).toEqual([]);
});

test('Ollama can be selected without a Foundry endpoint', async ({ page, request }) => {
  await request.post('/api/agent/config', { data: { endpoint: '' } });
  await page.goto('/');
  const model = page.getByRole('combobox', { name: 'LLM model', exact: true });
  await expect(model).toHaveValue('gemma');
  await expect(page.getByRole('combobox', { name: 'Reasoning effort', exact: true })).toHaveValue('none');
  await expect(page.getByRole('button', { name: 'Start LLM control', exact: true })).toBeEnabled();
  await model.selectOption('luna');
  await expect(page.getByRole('button', { name: 'Start LLM control', exact: true })).toBeDisabled();
  await model.selectOption('gemma');
  await expect(page.getByRole('button', { name: 'Start LLM control', exact: true })).toBeEnabled();
});

test('loaded project selects its configured deployment and explains missing model settings', async ({ page, request }) => {
  await request.post('/api/agent/config', { data: {
    endpoint: 'https://test.services.ai.azure.com/api/projects/test', models: [
      { id: 'luna', label: 'GPT-5.6 Luna', deployment: '' },
      { id: 'configured', label: 'GPT-5.2', deployment: 'GPT-5.2' },
    ],
  } });
  await page.goto('/');
  const selector = page.getByRole('combobox', { name: 'LLM model', exact: true });
  await expect(selector).toHaveValue('configured');
  await expect(page.locator('.model-connection-status')).toHaveText('Configured / GPT-5.2');
  await expect(page.getByRole('button', { name: 'Start LLM control', exact: true })).toBeEnabled();
  await selector.selectOption('luna');
  await expect(page.locator('.model-connection-status')).toHaveText('GPT-5.6 Luna: deployment name missing');
  await expect(page.getByRole('button', { name: 'Start LLM control', exact: true })).toBeDisabled();
  await page.locator('.agent-connection summary').click();
  await expect(page.getByRole('textbox', { name: 'Foundry endpoint', exact: true })).toHaveValue('https://test.services.ai.azure.com/api/projects/test');
  await page.getByRole('tab', { name: 'Voice', exact: true }).click();
  await expect(page.locator('.voice-state')).toHaveText('Realtime resource endpoint missing');
  await request.post('/api/voice/config', { data: { endpoint: 'https://test.openai.azure.com', deployment: '' } });
  await expect(page.locator('.voice-state')).toHaveText('GPT Realtime 2 deployment name missing');
  await page.getByRole('tab', { name: 'Chat', exact: true }).click();
  await selector.selectOption('configured');
  await expect(page.getByRole('button', { name: 'Start LLM control', exact: true })).toBeEnabled();
});

test('navigation plan advances skills, moves during inference, and cancels its buffer', async ({ page, request }) => {
  let observedMotionDuringInference = false;
  const firstPhysicsTime = new Map<string, number>();
  page.on('websocket', socket => {
    if (!socket.url().endsWith('/api/live')) return;
    socket.on('framereceived', ({ payload }) => {
      const state: LiveState = JSON.parse(String(payload));
      if (state.agent.phase !== 'thinking' || state.navigation?.active_step !== 1 || !state.busy) return;
      const key = `${state.agent.session_id}:${state.agent.turns}:${state.navigation.revision}`;
      const first = firstPhysicsTime.get(key);
      if (first !== undefined && state.snapshot.simulated_time_s > first) observedMotionDuringInference = true;
      if (first === undefined) firstPhysicsTime.set(key, state.snapshot.simulated_time_s);
    });
  });
  await request.post('/api/agent/config', { data: { endpoint: 'https://test.openai.azure.com' } });
  await page.setViewportSize({ width: 1440, height: 1100 });
  await page.goto('/');
  await expect(page.getByRole('button', { name: 'Single step', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await page.getByRole('button', { name: 'Navigation plan', exact: true }).click();
  await expect(page.getByRole('spinbutton', { name: 'Feedback interval (s)', exact: true })).toHaveValue('0.25');
  const start = page.waitForRequest(value => value.url().endsWith('/api/agent/start'));
  await page.getByRole('button', { name: 'Start LLM control', exact: true }).click();
  expect((await start).postDataJSON().execution_mode).toBe('navigation_plan');
  const plan = page.getByRole('region', { name: 'Navigation plan', exact: true });
  await expect(plan.locator('li')).toHaveCount(4);
  await expect(plan.locator('li').nth(0)).toHaveAttribute('data-status', 'completed');
  await expect(plan.locator('li').nth(1)).toHaveAttribute('data-status', 'running');
  await expect(plan.locator('li').nth(2)).toHaveAttribute('data-status', 'pending');
  await expect.poll(() => observedMotionDuringInference).toBe(true);
  await expect.poll(async () => (await spectatorPixels(page)).colors).toBeGreaterThan(50);
  const feed = page.getByRole('log', { name: 'Robot and LLM exchanges' });
  await expect(feed.locator('[data-kind="tool"]').filter({ hasText: 'replace_motion_buffer' })).not.toHaveCount(0);
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: 'test-results/navigation-plan-desktop.png', fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await expect.poll(async () => (await spectatorPixels(page)).colors).toBeGreaterThan(50);
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: 'test-results/navigation-plan-mobile.png', fullPage: true });
  await page.getByRole('button', { name: 'Cancel navigation plan', exact: true }).click();
  await expect(plan.locator('li').nth(1)).toHaveAttribute('data-status', 'cancelled');
  await expect(plan.locator('li').nth(2)).toHaveAttribute('data-status', 'cancelled');
  const cancelled: LiveState = await (await request.get('/api/state')).json();
  expect(cancelled.navigation?.remaining_s).toBe(0);
  expect(cancelled.agent.active).toBe(false);
  await page.getByRole('button', { name: 'Reset episode', exact: true }).click();
  await expect(plan).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Single step', exact: true })).toHaveAttribute('aria-pressed', 'true');
});

test('copy exchange feed includes filtered and collapsed payloads and handles clipboard denial', async ({ page, request }) => {
  await page.addInitScript(() => {
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: {
      writeText: async (text: string) => {
        const probe = window as ClipboardProbe;
        if (probe.rejectCopy) throw new Error('Clipboard denied');
        probe.copiedExchange = text;
      },
    } });
  });
  await request.post('/api/agent/config', { data: { endpoint: 'https://test.openai.azure.com' } });
  await page.goto('/');
  const copy = page.getByRole('button', { name: 'Copy exchange feed', exact: true });
  await expect(copy).toBeDisabled();
  await page.getByRole('spinbutton', { name: 'LLM turn limit', exact: true }).fill('1');
  await page.getByRole('button', { name: 'Start LLM control', exact: true }).click();
  const feed = page.getByRole('log', { name: 'Robot and LLM exchanges' });
  await expect(feed.locator('[data-kind="result"]')).toContainText('drive_base');
  await expect.poll(async () => (await (await request.get('/api/state')).json()).agent.active).toBe(false);
  const trace = await (await request.get('/api/agent/trace')).json();
  await expect(feed.locator('article')).toHaveCount(trace.events.length);
  await page.getByRole('button', { name: 'LLM', exact: true }).click();
  await expect(feed.locator('[data-kind="tool"]')).toHaveCount(0);
  await copy.click();
  const copied = await page.evaluate(() => JSON.parse((window as ClipboardProbe).copiedExchange));
  expect(copied.format).toBe('milo-exchange-feed-v1');
  expect(copied.session_id).toBe(trace.session_id);
  expect(copied.events).toEqual(trace.events);
  expect(copied.image_note).toContain('not embedded');
  expect(copied.earlier_events_discarded).toBe(false);
  await expect(page.getByRole('status').filter({ hasText: `Copied ${trace.events.length} exchanges` })).toBeVisible();
  await page.evaluate(() => { (window as ClipboardProbe).rejectCopy = true; });
  await copy.click();
  await expect(page.getByRole('status').filter({ hasText: 'Copy failed' })).toBeVisible();
  await page.evaluate(() => { (window as ClipboardProbe).rejectCopy = false; });
  await copy.click();
  await expect(page.getByRole('status').filter({ hasText: 'Copied' })).toBeVisible();
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: 'test-results/copy-exchanges-mobile.png', fullPage: true });
  await page.getByRole('button', { name: 'Reset episode', exact: true }).click();
  await expect(copy).toBeDisabled();
  await expect(page.getByRole('status').filter({ hasText: 'Copied' })).toHaveCount(0);
});

test('mobile exchange feed shows actual camera input and sensor details', async ({ page, request }) => {
  await request.post('/api/agent/config', { data: {
    endpoint: 'https://test.openai.azure.com', models: [{ id: 'luna', label: 'Test model', deployment: 'test-model' }],
  } });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/');
  await page.getByRole('spinbutton', { name: 'Feedback interval (s)', exact: true }).fill('.25');
  await page.getByRole('button', { name: 'Start LLM control', exact: true }).click();
  const feed = page.getByRole('log', { name: 'Robot and LLM exchanges' });
  await expect(feed.locator('[data-kind="feedback"]')).toHaveCount(2);
  await page.getByRole('checkbox', { name: 'Follow latest', exact: true }).uncheck();
  await page.getByRole('button', { name: 'Inputs', exact: true }).click();
  const firstInput = feed.locator('article').first();
  await firstInput.getByRole('img').scrollIntoViewIfNeeded();
  await expect(firstInput.getByRole('img')).toHaveJSProperty('naturalWidth', 640);
  await firstInput.getByText('Sensor payload', { exact: true }).click();
  await expect(firstInput.locator('.exchange-payload pre').first()).toContainText('"joints"');
  const dimensions = await page.evaluate(() => ({ viewport: innerWidth, page: document.documentElement.scrollWidth }));
  expect(dimensions.page).toBeLessThanOrEqual(dimensions.viewport);
  expect(await feed.evaluate(element => element.scrollWidth <= element.clientWidth)).toBeTruthy();
  await firstInput.getByText('Sensor payload', { exact: true }).click();
  await firstInput.getByRole('img').scrollIntoViewIfNeeded();
  await page.locator('.exchange-section').screenshot({ path: 'test-results/exchange-mobile.png' });
  await page.getByRole('button', { name: 'Stop', exact: true }).click();
});

test.describe('Realtime microphone control', () => {
  test('mic is on demand, speech packets drive tools, and audio replies play', async ({ page, request }) => {
    await page.addInitScript(() => {
      const state = { microphones: [] as MediaStreamTrack[], played: 0 };
      Object.assign(window, { voiceTest: state });
      navigator.mediaDevices.getUserMedia = async () => {
        const context = new AudioContext({ sampleRate: 24000 });
        const source = context.createOscillator();
        const destination = context.createMediaStreamDestination();
        source.connect(destination);
        source.start();
        await context.resume();
        const stream = destination.stream;
        const track = stream.getAudioTracks()[0];
        const stop = track.stop.bind(track);
        track.stop = () => { stop(); source.stop(); void context.close(); };
        state.microphones.push(...stream.getAudioTracks());
        return stream;
      };
      const originalStart = AudioBufferSourceNode.prototype.start;
      AudioBufferSourceNode.prototype.start = function(...args) { state.played++; return originalStart.apply(this, args); };
    });
    await page.setViewportSize({ width: 1440, height: 1100 });
    await page.goto('/');
    await page.getByRole('tab', { name: 'Voice', exact: true }).click();
    await expect(page.getByRole('button', { name: 'Start microphone', exact: true })).toBeDisabled();
    await expect(page.getByRole('region', { name: 'Voice control', exact: true })).toContainText('GPT REALTIME 2 / PREVIEW / AI VOICE');
    expect(await page.evaluate(() => (window as unknown as { voiceTest: { microphones: unknown[] } }).voiceTest.microphones.length)).toBe(0);
    await page.locator('.voice-connection summary').click();
    await expect(page.locator('.voice-connection')).toContainText('gpt-realtime-2 / low reasoning');
    await page.getByRole('textbox', { name: 'Realtime resource endpoint', exact: true }).fill('https://test.openai.azure.com');
    await page.getByRole('textbox', { name: 'Realtime deployment', exact: true }).fill('test-realtime');
    await page.getByRole('button', { name: 'Apply voice connection', exact: true }).click();
    const voiceConfig = (await (await request.get('/api/state')).json()).realtime;
    expect(voiceConfig).toMatchObject({ target_model: 'gpt-realtime-2', reasoning_effort: 'low', deployment: 'test-realtime' });
    await page.getByRole('spinbutton', { name: 'Feedback interval (s)', exact: true }).fill('.25');
    await expect(page.getByRole('button', { name: 'Start microphone', exact: true })).toBeEnabled();
    await page.getByRole('button', { name: 'Start microphone', exact: true }).click();
    const send = page.getByRole('button', { name: 'Send spoken command', exact: true });
    await expect.poll(async () => page.evaluate(() => (window as unknown as { voiceTest: { microphones: unknown[] } }).voiceTest.microphones.length)).toBe(1);
    await expect(send).toBeEnabled();
    await expect.poll(async () => Number(await page.locator('.voice-state').getAttribute('data-recorded-seconds'))).toBeGreaterThanOrEqual(.2);
    await send.click();
    await expect(page.locator('.voice-transcripts')).toContainText('Move forward');
    await expect(page.locator('.voice-transcripts')).toContainText('Movement complete.');
    await expect(page.getByRole('button', { name: 'Start microphone', exact: true })).toBeEnabled();
    const tracker = page.getByRole('group', { name: 'Token usage', exact: true });
    await expect(tracker.locator('.token-total dd')).toHaveText('30');
    await expect(tracker.locator('.token-input dd')).toHaveText('20');
    await expect(tracker.locator('.token-output dd')).toHaveText('10');
    await expect(tracker.locator('.token-scope')).toHaveText('Current run');
    expect((await (await request.get('/api/state')).json()).observation.odometry_m_rad[0]).toBeGreaterThan(.02);
    await expect(page.locator('.voice-state')).toHaveText('Idle / microphone off', { timeout: 9000 });
    await expect(page.locator('.status')).toHaveText('Agent idle');
    await expect(tracker.locator('.token-total dd')).toHaveText('30');
    const audio = await page.evaluate(() => {
      const state = (window as unknown as { voiceTest: { microphones: MediaStreamTrack[]; played: number } }).voiceTest;
      return { played: state.played, released: state.microphones.every(track => track.readyState === 'ended') };
    });
    expect(audio.played).toBeGreaterThan(0);
    expect(audio.released).toBe(true);
    await expect(page.getByRole('log', { name: 'Robot and LLM exchanges' })).toContainText('Spoken command');
    await page.locator('.voice-section').screenshot({ path: 'test-results/voice-desktop.png' });
    await page.setViewportSize({ width: 390, height: 844 });
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await page.locator('.voice-section').screenshot({ path: 'test-results/voice-mobile.png' });
    await page.getByRole('button', { name: 'End voice', exact: true }).click();
    await expect(page.locator('.voice-state')).toHaveText('Microphone off');
    await expect.poll(async () => (await (await request.get('/api/agent')).json()).active).toBe(false);
    expect((await (await request.get('/api/state')).json()).stopped).toBe(true);
    await expect(tracker.locator('.token-total dd')).toHaveText('30');
    await expect(tracker.locator('.token-scope')).toHaveText('Last run');
    await page.getByRole('button', { name: 'Start microphone', exact: true }).click();
    await expect(page.getByRole('button', { name: 'Send spoken command', exact: true })).toBeEnabled();
    await expect(tracker.locator('.token-total dd')).toHaveText('0');
    await page.getByRole('tab', { name: 'Chat', exact: true }).click();
    await expect(page.getByRole('tab', { name: 'Chat', exact: true })).toHaveAttribute('aria-selected', 'true');
    await expect(page.locator('.voice-section')).toHaveCount(0);
    expect(await page.evaluate(() => (window as unknown as { voiceTest: { microphones: MediaStreamTrack[] } }).voiceTest.microphones.every(track => track.readyState === 'ended'))).toBe(true);
    expect((await (await request.get('/api/agent')).json()).active).toBe(false);
    await page.getByRole('tab', { name: 'Voice', exact: true }).click();
    await page.getByRole('button', { name: 'Start microphone', exact: true }).click();
    await expect(page.getByRole('button', { name: 'Send spoken command', exact: true })).toBeEnabled();
    await page.getByRole('button', { name: 'Stop', exact: true }).click();
    await expect(page.locator('.voice-state')).toHaveText('Microphone off');
    expect(await page.evaluate(() => (window as unknown as { voiceTest: { microphones: MediaStreamTrack[] } }).voiceTest.microphones.every(track => track.readyState === 'ended'))).toBe(true);
  });

  test('microphone permission denial releases voice ownership', async ({ page, request }) => {
    await page.addInitScript(() => {
      navigator.mediaDevices.getUserMedia = async () => { throw new DOMException('Denied', 'NotAllowedError'); };
    });
    await request.post('/api/voice/config', { data: { endpoint: 'https://test.openai.azure.com', deployment: 'test-realtime' } });
    await page.goto('/');
    await page.getByRole('tab', { name: 'Voice', exact: true }).click();
    await page.getByRole('button', { name: 'Start microphone', exact: true }).click();
    await expect(page.locator('.voice-section [role="alert"]')).toContainText('Microphone permission was denied');
    await expect(page.locator('.voice-state')).toHaveText('Microphone off');
    await expect.poll(async () => (await (await request.get('/api/agent')).json()).active).toBe(false);
  });
});