import { test, expect } from '@playwright/test';
import type { HomeState } from '../src/HomeMapping';

test('home mapping controls preserve map identity, show observed geometry and reject unavailable destinations', async ({ page, request }) => {
  test.setTimeout(90000);
  await request.post('/api/challenges/load', { data: { challenge_id: 'park' } });
  let current: HomeState = { environment_id: 'standalone:park', stage: 'empty', map_id: null, name: null, revision: 0,
    localization: { status: 'unlocalized', age_s: 0, pose_m_rad: null, quality: null }, places: [], maps: [], coverage: null, task: null, error: null };
  const operations: Record<string, unknown>[] = [];
  let rejectSave = true;
  await page.route('**/api/home', async route => {
    if (route.request().method() === 'GET') return route.fulfill({ json: current });
    const body = route.request().postDataJSON(); operations.push(body);
    if (body.action === 'start_mapping') {
      current = { ...current, stage: 'mapping', name: 'Unsaved home', map_id: 'sensor-map', dirty: true,
        map: { width: 80, height: 60, origin_m: [-4, -3], resolution_m: .1,
          cells: Array.from({ length: 4800 }, (_, index) => {
            const column = index % 80, row = Math.floor(index / 80);
            return column < 8 || column > 70 || row < 8 || row > 50 ? -1 : column === 8 || column === 70 || row === 8 || row === 50 ? 100 : 0;
          }) },
        coverage: { free_m2: 25, visited_cells: 8, scan_count: 4 },
        localization: { status: 'localized', age_s: 0, pose_m_rad: [0, 0, 0], quality: { mean_residual_m: .04 } },
        live_obstacles_m: [[2, 1]], route_m: [], edges: [] };
    } else if (body.action === 'review') current = { ...current, stage: 'review' };
    else if (body.action === 'add_place') current = { ...current, places: [...current.places,
      { place_id: `place-${current.places.length}`, name: body.name, kind: body.kind, pose_m_rad: body.pose_m_rad ?? [0, 0, 0], reachable: true }] };
    else if (body.action === 'save_map') {
      if (rejectSave) return route.fulfill({ status: 409, json: { detail: 'Map revision conflict' } });
      current = { ...current, name: body.name, revision: 1, stage: 'loaded', dirty: false,
        maps: [{ map_id: 'sensor-map', environment_id: 'standalone:park', name: body.name, revision: 1 }] };
    } else if (body.action === 'load_map') current = { ...current, localization: { ...current.localization, status: 'unlocalized' }, places: current.places.map(place => ({ ...place, reachable: false })) };
    else if (body.action === 'localize') current = { ...current, localization: { ...current.localization, status: 'localized' }, places: current.places.map(place => ({ ...place, reachable: true })) };
    else if (body.action === 'navigate_to') current = { ...current, stage: 'navigation', route_m: [[0, 0], [1, 1]],
      task: { status: 'running', reason: 'Following observed route', segments: 1, retries: 0, visited_frontiers: 0 } };
    else if (body.action === 'cancel_task') current = { ...current, task: current.task ? { ...current.task, status: 'cancelled', reason: 'Cancelled by operator' } : null };
    return route.fulfill({ json: current });
  });
  await page.goto('/');
  const panel = page.locator('.home-mapping');
  await panel.locator('summary').click();
  await expect(panel.getByRole('button', { name: 'Start guided mapping', exact: true })).toBeEnabled();
  const started = page.waitForResponse(response => response.url().endsWith('/api/home') && response.request().method() === 'POST');
  await panel.getByRole('button', { name: 'Start guided mapping', exact: true }).click();
  expect((await started).ok()).toBe(true);
  const canvas = panel.getByLabel('Observed home map');
  await expect(canvas).toBeVisible();
  await expect(panel.getByRole('group', { name: 'Guided mapping controls' })).toBeVisible();
  await panel.getByRole('button', { name: 'Review map', exact: true }).click();
  await panel.getByRole('textbox', { name: 'Place name', exact: true }).fill('Home');
  await panel.getByRole('button', { name: 'Add place', exact: true }).click();
  await expect(panel.locator('.home-place-list')).toContainText('Home');
  await canvas.click({ position: { x: 60, y: 60 } });
  await panel.getByRole('textbox', { name: 'Place name', exact: true }).fill('Kitchen entrance');
  await panel.getByRole('combobox', { name: 'Place type', exact: true }).selectOption('doorway');
  await panel.getByRole('listbox', { name: 'Connected places', exact: true }).selectOption('place-0');
  await panel.getByRole('button', { name: 'Add place', exact: true }).click();
  const annotation = operations.filter(operation => operation.action === 'add_place')[1];
  expect(annotation.map_id).toBe('sensor-map'); expect(annotation.connects).toEqual(['place-0']);
  expect(annotation.pose_m_rad).toHaveLength(3);
  await panel.getByRole('textbox', { name: 'Map name', exact: true }).fill('Reusable home');
  await panel.getByRole('button', { name: 'Save map', exact: true }).click();
  await expect(panel.getByRole('alert')).toContainText('Map revision conflict');
  rejectSave = false;
  await panel.getByRole('button', { name: 'Save map', exact: true }).click();
  await expect(panel.getByRole('button', { name: 'Start guided mapping', exact: true })).toBeDisabled();
  await panel.getByRole('button', { name: 'Load saved home', exact: true }).click();
  await expect(panel.getByRole('button', { name: 'Navigate', exact: true })).toBeDisabled();
  await panel.getByRole('combobox', { name: 'Approximate location', exact: true }).selectOption('place-0');
  const localized = page.waitForResponse(response => response.url().endsWith('/api/home') && response.request().postDataJSON()?.action === 'localize');
  await panel.getByRole('button', { name: 'Localize', exact: true }).click();
  expect((await localized).ok()).toBe(true);
  await panel.getByRole('combobox', { name: 'Mapped destination', exact: true }).selectOption('place-1');
  await panel.getByRole('button', { name: 'Navigate', exact: true }).click();
  await expect(panel.locator('.home-task-status')).toContainText('Following observed route');
  await expect(panel.getByRole('button', { name: 'Localize', exact: true })).toBeDisabled();
  for (const width of [1440, 390, 320]) {
    await page.setViewportSize({ width, height: 1000 });
    const geometry = await panel.evaluate(element => ({ fits: element.scrollWidth <= element.clientWidth,
      controls: Array.from(element.querySelectorAll('button,input,select')).every(control => control.getBoundingClientRect().right <= innerWidth),
      overflow: Array.from(element.querySelectorAll('*')).filter(child => child.getBoundingClientRect().right > element.getBoundingClientRect().right + 1)
        .map(child => ({ tag: child.tagName, className: child.className, text: child.textContent?.slice(0, 60), width: child.getBoundingClientRect().width })) }));
    expect(geometry.fits, JSON.stringify({ width, ...geometry })).toBe(true); expect(geometry.controls).toBe(true);
    const colors = await canvas.evaluate((element: HTMLCanvasElement) => {
      const pixels = element.getContext('2d')!.getImageData(0, 0, element.width, element.height).data;
      const distinct = new Set<string>();
      for (let index = 0; index < pixels.length; index += 64) distinct.add(`${pixels[index]},${pixels[index + 1]},${pixels[index + 2]}`);
      return distinct.size;
    });
    expect(colors).toBeGreaterThan(5);
    await panel.screenshot({ path: `.runtime/home-mapping-${width}.png` });
  }
  await panel.getByRole('button', { name: 'Cancel home task', exact: true }).click();
  await expect(panel.locator('.home-task-status')).toContainText('cancelled');
  expect(operations.find(operation => operation.action === 'navigate_to')?.place_id).toBe('place-1');
  expect(operations.filter(operation => operation.action === 'start_mapping')).toHaveLength(1);
  await page.reload();
  await panel.locator('summary').click();
  await expect(panel.getByRole('textbox', { name: 'Map name', exact: true })).toHaveValue('Reusable home');
  await panel.getByRole('textbox', { name: 'Map name', exact: true }).fill('Draft renamed home');
  await expect.poll(() => page.evaluate(async () => {
    const response = await fetch('/api/home');
    const value = await response.json();
    return value.name;
  })).toBe('Reusable home');
  await expect(panel.getByRole('textbox', { name: 'Map name', exact: true })).toHaveValue('Draft renamed home');
});