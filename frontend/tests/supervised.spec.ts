import { test, expect, type Page, type WebSocketRoute } from '@playwright/test';
import { writeFile } from 'node:fs/promises';
import type { ExchangeFeed, LiveState } from '../src/types';

async function openChallengeMenu(page: Page) {
  if (!await page.getByRole('dialog', {name:'Load scenario',exact:true}).isVisible()) {
    await page.getByRole('button', {name:'Load scenario',exact:true}).click();
  }
}

async function pixels(page: Page) {
  return page.locator('.spectator canvas').evaluate((canvas: HTMLCanvasElement) => {
    const context = canvas.getContext('webgl2')!;
    const rgba = new Uint8Array(canvas.width * canvas.height * 4);
    context.readPixels(0, 0, canvas.width, canvas.height, context.RGBA, context.UNSIGNED_BYTE, rgba);
    const colors = new Set<string>();
    for (let offset = 0; offset < rgba.length; offset += 160) colors.add(`${rgba[offset]},${rgba[offset + 1]},${rgba[offset + 2]}`);
    return colors.size;
  });
}

test.beforeEach(async ({ request, page }) => {
  await page.addInitScript(() => {
    const address = new URL(location.href);
    address.searchParams.set('diagnostics', 'legacy');
    history.replaceState(null, '', address);
  });
  expect((await request.post('/api/test/preferences/reset')).ok()).toBe(true);
  const state = await (await request.post('/api/challenges/load', { data: { challenge_id: 'bench' } })).json();
  const spatial = await request.post('/api/spatial', {data: {run_id: state.run_id, episode_epoch: state.episode_epoch, enabled: false}});
  expect(spatial.ok()).toBe(true);
  await request.post('/api/agent/config', { data: { endpoint: '', models: [
    { id: 'luna', label: 'Luna', deployment: '', reasoning_efforts: ['low', 'medium', 'high'] },
  ] } });
  await request.post('/api/local-navigation/unload');
});

test('workspace preferences restore settings and challenge drafts without motion on reload', async ({page, request}) => {
  await request.post('/api/agent/config', {data: {endpoint: 'https://test.openai.azure.com', models: [
    {id: 'luna', label: 'Scripted Luna', deployment: 'test', reasoning_efforts: ['low', 'medium', 'high']}]}});
  const initial: LiveState = await (await request.get('/api/state')).json();
  const commands: string[] = [];
  page.on('request', message => {if (message.method() === 'POST' && !message.url().endsWith('/api/preferences')) commands.push(message.url());});
  await page.goto('/');
  await page.locator('.run-options > summary').click();
  await page.getByRole('combobox', {name: 'Supervisor reasoning', exact: true}).selectOption('medium');
  await page.getByRole('spinbutton', {name: 'Supervisor turn limit', exact: true}).fill('17');
  await page.getByRole('combobox', {name: 'Navigation controller', exact: true}).selectOption('luna_navigation');
  await page.getByRole('spinbutton', {name: 'Feedback interval (s)', exact: true}).fill('1.25');
  await page.getByRole('textbox', {name: 'Robot goal', exact: true}).fill('My saved bench goal.');
  await page.getByRole('checkbox', {name: 'Axes', exact: true}).check();
  await page.locator('.manual-disclosure > summary').click();
  await page.getByRole('tab', {name: 'Head', exact: true}).click();
  await page.getByRole('slider', {name: 'Head pitch', exact: true}).fill('0.44');
  await openChallengeMenu(page);
  await page.getByRole('combobox', {name: 'Predefined challenge', exact: true}).selectOption('furniture_circuit');
  await page.getByRole('combobox', {name: 'Object to circle', exact: true}).selectOption('sofa');
  await page.getByRole('combobox', {name: 'Circuit direction', exact: true}).selectOption('counterclockwise');
  await page.getByRole('combobox', {name: 'Map source', exact: true}).selectOption('none');
  await page.locator('.challenge-details > summary').click();
  await page.getByRole('button', {name: 'Close challenge menu', exact: true}).click();
  await expect.poll(async () => (await (await request.get('/api/preferences')).json()).preferences).toMatchObject({
    turns: 17, reasoning: 'medium', navigation_mode: 'luna_navigation', interval: 1.25, run_settings_open: true,
    manual_open: true, manual_tab: 'head', head_pitch: .44, axes: true, challenge_details_open: true,
    goals: {'standalone:bench::': 'My saved bench goal.'},
    challenge_selection: {challenge_id: 'furniture_circuit', orbit_target: 'sofa', orbit_direction: 'counterclockwise', reuse_saved_map: false}});
  await page.reload();
  await expect(page.locator('.run-options')).toHaveJSProperty('open', true);
  await expect(page.getByRole('spinbutton', {name: 'Supervisor turn limit', exact: true})).toHaveValue('17');
  await expect(page.getByRole('combobox', {name: 'Supervisor reasoning', exact: true})).toHaveValue('medium');
  await expect(page.getByRole('spinbutton', {name: 'Feedback interval (s)', exact: true})).toHaveValue('1.25');
  await expect(page.getByRole('textbox', {name: 'Robot goal', exact: true})).toHaveValue('My saved bench goal.');
  await expect(page.getByRole('tab', {name: 'Head', exact: true})).toHaveAttribute('aria-selected', 'true');
  await expect(page.getByRole('slider', {name: 'Head pitch', exact: true})).toHaveValue('0.44');
  await expect(page.getByRole('checkbox', {name: 'Axes', exact: true})).toBeChecked();
  for (const width of [1440, 390]) {
    await page.setViewportSize({width, height: 1000});
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await page.locator('.run-options').screenshot({path: `../.runtime/preferences-v1-review/settings-${width}.png`});
  }
  await openChallengeMenu(page);
  await expect(page.getByRole('combobox', {name: 'Object to circle', exact: true})).toHaveValue('sofa');
  await expect(page.getByRole('combobox', {name: 'Circuit direction', exact: true})).toHaveValue('counterclockwise');
  await expect(page.locator('.challenge-details')).toHaveJSProperty('open', true);
  const after: LiveState = await (await request.get('/api/state')).json();
  expect(after.run_id).toBe(initial.run_id);
  expect(after.snapshot).toEqual(initial.snapshot);
  expect(after.agent.active).toBe(false);
  expect(commands).toEqual([]);
});

test('workspace preferences come from the server in a fresh browser context', async ({page, request, browser}) => {
  await page.goto('/');
  await page.getByRole('textbox', {name: 'Robot goal', exact: true}).fill('Remember the practice bench.');
  await request.post('/api/preferences', {data:{exploration_budget:73}});
  await expect.poll(async () => (await (await request.get('/api/preferences')).json()).preferences.exploration_budget).toBe(73);
  const context = await browser.newContext();
  try {
    const fresh = await context.newPage();
    const commands: string[] = [];
    fresh.on('request', message => {if (message.method() === 'POST' && !message.url().endsWith('/api/preferences')) commands.push(message.url());});
    await fresh.goto('http://127.0.0.1:8001/');
    await fresh.locator('.run-options > summary').click();
    await expect(fresh.getByRole('spinbutton', {name: 'Mission budget', exact: true})).toHaveValue('73');
    await expect(fresh.getByRole('textbox', {name: 'Robot goal', exact: true})).toHaveValue('Remember the practice bench.');
    expect(commands).toEqual([]);
    expect((await (await request.get('/api/state')).json()).agent.active).toBe(false);
  } finally { await context.close(); }
});

test('workspace preferences retain the selected inspector tab and never apply a saved connection', async ({page, request}) => {
  await page.routeWebSocket('**/api/live', socket => {
    const server = socket.connectToServer();
    server.onMessage(message => {
      const state: LiveState = JSON.parse(String(message));
      state.agent.session_id = 'scripted-inspector-preferences';
      state.agent.phase = 'completed';
      state.agent.active = false;
      socket.send(JSON.stringify(state));
    });
  });
  const commands: string[] = [];
  page.on('request', message => {if (message.method() === 'POST' && !message.url().endsWith('/api/preferences')) commands.push(message.url());});
  await page.goto('/');
  await page.getByRole('tab', {name: 'Settings', exact: true}).click();
  await page.getByRole('textbox', {name: 'Foundry endpoint', exact: true}).fill('https://saved.example.com');
  await page.getByRole('textbox', {name: 'Luna deployment', exact: true}).fill('saved-deployment');
  await expect.poll(async () => (await (await request.get('/api/preferences')).json()).preferences.luna_deployment).toBe('saved-deployment');
  await page.reload();
  await expect(page.getByRole('tab', {name: 'Settings', exact: true})).toHaveAttribute('aria-selected', 'true');
  await expect(page.getByRole('textbox', {name: 'Foundry endpoint', exact: true})).toHaveValue('https://saved.example.com');
  await expect(page.getByRole('button', {name: 'Start LLM control', exact: true})).toBeDisabled();
  await expect(page.getByText('Connection changes not applied', {exact: true})).toBeVisible();
  expect((await (await request.get('/api/agent')).json()).configuration.endpoint).toBe('');
  expect(commands).toEqual([]);
});

test('workspace preferences expose save failures and retry without losing edits', async ({page, request}) => {
  await request.post('/api/preferences', {data:{goals:{'standalone:bench::':'Older saved goal.'}}});
  await page.addInitScript(() => localStorage.setItem('milo-workspace-preferences-v1', '{invalid'));
  let unavailable = true;
  await page.route('**/api/preferences', route => route.request().method() === 'POST' && unavailable
    ? route.fulfill({status:503,json:{detail:'Preference store unavailable'}}) : route.continue());
  await page.goto('/');
  await page.getByRole('textbox', {name: 'Robot goal', exact: true}).fill('Retain this edit.');
  await expect(page.getByRole('alert').filter({hasText:'Preferences are not saved'})).toBeVisible();
  await expect(page.getByRole('textbox', {name: 'Robot goal', exact: true})).toHaveValue('Retain this edit.');
  await page.reload();
  await expect(page.getByRole('textbox', {name: 'Robot goal', exact: true})).toHaveValue('Retain this edit.');
  await expect(page.getByRole('alert').filter({hasText:'Preferences are not saved'})).toBeVisible();
  unavailable = false;
  await page.getByRole('button', {name: 'Retry saving preferences', exact: true}).click();
  await expect(page.getByRole('alert').filter({hasText:'Preferences are not saved'})).toHaveCount(0);
  await expect.poll(async () => (await (await request.get('/api/preferences')).json()).preferences.goals['standalone:bench::']).toBe('Retain this edit.');
});

test('challenge menu closes after loading and preserves selection when reopened', async ({page, request}) => {
  let rejectLoad = false;
  await page.route('**/api/challenges/load', route => rejectLoad
    ? route.fulfill({status:503,json:{detail:'Scene temporarily unavailable'}}) : route.continue());
  await page.goto('/');
  const launcher=page.getByRole('button',{name:'Load scenario',exact:true});
  const menu=page.getByRole('dialog',{name:'Load scenario',exact:true});
  await expect(launcher).toBeVisible();
  await expect(menu).toBeHidden();
  await expect(page.getByLabel('Selected scenario preview',{exact:true})).toBeHidden();
  const original:LiveState=await(await request.get('/api/state')).json();
  await launcher.click();
  await expect(menu).toBeVisible();
  await expect(launcher).toHaveAttribute('aria-expanded','true');
  await menu.getByRole('combobox',{name:'Predefined challenge',exact:true}).selectOption('park');
  await menu.getByRole('button',{name:'Load selected scenario',exact:true}).click();
  await expect(menu).toBeHidden();
  await expect(launcher).toBeFocused();
  await expect(page.locator('.runbar h2')).toHaveText('Park in the Bay');
  await expect(page.getByLabel('Selected scenario preview',{exact:true})).toBeHidden();
  const loaded:LiveState=await(await request.get('/api/state')).json();
  expect(loaded.run_id).not.toBe(original.run_id);
  for(const width of [1440,768,390,320]){
    await page.setViewportSize({width,height:1000});
    expect((await launcher.boundingBox())!.height).toBeLessThan(60);
    await launcher.click();
    await expect(menu.getByRole('combobox',{name:'Predefined challenge',exact:true})).toHaveValue('park');
    await expect(menu.getByRole('button',{name:'Stop',exact:true})).toBeInViewport();
    const layout=await menu.evaluate(element=>({width:element.getBoundingClientRect().width,
      fits:element.scrollWidth<=element.clientWidth,viewport:innerWidth}));
    expect(layout.width).toBeLessThan(layout.viewport);
    expect(layout.fits).toBe(true);
    await menu.getByRole('button',{name:'Close challenge menu',exact:true}).click();
    await expect(launcher).toBeFocused();
  }
  await launcher.click();
  await menu.getByRole('combobox',{name:'Predefined challenge',exact:true}).selectOption('furniture_circuit');
  await menu.getByRole('combobox',{name:'Object to circle',exact:true}).selectOption('sofa');
  await menu.getByRole('combobox',{name:'Circuit direction',exact:true}).selectOption('counterclockwise');
  rejectLoad=true;
  await menu.getByRole('button',{name:'Load selected scenario',exact:true}).click();
  await expect(menu.getByRole('alert')).toContainText('Scene temporarily unavailable');
  await expect(menu).toBeVisible();
  expect((await(await request.get('/api/state')).json()).run_id).toBe(loaded.run_id);
  await page.keyboard.press('Escape');
  await expect(menu).toBeHidden();
  await expect(launcher).toBeFocused();
  await launcher.click();
  await expect(menu.getByRole('combobox',{name:'Object to circle',exact:true})).toHaveValue('sofa');
  await expect(menu.getByRole('combobox',{name:'Circuit direction',exact:true})).toHaveValue('counterclockwise');
  rejectLoad=false;
  await menu.getByRole('button',{name:'Load selected scenario',exact:true}).click();
  await expect(menu).toBeHidden();
  await expect(page.locator('.runbar h2')).toHaveText('Circle the Furniture');
});

test('test view navigation keeps styled accessible tabs across desktop and mobile', async ({page, context}) => {
  await page.route('**/api/test-results', route => route.fulfill({json:{batches:[],skipped:0,truncated:false,scope:'Saved evaluations'}}));
  await page.goto('/');
  const initial = await page.evaluate(async () => (await (await fetch('/api/state')).json()).run_id);
  const pages = context.pages().length;
  for (const width of [1440,768,390,320]) {
    await page.setViewportSize({width,height:900});
    for (const active of ['Test cockpit','Test archive']) {
      const navigation = page.getByRole('navigation',{name:'Test views'});
      const current = navigation.getByRole('link',{name:active,exact:true});
      if (active === 'Test archive') await current.click();
      await expect(current).toHaveAttribute('aria-current','page');
      await expect(navigation).toHaveCSS('display','flex');
      const headerLayout = await navigation.evaluate(element => {
        const header = element.closest('header')!;
        const bounds = header.getBoundingClientRect();
        const brand = header.querySelector('.brand')!.getBoundingClientRect();
        const menu = element.getBoundingClientRect();
        return {rightGap:bounds.right - menu.right - parseFloat(getComputedStyle(header).paddingRight),
          overlap:brand.left < menu.right && menu.left < brand.right && brand.top < menu.bottom && menu.top < brand.bottom,
          centerDifference:Math.abs(brand.top + brand.height / 2 - menu.top - menu.height / 2),
          height:bounds.height,overflow:document.documentElement.scrollWidth > innerWidth};
      });
      expect(Math.abs(headerLayout.rightGap)).toBeLessThan(2);
      expect(headerLayout.overlap).toBe(false);
      expect(headerLayout.overflow).toBe(false);
      if (width >= 768) {
        expect(headerLayout.centerDifference).toBeLessThan(2);
        expect(headerLayout.height).toBeLessThan(80);
      }
      const links = navigation.getByRole('link');
      for (const link of await links.all()) {
        await expect(link).toHaveCSS('display','flex');
        await expect(link).toHaveCSS('text-decoration-line','none');
      }
      const layout = await links.evaluateAll(elements => elements.map(element => {
        const bounds = element.getBoundingClientRect();
        const icon = element.querySelector('svg')!.getBoundingClientRect();
        return {left:bounds.left,right:bounds.right,top:bounds.top,height:bounds.height,
          contentFits:element.scrollWidth <= element.clientWidth,iconHeight:icon.height,
          iconCentered:Math.abs(icon.top + icon.height / 2 - (bounds.top + bounds.height / 2)) <= 2};
      }));
      expect(layout.every(link => link.height >= 44 && link.left >= 0 && link.right <= width && link.contentFits && link.iconHeight === 18 && link.iconCentered)).toBe(true);
      expect(layout[0].top).toBe(layout[1].top);
      expect(layout[0].right).toBeLessThan(layout[1].left);
      await links.first().focus();
      await page.keyboard.press('Tab');
      await expect(links.nth(1)).toBeFocused();
      await expect(links.nth(1)).toHaveCSS('outline-style','solid');
      await expect(current).toHaveCSS('border-bottom-color',await current.evaluate(element=>getComputedStyle(element).color));
      const beforeHover = await current.boundingBox();
      await current.hover();
      expect(await current.boundingBox()).toEqual(beforeHover);
      if (width === 1440 || width === 320) {
        await navigation.screenshot({path:test.info().outputPath(`navigation-${active.replace(' ','-')}-${width}.png`)});
        await navigation.locator('..').screenshot({path:`../.runtime/unified-header-v1/${active.replace(' ','-')}-${width}.png`});
      }
    }
    await page.getByRole('navigation',{name:'Test views'}).getByRole('link',{name:'Test cockpit',exact:true}).click();
    await expect(page).toHaveURL(url => url.pathname === '/' && !url.searchParams.has('view') && url.searchParams.get('diagnostics') === 'legacy');
  }
  expect(context.pages()).toHaveLength(pages);
  expect(await page.evaluate(async () => (await (await fetch('/api/state')).json()).run_id)).toBe(initial);
});

test('past test results open in the same tab with filters and return navigation', async ({page, context, request}) => {
  const requests: string[] = [];
  let fail = false;
  let routeFailed = false;
  const before: LiveState = await (await request.get('/api/state')).json();
  const trial = {case_id:'table-clockwise',challenge:'furniture_circuit',title:'Circle the Furniture',evidence:'real_model',status:'Verified pass',verified_success:true,
    physics_success:true,recording_complete:true,assisted:false,completion_s:132.5,elapsed_s:133,distance_m:11.7,contact_episodes:0,input_tokens:7190,output_tokens:370,
    inference_median_s:21.1,turns:1,termination:'completed',rendering:'enhanced',false_completion_claim:false,image_url:before.camera.url,
    trajectory_url:'/api/test-results/saved-0/trajectories/0'};
  const points=Array.from({length:41},(_,index)=>({x:-2+Math.cos(index*Math.PI/20),y:1+Math.sin(index*Math.PI/20),
    wall_s:index===40?20.003:index/2,segment:0,status:index===40?'completed':'in_progress',activity:'driving'}));
  const scene={source:'recorded_initial',simulated_s:0,geometry:[
    {key:'0:-1',type:3,dimensions:[10,8,.1],position:[0,0,0],quaternion:[0,0,0,1],color:[.7,.7,.7,1],name:'floor',texture:before.geometry.find(asset=>asset.texture)?.texture ?? null},
    {key:'1:-1',type:3,dimensions:[1.2,1,.8],position:[0,0,0],quaternion:[0,0,0,1],color:[.5,.2,.1,1],name:'test_table',texture:null}],
    poses:[{key:'0:-1',position:[-2,1,-.05],quaternion:[0,0,0,1]},{key:'1:-1',position:[-2,1,.4],quaternion:[0,0,0,1]}]};
  await context.route('**/api/test-results/*/trajectories/*',route=>routeFailed?route.fulfill({status:404,json:{detail:'Missing route'}}):
    route.fulfill({json:{points,bounds_m:[-3,0,-1,2],contacts:[points[10]],sample_count:41,downsampled:false,contact_markers_truncated:false,scene}}));
  const batches = Array.from({length:7},(_,index)=>({id:`saved-${index}`,name:`history-check-${index}`,date:`2026-09-13T1${index}:00:00Z`,date_source:'recorded',
    design:index===0?'motion-history-v1':'legacy-controller',source_sha256:'a'.repeat(64),mode:'luna_continuous',evidence:index===0?'real_model':'scripted_test',
    model:'test-deployment',reasoning:'high',budget_s:180,history:index===0?'enabled':'Not recorded',legacy:index!==0,source_changed:false,planned:2,successes:1,
    trials:[trial,{...trial,case_id:'missing-report',challenge:'park',title:'Park in the Bay',status:'Missing report',verified_success:false,physics_success:null,
      recording_complete:false,completion_s:null,input_tokens:null,output_tokens:null,contact_episodes:null,image_url:null,trajectory_url:null}]}));
  await context.route('**/api/test-results', route => fail?route.fulfill({status:503,json:{detail:'Unavailable'}}):route.fulfill({json:{batches,skipped:0,truncated:false,scope:'Saved evaluations'}}));
  const connected = page.waitForEvent('websocket');
  await page.goto('/');
  await connected;
  await expect(page.getByRole('link',{name:'Test cockpit',exact:true})).toHaveAttribute('aria-current','page');
  const archiveLink = page.getByRole('link',{name:'Test archive',exact:true});
  await expect(archiveLink).not.toHaveAttribute('target','_blank');
  const pageCount = context.pages().length;
  let openedPages = 0;
  const onPage = () => { openedPages++; };
  context.on('page',onPage);
  await archiveLink.click();
  const results = page;
  let sockets = 0;
  const onSocket = () => { sockets++; };
  results.on('websocket',onSocket);
  results.on('request',request=>{if(request.method()!=='GET')requests.push(request.url());});
  try {
    await expect(results).toHaveURL(/view=test-results/);
    await expect(results).toHaveTitle('Milo | Test results');
    expect(context.pages()).toHaveLength(pageCount);
    expect(openedPages).toBe(0);
    await expect(results.getByRole('link',{name:'Test archive',exact:true})).toHaveAttribute('aria-current','page');
    await expect(results.getByRole('combobox',{name:'Filter test evidence',exact:true})).toHaveValue('real_model');
    await results.getByRole('combobox',{name:'Filter test evidence',exact:true}).selectOption('all');
    await expect(results.getByLabel('Saved test runs').locator('.results-run')).toHaveCount(3);
    await results.getByRole('button',{name:'Show more',exact:true}).click();
    await expect(results.getByLabel('Saved test runs').locator('.results-run')).toHaveCount(7);
    await expect(results.getByLabel('Test run details')).toContainText('motion-history-v1');
    await expect(results.getByRole('img',{name:'Saved final camera: Circle the Furniture',exact:true})).toHaveJSProperty('naturalWidth',640);
    const environment=results.getByLabel('Recorded environment and robot route',{exact:true});
    await expect(environment).toBeVisible();
    await expect(results.getByText('Recorded initial environment',{exact:true})).toBeVisible();
    await expect(environment).toHaveAttribute('data-objects','2');
    await expect(environment).toHaveAttribute('data-textures-pending','0');
    await expect(environment).not.toHaveAttribute('data-texture-error','true');
    const scenePixels=()=>environment.evaluate((canvas:HTMLCanvasElement)=>{
      const context=canvas.getContext('webgl2')!;
      const data=new Uint8Array(canvas.width*canvas.height*4);
      context.readPixels(0,0,canvas.width,canvas.height,context.RGBA,context.UNSIGNED_BYTE,data);
      const colors=new Set<string>();
      for(let offset=0;offset<data.length;offset+=64)colors.add(`${data[offset]},${data[offset+1]},${data[offset+2]}`);
      return colors.size;
    });
    await expect.poll(scenePixels).toBeGreaterThan(20);
    const topImage=await environment.evaluate((canvas:HTMLCanvasElement)=>canvas.toDataURL());
    await environment.hover({position:{x:(await environment.boundingBox())!.width/2,y:(await environment.boundingBox())!.height/2}});
    await expect(results.locator('.results-scene-object')).toHaveText('test table');
    await results.getByRole('button',{name:'3D environment',exact:true}).click();
    await expect.poll(()=>environment.evaluate((canvas:HTMLCanvasElement)=>canvas.toDataURL())).not.toBe(topImage);
    const angledImage=await environment.evaluate((canvas:HTMLCanvasElement)=>canvas.toDataURL());
    const environmentBox=(await environment.boundingBox())!;
    await results.mouse.move(environmentBox.x+environmentBox.width/2,environmentBox.y+environmentBox.height/2);
    await results.mouse.down(); await results.mouse.move(environmentBox.x+environmentBox.width/2+50,environmentBox.y+environmentBox.height/2+20,{steps:5}); await results.mouse.up();
    await expect.poll(()=>environment.evaluate((canvas:HTMLCanvasElement)=>canvas.toDataURL())).not.toBe(angledImage);
    await results.getByRole('button',{name:'Rewind route',exact:true}).click();
    await expect(environment).toHaveAttribute('data-time','0');
    await results.getByRole('button',{name:'Play route',exact:true}).click();
    await expect.poll(async()=>Number(await environment.getAttribute('data-time'))).toBeGreaterThan(0);
    await results.getByRole('button',{name:'Pause route',exact:true}).click();
    await results.getByRole('slider',{name:'Route time',exact:true}).press('End');
    await expect(environment).toHaveAttribute('data-time','20.003');
    await results.getByRole('button',{name:'Path only',exact:true}).click();
    const map=results.getByRole('img',{name:'Recorded robot path: Circle the Furniture',exact:true});
    await expect(map).toBeVisible();
    await expect(map.locator('.results-route-contact')).toHaveCount(1);
    await expect(map.locator('.results-route-cursor')).toHaveAttribute('data-time','20.003');
    await results.getByRole('button',{name:'Rewind route',exact:true}).click();
    await expect(map.locator('.results-route-cursor')).toHaveAttribute('data-time','0');
    await results.getByRole('combobox',{name:'Route playback speed',exact:true}).selectOption('10');
    await results.getByRole('button',{name:'Play route',exact:true}).click();
    await expect.poll(async()=>Number(await map.locator('.results-route-cursor').getAttribute('data-time'))).toBeGreaterThan(0);
    await results.getByRole('button',{name:'Pause route',exact:true}).click();
    const cursorTime=await map.locator('.results-route-cursor').getAttribute('data-time');
    await expect(map.locator('.results-route-cursor')).toHaveAttribute('data-time',cursorTime!);
    const slider=results.getByRole('slider',{name:'Route time',exact:true});
    await slider.focus(); await slider.press('End');
    await expect(map.locator('.results-route-cursor')).toHaveAttribute('data-time','20.003');
    const geometry=await map.locator('.results-route-full').evaluate((element:SVGPolylineElement)=>{
      const box=element.getBBox(); return {width:box.width,height:box.height,points:element.points.numberOfItems};
    });
    expect(geometry.points).toBe(41);
    expect(geometry.width).toBeCloseTo(geometry.height,2);
    await results.getByRole('button',{name:'Park in the Bay missing-report',exact:true}).click();
    await expect(results.getByLabel('Selected trial details')).toContainText('Missing report');
    await expect(results.getByLabel('Selected trial details')).toContainText('Final camera not available');
    await expect(results.getByLabel('Selected trial details')).toContainText('Movement map not recorded');
    await results.goto('/?view=test-results&scenario=furniture_circuit');
    await expect(results.getByRole('combobox',{name:'Filter test scenario',exact:true})).toHaveValue('furniture_circuit');
    await expect(results.locator('.results-table tbody tr')).toHaveCount(1);
    await expect(results.locator('.results-trial-heading')).toContainText('1 / 1 verified passes');
    await results.getByRole('button',{name:'Path only',exact:true}).click();
    await expect(map).toBeVisible();
    await results.getByRole('searchbox',{name:'Search saved tests',exact:true}).fill('nonexistent');
    await expect(results.getByRole('heading',{name:'No matching tests',exact:true})).toBeVisible();
    await results.getByRole('button',{name:'Clear filters',exact:true}).click();
    await results.getByRole('combobox',{name:'Filter test evidence',exact:true}).selectOption('scripted_test');
    await expect(results.locator('.results-list-title')).toContainText('6');
    await results.getByRole('combobox',{name:'Filter test evidence',exact:true}).selectOption('real_model');
    await expect(results.locator('.results-list-title')).toContainText('1');
    const downloaded = results.waitForEvent('download');
    await results.getByRole('button',{name:'Download selected test results',exact:true}).click();
    expect((await downloaded).suggestedFilename()).toBe('milo-results-saved-0.json');
    for(const width of [1440,768,390,320]) {
      await results.setViewportSize({width,height:1000});
      expect(await results.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
      await expect(results.getByRole('link',{name:'Test cockpit',exact:true})).toBeVisible();
      await expect(results.getByRole('link',{name:'Test archive',exact:true})).toBeVisible();
      await expect(results.locator('.results-header h1 span')).toBeVisible();
      await expect(results.getByRole('button',{name:'Refresh saved results',exact:true})).toBeVisible();
      await results.getByRole('button',{name:'Top-down environment',exact:true}).click();
      await expect(environment).toBeVisible();
      await expect.poll(scenePixels).toBeGreaterThan(20);
      const bounds=await results.locator('.results-visuals').evaluate(element=>{
        const children=Array.from(element.children).map(child=>child.getBoundingClientRect());
        return {overlap:children[0].right>children[1].left && children[1].right>children[0].left && children[0].bottom>children[1].top,
          mapWidth:children[0].width};
      });
      expect(bounds.overlap).toBe(false); expect(bounds.mapWidth).toBeGreaterThan(250);
      await results.screenshot({path:`test-results/saved-results-${width}.png`,fullPage:true});
      await results.getByRole('button',{name:'3D environment',exact:true}).click();
      await expect.poll(scenePixels).toBeGreaterThan(20);
      await results.getByRole('button',{name:'Frame room',exact:true}).click();
      await expect(results.getByRole('button',{name:'Frame route',exact:true})).toBeVisible();
      await results.getByRole('button',{name:'Frame route',exact:true}).click();
      await results.getByRole('button',{name:'Path only',exact:true}).click();
    }
    routeFailed=true;
    await results.getByRole('button',{name:'Refresh saved results',exact:true}).click();
    await expect(results.getByRole('status')).toContainText('Recorded route unavailable');
    routeFailed=false;
    await results.getByRole('button',{name:'Retry movement map',exact:true}).click();
    await expect(environment).toBeVisible();
    fail=true;
    await results.getByRole('button',{name:'Refresh saved results',exact:true}).click();
    await expect(results.getByRole('alert')).toContainText('503');
    fail=false;
    await results.getByRole('button',{name:'Refresh saved results',exact:true}).click();
    await expect(results.getByRole('alert')).toHaveCount(0);
    batches.unshift({...batches[0],id:'newly-recorded',name:'Latest recorded apartment tests'});
    await results.evaluate(()=>window.dispatchEvent(new Event('focus')));
    await expect(results.locator('.results-list .results-run strong').first()).toHaveText('Latest recorded apartment tests');
    expect(sockets).toBe(0);
    expect(requests).toEqual([]);
    await expect(results.locator('.spectator, .robot-activity')).toHaveCount(0);
    await expect.poll(async()=>(await(await request.get('/api/state')).json()).stopped).toBe(true);
    results.off('websocket',onSocket);
    const pagesBeforeReturn = context.pages();
    const openedBeforeReturn = openedPages;
    await results.getByRole('link',{name:'Test cockpit',exact:true}).click();
    await expect(results).toHaveURL(url=>!url.searchParams.has('view'));
    await expect(results.getByRole('link',{name:'Test cockpit',exact:true})).toHaveAttribute('aria-current','page');
    expect(context.pages()).toEqual(pagesBeforeReturn);
    expect(openedPages).toBe(openedBeforeReturn);
  } finally { context.off('page',onPage); results.off('websocket',onSocket); }
  const after: LiveState = await (await request.get('/api/state')).json();
  expect(after.run_id).toBe(before.run_id);
  expect(after.snapshot).toEqual(before.snapshot);
  expect(after.stopped).toBe(true);
  expect(after.agent.active).toBe(false);
});


test('browser robot trials keep running in the archive and Stop saves their replay', async ({page, request}) => {
  test.setTimeout(60000);
  const before = await (await request.get('/api/test-results')).json();
  const existing = new Set(before.batches.map((batch:{id:string})=>batch.id));
  const state: LiveState = await (await request.post('/api/challenges/load',{data:{challenge_id:'park'}})).json();
  await request.post('/api/agent/config',{data:{endpoint:'https://test.openai.azure.com',models:[
    {id:'luna',label:'Scripted browser test',deployment:'test-deployment'}]}});
  await page.goto('/');
  await expect(page.getByRole('button',{name:'Start LLM control',exact:true})).toBeVisible();
  const response=await request.post('/api/agent/start',{data:{run_id:state.run_id,episode_epoch:state.episode_epoch,
    goal:'Record this scripted browser trial.',max_turns:2,feedback_interval_s:.25}});
  expect(response.ok()).toBe(true);
  await expect.poll(async()=>(await(await request.get('/api/state')).json()).snapshot.simulated_time_s).toBeGreaterThan(.5);
  const moving:LiveState=await(await request.get('/api/state')).json();
  let closedSockets=0;
  page.on('websocket',socket=>socket.on('close',()=>closedSockets++));
  await page.evaluate(()=>{(window as typeof window & {workspaceMarker?:boolean}).workspaceMarker=true;});
  await page.getByRole('link',{name:'Test archive',exact:true}).click();
  await page.getByRole('button',{name:'Show current test',exact:true}).click();
  await expect(page.getByLabel('Selected trial details')).toContainText('Running');
  const continued:LiveState=await(await request.get('/api/state')).json();
  expect(continued.agent.session_id).toBe(moving.agent.session_id);
  expect(continued.agent.active).toBe(true);
  expect(continued.stopped).toBe(false);
  await page.goBack();
  await expect(page.getByRole('textbox',{name:'Robot goal',exact:true})).toHaveValue('Record this scripted browser trial.');
  await page.goForward();
  await expect(page.getByRole('button',{name:'Show current test',exact:true})).toBeVisible();
  expect(await page.evaluate(()=>(window as typeof window & {workspaceMarker?:boolean}).workspaceMarker)).toBe(true);
  expect(closedSockets).toBe(0);
  await page.getByRole('button',{name:'Stop',exact:true}).click();
  await expect.poll(async()=>{
    const catalog=await(await request.get('/api/test-results')).json();
    return catalog.batches.filter((batch:{id:string;trials:{recording_complete:boolean}[]})=>!existing.has(batch.id)&&batch.trials[0]?.recording_complete).length;
  }).toBe(1);
  const catalog=await(await request.get('/api/test-results')).json();
  const batch=catalog.batches.find((item:{id:string})=>!existing.has(item.id));
  expect(batch.evidence).toBe('scripted_test');
  expect(batch.trials[0].termination).toBe('interrupted');
  expect(batch.architecture.version_key).toMatch(/^tool-step@0\.1\.0\+[a-f0-9]{12}$/);
  expect(batch.model_variant.deployment).toBe('test-deployment');
  const results=page;
  {
    await results.getByRole('searchbox',{name:'Search saved tests',exact:true}).fill(batch.name);
    await expect(results.getByRole('heading',{name:batch.name,exact:true})).toBeVisible();
    await expect(results.getByText('Recorded initial environment',{exact:true})).toBeVisible();
    await expect(results.getByLabel('Recorded environment and robot route',{exact:true})).toBeVisible();
    await expect(results.getByRole('img',{name:'Saved final camera: Park in the Bay',exact:true})).toHaveJSProperty('naturalWidth',640);
    await results.getByRole('slider',{name:'Route time',exact:true}).press('End');
    await results.screenshot({path:'test-results/automatic-browser-recording.png',fullPage:true});
  }
});

test('archive automatically finalizes a bounded robot trial without operator Stop', async ({page, request}) => {
  await request.post('/api/agent/config',{data:{endpoint:'https://test.openai.azure.com',models:[
    {id:'luna',label:'Scripted browser test',deployment:'test-deployment'}]}});
  await page.goto('/');
  await expect(page.getByRole('button',{name:'Start LLM control',exact:true})).toBeEnabled();
  const state:LiveState=await(await request.get('/api/state')).json();
  const started=await request.post('/api/agent/start',{data:{run_id:state.run_id,episode_epoch:state.episode_epoch,
    goal:'Archive bounded lifecycle fixture.',max_turns:4,feedback_interval_s:.25}});
  expect(started.ok()).toBe(true);
  await page.getByRole('link',{name:'Test archive',exact:true}).click();
  await page.getByRole('button',{name:'Show current test',exact:true}).click();
  await expect(page.getByLabel('Selected trial details')).toContainText('Running');
  await expect(page.locator('.results-live-run')).toContainText('Latest cockpit test',{timeout:30000});
  await expect(page.getByLabel('Selected trial details')).toContainText('Complete recording');
  const final:LiveState=await(await request.get('/api/state')).json();
  expect(final.run_id).toBe(state.run_id);
  expect(final.snapshot.simulated_time_s).toBeGreaterThan(7);
  expect(final.agent.active).toBe(false);
  expect(final.agent.outcome?.kind).not.toBe('interrupted');
  await expect.poll(async()=>Number(await page.getByLabel('Recorded environment and robot route',{exact:true}).getAttribute('data-time'))).toBeGreaterThan(7);
});

test('mocked model progress separates architecture task and assistance variants', async ({page}) => {
  const trial={case_id:'park',challenge:'park',title:'Park in the Bay',evidence:'real_model',status:'Verified pass',verified_success:true,
    physics_success:true,recording_complete:true,assisted:false,completion_s:10,elapsed_s:12,distance_m:1,contact_episodes:0,
    input_tokens:100,output_tokens:20,inference_median_s:1,turns:1,termination:'completed',rendering:'enhanced',false_completion_claim:false,
    image_url:null,trajectory_url:null,challenge_sha256:'scene-a',task_sha256:'task-a',environment:'standalone',goal:'Park'};
  const architecture={id:'observed-continuous',name:'Observed continuous',version:'0.1.0',revision:'abc',version_key:'observed-continuous@0.1.0+abc'};
  const model={provider:'foundry',deployment:'test-deployment',revision:'model-a',configuration:{reasoning:'high'}};
  const batch={id:'first',name:'Versioned trial',date:'2026-09-14T12:00:00Z',date_source:'recorded',design:architecture.version_key,
    source_sha256:'a'.repeat(64),mode:'luna_continuous',evidence:'real_model',model:'test-deployment',reasoning:'high',budget_s:180,
    history:'enabled',legacy:false,source_changed:false,planned:2,successes:1,variant_id:'architecture-a/model-a',architecture,model_variant:model,
    trials:[trial,{...trial,case_id:'failure',status:'Not passed',verified_success:false,physics_success:false,completion_s:null}]};
  const batches=[batch,{...batch,id:'assisted',name:'Assisted trial',planned:1,successes:0,trials:[{...trial,assisted:true,verified_success:false,completion_s:null}]},
    {...batch,id:'custom',name:'Custom task',planned:1,successes:0,trials:[{...trial,task_sha256:'custom-task',goal:'Inspect',verified_success:false,completion_s:null}]},
    {...batch,id:'next',name:'Next architecture',architecture:{...architecture,version:'0.2.0',version_key:'observed-continuous@0.2.0+def'},
      model_variant:{...model,revision:'model-b'},variant_id:'architecture-b/model-b',planned:1,successes:1,trials:[trial]},
    {...batch,id:'scripted',name:'Scripted check',evidence:'scripted_test'}];
  await page.route('**/api/test-results',route=>route.fulfill({json:{batches,skipped:0,truncated:false,scope:'Mocked metadata'}}));
  let sockets=0; page.on('websocket',()=>sockets++);
  await page.goto('/?view=test-results');
  await expect(page.getByRole('combobox',{name:'Filter test evidence',exact:true})).toHaveValue('real_model');
  await page.locator('.variant-progress summary').click();
  const rows=page.locator('.variant-progress tbody tr');
  await expect(rows).toHaveCount(4);
  await expect(rows.first()).toContainText('1 / 2');
  await expect(rows.first()).toContainText('10 s');
  await expect(rows.nth(1)).toContainText('Operator-assisted');
  await expect(rows.nth(2)).toContainText('Inspect');
  await page.getByRole('combobox',{name:'Filter architecture version',exact:true}).selectOption(architecture.version_key);
  await expect(rows).toHaveCount(3);
  await page.getByRole('combobox',{name:'Filter architecture version',exact:true}).selectOption('all');
  await page.getByRole('combobox',{name:'Filter model variant',exact:true}).selectOption('model-b');
  await expect(rows).toHaveCount(1);
  for(const width of [1440,320]) {
    await page.setViewportSize({width,height:1000});
    expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
    const activeTab=page.getByRole('navigation',{name:'Test views'}).getByRole('link',{name:'Test archive',exact:true});
    await expect(activeTab).toHaveAttribute('aria-current','page');
    expect((await activeTab.boundingBox())?.height).toBeGreaterThanOrEqual(44);
    await page.screenshot({path:`test-results/variant-progress-${width}.png`,fullPage:true});
  }
  expect(sockets).toBe(0);
});

test('dotted reference path is visible only in the cockpit world view', async ({page, request}) => {
  await request.post('/api/challenges/load',{data:{challenge_id:'furniture_circuit'}});
  await page.goto('/');
  const toggle=page.getByRole('checkbox',{name:'Show reference path',exact:true});
  const canvas=page.locator('.spectator canvas');
  await expect(toggle).toBeEnabled();
  await expect(canvas).toHaveAttribute('data-reference-points','122');
  const before:LiveState=await(await request.get('/api/state')).json();
  const camera=await(await request.get(before.camera.url)).body();
  for(const width of [1440,390,320]) {
    await page.setViewportSize({width,height:1000});
    await canvas.scrollIntoViewIfNeeded();
    await expect.poll(()=>pixels(page)).toBeGreaterThan(20);
    await toggle.uncheck();
    await expect(canvas).toHaveAttribute('data-reference-points','0');
    const without=await canvas.evaluate((element:HTMLCanvasElement)=>element.toDataURL());
    await toggle.check();
    await expect(canvas).toHaveAttribute('data-reference-points','122');
    await expect.poll(()=>canvas.evaluate((element:HTMLCanvasElement)=>element.toDataURL())).not.toBe(without);
    expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
    await canvas.screenshot({path:`test-results/dotted-reference-${width}.png`});
  }
  const after:LiveState=await(await request.get('/api/state')).json();
  expect(after.snapshot).toEqual(before.snapshot);
  expect(await(await request.get(after.camera.url)).body()).toEqual(camera);
  expect(JSON.stringify(after.observation)).not.toContain('reference');
});


test('test results distinguish initial errors from an empty archive', async ({page}) => {
  let unavailable = true;
  let sockets = 0;
  page.on('websocket',()=>sockets++);
  await page.route('**/api/test-results', route => unavailable?route.fulfill({status:503,json:{detail:'Unavailable'}}):route.fulfill({json:{batches:[],skipped:0,truncated:false,scope:'Saved evaluations'}}));
  await page.goto('/?view=test-results');
  await expect(page.getByRole('heading',{name:'Results unavailable',exact:true})).toBeVisible();
  await expect(page.getByRole('heading',{name:'No saved challenge evaluations',exact:true})).toHaveCount(0);
  unavailable=false;
  await page.getByRole('button',{name:'Refresh saved results',exact:true}).click();
  await expect(page.getByRole('alert')).toHaveCount(0);
  await expect(page.getByRole('heading',{name:'No saved challenge evaluations',exact:true})).toBeVisible();
  expect(sockets).toBe(0);
});


for (const failure of ['network', 'decode']) {
  test(`robot camera automatically recovers from a stalled ${failure} load`, async ({ page }) => {
    if (failure === 'network') {
      let first = true;
      await page.route('**/api/camera/**', route => {
        if (first) { first = false; return route.abort(); }
        return route.continue();
      });
    } else {
      await page.addInitScript(() => {
        const original = HTMLImageElement.prototype.decode;
        let first = true;
        HTMLImageElement.prototype.decode = function () {
          if (first && this.src.startsWith('blob:')) { first = false; return new Promise(() => {}); }
          return original.call(this);
        };
      });
    }
    await page.goto('/');
    await expect(page.getByAltText('Authoritative robot head camera')).toHaveJSProperty('naturalWidth', 640, {timeout: 12000});
    await expect(page.getByText('Camera reconnecting', {exact:true})).toHaveCount(0);
  });
}

test('expanded robot camera remains live, responsive, and stoppable', async ({ page, request }) => {
  await page.setViewportSize({width:1440,height:1000});
  await page.goto('/');
  const camera = page.getByAltText('Authoritative robot head camera');
  await expect(camera).toHaveJSProperty('naturalWidth', 640);
  await page.getByRole('button', {name:'Restore head camera', exact:true}).click();
  await page.getByRole('button', {name:'Expand robot camera', exact:true}).click();
  const dialog = page.getByRole('dialog', {name:'Robot camera', exact:true});
  const expanded = page.getByAltText('Expanded robot head camera');
  await expect(dialog).toBeVisible();
  await expect(expanded).toHaveJSProperty('naturalWidth', 640);
  const before = await expanded.getAttribute('data-frame');
  const state: LiveState = await (await request.get('/api/state')).json();
  const moved = await request.post('/api/command', {data:{run_id:state.run_id,episode_epoch:state.episode_epoch,
    observation_seq:state.observation.seq,action_id:'expanded-head-camera',tool:'set_head',
    arguments:{yaw_rad:.4,pitch_rad:.6,duration_s:1}}});
  expect((await moved.json()).status).toBe('ok');
  await expect(expanded).not.toHaveAttribute('data-frame', before!);
  await expect.poll(() => page.evaluate(() => {
    const camera = document.querySelector<HTMLImageElement>('img[alt="Authoritative robot head camera"]');
    const expanded = document.querySelector<HTMLImageElement>('img[alt="Expanded robot head camera"]');
    return Boolean(camera?.dataset.frame && expanded?.dataset.frame === camera.dataset.frame &&
      expanded.src === camera.src && expanded.naturalWidth === 640);
  })).toBe(true);
  for (const width of [1440,390,320]) {
    await page.setViewportSize({width,height:900});
    await expect(dialog.getByRole('button',{name:'Stop',exact:true})).toBeVisible();
    const geometry = await dialog.evaluate(element => {
      const bounds = element.getBoundingClientRect();
      const image = element.querySelector('img')!.getBoundingClientRect();
      const toolbar = element.querySelector('.robot-camera-toolbar')!.getBoundingClientRect();
      return {fits:bounds.left>=0 && bounds.right<=innerWidth && bounds.top>=0 && bounds.bottom<=innerHeight,
        noOverlap:image.top>=toolbar.bottom, imageHeight:image.height, noOverflow:element.scrollWidth<=element.clientWidth};
    });
    expect(geometry.fits && geometry.noOverlap && geometry.noOverflow).toBe(true);
    expect(geometry.imageHeight).toBeGreaterThan(100);
    await dialog.screenshot({path:`test-results/robot-camera-${width}.png`});
  }
  const stopped = page.waitForResponse(response=>response.url().endsWith('/api/stop') && response.request().method()==='POST');
  await dialog.getByRole('button',{name:'Stop',exact:true}).click();
  expect((await stopped).ok()).toBe(true);
  expect((await (await request.get('/api/state')).json()).stopped).toBe(true);
  await page.keyboard.press('Escape');
  await expect(dialog).not.toBeVisible();
  await expect(page.getByRole('button', {name:'Expand robot camera',exact:true})).toBeFocused();
});

test('spatial sensing pairs camera frames and preserves unknown space across desktop and mobile', async ({ page, request }) => {
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.goto('/');
  await page.locator('.spatial-section summary').click();
  await page.getByRole('checkbox', { name: 'Enable spatial sensing', exact: true }).check();
  const rgb = page.getByRole('img', { name: 'Spatial paired RGB', exact: true });
  const depth = page.getByRole('img', { name: 'Spatial metric depth', exact: true });
  await expect(rgb).toBeVisible();
  await expect(depth).toBeVisible();
  await expect.poll(async () => rgb.evaluate((image: HTMLImageElement) => image.naturalWidth)).toBe(160);
  expect(await rgb.getAttribute('data-sequence')).toBe(await depth.getAttribute('data-sequence'));
  const initial = await (await request.get('/api/spatial')).json();
  expect(initial.map.unknown_cells).toBeGreaterThan(20000);
  expect(initial.map.obstacle_cells).toBeGreaterThan(0);
  const state: LiveState = await (await request.get('/api/state')).json();
  const command = await request.post('/api/command', { data: { run_id: state.run_id, episode_epoch: state.episode_epoch,
    observation_seq: state.observation.seq, action_id: 'spatial-head-test', tool: 'set_head',
    arguments: { yaw_rad: .6, pitch_rad: .65, duration_s: 1 } } });
  expect((await command.json()).status).toBe('ok');
  await expect.poll(async () => Number(await rgb.getAttribute('data-sequence'))).toBeGreaterThan(initial.frame.sequence);
  for (const width of [1440, 390, 320]) {
    await page.setViewportSize({ width, height: 1000 });
    await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    const colors = await page.getByLabel('Observed floor and obstacle map').evaluate((canvas: HTMLCanvasElement) => {
      const data = canvas.getContext('2d')!.getImageData(0, 0, canvas.width, canvas.height).data;
      const colors = new Set<string>();
      for (let offset = 0; offset < data.length; offset += 4) colors.add(`${data[offset]},${data[offset + 1]},${data[offset + 2]}`);
      return colors.size;
    });
    expect(colors).toBeGreaterThan(3);
    await page.locator('.spatial-section').screenshot({ path: `test-results/spatial-${width}.png` });
  }
  const disabled = page.waitForResponse(response => response.url().endsWith('/api/spatial') && response.request().method() === 'POST');
  await page.getByRole('checkbox', { name: 'Enable spatial sensing', exact: true }).uncheck();
  expect((await disabled).ok()).toBe(true);
  await expect(rgb).toHaveCount(0);
  expect((await request.get(initial.frame.data_url)).status()).toBe(404);
});


test('scripted map overlays show sightings and fading footprints without granting motion', async ({ page, request }) => {
  await request.post('/api/challenges/load', {data: {challenge_id: 'park'}});
  const live: LiveState = await (await request.get('/api/state')).json();
  await request.post('/api/spatial', {data: {run_id: live.run_id, episode_epoch: live.episode_epoch, enabled: true}});
  const data = await (await request.get('/api/spatial')).json();
  data.frame = null;
  data.history = {trail_retention_s: 300, footprints: [{age_s: 60,
    polygon_m: [[-1.2, -.2], [-.8, -.2], [-.8, .2], [-1.2, .2]]}], labels: [
    {id: 'kitchen', label: 'Kitchen', evidence: 'Scripted stove and sink', position_m: [.5, .5], age_s: 10, stale: false, observation_seq: 1},
    {id: 'bathroom', label: 'Bathroom', evidence: 'Scripted tub and toilet', position_m: [.6, .5], age_s: 130, stale: true, observation_seq: 2}]};
  await page.route('**/api/spatial', route => route.fulfill({json: data}));
  await page.goto('/');
  await page.locator('.spatial-section > summary').click();
  const map = page.getByLabel('Observed floor and obstacle map');
  await expect(map).toBeVisible();
  const image = () => map.evaluate((canvas: HTMLCanvasElement) => canvas.toDataURL());
  const timeout = page.getByRole('combobox', {name: 'Trail timeout', exact: true});
  await timeout.selectOption('0');
  const absent = await image();
  await timeout.selectOption('30');
  expect(await image()).toBe(absent);
  await timeout.selectOption('120');
  await expect.poll(image).not.toBe(absent);
  const withLabels = await image();
  await page.getByRole('checkbox', {name: 'Luna sightings', exact: true}).uncheck();
  await expect.poll(image).not.toBe(withLabels);
  await page.getByRole('checkbox', {name: 'Luna sightings', exact: true}).check();
  await page.getByText('Unverified place sightings (2)', {exact: true}).click();
  await expect(page.getByText('Scripted stove and sink', {exact: true})).toBeVisible();
  for (const width of [1440, 390, 320]) {
    await page.setViewportSize({width, height: 1000});
    await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    expect(await map.evaluate((canvas: HTMLCanvasElement) => {
      const rgba = canvas.getContext('2d')!.getImageData(0, 0, 320, 320).data;
      const colors = new Set<string>();
      for (let offset = 0; offset < rgba.length; offset += 4) colors.add(`${rgba[offset]},${rgba[offset + 1]},${rgba[offset + 2]}`);
      return colors.size;
    })).toBeGreaterThan(5);
    await page.locator('.spatial-section').screenshot({path: `test-results/map-history-${width}.png`});
  }
  await page.getByRole('checkbox', {name: 'Luna sightings', exact: true}).uncheck();
  await timeout.selectOption('0');
  const noTrail = await image();
  await timeout.selectOption('120');
  await page.unroute('**/api/spatial');
  await page.route('**/api/spatial', route => route.abort());
  await page.clock.install();
  await page.clock.fastForward(121000);
  await expect.poll(image).toBe(noTrail);
  expect((await (await request.get('/api/state')).json()).snapshot.simulated_time_s).toBe(live.snapshot.simulated_time_s);
});


test('continuous local navigation tracks a camera destination without model calls or buffer stops', async ({ page, request }) => {
  await request.post('/api/challenges/load', {data:{challenge_id:'park'}});
  await page.setViewportSize({width:1440,height:1100});
  await page.goto('/');
  await page.getByRole('switch',{name:'Sensors and areas',exact:true}).check();
  await page.getByRole('tab',{name:'Telemetry',exact:true}).click();
  await page.locator('.spatial-section summary').click();
  await page.getByRole('checkbox',{name:'Fold arms during floor scan',exact:true}).check();
  const scan = page.waitForResponse(response=>response.url().endsWith('/api/continuous/scan'));
  await page.getByRole('button',{name:'Scan floor',exact:true}).click();
  expect((await scan).ok()).toBe(true);
  const rgb = page.getByAltText('Spatial paired RGB');
  await expect(rgb).toHaveJSProperty('naturalWidth',160);
  await page.getByRole('button',{name:'Select destination',exact:true}).click();
  const selection = page.getByRole('button',{name:'Drive continuously to floor point',exact:true});
  const bounds = await selection.boundingBox();
  const start = page.waitForResponse(response=>response.url().endsWith('/api/continuous/start'));
  await selection.click({position:{x:bounds!.width*.5,y:bounds!.height*.4}});
  const response = await start;
  expect(response.ok(), await response.text()).toBe(true);
  await expect(page.getByLabel('Continuous navigation status')).toHaveAttribute('data-status','running');
  await expect(page.getByRole('button',{name:'Start LLM control',exact:true})).toBeDisabled();
  await expect(page.getByRole('button',{name:'Drive forward',exact:true})).toBeHidden();
  await expect(page.getByRole('button',{name:'Stop',exact:true})).toBeVisible();
  const during: LiveState = await (await request.get('/api/state')).json();
  const initialFrame = during.camera.seq;
  const overlay = page.locator('.spectator canvas');
  await expect(overlay).toHaveAttribute('data-beam-state','Live');
  await expect(overlay).toHaveAttribute('data-zone-state','Observed');
  const firstAreaPose = await overlay.getAttribute('data-zone-origin');
  await expect.poll(()=>overlay.getAttribute('data-zone-origin')).not.toBe(firstAreaPose);
  await expect.poll(async()=>((await (await request.get('/api/state')).json()) as LiveState).camera.seq).toBeGreaterThan(initialFrame);
  await expect(page.getByLabel('Continuous navigation status')).toHaveAttribute('data-status','arrived',{timeout:30000});
  const sensor = await (await request.get('/api/spatial')).json();
  expect(sensor.continuous.buffer_stops).toBe(0);
  expect(sensor.history.footprints.length).toBeGreaterThan(3);
  expect(sensor.continuous.remaining_m).toBeLessThan(.06);
  expect(sensor.continuous.updates).toBeGreaterThan(20);
  const after: LiveState = await (await request.get('/api/state')).json();
  expect(after.agent.input_tokens+after.agent.output_tokens).toBe(0);
  expect(after.agent.active).toBe(false);
  expect(after.proximity.collisions).toHaveLength(0);
  await expect(overlay).toHaveAttribute('data-trail-visible','true');
  await expect.poll(async()=>Number(await overlay.getAttribute('data-trail-segments'))).toBeGreaterThan(10);
  const actualBase=after.snapshot.poses.find(pose=>pose.key===`${after.robot_body_id}:-1`)!;
  await expect.poll(async()=>{
    const endpoint=JSON.parse((await overlay.getAttribute('data-trail-end'))!);
    return Math.hypot(endpoint[0]-actualBase.position[0],endpoint[1]-actualBase.position[1]);
  }).toBeLessThan(.04);
  await writeFile('../.runtime/ground-trail-v1/physics-route.json',JSON.stringify({evidence:'scripted_camera_destination_real_physics',
    continuous:sensor.continuous,first_area_pose:firstAreaPose,final_area_pose:await overlay.getAttribute('data-zone-origin'),
    trail_segments:Number(await overlay.getAttribute('data-trail-segments')),trail_end:JSON.parse((await overlay.getAttribute('data-trail-end'))!),
    actual_base_position:actualBase.position,
    input_tokens:after.agent.input_tokens,output_tokens:after.agent.output_tokens,contacts:after.proximity.collisions},null,2));
  await page.locator('.world-panel').screenshot({path:'../.runtime/ground-trail-v1/physics-route.png'});
  for(const width of [1440,390,320]) {
    await page.setViewportSize({width,height:1000});
    await expect.poll(()=>page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
    await page.locator('.spatial-section').screenshot({path:`test-results/continuous-${width}.png`});
  }
});


test('motion history contact sheets and originals are visible and Stop remains authoritative', async ({page, request}) => {
  test.setTimeout(90000);
  await request.post('/api/agent/config', {data:{endpoint:'https://test.openai.azure.com',models:[
    {id:'luna',label:'Luna',deployment:'scripted-history',reasoning_efforts:['low','medium','high']}]}});
  await page.setViewportSize({width:1440,height:1000});
  await page.goto('/');
  await page.getByRole('textbox',{name:'Robot goal',exact:true}).fill('Review motion camera history.');
  await page.getByRole('button',{name:'Start LLM control',exact:true}).click();
  await expect.poll(async () => {
    const trace: ExchangeFeed = await (await request.get('/api/agent/trace')).json();
    return trace.events.some(entry => entry.kind === 'feedback' && !!entry.payload.historical_original);
  }, {timeout:60000}).toBe(true);
  await page.getByRole('tab',{name:'Trace',exact:true}).click();
  await page.getByRole('button',{name:'Inputs',exact:true}).click();
  const sheet = page.locator('img[alt^="Motion history / "]').first();
  const original = page.locator('img[alt^="Historical original view-"]').first();
  await page.locator('.exchange-entry').filter({has:sheet}).locator('.exchange-disclosure > summary').click();
  const originalDisclosure = page.locator('.exchange-entry').filter({has:original}).locator('.exchange-disclosure');
  if (!await originalDisclosure.evaluate((element: HTMLDetailsElement) => element.open)) await originalDisclosure.locator(':scope > summary').click();
  await sheet.scrollIntoViewIfNeeded();
  await expect(sheet).toHaveJSProperty('naturalWidth',332);
  await expect(sheet).toHaveJSProperty('naturalHeight',332);
  await original.scrollIntoViewIfNeeded();
  await expect(original).toHaveJSProperty('naturalWidth',160);
  await expect(original).toHaveJSProperty('naturalHeight',120);
  const trace: ExchangeFeed = await (await request.get('/api/agent/trace')).json();
  const sheetEvent = trace.events.find(entry => entry.kind === 'feedback' && entry.payload.camera_history?.frames.length);
  const originalEvent = trace.events.find(entry => entry.kind === 'feedback' && entry.payload.historical_original);
  expect(sheetEvent?.image_urls).toHaveLength(2);
  expect(originalEvent?.image_urls?.length).toBeLessThanOrEqual(3);
  await writeFile('test-results/motion-history-sheet.png',await (await request.get(sheetEvent!.image_urls![1])).body());
  await writeFile('test-results/motion-history-original.png',await (await request.get(originalEvent!.image_urls!.at(-1)!)).body());
  for(const width of [1440,390,320]) {
    await page.setViewportSize({width,height:1000});
    await original.scrollIntoViewIfNeeded();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await expect(page.getByRole('button',{name:'Stop',exact:true})).toBeVisible();
    await page.locator('.exchange-section').screenshot({path:`test-results/motion-history-trace-${width}.png`});
  }
  const before: LiveState = await (await request.get('/api/state')).json();
  const stopped = page.waitForResponse(response => response.url().endsWith('/api/stop'));
  await page.getByRole('button',{name:'Stop',exact:true}).click();
  expect((await stopped).ok()).toBe(true);
  const after: LiveState = await (await request.get('/api/state')).json();
  expect(after.stopped).toBe(true);
  expect(after.agent.active).toBe(false);
  expect(after.snapshot.simulated_time_s).toBe(before.snapshot.simulated_time_s);
  expect(after.observation.odometry_m_rad).toEqual(before.observation.odometry_m_rad);
});


test('scenario previews explain each task without loading or moving the robot', async ({page, request}) => {
  await page.setViewportSize({width:1440,height:1000});
  await page.goto('/');
  const goal = await page.getByRole('textbox',{name:'Robot goal',exact:true}).inputValue();
  await openChallengeMenu(page);
  const selector = page.getByRole('combobox',{name:'Predefined challenge',exact:true});
  const overview = page.getByLabel('Selected scenario preview', {exact:true});
  const presets = await (await request.get('/api/challenges')).json();
  await expect(selector.locator('option')).toHaveCount(presets.length + 1);
  const before: LiveState = await (await request.get('/api/state')).json();
  const sceneLoads: string[] = [];
  page.on('request', request => { if (request.url().endsWith('/api/challenges/load') && request.method() === 'POST') sceneLoads.push(request.postData() ?? ''); });
  for (const preset of [...presets, {id:'bench',title:'Practice bench',category:'Practice'}]) {
    await selector.selectOption(preset.id);
    await expect(overview.locator('strong')).toHaveText(preset.title);
    await expect(overview.locator('.scenario-facts')).toContainText(preset.category);
    await expect(overview.locator('.scenario-summary')).not.toBeEmpty();
    await expect(overview.locator('.scenario-completion')).not.toBeEmpty();
    await expect(overview.locator('.scenario-selection-state')).toHaveText(preset.id === 'bench' ? 'Loaded' : 'Not loaded');
    const image = overview.getByRole('img', {name:`Scene preview: ${preset.title}`,exact:true});
    await expect(image).toHaveJSProperty('naturalWidth',480);
    await expect(image).toHaveJSProperty('naturalHeight',300);
    await expect(overview.locator('.scenario-thumbnail')).toHaveAttribute('data-state','ready');
    const colors = await image.evaluate((source: HTMLImageElement) => {
      const canvas = document.createElement('canvas'); canvas.width=120; canvas.height=75;
      const context = canvas.getContext('2d')!; context.drawImage(source,0,0,120,75);
      const samples = context.getImageData(0,0,120,75).data;
      const distinct = new Set<string>();
      for(let offset=0;offset<samples.length;offset+=16) distinct.add(`${samples[offset]},${samples[offset+1]},${samples[offset+2]}`);
      return distinct.size;
    });
    expect(colors).toBeGreaterThan(20);
    await expect(page.locator('.challenge-goal')).toBeHidden();
  }
  let failImage = true;
  await page.route('**/scenario-previews/park.webp*', route => {
    if (failImage) { failImage=false; return route.fulfill({status:503,body:'Preview temporarily unavailable'}); }
    return route.continue();
  });
  await selector.selectOption('park');
  await expect(overview.getByText('Preview unavailable',{exact:true})).toBeVisible();
  await expect(page.getByRole('button',{name:'Load selected scenario',exact:true})).toBeEnabled();
  await overview.getByRole('button',{name:'Retry scene preview',exact:true}).click();
  await expect(overview.locator('.scenario-thumbnail')).toHaveAttribute('data-state','ready');
  await selector.selectOption('furniture_circuit');
  await page.getByRole('combobox',{name:'Object to circle',exact:true}).selectOption('floor lamp');
  await page.getByRole('combobox',{name:'Circuit direction',exact:true}).selectOption('counterclockwise');
  await expect(overview.locator('.scenario-completion')).toContainText('floor lamp counterclockwise');
  await expect(overview.locator('.scenario-thumbnail')).toHaveAttribute('data-state','ready');
  for (const width of [1440,768,390,320]) {
    await page.setViewportSize({width,height:1000});
    await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    const sizes = await overview.evaluate(element => {
      const image=element.querySelector('.scenario-thumbnail')!.getBoundingClientRect();
      const text=element.querySelector('.scenario-brief')!.getBoundingClientRect();
      return {imageWidth:image.width,imageHeight:image.height,overlap: image.left < text.right && image.right > text.left && image.top < text.bottom && image.bottom > text.top};
    });
    expect(sizes.imageWidth).toBeGreaterThanOrEqual(140);
    expect(sizes.imageHeight).toBeCloseTo(sizes.imageWidth * 5 / 8, 0);
    expect(sizes.overlap).toBe(false);
    const controls = await page.locator('.challenge-toolbar select').evaluateAll(elements => elements.map(element => {
      const select = element.getBoundingClientRect();
      return {width:select.width,left:select.left,right:select.right};
    }));
    expect(controls.every(control => control.width >= 140 && control.left >= 0 && control.right <= width), JSON.stringify({width,controls})).toBe(true);
    await page.locator('.scenario-setup').screenshot({path:`test-results/scenario-picker-${width}.png`});
  }
  const after: LiveState = await (await request.get('/api/state')).json();
  expect(after.run_id).toBe(before.run_id);
  expect(after.snapshot).toEqual(before.snapshot);
  expect(after.agent.active).toBe(false);
  expect(after.agent.input_tokens + after.agent.output_tokens).toBe(0);
  expect(sceneLoads).toEqual([]);
  await page.getByRole('button',{name:'Close challenge menu',exact:true}).click();
  await expect(page.getByRole('textbox',{name:'Robot goal',exact:true})).toHaveValue(goal);
});


test('furniture circuits expose object commands and retain selection on reset', async ({page, request}) => {
  await page.addInitScript(() => localStorage.setItem('milo-navigation-mode', 'luna_navigation'));
  await page.goto('/');
  await openChallengeMenu(page);
  await expect(page.getByRole('combobox',{name:'Object to circle',exact:true})).toHaveCount(0);
  await page.getByRole('combobox',{name:'Predefined challenge',exact:true}).selectOption('furniture_circuit');
  await page.getByRole('combobox',{name:'Object to circle',exact:true}).selectOption('chair');
  await page.getByRole('combobox',{name:'Circuit direction',exact:true}).selectOption('counterclockwise');
  const loading = page.waitForResponse(response => response.url().endsWith('/api/challenges/load'));
  await page.getByRole('button',{name:'Load selected scenario',exact:true}).click();
  const response = await loading;
  expect(response.ok()).toBe(true);
  expect(response.request().postDataJSON()).toMatchObject({challenge_id:'furniture_circuit',orbit_target:'chair',orbit_direction:'counterclockwise'});
  await expect(page.getByRole('heading',{name:'Circle the Furniture',exact:true})).toBeVisible();
  await expect(page.getByRole('textbox',{name:'Robot goal',exact:true})).toContainText('counterclockwise circuit around that chair');
  await openChallengeMenu(page);
  await page.locator('.challenge-details > summary').click();
  await expect(page.locator('.challenge-objectives')).toContainText('chair counterclockwise');
  await page.getByRole('button',{name:'Close challenge menu',exact:true}).click();
  const state: LiveState = await (await request.get('/api/state')).json();
  expect(state.challenge?.orbit).toEqual({target:'chair',direction:'counterclockwise'});
  expect(JSON.stringify(state.observation)).not.toContain('circuit_');
  expect(state.agent.active).toBe(false);
  for(const width of [1440,390,320]) {
    await page.setViewportSize({width,height:1000});
    await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await expect.poll(() => pixels(page)).toBeGreaterThan(40);
    await page.getByAltText('Authoritative robot head camera').scrollIntoViewIfNeeded();
    await expect(page.getByAltText('Authoritative robot head camera')).toHaveJSProperty('naturalWidth',640);
    await page.locator('.spectator-shell').screenshot({path:`test-results/furniture-circuit-${width}.png`});
  }
  if (await page.locator('.robot-options').getAttribute('open') === null) await page.locator('.robot-options > summary').click();
  await page.getByRole('button',{name:'Reset episode',exact:true}).click();
  await expect.poll(async () => (await (await request.get('/api/state')).json()).run_id).not.toBe(state.run_id);
  await openChallengeMenu(page);
  await expect(page.getByRole('combobox',{name:'Object to circle',exact:true})).toHaveValue('chair');
  await expect(page.getByRole('combobox',{name:'Circuit direction',exact:true})).toHaveValue('counterclockwise');
});


test('Luna supervision remains available alongside continuous local navigation', async ({ page, request }) => {
  await page.goto('/');
  await expect(page.getByRole('button', { name: 'Start LLM control', exact: true })).toBeHidden();
  await page.getByRole('textbox', { name: 'Foundry endpoint', exact: true }).fill('https://test.openai.azure.com');
  await page.getByRole('textbox', { name: 'Luna deployment', exact: true }).fill('scripted-supervisor');
  const saved = page.waitForResponse(response => response.url().endsWith('/api/agent/config') && response.request().method() === 'POST');
  await page.getByRole('button', { name: 'Apply Luna connection', exact: true }).click();
  expect((await saved).ok()).toBe(true);
  await page.locator('.run-options > summary').click();
  for (const width of [1440, 390]) {
    await page.setViewportSize({width,height:1000});
    const toggle = page.getByRole('checkbox',{name:'Plan while moving',exact:true});
    await expect(toggle).toBeVisible();
    const geometry = await toggle.evaluate(input => ({width:input.getBoundingClientRect().width,
      height:input.getBoundingClientRect().height, direction:getComputedStyle(input.parentElement!).flexDirection,
      overflow:document.documentElement.scrollWidth>innerWidth}));
    expect(geometry).toEqual({width:16,height:16,direction:'row',overflow:false});
  }
  await expect(page.getByRole('heading', { name: 'Luna', exact: true })).toBeVisible();
  await expect(page.getByRole('combobox', { name: 'Navigation controller', exact: true })).toHaveValue('luna_continuous');
  const composer = page.getByRole('checkbox', {name:'Motion skill composer', exact:true});
  await expect(composer).not.toBeChecked();
  await composer.check();
  await expect(page.getByLabel('Current architecture version')).toContainText('Composed motion skills');
  await composer.uncheck();
  await expect(page.getByLabel('Current architecture version')).toContainText('Observed continuous control');
  for (const name of ['Single step', 'Local SmolVLA navigation', 'Navigation plan', 'Luna + SmolVLA']) {
    await expect(page.getByRole('button', { name, exact: true })).toHaveCount(0);
  }
  await expect(page.getByRole('button', { name: 'Start LLM control', exact: true })).toBeEnabled();
  const state: LiveState = await (await request.get('/api/state')).json();
  expect(state.agent.active).toBe(false);
  expect(state.snapshot.simulated_time_s).toBe(0);
});

test('Nav2 primary exposes readiness and preserves an explicit built-in backup', async ({page, request}) => {
  await request.post('/api/agent/config', {data:{endpoint:'https://test.openai.azure.com',models:[
    {id:'luna',label:'Luna',deployment:'scripted-supervisor',reasoning_efforts:['low','medium','high']}]}});
  let ready = false;
  const payloads: Record<string, unknown>[] = [];
  await page.route('**/api/ros/status', route => route.fulfill({json:{enabled:true,ready,online:ready,message:'Nav2 bridge offline'}}));
  await page.route('**/api/test-variant?*', route => {
    const selected = new URL(route.request().url()).searchParams.get('navigation_backend') === 'nav2';
    return route.fulfill({json:{architecture:{name:selected?'Nav2 observed goal control':'Observed continuous control',version:'0.1.0',revision:'test'},
      supports_navigation_backend:true,supports_skill_composer:true,nav2:{enabled:true,ready,online:ready,message:'Nav2 bridge offline'}}});
  });
  await page.route('**/api/agent/start', async route => {
    payloads.push(route.request().postDataJSON());
    await route.fulfill({status:409,json:{detail:'Scripted payload capture; no inference'}});
  });
  await page.goto('/');
  await page.locator('.run-options > summary').click();
  const stack = page.getByRole('combobox',{name:'Navigation stack',exact:true});
  const start = page.getByRole('button',{name:'Start LLM control',exact:true});
  await expect(stack).toHaveValue('nav2');
  await expect(start).toBeDisabled();
  await expect(page.getByLabel('Nav2 readiness')).toContainText('offline');
  await expect(page.getByRole('checkbox',{name:'Motion skill composer',exact:true})).toHaveCount(0);
  ready = true;
  await expect(start).toBeEnabled({timeout:10000});
  await start.click();
  await expect.poll(()=>payloads.length).toBe(1);
  expect(payloads[0]).toMatchObject({navigation_backend:'nav2',continuous_handoff:false});
  expect(payloads[0]).not.toHaveProperty('skill_composer');
  await stack.selectOption('builtin');
  await expect(page.getByLabel('Current architecture version')).toContainText('Observed continuous control');
  await expect(page.getByRole('checkbox',{name:'Motion skill composer',exact:true})).toBeVisible();
  await start.click();
  await expect.poll(()=>payloads.length).toBe(2);
  expect(payloads[1]).toMatchObject({navigation_backend:'builtin'});
  for (const width of [1440,390]) {
    await page.setViewportSize({width,height:1000});
    expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
  }
  const state: LiveState = await (await request.get('/api/state')).json();
  expect(state.agent.active).toBe(false);
});

test('motion skill composer is opt-in and Start submits its selected mode', async ({page, request}) => {
  await request.post('/api/agent/config', {data:{endpoint:'https://test.openai.azure.com',models:[
    {id:'luna',label:'Luna',deployment:'scripted-supervisor',reasoning_efforts:['low','medium','high']}]}});
  await page.goto('/');
  await page.locator('.run-options > summary').click();
  await page.getByRole('checkbox',{name:'Motion skill composer',exact:true}).check();
  await expect(page.getByLabel('Current architecture version')).toContainText('Composed motion skills');
  const started = page.waitForResponse(response=>response.url().endsWith('/api/agent/start'));
  await page.getByRole('button',{name:'Start LLM control',exact:true}).click();
  const response = await started;
  expect(response.ok(),await response.text()).toBe(true);
  expect(response.request().postDataJSON()).toMatchObject({execution_mode:'luna_continuous',skill_composer:true});
  await page.getByRole('button',{name:'Stop',exact:true}).click();
  await expect.poll(async()=>((await (await request.get('/api/state')).json()) as LiveState).stopped).toBe(true);
  const stopped: LiveState = await (await request.get('/api/state')).json();
  await expect.poll(async()=>((await (await request.get('/api/state')).json()) as LiveState).agent.active,{timeout:15000}).toBe(false);
  const finalized: LiveState = await (await request.get('/api/state')).json();
  expect(finalized.snapshot.simulated_time_s).toBe(stopped.snapshot.simulated_time_s);
});

test('five-room kitchen search defaults to moving plans and Stop cancels preparation', async ({page, request}) => {
  await request.post('/api/challenges/load', {data:{challenge_id:'flat_kitchen'}});
  await request.post('/api/agent/config', {data:{endpoint:'https://test.openai.azure.com',models:[
    {id:'luna',label:'Luna',deployment:'scripted-supervisor',reasoning_efforts:['low','medium','high']}]}});
  await page.goto('/?graphics=enhanced');
  await expect(page.getByRole('heading',{name:'Find the Kitchen',exact:true})).toBeVisible();
  await page.locator('.run-options > summary').click();
  const mode = page.getByRole('combobox',{name:'Navigation controller',exact:true});
  const handoff = page.getByRole('checkbox',{name:'Plan while moving',exact:true});
  await expect(mode).toHaveValue('luna_continuous');
  await expect(handoff).toBeChecked();
  for (const width of [1440,390,320]) {
    await page.setViewportSize({width,height:1000});
    await expect.poll(()=>page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
    await expect(page.getByAltText('Authoritative robot head camera')).toHaveJSProperty('naturalWidth',640);
    await page.locator('.spectator-shell').screenshot({path:`test-results/flat-kitchen-enhanced-${width}.png`});
  }
  const before: LiveState = await (await request.get('/api/state')).json();
  expect(before.snapshot.simulated_time_s).toBe(0);
  expect(before.agent.active).toBe(false);
  const started = page.waitForResponse(response=>response.url().endsWith('/api/agent/start'));
  await page.getByRole('button',{name:'Start LLM control',exact:true}).click();
  const response = await started;
  expect(response.ok(),await response.text()).toBe(true);
  expect(response.request().postDataJSON()).toMatchObject({execution_mode:'luna_continuous',continuous_handoff:true,
    adaptive_navigation:true,compact_arms:true});
  const stopping = page.waitForResponse(reply=>reply.url().endsWith('/api/stop'));
  await page.getByRole('button',{name:'Stop',exact:true}).click();
  await expect.poll(async()=>((await (await request.get('/api/state')).json()) as LiveState).stopped).toBe(true);
  const latched: LiveState = await (await request.get('/api/state')).json();
  expect((await stopping).ok()).toBe(true);
  const stopped: LiveState = await (await request.get('/api/state')).json();
  expect(stopped.agent.active).toBe(false);
  expect(stopped.stopped).toBe(true);
  expect(stopped.snapshot.simulated_time_s).toBe(latched.snapshot.simulated_time_s);
  expect(stopped.local_navigation_model?.phase).toBe('unloaded');
  if (await page.locator('.robot-options').getAttribute('open') === null) await page.locator('.robot-options > summary').click();
  await page.getByRole('button',{name:'Reset episode',exact:true}).click();
  await expect(page.getByRole('heading',{name:'Find the Kitchen',exact:true})).toBeVisible();
  await expect(page.locator('.run-options')).toHaveJSProperty('open', true);
  await expect(handoff).toBeChecked();
});

test('continuous settings survive kitchen load and scripted preparation is stoppable', async ({page, request}) => {
  test.setTimeout(90000);
  await request.post('/api/agent/config', {data:{endpoint:'https://test.openai.azure.com',models:[
    {id:'luna',label:'Luna',deployment:'scripted-supervisor',reasoning_efforts:['low','medium','high']}]}});
  await page.goto('/');
  await page.locator('.spatial-section summary').click();
  const sensing = page.getByRole('checkbox',{name:'Enable spatial sensing',exact:true});
  const folding = page.getByRole('checkbox',{name:'Fold arms during floor scan',exact:true});
  await expect(sensing).toBeEnabled();
  await sensing.check();
  await expect(page.getByAltText('Spatial paired RGB')).toHaveJSProperty('naturalWidth',160);
  await folding.uncheck();
  await openChallengeMenu(page);
  await page.getByRole('combobox',{name:'Predefined challenge',exact:true}).selectOption('kitchen_bathroom');
  await page.getByRole('button',{name:'Load selected scenario',exact:true}).click();
  await expect.poll(async()=>((await (await request.get('/api/state')).json()) as LiveState).challenge?.id).toBe('kitchen_bathroom');
  await page.locator('.spatial-section summary').click();
  await expect(sensing).toBeChecked();
  await expect(folding).not.toBeChecked();
  await folding.check();
  await openChallengeMenu(page);
  await page.getByRole('combobox',{name:'Predefined challenge',exact:true}).selectOption('park');
  await page.getByRole('button',{name:'Load selected scenario',exact:true}).click();
  await expect.poll(async()=>((await (await request.get('/api/state')).json()) as LiveState).challenge?.id).toBe('park');
  await page.locator('.spatial-section summary').click();
  await expect(folding).toBeChecked();
  const started = page.waitForResponse(response=>response.url().endsWith('/api/agent/start'));
  await page.locator('.run-options > summary').click();
  await page.getByRole('checkbox',{name:'Plan while moving',exact:true}).check();
  await page.getByRole('button',{name:'Start LLM control',exact:true}).click();
  const payload = (await started).request().postDataJSON();
  expect(payload.execution_mode).toBe('luna_continuous');
  expect(payload.continuous_handoff).toBe(true);
  expect(payload.adaptive_navigation).toBe(true);
  await expect.poll(async()=>{
    const current: LiveState = await (await request.get('/api/state')).json();
    return current.agent.active && current.busy && current.snapshot.simulated_time_s > 0;
  },{timeout:65000}).toBe(true);
  await expect(sensing).toBeChecked();
  await expect(folding).toBeHidden();
  await expect(page.getByRole('button',{name:'Scan floor',exact:true})).toBeHidden();
  await expect(page.getByRole('region',{name:'Local model startup',exact:true})).toHaveCount(0);
  const moving: LiveState = await (await request.get('/api/state')).json();
  expect(moving.local_navigation_model?.phase).toBe('unloaded');
  await page.getByRole('button',{name:'Stop',exact:true}).click();
  await expect.poll(async()=>((await (await request.get('/api/state')).json()) as LiveState).agent.active).toBe(false);
  expect(((await (await request.get('/api/state')).json()) as LiveState).stopped).toBe(true);
});

for (const mode of ['luna_continuous', 'luna_navigation']) {
  test(`run chat redirects ${mode} without resetting the scene or movement memory`, async ({page, request}) => {
    test.setTimeout(120000);
    await request.post('/api/challenges/load', {data:{challenge_id:'park'}});
    await request.post('/api/agent/config', {data:{endpoint:'https://test.openai.azure.com',models:[
      {id:'luna',label:'Luna',deployment:'scripted-supervisor',reasoning_efforts:['low','medium','high']}]}});
    await page.goto('/');
    await page.locator('.run-options > summary').click();
    await page.getByRole('combobox',{name:'Navigation controller',exact:true}).selectOption(mode);
    await page.getByRole('button',{name:'Start LLM control',exact:true}).click();
    await page.getByRole('tab', {name:'Conversation',exact:true}).click();
    await expect.poll(async()=>{
      const state: LiveState = await (await request.get('/api/state')).json();
      return state.agent.active && (mode==='luna_continuous' ? state.continuous_navigation?.status==='running' : (state.agent.local_model?.requests_completed ?? 0)>=2);
    },{timeout:80000}).toBe(true);
    const before: LiveState = await (await request.get('/api/state')).json();
    const memoryBefore = before.agent.run_memory!.revision;
    const instruction = 'Inspect the doorway on the right. Do not repeat the completed turn unless a new view is needed.';
    await page.getByRole('textbox',{name:'New run instruction',exact:true}).fill(instruction);
    const applied=page.waitForResponse(response=>response.url().endsWith('/api/agent/instruction'));
    await page.getByRole('button',{name:'Send instruction',exact:true}).click();
    const response=await applied;
    expect(response.ok(), await response.text()).toBe(true);
    await expect(page.getByRole('log',{name:'Run conversation',exact:true})).toContainText(instruction);
    await expect(page.getByRole('log',{name:'Run conversation',exact:true})).toContainText('Instruction applied');
    const after: LiveState = await (await request.get('/api/state')).json();
    expect(after.run_id).toBe(before.run_id);
    expect(after.agent.session_id).not.toBe(before.agent.session_id);
    expect(after.agent.goal).toBe(instruction);
    expect(after.agent.execution_mode).toBe(mode);
    expect(after.snapshot.simulated_time_s).toBeGreaterThanOrEqual(before.snapshot.simulated_time_s);
    expect(after.agent.run_memory!.revision).toBeGreaterThanOrEqual(memoryBefore);
    expect(after.agent.run_memory!.recent_actions.some(action=>action.action==='operator_redirect')).toBe(true);
    expect(after.local_navigation_model?.load_count).toBe(before.local_navigation_model?.load_count);
    await page.getByRole('button',{name:'Stop',exact:true}).click();
    await expect(page.getByRole('button',{name:'Send instruction',exact:true})).toBeHidden();
    await page.getByRole('tab', {name:'Trace',exact:true}).click();
    await page.locator('.run-memory summary').click();
    await page.getByRole('tab', {name:'Conversation',exact:true}).click();
    for(const width of [1440,390]) {
      await page.setViewportSize({width,height:1000});
      expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
      await page.getByRole('region',{name:'Run chat',exact:true}).screenshot({path:`test-results/run-chat-${mode}-${width}.png`});
    }
    if (await page.locator('.robot-options').getAttribute('open') === null) await page.locator('.robot-options > summary').click();
    await page.getByRole('button',{name:'Reset episode',exact:true}).click();
    await expect(page.getByRole('tab', {name:'Conversation',exact:true})).toHaveCount(0);
    await expect(page.locator('.chat-transcript article')).toHaveCount(0);
    expect(((await (await request.get('/api/state')).json()) as LiveState).agent.run_memory!.recent_actions).toHaveLength(0);
  });
}

test('model readiness distinguishes weight loading from reused-model preparation', async ({ page, request }) => {
  const initial: LiveState = await (await request.get('/api/state')).json();
  let socket: WebSocketRoute;
  const resident: NonNullable<LiveState['local_navigation_model']> = {
    phase: 'ready', checkpoint: 'scripted-checkpoint', load_count: 1, process_id: 2468,
  };
  const local: NonNullable<LiveState['agent']['local_model']> = {
    run_id: 'scripted-run', checkpoint: 'scripted-checkpoint', phase: 'warming', requests_completed: 0,
    request_limit: 640, elapsed_s: 0, success: null,
  };
  await page.routeWebSocket('**/api/live', connection => {
    socket = connection;
    connection.send(JSON.stringify({...initial, local_navigation_model: resident,
      agent: {...initial.agent, active: true, session_id:'readiness-test', execution_mode:'luna_navigation', local_model: local}}));
  });
  const publish = (phase: typeof local.phase | null, residentPhase: typeof resident.phase) => socket.send(JSON.stringify({
    ...initial, local_navigation_model: {...resident, phase: residentPhase},
    agent: {...initial.agent, active: phase !== null, session_id:'readiness-test', execution_mode:'luna_navigation', local_model: phase === null ? null : {...local, phase}},
  }));
  await page.goto('/');
  const startup = page.getByRole('region', {name: 'Local model startup', exact: true});
  await expect(startup).toContainText('Reusing loaded model / preparing this run');
  await page.getByRole('tab', {name:'Settings',exact:true}).click();
  await expect(page.getByLabel('Local model process', {exact: true})).toContainText('Model loads: 1 / Process 2468');
  publish('loading', 'inferencing');
  await expect(startup).toContainText('Model already loaded / waiting for the previous prediction');
  publish('loading', 'loading');
  await expect(startup).toContainText('Loading model on GPU');
  publish(null, 'ready');
  await expect(startup).toHaveCount(0);
  await expect(page.getByRole('status', {name: 'Local model residency', exact: true})).toContainText('kept in GPU memory');
  await expect(page.getByLabel('Local model process', {exact: true})).toContainText('Model loads: 1 / Process 2468');
});

test('Luna supervises local motion in the selected scene and Stop retains loaded weights', async ({ page, request }) => {
  await request.post('/api/agent/config', { data: { endpoint: 'https://test.openai.azure.com', models: [
    { id: 'luna', label: 'Luna', deployment: 'scripted-supervisor', reasoning_efforts: ['low', 'medium', 'high'] },
  ] } });
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.setViewportSize({ width: 1440, height: 1100 });
  await page.goto('/');
  let loadCount: number | undefined;
  for (const challenge of ['park', 'recharge']) {
    await openChallengeMenu(page);
    await page.getByRole('combobox', { name: 'Predefined challenge', exact: true }).selectOption(challenge);
    await page.getByRole('button', { name: 'Load selected scenario', exact: true }).click();
    await expect.poll(async () => ((await (await request.get('/api/state')).json()) as LiveState).challenge?.id).toBe(challenge);
    const before: LiveState = await (await request.get('/api/state')).json();
    if (!await page.locator('.run-options').evaluate((element: HTMLDetailsElement) => element.open)) {
      await page.locator('.run-options > summary').click();
    }
    await page.getByRole('combobox', {name:'Navigation controller',exact:true}).selectOption('luna_navigation');
    const started = page.waitForResponse(response => response.url().endsWith('/api/agent/start') && response.request().method() === 'POST');
    await page.getByRole('button', { name: 'Start LLM control', exact: true }).click();
    const response = await started;
    expect(response.request().postDataJSON().execution_mode).toBe('luna_navigation');
    expect(response.ok()).toBe(true);
    await page.getByRole('tab', {name:'Settings',exact:true}).click();
    await expect(page.getByRole('button', { name: 'Unload local model', exact: true })).toBeDisabled();
    await expect.poll(async () => ((await (await request.get('/api/state')).json()) as LiveState).agent.local_model?.requests_completed,
      { timeout: 20000 }).toBe(2);
    const moving: LiveState = await (await request.get('/api/state')).json();
    expect(moving.run_id).toBe(before.run_id);
    expect(moving.challenge?.id).toBe(challenge);
    expect(moving.agent.goal).toBe(before.challenge!.goal);
    expect(moving.agent.model_id).toBe('luna');
    expect(moving.agent.execution_mode).toBe('luna_navigation');
    expect(moving.snapshot.simulated_time_s).toBeGreaterThan(1);
    expect(moving.agent.input_tokens).toBeGreaterThan(0);
    loadCount ??= moving.local_navigation_model!.load_count;
    expect(moving.local_navigation_model!.load_count).toBe(loadCount);
    const feed = await (await request.get('/api/agent/trace')).json();
    expect(feed.events.some((entry: { title: string }) => entry.title === 'Luna navigation decision')).toBe(true);
    expect(feed.events.some((entry: { title: string }) => entry.title === 'SmolVLA supervised motion')).toBe(true);
    for (const width of [1440, 390, 320]) {
      await page.setViewportSize({ width, height: 1000 });
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
      await expect.poll(() => pixels(page)).toBeGreaterThan(20);
      await page.getByRole('region', { name: 'Local model startup', exact: true }).screenshot({ path: `test-results/luna-smolvla-${challenge}-${width}.png` });
    }
    await page.getByRole('button', { name: 'Stop', exact: true }).click();
    await expect.poll(async () => ((await (await request.get('/api/state')).json()) as LiveState).agent.active).toBe(false);
    const stopped: LiveState = await (await request.get('/api/state')).json();
    expect(stopped.stopped).toBe(true);
    expect(stopped.navigation?.remaining_s).toBe(0);
    expect(stopped.local_navigation_model?.phase).toBe('ready');
    await expect(page.getByRole('button', { name: 'Start LLM control', exact: true })).toBeEnabled();
  }
  await page.getByRole('button', { name: 'Unload local model', exact: true }).click();
  await expect(page.getByRole('status', { name: 'Local model residency', exact: true })).toContainText('not loaded');
  expect(errors).toEqual([]);
});

test('observatory keeps scripted inputs decisions drafts and safety controls accessible', async ({page, request}) => {
  await request.post('/api/challenges/load', {data:{challenge_id:'kitchen_bathroom'}});
  const initial: LiveState = await (await request.get('/api/state')).json();
  const session = 'scripted-observatory';
  const reason = 'The doorway is visible ahead; inspect its threshold before selecting a route.';
  const events = [
    {id:1,turn:1,timestamp:1789250000,title:'Paired robot observation',kind:'feedback',image_url:initial.camera.url,
      payload:{observation:initial.observation,history_turns:[],images_in_request:1,tool_result_call_ids:[]}},
    {id:2,turn:1,timestamp:1789250001,title:'Luna continuous decision',kind:'response',image_url:null,
      payload:{text:'',status:'completed',latency_s:1.2,input_tokens:100,output_tokens:40,refusals:[],calls:[
        {call_id:'decision-1',name:'guide_continuous',arguments:JSON.stringify({action:'look',reason})},
        {call_id:'malformed',name:'guide_continuous',arguments:'{invalid'},
      ]}},
    {id:3,turn:1,timestamp:1789250002,title:'Head movement result',kind:'result',image_url:null,
      payload:{tool:'set_head',call_id:'decision-1',result:{status:'ok',actual_duration_s:.5,message:'Head pose reached.'}}},
    {id:4,turn:1,timestamp:1789250003,title:'Objective renewal',kind:'policy',image_url:null,
      payload:{action:'explore',status:'accepted',reason:'Fresh corridor confirmed.',operation_id:'scripted-objective'}},
    {id:5,turn:1,timestamp:1789250004,title:'Head command',kind:'tool',image_url:null,
      payload:{tool:'set_head',call_id:'decision-1',arguments:{yaw_rad:.4,pitch_rad:.5}}},
    {id:6,turn:1,timestamp:1789250005,title:'Session instructions',kind:'session',image_url:null,
      payload:{goal:'Find the kitchen.',instructions:'Scripted controller instructions.',tools:[{name:'set_head'}]}},
    {id:7,turn:1,timestamp:1789250006,title:'Motion rejected',kind:'result',image_url:null,
      payload:{tool:'drive_base',call_id:'blocked-command',result:{status:'rejected',error:'SPATIAL_STALE',message:'Sensor updates unavailable; motion stopped.'}}},
  ];
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.routeWebSocket('**/api/live', socket => socket.send(JSON.stringify({...initial,
    agent:{...initial.agent,active:true,execution_mode:'luna_continuous',session_id:session,phase:'thinking',trace_revision:events.length,turns:1,max_turns:80,goal:'Find the kitchen.'}})));
  await page.route('**/api/agent/trace?*', route => route.fulfill({json:{session_id:session,revision:events.length,first_id:1,capacity:200,events}}));
  await page.setViewportSize({width:1440,height:1000});
  await page.goto('/');
  const trace = page.getByRole('tab',{name:'Trace',exact:true});
  await expect(trace).toHaveAttribute('aria-selected','true');
  await expect(page.locator('.exchange-entry')).toHaveCount(events.length);
  await expect(page.locator('.exchange-disclosure[open]')).toHaveCount(0);
  await expect(page.locator('.exchange-preview.bad')).toContainText('SPATIAL_STALE');
  await expect(page.locator('.exchange-response .exchange-preview')).toContainText(reason);
  await expect(page.locator('.decision-summary')).toBeHidden();
  await page.locator('.exchange-response .exchange-disclosure > summary').focus();
  await page.keyboard.press('Enter');
  await expect(page.getByRole('checkbox',{name:'Follow latest',exact:true})).not.toBeChecked();
  await expect(page.locator('.decision-summary')).toContainText(reason);
  await expect(page.locator('.decision-summary')).toHaveCount(1);
  await expect(page.getByText('Reported reason',{exact:true})).toBeVisible();
  await page.getByRole('button',{name:'Inputs',exact:true}).click();
  await expect(page.locator('.exchange-entry')).toHaveCount(1);
  await expect(page.getByAltText(/^LLM input camera/)).toBeHidden();
  await page.locator('.exchange-disclosure > summary').click();
  await expect(page.getByAltText(/^LLM input camera/)).toHaveJSProperty('naturalWidth',640);
  await page.getByRole('button',{name:'LLM',exact:true}).click();
  await page.locator('.exchange-disclosure > summary').click();
  await expect(page.locator('.decision-summary')).toContainText(reason);
  await page.getByRole('button',{name:'Policy',exact:true}).click();
  await page.locator('.exchange-disclosure > summary').click();
  await page.getByText('Policy payload',{exact:true}).click();
  await expect(page.locator('.exchange-entry .exchange-payload pre')).toContainText('scripted-objective');
  await page.getByRole('button',{name:'Tools',exact:true}).click();
  await page.locator('.exchange-tool .exchange-disclosure > summary').click();
  await expect(page.locator('.exchange-arguments')).toContainText('yaw_rad');
  await page.getByRole('button',{name:'Session',exact:true}).click();
  await page.locator('.exchange-disclosure > summary').click();
  await page.getByText('Controller instructions',{exact:true}).click();
  await expect(page.locator('.exchange-entry .exchange-payload[open] pre')).toHaveText('Scripted controller instructions.');
  await page.context().grantPermissions(['clipboard-read','clipboard-write']);
  await page.getByRole('button',{name:'Copy exchange feed',exact:true}).click();
  await expect(page.getByRole('status').filter({hasText:'Copied 7 exchanges'})).toBeVisible();
  const copied = JSON.parse(await page.evaluate(() => navigator.clipboard.readText()));
  expect(copied.events).toEqual(events);
  await page.getByRole('button',{name:'All',exact:true}).click();
  await trace.focus();
  await page.keyboard.press('ArrowRight');
  await expect(page.getByRole('tab',{name:'Settings',exact:true})).toBeFocused();
  await expect(page.getByRole('combobox',{name:'Navigation controller',exact:true})).toBeDisabled();
  await page.locator('.run-options > summary').click();
  await expect(page.getByRole('combobox',{name:'Navigation controller',exact:true})).toBeHidden();
  await page.getByRole('tab',{name:'Settings',exact:true}).focus();
  await page.keyboard.press('Home');
  await expect(page.getByRole('tab',{name:'Conversation',exact:true})).toBeFocused();
  await page.getByRole('textbox',{name:'New run instruction',exact:true}).fill('Inspect the room on the left.');
  await trace.click();
  await page.getByRole('tab',{name:'Conversation',exact:true}).click();
  await expect(page.getByRole('textbox',{name:'New run instruction',exact:true})).toHaveValue('Inspect the room on the left.');
  await trace.click();
  for (const width of [1440, 1024, 390, 320]) {
    await page.setViewportSize({width,height:1000});
    await page.evaluate(()=>scrollTo(0,0));
    await expect.poll(()=>pixels(page)).toBeGreaterThan(20);
    expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
    if (width >= 1024) {
      const scene = (await page.locator('.spectator').boundingBox())!;
      const goal = (await page.getByRole('textbox',{name:'Robot goal',exact:true}).boundingBox())!;
      expect(goal.x).toBeGreaterThan(scene.x+scene.width);
      expect(goal.y).toBeLessThan(450);
    }
    await page.screenshot({path:`test-results/observatory-${width}.png`});
    await page.getByRole('link',{name:'Luna',exact:true}).click();
    await expect(trace).toBeInViewport();
    await page.locator('.mission-instructions > summary').click();
    await page.locator('.interaction-column').screenshot({path:`../.runtime/compact-cockpit-v1/activity-${width}.png`});
    await page.locator('.mission-instructions > summary').click();
    await page.locator('#controls').scrollIntoViewIfNeeded();
    await expect(page.getByRole('button',{name:'Stop',exact:true})).toBeInViewport();
    expect(await page.getByRole('button',{name:'Stop',exact:true}).evaluate(button=>{
      const rect=button.getBoundingClientRect();return button.contains(document.elementFromPoint(rect.x+rect.width/2,rect.y+rect.height/2));
    })).toBe(true);
  }
  expect(errors).toEqual([]);
});

test('setup comes first and controls follow the run lifecycle', async ({page, request}) => {
  const initial: LiveState = await (await request.get('/api/state')).json();
  let live = initial;
  let channel: WebSocketRoute;
  await page.routeWebSocket('**/api/live', socket => { channel = socket; socket.send(JSON.stringify(live)); });
  await page.route('**/api/agent/config', async route => {
    const settings = route.request().postDataJSON();
    live = {...live, agent: {...live.agent, configuration: {...live.agent.configuration, ...settings,
      models: settings.models.map((model: object) => ({...model, configured: true}))}}};
    channel.send(JSON.stringify(live));
    await route.fulfill({json: live.agent.configuration});
  });
  await page.route('**/api/agent/trace?*', route => route.fulfill({json:{session_id:live.agent.session_id,revision:0,first_id:0,capacity:200,events:[]}}));
  await page.setViewportSize({width:1440,height:1000});
  await page.goto('/');
  const scenario = page.getByRole('region', {name:'Predefined challenges',exact:true});
  await expect(scenario).toBeHidden();
  await expect(page.getByRole('button',{name:'Load scenario',exact:true})).toBeVisible();
  await expect(page.getByRole('textbox',{name:'Foundry endpoint',exact:true})).toBeVisible();
  await expect(page.getByRole('button',{name:'Start LLM control',exact:true})).toBeHidden();
  await expect(page.getByRole('tablist',{name:'Robot inspector',exact:true})).toHaveCount(0);
  await expect(page.getByRole('button',{name:'Drive forward',exact:true})).toBeHidden();
  await expect(page.locator('.challenge-goal')).toBeHidden();
  await page.getByRole('textbox',{name:'Foundry endpoint',exact:true}).fill('https://example.services.ai.azure.com');
  await page.getByRole('textbox',{name:'Luna deployment',exact:true}).fill('scripted-luna');
  await page.getByRole('button',{name:'Apply Luna connection',exact:true}).click();
  await expect(page.getByRole('textbox',{name:'Foundry endpoint',exact:true})).toBeHidden();
  const goal = page.getByRole('textbox',{name:'Robot goal',exact:true});
  await goal.fill('Inspect the doorway.');
  await page.locator('.run-options > summary').click();
  await expect(page.getByRole('combobox',{name:'Navigation controller',exact:true})).toBeVisible();
  await expect(page.getByRole('spinbutton',{name:'Feedback interval (s)',exact:true})).toHaveCount(0);
  await expect(page.getByRole('button',{name:'Unload local model',exact:true})).toHaveCount(0);
  await page.locator('.run-options > summary').click();
  await expect(goal).toHaveValue('Inspect the doorway.');
  let releaseLoad: () => void;
  const loading = new Promise<void>(resolve => {releaseLoad = resolve;});
  await page.route('**/api/challenges/load', async route => {
    await loading;
    await route.fulfill({status:503,json:{detail:'Scene temporarily unavailable'}});
  });
  await openChallengeMenu(page);
  await page.getByRole('combobox',{name:'Predefined challenge',exact:true}).selectOption('park');
  await page.getByRole('button',{name:'Load selected scenario',exact:true}).click();
  await expect(page.getByRole('button',{name:'Loading...',exact:true})).toBeDisabled();
  await expect(page.getByRole('button',{name:'Start LLM control',exact:true,includeHidden:true})).toBeDisabled();
  releaseLoad!();
  await expect(scenario.getByRole('alert')).toContainText('Scene temporarily unavailable');
  await expect(page.getByRole('button',{name:'Load selected scenario',exact:true})).toBeEnabled();
  await page.getByRole('button',{name:'Close challenge menu',exact:true}).click();
  await expect(goal).toHaveValue('Inspect the doorway.');
  for (const width of [1440,390,320]) {
    await page.setViewportSize({width,height:1000});
    await page.evaluate(() => scrollTo(0,0));
    await openChallengeMenu(page);
    const loadingBounds = (await page.getByRole('dialog',{name:'Load scenario',exact:true}).boundingBox())!;
    expect(loadingBounds.y).toBeGreaterThanOrEqual(0);
    expect(loadingBounds.y + loadingBounds.height).toBeLessThanOrEqual(1000);
    await expect(page.getByRole('button',{name:'Load selected scenario',exact:true})).toBeInViewport();
    await page.screenshot({path:`test-results/setup-journey-${width}.png`});
    await page.getByRole('button',{name:'Close challenge menu',exact:true}).click();
    const sceneBounds = (await page.locator('.spectator').boundingBox())!;
    if (width < 850) expect((await goal.boundingBox())!.y).toBeLessThan(sceneBounds.y);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  }
  live = {...live, agent:{...live.agent, active:true, session_id:'journey-test',execution_mode:'luna_continuous',phase:'thinking',goal:'Inspect the doorway.'}};
  channel!.send(JSON.stringify(live));
  await expect(page.getByRole('tab',{name:'Trace',exact:true})).toHaveAttribute('aria-selected','true');
  await expect(page.getByRole('button',{name:'Take manual control',exact:true})).toBeVisible();
  await page.getByRole('tab',{name:'Conversation',exact:true}).click();
  await page.getByRole('textbox',{name:'New run instruction',exact:true}).fill('Look left.');
  await page.getByRole('tab',{name:'Trace',exact:true}).click();
  await page.getByRole('tab',{name:'Conversation',exact:true}).click();
  await expect(page.getByRole('textbox',{name:'New run instruction',exact:true})).toHaveValue('Look left.');
  live = {...live,stopped:true,agent:{...live.agent,active:false,phase:'completed'}};
  channel!.send(JSON.stringify(live));
  await expect(page.getByRole('textbox',{name:'New run instruction',exact:true})).toBeHidden();
  await page.locator('.manual-disclosure > summary').click();
  await expect(page.locator('.manual-disclosure')).toHaveAttribute('open','');
  await expect(page.getByRole('button',{name:'Drive forward',exact:true})).toBeDisabled();
  await expect(page.getByRole('button',{name:'Stop',exact:true})).toBeInViewport();
});

test('manual controls and challenge reset remain usable without Luna', async ({ page, request }) => {
  await page.goto('/');
  await page.locator('.manual-disclosure > summary').click();
  if (await page.locator('.robot-options').getAttribute('open') === null) await page.locator('.robot-options > summary').click();
  await page.getByRole('button', { name: 'Enable manual control', exact: true }).isDisabled();
  const before: LiveState = await (await request.get('/api/state')).json();
  const done = page.waitForResponse(response => response.url().endsWith('/api/command') && response.request().postDataJSON()?.tool === 'drive_base');
  await page.getByRole('button', { name: 'Drive backward', exact: true }).click();
  expect((await (await done).json()).status).toBe('ok');
  await expect(page.locator('.event-list')).toContainText('drive_base');
  await page.getByRole('button', { name: 'Stop', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Drive backward', exact: true })).toBeDisabled();
  if (await page.locator('.robot-options').getAttribute('open') === null) await page.locator('.robot-options > summary').click();
  await page.getByRole('button', { name: 'Reset episode', exact: true }).click();
  await expect.poll(async () => ((await (await request.get('/api/state')).json()) as LiveState).run_id).not.toBe(before.run_id);
  await expect(page.getByRole('button', { name: 'Drive backward', exact: true })).toBeEnabled();
});