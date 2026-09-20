import {test, expect} from '@playwright/test';
import {mkdir, mkdtemp, readFile, realpath, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import type {LiveState, NavigationDiagnostics} from '../src/types';

test.beforeEach(async ({request}) => {
  await request.post('/api/test/preferences/reset');
  await request.post('/api/challenges/load', {data:{challenge_id:'park',reuse_saved_map:false}});
  await request.post('/api/agent/config', {data:{endpoint:'https://test.openai.azure.com',models:[
    {id:'luna',label:'Scripted Luna',deployment:'scripted',reasoning_efforts:['low','medium','high']}]}});
});

test('spatial memory drawer creates checkpoints and independent profiles without motion', async ({page,request}) => {
  const initial:LiveState=await(await request.get('/api/state')).json();
  const errors:string[]=[];page.on('pageerror',error=>errors.push(error.message));
  await page.goto('/');
  await page.getByRole('button',{name:'Open configuration',exact:true}).click();
  const panel=page.getByRole('region',{name:'Spatial memory',exact:true});
  const profile=panel.getByRole('combobox',{name:'Knowledge profile',exact:true});
  const name=panel.getByRole('textbox',{name:'Knowledge name',exact:true});
  await expect(profile).toBeEnabled();
  await name.fill('Browser fixture knowledge');
  await panel.getByRole('button',{name:'Start fresh knowledge',exact:true}).click();
  await expect(panel.getByRole('status')).toHaveText('Browser fixture knowledge');
  const parent=await(await request.get('/api/memory')).json();
  const identity={run_id:initial.run_id,episode_epoch:initial.episode_epoch};
  expect((await request.post('/api/home',{data:{...identity,action:'start_mapping'}})).ok()).toBe(true);
  expect((await request.post('/api/home',{data:{...identity,action:'save_map',name:'Browser fixture map'}})).ok()).toBe(true);
  for(const label of ['Fixture kitchen','Fixture hall']) {
    const result=await request.post('/api/memory',{data:{...identity,context_id:parent.scope.context_id,
      action:'record_observation',kind:'room',label,description:'Scripted semantic fixture on the current camera; no autonomous room recognition claim.'}});
    expect(result.ok(),await result.text()).toBe(true);
  }
  await name.fill('Two fixture rooms');
  await panel.getByRole('button',{name:'Save checkpoint',exact:true}).click();
  await expect(panel.getByRole('combobox',{name:'Knowledge checkpoint',exact:true})).toBeVisible();
  await expect(panel).toContainText('2 rooms / 0 objects / 0 pathways');
  await panel.getByText('Remembered rooms and objects',{exact:true}).click();
  await expect(panel.getByText('Fixture kitchen',{exact:true})).toBeVisible();
  for(const width of [1440,390,320]) {
    await page.setViewportSize({width,height:1000});
    await panel.scrollIntoViewIfNeeded();
    expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
    const dialog=page.getByRole('dialog',{name:'Robot configuration',exact:true});
    expect(await dialog.evaluate(element=>element.scrollWidth<=element.clientWidth)).toBe(true);
    await expect(dialog.getByRole('button',{name:'Stop',exact:true})).toBeInViewport();
    await expect(dialog.getByRole('button',{name:'Reset episode',exact:true})).toBeInViewport();
    await page.screenshot({path:`../.runtime/spatial-memory-v1/memory-${width}.png`});
  }
  await name.fill('Independent browser fork');
  await panel.getByRole('button',{name:'Fork checkpoint',exact:true}).click();
  await expect(panel.getByRole('status')).toHaveText('Independent browser fork');
  const fork=await(await request.get('/api/memory')).json();
  expect(fork.scope.profile_id).not.toBe(parent.scope.profile_id);
  expect(fork.scope.map_id).not.toBe(parent.scope.map_id);
  expect((await request.post('/api/memory',{data:{...identity,context_id:fork.scope.context_id,
    action:'record_observation',kind:'room',label:'Fixture extra view',description:'Unlocalized fork fixture; viewing pose only.'}})).ok()).toBe(true);
  await expect(panel).toContainText('3 rooms / 0 objects / 0 pathways');
  await profile.selectOption(parent.scope.profile_id);
  await expect(panel.getByRole('status')).toHaveText('Browser fixture knowledge');
  await expect(panel).toContainText('2 rooms / 0 objects / 0 pathways');
  expect((await request.post('/api/memory',{data:{...identity,context_id:fork.scope.context_id,action:'fresh_profile',name:'Obsolete'}})).status()).toBe(409);
  await page.getByRole('button',{name:'Save configuration',exact:true}).click();
  await expect(page.getByRole('dialog',{name:'Robot configuration',exact:true})).toBeHidden();
  expect((await(await request.get('/api/state')).json()).snapshot).toEqual(initial.snapshot);
  await page.reload();await page.getByRole('button',{name:'Open configuration',exact:true}).click();
  await expect(profile).toHaveValue(parent.scope.profile_id);
  await expect(panel).toContainText('2 rooms / 0 objects / 0 pathways');
  const switched=await request.post('/api/challenges/load',{data:{challenge_id:'park',reuse_saved_map:true,environment_instance_id:'isolated-browser-flat'}});
  expect(switched.ok()).toBe(true);
  const isolated=await(await request.get('/api/memory')).json();
  expect(isolated.scope.environment_id).not.toBe(parent.scope.environment_id);
  expect(isolated.rooms).toEqual([]);
  const foreign=await request.post('/api/memory',{data:{run_id:isolated.scope.run_id,episode_epoch:isolated.scope.episode_epoch,
    context_id:isolated.scope.context_id,action:'select_profile',profile_id:parent.scope.profile_id}});
  expect(foreign.status()).toBe(409);
  expect(errors).toEqual([]);
});

test('configuration drawer saves then collapses and retains failures beside permanent reset', async ({page,request}) => {
  const initial:LiveState=await(await request.get('/api/state')).json();
  const errors:string[]=[];page.on('pageerror',error=>errors.push(error.message));
  const directory=await realpath(await mkdtemp(join(tmpdir(),'milo-configuration-')));
  await page.goto('/');
  const open=page.getByRole('button',{name:'Open configuration',exact:true});
  const reset=page.getByRole('button',{name:'Reset episode',exact:true});
  const drawer=page.getByRole('dialog',{name:'Robot configuration',exact:true});
  const save=drawer.getByRole('button',{name:'Save configuration',exact:true});
  await expect(drawer).toBeHidden();await expect(reset).toBeVisible();
  await expect(page.locator('.robot-options')).not.toHaveAttribute('open','');
  await page.getByRole('button',{name:'Minimize Luna console',exact:true}).click();
  await open.click();await expect(drawer).toBeVisible();
  const threshold=drawer.getByRole('spinbutton',{name:'Luna token threshold',exact:true});
  await threshold.fill('64000');
  await drawer.getByRole('textbox',{name:'Recording folder',exact:true}).fill(directory);
  for(const width of [1440,390,320]) {
    await page.setViewportSize({width,height:900});
    const layout=await drawer.evaluate(element=>{
      const bounds=element.getBoundingClientRect();
      const controls=[...element.querySelectorAll('.run-actions button, .robot-options > summary')]
        .filter(control=>control.getClientRects().length).map(control=>control.getBoundingClientRect());
      return {right:bounds.right,width:bounds.width,viewport:innerWidth,overflow:element.scrollWidth>element.clientWidth,
        fits:controls.every(control=>control.left>=bounds.left && control.right<=bounds.right),
        overlap:controls.some((control,index)=>controls.slice(index+1).some(other=>control.left<other.right && other.left<control.right && control.top<other.bottom && other.top<control.bottom))};
    });
    expect(Math.abs(layout.right-width)).toBeLessThan(1);expect(layout.width).toBeLessThanOrEqual(540);
    expect(layout.overflow).toBe(false);expect(layout.fits).toBe(true);expect(layout.overlap).toBe(false);
    await expect(drawer.getByRole('button',{name:'Stop',exact:true})).toBeInViewport();
    await expect(reset).toBeInViewport();await expect(save).toBeInViewport();
    await page.screenshot({path:`../.runtime/configuration-drawer-v1/drawer-${width}.png`});
  }
  await save.click();await expect(drawer).toBeHidden();await expect(open).toBeFocused();
  const saved=(await(await request.get('/api/preferences')).json()).preferences;
  expect(saved.max_model_tokens).toBe(64000);expect(saved.recording_directory).toBe(directory);
  await expect(page.getByRole('button',{name:'Restore Luna console',exact:true})).toBeVisible();
  await open.click();await expect(threshold).toHaveValue('64000');
  let reject=true;let release:()=>void=()=>{};
  await page.route('**/api/preferences',async route=>{
    if(route.request().method()!=='POST' || !reject) return route.continue();
    await new Promise<void>(resolve=>{release=resolve;});
    await route.fulfill({status:503,json:{detail:'Scripted save failure'}});
  });
  await threshold.fill('68000');await save.click();
  await expect(drawer.getByRole('button',{name:'Saving configuration',exact:true})).toBeDisabled();
  await expect(drawer.getByRole('button',{name:'Stop',exact:true})).toBeEnabled();
  await expect(drawer).toBeVisible();release();
  await expect(drawer.getByRole('alert')).toContainText('Configuration could not be saved');
  await expect(threshold).toHaveValue('68000');await expect(save).toBeEnabled();
  reject=false;await save.click();await expect(drawer).toBeHidden();
  expect((await(await request.get('/api/preferences')).json()).preferences.max_model_tokens).toBe(68000);
  await open.click();await page.keyboard.press('Escape');await expect(drawer).toBeHidden();
  await expect(open).toBeFocused();
  await page.reload();await expect(drawer).toBeHidden();await open.click();
  await expect(threshold).toHaveValue('68000');
  await drawer.getByRole('button',{name:'Close configuration',exact:true}).click();
  await expect(drawer).toBeHidden();
  expect((await(await request.get('/api/state')).json()).snapshot).toEqual(initial.snapshot);
  expect(errors).toEqual([]);
});

test('Qwen configuration saves its bounded local profile and reports validation details', async ({page,request}) => {
  await page.goto('/');
  await page.getByRole('button',{name:'Open configuration',exact:true}).click();
  const drawer=page.getByRole('dialog',{name:'Robot configuration',exact:true});
  await drawer.getByRole('combobox',{name:'Mission controller',exact:true}).selectOption('qwen');
  const saved=page.waitForResponse(response=>response.url().endsWith('/api/agent/config') && response.request().method()==='POST');
  await drawer.getByRole('button',{name:'Save configuration',exact:true}).click();
  const response=await saved;
  expect(response.ok(),await response.text()).toBe(true);
  const state:LiveState=await(await request.get('/api/state')).json();
  expect(state.agent.configuration.models.find(model=>model.id==='qwen')).toMatchObject({
    provider:'ollama',deployment:'qwen3-vl:4b-instruct-q4_K_M',context_window:16384,configured:true});
  await expect(drawer).toBeHidden();

  await page.getByRole('button',{name:'Open configuration',exact:true}).click();
  await page.route('**/api/agent/config',route=>route.fulfill({status:422,json:{detail:[{
    loc:['body','models',3,'context_window'],msg:'Input should be less than or equal to 49152',type:'less_than_equal'}]}}));
  await drawer.getByRole('textbox',{name:'Local model tag',exact:true}).fill('qwen-test-tag');
  await drawer.getByRole('button',{name:'Save configuration',exact:true}).click();
  await expect(drawer.getByRole('alert')).toContainText('models.3.context_window: Input should be less than or equal to 49152');
  await expect(drawer).toBeVisible();
});

test('hybrid defaults keep Qwen operational and give Luna task-only supervision', async ({page}) => {
  let payload:Record<string,unknown>|null=null;
  let baselinePayload:Record<string,unknown>|null=null;
  await page.route('**/api/agent/local-readiness?*',route=>route.fulfill({json:{ready:true,status:'installed',message:'Local vision model installed'}}));
  await page.route('**/api/mission/start',async route=>{
    payload=route.request().postDataJSON();
    await route.fulfill({status:409,json:{detail:'Payload captured without starting model inference'}});
  });
  await page.route('**/api/regression/start',async route=>{
    baselinePayload=route.request().postDataJSON();
    await route.fulfill({status:409,json:{detail:'Baseline payload captured without model inference'}});
  });
  await page.goto('/');
  await page.getByRole('button',{name:'Open configuration',exact:true}).click();
  const drawer=page.getByRole('dialog',{name:'Robot configuration',exact:true});
  await expect(drawer.getByRole('combobox',{name:'Mission controller',exact:true})).toHaveValue('hybrid');
  await drawer.getByRole('button',{name:'Save configuration',exact:true}).click();
  await expect(drawer).toBeHidden();
  await page.getByRole('button',{name:'Open configuration',exact:true}).click();
  await expect(drawer.getByLabel('Luna task supervision readiness')).toContainText('immutable task plan only');
  await drawer.getByRole('button',{name:'Close configuration',exact:true}).click();
  await page.getByRole('button',{name:'Start mission',exact:true}).click();
  await expect.poll(()=>payload).not.toBeNull();
  expect(payload).toMatchObject({model_id:'qwen',reasoning:'none',task_supervisor_model_id:'luna',
    task_supervisor_reasoning:'low',max_task_supervisor_requests:4,max_task_supervisor_tokens:100000});
  expect(payload).not.toHaveProperty('automatic_fallback');
  await expect(page.getByRole('alert')).toContainText('Payload captured without starting model inference');
  const panel=page.locator('.regression-panel');
  await panel.locator(':scope > summary').click();
  const supervisor=panel.getByRole('combobox',{name:'Baseline supervisor',exact:true});
  await expect(supervisor).toHaveValue('hybrid');
  for(const selection of ['hybrid','qwen','luna']) {
    await supervisor.selectOption(selection);
    baselinePayload=null;
    await panel.getByRole('button',{name:'Start baseline',exact:true}).click();
    await expect.poll(()=>baselinePayload).not.toBeNull();
    expect(baselinePayload).toMatchObject({model_id:selection==='hybrid'?'qwen':selection,
      task_supervisor_model_id:selection==='hybrid'?'luna':null});
    if(selection==='hybrid') expect(baselinePayload).toHaveProperty('reasoning','none');
    await expect(panel.getByRole('alert')).toContainText('Baseline payload captured');
  }
  await page.reload();
  await panel.locator(':scope > summary').click();
  await expect(supervisor).toHaveValue('hybrid');
});

test('regression includes four parking starts with distinct loadable scene previews', async ({page,request}) => {
  test.setTimeout(90000);
  const initial:LiveState=await(await request.get('/api/state')).json();
  expect(initial.regression?.suite_id).toBe('observable-navigation-v2');
  expect(initial.regression?.cases.slice(0,4).map(entry=>entry.challenge_id)).toEqual(['park','park_left','park_right','park_far']);
  await page.goto('/');
  const panel=page.locator('.regression-panel');
  await expect(panel.locator(':scope > summary')).toContainText('8 cases · 20 min mission budget');
  await panel.locator(':scope > summary').click();
  await expect(panel.locator('.regression-cases > li')).toHaveCount(8);
  await expect(panel.getByRole('button',{name:'Start baseline',exact:true})).toHaveAttribute('title',/96 primary model requests and 640,000 primary tokens/);
  for(const image of await panel.locator('.regression-cases > li:nth-child(-n+4) img').all()) {
    await expect.poll(()=>image.evaluate(element=>(element as HTMLImageElement).naturalWidth)).toBeGreaterThan(0);
  }
  for(const width of [1440,390]) {
    await page.setViewportSize({width,height:1000});
    expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
    await panel.screenshot({path:`../.runtime/parking-variations-v1/suite-${width}.png`});
  }
  for(const identifier of ['park_left','park_right','park_far']) {
    await page.getByRole('button',{name:'Choose challenge',exact:true}).click();
    const dialog=page.getByRole('dialog',{name:'Load challenge',exact:true});
    await dialog.getByRole('combobox',{name:'Predefined challenge',exact:true}).selectOption(identifier);
    await dialog.getByRole('combobox',{name:'Map source',exact:true}).selectOption('none');
    await expect(dialog.locator('.scenario-thumbnail img')).toHaveAttribute('src',`/scenario-previews/${identifier}.webp`);
    await expect(dialog.locator('.scenario-thumbnail')).toHaveAttribute('data-state','ready');
    const loaded=page.waitForResponse(response=>response.url().endsWith('/api/challenges/load')&&response.request().method()==='POST');
    await dialog.getByRole('button',{name:'Load challenge',exact:true}).click();
    const response=await loaded;
    expect(response.ok()).toBe(true);
    const loadedState:LiveState=await response.json();
    expect(loadedState.challenge?.id).toBe(identifier);
    await expect(dialog).toBeHidden();
    await expect(page.getByRole('heading',{name:loadedState.challenge!.title,exact:true})).toBeVisible();
  }
  await expect.poll(async()=>(await(await request.get('/api/preferences')).json()).preferences.challenge_selection?.challenge_id).toBe('park_far');
  await page.reload();
  await page.getByRole('button',{name:'Choose challenge',exact:true}).click();
  await expect(page.getByRole('combobox',{name:'Predefined challenge',exact:true})).toHaveValue('park_far');
  const current:LiveState=await(await request.get('/api/state')).json();
  expect(current.agent.active).toBe(false);
  expect(current.regression?.active).toBe(false);
});

test('regression cases expose their recorded paths without changing robot state', async ({page,request}) => {
  const initial:LiveState=await(await request.get('/api/state')).json();
  const live={...initial,regression:{suite_id:'observable-navigation-v1',sequence_id:'paths',active:false,phase:'completed' as const,
    cases:[{id:'parking',challenge_id:'park' as const,title:'Park in the green bay',budget_s:120,status:'failed' as const,
      elapsed_s:42,trajectory_url:'/api/regression/cases/parking/trajectory',evaluation:{schema_version:1,evaluation_only:true,
        kind:'arrival',metric:'Target approach',progress_pct:75,physics_complete:false,completed_objectives:0,total_objectives:1,
        initial_gap_m:1.4,remaining_m:.35,remaining_pct:25,center_distance_m:.6,detail:'Move fully inside the zone',
        target_xy_m:[1.6,.5],target_bounds_m:[1.3,.1,1.9,.9],checks:[{label:'Full footprint inside',complete:false},{label:'Stopped',complete:true}]}}]}};
  await page.routeWebSocket('**/api/live',socket=>socket.send(JSON.stringify(live)));
  await page.route('**/api/regression/cases/parking/trajectory',route=>route.fulfill({json:{coordinate_frame:'recorded_world_xy_m',
    sample_count:4,distance_m:1.2,bounds_m:[0,0,1,.5],downsampled:false,contact_markers_truncated:false,
    points:[{x:0,y:0,wall_s:0,segment:0},{x:.4,y:.1,wall_s:1,segment:0},{x:.8,y:.5,wall_s:2,segment:0},{x:1,y:.5,wall_s:3,segment:0}],contacts:[]}}));
  await page.goto('/');
  const panel=page.locator('.regression-panel');
  if(!(await panel.getAttribute('open')))await panel.locator(':scope > summary').click();
  const item=panel.locator('.regression-cases > li').first();
  await item.getByText('Path',{exact:true}).click();
  await expect(item.getByRole('img',{name:'Recorded path for Park in the green bay'})).toBeVisible();
  await expect(item).toContainText('1.20 m / 4 samples');
  await expect(item.getByRole('progressbar')).toHaveAttribute('value','75');
  await expect(item).toContainText('75.0%');
  await expect(item).toContainText('0.35 m remaining (25.0% of initial gap)');
  await expect(item.locator('.regression-case-status')).toHaveText('failed');
  await expect(item).toContainText('Objectives 0/1');
  await expect(item.locator('.regression-path-target')).toHaveCount(1);
  for(const width of [1440,390,320]){
    await page.setViewportSize({width,height:900});
    expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
    await expect(item.locator('.regression-path-line')).toHaveCount(1);
  }
  const current:LiveState=await(await request.get('/api/state')).json();
  expect(current.snapshot).toEqual(initial.snapshot);
});

test('recording settings save local folders and preserve failed drafts without motion', async ({page,request}) => {
  const directory=await realpath(await mkdtemp(join(tmpdir(),'milo-recording-settings-')));
  const target=join(directory,'My test runs');
  const blocked=join(directory,'not-a-folder');await writeFile(blocked,'preserve');
  const initial:LiveState=await(await request.get('/api/state')).json();
  const errors:string[]=[];page.on('pageerror',error=>errors.push(error.message));
  await page.goto('/');await page.getByRole('tab',{name:'Settings',exact:true}).click();
  const panel=page.getByRole('region',{name:'Test run recording',exact:true});
  const folder=panel.getByRole('textbox',{name:'Recording folder',exact:true});
  const enabled=panel.getByRole('switch',{name:'Record test runs',exact:true});
  const save=panel.getByRole('button',{name:'Save recording settings',exact:true});
  await expect(enabled).toBeChecked();await expect(save).toBeDisabled();
  await folder.fill(target);await enabled.uncheck();await save.click();
  await expect(panel.getByRole('status',{name:'Recording status',exact:true})).toHaveText('Recording off');
  await expect(folder).toHaveValue(target);
  await page.reload();await page.getByRole('button',{name:'Open configuration',exact:true}).click();
  await expect(folder).toHaveValue(target);await expect(enabled).not.toBeChecked();
  await folder.fill(blocked);await save.click();await expect(panel.getByRole('alert')).toContainText('writable local folder');
  await expect(folder).toHaveValue(blocked);
  expect((await(await request.get('/api/recording')).json()).directory).toBe(target);
  expect(await readFile(blocked,'utf8')).toBe('preserve');
  await folder.fill(target);await enabled.check();await save.click();
  await expect(panel.getByRole('status',{name:'Recording status',exact:true})).toHaveText('Ready for next run');
  for(const width of [1440,390,320]) {
    await page.setViewportSize({width,height:1000});await panel.scrollIntoViewIfNeeded();
    expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
    expect(await save.evaluate(element=>element.scrollWidth<=element.clientWidth)).toBe(true);
    await panel.screenshot({path:`../.runtime/recording-settings-v1/settings-${width}.png`});
  }
  expect(errors).toEqual([]);
  const after:LiveState=await(await request.get('/api/state')).json();expect(after.snapshot).toEqual(initial.snapshot);
  expect(after.agent.active).toBe(false);
});

test('powered-on chat starts custom missions and redirects only the active session', async ({page,request}) => {
  await request.post('/api/preferences',{data:{max_model_requests:4,max_model_tokens:22000,exploration_budget:75,turns:9,reasoning:'low'}});
  const initial:LiveState=await(await request.get('/api/state')).json();
  let live=initial;
  let publish:(value:LiveState)=>void=()=>{};
  let disconnect:()=>void=()=>{};
  await page.routeWebSocket('**/api/live',socket=>{publish=value=>{live=value;socket.send(JSON.stringify(value));};disconnect=()=>socket.close();publish(initial);});
  const starts:Record<string,unknown>[]=[],redirects:Record<string,unknown>[]=[];
  let release:()=>void=()=>{};
  let reject=true;
  await page.route('**/api/mission/start',async route=>{
    const body=route.request().postDataJSON();starts.push(body);
    if(reject) {
      await new Promise<void>(resolve=>{release=resolve;});
      await route.fulfill({status:409,json:{detail:'Instruction cancelled by Stop'}});
    } else {
      publish({...live,busy:true,agent:{...live.agent,active:true,session_id:'chat-start',goal:body.goal,execution_mode:'luna_continuous',unified_mission:true}});
      await route.fulfill({json:{}});
    }
  });
  await page.route('**/api/agent/instruction',async route=>{
    const body=route.request().postDataJSON();redirects.push(body);
    publish({...live,agent:{...live.agent,session_id:'chat-redirect',goal:body.message,run_messages:[
      {id:'user-redirect',role:'user',text:body.message,status:'applied',timestamp:1},
      {id:'controller-redirect',role:'assistant',text:'Instruction applied. Replanning from the current position.',status:'applied',timestamp:1}]}});
    await route.fulfill({json:{}});
  });
  const errors:string[]=[];page.on('pageerror',error=>errors.push(error.message));
  await page.goto('/');
  const composer=page.getByRole('textbox',{name:'New run instruction',exact:true});
  const send=page.getByRole('button',{name:'Send instruction',exact:true});
  await expect(composer).toBeVisible();await expect(composer).toBeEnabled();await expect(send).toBeDisabled();
  await composer.fill('  Inspect the doorway and wait for my next instruction.  ');
  for(const width of [1440,390,320]) {
    await page.setViewportSize({width,height:1000});
    await composer.scrollIntoViewIfNeeded();
    expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
    await expect(send).toBeEnabled();
    await page.locator('.agent-section').screenshot({path:`../.runtime/powered-chat-v1/chat-${width}.png`});
  }
  publish({...live,power:{...live.power!,on:false,mode:'off'}});
  await expect(composer).toBeDisabled();await expect(send).toBeDisabled();
  await expect(page.getByLabel('Instruction availability')).toHaveText('Robot is off');
  publish({...live,power:{...live.power!,on:true,mode:'idle'}});
  await expect(send).toBeEnabled();expect(starts).toEqual([]);
  await send.click();await expect.poll(()=>starts.length).toBe(1);
  await expect(send).toBeDisabled();await expect(page.getByRole('button',{name:'Start mission',exact:true})).toBeDisabled();
  release();
  await expect(page.getByRole('alert').filter({hasText:'Instruction cancelled by Stop'})).toBeVisible();
  await expect(composer).toHaveValue('  Inspect the doorway and wait for my next instruction.  ');
  reject=false;await send.click();await expect.poll(()=>starts.length).toBe(2);
  expect(starts[1]).toMatchObject({run_id:initial.run_id,episode_epoch:initial.episode_epoch,
    goal:'Inspect the doorway and wait for my next instruction.',unified_mission:true,mission_local_only:false,
    execution_mode:'luna_continuous',navigation_backend:'builtin',images_per_request:2,map_context:true,
    max_model_requests:4,max_model_tokens:22000,mission_budget_s:75,max_turns:9,reasoning:'low'});
  await expect(composer).toHaveValue('');
  await expect(page.getByRole('log',{name:'Run conversation',exact:true})).toContainText(String(starts[1].goal));
  await composer.fill('Return to the starting position.');await send.click();
  await expect.poll(()=>redirects.length).toBe(1);
  expect(redirects[0]).toEqual({run_id:initial.run_id,episode_epoch:initial.episode_epoch,session_id:'chat-start',message:'Return to the starting position.'});
  await expect(page.getByRole('log',{name:'Run conversation',exact:true})).toContainText('Instruction applied.');
  await expect(composer).toHaveValue('');
  publish({...live,busy:false,agent:{...live.agent,active:false}});
  await expect(composer).toBeEnabled();await composer.fill('Keep this unsent draft.');
  disconnect();await expect(composer).toBeDisabled();await expect(send).toBeDisabled();
  await expect(composer).toHaveValue('Keep this unsent draft.');
  expect(starts).toHaveLength(2);expect(redirects).toHaveLength(1);expect(errors).toEqual([]);
  const after:LiveState=await(await request.get('/api/state')).json();
  expect(after.snapshot).toEqual(initial.snapshot);expect(after.agent.active).toBe(false);
});

test('idle chat honors readiness and settings while stop words never start inference', async ({page}) => {
  const commands:string[]=[];
  await page.route('**/api/stop',route=>{commands.push('stop');return route.fulfill({json:{}});});
  await page.route('**/api/mission/start',route=>{commands.push('start');return route.fulfill({json:{}});});
  await page.goto('/');
  const composer=page.getByRole('textbox',{name:'New run instruction',exact:true});
  const send=page.getByRole('button',{name:'Send instruction',exact:true});
  await composer.fill('Inspect the doorway.');
  await expect(send).toBeEnabled();
  await page.getByRole('tab',{name:'Settings',exact:true}).click();
  const budget=page.getByRole('spinbutton',{name:'Luna token threshold',exact:true});
  await budget.fill('0');await expect(send).toBeDisabled();await budget.fill('12000');
  await page.locator('.mission-diagnostics > summary').click();
  const profile=page.getByRole('combobox',{name:'Diagnostic execution profile',exact:true});
  await profile.selectOption('local');await expect(send).toBeDisabled();
  await page.getByRole('button',{name:'Close configuration',exact:true}).click();
  await expect(page.getByLabel('Instruction availability')).toContainText('Luna mission profile');
  await composer.fill('STOP!');await expect(send).toBeEnabled();await send.click();
  await expect.poll(()=>commands).toEqual(['stop']);await expect(composer).toHaveValue('');
  await page.getByRole('tab',{name:'Settings',exact:true}).click();await profile.selectOption('unified');
  await page.locator('.agent-connection > summary').click();
  await page.getByRole('textbox',{name:'Luna deployment',exact:true}).fill('unapplied');
  await page.getByRole('button',{name:'Close configuration',exact:true}).click();
  await composer.fill('Inspect the doorway.');await expect(send).toBeDisabled();
  await expect(page.getByLabel('Instruction availability')).toHaveText('Connection changes not applied');
  await page.route('**/api/mission/capabilities',route=>route.fulfill({status:404,json:{detail:'Old server'}}));
  await page.reload();await composer.fill('Inspect the doorway.');await expect(send).toBeDisabled();
  await composer.fill('pause');await expect(send).toBeEnabled();await send.click();
  await expect.poll(()=>commands).toEqual(['stop','stop']);
});

test('viewport-first console keeps setup compact and token budget adjustable without motion', async ({page,request}) => {
  const initial:LiveState=await(await request.get('/api/state')).json();
  const errors:string[]=[];
  page.on('pageerror',error=>errors.push(error.message));
  const sent:Record<string,unknown>[]=[];
  await page.route('**/api/mission/start',route=>{sent.push(route.request().postDataJSON());return route.fulfill({status:409,json:{detail:'Scripted contract capture'}});});
  await page.setViewportSize({width:1440,height:1000});
  await page.goto('/');
  const consolePanel=page.getByRole('region',{name:'LLM control',exact:true});
  await expect(page.getByRole('button',{name:'Start mission',exact:true})).toBeEnabled();
  await expect(page.getByRole('dialog')).toHaveCount(0);
  await expect(page.getByRole('button',{name:'Choose challenge',exact:true})).toBeVisible();
  await expect(page.getByRole('textbox',{name:'Robot goal',exact:true})).toBeHidden();
  await expect(page.getByRole('tabpanel',{name:'Conversation',exact:true})).toBeVisible();
  await expect(page.getByRole('region',{name:'Motion diagnostics',exact:true})).toBeHidden();
  for(const width of [1440,1024,390,320]) {
    await page.setViewportSize({width,height:1000});
    await page.evaluate(()=>scrollTo(0,0));
    const layout=await page.evaluate(()=>{
      const world=document.querySelector('.world-panel')!.getBoundingClientRect();
      const consolePanel=document.querySelector('.interaction-column')!.getBoundingClientRect();
      const readout=document.querySelector('.live-telemetry')!.getBoundingClientRect();
      return {worldWidth:world.width,consoleWidth:consolePanel.width,worldBottom:world.bottom,readoutTop:readout.top,
        readoutBottom:readout.bottom,consoleTop:consolePanel.top,overflow:document.documentElement.scrollWidth>innerWidth};
    });
    expect(layout.overflow).toBe(false);
    expect(Math.abs(layout.worldWidth-layout.consoleWidth)).toBeLessThan(2);
    expect(layout.readoutTop).toBeGreaterThanOrEqual(layout.worldBottom-1);
    expect(layout.consoleTop).toBeGreaterThanOrEqual(layout.readoutBottom-1);
    await expect(page.getByRole('button',{name:'Stop',exact:true})).toBeInViewport();
    await expect(page.locator('.spectator canvas')).toBeInViewport();
    await expect.poll(async()=>page.locator('.spectator canvas').evaluate((canvas:HTMLCanvasElement)=>{
      const context=canvas.getContext('webgl2')!;
      const pixels=new Uint8Array(canvas.width*canvas.height*4);
      context.readPixels(0,0,canvas.width,canvas.height,context.RGBA,context.UNSIGNED_BYTE,pixels);
      const colors=new Set<string>();
      for(let offset=0;offset<pixels.length;offset+=160) colors.add(`${pixels[offset]},${pixels[offset+1]},${pixels[offset+2]}`);
      return colors.size;
    })).toBeGreaterThan(40);
    await page.screenshot({path:`../.runtime/viewport-console-v1/layout-${width}.png`});
  }
  const canvas=page.locator('.spectator canvas');
  const beforeOrbit=await canvas.evaluate((element:HTMLCanvasElement)=>element.toDataURL());
  await canvas.hover({position:{x:80,y:220}});
  await page.mouse.wheel(0,120);
  await expect.poll(()=>canvas.evaluate((element:HTMLCanvasElement)=>element.toDataURL())).not.toBe(beforeOrbit);
  await page.locator('.live-telemetry > summary').click();
  await expect(page.getByRole('definition').filter({hasText:'Not reported'})).toHaveCount(0);
  await expect(page.getByLabel('Live telemetry summary')).toBeHidden();
  await page.getByRole('button',{name:'Minimize Luna console',exact:true}).click();
  await expect(page.getByRole('tablist',{name:'Robot inspector'})).toBeHidden();
  await expect(page.getByRole('button',{name:'Start mission',exact:true})).toBeEnabled();
  await page.getByRole('button',{name:'Restore Luna console',exact:true}).click();
  await page.getByRole('tab',{name:'Telemetry',exact:true}).click();
  await expect(page.getByRole('region',{name:'Motion diagnostics',exact:true})).toBeVisible();
  await page.locator('.sensor-panel > summary').click();
  await expect(page.getByRole('region',{name:'Collision and distance sensors',exact:true})).toBeVisible();
  await page.getByRole('tab',{name:'Settings',exact:true}).click();
  const slider=page.getByRole('slider',{name:'Luna token budget',exact:true});
  const exact=page.getByRole('spinbutton',{name:'Luna token threshold',exact:true});
  await exact.fill('50000');
  await expect(slider).toHaveValue('25');
  await slider.focus(); await page.keyboard.press('ArrowRight');
  await expect(exact).toHaveValue('52000');
  await expect(slider).toHaveAttribute('aria-valuetext','52,000 tokens');
  await expect.poll(async()=>(await(await request.get('/api/preferences')).json()).preferences.max_model_tokens).toBe(52000);
  await page.getByRole('button',{name:'Start mission',exact:true}).click();
  await expect.poll(()=>sent.length).toBe(1);
  expect(sent[0].max_model_tokens).toBe(52000);
  await page.reload();
  await page.getByRole('button',{name:'Open configuration',exact:true}).click();
  await expect(exact).toHaveValue('52000');
  expect(errors).toEqual([]);
  const after=await(await request.get('/api/state')).json();
  expect(after.snapshot).toEqual(initial.snapshot);
  expect(after.agent.active).toBe(false);
  await expect(consolePanel).toBeVisible();
});

test('Luna usage and thresholds stay visible beside controls across viewports', async ({page,request}) => {
  const initial:LiveState=await(await request.get('/api/state')).json();
  const live:LiveState={...initial,agent:{...initial.agent,session_id:'usage-test',active:false,
    input_tokens:104932,output_tokens:1341,
    inference_budget:{requests:9,max_requests:12,tokens:106273,max_tokens:100000,usage_unknown:false}}};
  let publish:(value:LiveState)=>void=()=>{};
  await page.routeWebSocket('**/api/live',socket=>{publish=value=>socket.send(JSON.stringify(value));publish(live);});
  await page.goto('/');
  const usage=page.getByRole('group',{name:'Luna usage and limits',exact:true});
  await expect(usage).toContainText('106,273 / 100,000');
  await expect(usage).toContainText('9 / 12');
  await expect(usage).toHaveAttribute('data-limit-reached','true');
  for(const width of [1440,850,390,320]) {
    await page.setViewportSize({width,height:1000});
    await page.locator('.agent-section').scrollIntoViewIfNeeded();
    await expect(usage).toBeInViewport();
    await expect(page.getByRole('button',{name:'Stop',exact:true})).toBeInViewport();
    expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
    expect(await usage.evaluate(element=>{
      const parent=element.getBoundingClientRect();
      return [...element.querySelectorAll('span,strong')].every(child=>{
        const bounds=child.getBoundingClientRect();
        return bounds.left>=parent.left-1 && bounds.right<=parent.right+1;
      });
    })).toBe(true);
    await page.screenshot({path:`../.runtime/semantic-mission-v1/usage-${width}.png`});
  }
  publish({...live,agent:{...live.agent,input_tokens:1999999,output_tokens:500,
    inference_budget:{...live.agent.inference_budget!,max_tokens:2000000,usage_unknown:true}}});
  await expect(usage).toContainText('2,000,499 / 2,000,000');
  await expect(usage).toContainText('Usage incomplete');
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
});

test('unified mission is the single default even with old saved diagnostic choices', async ({page,request}) => {
  await request.post('/api/preferences',{data:{control_mode:'exploration',navigation_mode:'luna_navigation',navigation_backend:'nav2',turns:17}});
  const sent: Record<string,unknown>[]=[];
  await page.route('**/api/mission/start',route=>{sent.push(route.request().postDataJSON());return route.fulfill({status:409,json:{detail:'Scripted contract capture'}});});
  await page.goto('/');
  await expect(page.getByRole('button',{name:'Task / Luna',exact:true})).toHaveCount(0);
  await expect(page.getByRole('button',{name:'Explore / local',exact:true})).toHaveCount(0);
  await expect(page.getByRole('button',{name:'Start mission',exact:true})).toBeEnabled();
  await page.getByRole('button',{name:'Start mission',exact:true}).click();
  await expect.poll(()=>sent.length).toBe(1);
  expect(sent[0]).toMatchObject({unified_mission:true,navigation_backend:'builtin',execution_mode:'luna_continuous',
    map_context:true,mission_local_only:false,images_per_request:2,max_turns:17});
  expect(sent[0]).not.toHaveProperty('continuous_handoff');
  await page.getByRole('tab',{name:'Settings',exact:true}).click();
  await page.locator('.mission-diagnostics > summary').click();
  await page.getByRole('combobox',{name:'Diagnostic execution profile',exact:true}).selectOption('local');
  await page.reload();
  await page.getByRole('button',{name:'Open configuration',exact:true}).click();
  await page.locator('.mission-diagnostics > summary').click();
  await expect(page.getByRole('combobox',{name:'Diagnostic execution profile',exact:true})).toHaveValue('unified');
});

test('old backend cannot silently fall back from a unified mission',async({page})=>{
  await page.route('**/api/mission/capabilities',route=>route.fulfill({status:404,json:{detail:'Old server'}}));
  await page.goto('/');
  await expect(page.getByRole('button',{name:'Start mission',exact:true})).toBeDisabled();
  await expect(page.getByRole('alert').filter({hasText:'Mission controller unavailable'})).toBeVisible();
});

test('motion diagnostics expose buffer timing and sensor loss without authorizing motion', async ({page,request}) => {
  const initial:LiveState=await(await request.get('/api/state')).json();
  const diagnostics:NavigationDiagnostics={clock:'monotonic',authorization_id:'recorded-buffer',scope:'short_motion_buffer',
    issuer:'navigation_runtime',renewal_owner:'local_continuous_controller',task_id:'recorded-task',objective_id:'recorded-objective',
    issued_at_s:100,renewed_at_s:100.3,expires_at_s:100.8,
    last_renewal:{at_s:100.3,result:'accepted',rejection_reason:null,expires_at_s:100.8},last_controller_tick_at_s:100.3,
    maximum_recent_tick_gap_s:.266,tick_window_s:5,sensor_at_last_command:{kind:'observed_depth',clock:'monotonic',captured_at_s:99.863,age_s:.437},
    stop:{at_s:101.05,initiator:'watchdog',reason:'BUFFER_EXPIRED: Motion authorization expired'},
    recent_events:[{event:'renewal',at_s:100.3,result:'accepted',expires_at_s:100.8},{event:'stopped',at_s:101.05,initiator:'watchdog',reason:'BUFFER_EXPIRED'}]};
  let live:LiveState={...initial,stopped:true,navigation:{...initial.navigation!,status:'cancelled',reason:'Navigation cancelled',diagnostics},
    agent:{...initial.agent,session_id:'scripted-diagnostics',phase:'completed',unified_mission:true,
      outcome:{kind:'limited',message:'Luna token threshold reached; robot returned to idle',source:'controller',timestamp:1},
      mission:{mission_id:'recorded-mission',phase:'blocked',remaining_s:0,reason:'Luna token threshold reached',plan:null,receipts:{},
        objective:{action:'explore',status:'ended',remaining_s:0,remaining_travel_m:5.49,
          authorization:{id:'recorded-objective',scope:'exploration_objective_not_trajectory',issuer:'mission_supervisor',renewal_owner:'mission_supervisor',
            validator:'simulation_worker',clock:'monotonic',issued_at_s:90,renewed_at_s:null,expires_at_s:140,last_renewal:null,stop:null,recent_events:[]}}}}};
  let publish:(value:LiveState)=>void=()=>{};
  let disconnect:()=>void=()=>{};
  let sensorMode='fresh';
  await page.routeWebSocket('**/api/live',socket=>{publish=value=>socket.send(JSON.stringify(value));disconnect=()=>socket.close();publish(live);});
  await page.route('**/api/spatial',route=>route.fulfill({status:sensorMode==='failed'?503:200,json:{enabled:sensorMode!=='off',paused:false,error:null,frame:null,
    map:{width:4,height:4,cells:Array(16).fill(0),resolution_m:.1,origin_m:[0,0],robot_odometry_m_rad:[0,0,0],
      age_s:sensorMode==='stale'?2:.2,stale:sensorMode==='stale',observed_floor_cells:10,obstacle_cells:6},footprint:null}}));
  const commands:string[]=[];
  const extraPolls:string[]=[];
  page.on('request',message=>{
    if(message.method()==='POST' && !message.url().endsWith('/api/preferences')) commands.push(message.url());
    if(message.url().endsWith('/api/home')) extraPolls.push(message.url());
  });
  const errors:string[]=[];
  page.on('pageerror',error=>errors.push(error.message));
  await page.goto('/');
  await page.getByRole('tab',{name:'Telemetry',exact:true}).click();
  const panel=page.getByRole('region',{name:'Motion diagnostics',exact:true});
  await expect(panel).toContainText('BUFFER_EXPIRED');
  await expect(panel).toContainText('Fresh at receipt');
  await expect(page.getByRole('status',{name:'Robot status',exact:true})).toContainText('Task limit reached');
  await panel.getByText('Sensor and timing details',{exact:true}).click();
  const reading=(label:string)=>panel.locator('dl > div').filter({has:page.locator('dt').filter({hasText:new RegExp(`^${label}$`)})}).locator('dd');
  await expect(reading('Authorized window')).toHaveText('0.500 s / monotonic');
  await expect(reading('Last tick to stop')).toHaveText('0.750 s');
  await expect(reading('Stop relative to expiry')).toHaveText('0.250 s after');
  await expect(reading('Sensor age at command')).toContainText('0.437 s');
  await expect(reading('Depth sample age at receipt')).toHaveText('0.200 s / server monotonic');
  await expect(reading('Last renewal')).toContainText('accepted');
  await panel.getByText('Authorization timeline',{exact:true}).click();
  await expect(reading('Recorded max tick gap')).toHaveText('0.266 s / 5 s window');
  await expect(panel).toContainText('exploration_objective_not_trajectory');
  await expect(panel).toContainText('recorded-objective');
  await page.context().grantPermissions(['clipboard-read','clipboard-write']);
  await panel.getByRole('button',{name:'Copy motion diagnostics',exact:true}).click();
  await expect(panel.getByRole('status')).toHaveText('Diagnostics copied');
  const copied=JSON.parse(await page.evaluate(()=>navigator.clipboard.readText()));
  expect(copied.run_id).toBe(initial.run_id);
  expect(copied.diagnostics).toEqual(diagnostics);
  expect(copied.controller_outcome.message).toContain('token threshold');
  expect(copied).not.toHaveProperty('geometry');
  sensorMode='stale';
  await expect(panel).toContainText('Stale at receipt');
  await expect(reading('Sensor age at command')).toContainText('0.437 s');
  sensorMode='failed';
  await expect(panel).toContainText('Unavailable / last received');
  await expect(panel).toContainText('Spatial sensor unavailable');
  sensorMode='off';
  await expect(panel).toContainText('Sensing off');
  sensorMode='fresh';
  await expect(panel).toContainText('Fresh at receipt');
  for(const width of [1440,390,320]){
    await page.setViewportSize({width,height:1000});
    expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
    await panel.screenshot({path:`../.runtime/cockpit-diagnostics-v1/details-${width}.png`,style:'.command-bar { visibility: hidden; }'});
  }
  live={...live,navigation:{...live.navigation!,diagnostics:{...diagnostics,clock:'test',issued_at_s:null,renewed_at_s:null,
    sensor_at_last_command:null,last_renewal:{at_s:101.1,result:'rejected',rejection_reason:'SPATIAL_STALE'}}}};
  publish(live);
  await expect(reading('Last renewal')).toHaveText('rejected / SPATIAL_STALE');
  await expect(reading('Sensor age at command')).toContainText('Not recorded');
  await expect(reading('Authorized window')).toHaveText('Not recorded / test');
  disconnect();
  await expect(panel).toContainText('Disconnected / last received');
  expect(commands).toEqual([]);
  expect(extraPolls).toEqual([]);
  expect(errors).toEqual([]);
});

test('recorded route failures expose immutable sensor and expiry evidence on demand', async ({page,request}) => {
  const initial:LiveState=await(await request.get('/api/state')).json();
  const task={task_id:'history-task',status:'failed',reason:'BLOCKED: route retries exhausted',frontier_id:'2:1',segments:2,retries:2,
    route_failures:[{phase:'controller',reason:'BUFFER_EXPIRED: Motion authorization expired',segment:1,frontier_id:'2:1',elapsed_s:1.2,clock:'monotonic',
      motion_diagnostics:{clock:'monotonic',authorization_id:'history-buffer',scope:'short_motion_buffer',issuer:'navigation_runtime',renewal_owner:'local_continuous_controller',
        task_id:'history-task',objective_id:'history-objective',issued_at_s:10,renewed_at_s:10.2,expires_at_s:10.7,
        last_renewal:{at_s:10.2,result:'accepted'},last_controller_tick_at_s:10.2,maximum_recent_tick_gap_s:.2,tick_window_s:5,
        sensor_at_last_command:{kind:'observed_depth',clock:'monotonic',age_s:.8},
        stop:{at_s:10.9,initiator:'watchdog',reason:'BUFFER_EXPIRED'},recent_events:[]}},
      {phase:'controller',reason:'OBSERVED_PATH_BLOCKED',segment:2,frontier_id:'2:1',elapsed_s:2}]};
  const events=[
    {id:1,kind:'policy',title:'Mission operation finished',timestamp:1,turn:2,image_url:null,payload:{task}},
    {id:2,kind:'feedback',title:'Mission camera and observed map',timestamp:2,turn:3,image_url:initial.camera.url,
      payload:{observation:{...initial.observation,spatial:{task,localization:{status:'localized',age_s:.12,sample_clock:'monotonic',
        tracking_method:'wheel_odometry_with_fixed_pose_scan_consistency',continuous_pose_correction:false}}},
        history_turns:[],images_in_request:1,tool_result_call_ids:[],
        observed_map_snapshot:{age_s:.719,capture_clock:'monotonic',geometry_age_s:.505,frame:'map',revision:'snapshot-hash'}}},
    {id:3,kind:'response',title:'Luna mission decision',timestamp:3,turn:3,image_url:null,
      payload:{text:'Retry after a buffer expired.',status:'completed',latency_s:6,input_tokens:15000,output_tokens:200,calls:[],refusals:[]}},
    {id:4,kind:'feedback',title:'Operator stopped',timestamp:4,turn:3,image_url:initial.camera.url,
      payload:{observation:{...initial.observation,navigation:{...initial.navigation,status:'cancelled',
        diagnostics:{...task.route_failures[0].motion_diagnostics,stop:{at_s:11,initiator:'external_stop',reason:'Stop requested'}}}},
        history_turns:[],images_in_request:1,tool_result_call_ids:[]}},
  ];
  await page.routeWebSocket('**/api/live',socket=>socket.send(JSON.stringify({...initial,agent:{...initial.agent,session_id:'history-session',trace_revision:events.length}})));
  await page.route('**/api/agent/trace?*',route=>route.fulfill({json:{session_id:'history-session',revision:events.length,first_id:1,capacity:200,events}}));
  const errors:string[]=[];
  page.on('pageerror',error=>errors.push(error.message));
  await page.goto('/');
  await page.getByRole('tab',{name:'Trace',exact:true}).click();
  await page.getByRole('button',{name:'Failures',exact:true}).click();
  await expect(page.locator('.exchange-entry')).toHaveCount(2);
  const policy=page.locator('.exchange-policy');
  await expect(policy).toContainText('BLOCKED: route retries exhausted');
  await policy.locator('.exchange-disclosure > summary').click();
  await policy.getByText('Segment 1',{exact:true}).click();
  const recorded=policy.getByRole('region',{name:'Recorded motion diagnostics',exact:true});
  await expect(recorded).toContainText('Recorded at this event');
  await recorded.getByText('Sensor and timing details',{exact:true}).click();
  await expect(recorded).toContainText('0.800 s / observed_depth');
  await expect(recorded).toContainText('0.700 s');
  await expect(recorded).toContainText('0.200 s after');
  await policy.getByText('Segment 2',{exact:true}).click();
  await expect(policy).toContainText('Timing not recorded');
  const input=page.locator('.exchange-feedback');
  await input.locator('.exchange-disclosure > summary').click();
  await expect(input).toContainText('0.120 s / monotonic');
  await expect(input).toContainText('0.719 s / monotonic');
  await expect(input).toContainText('0.505 s / Unix-derived');
  await expect(input.locator('dl > div').filter({hasText:'Continuous pose correction'})).toContainText('No');
  await page.getByRole('button',{name:'Policy',exact:true}).click();
  await expect(recorded).toBeVisible();
  await expect(recorded).toContainText('0.800 s / observed_depth');
  for(const width of [1440,390,320]) {
    await page.setViewportSize({width,height:1000});
    expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
    await page.locator('.exchange-section').screenshot({path:`../.runtime/cockpit-diagnostics-v1/history-${width}.png`,style:'.command-bar { visibility: hidden; }'});
  }
  expect(errors).toEqual([]);
});

test('compact mission panels preserve drafts and report live policy without issuing commands', async ({page,request}) => {
  const initial:LiveState = await (await request.get('/api/state')).json();
  let publish:(value:LiveState)=>void = () => {};
  let disconnect:()=>void = () => {};
  await page.routeWebSocket('**/api/live', socket => {
    publish = value => socket.send(JSON.stringify(value));
    disconnect = () => socket.close();
    publish(initial);
  });
  const commands:string[]=[];
  page.on('request', message => {if (message.method()==='POST' && !message.url().endsWith('/api/preferences')) commands.push(message.url());});
  const errors:string[]=[];
  page.on('pageerror',error=>errors.push(error.message));
  await page.goto('/');
  await page.getByRole('tab',{name:'Telemetry',exact:true}).click();
  const policy = page.getByLabel('Active policy',{exact:true});
  await expect(policy).toContainText('No active policy');
  await policy.locator(':scope > summary').click();
  await expect(policy).toContainText('No policy has run in this episode.');
  publish({...initial,continuous_navigation:{status:'running',reason:'Observed local destination',remaining_m:1,updates:2,buffer_stops:0}});
  await expect(policy.locator('summary').first()).toContainText('Local navigation');
  await expect(policy).toContainText('Observed local destination');
  await expect(policy).not.toContainText('Model profile');
  publish(initial);
  await expect(policy.locator('summary').first()).toContainText('No active policy');
  const goal = page.getByRole('textbox',{name:'Robot goal',exact:true});
  await page.locator('.mission-instructions > summary').click();
  await goal.fill('Keep this draft while inspecting model output.');
  const instructions = page.locator('.mission-instructions');
  await instructions.locator('summary').click();
  await expect(goal).toBeHidden();
  await expect(instructions.locator('summary')).toContainText('Keep this draft');
  await instructions.locator('summary').focus();
  await page.keyboard.press('Enter');
  await expect(goal).toHaveValue('Keep this draft while inspecting model output.');
  await instructions.locator('summary').click();
  const running:LiveState = {...initial,stopped:false,agent:{...initial.agent,active:true,session_id:'scripted-policy-panel',
    unified_mission:true,execution_mode:'luna_continuous',navigation_backend:'builtin',phase:'thinking',goal:'Explore nearby space.',
    mission:{mission_id:'scripted-mission',phase:'exploring',remaining_s:120,reason:'Review pending; bounded exploration continues.',
      plan:{kind:'explore',target:'',return_home:false},receipts:{},
      objective:{action:'explore',status:'active',remaining_s:8,remaining_travel_m:1.4}}}};
  publish(running);
  await expect(policy.locator('summary').first()).toContainText('Unified mission');
  await expect(policy).toContainText('Built-in');
  await expect(policy).toContainText('explore / active');
  await expect(policy).toContainText('8.0 s / 1.40 m remaining');
  await expect(instructions).not.toHaveAttribute('open');
  const composer=page.getByRole('textbox',{name:'New run instruction',exact:true});
  await composer.fill('Retain this unsent follow-up.');
  await page.getByRole('tab',{name:'Settings',exact:true}).click();
  await expect(page.getByRole('slider',{name:'Luna token budget',exact:true})).toBeDisabled();
  const settings=page.locator('.run-options');
  await expect(settings).toHaveAttribute('open');
  await settings.locator(':scope > summary').click();
  await expect(page.getByRole('spinbutton',{name:'Mission budget',exact:true})).toBeHidden();
  await expect(composer).toHaveValue('Retain this unsent follow-up.');
  await page.getByRole('button',{name:'Close configuration',exact:true}).click();
  await page.getByRole('tab',{name:'Trace',exact:true}).click();
  await expect(composer).toBeVisible();
  await page.getByRole('tab',{name:'Telemetry',exact:true}).click();
  await policy.getByText('Reported policy state',{exact:true}).click();
  await expect(policy.locator('pre')).toContainText('scripted-mission');
  publish({...running,agent:{...running.agent,mission:{...running.agent.mission!,objective:{action:'explore',status:'active',remaining_s:4,remaining_travel_m:.8}}}});
  await expect(policy).toContainText('4.0 s / 0.80 m remaining');
  await expect(policy).toHaveAttribute('open');
  await policy.getByText('Reported policy state',{exact:true}).click();
  for (const width of [1440,1024,390,320]) {
    await page.setViewportSize({width,height:1000});
    await page.getByRole('link',{name:'Luna',exact:true}).click();
    expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
    await expect(composer).toHaveValue('Retain this unsent follow-up.');
    await page.locator('.interaction-column').screenshot({path:`../.runtime/compact-cockpit-v1/policy-${width}.png`});
  }
  publish({...running,stopped:true,agent:{...running.agent,active:false,phase:'stopped'}});
  await expect(policy.locator('summary').first()).toContainText('No active policy');
  await expect(policy).toContainText('Last reported controller');
  await expect(composer).toBeEnabled();
  disconnect();
  await expect(policy.locator('summary').first()).toContainText('Last received / disconnected');
  expect(commands).toEqual([]);
  expect(errors).toEqual([]);
  const after:LiveState = await (await request.get('/api/state')).json();
  expect(after.snapshot).toEqual(initial.snapshot);
  expect(after.agent.active).toBe(false);
});

test('power cycling stays idle and task cost limits persist', async ({page, request}) => {
  await page.setViewportSize({width:1440,height:1000});
  await page.goto('/');
  const initial = await (await request.get('/api/state')).json();
  const power = page.getByRole('switch', {name:'Robot power',exact:true});
  await expect(power).toHaveAttribute('aria-checked','true');
  await power.click();
  await expect(page.getByRole('status',{name:'Robot status',exact:true})).toContainText('Off');
  await expect(page.getByRole('button',{name:'Start mission',exact:true})).toBeDisabled();
  await expect(power).toBeEnabled();
  await power.click();
  await expect(page.getByRole('status',{name:'Robot status',exact:true})).toContainText('On / Idle');
  await expect(page.getByRole('button',{name:'Start mission',exact:true})).toBeEnabled();
  const idle = await (await request.get('/api/state')).json();
  expect(idle.snapshot).toEqual(initial.snapshot);
  expect(idle.run_id).toBe(initial.run_id);
  expect(idle.agent.active).toBe(false);
  expect(idle.agent.input_tokens).toBe(0);
  await page.getByRole('tab',{name:'Settings',exact:true}).click();
  await page.getByRole('spinbutton',{name:'Luna request limit',exact:true}).fill('3');
  await page.getByRole('spinbutton',{name:'Luna token threshold',exact:true}).fill('12000');
  await expect.poll(async()=>(await(await request.get('/api/preferences')).json()).preferences.max_model_tokens).toBe(12000);
  await page.reload();
  await page.getByRole('button',{name:'Open configuration',exact:true}).click();
  await expect(page.getByRole('spinbutton',{name:'Luna request limit',exact:true})).toHaveValue('3');
  const sent:Record<string,unknown>[]=[];
  await page.route('**/api/mission/start',route=>{sent.push(route.request().postDataJSON());return route.fulfill({status:409,json:{detail:'Scripted contract capture'}});});
  await page.getByRole('button',{name:'Start mission',exact:true}).click();
  await expect.poll(()=>sent.length).toBe(1);
  expect(sent[0]).toMatchObject({max_model_requests:3,max_model_tokens:12000});
  for(const width of [1440,390,320]) {
    await page.setViewportSize({width,height:1000});
    expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
    await expect(power).toBeVisible();
    await page.screenshot({path:`../.runtime/unified-controls-v1/ui-${width}.png`});
  }
});

test('robot controls appear once and stay reachable through dialogs', async ({page,request}) => {
  const errors:string[]=[];
  page.on('pageerror',error=>errors.push(error.message));
  await page.setViewportSize({width:1440,height:1000});
  await page.goto('/');
  const bar=page.getByRole('region',{name:'Robot controls',exact:true});
  const stop=page.getByRole('button',{name:'Stop',exact:true});
  const start=page.getByRole('button',{name:'Start mission',exact:true});
  const status=page.getByRole('status',{name:'Robot status',exact:true});
  await expect(start).toBeEnabled();
  await page.getByRole('tab',{name:'Telemetry',exact:true}).click();
  await page.locator('.home-mapping > summary').click();
  await page.locator('.spatial-section > summary').click();
  await expect(page.locator('button.stop-button')).toHaveCount(1);
  await page.locator('.home-mapping > summary').click();
  await page.locator('.spatial-section > summary').click();
  for(const width of [1440,390,320]) {
    await page.setViewportSize({width,height:1000});
    await expect(start).toHaveCount(1);
    await expect(stop).toHaveCount(1);
    await expect(page.getByRole('button',{name:'Open configuration',exact:true})).toBeVisible();
    await expect(page.getByRole('button',{name:'Reset episode',exact:true})).toBeVisible();
    await expect(status).toHaveCount(1);
    expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
    await page.evaluate(()=>scrollTo(0,document.body.scrollHeight));
    const layout=await bar.evaluate(element=>{
      const bounds=element.getBoundingClientRect();
      const controls=[...element.querySelectorAll('.run-actions button, .robot-options > summary')]
        .filter(control=>control.getClientRects().length>0).map(control=>control.getBoundingClientRect());
      const stopButton=element.querySelector<HTMLButtonElement>('.stop-button')!;
      const stopBounds=stopButton.getBoundingClientRect();
      return {top:bounds.top,bottom:bounds.bottom,contained:controls.every(control=>control.left>=bounds.left && control.right<=bounds.right),
        overlap:controls.some((control,index)=>controls.slice(index+1).some(other=>control.left<other.right && other.left<control.right && control.top<other.bottom && other.top<control.bottom)),
        hit:stopButton.contains(document.elementFromPoint(stopBounds.left+stopBounds.width/2,stopBounds.top+stopBounds.height/2))};
    });
    expect(layout.top).toBeGreaterThanOrEqual(0);
    expect(layout.bottom).toBeLessThan(1000);
    expect(layout.contained).toBe(true);
    expect(layout.overlap).toBe(false);
    expect(layout.hit).toBe(true);
    await page.evaluate(()=>scrollTo(0,0));
    await page.screenshot({path:`../.runtime/unified-controls-v1/controls-${width}.png`});
  }
  await page.getByRole('button',{name:'Restore head camera',exact:true}).click();
  await page.getByRole('button',{name:'Restore spatial map',exact:true}).click();
  for(const [opener,name] of [['Expand robot camera','Robot camera'],['Expand spatial map','Spatial map'],['Choose challenge','Load challenge']]) {
    await page.getByRole('button',{name:opener,exact:true}).click();
    const dialog=page.getByRole('dialog',{name,exact:true});
    await expect(dialog).toBeVisible();
    await expect(dialog.getByRole('region',{name:'Robot controls',exact:true})).toHaveCount(1);
    await expect(page.locator('.command-bar')).toHaveCount(1);
    await expect(start).toHaveCount(1);
    await expect(stop).toHaveCount(1);
    await expect(status).toHaveCount(1);
    expect(await dialog.evaluate(element=>element.scrollWidth<=element.clientWidth)).toBe(true);
    await stop.click();
    await expect.poll(async()=>(await(await request.get('/api/state')).json()).stopped).toBe(true);
    await dialog.getByRole('button',{name:'Open configuration',exact:true}).click();
    const configuration=page.getByRole('dialog',{name:'Robot configuration',exact:true});
    await expect(dialog).not.toBeVisible();await expect(configuration).toBeVisible();
    await expect(configuration.getByRole('button',{name:'Reset episode',exact:true})).toBeVisible();
    await expect(configuration.getByRole('button',{name:'Stop',exact:true})).toBeEnabled();
    await page.keyboard.press('Escape');
    await expect(configuration).not.toBeVisible();
    await expect(dialog).not.toBeVisible();
    await expect(page.locator('.observatory > .command-bar')).toHaveCount(1);
  }
  await page.getByLabel('Robot options',{exact:true}).click();
  await page.getByRole('button',{name:'Enable manual control',exact:true}).click();
  await expect.poll(async()=>(await(await request.get('/api/state')).json()).stopped).toBe(false);
  await page.getByLabel('Robot options',{exact:true}).click();
  const before=await(await request.get('/api/state')).json();
  await page.getByRole('button',{name:'Reset episode',exact:true}).click();
  await expect.poll(async()=>(await(await request.get('/api/state')).json()).run_id).not.toBe(before.run_id);
  await expect(start).toBeEnabled();
  expect(errors).toEqual([]);
});

test('one robot status prioritizes current activity power and connection',async({page,request})=>{
  const initial=await(await request.get('/api/state')).json();
  let publish:(value:typeof initial)=>void=()=>{};
  let disconnect:()=>void=()=>{};
  await page.routeWebSocket('**/api/live',socket=>{
    publish=value=>socket.send(JSON.stringify(value));
    disconnect=()=>socket.close();
    publish(initial);
  });
  await page.goto('/');
  const status=page.getByRole('status',{name:'Robot status',exact:true});
  await expect(status).toHaveCount(1);
  const reported={...initial,stopped:false,busy:false,result:null,agent:{...initial.agent,active:false,phase:'completed',
    outcome:{kind:'completed',source:'agent',message:'Goal reached',timestamp:Date.now()/1000}}};
  publish(reported);
  await expect(status).toContainText('Agent reports completion');
  await expect(page.getByText('Agent report / scenario not verified complete',{exact:true})).toBeVisible();
  publish({...reported,agent:{...reported.agent,active:true,phase:'thinking'}});
  await expect(status).toContainText('Model thinking');
  await expect(page.getByRole('button',{name:'Start mission',exact:true})).toBeDisabled();
  publish({...reported,power:{...initial.power,on:false,mode:'off'}});
  await expect(status).toContainText('Off');
  await expect(page.getByText('Agent reports completion',{exact:true})).toHaveCount(0);
  disconnect();
  await expect(status).toContainText('Connection lost');
  await expect(page.getByRole('button',{name:'Stop',exact:true})).toBeDisabled();
  await expect(page.getByRole('status',{name:'Task outcome',exact:true})).toHaveCount(0);
});

test('unified mission shows actual observed map input and Stop revokes the shared session',async({page,request})=>{
  test.setTimeout(100000);
  const errors:string[]=[];
  page.on('pageerror',error=>errors.push(error.message));
  await page.setViewportSize({width:1440,height:1100});
  await page.goto('/');
  const destination=await realpath(await mkdtemp(join(tmpdir(),'milo-recorded-mission-')));
  await page.getByRole('tab',{name:'Settings',exact:true}).click();
  const recordingPanel=page.getByRole('region',{name:'Test run recording',exact:true});
  await recordingPanel.getByRole('textbox',{name:'Recording folder',exact:true}).fill(destination);
  await recordingPanel.getByRole('button',{name:'Save recording settings',exact:true}).click();
  await expect(recordingPanel.getByText('Recording settings saved',{exact:true})).toBeVisible();
  await page.getByRole('button',{name:'Save configuration',exact:true}).click();
  await expect(page.getByRole('dialog',{name:'Robot configuration',exact:true})).toBeHidden();
  await page.getByRole('tab',{name:'Conversation',exact:true}).click();
  const initial:LiveState=await(await request.get('/api/state')).json();
  const custom='Explore nearby observed space and report what you can see.';
  const composer=page.getByRole('textbox',{name:'New run instruction',exact:true});
  await composer.fill(custom);
  const starting=page.waitForResponse(response=>response.url().endsWith('/api/mission/start') && response.request().method()==='POST');
  await page.getByRole('button',{name:'Send instruction',exact:true}).click();
  expect((await starting).ok()).toBe(true);
  await expect(composer).toHaveValue('');
  await expect(page.getByRole('log',{name:'Run conversation',exact:true})).toContainText(custom);
  const active:LiveState=await(await request.get('/api/state')).json();
  expect(active.run_id).toBe(initial.run_id);expect(active.episode_epoch).toBe(initial.episode_epoch);
  expect(active.agent.goal).toBe(custom);
  await expect(page.getByRole('status',{name:'Run recording',exact:true})).toHaveText('Recording');
  await page.getByRole('tab',{name:'Settings',exact:true}).click();
  await expect(recordingPanel.getByRole('textbox',{name:'Recording folder',exact:true})).toBeDisabled();
  await expect(recordingPanel.getByRole('switch',{name:'Record test runs',exact:true})).toBeDisabled();
  await page.getByRole('button',{name:'Close configuration',exact:true}).click();
  await page.getByRole('tab',{name:'Trace',exact:true}).click();
  await expect(page.getByRole('status',{name:'Robot status',exact:true})).toBeVisible();
  await page.locator('.exchange-entry').filter({has:page.locator('img[alt="Observed map supplied to Luna"]')})
    .first().locator('.exchange-disclosure > summary').click({timeout:70000});
  await expect(page.getByAltText('Observed map supplied to Luna').first()).toHaveJSProperty('naturalWidth',512,{timeout:70000});
  const trace=await(await request.get('/api/agent/trace')).json();
  const input=trace.events.find((event:{title:string})=>event.title==='Mission camera and observed map');
  expect(input.payload.image_roles).toEqual(['current_head','observed_map']);
  expect(input.payload.observed_map_snapshot.cells.length).toBe(input.payload.observed_map_snapshot.width*input.payload.observed_map_snapshot.height);
  for(const width of [1440,390]){
    await page.setViewportSize({width,height:1100});
    expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
    await page.locator('.agent-section').screenshot({path:`../.runtime/unified-mission-v1/ui-${width}.png`});
  }
  await page.getByRole('button',{name:'Stop',exact:true}).click();
  await expect.poll(async()=>(await(await request.get('/api/agent')).json()).active).toBe(false);
  const stopped=await(await request.get('/api/state')).json();
  expect(stopped.agent.mission.phase).toBe('cancelled');
  expect(stopped.stopped).toBe(true);
  await expect.poll(async()=>(await(await request.get('/api/recording')).json()).active).toBe(false);
  const recording:NonNullable<LiveState['recording']>=await(await request.get('/api/recording')).json();
  expect(recording.run_directory!.startsWith(destination)).toBe(true);
  const score=JSON.parse(await readFile(join(recording.run_directory!,'scorecard.json'),'utf8'));
  expect(score.complete_recording).toBe(true);expect(score.samples).toBeGreaterThan(0);
  expect((await readFile(join(recording.run_directory!,'trajectory.jsonl'),'utf8')).length).toBeGreaterThan(0);
  expect((await readFile(join(recording.run_directory!,'replay.html'),'utf8')).length).toBeGreaterThan(0);
  const catalog=await(await request.get('/api/test-results')).json();
  const batch=catalog.batches.find((entry:{session_id:string})=>entry.session_id===active.agent.session_id);
  expect(batch).toBeTruthy();expect((await request.get(batch.trials[0].replay_url)).ok()).toBe(true);
  await page.getByRole('tab',{name:'Settings',exact:true}).click();
  await expect(recordingPanel).toContainText(recording.run_directory!);
  await page.context().grantPermissions(['clipboard-read','clipboard-write']);
  await recordingPanel.getByRole('button',{name:'Copy recording path',exact:true}).click();
  expect(await page.evaluate(()=>navigator.clipboard.readText())).toBe(recording.run_directory);
  await writeFile('../.runtime/recording-settings-v1/browser-recording.json',JSON.stringify({evidence:'scripted_inference_real_physics',
    destination:recording.run_directory,samples:score.samples,complete:score.complete_recording,archive_id:batch.id},null,2));
  await expect(composer).toBeEnabled();
  expect(errors).toEqual([]);
});

test('movement practice renders, completes a measured sequence and stops a custom spin', async ({page,request}) => {
  test.setTimeout(240000);
  const errors:string[]=[];page.on('pageerror',error=>errors.push(error.message));
  await request.post('/api/preferences',{data:{exploration_budget:150,max_model_requests:2,recording_enabled:false}});
  await page.setViewportSize({width:1440,height:1000});
  await page.goto('/');
  await page.getByRole('button',{name:'Choose challenge',exact:true}).click();
  const menu=page.getByRole('dialog',{name:'Load challenge',exact:true});
  const selector=menu.getByRole('combobox',{name:'Predefined challenge',exact:true});
  await selector.selectOption('movement_practice');
  await menu.getByRole('combobox',{name:'Map source',exact:true}).selectOption('none');
  await menu.getByRole('button',{name:'Load challenge',exact:true}).click();
  await expect(menu).toBeHidden();
  await expect(page.getByRole('heading',{name:'Movement Practice',exact:true})).toBeVisible();
  const canvas=page.locator('.spectator canvas');
  const colors=()=>canvas.evaluate((surface:HTMLCanvasElement)=>{
    const context=surface.getContext('webgl2')!;
    const data=new Uint8Array(surface.width*surface.height*4);
    context.readPixels(0,0,surface.width,surface.height,context.RGBA,context.UNSIGNED_BYTE,data);
    const distinct=new Set<string>();
    for(let offset=0;offset<data.length;offset+=160) distinct.add(`${data[offset]},${data[offset+1]},${data[offset+2]}`);
    return distinct.size;
  });
  await expect.poll(colors).toBeGreaterThan(30);
  if(process.env.MILO_CAPTURE_PREVIEWS==='1') {
    const encoded=await canvas.evaluate((surface:HTMLCanvasElement)=>{
      const original=document.createElement('canvas');original.width=surface.width;original.height=surface.height;
      const originalContext=original.getContext('2d')!;originalContext.drawImage(surface,0,0);
      const data=originalContext.getImageData(0,0,surface.width,surface.height).data;
      let left=surface.width,right=0,top=surface.height,bottom=0;
      for(let row=0;row<surface.height;row+=2) for(let column=0;column<surface.width;column+=2) {
        const offset=(row*surface.width+column)*4;
        if(Math.max(...[0,1,2].map(channel=>Math.abs(data[offset+channel]-data[channel])))<=8) continue;
        left=Math.min(left,column);right=Math.max(right,column);top=Math.min(top,row);bottom=Math.max(bottom,row);
      }
      left=Math.max(0,left-20);right=Math.min(surface.width,right+20);
      top=Math.max(0,top-20);bottom=Math.min(surface.height,bottom+20);
      const cropWidth=Math.max(1,right-left),cropHeight=Math.max(1,bottom-top);
      const image=document.createElement('canvas');image.width=480;image.height=300;
      const context=image.getContext('2d')!;
      context.fillStyle=getComputedStyle(document.documentElement).getPropertyValue('--cp-surface-soft').trim();
      context.fillRect(0,0,480,300);
      const scale=Math.min(480/cropWidth,300/cropHeight);
      context.drawImage(original,left,top,cropWidth,cropHeight,(480-cropWidth*scale)/2,(300-cropHeight*scale)/2,cropWidth*scale,cropHeight*scale);
      return image.toDataURL('image/webp',.9).split(',')[1];
    });
    const directory=new URL('../public/scenario-previews/',import.meta.url);
    await mkdir(directory,{recursive:true});
    await writeFile(new URL('movement_practice.webp',directory),Buffer.from(encoded,'base64'));
  }
  for(const width of [1440,390,320]) {
    await page.setViewportSize({width,height:1000});
    await expect.poll(colors).toBeGreaterThan(30);
    expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
    await canvas.scrollIntoViewIfNeeded();
    await page.screenshot({path:`../.runtime/movement-practice-v1/scene-${width}.png`});
    await page.getByRole('button',{name:'Choose challenge',exact:true}).click();
    if(process.env.MILO_CAPTURE_PREVIEWS!=='1') {
      await expect(menu.getByAltText('Scene preview: Movement Practice')).toHaveJSProperty('naturalWidth',480);
    }
    expect(await menu.evaluate(element=>element.scrollWidth<=element.clientWidth)).toBe(true);
    await page.screenshot({path:`../.runtime/movement-practice-v1/picker-${width}.png`});
    await menu.getByRole('button',{name:'Close challenge menu',exact:true}).click();
  }
  await page.setViewportSize({width:1440,height:1000});
  const initial:LiveState=await(await request.get('/api/state')).json();
  await page.getByRole('button',{name:'Start mission',exact:true}).click();
  await expect.poll(async()=>(await(await request.get('/api/agent')).json()).mission?.phase,{timeout:150000}).toBe('completed');
  await expect.poll(async()=>(await(await request.get('/api/agent')).json()).active).toBe(false);
  const completed:LiveState=await(await request.get('/api/state')).json();
  expect(completed.agent.error).toBeNull();
  expect(completed.challenge?.status).toBe('completed');
  expect(completed.challenge?.completed_objectives).toBe(3);
  expect(completed.agent.inference_budget?.requests).toBe(1);
  expect(completed.stopped).toBe(true);
  await page.screenshot({path:'../.runtime/movement-practice-v1/completed.png'});
  const resetting=page.waitForResponse(response=>response.url().endsWith('/api/reset') && response.request().method()==='POST');
  await page.getByRole('button',{name:'Reset episode',exact:true}).click();
  expect((await resetting).ok()).toBe(true);
  await expect.poll(async()=>(await(await request.get('/api/state')).json()).run_id).not.toBe(initial.run_id);
  const reset:LiveState=await(await request.get('/api/state')).json();
  await expect(page.locator('.viewport-footer')).toContainText(`Epoch ${reset.episode_epoch}`);
  const composer=page.getByRole('textbox',{name:'New run instruction',exact:true});
  await expect(composer).toBeEnabled();
  await composer.fill('Spin three times around on the spot.');
  const starting=page.waitForResponse(response=>response.url().endsWith('/api/mission/start') && response.request().method()==='POST');
  await page.getByRole('button',{name:'Send instruction',exact:true}).click();
  expect((await starting).ok()).toBe(true);
  await expect(page.getByRole('log',{name:'Run conversation',exact:true})).toContainText('Spin three times around on the spot.');
  await expect.poll(async()=>{
    const state:LiveState=await(await request.get('/api/state')).json();
    const rotation=state.snapshot.poses.find(pose=>pose.key===`${state.robot_body_id}:-1`)!.quaternion;
    return Math.abs(2*Math.atan2(rotation[2],rotation[3]));
  },{timeout:60000}).toBeGreaterThan(.15);
  await page.getByRole('button',{name:'Stop',exact:true}).click();
  await expect.poll(async()=>(await(await request.get('/api/agent')).json()).active).toBe(false);
  const stopped:LiveState=await(await request.get('/api/state')).json();
  expect(stopped.agent.mission?.phase).toBe('cancelled');expect(stopped.stopped).toBe(true);
  expect((await(await request.get('/api/state')).json()).snapshot).toEqual(stopped.snapshot);
  expect(errors).toEqual([]);
});