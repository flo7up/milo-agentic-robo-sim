import { test, expect } from '@playwright/test';
import type { HomeState } from '../src/HomeMapping';
import type { Batch } from '../src/TestResults';

test.beforeEach(async ({request}) => {
  expect((await request.post('/api/test/preferences/reset')).ok()).toBe(true);
});

test('baseline progress separates gains regressions gaps and incompatible runs', async ({page}) => {
  const titles = ['Navigate', 'Return Home', 'Room entry'];
  function batch(id: string, date: string, passed: number[], cohort = 'matched', eligible = true): Batch {
    const tasks = titles.map((title, index) => ({task_id: String(index), title, planned:3, passed:passed[index], failed:index===2?0:3-passed[index],blocked:index===2?3:0,invalid:0,not_run:0}));
    return {id,name:id,date,date_source:'recorded',design:id,source_sha256:id,mode:'home_mapping',evidence:'scripted_test',model:'None',reasoning:'None',budget_s:60,history:'None',legacy:false,source_changed:!eligible,planned:9,successes:0,
      benchmark:{suite_id:'household-foundation-v1',suite_sha256:'suite',map_sha256:cohort,preflight:false,tasks,comparison:{eligible,reasons:eligible?[]:['Source stability not established'],cohort_id:cohort,experiment_id:id,baseline_experiment_id:''}},
      trials:tasks.flatMap((task,index)=>Array.from({length:3},(_,repeat)=>({case_id:`${index}-${repeat}`,challenge:'flat_kitchen',title:task.title,evidence:'scripted_test',status:repeat<task.passed?'Benchmark pass':index===2?'Blocked prerequisite':'Benchmark failed',verified_success:false,physics_success:false,recording_complete:true,assisted:false,completion_s:null,elapsed_s:id==='baseline'?10+repeat*2:7+repeat*3,distance_m:1,contact_episodes:0,input_tokens:0,output_tokens:0,inference_median_s:null,turns:0,termination:'test',rendering:'enhanced',false_completion_claim:false,image_url:null,
        benchmark:{suite_id:'household-foundation-v1',task_id:task.task_id,status:repeat<task.passed?'passed':index===2?'blocked':'failed',reason:`${id} ${task.title} ${repeat}`,criteria:'Fixed task criteria',budget_s:60,setup_s:1,position_error_m:null,new_free_m2:null}})))};
  }
  const batches=[batch('candidate','2026-09-15T10:00:00Z',[3,2,0]),batch('baseline','2026-09-14T10:00:00Z',[0,3,0]),batch('different-map','2026-09-16T10:00:00Z',[3,3,0],'other'),batch('drift','2026-09-17T10:00:00Z',[3,3,0],'matched',false)];
  await page.route('**/api/test-results',route=>route.fulfill({json:{batches,skipped:0,truncated:false}}));
  await page.goto('/?view=test-results');
  const progress=page.getByRole('region',{name:'Baseline progress',exact:true});
  await expect(progress).toBeVisible();
  await expect(progress.getByRole('combobox',{name:'Baseline run',exact:true})).toHaveValue('baseline');
  const rows=progress.getByRole('table',{name:'Task changes against baseline'}).locator('tbody tr');
  await expect(rows.nth(0)).toHaveAttribute('data-change','improved');
  await expect(rows.nth(1)).toHaveAttribute('data-change','regressed');
  await expect(rows.nth(2)).toContainText('3 blocked');
  await expect(rows.nth(1)).toContainText('2 identical successful cases');
  await expect(progress.getByRole('combobox',{name:'Candidate run',exact:true}).locator('option')).toHaveCount(1);
  await progress.getByText('Not comparable (2)',{exact:true}).click();
  await expect(progress).toContainText('Source stability not established');
  for(const width of [1440,768,390,320]) {
    await page.setViewportSize({width,height:1000});
    expect(await progress.evaluate(element=>element.scrollWidth<=element.clientWidth)).toBe(true);
    expect(await progress.locator('select').evaluateAll(selects=>selects.every(select=>select.getBoundingClientRect().right<=innerWidth))).toBe(true);
    await progress.screenshot({path:`.runtime/baseline-progress-${width}.png`});
  }
  await rows.nth(1).getByRole('button').click();
  await expect(page.getByRole('region',{name:'Selected trial details'})).toContainText('candidate Return Home 2');
  await page.getByRole('button',{name:'Progress',exact:true}).click();
  await progress.getByRole('combobox',{name:'Baseline run',exact:true}).selectOption('candidate');
  await page.reload();
  await expect(progress.getByRole('combobox',{name:'Baseline run',exact:true})).toHaveValue('candidate');
  await progress.getByRole('combobox',{name:'Baseline run',exact:true}).selectOption('different-map');
  await expect(progress.getByRole('combobox',{name:'Candidate run',exact:true})).toBeDisabled();
  await expect(progress).toContainText('no complete matched candidate yet');
});

test('scenario map reuse is optional and survives reset without deleting saved maps', async ({page,request}) => {
  test.setTimeout(90000);
  await request.post('/api/challenges/load',{data:{challenge_id:'park'}});
  await page.goto('/');
  await page.getByRole('button',{name:'Load scenario',exact:true}).click();
  const dialog=page.getByRole('dialog',{name:'Load scenario',exact:true});
  await dialog.getByRole('combobox',{name:'Map source',exact:true}).selectOption('none');
  const loaded=page.waitForResponse(response=>response.url().endsWith('/api/challenges/load')&&response.request().method()==='POST');
  await dialog.getByRole('button',{name:'Load selected scenario',exact:true}).click();
  expect((await loaded).request().postDataJSON().reuse_saved_map).toBe(false);
  await expect.poll(async()=>(await(await request.get('/api/state')).json()).map_setup.reuse_saved_map).toBe(false);
  const reset=await request.post('/api/reset');
  expect((await reset.json()).map_setup).toMatchObject({reuse_saved_map:false,map_id:null});
  await page.getByRole('button',{name:'Load scenario',exact:true}).click();
  await expect(dialog.getByRole('combobox',{name:'Map source',exact:true})).toHaveValue('none');
  await dialog.getByRole('combobox',{name:'Map source',exact:true}).selectOption('saved');
  await dialog.getByRole('button',{name:'Load selected scenario',exact:true}).click();
  await expect.poll(async()=>(await(await request.get('/api/state')).json()).map_setup.reuse_saved_map).toBe(true);
});

test('local-only diagnostics use the unified mission endpoint and expose Stop', async ({page,request}) => {
  await request.post('/api/challenges/load',{data:{challenge_id:'park',reuse_saved_map:false}});
  await request.post('/api/resume');
  const operations: string[]=[];
  await page.route('**/api/mission/start',route=>{const body=route.request().postDataJSON();expect(body.mission_local_only).toBe(true);expect(body.unified_mission).toBe(true);operations.push('mission');return route.fulfill({status:409,json:{detail:'Scripted start capture; no motion'}});});
  await page.route('**/api/stop',route=>{operations.push('stop');return route.fulfill({json:{stopped:true}});});
  page.on('request',request=>{if(request.method()==='POST'&&request.url().includes('/api/agent/'))operations.push('agent-request');});
  await page.goto('/');
  await page.locator('.run-options > summary').click();
  await page.locator('.mission-diagnostics > summary').click();
  await page.getByRole('combobox',{name:'Diagnostic execution profile',exact:true}).selectOption('local');
  const panel=page.getByRole('region',{name:'LLM control',exact:true});
  await page.getByRole('button',{name:'Start mission',exact:true}).click();
  await expect(panel).toContainText('Scripted start capture');
  for(const width of [1440,390,320]){
    await page.setViewportSize({width,height:1000});
    const layout=await panel.evaluate(element=>({width:element.clientWidth,scroll:element.scrollWidth,
      outside:[...element.querySelectorAll('*')].filter(child=>child.getBoundingClientRect().right>element.getBoundingClientRect().right+1)
        .map(child=>({tag:child.tagName,class:child.className,right:child.getBoundingClientRect().right,text:child.textContent?.slice(0,60)})).slice(0,8)}));
    expect(layout.scroll,JSON.stringify(layout)).toBeLessThanOrEqual(layout.width);
  }
  await page.getByRole('button',{name:'Stop',exact:true}).click();
  expect(operations).toEqual(['mission','stop']);
});

test('standardized benchmark shows task denominators blocked cases and criteria without scenario pass claims', async ({page}) => {
  const tasks = ['Localize after restart', 'Navigate to checkpoint', 'Return Home', 'Occupied destination', 'Explore partial map', 'Stop during motion', 'Room to room'];
  const benchmark = {suite_id:'household-foundation-v1',suite_sha256:'suite-fixed',map_sha256:'map-fixed',preflight:false,
    tasks:tasks.map((title,index)=>({task_id:String(index),title,planned:3,passed:index===0?3:0,failed:index>0&&index<6?3:0,blocked:index===6?3:0,invalid:0,not_run:0}))};
  const trial={case_id:'localize-r1',challenge:'flat_kitchen',title:tasks[0],evidence:'scripted_test',status:'Benchmark pass',verified_success:false,physics_success:false,recording_complete:true,assisted:false,completion_s:null,elapsed_s:.4,distance_m:0,contact_episodes:0,input_tokens:0,output_tokens:0,inference_median_s:null,turns:0,termination:'passed',rendering:'enhanced',false_completion_claim:false,image_url:null,trajectory_url:null,
    benchmark:{suite_id:'household-foundation-v1',task_id:'localize',status:'passed',reason:'Independent pose check passed',criteria:'Pose error at most 0.15 m',budget_s:20,setup_s:15,position_error_m:.01,new_free_m2:null}};
  await page.route('**/api/test-results',route=>route.fulfill({json:{batches:[{id:'baseline',name:'Household baseline',date:'2026-09-14T12:00:00Z',date_source:'recorded',design:'household-foundation-v1',source_sha256:'fixed',mode:'home_mapping',evidence:'scripted_test',model:'None',reasoning:'None',budget_s:null,history:'None',legacy:false,source_changed:false,planned:21,successes:0,trials:[trial],benchmark}],skipped:0,truncated:false}}));
  await page.goto('/?view=test-results&evidence=scripted_test');
  const panel=page.getByRole('region',{name:'Standardized benchmark baseline'});
  await expect(panel).toContainText('household-foundation-v1');
  await expect(panel.locator('tbody tr')).toHaveCount(7);
  await expect(panel.locator('tbody tr').first()).toContainText('3 / 3');
  await expect(panel.locator('tbody tr').last()).toContainText('0 / 3');
  await expect(page.getByRole('region',{name:'Selected trial details'})).toContainText('Pose error at most 0.15 m');
  await expect(page.locator('.results-trial-heading')).toContainText('no model inference');
  for(const width of [1440,390,320]){
    await page.setViewportSize({width,height:1000});
    expect(await page.locator('.results-page').evaluate(element=>element.scrollWidth<=element.clientWidth)).toBe(true);
    await panel.screenshot({path:`.runtime/household-baseline-${width}.png`});
  }
});


test('spatial replay synchronizes past images map coverage and failure events without robot writes', async ({page}) => {
  const prefix='/api/test-results/replay-fixture/replay/0';
  const coverage=(amount:number)=>({known_cells:amount*100,free_m2:amount,visited_cells:amount,scan_count:amount});
  const map=(amount:number)=>({map_id:'frozen-home',revision:1,name:'Recorded home',sha256:'fixture',width:400,height:400,resolution_m:.1,origin_m:[-20,-20],
    cells:Array.from({length:160000},(_,index)=>{const column=index%400,row=Math.floor(index/400);return column>=180&&column<200+amount*5&&row>=185&&row<210?column===180||row===185?100:0:-1;}),
    visited_indices:[200*400+190,200*400+191],places:[{place_id:'home',name:'Home',pose_m_rad:[-1,0,0]}],edges:[]});
  const frames=[0,2,4].map((seconds,index)=>({wall_s:seconds,simulated_s:seconds,index:index+1,activity:seconds===4?'blocked':'driving',status:'in_progress',contact:false,assisted:false,odometry_m_rad:[seconds/4,0,0],
    camera_url:`${prefix}/media/camera-${index+1}.png`,depth_url:null,camera_simulated_s:seconds,depth_simulated_s:null,
    map_url:`${prefix}/media/home-${index+1}.json`,telemetry_url:`${prefix}/media/telemetry-${index+1}.json`,
    home:{map:{map_id:'frozen-home',revision:1,sequence:index+1,wall_s:seconds},coverage:coverage(index+1),pose_m_rad:[seconds/4,0,0],sampled_wall_s:seconds,localization:{status:seconds===4?'unlocalized':'localized'},depth_age_s:.2,lidar_age_s:.1,
      task:{kind:'navigate',status:seconds===4?'failed':'running',reason:seconds===4?'LOCALIZATION_LOST':'Following route',segments:index+1,retries:index},stage:'navigation'}}));
  const spatial={complete:true,snapshots:3,events:2,map_id:'frozen-home',map_revision:1,initial:coverage(1),final:coverage(3),final_task:{kind:'navigate',status:'failed'}};
  const trial={case_id:'navigation',challenge:'park',title:'Recorded route',evidence:'scripted_test',status:'Not passed',verified_success:false,physics_success:false,recording_complete:true,assisted:false,completion_s:null,elapsed_s:4,distance_m:1,contact_episodes:0,input_tokens:0,output_tokens:0,inference_median_s:null,turns:0,termination:'failed',rendering:'tiny',false_completion_claim:false,image_url:null,trajectory_url:null,replay_url:prefix,spatial_progress:spatial};
  const writes:string[]=[];
  page.on('request', request=>{if(request.method()!=='GET'&&request.url().includes('/api/'))writes.push(request.url());});
  await page.route('**/api/test-results',route=>route.fulfill({json:{batches:[{id:'replay-fixture',name:'Scripted replay fixture',date:'2026-09-14T12:00:00Z',date_source:'recorded',design:'spatial-replay-v1',source_sha256:'fixture',mode:'home_mapping',evidence:'scripted_test',model:'None',reasoning:'None',budget_s:4,history:'None',legacy:false,source_changed:false,planned:1,successes:0,trials:[trial]}],skipped:0,truncated:false}}));
  await page.route(`**${prefix}`,route=>route.fulfill({json:{frames,events:[{id:1,wall_s:0,stage:'navigation',localization:'localized',status:'running',reason:'Task started',segments:1,retries:0},{id:2,wall_s:4,stage:'navigation',localization:'unlocalized',status:'failed',reason:'LOCALIZATION_LOST',segments:3,retries:2}],evidence:'scripted_test',sample_count:81,recording_complete:true,spatial_progress:spatial}}));
  await page.route(`**${prefix}/media/*`,route=>{
    const name=route.request().url().split('/').at(-1)!;
    if(name.endsWith('.png'))return route.fulfill({contentType:'image/png',body:Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jD1sAAAAASUVORK5CYII=','base64')});
    const amount=Number(name.match(/\d+/)![0]);
    return route.fulfill({json:name.startsWith('home')?map(amount):{live_obstacles_m:[[1,1]],route_m:[[0,0],[1,0]]}});
  });
  await page.goto('/?view=test-results');
  await page.getByRole('combobox',{name:'Filter test evidence',exact:true}).selectOption('scripted_test');
  const replay=page.getByRole('region',{name:'Synchronized spatial replay',exact:true});
  await expect(replay).toBeVisible();
  await expect(page.locator('.spatial-progress')).toContainText('2 m2');
  const image=replay.getByRole('img',{name:'Recorded head camera',exact:true});
  await expect(image).toHaveAttribute('src',`${prefix}/media/camera-1.png`);
  await replay.getByRole('slider',{name:'Replay time',exact:true}).fill('1.9');
  await expect(image).toHaveAttribute('src',`${prefix}/media/camera-1.png`);
  await replay.getByRole('slider',{name:'Replay time',exact:true}).fill('2');
  await expect(image).toHaveAttribute('src',`${prefix}/media/camera-2.png`);
  await replay.getByRole('button',{name:'Jump to event 2: LOCALIZATION_LOST',exact:true}).click();
  await expect(replay.getByRole('slider',{name:'Replay time',exact:true})).toHaveValue('4');
  await expect(replay.locator('.replay-task-reason')).toHaveText('LOCALIZATION_LOST');
  await expect(replay.getByLabel('Recorded observed map')).toHaveAttribute('data-map-id','frozen-home');
  for(const width of [1440,390,320]){
    await page.setViewportSize({width,height:1000});
    const geometry=await replay.evaluate(element=>({fits:element.scrollWidth<=element.clientWidth,viewport:innerWidth}));
    expect(geometry.fits).toBe(true);
    const colors=await replay.getByLabel('Recorded observed map').evaluate((canvas:HTMLCanvasElement)=>{const pixels=canvas.getContext('2d')!.getImageData(0,0,640,480).data;const colors=new Set();for(let index=0;index<pixels.length;index+=64)colors.add(`${pixels[index]},${pixels[index+1]},${pixels[index+2]}`);return colors.size;});
    expect(colors).toBeGreaterThan(5);
    await replay.screenshot({path:`.runtime/spatial-replay-${width}.png`});
  }
  await replay.getByRole('button',{name:'Rewind replay',exact:true}).click();
  await expect(image).toHaveAttribute('src',`${prefix}/media/camera-1.png`);
  await replay.getByRole('button',{name:'Play replay',exact:true}).click();
  await expect.poll(async()=>Number(await replay.getByRole('slider',{name:'Replay time',exact:true}).inputValue())).toBeGreaterThan(.2);
  await replay.getByRole('button',{name:'Pause replay',exact:true}).click();
  expect(writes).toEqual([]);
});

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
    else if (body.action === 'guided_to') current = { ...current, stage: 'mapping', dirty: true, route_m: [[0, 0], body.pose_m_rad.slice(0, 2)],
      task: { status: 'completed', reason: 'Observed survey waypoint reached', segments: 1, retries: 0, visited_frontiers: 0 } };
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
    else if (body.action === 'continue_mapping') current = { ...current, stage: 'mapping', dirty: true };
    else if (body.action === 'review_room') current = { ...current, dirty: true,
      places: current.places.map(place => place.place_id === 'place-1' ? { ...place, identity_status: 'operator_confirmed' } : place),
      room_observations: current.room_observations?.map(observation => ({ ...observation, review_status: 'operator_confirmed' })) };
    return route.fulfill({ json: current });
  });
  await page.route('**/api/stop', route => {
    current = {...current, task:current.task ? {...current.task,status:'cancelled',reason:'Cancelled by operator'} : null};
    return route.fulfill({json:{stopped:true}});
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
  await page.route('**/api/command', route => route.fulfill({ json: { status: 'error', error: 'GUIDED_MAP_BLOCKED', message: 'Guided rotation lacks observed clearance' } }));
  await panel.getByRole('button', { name: 'Map: rotate left', exact: true }).click();
  await expect(panel.getByRole('alert')).toContainText('Guided rotation lacks observed clearance');
  await page.unroute('**/api/command');
  await expect(panel.getByRole('button', { name: 'Survey to selected point', exact: true })).toBeDisabled();
  await canvas.click({ position: { x: 220, y: 180 } });
  await panel.getByRole('button', { name: 'Survey to selected point', exact: true }).click();
  expect(operations.find(operation => operation.action === 'guided_to')?.pose_m_rad).toHaveLength(3);
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
  await page.getByRole('button', { name: 'Stop', exact: true }).click();
  await expect(panel.locator('.home-task-status')).toContainText('cancelled');
  expect(operations.find(operation => operation.action === 'navigate_to')?.place_id).toBe('place-1');
  expect(operations.filter(operation => operation.action === 'start_mapping')).toHaveLength(1);
  current = { ...current, room_verification: { status: 'not_currently_verified', identity_verified: false },
    room_observations: [{ observation_id: 'room-evidence', place_id: 'place-1', label: 'Kitchen entrance',
      evidence: 'A sink and stove are visible beyond the doorway.', confidence: .7, review_status: 'tentative',
      room_matches: null, observed_unix_s: 1789387200, image_url: '/api/home/sensor-map/rooms/room-evidence/image.png' }] };
  await page.route('**/api/home/*/rooms/*/image.png', route => route.fulfill({ contentType: 'image/png',
    body: Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jD1sAAAAASUVORK5CYII=', 'base64') }));
  await panel.getByText('Room observations (1)', { exact: true }).click();
  await expect(panel.getByRole('img', { name: 'Room evidence for Kitchen entrance', exact: true })).toBeVisible();
  await expect(panel.getByLabel('Current room report')).toContainText('not currently verified');
  await panel.getByRole('button', { name: 'Confirm Kitchen entrance', exact: true }).click();
  await expect(panel.locator('.home-place-list')).toContainText('operator confirmed');
  expect(operations.find(operation => operation.action === 'review_room')?.evidence_id).toBe('room-evidence');
  await panel.getByRole('button', { name: 'Continue mapping', exact: true }).click();
  expect(operations.find(operation => operation.action === 'continue_mapping')?.map_id).toBe('sensor-map');
  for (const width of [1440, 390, 320]) {
    await page.setViewportSize({ width, height: 1000 });
    expect(await panel.evaluate(element => element.scrollWidth <= element.clientWidth)).toBe(true);
    await panel.screenshot({ path: `.runtime/room-review-${width}.png` });
  }
  await page.reload();
  await panel.locator(':scope > summary').click();
  await expect(panel.getByRole('textbox', { name: 'Map name', exact: true })).toHaveValue('Reusable home');
  await panel.getByRole('textbox', { name: 'Map name', exact: true }).fill('Draft renamed home');
  await expect.poll(() => page.evaluate(async () => {
    const response = await fetch('/api/home');
    const value = await response.json();
    return value.name;
  })).toBe('Reusable home');
  await expect(panel.getByRole('textbox', { name: 'Map name', exact: true })).toHaveValue('Draft renamed home');
});