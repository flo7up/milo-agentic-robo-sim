import {test, expect} from '@playwright/test';
import type {LiveState, NavigationDiagnostics} from '../src/types';

test.beforeEach(async ({request}) => {
  await request.post('/api/test/preferences/reset');
  await request.post('/api/challenges/load', {data:{challenge_id:'park',reuse_saved_map:false}});
  await request.post('/api/agent/config', {data:{endpoint:'https://test.openai.azure.com',models:[
    {id:'luna',label:'Scripted Luna',deployment:'scripted',reasoning_efforts:['low','medium','high']}]}});
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
  await page.locator('.run-options > summary').click();
  await page.locator('.mission-diagnostics > summary').click();
  await page.getByRole('combobox',{name:'Diagnostic execution profile',exact:true}).selectOption('local');
  await page.reload();
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
  const settings=page.locator('.run-options');
  await expect(settings).toHaveAttribute('open');
  await settings.locator(':scope > summary').click();
  await expect(page.getByRole('spinbutton',{name:'Mission budget',exact:true})).toBeHidden();
  await expect(composer).toHaveValue('Retain this unsent follow-up.');
  await page.getByRole('tab',{name:'Trace',exact:true}).click();
  await expect(composer).toBeVisible();
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
  await expect(composer).toBeHidden();
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
  await page.locator('.run-options > summary').click();
  await page.getByRole('spinbutton',{name:'Luna request limit',exact:true}).fill('3');
  await page.getByRole('spinbutton',{name:'Luna token threshold',exact:true}).fill('12000');
  await expect.poll(async()=>(await(await request.get('/api/preferences')).json()).preferences.max_model_tokens).toBe(12000);
  await page.reload();
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
  await page.locator('.home-mapping > summary').click();
  await page.locator('.spatial-section > summary').click();
  await expect(page.locator('button.stop-button')).toHaveCount(1);
  await page.locator('.home-mapping > summary').click();
  await page.locator('.spatial-section > summary').click();
  for(const width of [1440,390,320]) {
    await page.setViewportSize({width,height:1000});
    await expect(start).toHaveCount(1);
    await expect(stop).toHaveCount(1);
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
    await page.keyboard.press('Escape');
    await expect(dialog).not.toBeVisible();
    await expect(page.locator('.observatory > .command-bar')).toHaveCount(1);
  }
  await page.getByLabel('Robot options',{exact:true}).click();
  await page.getByRole('button',{name:'Enable manual control',exact:true}).click();
  await expect.poll(async()=>(await(await request.get('/api/state')).json()).stopped).toBe(false);
  const before=await(await request.get('/api/state')).json();
  await page.getByRole('button',{name:'Reset episode',exact:true}).click();
  await expect.poll(async()=>(await(await request.get('/api/state')).json()).run_id).not.toBe(before.run_id);
  await page.getByLabel('Robot options',{exact:true}).click();
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
  await page.getByRole('button',{name:'Start mission',exact:true}).click();
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
  expect(errors).toEqual([]);
});