import {test, expect} from '@playwright/test';

test.beforeEach(async ({request}) => {
  await request.post('/api/test/preferences/reset');
  await request.post('/api/challenges/load', {data:{challenge_id:'park',reuse_saved_map:false}});
  await request.post('/api/agent/config', {data:{endpoint:'https://test.openai.azure.com',models:[
    {id:'luna',label:'Scripted Luna',deployment:'scripted',reasoning_efforts:['low','medium','high']}]}});
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