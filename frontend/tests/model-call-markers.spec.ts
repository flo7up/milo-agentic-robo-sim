import { test, expect } from '@playwright/test';
import type { LiveState, ModelCall } from '../src/types';

test('model call dots track dispatch positions and link both ways with chat across reset and resize', async ({page,request})=>{
  await request.post('/api/test/preferences/reset');
  const loaded=await request.post('/api/challenges/load',{data:{challenge_id:'bench'}});
  expect(loaded.ok()).toBe(true);
  const initial:LiveState=await(await request.get('/api/state')).json();
  const base=initial.snapshot.poses.find(pose=>pose.key===`${initial.robot_body_id}:-1`)!;
  let live:LiveState={...initial,agent:{...initial.agent,session_id:'call-locations',phase:'acting',active:true,goal:'Explore the observed area.'}};
  let publish:(value:LiveState)=>void=()=>{};
  await page.routeWebSocket('**/api/live',socket=>{publish=value=>{live=value;socket.send(JSON.stringify(value));};publish(live);});
  const commands:string[]=[],errors:string[]=[];
  page.on('request',message=>{if(message.method()==='POST' && !message.url().endsWith('/api/preferences')) commands.push(message.url());});
  page.on('pageerror',error=>errors.push(error.message));
  await page.goto('/');
  const canvas=page.locator('.spectator canvas');
  await expect(canvas).toBeVisible();
  const move=async(x:number,y:number)=>{
    publish({...live,snapshot:{...live.snapshot,simulated_time_s:live.snapshot.simulated_time_s+.1,
      poses:live.snapshot.poses.map(pose=>pose.key.startsWith(`${live.robot_body_id}:`)?{...pose,position:[pose.position[0]+x,pose.position[1]+y,pose.position[2]]}:pose)}});
    await expect(page.getByLabel('Simulation time')).toContainText(live.snapshot.simulated_time_s.toFixed(2));
  };
  const calls:ModelCall[]=[];
  for(let index=0;index<6;index++) {
    if(index>1) for(let step=0;step<5;step++) await move(-.04,0);
    const pose=live.snapshot.poses.find(pose=>pose.key===`${live.robot_body_id}:-1`)!;
    calls.push({id:`call-locations:${index+1}`,number:index+1,session_id:'call-locations',run_id:live.run_id,episode_epoch:live.episode_epoch,
      model:index===0?'Luna':'Qwen',kind:index===0?'task_supervision':'decision',timestamp:1000+index,simulated_time_s:live.snapshot.simulated_time_s,
      position_world_m:[pose.position[0],pose.position[1]],status:index===5?'thinking':'responded',summary:index===5?'Waiting for model response.':`Explore opening ${index+1}.`});
    publish({...live,agent:{...live.agent,model_calls:[...calls],run_messages:index===5?[{id:'reason-5',role:'assistant',source:'model',status:'reported',text:'Continue past the junction.',model_call_id:calls[4].id}]:[]}});
    await expect(page.locator('.model-call-marker')).toHaveCount(index+1);
  }
  const first=page.getByRole('button',{name:/^Model call 1:/});
  const second=page.getByRole('button',{name:/^Model call 2:/});
  const firstBox=await first.boundingBox(),secondBox=await second.boundingBox();
  expect(firstBox!.width).toBeCloseTo(firstBox!.height,0);
  expect(Math.abs(firstBox!.x-secondBox!.x)>25 || Math.abs(firstBox!.y-secondBox!.y)>25).toBe(true);
  const anchors=await page.locator('.model-call-markers circle').evaluateAll(nodes=>nodes.map(node=>[node.getAttribute('cx'),node.getAttribute('cy')]));
  expect(anchors[0]).toEqual(anchors[1]);
  await move(-.04,0);
  expect(await page.locator('.model-call-markers circle').evaluateAll(nodes=>nodes.map(node=>[node.getAttribute('cx'),node.getAttribute('cy')]))).toEqual(anchors);
  const chat=page.getByRole('region',{name:'Run chat',exact:true});
  await expect(chat.locator('[data-model-call]')).toHaveCount(6);
  await expect(chat.locator('[data-model-call="5"]')).toContainText('Continue past the junction.');
  await expect(chat.locator('[data-model-call="6"]')).toContainText('Waiting for model response.');
  await page.getByRole('button',{name:'Hide assistant details'}).click();
  await first.click();
  await expect(chat).toBeVisible();
  await expect(chat.locator('[data-model-call="1"]')).toBeFocused();
  await expect(first).toHaveAttribute('aria-pressed','true');
  const toggle=page.getByRole('switch',{name:'Travelled path',exact:true});
  await toggle.uncheck();await expect(first).toBeHidden();
  await chat.getByRole('button',{name:'Show model call 5 on path',exact:true}).click();
  await expect(toggle).toBeChecked();
  await expect(page.getByRole('button',{name:/^Model call 5:/})).toHaveAttribute('aria-pressed','true');
  for(const width of [1440,390]) {
    await page.setViewportSize({width,height:1000});
    await page.locator('.world-panel').scrollIntoViewIfNeeded();
    await expect(page.getByRole('button',{name:/^Model call 5:/})).toBeVisible();
    expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
    await page.locator('.world-panel').screenshot({path:`../.runtime/model-call-markers-v1/path-${width}.png`});
  }
  await page.setViewportSize({width:1440,height:1000});
  await chat.screenshot({path:'../.runtime/model-call-markers-v1/chat.png'});
  await page.reload();
  await expect(page.locator('.model-call-marker')).toHaveCount(6);
  publish({...live,episode_epoch:live.episode_epoch+1});
  await expect(page.locator('.model-call-marker')).toHaveCount(0);
  await expect(chat.locator('[data-model-call]')).toHaveCount(0);
  publish({...live,agent:{...live.agent,session_id:'new-session'}});
  await expect(page.locator('.model-call-marker')).toHaveCount(0);
  expect(commands).toEqual([]);expect(errors).toEqual([]);
  const after:LiveState=await(await request.get('/api/state')).json();
  expect(after.snapshot).toEqual(initial.snapshot);
  expect(base.position).toEqual(initial.snapshot.poses.find(pose=>pose.key===`${initial.robot_body_id}:-1`)!.position);
});
