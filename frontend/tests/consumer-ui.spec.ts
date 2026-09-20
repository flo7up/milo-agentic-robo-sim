import { test, expect } from '@playwright/test';
import type { LiveState, ExchangeFeed } from '../src/types';

test('consumer cockpit exposes model calls and progressively loads their evidence', async ({page,request}) => {
  await request.post('/api/test/preferences/reset');
  await request.post('/api/challenges/load',{data:{challenge_id:'park',reuse_saved_map:false}});
  const initial:LiveState=await(await request.get('/api/state')).json();
  const live:LiveState={...initial,agent:{...initial.agent,session_id:'consumer-fixture',active:true,phase:'thinking',model_id:'qwen',
    inference_latency_s:1.25,input_tokens:1400,output_tokens:100,trace_revision:2,
    inference_budget:{requests:2,max_requests:12,tokens:1500,max_tokens:100000,usage_unknown:false},
    task_supervision:{model_id:'luna',requests:1,max_requests:4,tokens:500,input_tokens:450,output_tokens:50,max_tokens:100000,usage_unknown:false,status:'ready',guidance:''}}};
  let publish:(value:LiveState)=>void=()=>{};
  await page.routeWebSocket('**/api/live',socket=>{publish=value=>socket.send(JSON.stringify(value));publish(live);});
  let traceReads=0;
  const feed:ExchangeFeed={session_id:'consumer-fixture',revision:2,first_id:1,capacity:200,events:[{
    id:2,turn:1,timestamp:1789916400,title:'Navigation decision',kind:'response',image_url:null,
    payload:{text:'The path ahead is clear. I will move toward the bay.',text_truncated:false,status:'completed',latency_s:1.25,
      input_tokens:1400,output_tokens:100,calls:[],calls_truncated:false,refusals:[]}}]};
  await page.route('**/api/agent/trace?*',route=>{traceReads++;return route.fulfill({json:feed});});
  const errors:string[]=[];page.on('pageerror',error=>errors.push(error.message));
  await page.goto('/');
  const activity=page.getByRole('region',{name:'Model activity',exact:true});
  const stats=activity.getByRole('group',{name:'Run statistics'});
  await expect(activity).toContainText('Waiting for a model response');
  await expect(stats.getByRole('button',{name:'Model calls: view details'})).toContainText('3');
  await expect(stats).toContainText('2,000');
  await expect(stats).toContainText('1.3 s');
  await expect(page.getByLabel('Live telemetry summary')).toBeHidden();
  await expect(page.getByRole('dialog',{name:'Robot configuration'})).toBeHidden();
  await expect(page.getByRole('button',{name:'Load scenario',exact:true})).toBeVisible();
  const controls=page.getByRole('group',{name:'Run controls'});
  await expect(controls.getByRole('button')).toHaveCount(3);
  expect(traceReads).toBe(0);
  for(const width of [1440,768,390,320]) {
    await page.setViewportSize({width,height:1000});
    await activity.scrollIntoViewIfNeeded();
    expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
    await expect(activity).toBeInViewport();
    await expect(page.getByRole('button',{name:'Stop',exact:true})).toBeInViewport();
    await page.screenshot({path:`../.runtime/consumer-cockpit-disclosure-v1/cockpit-${width}.png`,fullPage:true});
  }
  await activity.locator('.model-drilldown').click();
  await expect(page.getByRole('tab',{name:'Model calls',exact:true})).toHaveAttribute('aria-selected','true');
  await expect(page.getByRole('tab',{name:'Model calls',exact:true})).toBeFocused();
  await expect(page.getByText('Navigation decision',{exact:true})).toBeVisible();
  expect(traceReads).toBe(1);
  await page.getByRole('button',{name:'Reported tokens: view usage',exact:true}).click();
  await expect(page.getByRole('tabpanel',{name:'Details',exact:true})).toBeVisible();
  await expect(page.getByRole('region',{name:'Exchange feed',exact:true})).toBeHidden();
  publish({...live,agent:{...live.agent,active:false,phase:'stopped',trace_revision:3,
    inference_budget:{...live.agent.inference_budget!,usage_unknown:true}}});
  await expect(activity).toContainText('Run ended');
  await expect(activity).toContainText('Usage incomplete');
  expect(traceReads).toBe(1);
  feed.revision=3;
  feed.events.push({...feed.events[0],id:3,title:'Follow-up decision'});
  await activity.locator('.model-drilldown').click();
  await expect(page.getByText('Follow-up decision',{exact:true})).toBeVisible();
  expect(traceReads).toBe(2);
  await page.getByRole('button',{name:'Robot settings',exact:true}).click();
  const drawer=page.getByRole('dialog',{name:'Robot configuration',exact:true});
  expect(await drawer.evaluate(element=>element.getBoundingClientRect().left)).toBe(0);
  await page.keyboard.press('Escape');
  await expect(drawer).toBeHidden();
  await expect(page.getByRole('button',{name:'Robot settings',exact:true})).toBeFocused();
  expect(errors).toEqual([]);
});
