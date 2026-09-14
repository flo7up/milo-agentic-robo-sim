import {mkdir, readFile, readdir, writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {resolve} from 'node:path';

const base = process.argv[2] ?? 'http://127.0.0.1:8013';
const output = resolve(process.argv[3] ?? `.runtime/nav2-smoke-${Date.now()}`);
await mkdir(output, {recursive:false});
const call = async (path, body) => {
  const response = await fetch(`${base}/api/${path}`, {signal:AbortSignal.timeout(5000), ...(body === undefined ? {} : {
    method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body),
  })});
  if (!response.ok) throw new Error(`${path}: ${response.status} ${await response.text()}`);
  return response.json();
};
const initial = await call('state');
if (initial.agent.active || initial.busy) throw new Error('An existing controller is active; do not interrupt it');
const root = resolve(import.meta.dirname, '..');
const sourceFiles = [...(await readdir(resolve(root, 'backend'))).filter(name=>name.endsWith('.py')).map(name=>`backend/${name}`),
  ...(await readdir(resolve(root, 'ros'))).filter(name=>name.endsWith('.py') || name.endsWith('.yaml') || name==='Dockerfile').map(name=>`ros/${name}`),
  'assets/milo.urdf', 'pyproject.toml', 'start.ps1', 'scripts/check_nav2.mjs'];
const sources = async()=> Object.fromEntries(await Promise.all(sourceFiles.sort().map(async path=>
  [path,createHash('sha256').update(await readFile(resolve(root,path))).digest('hex')])));
const socket = new WebSocket(`${base.replace(/^http/, 'ws')}/api/live`);
await new Promise((resolveReady, reject) => {
  socket.addEventListener('message', resolveReady, {once:true});
  socket.addEventListener('error', reject, {once:true});
});
const started = Date.now();
const report = {evidence:'real_nav2_scripted_goals', started_at:new Date().toISOString(), run_id:initial.run_id,
  design:'nav2-supervised-v1-integration', rendering:initial.rendering, challenge:initial.challenge,
  episode_epoch:initial.episode_epoch, source_before:await sources(), goal_results:[], success:false};
const until = async (predicate, budgetMs) => {
  const deadline = Date.now() + budgetMs;
  while (Date.now() < deadline) {
    const result = await predicate();
    if (result) return result;
    await new Promise(resolveWait => setTimeout(resolveWait, 200));
  }
  throw new Error('Bounded Nav2 smoke check timed out');
};
try {
  await until(async()=> (await call('ros/status')).ready, 90000);
  for (const delta of [.7, .35]) {
    const state = await call('state');
    if (state.run_id !== initial.run_id || state.agent.active || state.busy) throw new Error('Test authority changed');
    if (state.stopped) await call('resume', {});
    const sensor = await call('ros/sensors');
    if (sensor.run_id !== initial.run_id || sensor.episode_epoch !== initial.episode_epoch) throw new Error('Sensor episode changed');
    const pose = sensor.odometry_m_rad;
    const goal = await call('ros/start', {run_id:state.run_id, episode_epoch:state.episode_epoch,
      target_m_rad:[pose[0] + delta * Math.cos(pose[2]), pose[1] + delta * Math.sin(pose[2]), pose[2]]});
    const samples = [];
    const entry = {session:goal, samples};
    report.goal_results.push(entry);
    const final = await until(async()=> {
      const current = await call('state');
      if (current.run_id !== initial.run_id) throw new Error('Episode changed during smoke check');
      samples.push({wall_s:(Date.now()-started)/1000, simulated_s:current.snapshot.simulated_time_s,
        navigation:current.ros_navigation, runtime:current.navigation, pose:current.snapshot.robot, collisions:current.proximity.collisions});
      return current.ros_navigation?.session_id === goal.session_id && current.ros_navigation.status !== 'running' ? current : null;
    }, 65000);
    const finalSensor = await call('ros/sensors');
    Object.assign(entry, {result:final.ros_navigation, odometry:finalSensor.odometry_m_rad,
      travel_m:final.navigation.travel_m});
    if (final.ros_navigation.status !== 'arrived') throw new Error(final.ros_navigation.reason);
    if (samples.some(sample=>sample.collisions.length)) throw new Error('Sampled contact during navigation');
  }
  const sensor = await call('ros/sensors');
  const pose = sensor.odometry_m_rad;
  const interrupted = await call('ros/start', {run_id:initial.run_id, episode_epoch:initial.episode_epoch,
    target_m_rad:[pose[0]+.7*Math.cos(pose[2]),pose[1]+.7*Math.sin(pose[2]),pose[2]]});
  const moving = await until(async()=> {
    const current = await call('state');
    if (current.ros_navigation?.session_id !== interrupted.session_id || current.ros_navigation.status !== 'running') {
      throw new Error('Interrupt test goal ended before moving');
    }
    return current.navigation.travel_m > .02 ? current : null;
  }, 15000);
  await call('stop', {});
  const stopped = await until(async()=> {
    const current = await call('state');
    return current.stopped && !current.busy && current.ros_navigation.status !== 'running' ? current : null;
  }, 5000);
  const rejected = async()=> {
    const response = await fetch(`${base}/api/ros/velocity`, {method:'POST', signal:AbortSignal.timeout(5000),
      headers:{'Content-Type':'application/json'}, body:JSON.stringify({session_id:interrupted.session_id,
        sensor_sequence:sensor.sequence,command_sequence:moving.ros_navigation.command_sequence+100,
        linear_mps:.1,angular_radps:0})});
    if (response.status !== 409) throw new Error(`Old command was not rejected: ${response.status}`);
    return response.status;
  };
  const afterStop = await rejected();
  for (let sample=0;sample<4;sample++) {
    await new Promise(resolveWait=>setTimeout(resolveWait,200));
    const current = await call('state');
    if (current.snapshot.simulated_time_s !== stopped.snapshot.simulated_time_s) throw new Error('Physics advanced after Stop');
  }
  await call('resume', {});
  const afterResume = await rejected();
  report.interruption = {session_id:interrupted.session_id, status:stopped.ros_navigation.status,
    after_stop_http:afterStop,after_resume_http:afterResume,frozen_simulated_s:stopped.snapshot.simulated_time_s};
  report.source_after = await sources();
  report.source_unchanged = JSON.stringify(report.source_before) === JSON.stringify(report.source_after);
  if (!report.source_unchanged) throw new Error('Source changed during the integration check');
  report.success = true;
} catch (error) {
  report.error = String(error);
  process.exitCode = 1;
} finally {
  try { await call('stop', {}); }
  catch (error) { report.cleanup_error=String(error); report.success=false; process.exitCode=1; }
  socket.close();
  report.finished_at = new Date().toISOString();
  await writeFile(resolve(output, 'report.json'), JSON.stringify(report, null, 2));
}
const {source_before,source_after,...summary} = report;
console.log(JSON.stringify({...summary, goal_results:report.goal_results.map(({samples,...result})=>({...result,samples:samples.length}))}, null, 2));