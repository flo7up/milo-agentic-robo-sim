import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { Activity, ArrowDown, ArrowUp, Battery, BatteryCharging, Camera, Check, CircleStop, Crosshair, Eye, Hand, LoaderCircle, Mic, Move3D, Pause, Play, RotateCcw, RotateCw, Settings2, Timer, TriangleAlert, Volume2, Wifi, WifiOff } from 'lucide-react';
import { Spectator } from './Spectator';
import { HeadCamera } from './HeadCamera';
import { AgentControl } from './AgentControl';
import { ChallengePicker } from './ChallengePicker';
import { OutcomeFeedback } from './OutcomeFeedback';
import type { LiveState, ManualPlacement, ProximitySensors, Result } from './types';
import './style.css';

export async function api(path: string, body?: unknown) {
  const response = await fetch(`/api/${path}`, body === undefined ? undefined : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(typeof data.detail === 'string' ? data.detail : `Request failed (${response.status})`);
  }
  return response.json();
}

function IconButton({ title, children, onClick, disabled = false, danger = false }: { title: string; children: React.ReactNode; onClick: () => void; disabled?: boolean; danger?: boolean }) {
  return <button className={`icon-button ${danger ? 'danger' : ''}`} title={title} aria-label={title} onClick={onClick} disabled={disabled}>{children}</button>;
}

function NumberInput({ label, value, set, min, max, step = .01 }: { label: string; value: number; set: (value: number) => void; min: number; max: number; step?: number }) {
  return <label className="number-field"><span>{label}</span><input type="number" min={min} max={max} step={step} value={value} onChange={event => set(Number(event.target.value))} /></label>;
}

const tokenFormat = new Intl.NumberFormat('en-US');

function TokenCounter({ agent, connected }: { agent: LiveState['agent'] | undefined; connected: boolean }) {
  const counts = [
    { label: 'Total', value: (agent?.input_tokens ?? 0) + (agent?.output_tokens ?? 0) },
    { label: 'Input', value: agent?.input_tokens ?? 0 },
    { label: 'Output', value: agent?.output_tokens ?? 0 },
  ];
  const scope = !connected ? 'Last received' : !agent?.session_id ? 'No run yet' : agent.phase === 'sleeping' ? 'Idle run' : agent.active || agent.auto_wake ? 'Current run' : 'Last run';
  return <div className="token-tracker" role="group" aria-label="Token usage"
    title="Provider-reported tokens for this text or voice run. Updates after each model response; interrupted or unreported requests may be missing. Resets on a new run or episode.">
    <div className="token-heading"><strong>Reported tokens</strong><span className="token-scope">{scope}</span></div>
    <dl className="token-counts">{counts.map(({ label, value }) => <div className={`token-${label.toLowerCase()}`} key={label}>
      <dt>{label}</dt><dd className={value >= 1000000 ? 'token-large' : ''}>{agent ? tokenFormat.format(value) : '--'}</dd>
    </div>)}</dl>
  </div>;
}

function RobotActivity({ state, connected }: { state: LiveState | null; connected: boolean }) {
  const phases: Record<string, { title: string; icon: typeof Activity }> = {
    starting: { title: 'Starting controller', icon: LoaderCircle },
    thinking: { title: 'Model thinking', icon: LoaderCircle },
    waiting: { title: 'Waiting for feedback', icon: Timer },
    acting: { title: 'Preparing command', icon: Timer },
    voice_ready: { title: 'Voice ready', icon: Mic },
    listening: { title: 'Listening', icon: Mic },
    speaking: { title: 'Milo speaking', icon: Volume2 },
  };
  let activity = { kind: 'idle', title: 'Robot idle', detail: 'Manual control', icon: Pause };
  if (!connected || !state) {
    activity = { kind: 'offline', title: state ? 'Connection lost' : 'Connecting', detail: 'Robot state unavailable', icon: WifiOff };
  } else if (state.agent.auto_wake && !state.busy && state.agent.phase === 'completed') {
    activity = { kind: 'settling', title: state.agent.idle_reason ?? 'Task ended', detail: 'Waiting for a quiet camera', icon: Timer };
  } else if (state.stopped) {
    activity = { kind: 'stopped', title: state.busy ? 'Stopping robot' : 'Robot stopped', detail: state.busy ? 'Stop requested' : 'Motion disabled', icon: CircleStop };
  } else if (state.busy) {
    activity = { kind: 'running', title: 'Robot running', detail: `Executing command / ${state.agent.active ? state.agent.mode === 'voice' ? 'Voice control' : state.agent.mode === 'chat' ? 'Chat control' : 'LLM control' : 'Manual control'}`, icon: Activity };
  } else if (state.agent.phase === 'error') {
    activity = { kind: 'error', title: 'Control error', detail: 'Robot holding position', icon: TriangleAlert };
  } else if (state.agent.phase === 'sleeping') {
    activity = { kind: 'sleeping', title: 'Agent idle', detail: state.agent.auto_wake ? 'Watching camera / no inference' : 'Session limit reached / awaiting user', icon: Pause };
  } else if (state.agent.phase === 'waking') {
    activity = { kind: 'waking', title: 'Agent waking', detail: state.agent.wake_reason ?? 'New interaction', icon: Activity };
  } else if (state.agent.active) {
    activity = { kind: state.agent.phase, ...(phases[state.agent.phase] ?? phases.starting), detail: 'Robot holding position' };
  } else if (state.agent.phase === 'completed') {
    activity = { kind: 'completed', title: 'Run finished', detail: 'Robot holding position', icon: Check };
  }
  const StatusIcon = activity.icon;
  return <section className="robot-activity" data-state={activity.kind} aria-label="Robot activity">
    <div className="activity-summary" role="status" aria-live="polite" aria-atomic="true">
      <StatusIcon className="activity-icon" size={24} aria-hidden="true" />
      <div><strong className="status">{activity.title}</strong><span className="activity-detail">{activity.detail}</span></div>
    </div>
    <div className="activity-readouts">
      <TokenCounter agent={state?.agent} connected={connected} />
      <div className="activity-clock" aria-label="Simulation time"><strong>{connected && state ? state.snapshot.simulated_time_s.toFixed(2) : '--'}</strong><span>s simulated</span></div>
    </div>
    {state && <OutcomeFeedback state={state} connected={connected} />}
  </section>;
}

function ProximityPanel({ sensors }: { sensors: ProximitySensors }) {
  const positions: Record<string, [number, number]> = { front_left: [1, 1], front: [2, 1], front_right: [3, 1], left: [1, 2],
    right: [3, 2], rear_left: [1, 3], rear: [2, 3], rear_right: [3, 3] };
  return <section className="proximity-panel" aria-label="Collision and distance sensors" data-simulated-time={sensors.simulated_time_s}
    title="Eight fixed beams at chassis level, measured from the sensor mount, up to 2 m. Narrow beams can miss obstacles between them. Readings do not steer or stop the robot.">
    <h3>Distance sensors <span className="tag">2 m range</span></h3>
    <dl className="distance-grid">
      <div className="sensor-robot" aria-hidden="true"><Move3D size={26} /></div>
      {sensors.distances.map(reading => <div key={reading.direction} className={`distance-reading ${reading.distance_m !== null && reading.distance_m < .25 ? 'near' : ''}`}
        style={{ gridColumn: positions[reading.direction]?.[0], gridRow: positions[reading.direction]?.[1] }}>
        <dt>{reading.direction.replaceAll('_', ' ')}</dt>
        <dd>{reading.status === 'occluded' ? 'Occluded' : reading.distance_m === null ? '>2.00 m' : `${reading.distance_m.toFixed(2)} m`}</dd>
      </div>)}
    </dl>
    <div className={`collision-status ${sensors.collisions.length ? 'contact' : ''}`} role="status">
      {sensors.collisions.length ? <TriangleAlert size={16} /> : <Check size={16} />}
      <span>{sensors.collisions.length ? 'Collision detected' : 'No collision'}</span>
    </div>
    {sensors.collisions.length > 0 && <ul className="collision-readings">{sensors.collisions.map(contact => <li key={contact.direction}>{contact.direction} <strong>{contact.force_n.toFixed(1)} N</strong></li>)}</ul>}
  </section>;
}

function App() {
  const [state, setState] = useState<LiveState | null>(null);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState('');
  const [pending, setPending] = useState(false);
  const [axes, setAxes] = useState(false);
  const [tab, setTab] = useState('drive');
  const [arm, setArm] = useState('left');
  const [yaw, setYaw] = useState(0);
  const [pitch, setPitch] = useState(.7);
  const [duration, setDuration] = useState(2);
  const [position, setPosition] = useState([.36, .25, -.12]);
  const [joints, setJoints] = useState([0, -.6, 1.8, 0, -1.2, 0]);
  const [opening, setOpening] = useState(.11);
  const [force, setForce] = useState(20);
  const [events, setEvents] = useState<{ tool: string; result: Result }[]>([]);
  const currentRun = useRef(state?.run_id);
  currentRun.current = state?.run_id;
  useEffect(() => { setEvents([]); setError(''); }, [state?.run_id]);
  useEffect(() => {
    let active = true;
    let socket: WebSocket;
    let reconnect: ReturnType<typeof setTimeout>;
    const connect = () => {
      socket = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/api/live`);
      socket.onmessage = event => { if (active) { setState(JSON.parse(event.data)); setConnected(true); } };
      socket.onclose = () => { if (active) { setConnected(false); reconnect = setTimeout(connect, 1500); } };
    };
    connect();
    return () => { active = false; clearTimeout(reconnect); socket.close(); };
  }, []);

  async function control(path: string) {
    setError('');
    try {
      await api(path, {});
      if (path === 'reset') setEvents([]);
    } catch (failure) { setError(String(failure)); }
  }

  async function execute(tool: string, arguments_: object = {}) {
    if (!state || state.agent.active) return;
    setPending(true);
    setError('');
    try {
      const result: Result = await api('command', { run_id: state.run_id, episode_epoch: state.episode_epoch, action_id: crypto.randomUUID(), observation_seq: state.observation.seq, tool, arguments: arguments_ });
      if (result.observation.run_id !== currentRun.current) return;
      setEvents(previous => [...previous.slice(-49), { tool, result }]);
      if (result.status !== 'ok') setError(`${result.error}: ${result.message}`);
    } catch (failure) { setError(String(failure)); }
    finally { setPending(false); }
  }

  async function placeRobot(placement: ManualPlacement) {
    if (!connected || pending || state?.busy || state?.stopped || state?.agent.active) throw new Error('Robot is not available for manual placement.');
    setPending(true);
    try {
      const placed: LiveState = await api('robot/placement', placement);
      setState(previous => previous?.run_id === placed.run_id && previous.observation.seq <= placed.observation.seq ? placed : previous);
    } finally { setPending(false); }
  }

  const disabled = !connected || pending || !!state?.busy || !!state?.stopped || !!state?.agent.active;
  const jointLimits = [1.8, 2.5, 2.7, 3, 2.5, 3];
  const observation = state?.observation;
  return <>
    <header className="topbar">
      <div className="brand"><span className="brand-mark"><Move3D size={26} /></span><div><h1>Milo <span>/ Embodied Lab</span></h1><p>Robot visual control research</p></div></div>
      <div className="connection">{connected ? <Wifi size={16} /> : <WifiOff size={16} />} {connected ? 'Local simulator' : 'Reconnecting'}</div>
    </header>
    <main>
      <section className="runbar">
        <div><span className="eyebrow">{state?.challenge ? 'PREDEFINED CHALLENGE' : 'MECHANICAL BENCH'}</span><h2>{state?.challenge?.title ?? 'Floor pickup & release'}</h2><p>{state?.challenge?.skill ?? 'Position either arm, close around the cube, lift, and release.'}</p></div>
        <div className="run-actions"><span className="tag">{state?.agent.active ? `${state.agent.mode === 'voice' ? 'Voice' : 'LLM'} / bounded tools` : state?.assisted ? 'Manual / assisted' : 'Manual ready'}</span><IconButton title="Resume manual control" disabled={!state?.stopped} onClick={() => control('resume')}><Play size={20} /></IconButton><IconButton title="Reset episode" onClick={() => control('reset')}><RotateCcw size={20} /></IconButton><button className="stop-button" onClick={() => control('stop')}><CircleStop size={18} /> Stop</button></div>
      </section>
      <div className="mode-strip"><span>{state?.agent.active ? state.agent.mode === 'voice' ? 'Voice control' : 'LLM control' : 'Manual control'}</span><span>step_locked</span><span>RGB + proprioception</span><span>assisted_constraint</span><span>1 / 240 s physics</span></div>
      {error && <div className="error" role="alert">{error}</div>}
      {state && <ChallengePicker state={state} connected={connected} request={api} />}
      <RobotActivity state={state} connected={connected} />
      <section className="view-grid">
        <div className="world-panel">
          <div className="panel-header"><h3><Move3D size={16} /> Spectator</h3><label className="toggle"><input type="checkbox" checked={axes} onChange={event => setAxes(event.target.checked)} /> Axes</label></div>
          {state ? <Spectator state={state} axes={axes} enabled={!disabled} onPlace={placeRobot} /> : <div className="loading">Connecting to physics worker...</div>}
          <div className="viewport-footer"><span>{state?.manual_placements ? `Manual placements: ${state.manual_placements}` : 'MILO-01'}</span><span>{state?.snapshot.simulated_time_s.toFixed(2) ?? '0.00'} s simulated</span><span>Epoch {state?.episode_epoch ?? '-'}</span></div>
        </div>
        <div className="camera-panel">
          <div className="panel-header"><h3><Camera size={16} /> Head camera</h3><span className="tag">AUTHORITATIVE RGB</span></div>
          {state && <HeadCamera key={state.run_id} frame={state.camera} />}
          <div className="sensors">
            <h3>Proprioception</h3>
            <div className="sensor-row"><span>Head yaw / pitch</span><strong>{observation?.head_rad.map(value => value.toFixed(2)).join(' / ') ?? '-'} rad</strong></div>
            <div className="sensor-row"><span>Encoder odometry</span><strong>{observation?.odometry_m_rad.slice(0, 2).map(value => value.toFixed(2)).join(', ') ?? '-'} m</strong></div>
            {['left', 'right'].map(side => <div className="gripper-reading" key={side}><span>{side} gripper</span><strong>{((observation?.grippers[side].aperture_m ?? 0) * 1000).toFixed(0)} mm</strong><span>{observation?.grippers[side].load_n.toFixed(2) ?? '-'} N</span><span title="Inner and outer finger contact" className="contacts">{observation?.grippers[side].contact.map((value, index) => <i key={index} className={value ? 'on' : ''} />)}</span></div>)}
            {state?.proximity && <ProximityPanel sensors={state.proximity} />}
            {observation?.battery && <div className={`battery-reading ${observation.battery.low ? 'battery-low' : ''}`} aria-label="Battery status">
              <span>{observation.battery.charging ? <BatteryCharging size={16} /> : <Battery size={16} />} Battery</span>
              <meter aria-label="Battery level" min={0} max={100} low={35} high={90} optimum={100} value={observation.battery.charge_pct} />
              <strong>{observation.battery.charge_pct.toFixed(1)}%</strong>
              <small>{observation.battery.charging ? 'Charging' : observation.battery.charge_pct === 0 ? 'Battery empty' : observation.battery.low ? 'Low battery / return to charger' : 'Battery ready'}</small>
            </div>}
          </div>
        </div>
      </section>
      {state && <AgentControl key={state.run_id} state={state} connected={connected} request={api} />}
      <section className="controls-section">
        <div className="panel-header"><h3><Settings2 size={16} /> Manual control</h3><NumberInput label="Duration (s)" value={duration} set={setDuration} min={.1} max={2} step={.1} /></div>
        <div className="tabs" role="tablist">{[['drive', 'Base', <Move3D size={16} />], ['head', 'Head', <Eye size={16} />], ['arms', 'Arms & grippers', <Hand size={16} />]].map(([key, text, icon]) => <button role="tab" aria-selected={tab === key} key={String(key)} onClick={() => setTab(String(key))}>{icon}{text}</button>)}</div>
        <div className="control-body">
          {tab === 'drive' && <div className="drive-controls"><div className="direction-pad"><IconButton title="Rotate left" disabled={disabled} onClick={() => execute('drive_base', { linear_mps: 0, angular_radps: .6, duration_s: duration })}><RotateCcw /></IconButton><IconButton title="Drive forward" disabled={disabled} onClick={() => execute('drive_base', { linear_mps: .2, angular_radps: 0, duration_s: duration })}><ArrowUp /></IconButton><IconButton title="Rotate right" disabled={disabled} onClick={() => execute('drive_base', { linear_mps: 0, angular_radps: -.6, duration_s: duration })}><RotateCw /></IconButton><span /><IconButton title="Drive backward" disabled={disabled} onClick={() => execute('drive_base', { linear_mps: -.2, angular_radps: 0, duration_s: duration })}><ArrowDown /></IconButton></div><div className="limits"><span>Linear command <strong>0.20 m/s</strong></span><span>Angular command <strong>0.60 rad/s</strong></span><span>Maximum duration <strong>2.00 s</strong></span></div><button onClick={() => execute('observe')} disabled={!connected || pending || !!state?.agent.active}><Camera size={16} /> Observe</button></div>}
          {tab === 'head' && <div className="head-controls"><label>Yaw <output>{yaw.toFixed(2)} rad</output><input aria-label="Head yaw" type="range" min={-1.5} max={1.5} step={.01} value={yaw} onChange={event => setYaw(Number(event.target.value))} /></label><label>Pitch <output>{pitch.toFixed(2)} rad</output><input aria-label="Head pitch" type="range" min={-.7} max={1.15} step={.01} value={pitch} onChange={event => setPitch(Number(event.target.value))} /></label><button disabled={disabled} className="primary" onClick={() => execute('set_head', { yaw_rad: yaw, pitch_rad: pitch, duration_s: duration })}><Eye size={16} /> Set head</button></div>}
          {tab === 'arms' && <>
            <div className="arm-selector"><label>Arm<select value={arm} onChange={event => { setArm(event.target.value); setPosition(previous => [previous[0], event.target.value === 'left' ? .25 : -.25, previous[2]]); }}>{['left', 'right'].map(side => <option key={side}>{side}</option>)}</select></label><span>Base frame / meters / xyzw</span></div>
            <div className="arm-grid"><div className="joint-controls">{joints.map((value, index) => <label key={index}><span>J{index + 1}</span><input aria-label={`Joint ${index + 1}`} type="range" min={-jointLimits[index]} max={jointLimits[index]} step={.01} value={value} onChange={event => setJoints(previous => previous.map((entry, entryIndex) => entryIndex === index ? Number(event.target.value) : entry))} /><output>{value.toFixed(2)} rad</output></label>)}<button disabled={disabled} onClick={() => execute('set_arm_joints', { arm, joint_positions_rad: joints, duration_s: duration })}><Settings2 size={16} /> Set joints</button></div><div className="cartesian-controls"><h4>Palm target</h4><div className="xyz-fields">{position.map((value, index) => <NumberInput key={index} label={`${['X', 'Y', 'Z'][index]} (m)`} min={-1} max={1} value={value} set={number => setPosition(previous => previous.map((entry, entryIndex) => index === entryIndex ? number : entry))} />)}</div><p className="readout">Orientation: [0, 0, 0, 1]</p><button className="primary" disabled={disabled} onClick={() => execute('move_end_effector', { arm, position_m: position, orientation_xyzw: [0, 0, 0, 1], frame: 'base', duration_s: duration })}><Crosshair size={16} /> Move palm</button></div><div className="gripper-controls"><h4>Parallel-jaw gripper</h4><NumberInput label="Opening (m)" min={0} max={.11} value={opening} set={setOpening} /><NumberInput label="Max force (N)" min={1} max={35} step={1} value={force} set={setForce} /><button disabled={disabled} onClick={() => execute('set_gripper', { arm, opening_m: opening, max_force_n: force })}><Hand size={16} /> Set gripper</button></div></div>
          </>}
        </div>
        <button className="wait-button" disabled={disabled} onClick={() => execute('wait', { duration_s: duration })}><Timer size={16} /> Wait</button>
      </section>
      <section className="timeline"><div className="panel-header"><h3>Action timeline</h3><span>{events.length} commands</span></div>{events.length ? <div className="event-list">{[...events].reverse().map(({ tool, result }) => <div className="event" key={result.action_id}><span className={result.status === 'ok' ? 'ok' : 'bad'}>{result.status}</span><strong>{tool}</strong><span>{result.actual_duration_s.toFixed(2)} s</span><span>Frame {result.observation.seq}</span><span>{result.error ?? 'Completed'}</span></div>)}</div> : <p className="empty">No commands recorded in this episode.</p>}</section>
      <footer>Prototype research simulator. Not a real-hardware safety validation.</footer>
    </main>
  </>;
}

createRoot(document.getElementById('root')!).render(<App />);