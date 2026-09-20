import React, { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { createRoot } from 'react-dom/client';
import { Activity, ArrowDown, ArrowUp, Battery, BatteryCharging, Camera, Check, CircleStop, Crosshair, Eye, Hand, LoaderCircle, Map, MessageSquare, Mic, Minus, Move3D, Pause, Play, Power, Radar, RotateCcw, RotateCw, Route, Settings2, Timer, TriangleAlert, Volume2, Wifi, WifiOff } from 'lucide-react';
import { Spectator } from './Spectator';
import { HeadCamera } from './HeadCamera';
import { SpatialSensing } from './SpatialSensing';
import { HomeMapping } from './HomeMapping';
import { LunaNavigationControl as AgentControl } from './LunaNavigationControl';
import { ChallengePicker } from './ChallengePicker';
import { RegressionControl } from './RegressionControl';
import { robotOutcome } from './OutcomeFeedback';
import { RobotControlSurface } from './RobotControlSurface';
import { ViewNavigation, type TestView } from './ViewNavigation';
import { PreferencesProvider, usePreference } from './Preferences';
import type { LiveState, ManualPlacement, ProximitySensors, Result, SpatialTelemetry } from './types';
import './style.css';

export async function api(path: string, body?: unknown) {
  const response = await fetch(`/api/${path}`, body === undefined ? undefined : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    const validation = Array.isArray(data.detail) ? data.detail.map((entry: {loc?:unknown;msg?:unknown}) => {
      const location = Array.isArray(entry.loc) ? entry.loc.filter(part => part !== 'body').join('.') : '';
      const message = typeof entry.msg === 'string' ? entry.msg : 'Invalid value';
      return location ? `${location}: ${message}` : message;
    }).join('; ') : null;
    throw new Error(typeof data.detail === 'string' ? data.detail : validation || `Request failed (${response.status})`);
  }
  return response.json();
}

function IconButton({ title, children, onClick, disabled = false, danger = false }: { title: string; children: React.ReactNode; onClick: () => void; disabled?: boolean; danger?: boolean }) {
  return <button className={`icon-button ${danger ? 'danger' : ''}`} title={title} aria-label={title} onClick={onClick} disabled={disabled}>{children}</button>;
}

function NumberInput({ label, value, set, min, max, step = .01 }: { label: string; value: number; set: (value: number) => void; min: number; max: number; step?: number }) {
  return <label className="number-field"><span>{label}</span><input type="number" min={min} max={max} step={step} value={value} onChange={event => set(Number(event.target.value))} /></label>;
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
  } else if (state.power?.on === false) {
    activity = { kind: 'off', title: 'Off', detail: 'Motion and sensing disabled', icon: Power };
  } else if (state.agent.auto_wake && !state.busy && state.agent.phase === 'completed') {
    activity = { kind: 'settling', title: state.agent.idle_reason ?? 'Task ended', detail: 'Waiting for a quiet camera', icon: Timer };
  } else if (state.stopped) {
    activity = { kind: 'stopped', title: state.busy ? 'Stopping robot' : 'Robot stopped', detail: state.busy ? 'Stop requested' : 'Motion disabled', icon: CircleStop };
  } else if (state.continuous_navigation?.status === 'running') {
    activity = { kind: 'running', title: 'Robot navigating', detail: `Continuous local control / ${state.continuous_navigation.remaining_m.toFixed(2)} m remaining`, icon: Activity };
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
  const reportedOutcome = state && connected && state.power?.on !== false && !state.busy && !state.agent.active ? robotOutcome(state, connected) : null;
  const outcome = reportedOutcome?.kind === 'stopped' && !state?.stopped ? null : reportedOutcome;
  if (outcome) activity = { kind: outcome.kind, title: outcome.title, detail: outcome.message, icon: outcome.kind === 'success' ? Check : ['warning', 'error'].includes(outcome.kind) ? TriangleAlert : CircleStop };
  else if (connected && state && state.power?.on !== false && !state.busy && !state.agent.active && !state.stopped && state.agent.phase !== 'error') {
    activity = { kind: 'idle', title: 'On / Idle', detail: 'Ready for a task', icon: Pause };
  }
  if (state?.agent.active && state.agent.mission && !state.stopped) activity.detail = `${state.agent.mission.plan?.target || state.agent.mission.phase.replaceAll('_', ' ')} / ${state.agent.mission.remaining_s.toFixed(0)} s remaining`;
  const StatusIcon = activity.icon;
  return <section className="robot-activity" data-state={activity.kind} aria-label="Robot activity">
    <div className="activity-summary" role="status" aria-label="Robot status" aria-live="polite" aria-atomic="true">
      <StatusIcon className="activity-icon" size={24} aria-hidden="true" />
      <div><strong className="status">{activity.title}</strong><span className="activity-detail">{activity.detail}</span></div>
    </div>
    <div className="activity-readouts">
      <div className="activity-clock" aria-label="Simulation time"><strong>{connected && state ? state.snapshot.simulated_time_s.toFixed(2) : '--'}</strong><span>s simulated</span></div>
    </div>
    {outcome && <small className="outcome-evidence">{outcome.evidence}</small>}
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

function App({ onNavigate, onLiveState }: { onNavigate: (view: TestView) => void; onLiveState: (state: LiveState | null) => void }) {
  const [state, setState] = useState<LiveState | null>(null);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState('');
  const [pending, setPending] = useState(false);
  const [sceneLoading, setSceneLoading] = useState(false);
  const [spatialHud, setSpatialHud] = useState<HTMLDivElement | null>(null);
  const [cameraMinimized, setCameraMinimized] = useState(true);
  const [liveReadoutsOpen, setLiveReadoutsOpen] = useState(true);
  const [spatialTelemetry, setSpatialTelemetry] = useState<SpatialTelemetry | null>(null);
  const [showMotionZones,setShowMotionZones]=useState(() => {
    try { return sessionStorage.getItem('milo-sensors-and-areas') === 'true'; }
    catch { return false; }
  });
  useEffect(() => {
    try { sessionStorage.setItem('milo-sensors-and-areas', String(showMotionZones)); }
    catch {}
  }, [showMotionZones]);
  const [showTravelledPath,setShowTravelledPath]=useState(() => {
    try { return sessionStorage.getItem('milo-travelled-path') !== 'false'; }
    catch { return true; }
  });
  useEffect(() => {
    try { sessionStorage.setItem('milo-travelled-path', String(showTravelledPath)); }
    catch {}
  }, [showTravelledPath]);
  const [powerPending, setPowerPending] = useState(false);
  const [commandHost, setCommandHost] = useState<HTMLDivElement | null>(null);
  const [settingsHost, setSettingsHost] = useState<HTMLDivElement | null>(null);
  const [controlSurface, setControlSurface] = useState<HTMLElement | null>(null);
  const [manualOpen, setManualOpen] = usePreference('manual_open', false);
  const [axes, setAxes] = usePreference('axes', false);
  const [tab, setTab] = usePreference('manual_tab', 'drive');
  const [arm, setArm] = usePreference('manual_arm', 'left');
  const [yaw, setYaw] = usePreference('head_yaw', 0);
  const [pitch, setPitch] = usePreference('head_pitch', .7);
  const [duration, setDuration] = usePreference('duration', 2);
  const [position, setPosition] = usePreference('position', [.36, .25, -.12]);
  const [joints, setJoints] = usePreference('joints', [0, -.6, 1.8, 0, -1.2, 0]);
  const [opening, setOpening] = usePreference('opening', .11);
  const [force, setForce] = usePreference('force', 20);
  const [events, setEvents] = useState<{ tool: string; result: Result }[]>([]);
  const currentRun = useRef(state?.run_id);
  currentRun.current = state?.run_id;
  useEffect(() => { onLiveState(state); }, [state, onLiveState]);
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

  async function togglePower() {
    if (!state?.power) return;
    setPowerPending(true); setError('');
    try {
      setState(await api('power', {run_id:state.run_id, episode_epoch:state.episode_epoch, on:!state.power.on}));
    } catch (failure) { setError(String(failure)); }
    finally { setPowerPending(false); }
  }
  const powered = state?.power?.on !== false;
  const disabled = !connected || !powered || powerPending || pending || sceneLoading || !!state?.busy || !!state?.stopped || !!state?.agent.active || !!state?.regression?.active;
  const jointLimits = [1.8, 2.5, 2.7, 3, 2.5, 3];
  const observation = state?.observation;
  const inferenceBudget = state?.agent.inference_budget;
  const reportedTokens = (state?.agent.input_tokens ?? 0) + (state?.agent.output_tokens ?? 0);
  const telemetryContent = <>
    <details className="sensor-panel compact-disclosure">
      <summary><Activity size={16} />Robot sensors</summary>
      <div className="sensors">
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
    </details>
    <div id="map">{state && <><HomeMapping key={`home-${state.run_id}`} runId={state.run_id} epoch={state.episode_epoch}
      connected={connected} busy={sceneLoading || state.busy || state.agent.active} stopped={state.stopped} request={api} />
      <SpatialSensing key={`spatial-${state.run_id}`} runId={state.run_id} epoch={state.episode_epoch}
      connected={connected} busy={sceneLoading || state.busy || state.agent.active} request={api} hudHost={spatialHud} onTelemetry={setSpatialTelemetry} showMotionZones={showMotionZones} /></>}</div>
  </>;
  const commandBar = <section className="command-bar" aria-label="Robot controls">
    <RobotActivity state={state} connected={connected} />
    <div className="run-actions">{state?.power && <div className="robot-power-control">
      <button className="icon-button" type="button" role="switch" aria-label="Robot power" aria-checked={state.power.on}
        title={state.power.on ? 'Turn robot off' : 'Turn robot on'} disabled={!connected || powerPending || sceneLoading}
        onClick={() => void togglePower()}><Power size={19} /></button></div>}
      <div ref={setCommandHost} className="mission-start" />
      <button className="stop-button" disabled={!connected} onClick={() => control('stop')}><CircleStop size={18} /> Stop</button>
      <div ref={setSettingsHost} className="mission-settings" />
      <button className="icon-button" type="button" title="Reset scene" aria-label="Reset episode"
        disabled={!connected || sceneLoading || powerPending} onClick={() => control('reset')}><RotateCcw size={18} /></button>
      <details className="robot-options"><summary title="Manual control options" aria-label="Robot options"><Hand size={19} /></summary><div>
        <button disabled={!connected || !powered || (!state?.stopped && !state?.agent.active)} onClick={() => control(state?.agent.active ? 'agent/takeover' : 'resume')}><Hand size={16} />{state?.agent.active ? 'Take manual control' : 'Enable manual control'}</button>
      </div></details></div>
    {inferenceBudget && <div className="command-budget" role="group" aria-label="Luna usage and limits"
      data-limit-reached={reportedTokens >= inferenceBudget.max_tokens}
      title="Provider-reported usage. The threshold is checked after each request, so a response may cross it.">
      <span>{!connected ? 'Last received' : state?.agent.active ? 'Current run' : 'Last run'}</span>
      <span>Tokens <strong>{reportedTokens.toLocaleString('en-US')} / {inferenceBudget.max_tokens.toLocaleString('en-US')}</strong></span>
      <span>Requests <strong>{inferenceBudget.requests} / {inferenceBudget.max_requests}</strong></span>
      {inferenceBudget.usage_unknown && <span>Usage incomplete</span>}
    </div>}
  </section>;
  return <RobotControlSurface.Provider value={setControlSurface}>
    <header className="topbar cockpit-header unified-header">
      <div className="brand"><span className="brand-mark"><Move3D size={26} /></span><div><h1>Milo <span>/ Robot observatory</span></h1></div></div>
      <ViewNavigation current="cockpit" onNavigate={onNavigate} />
    </header>
    <main className="observatory">
      <section className="runbar">
        <div className="run-context"><div><h2>{state?.challenge?.title ?? 'Practice bench'}</h2></div>
          {state?.regression?.active && <span className="tag" role="status">Baseline {(state.regression.current_index ?? 0)+1} / {state.regression.cases.length} · {state.regression.phase.replaceAll('_',' ')}</span>}
          {state && <ChallengePicker state={state} connected={connected} request={api} onLoadingChange={setSceneLoading} />}</div>
        <nav className="workspace-nav" aria-label="Workspace"><a href="#observe"><Eye size={15} />World</a><a href="#interact"><MessageSquare size={15} />Luna</a><a href="#controls" onClick={() => setManualOpen(true)}><Hand size={15} />Controls</a></nav>
      </section>
      {controlSurface ? createPortal(commandBar, controlSurface) : commandBar}
      {error && <div className="error" role="alert">{error}</div>}
      {state?.preference_error && <div className="error" role="alert">{state.preference_error}</div>}
      <div className="observatory-workspace" data-stage={state?.agent.session_id || state?.agent.active ? 'run' : 'setup'}>
      <div className="observation-column" id="observe">
      <section className="view-grid">
        <div className="world-panel">
          <div className="panel-header world-options"><div className="world-title-controls">
            <h3><Move3D size={16} /> World view <span className="tag">Operator only</span></h3>
            <label className="toggle sensor-overlay-toggle" title="Show sensor beams, robot footprint and observed clearance areas. Display only; does not enable sensing or authorize motion.">
              <Radar size={16} aria-hidden="true" />Sensors &amp; areas
              <input type="checkbox" role="switch" aria-label="Sensors and areas" checked={showMotionZones}
                onChange={event=>setShowMotionZones(event.target.checked)} />
            </label>
            <label className="toggle sensor-overlay-toggle" title="Blue ground trail of recent travel sampled in this browser view, not a planned route. Gaps mark missing observations or manual placement; resets with the scene or page reload.">
              <Route size={16} aria-hidden="true" />Travelled path
              <input type="checkbox" role="switch" aria-label="Travelled path" checked={showTravelledPath}
                onChange={event=>setShowTravelledPath(event.target.checked)} />
            </label></div>
            <label className="toggle"><input type="checkbox" checked={axes} onChange={event => setAxes(event.target.checked)} /> Axes</label></div>
          <div className="world-viewport">
            {state ? <Spectator state={state} axes={axes} enabled={!disabled} onPlace={placeRobot} showZones={showMotionZones} showTrail={showTravelledPath} telemetry={spatialTelemetry} connected={connected} /> : <div className="loading">Connecting to physics worker...</div>}
            <aside className="viewport-hud" aria-label="Robot sensor HUD">
              <section className="viewport-widget" aria-label="Head camera HUD" data-minimized={cameraMinimized}>
                <div className="viewport-widget-heading"><h3><Camera size={14} /> Head camera</h3></div>
                  <button className={cameraMinimized ? 'viewport-widget-restore' : 'icon-button viewport-widget-toggle'} type="button" aria-label={cameraMinimized ? 'Restore head camera' : 'Minimize head camera'}
                    title={cameraMinimized ? 'Restore head camera' : 'Minimize head camera'} aria-expanded={!cameraMinimized} aria-controls="head-camera-hud-content"
                    onClick={() => setCameraMinimized(value => !value)}>{!cameraMinimized && <Minus size={16} />}</button>
                <div id="head-camera-hud-content">
                  {state && <HeadCamera key={state.run_id} frame={state.camera} connected={connected && powered} compact={cameraMinimized} />}
                </div>
              </section>
              <div ref={setSpatialHud} />
            </aside>
          </div>
          <div className="viewport-footer"><span>{state?.manual_placements ? `Manual placements: ${state.manual_placements}` : 'MILO-01'}</span><span>Epoch {state?.episode_epoch ?? '-'}</span></div>
        </div>
      </section>
      {state && <RegressionControl state={state} connected={connected && !sceneLoading && !powerPending} request={api}/>}
      <details className="live-telemetry" open={liveReadoutsOpen} onToggle={event=>setLiveReadoutsOpen(event.currentTarget.open)}>
        <summary><Activity size={15} />Live telemetry <span>{!connected ? 'Disconnected / last received' : 'Latest received'}</span></summary>
        <dl aria-label="Live telemetry summary">
          <div><dt>Encoder position</dt><dd>{observation?.odometry_m_rad.slice(0,2).map(value=>value.toFixed(2)).join(', ') ?? '-'} m</dd></div>
          <div><dt>Motion buffer</dt><dd>{state?.navigation?.status ?? 'Idle'}</dd></div>
          <div><dt>Contacts</dt><dd>{state?.proximity?.collisions.length ?? '-'} <span>/ {state?.proximity?.simulated_time_s.toFixed(2) ?? '-'} s sim</span></dd></div>
          <div><dt>Battery</dt><dd>{observation?.battery ? `${observation.battery.charge_pct.toFixed(0)}%` : 'Not reported'}</dd></div>
        </dl>
      </details>
      </div>
      <aside className="interaction-column" id="interact" aria-label="Robot interaction">
        {state && <AgentControl key={state.run_id} state={state} connected={connected && !sceneLoading && !powerPending} request={api} commandHost={commandHost}
          settingsHost={settingsHost} telemetryContent={telemetryContent}
          spatialTelemetry={spatialTelemetry?.run_id === state.run_id && spatialTelemetry.episode_epoch === state.episode_epoch ? spatialTelemetry : null} />}
      </aside>
      </div>
      <div className="lab-tools" id="controls">
      <details className="manual-disclosure" open={manualOpen && !state?.agent.active} onToggle={event => {if (!state?.agent.active) setManualOpen(event.currentTarget.open);}}>
      <summary><Hand size={16} />Manual controls</summary>
      <section className="controls-section">
        <div className="panel-header"><h3><Settings2 size={16} /> Manual control</h3><NumberInput label="Duration (s)" value={duration} set={setDuration} min={.1} max={2} step={.1} /></div>
        <div className="tabs" role="tablist">{[['drive', 'Base', <Move3D size={16} />], ['head', 'Head', <Eye size={16} />], ['arms', 'Arms & grippers', <Hand size={16} />]].map(([key, text, icon]) => <button role="tab" aria-selected={tab === key} key={String(key)} onClick={() => setTab(String(key) as typeof tab)}>{icon}{text}</button>)}</div>
        <div className="control-body">
          {tab === 'drive' && <div className="drive-controls"><div className="direction-pad"><IconButton title="Rotate left" disabled={disabled} onClick={() => execute('drive_base', { linear_mps: 0, angular_radps: .6, duration_s: duration })}><RotateCcw /></IconButton><IconButton title="Drive forward" disabled={disabled} onClick={() => execute('drive_base', { linear_mps: .2, angular_radps: 0, duration_s: duration })}><ArrowUp /></IconButton><IconButton title="Rotate right" disabled={disabled} onClick={() => execute('drive_base', { linear_mps: 0, angular_radps: -.6, duration_s: duration })}><RotateCw /></IconButton><span /><IconButton title="Drive backward" disabled={disabled} onClick={() => execute('drive_base', { linear_mps: -.2, angular_radps: 0, duration_s: duration })}><ArrowDown /></IconButton></div><div className="limits"><span>Linear command <strong>0.20 m/s</strong></span><span>Angular command <strong>0.60 rad/s</strong></span><span>Maximum duration <strong>2.00 s</strong></span></div><button onClick={() => execute('observe')} disabled={!connected || pending || !!state?.agent.active}><Camera size={16} /> Observe</button></div>}
          {tab === 'head' && <div className="head-controls"><label>Yaw <output>{yaw.toFixed(2)} rad</output><input aria-label="Head yaw" type="range" min={-1.5} max={1.5} step={.01} value={yaw} onChange={event => setYaw(Number(event.target.value))} /></label><label>Pitch <output>{pitch.toFixed(2)} rad</output><input aria-label="Head pitch" type="range" min={-.7} max={1.15} step={.01} value={pitch} onChange={event => setPitch(Number(event.target.value))} /></label><button disabled={disabled} className="primary" onClick={() => execute('set_head', { yaw_rad: yaw, pitch_rad: pitch, duration_s: duration })}><Eye size={16} /> Set head</button></div>}
          {tab === 'arms' && <>
            <div className="arm-selector"><label>Arm<select value={arm} onChange={event => { setArm(event.target.value as typeof arm); setPosition(previous => [previous[0], event.target.value === 'left' ? .25 : -.25, previous[2]]); }}>{['left', 'right'].map(side => <option key={side}>{side}</option>)}</select></label><span>Base frame / meters / xyzw</span></div>
            <div className="arm-grid"><div className="joint-controls">{joints.map((value, index) => <label key={index}><span>J{index + 1}</span><input aria-label={`Joint ${index + 1}`} type="range" min={-jointLimits[index]} max={jointLimits[index]} step={.01} value={value} onChange={event => setJoints(previous => previous.map((entry, entryIndex) => entryIndex === index ? Number(event.target.value) : entry))} /><output>{value.toFixed(2)} rad</output></label>)}<button disabled={disabled} onClick={() => execute('set_arm_joints', { arm, joint_positions_rad: joints, duration_s: duration })}><Settings2 size={16} /> Set joints</button></div><div className="cartesian-controls"><h4>Palm target</h4><div className="xyz-fields">{position.map((value, index) => <NumberInput key={index} label={`${['X', 'Y', 'Z'][index]} (m)`} min={-1} max={1} value={value} set={number => setPosition(previous => previous.map((entry, entryIndex) => index === entryIndex ? number : entry))} />)}</div><p className="readout">Orientation: [0, 0, 0, 1]</p><button className="primary" disabled={disabled} onClick={() => execute('move_end_effector', { arm, position_m: position, orientation_xyzw: [0, 0, 0, 1], frame: 'base', duration_s: duration })}><Crosshair size={16} /> Move palm</button></div><div className="gripper-controls"><h4>Parallel-jaw gripper</h4><NumberInput label="Opening (m)" min={0} max={.11} value={opening} set={setOpening} /><NumberInput label="Max force (N)" min={1} max={35} step={1} value={force} set={setForce} /><button disabled={disabled} onClick={() => execute('set_gripper', { arm, opening_m: opening, max_force_n: force })}><Hand size={16} /> Set gripper</button></div></div>
          </>}
        </div>
        <button className="wait-button" disabled={disabled} onClick={() => execute('wait', { duration_s: duration })}><Timer size={16} /> Wait</button>
      </section>
      {events.length > 0 && <section className="timeline"><div className="panel-header"><h3>Action timeline</h3><span>{events.length} commands</span></div><div className="event-list">{[...events].reverse().map(({ tool, result }) => <div className="event" key={result.action_id}><span className={result.status === 'ok' ? 'ok' : 'bad'}>{result.status}</span><strong>{tool}</strong><span>{result.actual_duration_s.toFixed(2)} s</span><span>Frame {result.observation.seq}</span><span>{result.error ?? 'Completed'}</span></div>)}</div></section>}
      </details>
      </div>
      <footer>Prototype research simulator. Not a real-hardware safety validation.</footer>
    </main>
  </RobotControlSurface.Provider>;
}

const Archive = React.lazy(() => import('./TestResults').then(module => ({default:module.TestResults})));

function TestWorkspace() {
  const initialView = new URLSearchParams(location.search).get('view') === 'test-results' ? 'archive' : 'cockpit';
  const [view, setView] = useState<TestView>(initialView);
  const [cockpitOpened, setCockpitOpened] = useState(initialView === 'cockpit');
  const [archiveOpened, setArchiveOpened] = useState(initialView === 'archive');
  const [liveState, setLiveState] = useState<LiveState | null>(null);
  function selectView(next: TestView) {
    setView(next);
    if (next === 'cockpit') setCockpitOpened(true);
    else setArchiveOpened(true);
  }
  function navigate(next: TestView) {
    if (next !== view) {
      const url = new URL(location.href);
      if (next === 'archive') url.searchParams.set('view', 'test-results');
      else url.searchParams.delete('view');
      history.pushState(null, '', `${url.pathname}${url.search}${url.hash}`);
      selectView(next);
    }
  }
  useEffect(() => {
    const onPopState = () => selectView(new URLSearchParams(location.search).get('view') === 'test-results' ? 'archive' : 'cockpit');
    window.addEventListener('popstate', onPopState);
    return () => window.removeEventListener('popstate', onPopState);
  }, []);
  useEffect(() => { document.title = view === 'archive' ? 'Milo | Test results' : 'Milo | Embodied Robot Lab'; }, [view]);
  return <>
    {cockpitOpened && <div hidden={view !== 'cockpit'} data-test-view="cockpit"><App onNavigate={navigate} onLiveState={setLiveState} /></div>}
    {archiveOpened && <div hidden={view !== 'archive'} data-test-view="archive"><React.Suspense fallback={<p role="status">Loading test archive...</p>}>
      <Archive onNavigate={navigate} liveState={liveState} onStop={() => api('stop', {})} />
    </React.Suspense></div>}
  </>;
}

if (new URLSearchParams(location.search).has('cameraWorker')) {
  void import('./gpuCamera').then(({ installGpuCamera }) => installGpuCamera());
} else {
  createRoot(document.getElementById('root')!).render(<PreferencesProvider><TestWorkspace /></PreferencesProvider>);
}