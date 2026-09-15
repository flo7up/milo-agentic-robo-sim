import { useEffect, useState } from 'react';
import { Bot, Check, CircleStop, Compass, Hand, Play, Power, Settings2, Timer, Send, MessageSquare, History, Radio } from 'lucide-react';
import { ExchangeFeed } from './ExchangeFeed';
import { LocalModelProgress } from './LocalModelProgress';
import type { LiveState, Reasoning } from './types';

export function LunaNavigationControl({ state, connected, request }: {
  state: LiveState; connected: boolean; request: (path: string, body?: unknown) => Promise<unknown>;
}) {
  const agent = state.agent;
  const kitchenSearch = state.challenge?.id === 'flat_kitchen';
  const circleFurniture = state.challenge?.id === 'furniture_circuit';
  const luna = agent.configuration.models.find(model => model.id === 'luna' && model.provider === 'foundry');
  const [goal, setGoal] = useState(state.challenge?.goal ?? 'Inspect the scene and navigate safely.');
  const [interval, setInterval] = useState(.25);
  const [turns, setTurns] = useState(80);
  const [reasoning, setReasoning] = useState<Reasoning>(luna?.reasoning_efforts.includes('high') ? 'high' : luna?.reasoning_efforts[0] ?? 'high');
  const [endpoint, setEndpoint] = useState(agent.configuration.endpoint);
  const [deployment, setDeployment] = useState(luna?.deployment ?? '');
  const [pending, setPending] = useState(false);
  const [error, setError] = useState('');
  const [controlMode, setControlMode] = useState<'task' | 'exploration'>('task');
  const [explorationBudget, setExplorationBudget] = useState(180);
  const [home, setHome] = useState<{map_id: string | null; name: string | null; revision: number;
    localization: {status: string}; expansion_allowed?: boolean; coverage: {free_m2: number} | null;
    task: {status: string; reason: string; local_exploration?: boolean; continuations?: number; visited_frontiers: number} | null} | null>(null);
  useEffect(() => {
    if (controlMode !== 'exploration' || !connected || pending) return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    async function refresh() {
      try {
        const response = await fetch('/api/home?compact=true', {signal:controller.signal});
        if (!response.ok) throw new Error('Local exploration state unavailable');
        const value = await response.json();
        if (!controller.signal.aborted) setHome(value);
      } catch (failure) { if (!controller.signal.aborted) setError(String(failure)); }
      finally { if (!controller.signal.aborted) timer = setTimeout(refresh, 1000); }
    }
    void refresh();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [controlMode, connected, pending, state.run_id]);
  const [instruction, setInstruction] = useState('');
  const [sending, setSending] = useState(false);
  const [instructionError, setInstructionError] = useState('');
  const [handoff, setHandoff] = useState(kitchenSearch);
  const [skillComposer, setSkillComposer] = useState(false);
  const [preferredBackend, setPreferredBackend] = useState<'nav2' | 'builtin'>(() =>
    localStorage.getItem('milo-navigation-backend') === 'builtin' ? 'builtin' : 'nav2');
  const [nav2, setNav2] = useState({enabled:false, ready:false, message:'Nav2 bridge is offline'});
  const [supportsNavigationBackend, setSupportsNavigationBackend] = useState(false);
  useEffect(() => { localStorage.setItem('milo-navigation-backend', preferredBackend); }, [preferredBackend]);
  const [aiRoutes, setAiRoutes] = useState(() => localStorage.getItem('milo-ai-generated-routes') === 'true');
  useEffect(() => { localStorage.setItem('milo-ai-generated-routes', String(aiRoutes)); }, [aiRoutes]);
  const [adaptive, setAdaptive] = useState(() => localStorage.getItem('milo-adaptive-navigation') !== 'false');
  useEffect(() => { localStorage.setItem('milo-adaptive-navigation', String(adaptive)); }, [adaptive]);
  const [inspector, setInspector] = useState('trace');
  const [connectionOpen, setConnectionOpen] = useState(!luna?.configured);
  const hasRun = !!agent.session_id || agent.active;
  useEffect(() => { setConnectionOpen(!luna?.configured); }, [luna?.configured]);
  const [navigationMode, setNavigationMode] = useState<'luna_continuous' | 'luna_navigation'>(() =>
    !kitchenSearch && !circleFurniture && localStorage.getItem('milo-navigation-mode') === 'luna_navigation' ? 'luna_navigation' : 'luna_continuous');
  useEffect(() => { localStorage.setItem('milo-navigation-mode', navigationMode); }, [navigationMode]);
  const continuous = (agent.active ? agent.execution_mode : navigationMode) === 'luna_continuous';
  const backend = continuous && supportsNavigationBackend && nav2.enabled ? preferredBackend : 'builtin';
  const [variant, setVariant] = useState<{name:string;version:string;revision:string} | null>(null);
  const [supportsAiRoutes, setSupportsAiRoutes] = useState(false);
  const [supportsSkillComposer, setSupportsSkillComposer] = useState(false);
  useEffect(() => {
    const controller = new AbortController();
    void fetch(`/api/test-variant?execution_mode=${navigationMode}&model_id=luna&reasoning=${reasoning}&skill_composer=${navigationMode === 'luna_continuous' && backend === 'builtin' && skillComposer}&navigation_backend=${backend}`, {signal:controller.signal})
      .then(async response => response.ok ? response.json() : null)
      .then(data => { if (!controller.signal.aborted) { setVariant(data?.architecture ?? null); setSupportsAiRoutes(data?.supports_ai_generated_routes === true); setSupportsSkillComposer(data?.supports_skill_composer === true); setSupportsNavigationBackend(data?.supports_navigation_backend === true); if (data?.nav2) setNav2(data.nav2); } })
      .catch(() => {});
    return () => controller.abort();
  }, [navigationMode, reasoning, luna?.deployment, skillComposer, backend]);
  useEffect(() => {
    if (!supportsNavigationBackend) return;
    const controller = new AbortController();
    const refresh = () => { void fetch('/api/ros/status', {signal:controller.signal}).then(response => response.ok ? response.json() : null)
      .then(data => { if (data && !controller.signal.aborted) setNav2(data); }).catch(() => {}); };
    refresh();
    const timer = window.setInterval(refresh, 2000);
    return () => { window.clearInterval(timer); controller.abort(); };
  }, [supportsNavigationBackend]);
  const resident = state.local_navigation_model;
  useEffect(() => { if (agent.active && agent.goal) setGoal(agent.goal); }, [agent.session_id, agent.goal, agent.active]);
  async function sendInstruction() {
    setSending(true); setInstructionError('');
    try {
      await request('agent/instruction', {run_id:state.run_id,episode_epoch:state.episode_epoch,
        session_id:agent.session_id,message:instruction});
      setInstruction('');
    } catch (failure) { setInstructionError(failure instanceof Error ? failure.message : String(failure)); }
    finally { setSending(false); }
  }
  useEffect(() => {
    setEndpoint(agent.configuration.endpoint);
    setDeployment(luna?.deployment ?? '');
    if (luna) setReasoning(current => luna.reasoning_efforts.includes(current) ? current : luna.reasoning_efforts[0]);
  }, [agent.configuration.endpoint, luna?.deployment, luna?.reasoning_efforts.join(',')]);
  async function perform(path: string, body: unknown) {
    setPending(true); setError('');
    try { await request(path, body); }
    catch (failure) { setError(failure instanceof Error ? failure.message : String(failure)); }
    finally { setPending(false); }
  }
  const residentLabels = { unloaded: continuous ? 'Model not loaded / SmolVLA inactive' : 'Model not loaded / loads on first Start', loading: 'Loading model into GPU memory',
    ready: 'Model ready / kept in GPU memory', inferencing: 'Model loaded / processing a request', error: 'Model worker unavailable / reloads on Start' };
  const exploring = home?.task?.status === 'running';
  const modeSelector = <div className="control-mode-selector" role="group" aria-label="Robot control mode">
    <button type="button" aria-pressed={controlMode === 'task'} disabled={pending || agent.active || state.busy || exploring} onClick={() => setControlMode('task')}><Bot size={16}/>Task / Luna</button>
    <button type="button" aria-pressed={controlMode === 'exploration'} disabled={pending || agent.active || state.busy} onClick={() => setControlMode('exploration')}><Compass size={16}/>Explore / local</button>
  </div>;
  if (controlMode === 'exploration') return <section className="agent-section local-exploration" aria-label="Local exploration">
    {modeSelector}
    <div className="panel-header"><h3><Compass size={17}/>Explore locally</h3><span className="tag">No model inference</span></div>
    <dl className="results-facts"><div><dt>Map</dt><dd>{home?.name ?? 'New unsaved sensor map'}</dd></div><div><dt>Localization</dt><dd>{home?.map_id ? home.localization.status : 'Starts at current robot pose'}</dd></div><div><dt>Observed free</dt><dd>{home?.coverage ? `${home.coverage.free_m2.toFixed(1)} m2` : 'Not measured'}</dd></div><div><dt>Model calls</dt><dd>0</dd></div></dl>
    <form onSubmit={event => { event.preventDefault(); setPending(true); setError(''); void (async () => {
      try {
        await request('continuous/scan', {run_id:state.run_id,episode_epoch:state.episode_epoch,compact_arms:true});
        const result = await request('home', {run_id:state.run_id,episode_epoch:state.episode_epoch,action:'start_exploration',time_budget:explorationBudget});
        setHome(result as NonNullable<typeof home>);
      } catch (failure) { setError(failure instanceof Error ? failure.message : String(failure)); }
      finally { setPending(false); }
    })(); }}>
      <label>Exploration budget (s)<input aria-label="Local exploration budget" type="number" min={1} max={300} value={explorationBudget} disabled={pending || exploring} onChange={event => setExplorationBudget(Number(event.target.value))}/></label>
      <div className="agent-actions"><button className="primary" type="submit" disabled={!connected || !home || pending || state.busy || state.stopped || agent.active || exploring || home.expansion_allowed === false || (home.map_id !== null && home.localization.status !== 'localized') || explorationBudget < 1 || explorationBudget > 300}><Play size={16}/>Start exploration</button>
        <button type="button" className="stop-button" disabled={!connected} onClick={() => void request('home', {run_id:state.run_id,episode_epoch:state.episode_epoch,action:'cancel_task'}).catch(failure => setError(String(failure)))}><CircleStop size={16}/>Stop exploration</button></div>
    </form>
    {state.stopped && <p role="status">Stopped / manual resume required</p>}
    {home?.map_id && home.localization.status !== 'localized' && <p role="status">Saved map loaded / localization required in Home map</p>}
    {home?.task?.local_exploration && <div className="home-task-status" role="status"><strong>{home.task.status}</strong><span>{home.task.reason}</span><small>{home.task.visited_frontiers} frontiers / {home.task.continuations ?? 0} rolling continuations</small></div>}
    {error && <p className="error" role="alert">{error}</p>}
  </section>;
  return <section className="agent-section" aria-label="LLM control">
    {modeSelector}
    <div className="panel-header"><h3><Bot size={17} /> Luna</h3>
      <span className="tag">{agent.active ? 'Supervisor active' : luna?.configured ? 'Ready for a goal' : 'Connection required'}</span></div>
    <details className="agent-connection setup-connection" hidden={agent.active} open={connectionOpen}
      onToggle={event => setConnectionOpen(event.currentTarget.open)}><summary><Settings2 size={15} /> Luna connection</summary>
      <form onSubmit={event => {
        event.preventDefault();
        const models = agent.configuration.models.map(({configured: _, ...model}) => model.id === 'luna'
          ? {...model, provider: 'foundry', deployment} : model);
        if (!models.some(model => model.id === 'luna')) models.push({id:'luna',label:'Luna',deployment,provider:'foundry',reasoning_efforts:['low','medium','high']});
        void perform('agent/config', {endpoint, ollama_endpoint:agent.configuration.ollama_endpoint, models});
      }}><fieldset disabled={!connected || agent.active || pending}>
        <label>Foundry endpoint<input aria-label="Foundry endpoint" type="url" value={endpoint} onChange={event => setEndpoint(event.target.value)} /></label>
        <label>Luna deployment<input aria-label="Luna deployment" value={deployment} onChange={event => setDeployment(event.target.value)} /></label>
        <button type="submit"><Check size={16} /> Apply Luna connection</button>
      </fieldset></form>
    </details>
    <div className="controller-caption" hidden={!luna?.configured && !hasRun}>{continuous ? (agent.active ? agent.navigation_backend : backend) === 'nav2' ? 'Nav2 navigation' : 'Continuous local navigation' : 'SmolVLA primitives'}<span className={luna?.configured ? 'ok' : 'bad'}>{luna?.configured ? 'Configured' : 'Connection required'}</span></div>
    {!continuous && agent.local_model && <LocalModelProgress live={agent.local_model} connected={connected} resident={resident} />}
    <form className="agent-form" onSubmit={event => {
      event.preventDefault();
      void perform('agent/start', { run_id: state.run_id, episode_epoch: state.episode_epoch,
        execution_mode: navigationMode, compact_arms: localStorage.getItem('milo-compact-arms') !== 'false',
        continuous_handoff: continuous && backend === 'builtin' && handoff,
        ...(continuous && supportsNavigationBackend ? {navigation_backend:backend} : {}),
        ...(continuous && backend === 'builtin' && skillComposer && supportsSkillComposer ? {skill_composer:true} : {}),
        ...(continuous && backend === 'builtin' && aiRoutes && supportsAiRoutes ? {ai_generated_routes:true} : {}),
        adaptive_navigation: adaptive,
        model_id: 'luna', reasoning, goal, max_turns: turns, feedback_interval_s: interval });
    }}>
      <div className="agent-goal-row"><label>Robot goal<textarea aria-label="Robot goal" value={goal} required maxLength={2000} rows={2}
        disabled={agent.active || pending} onChange={event => setGoal(event.target.value)} /></label>
        <div className="agent-actions" hidden={!luna?.configured && !hasRun}>{agent.active ? <>
          {!continuous && <button type="button" disabled={pending || !connected || interval < .25 || interval > 30}
            onClick={() => void perform('agent/rate', {feedback_interval_s: interval})}><Timer size={16} /> Apply rate</button>
          }
          <button type="button" disabled={pending || !connected} onClick={() => void perform('agent/takeover', {})}><Hand size={16} /> Take manual control</button>
        </> : <button type="submit" className="primary" disabled={!connected || pending || state.busy || !luna?.configured || !goal.trim()
          || (continuous && backend === 'builtin' && aiRoutes && !supportsAiRoutes) || (backend === 'nav2' && !nav2.ready)
          || !Number.isInteger(turns) || turns < 1 || turns > 80 || interval < .25 || interval > 30}>
          <Play size={16} /> Start LLM control</button>}</div></div>
    </form>
    {(error || agent.error) && <div className="error" role="alert">{error || agent.error}</div>}
    {agent.message && <p className="agent-message">{agent.message}</p>}
    {hasRun && <div className="agent-metrics"><span>Supervisor turns <strong>{agent.turns} / {agent.max_turns}</strong></span>
      <span>{continuous ? 'Route updates' : 'Local commands'} <strong>{continuous ? state.continuous_navigation?.updates ?? 0 : agent.local_model?.requests_completed ?? 0}</strong></span>
      <span>{continuous ? 'Luna inference' : 'Local inference'} <strong>{agent.inference_latency_s?.toFixed(2) ?? '-'} s</strong></span>
      <span>{agent.local_model?.instruction ?? ''}</span></div>}
    {hasRun && <div className="inspector-tabs tabs" role="tablist" aria-label="Robot inspector" onKeyDown={event => {
      const keys = ['conversation', 'trace', 'settings'];
      const index = keys.indexOf(inspector);
      const next = event.key === 'ArrowRight' ? (index + 1) % keys.length : event.key === 'ArrowLeft' ? (index + keys.length - 1) % keys.length : event.key === 'Home' ? 0 : event.key === 'End' ? keys.length - 1 : -1;
      if (next < 0) return;
      event.preventDefault(); setInspector(keys[next]);
      event.currentTarget.querySelectorAll<HTMLButtonElement>('[role="tab"]')[next].focus();
    }}>
      {([{id:'conversation',label:'Conversation',icon:MessageSquare},{id:'trace',label:'Trace',icon:Radio},{id:'settings',label:'Settings',icon:Settings2}]).map(({id,label,icon:Icon}) =>
        <button type="button" key={id} role="tab" id={`inspector-${id}`} aria-controls={`panel-${id}`} aria-selected={inspector === id}
          tabIndex={inspector === id ? 0 : -1} onClick={() => setInspector(id)}><Icon size={15} />{label}</button>)}
    </div>}
    <div id="panel-conversation" role="tabpanel" aria-labelledby="inspector-conversation" hidden={!hasRun || inspector !== 'conversation'}>
    <section className="chat-section" aria-label="Run chat">
      <div className="panel-header"><h3><MessageSquare size={17} />Run chat</h3>
        <span className="tag" role="status">{sending ? 'Updating instruction' : agent.active ? 'Run active' : 'Run stopped'}</span></div>
      <div className="chat-transcript" role="log" aria-label="Run conversation" tabIndex={0}>
        {!agent.run_messages?.length && <p className="empty">No messages in this run.</p>}
        {(agent.run_messages ?? []).map(message => <article key={message.id} className={`chat-message chat-${message.role}`}>
          <strong>{message.role === 'user' ? 'You' : message.source === 'model' ? 'Luna' : 'Controller'} / {message.status}</strong>
          <p>{message.text}</p></article>)}
      </div>
      <form className="chat-composer" hidden={!agent.active} onSubmit={event => {event.preventDefault();void sendInstruction();}}>
        <label>New instruction<textarea aria-label="New run instruction" value={instruction} rows={2} maxLength={2000}
          disabled={!connected || sending || !agent.active} onChange={event=>setInstruction(event.target.value)} /></label>
        <button type="submit" disabled={!connected || sending || !agent.active || !instruction.trim()}><Send size={16} />Send instruction</button>
      </form>
      {instructionError && <p className="error" role="alert">{instructionError}</p>}
    </section>
    </div>
    <div id="panel-trace" role="tabpanel" aria-labelledby="inspector-trace" hidden={!hasRun || inspector !== 'trace'}>
    <ExchangeFeed agent={agent} visible={hasRun && inspector === 'trace'} />
    <details className="run-memory" aria-label="Episode movement memory">
      <summary><History size={15} />Movement memory</summary>
      <div className="agent-metrics"><span>Route samples <strong>{agent.run_memory?.visited_positions_m.length ?? 0}</strong></span>
        <span>No-progress actions <strong>{agent.run_memory?.progress?.stagnant_actions ?? 0}</strong></span>
        <span>Recovery attempts <strong>{agent.run_memory?.progress?.recovery_attempts ?? 0} / {agent.run_memory?.progress?.recovery_limit ?? 2}</strong></span>
        <span>Headings inspected here <strong>{agent.run_memory?.inspected_heading_sectors_here.length ?? 0} / 24</strong></span>
        <span>Rotation near this position <strong>{((agent.run_memory?.rotation_without_translation_rad ?? 0)*180/Math.PI).toFixed(0)} deg</strong></span></div>
      <ol className="memory-actions">{(agent.run_memory?.recent_actions ?? []).map((action,index)=><li key={`${agent.run_memory?.revision}-${index}`}>
        <strong>{action.action}</strong><span>{action.status} / {action.distance_m.toFixed(2)} m / {(action.turn_rad*180/Math.PI).toFixed(0)} deg</span>
        {action.reason && <span>{action.reason}</span>}</li>)}</ol>
    </details>
    </div>
    <details className="run-options" open={hasRun || undefined} hidden={(!luna?.configured && !hasRun) || (hasRun && inspector !== 'settings')}>
    <summary hidden={hasRun}><Settings2 size={15} />Run settings</summary>
    <div id="panel-settings" role={hasRun ? 'tabpanel' : undefined} aria-labelledby={hasRun ? 'inspector-settings' : undefined}>
      {!agent.active && variant && <dl className="test-variant-preview" aria-label="Current architecture version"><div><dt>Architecture</dt><dd>{variant.name}</dd></div><div><dt>Version</dt><dd>{variant.version}+{variant.revision}</dd></div></dl>}
      <div className="agent-settings">
        <label>Navigation controller<select aria-label="Navigation controller" value={navigationMode} disabled={agent.active || pending}
          onChange={event => setNavigationMode(event.target.value as typeof navigationMode)}>
          <option value="luna_continuous">Continuous local control</option><option value="luna_navigation">SmolVLA primitives</option></select></label>
        {continuous && supportsNavigationBackend && <label>Navigation stack<select aria-label="Navigation stack" value={backend} disabled={agent.active || pending}
          onChange={event => setPreferredBackend(event.target.value as 'nav2' | 'builtin')}>
          <option value="nav2" disabled={!nav2.enabled}>Nav2 (primary)</option><option value="builtin">Built-in (backup)</option></select></label>}
        <label>Supervisor reasoning<select aria-label="Supervisor reasoning" value={reasoning} disabled={agent.active || pending}
          onChange={event => setReasoning(event.target.value as Reasoning)}>
          {(luna?.reasoning_efforts ?? ['low', 'medium', 'high']).map(effort => <option value={effort} key={effort}>{effort}</option>)}</select></label>
        <label>Supervisor turn limit<input aria-label="Supervisor turn limit" type="number" min={1} max={80} value={turns}
          disabled={agent.active || pending} onChange={event => setTurns(Number(event.target.value))} /></label>
        {!continuous && <label>Local feedback interval (s)<input aria-label="Feedback interval (s)" type="number" min={.25} max={30} step={.25}
          value={interval} onChange={event => setInterval(Number(event.target.value))} /></label>}
      </div>
      {continuous && backend === 'nav2' && <p role="status" aria-label="Nav2 readiness">{nav2.ready ? 'Nav2 ready' : nav2.message}</p>}
      {continuous && backend === 'builtin' && <label className="toggle handoff-toggle" title={supportsAiRoutes ? 'Experimental generic waypoint control' : 'This backend does not support generic AI routes. Uncheck to use observed continuous control.'}><input type="checkbox" aria-label="AI-generated routes" checked={aiRoutes}
        disabled={agent.active || pending || (!supportsAiRoutes && !aiRoutes)} onChange={event => setAiRoutes(event.target.checked)} />AI-generated routes ({supportsAiRoutes ? 'experimental' : 'unavailable'})</label>}
      {continuous && backend === 'builtin' && !aiRoutes && <label className="toggle handoff-toggle"><input type="checkbox" aria-label="Plan while moving" checked={handoff}
        disabled={agent.active || pending} onChange={event => setHandoff(event.target.checked)} />Plan while moving (experimental)</label>}
      {continuous && backend === 'builtin' && !aiRoutes && <label className="toggle handoff-toggle"><input type="checkbox" aria-label="Motion skill composer" checked={skillComposer}
        disabled={agent.active || pending || !supportsSkillComposer} onChange={event => setSkillComposer(event.target.checked)} />Motion skill composer (experimental)</label>}
      {continuous && !aiRoutes && <label className="toggle handoff-toggle"><input type="checkbox" aria-label="Adaptive exploration" checked={adaptive}
        disabled={agent.active || pending} onChange={event => setAdaptive(event.target.checked)} />Adaptive exploration</label>}
    {(!continuous || (resident && resident.phase !== 'unloaded')) && <div className="policy-readiness">
      <div className="panel-header"><h4>{resident?.checkpoint ?? 'Local navigation checkpoint'} / local CUDA</h4>
        <button type="button" className="icon-button" aria-label="Unload local model" title="Unload SmolVLA and release GPU memory"
          disabled={!connected || pending || agent.active || state.busy || !resident || resident.phase === 'unloaded'}
          onClick={() => void perform('local-navigation/unload', {})}><Power size={16} /></button></div>
      <p role="status" aria-label="Local model residency">{connected ? residentLabels[resident?.phase ?? 'unloaded'] : 'Model residency unavailable'}</p>
      {resident && <p className="context-note" aria-label="Local model process" title="Model loads in this backend process; scene resets do not reset this count.">
        Server {location.host} / Model loads: {resident.load_count}{resident.process_id ? ` / Process ${resident.process_id}` : ''}</p>}
      <p role="status" aria-label="Luna connection">{luna?.configured ? `Luna supervisor / ${luna.deployment}` : 'Luna connection required'}</p>
      <p className="context-note">{continuous ? 'Continuous local controller / SmolVLA inactive' : 'Recovery-trained checkpoint / SmolVLA active on Start'}</p>
    </div>}
    </div>
    </details>
  </section>;
}