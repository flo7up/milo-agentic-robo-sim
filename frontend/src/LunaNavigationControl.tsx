import { useEffect, useState } from 'react';
import { createPortal } from 'react-dom';
import { Bot, Check, ChevronRight, Play, Power, Route, Settings2, Timer, Send, MessageSquare, History, Radio } from 'lucide-react';
import { ExchangeFeed } from './ExchangeFeed';
import { MotionDiagnostics } from './MotionDiagnostics';
import { LocalModelProgress } from './LocalModelProgress';
import { usePreference } from './Preferences';
import type { LiveState, Reasoning, SpatialTelemetry } from './types';

function normalizedEndpoint(value: string) {
  try {
    const endpoint = new URL(value.trim());
    const path = endpoint.pathname.replace(/\/+$/, '').replace(/^\/openai\/v1$/, '');
    return `${endpoint.origin}${path}${endpoint.search}${endpoint.hash}`;
  } catch { return value.trim(); }
}

function TokenCounter({ agent, connected }: { agent: LiveState['agent']; connected: boolean }) {
  const counts = [
    {label:'Total',value:agent.input_tokens + agent.output_tokens},
    {label:'Input',value:agent.input_tokens},
    {label:'Output',value:agent.output_tokens},
  ];
  return <div className="token-tracker" role="group" aria-label="Token usage"
    title="Provider-reported tokens for this run; interrupted or unreported requests may be missing. Resets on a new run or episode.">
    <div className="token-heading"><strong>Reported tokens</strong><span className="token-scope">{!connected ? 'Last received' : agent.active ? 'Current run' : 'Last run'}</span></div>
    <dl className="token-counts">{counts.map(({label,value})=><div className={`token-${label.toLowerCase()}`} key={label}>
      <dt>{label}</dt><dd className={value>=1000000 ? 'token-large' : ''}>{value.toLocaleString('en-US')}</dd>
    </div>)}</dl>
  </div>;
}

export function LunaNavigationControl({ state, connected, request, commandHost, spatialTelemetry }: {
  state: LiveState; connected: boolean; request: (path: string, body?: unknown) => Promise<unknown>; commandHost: HTMLElement | null;
  spatialTelemetry?:SpatialTelemetry | null;
}) {
  const agent = state.agent;
  const kitchenSearch = state.challenge?.id === 'flat_kitchen';
  const luna = agent.configuration.models.find(model => model.id === 'luna' && model.provider === 'foundry');
  const goalKey = `${state.challenge?.environment ?? 'standalone'}:${state.challenge?.id ?? 'bench'}:${state.challenge?.orbit?.target ?? ''}:${state.challenge?.orbit?.direction ?? ''}`;
  const [goals, setGoals] = usePreference('goals', {});
  const goal = agent.active && agent.goal ? agent.goal : goals[goalKey] ?? state.challenge?.goal ?? 'Inspect the scene and navigate safely.';
  const setGoal = (value: string) => setGoals(previous => ({...previous, [goalKey]: value}));
  const [interval, setInterval] = usePreference('interval', .25);
  const [turns, setTurns] = usePreference('turns', 80);
  const [maxRequests, setMaxRequests] = usePreference('max_model_requests', 12);
  const [maxTokens, setMaxTokens] = usePreference('max_model_tokens', 100000);
  const [preferredReasoning, setReasoning] = usePreference('reasoning', 'high');
  const reasoning = luna?.reasoning_efforts.includes(preferredReasoning) ? preferredReasoning : luna?.reasoning_efforts[0] ?? 'high';
  const [endpoint, setEndpoint] = usePreference('luna_endpoint', agent.configuration.endpoint);
  const [deployment, setDeployment] = usePreference('luna_deployment', luna?.deployment ?? '');
  const connectionChanged = normalizedEndpoint(endpoint) !== normalizedEndpoint(agent.configuration.endpoint) || deployment !== (luna?.deployment ?? '');
  const [pending, setPending] = useState(false);
  const [error, setError] = useState('');
  const [explorationBudget, setExplorationBudget] = usePreference('exploration_budget', 180);
  const [diagnostic, setDiagnostic] = useState<'unified' | 'local' | 'legacy'>(() =>
    new URLSearchParams(location.search).get('diagnostics') === 'legacy' ? 'legacy' : 'unified');
  const [mapContext, setMapContext] = usePreference('mission_map_context', true);
  const unified = diagnostic !== 'legacy';
  const localOnly = diagnostic === 'local';
  const [capabilities, setCapabilities] = useState<{ready: boolean; architecture?: {name:string;version:string;revision:string}}>({ready:false});
  useEffect(() => {
    const controller = new AbortController();
    void fetch('/api/mission/capabilities', {signal:controller.signal}).then(response => response.ok ? response.json() : null)
      .then(value => {if (!controller.signal.aborted) setCapabilities({ready:value?.version===1 && value?.unified_mission===true, architecture:value?.architecture});})
      .catch(() => {});
    return () => controller.abort();
  }, [state.run_id]);
  const [instruction, setInstruction] = useState('');
  const [sending, setSending] = useState(false);
  const [instructionError, setInstructionError] = useState('');
  const [handoff, setHandoff] = usePreference('handoff', kitchenSearch);
  const [skillComposer, setSkillComposer] = usePreference('skill_composer', false);
  const [preferredBackend, setPreferredBackend] = usePreference('navigation_backend', 'nav2');
  const [nav2, setNav2] = useState({enabled:false, ready:false, message:'Nav2 bridge is offline'});
  const [supportsNavigationBackend, setSupportsNavigationBackend] = useState(false);
  const [aiRoutes, setAiRoutes] = usePreference('ai_routes', false);
  const [adaptive, setAdaptive] = usePreference('adaptive', true);
  const [inspector, setInspector] = usePreference('inspector', 'trace');
  const [connectionOpen, setConnectionOpen] = usePreference('connection_open', !luna?.configured);
  const [runSettingsOpen, setRunSettingsOpen] = usePreference('run_settings_open', false);
  const [instructionsOpen, setInstructionsOpen] = useState(true);
  const [compactArms] = usePreference('compact_arms', true);
  const hasRun = !!agent.session_id || agent.active;
  const [preferredMode, setNavigationMode] = usePreference('navigation_mode', 'luna_continuous');
  const navigationMode = unified ? 'luna_continuous' : preferredMode;
  const continuous = (agent.active ? agent.execution_mode : navigationMode) === 'luna_continuous';
  const backend = !unified && continuous && supportsNavigationBackend && nav2.enabled ? preferredBackend : 'builtin';
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
  async function sendInstruction() {
    setSending(true); setInstructionError('');
    try {
      await request('agent/instruction', {run_id:state.run_id,episode_epoch:state.episode_epoch,
        session_id:agent.session_id,message:instruction});
      setInstruction('');
    } catch (failure) { setInstructionError(failure instanceof Error ? failure.message : String(failure)); }
    finally { setSending(false); }
  }
  async function perform(path: string, body: unknown) {
    setPending(true); setError('');
    try { await request(path, body); }
    catch (failure) { setError(failure instanceof Error ? failure.message : String(failure)); }
    finally { setPending(false); }
  }
  const residentLabels = { unloaded: continuous ? 'Model not loaded / SmolVLA inactive' : 'Model not loaded / loads on first Start', loading: 'Loading model into GPU memory',
    ready: 'Model ready / kept in GPU memory', inferencing: 'Model loaded / processing a request', error: 'Model worker unavailable / reloads on Start' };
  const policyLabels = { single_step:'Single-step tools', navigation_plan:'Buffered navigation plan', supervised_policy:'Supervised SmolVLA',
    local_navigation:'Local SmolVLA navigation', luna_navigation:'Luna + SmolVLA primitives', luna_continuous:'Continuous local control' };
  const localMotionActive = state.continuous_navigation?.status === 'running' || state.navigation?.status === 'running'
    || state.skill?.status === 'running' || state.skill?.status === 'awaiting_policy';
  const policyActive = agent.active || localMotionActive;
  const policyName = !agent.active && localMotionActive ? state.skill?.skill ? `Local skill: ${state.skill.skill}` : 'Local navigation'
    : agent.unified_mission ? 'Unified mission' : policyLabels[agent.execution_mode];
  const policyStatus = !connected ? 'Last received / disconnected' : policyActive ? policyName : 'No active policy';
  function selectInspector(next: typeof inspector) {
    setInspector(next);
    if (next === 'settings') setRunSettingsOpen(true);
  }
  return <section className="agent-section" aria-label="LLM control">
    <div className="panel-header"><h3><Bot size={17} /> {unified ? 'Mission' : 'Luna'}</h3></div>
    {unified && !capabilities.ready && <p role="alert">Mission controller unavailable on this server. Restart with the current backend.</p>}
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
    {!continuous && agent.local_model && <LocalModelProgress live={agent.local_model} connected={connected} resident={resident} />}
    <form id="mission-form" className="agent-form" onSubmit={event => {
      event.preventDefault();
      if (unified) {
        void perform('mission/start', {run_id:state.run_id, episode_epoch:state.episode_epoch, execution_mode:'luna_continuous',
          unified_mission:true, mission_local_only:localOnly, map_context:mapContext, mission_budget_s:explorationBudget,
          images_per_request:2, context_tokens:8192, navigation_backend:'builtin', compact_arms:compactArms,
          model_id:'luna', reasoning, goal:localOnly ? 'Explore the observed environment within the mission budget.' : goal,
          max_turns:turns, feedback_interval_s:interval, max_model_requests:maxRequests, max_model_tokens:maxTokens});
        return;
      }
      void perform('agent/start', { run_id: state.run_id, episode_epoch: state.episode_epoch,
        execution_mode: navigationMode, compact_arms: compactArms,
        continuous_handoff: continuous && backend === 'builtin' && handoff,
        ...(continuous && supportsNavigationBackend ? {navigation_backend:backend} : {}),
        ...(continuous && backend === 'builtin' && skillComposer && supportsSkillComposer ? {skill_composer:true} : {}),
        ...(continuous && backend === 'builtin' && aiRoutes && supportsAiRoutes ? {ai_generated_routes:true} : {}),
        adaptive_navigation: adaptive,
        model_id: 'luna', reasoning, goal, max_turns: turns, feedback_interval_s: interval,
        max_model_requests:maxRequests, max_model_tokens:maxTokens });
    }}>
      <details className="mission-instructions compact-disclosure" open={instructionsOpen} onToggle={event => setInstructionsOpen(event.currentTarget.open)}>
      <summary><ChevronRight className="disclosure-chevron" size={15} /><MessageSquare size={15} /><strong>Instructions</strong>
        {!instructionsOpen && <span className="disclosure-preview">{goal}</span>}</summary>
      <div className="agent-goal-row"><label>Robot goal<textarea aria-label="Robot goal" value={goal} required maxLength={2000} rows={4}
        disabled={agent.active || pending} onChange={event => setGoal(event.target.value)} /></label>
      </div>
      </details>
      {commandHost && createPortal(<button type="submit" form="mission-form" className="primary" aria-label={unified ? 'Start mission' : 'Start LLM control'} title={unified ? 'Start mission' : 'Start LLM control'} disabled={agent.active || !connected || pending || state.power?.on === false || state.busy || (!localOnly && (!luna?.configured || connectionChanged || !goal.trim()))
          || !Number.isInteger(maxRequests) || maxRequests < 1 || maxRequests > 200 || !Number.isInteger(maxTokens) || maxTokens < 1 || maxTokens > 2000000
          || (unified && (!capabilities.ready || explorationBudget < 5 || explorationBudget > 300))
          || (!unified && ((continuous && backend === 'builtin' && aiRoutes && !supportsAiRoutes) || (backend === 'nav2' && !nav2.ready)))
          || !Number.isInteger(turns) || turns < 1 || turns > 80 || interval < .25 || interval > 30}>
          <Play size={16} /> Start</button>, commandHost)}
    </form>
    {error && <div className="error" role="alert">{error}</div>}
    <MotionDiagnostics navigation={state.navigation} connected={connected} state={state} spatial={spatialTelemetry} />
    <details className="active-policy compact-disclosure" aria-label="Active policy">
      <summary><ChevronRight className="disclosure-chevron" size={15} /><Route size={15} /><strong>Active policy</strong><span className="disclosure-preview">{policyStatus}</span></summary>
      {hasRun || localMotionActive ? <>
        <dl className="policy-facts">
          <div><dt>{policyActive && connected ? 'Controller' : 'Last reported controller'}</dt><dd>{policyName}</dd></div>
          {hasRun && <>
          <div><dt>{agent.active ? 'Execution mode' : 'Last run mode'}</dt><dd>{agent.execution_mode}</dd></div>
          <div><dt>{agent.active ? 'Navigation backend' : 'Last run backend'}</dt><dd>{agent.navigation_backend === 'nav2' ? 'Nav2' : agent.navigation_backend === 'builtin' ? 'Built-in' : 'Not reported'}</dd></div>
          <div><dt>{agent.active ? 'Phase' : 'Last run phase'}</dt><dd>{agent.mission?.phase ?? agent.phase}</dd></div>
          <div><dt>{agent.active ? 'Model profile' : 'Last run model profile'}</dt><dd>{agent.model_id || 'Not reported'} / {agent.reasoning}</dd></div>
          </>}
          {agent.mission?.plan && <div><dt>{agent.active ? 'Mission' : 'Last mission'}</dt><dd>{agent.mission.plan.kind} / {agent.mission.plan.target || 'No named target'}</dd></div>}
          {agent.mission?.objective && <div><dt>{agent.active ? 'Objective' : 'Last objective'}</dt><dd>{agent.mission.objective.action} / {agent.mission.objective.status}
            <br />{agent.mission.objective.remaining_s.toFixed(1)} s / {agent.mission.objective.remaining_travel_m.toFixed(2)} m remaining</dd></div>}
          {state.continuous_navigation && <div><dt>Local motion</dt><dd>{state.continuous_navigation.status} / {state.continuous_navigation.reason}</dd></div>}
          {state.skill?.skill && <div><dt>Skill</dt><dd>{state.skill.skill} / {state.skill.status}</dd></div>}
          {state.skill?.checkpoint && <div><dt>Skill checkpoint</dt><dd>{state.skill.checkpoint}</dd></div>}
        </dl>
        {agent.mission?.reason && <p className="exchange-text">{agent.mission.reason}</p>}
        <details className="exchange-payload"><summary>Reported policy state</summary><pre>{JSON.stringify({
          session_id:agent.session_id, active:agent.active, execution_mode:agent.execution_mode, navigation_backend:agent.navigation_backend,
          mission:agent.mission, navigation:state.navigation, continuous_navigation:state.continuous_navigation, skill:state.skill,
        }, null, 2)}</pre></details>
      </> : <p className="empty">No policy has run in this episode.</p>}
    </details>
    {hasRun && <TokenCounter agent={agent} connected={connected} />}
    {agent.inference_budget && <div className="agent-metrics" aria-label="Luna task budget">
      <span>Luna requests <strong>{agent.inference_budget.requests} / {agent.inference_budget.max_requests}</strong></span>
      <span>Token threshold <strong>{agent.inference_budget.max_tokens.toLocaleString()}</strong></span>
    </div>}
    {hasRun && <div className="agent-metrics"><span>Supervisor turns <strong>{agent.turns} / {agent.max_turns}</strong></span>
      <span>{continuous ? 'Route updates' : 'Local commands'} <strong>{continuous ? state.continuous_navigation?.updates ?? 0 : agent.local_model?.requests_completed ?? 0}</strong></span>
      <span>{continuous ? 'Luna inference' : 'Local inference'} <strong>{agent.inference_latency_s?.toFixed(2) ?? '-'} s</strong></span>
      <span>{agent.local_model?.instruction ?? ''}</span></div>}
    {hasRun && <div className="inspector-tabs tabs" role="tablist" aria-label="Robot inspector" onKeyDown={event => {
      const keys = ['conversation', 'trace', 'settings'] as const;
      const index = keys.indexOf(inspector);
      const next = event.key === 'ArrowRight' ? (index + 1) % keys.length : event.key === 'ArrowLeft' ? (index + keys.length - 1) % keys.length : event.key === 'Home' ? 0 : event.key === 'End' ? keys.length - 1 : -1;
      if (next < 0) return;
      event.preventDefault(); selectInspector(keys[next]);
      event.currentTarget.querySelectorAll<HTMLButtonElement>('[role="tab"]')[next].focus();
    }}>
      {([{id:'conversation',label:'Conversation',icon:MessageSquare},{id:'trace',label:'Trace',icon:Radio},{id:'settings',label:'Settings',icon:Settings2}] as const).map(({id,label,icon:Icon}) =>
        <button type="button" key={id} role="tab" id={`inspector-${id}`} aria-controls={`panel-${id}`} aria-selected={inspector === id}
          tabIndex={inspector === id ? 0 : -1} onClick={() => selectInspector(id)}><Icon size={15} />{label}</button>)}
    </div>}
    <div id="panel-conversation" role="tabpanel" aria-labelledby="inspector-conversation" hidden={!hasRun || inspector !== 'conversation'}>
    <section className="chat-section" aria-label="Run chat">
      <div className="panel-header"><h3><MessageSquare size={17} />Run chat</h3></div>
      <div className="chat-transcript" role="log" aria-label="Run conversation" tabIndex={0}>
        {!agent.run_messages?.length && <p className="empty">No messages in this run.</p>}
        {(agent.run_messages ?? []).map(message => <article key={message.id} className={`chat-message chat-${message.role}`}>
          <strong>{message.role === 'user' ? 'You' : message.source === 'model' ? 'Luna' : 'Controller'} / {message.status}</strong>
          <p>{message.text}</p></article>)}
      </div>
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
    {connectionChanged && <p role="status">Connection changes not applied</p>}
    <details className="run-options compact-disclosure" open={runSettingsOpen} onToggle={event => setRunSettingsOpen(event.currentTarget.open)}
      hidden={(hasRun && inspector !== 'settings')}>
    <summary><ChevronRight className="disclosure-chevron" size={15} /><Settings2 size={15} /><strong>Run settings</strong></summary>
    <div id="panel-settings" role={hasRun ? 'tabpanel' : undefined} aria-labelledby={hasRun ? 'inspector-settings' : undefined}>
      {!agent.active && (unified ? capabilities.architecture : variant) && <dl className="test-variant-preview" aria-label="Current architecture version"><div><dt>Architecture</dt><dd>{(unified ? capabilities.architecture : variant)?.name}</dd></div><div><dt>Version</dt><dd>{(unified ? capabilities.architecture : variant)?.version}+{(unified ? capabilities.architecture : variant)?.revision}</dd></div></dl>}
      <div className="agent-settings">
        {!unified && <label>Navigation controller<select aria-label="Navigation controller" value={navigationMode} disabled={agent.active || pending}
          onChange={event => setNavigationMode(event.target.value as typeof navigationMode)}>
          <option value="luna_continuous">Continuous local control</option><option value="luna_navigation">SmolVLA primitives</option></select></label>}
        {!unified && continuous && supportsNavigationBackend && <label>Navigation stack<select aria-label="Navigation stack" value={backend} disabled={agent.active || pending}
          onChange={event => setPreferredBackend(event.target.value as 'nav2' | 'builtin')}>
          <option value="nav2" disabled={!nav2.enabled}>Nav2 (primary)</option><option value="builtin">Built-in (backup)</option></select></label>}
        <label>Supervisor reasoning<select aria-label="Supervisor reasoning" value={reasoning} disabled={agent.active || pending}
          onChange={event => setReasoning(event.target.value as Reasoning)}>
          {(luna?.reasoning_efforts ?? ['low', 'medium', 'high']).map(effort => <option value={effort} key={effort}>{effort}</option>)}</select></label>
        <label>Supervisor turn limit<input aria-label="Supervisor turn limit" type="number" min={1} max={80} value={turns}
          disabled={agent.active || pending} onChange={event => setTurns(Number(event.target.value))} /></label>
        <label>Luna request limit<input aria-label="Luna request limit" type="number" min={1} max={200} value={maxRequests}
          disabled={agent.active || pending} onChange={event => setMaxRequests(Number(event.target.value))} /></label>
        <label>Luna token threshold<input aria-label="Luna token threshold" type="number" min={1} max={2000000} step={1000} value={maxTokens}
          title="Checked using reported usage after each request; one response may cross this threshold."
          disabled={agent.active || pending} onChange={event => setMaxTokens(Number(event.target.value))} /></label>
        {unified && <label>Mission budget (s)<input aria-label="Mission budget" type="number" min={5} max={300} value={explorationBudget}
          disabled={agent.active || pending} onChange={event=>setExplorationBudget(Number(event.target.value))}/></label>}
        {!continuous && <label>Local feedback interval (s)<input aria-label="Feedback interval (s)" type="number" min={.25} max={30} step={.25}
          value={interval} onChange={event => setInterval(Number(event.target.value))} /></label>}
        {!continuous && agent.active && <button type="button" disabled={pending || !connected || interval < .25 || interval > 30}
          onClick={() => void perform('agent/rate', {feedback_interval_s: interval})}><Timer size={16} />Apply rate</button>}
      </div>
      <details className="mission-diagnostics"><summary><Settings2 size={15}/>Diagnostics</summary>
        <label>Execution profile<select aria-label="Diagnostic execution profile" value={diagnostic} disabled={agent.active || pending}
          onChange={event=>setDiagnostic(event.target.value as typeof diagnostic)}><option value="unified">Unified mission</option>
          <option value="local">Unified local-only exploration</option><option value="legacy">Legacy controller comparison</option></select></label>
        {unified && <label className="toggle"><input type="checkbox" aria-label="Observed map context" checked={mapContext} disabled={agent.active || pending}
          onChange={event=>setMapContext(event.target.checked)}/>Observed map context</label>}
      </details>
      {!unified && <>
      {continuous && backend === 'nav2' && <p role="status" aria-label="Nav2 readiness">{nav2.ready ? 'Nav2 ready' : nav2.message}</p>}
      {continuous && backend === 'builtin' && <label className="toggle handoff-toggle" title={supportsAiRoutes ? 'Experimental generic waypoint control' : 'This backend does not support generic AI routes. Uncheck to use observed continuous control.'}><input type="checkbox" aria-label="AI-generated routes" checked={aiRoutes}
        disabled={agent.active || pending || (!supportsAiRoutes && !aiRoutes)} onChange={event => setAiRoutes(event.target.checked)} />AI-generated routes ({supportsAiRoutes ? 'experimental' : 'unavailable'})</label>}
      {continuous && backend === 'builtin' && !aiRoutes && <label className="toggle handoff-toggle"><input type="checkbox" aria-label="Plan while moving" checked={handoff}
        disabled={agent.active || pending} onChange={event => setHandoff(event.target.checked)} />Plan while moving (experimental)</label>}
      {continuous && backend === 'builtin' && !aiRoutes && <label className="toggle handoff-toggle"><input type="checkbox" aria-label="Motion skill composer" checked={skillComposer}
        disabled={agent.active || pending || !supportsSkillComposer} onChange={event => setSkillComposer(event.target.checked)} />Motion skill composer (experimental)</label>}
      {continuous && !aiRoutes && <label className="toggle handoff-toggle"><input type="checkbox" aria-label="Adaptive exploration" checked={adaptive}
        disabled={agent.active || pending} onChange={event => setAdaptive(event.target.checked)} />Adaptive exploration</label>}
      </>}
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
    <form className="chat-composer" hidden={!agent.active} onSubmit={event => {event.preventDefault();void sendInstruction();}}>
      <label>New instruction<textarea aria-label="New run instruction" value={instruction} rows={2} maxLength={2000}
        disabled={!connected || sending || !agent.active} onChange={event=>setInstruction(event.target.value)} /></label>
      <button type="submit" disabled={!connected || sending || !agent.active || !instruction.trim()}><Send size={16} />Send instruction</button>
    </form>
    {instructionError && <p className="error" role="alert">{instructionError}</p>}
  </section>;
}