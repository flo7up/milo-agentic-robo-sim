import { useEffect, useRef, useState, type ReactNode } from 'react';
import { createPortal } from 'react-dom';
import { Activity, Bot, Check, ChevronRight, CircleDot, Copy, Folder, Minus, Plus, Play, Power, Route, Save, Settings2, Timer, Send, MessageSquare, History, Radio, X } from 'lucide-react';
import { ExchangeFeed } from './ExchangeFeed';
import { ModelActivity } from './ModelActivity';
import { MotionDiagnostics } from './MotionDiagnostics';
import { LocalModelProgress } from './LocalModelProgress';
import { usePreference, useSavePreferences } from './Preferences';
import { RobotControlSlot } from './RobotControlSurface';
import { MemoryControls } from './MemoryControls';
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

export function LunaNavigationControl({ state, connected, request, commandHost, settingsHost, spatialTelemetry, telemetryContent, robotAccess }: {
  state: LiveState; connected: boolean; request: (path: string, body?: unknown) => Promise<unknown>; commandHost: HTMLElement | null;
  settingsHost?:HTMLElement | null;
  spatialTelemetry?:SpatialTelemetry | null;
  telemetryContent?:ReactNode;
  robotAccess?:ReactNode;
}) {
  const agent = state.agent;
  const kitchenSearch = state.challenge?.id === 'flat_kitchen';
  const luna = agent.configuration.models.find(model => model.id === 'luna' && model.provider === 'foundry');
  const qwen = agent.configuration.models.find(model => model.id === 'qwen' && model.provider === 'ollama');
  const [preferredController, setPreferredController] = usePreference('mission_controller', 'hybrid');
  const [diagnostic, setDiagnostic] = useState<'unified' | 'local' | 'legacy'>(() =>
    new URLSearchParams(location.search).get('diagnostics') === 'legacy' ? 'legacy' : 'unified');
  const unified = diagnostic !== 'legacy';
  const localOnly = diagnostic === 'local' || (unified && preferredController === 'policy');
  const hybrid = unified && preferredController === 'hybrid';
  const useLocalModel = unified && !localOnly && ['qwen','hybrid'].includes(preferredController);
  const selectedModel = useLocalModel ? qwen : luna;
  const modelId = useLocalModel ? 'qwen' : 'luna';
  const modelName = hybrid ? 'Qwen + Luna' : useLocalModel ? 'Qwen' : 'Luna';
  const goalKey = `${state.challenge?.environment ?? 'standalone'}:${state.challenge?.id ?? 'bench'}:${state.challenge?.orbit?.target ?? ''}:${state.challenge?.orbit?.direction ?? ''}`;
  const [goals, setGoals] = usePreference('goals', {});
  const goal = agent.active && agent.goal ? agent.goal : goals[goalKey] ?? state.challenge?.goal ?? 'Inspect the scene and navigate safely.';
  const setGoal = (value: string) => setGoals(previous => ({...previous, [goalKey]: value}));
  const [interval, setInterval] = usePreference('interval', .25);
  const kitchenBudget = kitchenSearch && unified;
  const [turns, setTurns] = usePreference(kitchenBudget ? 'kitchen_turns' : 'turns', kitchenBudget ? 60 : 80);
  const [maxRequests, setMaxRequests] = usePreference(kitchenBudget ? 'kitchen_max_model_requests' : 'max_model_requests', kitchenBudget ? 60 : 12);
  const [maxTokens, setMaxTokens] = usePreference(kitchenBudget ? 'kitchen_max_model_tokens' : 'max_model_tokens', kitchenBudget ? 400000 : 100000);
  const [preferredReasoning, setReasoning] = usePreference('reasoning', 'high');
  const reasoning = useLocalModel ? 'none' : luna?.reasoning_efforts.includes(preferredReasoning) ? preferredReasoning : luna?.reasoning_efforts[0] ?? 'high';
  const taskSupervisorReasoning = luna?.reasoning_efforts.includes('low') ? 'low' : luna?.reasoning_efforts[0] ?? 'none';
  const [endpoint, setEndpoint] = usePreference('luna_endpoint', agent.configuration.endpoint);
  const [deployment, setDeployment] = usePreference('luna_deployment', luna?.deployment ?? '');
  const [localEndpoint, setLocalEndpoint] = usePreference('local_model_endpoint', agent.configuration.ollama_endpoint);
  const [localTag, setLocalTag] = usePreference('local_model_tag', qwen?.deployment ?? 'qwen3-vl:4b-instruct-q4_K_M');
  const cloudConnectionChanged = normalizedEndpoint(endpoint) !== normalizedEndpoint(agent.configuration.endpoint) || deployment !== (luna?.deployment ?? '');
  const localConnectionChanged = normalizedEndpoint(localEndpoint) !== normalizedEndpoint(agent.configuration.ollama_endpoint) || localTag !== (qwen?.deployment ?? '');
  const selectedConnectionChanged = hybrid ? localConnectionChanged || cloudConnectionChanged : useLocalModel ? localConnectionChanged : cloudConnectionChanged;
  const connectionChanged = cloudConnectionChanged || ((useLocalModel || !!qwen) && localConnectionChanged);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState('');
  const [configurationOpen, setConfigurationOpen] = useState(false);
  const [configurationSaving, setConfigurationSaving] = useState(false);
  const [configurationError, setConfigurationError] = useState('');
  const [configurationHost, setConfigurationHost] = useState<HTMLDivElement | null>(null);
  const configurationDialog = useRef<HTMLDialogElement>(null);
  const restoreConfigurationFocus = useRef(false);
  const savePreferences = useSavePreferences();
  function closeConfiguration() {
    restoreConfigurationFocus.current = true;
    setConfigurationOpen(false);
  }
  useEffect(() => {
    if (configurationOpen) configurationDialog.current?.showModal();
    else configurationDialog.current?.close();
  }, [configurationOpen]);
  useEffect(() => {
    const button = settingsHost?.querySelector('button');
    if (!configurationOpen && restoreConfigurationFocus.current && button?.isConnected && !button.closest('dialog:not([open])')) {
      button.focus();
      restoreConfigurationFocus.current = false;
    }
  }, [configurationOpen, settingsHost]);
  const [recordingDraft, setRecordingDraft] = useState<{enabled:boolean;directory:string} | null>(null);
  const [recordingSaving, setRecordingSaving] = useState(false);
  const [recordingError, setRecordingError] = useState('');
  const [recordingMessage, setRecordingMessage] = useState('');
  const recording = state.recording;
  const recordingChoice = recordingDraft ?? {enabled:recording?.enabled ?? true,directory:recording?.directory ?? ''};
  const recordingLocked = !connected || !recording || agent.active || state.busy || recording.active || pending || recordingSaving;
  async function saveRecording() {
    if (recordingLocked || !recordingDraft) return;
    setRecordingSaving(true); setRecordingError(''); setRecordingMessage('');
    try {
      await request('preferences', {recording_enabled:recordingChoice.enabled,recording_directory:recordingChoice.directory});
      setRecordingDraft(null); setRecordingMessage('Recording settings saved');
    } catch (failure) {setRecordingError(failure instanceof Error ? failure.message : String(failure));}
    finally {setRecordingSaving(false);}
  }
  async function copyRecordingPath() {
    try {await navigator.clipboard.writeText(recording!.run_directory!); setRecordingMessage('Recording path copied');}
    catch {setRecordingError('Could not copy the recording path');}
  }
  const [explorationBudget, setExplorationBudget] = usePreference('exploration_budget', 180);
  const [mapContext, setMapContext] = usePreference('mission_map_context', true);
  const [capabilities, setCapabilities] = useState<{ready: boolean; localSupervisor?:boolean; architecture?: {name:string;version:string;revision:string}}>({ready:false});
  useEffect(() => {
    const controller = new AbortController();
    void fetch('/api/mission/capabilities', {signal:controller.signal}).then(response => response.ok ? response.json() : null)
      .then(value => {if (!controller.signal.aborted) setCapabilities({ready:value?.version===1 && value?.unified_mission===true,
        localSupervisor:value?.local_supervisor?.transport==='mission_json_v1', architecture:value?.architecture});})
      .catch(() => {});
    return () => controller.abort();
  }, [state.run_id]);
  const [localCheckRevision, setLocalCheckRevision] = useState(0);
  const [localReadiness, setLocalReadiness] = useState<{ready:boolean;message:string;key:string} | null>(null);
  const localReadinessKey = JSON.stringify([state.run_id, agent.configuration.ollama_endpoint, qwen?.deployment, localCheckRevision]);
  useEffect(() => {
    if (!useLocalModel || !connected || !capabilities.localSupervisor || localConnectionChanged || agent.active) return;
    const controller = new AbortController();
    let active = true;
    const timeout = window.setTimeout(()=>controller.abort(), 7000);
    setLocalReadiness(null);
    void fetch('/api/agent/local-readiness?model_id=qwen', {signal:controller.signal}).then(async response => {
      if (!response.ok) throw new Error('Local model readiness unavailable');
      return response.json();
    }).then(value=>{
      if (!controller.signal.aborted) setLocalReadiness({ready:value.ready===true,message:value.message,key:localReadinessKey});
    }).catch(()=>{if (active) setLocalReadiness({ready:false,message:'Local model readiness unavailable',key:localReadinessKey});})
      .finally(()=>window.clearTimeout(timeout));
    return ()=>{active=false;controller.abort();window.clearTimeout(timeout);};
  }, [useLocalModel, connected, capabilities.localSupervisor, localConnectionChanged, agent.active, localReadinessKey]);
  const localModelReady = !!capabilities.localSupervisor && localReadiness?.key === localReadinessKey && localReadiness.ready;
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
  const [savedInspector, setInspector] = usePreference('inspector', 'conversation');
  const [telemetrySelected, setTelemetrySelected] = useState(false);
  const inspector = telemetrySelected ? 'telemetry' : savedInspector === 'settings' ? 'conversation' : savedInspector;
  const [consoleMinimized, setConsoleMinimized] = useState(false);
  const [connectionOpen, setConnectionOpen] = usePreference('connection_open', !luna?.configured);
  const [runSettingsOpen, setRunSettingsOpen] = usePreference('run_settings_open', false);
  const [instructionsOpen, setInstructionsOpen] = useState(false);
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
    void fetch(`/api/test-variant?execution_mode=${navigationMode}&model_id=${modelId}&reasoning=${reasoning}&skill_composer=${navigationMode === 'luna_continuous' && backend === 'builtin' && skillComposer}&navigation_backend=${backend}${hybrid ? '&task_supervisor_model_id=luna' : ''}`, {signal:controller.signal})
      .then(async response => response.ok ? response.json() : null)
      .then(data => { if (!controller.signal.aborted) { setVariant(data?.architecture ?? null); setSupportsAiRoutes(data?.supports_ai_generated_routes === true); setSupportsSkillComposer(data?.supports_skill_composer === true); setSupportsNavigationBackend(data?.supports_navigation_backend === true); if (data?.nav2) setNav2(data.nav2); } })
      .catch(() => {});
    return () => controller.abort();
  }, [navigationMode, modelId, reasoning, selectedModel?.deployment, skillComposer, backend, hybrid]);
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
  const startUnavailable = agent.active || state.regression?.active || !connected || pending || sending || recordingSaving || configurationSaving || state.power?.on === false || state.busy
    || (!localOnly && (!selectedModel?.configured || selectedConnectionChanged || (useLocalModel && !localModelReady)))
    || (hybrid && !luna?.configured)
    || !Number.isInteger(maxRequests) || maxRequests < 1 || maxRequests > 200 || !Number.isInteger(maxTokens) || maxTokens < 1 || maxTokens > 2000000
    || (unified && (!capabilities.ready || !Number.isFinite(explorationBudget) || explorationBudget < 5 || explorationBudget > 300))
    || (!unified && ((continuous && backend === 'builtin' && aiRoutes && !supportsAiRoutes) || (backend === 'nav2' && !nav2.ready)))
    || !Number.isInteger(turns) || turns < 1 || turns > 80 || !Number.isFinite(interval) || interval < .25 || interval > 30;
  const stopInstruction = ['stop','pause','halt','cancel'].includes(instruction.trim().toLowerCase().replace(/[.!]+$/, ''));
  const instructionUnavailable = !connected ? 'Disconnected' : state.power?.on === false ? 'Robot is off'
    : state.regression?.active && !stopInstruction ? 'Stop the regression baseline before sending an instruction'
    : pending || sending || recordingSaving || configurationSaving ? 'Sending request' : stopInstruction ? '' : localOnly ? 'Custom instructions require a model supervisor'
    : agent.active ? (!agent.session_id ? 'Waiting for the active session' : '')
    : state.busy ? 'Robot is busy' : !selectedModel?.configured ? `${modelName} connection required`
    : selectedConnectionChanged ? 'Connection changes not applied' : useLocalModel && !localModelReady ? 'Local model not ready'
    : startUnavailable ? 'Check mission settings and controller readiness' : '';
  function startRequest(target: string) {
    if (unified) return request('mission/start', {run_id:state.run_id, episode_epoch:state.episode_epoch, execution_mode:'luna_continuous',
      unified_mission:true, mission_local_only:localOnly, map_context:mapContext, mission_budget_s:explorationBudget,
      images_per_request:2, context_tokens:8192, navigation_backend:'builtin', compact_arms:compactArms,
      model_id:modelId, reasoning, goal:localOnly ? 'Explore the observed environment within the mission budget.' : target,
      ...(hybrid ? {task_supervisor_model_id:'luna',task_supervisor_reasoning:taskSupervisorReasoning,
        max_task_supervisor_requests:4,max_task_supervisor_tokens:100000} : {}),
      max_turns:turns, feedback_interval_s:interval, max_model_requests:maxRequests, max_model_tokens:maxTokens});
    return request('agent/start', {run_id:state.run_id, episode_epoch:state.episode_epoch,
      execution_mode:navigationMode, compact_arms:compactArms,
      continuous_handoff:continuous && backend === 'builtin' && handoff,
      ...(continuous && supportsNavigationBackend ? {navigation_backend:backend} : {}),
      ...(continuous && backend === 'builtin' && skillComposer && supportsSkillComposer ? {skill_composer:true} : {}),
      ...(continuous && backend === 'builtin' && aiRoutes && supportsAiRoutes ? {ai_generated_routes:true} : {}),
      adaptive_navigation:adaptive, model_id:'luna', reasoning, goal:target, max_turns:turns, feedback_interval_s:interval,
      max_model_requests:maxRequests, max_model_tokens:maxTokens});
  }
  async function startMission() {
    if (startUnavailable || (!localOnly && !goal.trim())) return;
    setPending(true); setError('');
    try { await startRequest(goal); }
    catch (failure) { setError(failure instanceof Error ? failure.message : String(failure)); }
    finally { setPending(false); }
  }
  async function sendInstruction() {
    const message = instruction.trim();
    if (instructionUnavailable || !message) return;
    setSending(true); setInstructionError('');
    try {
      if (stopInstruction) await request('stop');
      else if (agent.active) await request('agent/instruction', {run_id:state.run_id,episode_epoch:state.episode_epoch,
        session_id:agent.session_id,message});
      else {
        await startRequest(message);
        setGoal(message);
      }
      setInstruction('');
      selectInspector('conversation');
    } catch (failure) { setInstructionError(failure instanceof Error ? failure.message : String(failure)); }
    finally { setSending(false); }
  }
  async function perform(path: string, body: unknown) {
    setPending(true); setError('');
    try { await request(path, body); }
    catch (failure) { setError(failure instanceof Error ? failure.message : String(failure)); }
    finally { setPending(false); }
  }
  function connectionValues() {
    const models = agent.configuration.models.map(({configured: _, ...model}) => model.id === 'luna'
      ? {...model, provider: 'foundry', deployment} : model.id === 'qwen'
      ? {...model, provider:'ollama', deployment:localTag, reasoning_efforts:['none'] as Reasoning[], context_window:16384} : model);
    if (!models.some(model => model.id === 'luna')) models.push({id:'luna',label:'Luna',deployment,provider:'foundry',reasoning_efforts:['low','medium','high']});
    if (useLocalModel && !models.some(model => model.id === 'qwen')) models.push({id:'qwen',label:'Qwen3-VL 4B (local)',deployment:localTag,
      provider:'ollama',reasoning_efforts:['none'],context_window:16384});
    return {endpoint, ollama_endpoint:(useLocalModel || !!qwen) ? localEndpoint : agent.configuration.ollama_endpoint, models};
  }
  const configurationLocked = !connected || agent.active || state.busy || pending || sending || recordingSaving || configurationSaving || !!recording?.active;
  async function saveConfiguration() {
    if (configurationLocked) return;
    const fields = configurationDialog.current?.querySelectorAll<HTMLInputElement | HTMLSelectElement>('input, select');
    if (fields && [...fields].some(field => !field.reportValidity())) return;
    setConfigurationSaving(true); setConfigurationError('');
    try {
      if (connectionChanged) await request('agent/config', connectionValues());
      if (recordingDraft) {
        if (!recording) throw new Error('Recording settings require a backend update.');
        await request('preferences', {recording_enabled:recordingChoice.enabled,recording_directory:recordingChoice.directory});
        setRecordingDraft(null);
      }
      await savePreferences();
      closeConfiguration();
    } catch (failure) { setConfigurationError(failure instanceof Error ? failure.message : String(failure)); }
    finally { setConfigurationSaving(false); }
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
  function selectInspector(next: typeof savedInspector | 'telemetry') {
    if (next === 'settings') {
      setRunSettingsOpen(true);
      setConfigurationOpen(true);
      return;
    }
    setConsoleMinimized(false);
    setTelemetrySelected(next === 'telemetry');
    if (next !== 'telemetry') setInspector(next);
  }
  return <section className="agent-section" aria-label="LLM control">
    {settingsHost && createPortal(<button type="button" aria-label="Robot settings" title="Robot configuration"
      aria-haspopup="dialog" aria-controls="robot-configuration" aria-expanded={configurationOpen} onClick={event => {
        if (configurationOpen) return;
        event.currentTarget.closest('dialog')?.close();
        selectInspector('settings');
      }}><Settings2 size={17}/>Robot settings</button>, settingsHost)}
    {createPortal(<dialog id="robot-configuration" className="robot-configuration" ref={configurationDialog} aria-labelledby="configuration-title"
      onClose={closeConfiguration}>
      <div className="configuration-heading"><h2 id="configuration-title"><Settings2 size={18}/>Robot configuration</h2>
        <button type="button" className="icon-button" aria-label="Close configuration" title="Close configuration"
          onClick={closeConfiguration}><X size={18}/></button></div>
      <RobotControlSlot active={configurationOpen}/>
      <div className="configuration-content">{robotAccess}<fieldset disabled={configurationSaving}>
        <div ref={setConfigurationHost} className="configuration-fields"/>
      </fieldset></div>
      <div className="configuration-footer">
        {configurationError && <p className="error" role="alert">{configurationError}</p>}
        {error && <p className="error" role="alert">{error}</p>}
        {connectionChanged && <p role="status">Connection changes not applied</p>}
        <button type="button" className="primary" disabled={configurationLocked} onClick={()=>void saveConfiguration()}>
          <Save size={16}/>{configurationSaving ? 'Saving configuration' : 'Save configuration'}</button>
      </div>
    </dialog>, document.body)}
    <div className="panel-header console-heading"><h3><Bot size={17} />Robot assistant</h3>
      {recording?.active && <span className="tag" role="status" aria-label="Run recording"><CircleDot size={13}/>{recording.status === 'finalizing' ? 'Saving recording' : 'Recording'}</span>}
      <button type="button" className="icon-button" aria-label={consoleMinimized ? 'Show assistant details' : 'Hide assistant details'}
        title={consoleMinimized ? 'Show assistant details' : 'Hide assistant details'} aria-expanded={!consoleMinimized} aria-controls="luna-console-body"
        onClick={()=>setConsoleMinimized(value=>!value)}>{consoleMinimized ? <Plus size={16}/> : <Minus size={16}/>}</button>
    </div>
    <ModelActivity agent={agent} connected={connected} modelName={localOnly ? 'Local exploration' : modelName}
      onCalls={()=>{selectInspector('trace');requestAnimationFrame(()=>document.getElementById('inspector-trace')?.focus());}}
      onDetails={()=>{selectInspector('telemetry');requestAnimationFrame(()=>document.getElementById('inspector-telemetry')?.focus());}} />
    <div id="luna-console-body" hidden={consoleMinimized}>
    <div className="inspector-tabs tabs" role="tablist" aria-label="Robot inspector" onKeyDown={event => {
      const keys = ['conversation', 'trace', 'telemetry'] as const;
      const index = keys.indexOf(inspector);
      const next = event.key === 'ArrowRight' ? (index + 1) % keys.length : event.key === 'ArrowLeft' ? (index + keys.length - 1) % keys.length : event.key === 'Home' ? 0 : event.key === 'End' ? keys.length - 1 : -1;
      if (next < 0) return;
      event.preventDefault(); selectInspector(keys[next]);
      event.currentTarget.querySelectorAll<HTMLButtonElement>('[role="tab"]')[next].focus();
    }}>
      {([{id:'conversation',label:'Conversation',icon:MessageSquare},{id:'trace',label:'Model calls',icon:Radio},
        {id:'telemetry',label:'Details',icon:Activity}] as const).map(({id,label,icon:Icon}) =>
        <button type="button" key={id} role="tab" id={`inspector-${id}`} aria-controls={`panel-${id}`} aria-selected={inspector === id}
          tabIndex={inspector === id ? 0 : -1} onClick={() => selectInspector(id)}><Icon size={15} />{label}</button>)}
    </div>
    {unified && !capabilities.ready && <p role="alert">Mission controller unavailable on this server. Restart with the current backend.</p>}
    {configurationHost && createPortal(<div className="mission-controller-choice">
      <label>Mission controller<select aria-label="Mission controller" value={unified ? preferredController : 'luna'} disabled={configurationLocked || !unified}
        onChange={event=>{setPreferredController(event.target.value as typeof preferredController);setConnectionOpen(true);}}>
        <option value="luna">Luna + existing policies</option><option value="qwen">Local Qwen + existing policies</option>
        <option value="hybrid">Qwen + Luna task supervision</option>
        <option value="policy">Policy-only exploration (backup)</option></select></label>
    </div>, configurationHost)}
    {configurationHost && createPortal(<details className={`agent-connection setup-connection${useLocalModel ? ' local-supervisor-connection' : ''}`} hidden={agent.active} open={connectionOpen}
      onToggle={event => setConnectionOpen(event.currentTarget.open)}><summary><Settings2 size={15} /> {useLocalModel ? 'Local model connection' : 'Luna connection'}</summary>
      <form onSubmit={event => {
        event.preventDefault();
        void perform('agent/config', connectionValues());
      }}><fieldset disabled={!connected || agent.active || pending}>
        {useLocalModel ? <>
          <label>Ollama endpoint<input aria-label="Ollama endpoint" type="url" value={localEndpoint} onChange={event=>setLocalEndpoint(event.target.value)}/></label>
          <label>Local model tag<input aria-label="Local model tag" value={localTag} maxLength={120} onChange={event=>setLocalTag(event.target.value)}/></label>
        </> : <>
          <label>Foundry endpoint<input aria-label="Foundry endpoint" type="url" value={endpoint} onChange={event => setEndpoint(event.target.value)} /></label>
          <label>Luna deployment<input aria-label="Luna deployment" value={deployment} onChange={event => setDeployment(event.target.value)} /></label>
        </>}
        <button type="submit"><Check size={16} />{useLocalModel ? 'Apply local connection' : 'Apply Luna connection'}</button>
      </fieldset></form>
      {useLocalModel && <>
        <p role="status" aria-label="Local supervisor readiness">{!capabilities.localSupervisor ? 'Backend update required' : localConnectionChanged ? 'Apply local connection first'
          : localReadiness?.key === localReadinessKey ? localReadiness.message : 'Checking local model'}</p>
        <button type="button" disabled={configurationLocked || localConnectionChanged || !capabilities.localSupervisor}
          onClick={()=>setLocalCheckRevision(value=>value+1)}><Check size={16}/>Check local model</button>
        {hybrid && <p role="status" aria-label="Luna task supervision readiness">{luna?.configured
          ? 'Luna sets the immutable task plan only; Qwen owns current-observation decisions' : 'Configure Luna before hybrid task supervision'}</p>}
      </>}
    </details>, configurationHost)}
    {!continuous && agent.local_model && <LocalModelProgress live={agent.local_model} connected={connected} resident={resident} />}
    <form id="mission-form" className="agent-form" onSubmit={event => {
      event.preventDefault();
      void startMission();
    }}>
      <details className="mission-instructions compact-disclosure" open={instructionsOpen} onToggle={event => setInstructionsOpen(event.currentTarget.open)}>
      <summary><ChevronRight className="disclosure-chevron" size={15} /><MessageSquare size={15} /><strong>Instructions</strong>
        {!instructionsOpen && <span className="disclosure-preview">{goal}</span>}</summary>
      <div className="agent-goal-row"><label>Robot goal<textarea aria-label="Robot goal" value={goal} required maxLength={2000} rows={4}
        disabled={agent.active || pending} onChange={event => setGoal(event.target.value)} /></label>
      </div>
      </details>
        {commandHost && createPortal(<button type="submit" form="mission-form" className="primary" aria-label={unified ? 'Start mission' : 'Start LLM control'} title={unified ? 'Start mission' : 'Start LLM control'} disabled={startUnavailable || (!localOnly && !goal.trim())}>
          <Play size={16} /> Start</button>, commandHost)}
    </form>
    {error && <div className="error" role="alert">{error}</div>}
    <div id="panel-telemetry" role="tabpanel" aria-labelledby="inspector-telemetry" hidden={inspector !== 'telemetry'}>
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
      <span>{agent.model_id === 'qwen' ? 'Qwen' : 'Luna'} requests <strong>{agent.inference_budget.requests} / {agent.inference_budget.max_requests}</strong></span>
      <span>Token threshold <strong>{agent.inference_budget.max_tokens.toLocaleString()}</strong></span>
    </div>}
    {agent.task_supervision && <div className="agent-metrics" aria-label="Luna task supervision budget">
      <span>Luna task reviews <strong>{agent.task_supervision.requests} / {agent.task_supervision.max_requests}</strong></span>
      <span>Tokens <strong>{agent.task_supervision.tokens.toLocaleString()} / {agent.task_supervision.max_tokens.toLocaleString()}</strong></span>
      <span>Motion authority <strong>None</strong></span>
    </div>}
    {hasRun && <div className="agent-metrics"><span>Supervisor turns <strong>{agent.turns} / {agent.max_turns}</strong></span>
      <span>{continuous ? 'Route updates' : 'Local commands'} <strong>{continuous ? state.continuous_navigation?.updates ?? 0 : agent.local_model?.requests_completed ?? 0}</strong></span>
      <span>{continuous ? `${agent.model_id === 'qwen' ? 'Qwen' : 'Luna'} inference` : 'Local inference'} <strong>{agent.inference_latency_s?.toFixed(2) ?? '-'} s</strong></span>
      <span>{agent.local_model?.instruction ?? ''}</span></div>}
    {telemetryContent}
    </div>
    <div id="panel-conversation" role="tabpanel" aria-labelledby="inspector-conversation" hidden={inspector !== 'conversation'}>
    <section className="chat-section" aria-label="Run chat">
      <div className="panel-header"><h3><MessageSquare size={17} />Run chat</h3></div>
      <div className="chat-transcript" role="log" aria-label="Run conversation" tabIndex={0}>
        {!hasRun && !agent.run_messages?.length && <p className="empty">No messages yet.</p>}
        {(agent.run_messages ?? []).map(message => <article key={message.id} className={`chat-message chat-${message.role}`}>
          <strong>{message.role === 'user' ? 'You' : message.source === 'model' ? agent.configuration.models.find(model=>model.id===agent.model_id)?.label ?? modelName : 'Controller'} / {message.status}</strong>
          <p>{message.text}</p></article>)}
        {hasRun && agent.goal && !agent.run_messages?.some(message=>message.role==='user' && message.text===agent.goal) &&
          <article className="chat-message chat-user"><strong>You / mission instruction</strong><p>{agent.goal}</p></article>}
      </div>
    </section>
    </div>
    <div id="panel-trace" role="tabpanel" aria-labelledby="inspector-trace" hidden={inspector !== 'trace'}>
    <ExchangeFeed agent={agent} visible={inspector === 'trace' && !consoleMinimized} />
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
    {configurationHost && createPortal(<details className="run-options compact-disclosure" open={runSettingsOpen} onToggle={event => setRunSettingsOpen(event.currentTarget.open)}>
    <summary><ChevronRight className="disclosure-chevron" size={15} /><Settings2 size={15} /><strong>Run settings</strong></summary>
    <div id="panel-settings">
      <MemoryControls active={configurationOpen} locked={!connected || agent.active || state.busy || pending || configurationSaving}
        runId={state.run_id} epoch={state.episode_epoch} request={request}/>
      <section className="recording-settings" aria-label="Test run recording">
        <div className="panel-header"><h4><CircleDot size={16}/> Recording</h4>
          <span className="tag" role="status" aria-label="Recording status">{!connected ? 'Disconnected' : !recording ? 'Backend update required'
            : recording.error ? 'Recording failed' : recording.status === 'recording' ? 'Recording' : recording.status === 'finalizing' ? 'Saving recording'
            : recording.enabled ? 'Ready for next run' : 'Recording off'}</span></div>
        <form onSubmit={event=>{event.preventDefault();void saveRecording();}}>
          <fieldset disabled={recordingLocked}>
            <label className="toggle"><input type="checkbox" role="switch" aria-label="Record test runs" checked={recordingChoice.enabled}
              onChange={event=>{setRecordingDraft({...recordingChoice,enabled:event.target.checked});setRecordingMessage('');}}/>Record test runs</label>
            <label><span className="recording-folder-label"><Folder size={15}/> Recording folder</span><input aria-label="Recording folder" value={recordingChoice.directory} maxLength={1024}
              placeholder=".runtime/performance" spellCheck={false} title="Local backend folder. Relative paths use the workspace root; empty restores the default. Applies to future runs."
              onChange={event=>{setRecordingDraft({...recordingChoice,directory:event.target.value});setRecordingMessage('');}}/></label>
            <button type="submit" disabled={!recordingDraft || recordingSaving}><Save size={16}/>{recordingSaving ? 'Saving' : 'Save recording settings'}</button>
          </fieldset>
        </form>
        {recording?.run_directory && <div className="recording-location"><div><strong>{recording.active ? 'Current recording' : 'Last recording'}</strong>
          <code>{recording.run_directory}</code></div><button type="button" className="icon-button" aria-label="Copy recording path" title="Copy recording folder path"
            onClick={()=>void copyRecordingPath()}><Copy size={16}/></button></div>}
        {recording?.samples != null && <span className="exchange-meta">{recording.samples.toLocaleString()} samples</span>}
        {recordingMessage && <p role="status" className="exchange-meta">{recordingMessage}</p>}
        {(recordingError || recording?.error) && <p role="alert" className="error">{recordingError || recording?.error}</p>}
      </section>
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
          {(useLocalModel ? ['none'] : luna?.reasoning_efforts ?? ['low', 'medium', 'high']).map(effort => <option value={effort} key={effort}>{effort}</option>)}</select></label>
        <label>Supervisor turn limit<input aria-label="Supervisor turn limit" type="number" min={1} max={80} value={turns}
          disabled={agent.active || pending} onChange={event => setTurns(Number(event.target.value))} /></label>
        <label>Luna request limit<input aria-label="Luna request limit" type="number" min={1} max={200} value={maxRequests}
          disabled={agent.active || pending} onChange={event => setMaxRequests(Number(event.target.value))} /></label>
        <div className="token-budget-control">
        <label htmlFor="token-budget-slider">Luna token budget <output>{maxTokens.toLocaleString('en-US')}</output></label>
        <input id="token-budget-slider" aria-label="Luna token budget" type="range" min={0} max={100} step={1}
          value={maxTokens <= 100000 ? maxTokens / 2000 : 50 + (maxTokens - 100000) / 38000}
          aria-valuetext={`${maxTokens.toLocaleString('en-US')} tokens`}
          title="Next run token threshold; reported usage can exceed it by one response."
          disabled={agent.active || pending} onChange={event => {
            const value = Number(event.target.value);
            setMaxTokens(Math.max(1, value <= 50 ? value * 2000 : 100000 + (value - 50) * 38000));
          }} />
        <label className="token-budget-exact">Exact threshold<input aria-label="Luna token threshold" type="number" min={1} max={2000000} step={1} value={maxTokens}
          title="Checked using reported usage after each request; one response may cross this threshold."
          disabled={agent.active || pending} onChange={event => setMaxTokens(Number(event.target.value))} /></label>
        </div>
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
    </details>, configurationHost)}
    <form className="chat-composer" onSubmit={event => {event.preventDefault();void sendInstruction();}}>
      <label>New instruction<textarea aria-label="New run instruction" value={instruction} rows={2} maxLength={2000}
        disabled={!connected || sending || pending || state.power?.on === false} onChange={event=>setInstruction(event.target.value)} /></label>
      <button type="submit" title={instructionUnavailable || (stopInstruction ? 'Stop the robot without a model request' : agent.active ? 'Update the current mission' : 'Start a mission with this instruction')}
        disabled={!!instructionUnavailable || !instruction.trim()}><Send size={16} />Send instruction</button>
    </form>
    {!!instructionUnavailable && !sending && !pending && <p className="exchange-meta" role="status" aria-label="Instruction availability">{instructionUnavailable}</p>}
    {instructionError && <p className="error" role="alert">{instructionError}</p>}
    </div>
  </section>;
}
