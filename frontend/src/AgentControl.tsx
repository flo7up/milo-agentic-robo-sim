import { useEffect, useRef, useState } from 'react';
import { Bot, Check, Cpu, Hand, ListChecks, MessageSquare, Mic, Play, Plus, Power, RefreshCw, Route, Send, Settings2, Square, StepForward, Timer } from 'lucide-react';
import { ExchangeFeed } from './ExchangeFeed';
import { LocalModelProgress } from './LocalModelProgress';
import { VoiceControl } from './VoiceControl';
import type { ExecutionMode, LiveState, Reasoning } from './types';

const phaseLabels: Record<string, string> = {
  idle: 'Manual control', starting: 'Starting', thinking: 'Awaiting model', acting: 'Executing tool',
  waiting: 'Waiting for feedback', stopped: 'Stopped', completed: 'Session ended', error: 'Connection or control error',
  voice_ready: 'Voice connected', listening: 'Microphone active', speaking: 'Milo speaking',
  sleeping: 'Agent idle', waking: 'Waking up',
};

export function AgentControl({ state, connected, request }: {
  state: LiveState; connected: boolean; request: (path: string, body?: unknown) => Promise<unknown>;
}) {
  const agent = state.agent;
  const configuration = agent.configuration;
  const defaultProfile = configuration.models.find(entry => entry.id === configuration.default_model_id) ?? configuration.models[0];
  const [modelId, setModelId] = useState(configuration.default_model_id);
  const [reasoning, setReasoning] = useState<Reasoning>(defaultProfile.reasoning_efforts.includes('low') ? 'low' : defaultProfile.reasoning_efforts[0]);
  const [executionMode, setExecutionMode] = useState<ExecutionMode>(() => {
    if (agent.active) return agent.execution_mode;
    try {
      const saved = localStorage.getItem('milo-execution-mode');
      if (saved && ['single_step', 'navigation_plan', 'supervised_policy', 'local_navigation'].includes(saved)) return saved as ExecutionMode;
    } catch {}
    return 'local_navigation';
  });
  const [policyEndpoint, setPolicyEndpoint] = useState(agent.policy?.endpoint ?? 'http://127.0.0.1:8085');
  const [policyCheck, setPolicyCheck] = useState<{ endpoint: string; ready: boolean; message: string; checkpoint?: string } | null>(null);
  const [checkingPolicy, setCheckingPolicy] = useState(false);
  const [policyRetry, setPolicyRetry] = useState(0);
  const [imagesPerRequest, setImagesPerRequest] = useState(agent.images_per_request ?? 1);
  const [contextTokens, setContextTokens] = useState(agent.context_tokens ?? 4096);
  const [interval, setInterval] = useState(executionMode === 'local_navigation' ? 1 : agent.feedback_interval_s);
  const [goal, setGoal] = useState(state.challenge?.goal ?? 'Inspect the area, approach a visible cube, and stop before contact.');
  const [turnLimit, setTurnLimit] = useState(executionMode === 'local_navigation' ? 30 : state.challenge?.suggested_turn_limit ?? 30);
  const [pending, setPending] = useState(false);
  const [switching, setSwitching] = useState(false);
  const [message, setMessage] = useState('');
  const transcript = useRef<HTMLDivElement>(null);
  const followChat = useRef(true);
  const focusMode = useRef<'chat' | 'voice' | null>(null);
  const [error, setError] = useState('');
  const [endpoint, setEndpoint] = useState(configuration.endpoint);
  const [ollamaEndpoint, setOllamaEndpoint] = useState(configuration.ollama_endpoint);
  const [provider, setProvider] = useState<'foundry' | 'ollama'>(defaultProfile.provider);
  const [label, setLabel] = useState('GPT-5.6 Luna');
  const [deployment, setDeployment] = useState('');
  const [addingModel, setAddingModel] = useState(false);
  const profile = configuration.models.find(entry => entry.id === modelId) ?? configuration.models[0];
  const mode = state.interaction_mode ?? 'chat';
  const localNavigation = mode === 'chat' && executionMode === 'local_navigation';
  const residentModel = state.local_navigation_model;
  const residencyLabels = {
    unloaded: 'Model not loaded / loads on first Start',
    loading: 'Loading model into GPU memory',
    ready: 'Model ready / kept in GPU memory',
    inferencing: 'Model loaded / processing a request',
    error: 'Model worker unavailable / reloads on Start',
  };
  useEffect(() => {
    if (executionMode !== 'supervised_policy' || mode !== 'chat' || !connected || agent.active) return;
    const controller = new AbortController();
    setPolicyCheck(null);
    setCheckingPolicy(true);
    async function check() {
      try {
        const response = await fetch('/api/policy/check', { method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ endpoint: policyEndpoint }), signal: controller.signal });
        if (!response.ok) throw new Error(response.status === 404
          ? 'Backend update required for policy readiness checks.'
          : response.status === 422 ? 'Use a local HTTP policy endpoint, such as http://127.0.0.1:8085.' : 'Policy readiness check failed.');
        const result = await response.json();
        if (!controller.signal.aborted) setPolicyCheck({ endpoint: policyEndpoint, ready: result.ready === true,
          message: result.message, checkpoint: result.metadata?.checkpoint });
      } catch (failure) {
        if (!controller.signal.aborted) setPolicyCheck({ endpoint: policyEndpoint, ready: false,
          message: failure instanceof Error ? failure.message : 'Policy readiness check failed.' });
      } finally {
        if (!controller.signal.aborted) setCheckingPolicy(false);
      }
    }
    void check();
    return () => controller.abort();
  }, [executionMode, mode, connected, agent.active, policyEndpoint, policyRetry]);
  const lastMessage = agent.chat_messages?.at(-1)?.id;
  useEffect(() => {
    if (followChat.current && transcript.current) transcript.current.scrollTop = transcript.current.scrollHeight;
  }, [lastMessage, mode]);
  useEffect(() => {
    if (!pending && !switching && focusMode.current === mode) {
      document.getElementById(`mode-${mode}`)?.focus();
      focusMode.current = null;
    }
  }, [mode, pending, switching]);
  const profileSignature = JSON.stringify(configuration.models);
  useEffect(() => {
    setEndpoint(configuration.endpoint);
    setOllamaEndpoint(configuration.ollama_endpoint);
    setReasoning(current => profile.reasoning_efforts.includes(current) ? current
      : profile.reasoning_efforts.includes('low') ? 'low' : profile.reasoning_efforts[0]);
    if (!addingModel) {
      setLabel(profile.label);
      setDeployment(profile.deployment);
      setProvider(profile.provider);
    }
  }, [configuration.endpoint, configuration.ollama_endpoint, profileSignature, profile.id, addingModel]);

  async function perform(path: string, body: unknown) {
    setPending(true);
    setError('');
    try { await request(path, body); return true; }
    catch (failure) { setError(String(failure)); return false; }
    finally { setPending(false); }
  }

  async function saveConnection(event: React.FormEvent) {
    event.preventDefault();
    const savedId = addingModel ? crypto.randomUUID() : profile.id;
    const edited = { id: savedId, label, deployment, provider,
      reasoning_efforts: provider === 'ollama' ? ['none'] : profile.provider === 'ollama' ? ['low', 'medium', 'high'] : profile.reasoning_efforts };
    const models = configuration.models.map(({ configured: _, ...entry }) => entry);
    const saved = await perform('agent/config', {
      endpoint, ollama_endpoint: ollamaEndpoint,
      models: addingModel ? [...models, edited] : models.map(entry => entry.id === savedId ? edited : entry),
    });
    if (saved) { setModelId(savedId); setAddingModel(false); }
  }

  async function switchMode(next: 'chat' | 'voice') {
    if (next === mode || switching) return;
    setSwitching(true);
    try {
      if (!await perform('agent/mode', { run_id: state.run_id, episode_epoch: state.episode_epoch, mode: next })) focusMode.current = null;
    }
    finally { setSwitching(false); }
  }

  async function sendMessage(event: React.FormEvent) {
    event.preventDefault();
    followChat.current = true;
    const sent = await perform('agent/chat', { run_id: state.run_id, episode_epoch: state.episode_epoch,
      model_id: profile.id, reasoning, goal, feedback_interval_s: interval, max_turns: turnLimit, message: message.trim(),
      execution_mode: executionMode,
      ...(executionMode === 'supervised_policy' ? { policy: { endpoint: policyEndpoint } } : {}),
      images_per_request: imagesPerRequest, context_tokens: contextTokens,
      conversation_id: agent.mode === 'chat' ? agent.session_id : null });
    if (sent) setMessage('');
  }

  const activeProfile = configuration.models.find(entry => entry.id === agent.model_id);
  const validContext = Number.isInteger(imagesPerRequest) && imagesPerRequest >= 1 && imagesPerRequest <= 8
    && Number.isInteger(contextTokens) && contextTokens >= 0 && contextTokens <= 32768;
  const navigation = state.navigation;
  const skill = state.skill;
  const cloudSupervisor = executionMode !== 'supervised_policy' || profile.provider === 'foundry';
  const policyAvailable = executionMode !== 'supervised_policy' || (policyCheck?.endpoint === policyEndpoint && policyCheck.ready && !checkingPolicy);
  const contextUsage = agent.mode !== 'voice' ? agent.context_usage : null;
  const skillLabels = { inspect_room: 'Inspect room', locate_doorway: 'Locate doorway', approach: 'Approach', cross: 'Cross' };
  const configurationStatus = profile.provider === 'ollama'
    ? !configuration.ollama_endpoint ? 'Ollama endpoint missing'
      : !profile.deployment.trim() ? 'Ollama model tag missing' : `Configured locally / ${profile.deployment}`
    : !configuration.endpoint ? 'Foundry endpoint missing'
    : !profile.deployment.trim() ? `${profile.label}: deployment name missing`
    : `Configured / ${profile.deployment}`;
  const voiceConfigurationStatus = !state.realtime.endpoint ? 'Realtime resource endpoint missing'
    : !state.realtime.deployment.trim() ? 'GPT Realtime 2 deployment name missing' : `Configured / ${state.realtime.deployment}`;
  return <section className="agent-section" aria-label="LLM control">
    <div className="panel-header">
      <h3><Bot size={17} /> Agent interaction <span className="tag">{localNavigation ? 'LOCAL SMOLVLA' : mode === 'chat' && profile.provider === 'ollama' ? 'OLLAMA LOCAL' : 'MICROSOFT FOUNDRY'}</span></h3>
      <span className={`agent-phase ${agent.phase === 'error' ? 'bad' : ''}`} role="status">{phaseLabels[agent.phase] ?? agent.phase}</span>
    </div>
    {(!localNavigation || agent.local_model) && <LocalModelProgress live={agent.local_model} connected={connected} />}
    <div className="interaction-tabs tabs" role="tablist" aria-label="Interaction mode">
      {(['chat', 'voice'] as const).map(option => <button key={option} id={`mode-${option}`} role="tab" type="button"
        aria-selected={mode === option} aria-controls={`panel-${option}`} tabIndex={mode === option ? 0 : -1} disabled={switching || pending || !connected}
        title={mode === option ? `${option === 'chat' ? 'Chat' : 'Voice'} mode` : 'Switch mode and stop current control'}
        onKeyDown={event => {
          if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
          event.preventDefault();
          const next = event.key === 'Home' ? 'chat' : event.key === 'End' ? 'voice' : mode === 'chat' ? 'voice' : 'chat';
          focusMode.current = next;
          void switchMode(next);
        }}
        onClick={() => void switchMode(option)}>{option === 'chat' ? <MessageSquare size={17} /> : <Mic size={17} />}{option === 'chat' ? 'Chat' : 'Voice'}</button>)}
    </div>
    <div id={`panel-${mode}`} role="tabpanel" aria-labelledby={`mode-${mode}`} aria-busy={switching}>
    <form className="agent-form" onSubmit={event => {
      event.preventDefault();
      if (mode !== 'chat') return;
      void perform('agent/start', { run_id: state.run_id, episode_epoch: state.episode_epoch,
        model_id: profile.id, reasoning, goal, feedback_interval_s: interval, max_turns: turnLimit, execution_mode: executionMode,
        ...(executionMode === 'supervised_policy' ? { policy: { endpoint: policyEndpoint } } : {}),
        images_per_request: imagesPerRequest, context_tokens: contextTokens });
    }}>
      {mode === 'chat' && <div className="execution-modes tabs" role="group" aria-label="Execution mode">
        {(['local_navigation', 'single_step', 'navigation_plan', 'supervised_policy'] as const).map(value => <button type="button" key={value}
          aria-pressed={executionMode === value} disabled={agent.active || state.busy || pending || switching}
          title={value === 'local_navigation' ? 'Local SmolVLA drives the current scene using its head camera, sensors and Robot goal' : value === 'single_step' ? 'One bounded action per model response' : value === 'navigation_plan' ? 'Feedback-checked navigation skills with a short, expiring motion buffer' : 'Cloud supervision with a local, Milo-trained left-arm policy'}
          onClick={() => {
            setExecutionMode(value);
            try { localStorage.setItem('milo-execution-mode', value); } catch {}
            if (value === 'local_navigation') { setInterval(1); setTurnLimit(30); }
            if (value === 'navigation_plan') setInterval(.25);
            if (value === 'supervised_policy') {
              setInterval(1);
              const supervisor = configuration.models.find(entry => entry.id === 'luna' && entry.provider === 'foundry')
                ?? configuration.models.find(entry => entry.provider === 'foundry');
              if (supervisor) { setModelId(supervisor.id); setReasoning(supervisor.reasoning_efforts.includes('low') ? 'low' : supervisor.reasoning_efforts[0]); }
            }
          }}>
          {value === 'local_navigation' ? <Cpu size={16} /> : value === 'single_step' ? <StepForward size={16} /> : value === 'navigation_plan' ? <Route size={16} /> : <Bot size={16} />}
          {value === 'local_navigation' ? 'Local SmolVLA navigation' : value === 'single_step' ? 'Single step' : value === 'navigation_plan' ? 'Navigation plan' : 'Luna + SmolVLA'}
        </button>)}
      </div>}
      {localNavigation && <div className="policy-readiness">
        <div className="panel-header"><h4>navigation-stop-balanced-1500 / local CUDA</h4>
          <button type="button" className="icon-button" aria-label="Unload local model" title="Unload SmolVLA and release GPU memory"
            disabled={!connected || pending || agent.active || state.busy || !residentModel || residentModel.phase === 'unloaded'}
            onClick={() => void perform('local-navigation/unload', {})}><Power size={16} /></button>
        </div>
        <p role="status" aria-label="Local model residency">{!connected ? 'Model residency unavailable' : residencyLabels[residentModel?.phase ?? 'unloaded']}</p>
        <p>{state.challenge?.title ?? 'Practice bench'} / {agent.active ? 'Local model controls the browser robot' : 'Start LLM control to run the current goal'}</p>
        <p className="context-note">Experimental navigation. Checkpoint validated for green-bay parking only; other challenge outcomes are unverified. Navigation only, no manipulation.</p>
      </div>}
      <div className={`agent-settings ${mode === 'voice' ? 'voice-settings' : ''}`}>
        {mode === 'chat' && executionMode === 'supervised_policy' && <label>SmolVLA endpoint<input
          aria-label="SmolVLA endpoint" type="url" required value={policyEndpoint} disabled={agent.active || pending}
          onChange={event => setPolicyEndpoint(event.target.value)} /></label>}
        {mode === 'chat' && !localNavigation && <>
        <label>Model<select aria-label="LLM model" value={profile.id} disabled={agent.active || pending} onChange={event => {
          setModelId(event.target.value); setAddingModel(false);
          const selected = configuration.models.find(entry => entry.id === event.target.value)!;
          setReasoning(selected.reasoning_efforts.includes('low') ? 'low' : selected.reasoning_efforts[0]);
        }}>{configuration.models.map(entry => <option value={entry.id} key={entry.id}>{entry.label}</option>)}</select></label>
        <label>Reasoning<select aria-label="Reasoning effort" value={reasoning} disabled={agent.active || pending} onChange={event => setReasoning(event.target.value as Reasoning)}>
          {profile.reasoning_efforts.map(effort => <option value={effort} key={effort}>{effort[0].toUpperCase() + effort.slice(1)}</option>)}
        </select></label>
        </>}
        <label>Feedback interval (s)<input aria-label="Feedback interval (s)" title="Minimum wall-clock interval between fresh model inputs; inference and motion may take longer." type="number" min={.25} max={30} step={.25} required value={interval} onChange={event => setInterval(Number(event.target.value))} /></label>
        <label>Turn limit<input aria-label="LLM turn limit" type="number" min={1} max={localNavigation ? 40 : 200} step={1} required value={turnLimit} disabled={agent.active || pending} onChange={event => setTurnLimit(Number(event.target.value))} /></label>
        {mode === 'chat' && !localNavigation && <>
          <label>Images per request<input aria-label="Images per request" type="number" min={1} max={8} step={1} required
            title="Maximum distinct head-camera frames per request. Current frame first, plus recent real observations as they become available. More images use more tokens."
            value={imagesPerRequest} disabled={agent.active || pending} onChange={event => setImagesPerRequest(Number(event.target.value))} /></label>
          <label>Retained context (tokens)<input aria-label="Retained context (tokens)" type="number" min={0} max={32768} step={1} required
            title="Conservative estimated text-token budget for past turns and action summaries. 0 clears text history. Current sensors, goal, tools and selected images are additional. Larger Ollama settings can use more memory."
            value={contextTokens} disabled={agent.active || pending} onChange={event => setContextTokens(Number(event.target.value))} /></label>
        </>}
      </div>
      {mode === 'chat' && executionMode === 'supervised_policy' && <div className="policy-readiness" aria-busy={checkingPolicy}>
        <div className="panel-header"><h4>Left-arm policy connection</h4>
          <button type="button" className="icon-button" aria-label="Check policy connection" title="Check local server and checkpoint compatibility without running inference"
            disabled={!connected || agent.active || pending || checkingPolicy} onClick={() => setPolicyRetry(value => value + 1)}><RefreshCw size={16} /></button>
        </div>
        <p>Left-arm pick/place only. For driving, select Local SmolVLA navigation.</p>
        <p role="status" aria-label="Policy connection status" className={policyAvailable ? 'ok' : 'bad'}>
          {!connected ? 'Connection unavailable' : checkingPolicy || policyCheck?.endpoint !== policyEndpoint
            ? 'Checking policy service...' : policyCheck.message}
        </p>
        {policyCheck?.checkpoint && <p>{policyCheck.checkpoint}</p>}
      </div>}
      <div className="agent-goal-row">
        {mode === 'chat' && <label>Robot goal<textarea aria-label="Robot goal" required maxLength={2000} rows={2} value={goal} disabled={agent.active || pending} onChange={event => setGoal(event.target.value)} /></label>}
        <div className="agent-actions">
          {agent.active ? <>
            <button type="button" disabled={pending || !connected || interval < .25 || interval > 30} onClick={() => perform('agent/rate', { feedback_interval_s: interval })}><Timer size={16} /> Apply rate</button>
            <button type="button" className="primary" disabled={pending || !connected} onClick={() => perform('agent/takeover', {})}><Hand size={16} /> Take manual control</button>
          </> : mode === 'chat' && <button type="submit" className="primary" disabled={pending || !connected || state.busy || (!localNavigation && (!profile.configured || !validContext)) || !goal.trim() || !cloudSupervisor || !policyAvailable}><Play size={16} />Start LLM control</button>}
          {!agent.active && agent.auto_wake && <button type="button" disabled={pending || !connected} onClick={() => perform('agent/takeover', {})}><Hand size={16} /> Take manual control</button>}
        </div>
      </div>
    </form>
    {mode === 'chat' && !localNavigation && <section className="request-context" aria-label="Request context">
      <div className="panel-header"><h4>Request context</h4>
        <span className="tag">{!connected ? 'Last received' : contextUsage ? `Request ${contextUsage.turn}` : 'No request yet'}</span>
      </div>
      {contextUsage ? <>
        <dl className="context-readings">
          <div title="Conservative UTF-8 byte estimate for replayed text and recent-action summaries. Excludes current instructions, sensors, tools and images; not the full model context window.">
            <dt>Retained history (est.)</dt>
            <dd>{contextUsage.retained_tokens_estimate.toLocaleString('en-US')} / {contextUsage.retained_budget.toLocaleString('en-US')} tokens</dd>
          </div>
          <div><dt>Retained turns</dt><dd>{contextUsage.retained_turns}</dd></div>
          <div><dt>Images sent</dt><dd>{contextUsage.images_sent} / {contextUsage.image_limit}</dd></div>
          <div title="Input tokens reported by the provider for this request, including its multimodal accounting. Not cumulative run usage. No full-window percentage is inferred.">
            <dt>Request input (reported)</dt><dd>{contextUsage.input_tokens != null
              ? `${contextUsage.input_tokens.toLocaleString('en-US')} tokens`
              : contextUsage.response_received ? 'Not reported' : 'Not yet reported'}</dd>
          </div>
        </dl>
        {contextUsage.retained_budget > 0 ? <meter aria-label="Retained history budget" min={0} max={contextUsage.retained_budget}
          value={Math.min(contextUsage.retained_tokens_estimate, contextUsage.retained_budget)}
          aria-valuetext={`${contextUsage.retained_tokens_estimate} of ${contextUsage.retained_budget} estimated retained-history tokens`}
          title="Retained-history budget only, not the full model context limit" />
          : <p className="context-note">History disabled</p>}
        <div className="context-notes">
          <span title="This deployment's full input/output context limit is not configured; history budget is only one part of a request.">Full model window: not configured</span>
          {contextUsage.instructions_repeated && <span title="System instructions and the active run command are resent on each request outside the trimmed history.">Instructions resent each request</span>}
        </div>
        <details><summary>Active command (resent)</summary><p>{contextUsage.active_command}</p></details>
      </> : <p className="context-note">No request measurements available</p>}
    </section>}
    {mode === 'chat' && (executionMode === 'navigation_plan' || !!navigation?.steps.length) && <section className="navigation-plan" aria-label="Navigation plan">
      <div className="panel-header"><h4><ListChecks size={16} /> Task plan</h4>
        <button type="button" className="icon-button" aria-label="Cancel navigation plan" title="Cancel plan, discard buffered motion, and take manual control"
          disabled={!agent.active || pending || !connected} onClick={() => void perform('agent/takeover', {})}><Square size={16} /></button>
      </div>
      <div className="navigation-metrics">
        <span>Buffer <strong>{navigation?.remaining_s.toFixed(2) ?? '0.00'} s</strong></span>
        <span>Expires in <strong>{navigation?.expires_in_s.toFixed(2) ?? '0.00'} s</strong></span>
        <span>Revision <strong>{navigation?.revision ?? 0}</strong></span>
      </div>
      <progress aria-label="Buffered motion" max={2} value={navigation?.remaining_s ?? 0} />
      <p className="navigation-state" role="status" aria-label="Navigation status">{connected ? navigation?.reason || 'No plan submitted' : 'Navigation state unavailable'}</p>
      <ol>{navigation?.steps.map((step, index) => <li key={index} data-status={step.status} aria-current={navigation.active_step === index ? 'step' : undefined}>
        <div><strong>{skillLabels[step.skill]}</strong><span>{step.status}</span></div><p>{step.goal}</p>
        {step.evidence && <details><summary>Model evidence</summary><p>{step.evidence}</p></details>}
      </li>)}</ol>
    </section>}
    {mode === 'chat' && (executionMode === 'supervised_policy' || skill) && <section className="navigation-plan" aria-label="SmolVLA skill">
      <div className="panel-header"><h4><Bot size={16} /> Local manipulation</h4>
        <button type="button" className="icon-button" aria-label="Cancel policy skill" title="Stop supervision and policy motion"
          disabled={!agent.active || pending || !connected} onClick={() => void perform('agent/takeover', {})}><Square size={16} /></button>
      </div>
      <div className="navigation-metrics">
        <span>Buffer <strong>{skill?.remaining_s.toFixed(2) ?? '0.00'} s</strong></span>
        <span>Policy <strong>{skill?.policy_latency_s?.toFixed(2) ?? '-'} s</strong></span>
        <span>Requests <strong>{skill?.policy_requests ?? 0}</strong></span>
        <span>Rejected <strong>{skill?.rejected_chunks ?? 0}</strong></span>
      </div>
      <progress aria-label="Policy motion buffer" max={1} value={skill?.remaining_s ?? 0} />
      <p className="navigation-state" role="status" aria-label="Policy skill status">{!connected ? 'Policy state unavailable' : !cloudSupervisor ? 'Cloud supervisor required' : skill?.reason ?? 'Milo-trained checkpoint required'}</p>
      {skill?.instruction && <p>{skill.instruction}</p>}
      {skill?.checkpoint && <div className="exchange-meta">{skill.checkpoint}</div>}
      {skill?.completion_source === 'supervisor' && <p>Supervisor-reported completion</p>}
    </section>}
    {mode === 'chat' && !localNavigation && <section className="chat-section" aria-label="Chat control">
      <div className="chat-transcript" ref={transcript} role="log" aria-label="Chat conversation" aria-live="polite" aria-relevant="additions text"
        tabIndex={0} onScroll={event => {
          const element = event.currentTarget;
          followChat.current = element.scrollHeight - element.clientHeight - element.scrollTop < 20;
        }}>
        {(agent.chat_messages ?? []).map(entry => <article className={`chat-message chat-${entry.role}`} key={entry.id}>
          <strong>{entry.role === 'user' ? 'You' : 'Milo'}</strong><p>{entry.text}</p>
        </article>)}
      </div>
      {agent.active && agent.mode === 'chat' && <span className="chat-progress" role="status">{phaseLabels[agent.phase] ?? 'Responding'}</span>}
      <form className="chat-composer" onSubmit={event => void sendMessage(event)}>
        <label>Message Milo<textarea aria-label="Chat message" rows={2} maxLength={2000} required value={message}
          disabled={pending || switching || !connected} onChange={event => setMessage(event.target.value)} /></label>
        <button type="submit" className="primary icon-button" aria-label="Send chat message" title="Send chat message"
          disabled={pending || switching || !connected || agent.active || state.busy || !profile.configured || !message.trim() || !goal.trim() || !validContext || !cloudSupervisor || !policyAvailable || interval < .25 || interval > 30 || turnLimit < 1 || turnLimit > 200}><Send size={20} /></button>
      </form>
    </section>}
    {mode === 'voice' && !switching && <VoiceControl state={state} connected={connected} request={request} interval={interval} maxTurns={turnLimit} />}
    </div>
    <div className="agent-metrics">
      <span className="model-connection-status">{localNavigation ? 'Local SmolVLA / navigation checkpoint' : agent.active ? agent.mode === 'voice' ? `${state.realtime.deployment} / voice / ${state.realtime.reasoning_effort}` : `${activeProfile?.label ?? agent.model_id} / ${agent.reasoning}` : mode === 'voice' ? voiceConfigurationStatus : configurationStatus}</span>
      <span>Target interval <strong>{(agent.active ? agent.feedback_interval_s : interval).toFixed(2)} s</strong></span>
      <span>Actual interval <strong>{agent.observed_interval_s?.toFixed(2) ?? '-'} s</strong></span>
      <span>Inference <strong>{agent.inference_latency_s?.toFixed(2) ?? '-'} s</strong></span>
      <span>Turns <strong>{agent.turns} / {agent.session_id ? agent.max_turns : turnLimit}</strong></span>
      {agent.auto_wake && <span className="idle-monitor">Camera quiet <strong>{agent.camera_unchanged_s.toFixed(1)} s</strong> / Auto wake armed</span>}
    </div>
    {(error || agent.error) && <div className="error" role="alert">{error || agent.error}</div>}
    {agent.message && agent.mode !== 'chat' && <p className="agent-message">{agent.message}</p>}
    {mode === 'chat' && !localNavigation && <details className="agent-connection">
      <summary><Settings2 size={15} /> Model connection</summary>
      <form onSubmit={saveConnection}>
        <fieldset disabled={agent.active || pending}>
          <label>Provider<select aria-label="Model provider" value={provider} onChange={event => setProvider(event.target.value as 'foundry' | 'ollama')}>
            <option value="foundry">Microsoft Foundry</option><option value="ollama">Ollama (local)</option>
          </select></label>
          {provider === 'ollama'
            ? <label>Local endpoint<input aria-label="Ollama endpoint" type="url" required value={ollamaEndpoint} placeholder="http://127.0.0.1:11434" onChange={event => setOllamaEndpoint(event.target.value)} /></label>
            : <label>Resource or project endpoint<input aria-label="Foundry endpoint" type="url" required value={endpoint} placeholder="https://your-resource.services.ai.azure.com/api/projects/your-project" onChange={event => setEndpoint(event.target.value)} /></label>}
          <label>Model label<input aria-label="Model label" required maxLength={120} value={label} onChange={event => setLabel(event.target.value)} /></label>
          <label>{provider === 'ollama' ? 'Model tag' : 'Deployment name'}<input aria-label={provider === 'ollama' ? 'Ollama model tag' : 'Foundry deployment name'} required maxLength={120} value={deployment} onChange={event => setDeployment(event.target.value)} /></label>
          <div className="agent-actions">
            <button type="submit"><Check size={16} /> {addingModel ? 'Add model' : 'Apply connection'}</button>
            <button type="button" title={addingModel ? 'Cancel new model' : 'Add another model deployment'} disabled={configuration.models.length >= 20} onClick={() => {
              setAddingModel(!addingModel); setLabel(addingModel ? profile.label : ''); setDeployment(addingModel ? profile.deployment : '');
              setProvider(addingModel ? profile.provider : 'foundry');
            }}><Plus size={16} /> {addingModel ? 'Cancel' : 'New model'}</button>
          </div>
        </fieldset>
      </form>
    </details>}
    <ExchangeFeed agent={agent} />
  </section>;
}