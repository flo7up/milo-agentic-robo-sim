import { useEffect, useRef, useState } from 'react';
import { Bot, Check, Hand, ListChecks, MessageSquare, Mic, Play, Plus, Route, Send, Settings2, Square, StepForward, Timer } from 'lucide-react';
import { ExchangeFeed } from './ExchangeFeed';
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
  const [executionMode, setExecutionMode] = useState<ExecutionMode>(agent.execution_mode ?? 'single_step');
  const [interval, setInterval] = useState(agent.feedback_interval_s);
  const [goal, setGoal] = useState(state.challenge?.goal ?? 'Inspect the area, approach a visible cube, and stop before contact.');
  const [turnLimit, setTurnLimit] = useState(state.challenge?.suggested_turn_limit ?? 30);
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
      conversation_id: agent.mode === 'chat' ? agent.session_id : null });
    if (sent) setMessage('');
  }

  const activeProfile = configuration.models.find(entry => entry.id === agent.model_id);
  const navigation = state.navigation;
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
      <h3><Bot size={17} /> Agent interaction <span className="tag">{mode === 'chat' && profile.provider === 'ollama' ? 'OLLAMA LOCAL' : 'MICROSOFT FOUNDRY'}</span></h3>
      <span className={`agent-phase ${agent.phase === 'error' ? 'bad' : ''}`} role="status">{phaseLabels[agent.phase] ?? agent.phase}</span>
    </div>
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
        model_id: profile.id, reasoning, goal, feedback_interval_s: interval, max_turns: turnLimit, execution_mode: executionMode });
    }}>
      {mode === 'chat' && <div className="execution-modes tabs" role="group" aria-label="Execution mode">
        {(['single_step', 'navigation_plan'] as const).map(value => <button type="button" key={value}
          aria-pressed={executionMode === value} disabled={agent.active || state.busy || pending || switching}
          title={value === 'single_step' ? 'One bounded action per model response' : 'Feedback-checked navigation skills with a short, expiring motion buffer'}
          onClick={() => { setExecutionMode(value); if (value === 'navigation_plan') setInterval(.25); }}>
          {value === 'single_step' ? <StepForward size={16} /> : <Route size={16} />}{value === 'single_step' ? 'Single step' : 'Navigation plan'}
        </button>)}
      </div>}
      <div className={`agent-settings ${mode === 'voice' ? 'voice-settings' : ''}`}>
        {mode === 'chat' && <>
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
        <label>Turn limit<input aria-label="LLM turn limit" type="number" min={1} max={200} step={1} required value={turnLimit} disabled={agent.active || pending} onChange={event => setTurnLimit(Number(event.target.value))} /></label>
      </div>
      <div className="agent-goal-row">
        {mode === 'chat' && <label>Robot goal<textarea aria-label="Robot goal" required maxLength={2000} rows={2} value={goal} disabled={agent.active || pending} onChange={event => setGoal(event.target.value)} /></label>}
        <div className="agent-actions">
          {agent.active ? <>
            <button type="button" disabled={pending || !connected || interval < .25 || interval > 30} onClick={() => perform('agent/rate', { feedback_interval_s: interval })}><Timer size={16} /> Apply rate</button>
            <button type="button" className="primary" disabled={pending || !connected} onClick={() => perform('agent/takeover', {})}><Hand size={16} /> Take manual control</button>
          </> : mode === 'chat' && <button type="submit" className="primary" disabled={pending || !connected || state.busy || !profile.configured || !goal.trim()}><Play size={16} /> Start LLM control</button>}
          {!agent.active && agent.auto_wake && <button type="button" disabled={pending || !connected} onClick={() => perform('agent/takeover', {})}><Hand size={16} /> Take manual control</button>}
        </div>
      </div>
    </form>
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
    {mode === 'chat' && <section className="chat-section" aria-label="Chat control">
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
          disabled={pending || switching || !connected || agent.active || state.busy || !profile.configured || !message.trim() || !goal.trim() || interval < .25 || interval > 30 || turnLimit < 1 || turnLimit > 200}><Send size={20} /></button>
      </form>
    </section>}
    {mode === 'voice' && !switching && <VoiceControl state={state} connected={connected} request={request} interval={interval} maxTurns={turnLimit} />}
    </div>
    <div className="agent-metrics">
      <span className="model-connection-status">{agent.active ? agent.mode === 'voice' ? `${state.realtime.deployment} / voice / ${state.realtime.reasoning_effort}` : `${activeProfile?.label ?? agent.model_id} / ${agent.reasoning}` : mode === 'voice' ? voiceConfigurationStatus : configurationStatus}</span>
      <span>Target interval <strong>{(agent.active ? agent.feedback_interval_s : interval).toFixed(2)} s</strong></span>
      <span>Actual interval <strong>{agent.observed_interval_s?.toFixed(2) ?? '-'} s</strong></span>
      <span>Inference <strong>{agent.inference_latency_s?.toFixed(2) ?? '-'} s</strong></span>
      <span>Turns <strong>{agent.turns} / {agent.session_id ? agent.max_turns : turnLimit}</strong></span>
      {agent.auto_wake && <span className="idle-monitor">Camera quiet <strong>{agent.camera_unchanged_s.toFixed(1)} s</strong> / Auto wake armed</span>}
    </div>
    {(error || agent.error) && <div className="error" role="alert">{error || agent.error}</div>}
    {agent.message && agent.mode !== 'chat' && <p className="agent-message">{agent.message}</p>}
    {mode === 'chat' && <details className="agent-connection">
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