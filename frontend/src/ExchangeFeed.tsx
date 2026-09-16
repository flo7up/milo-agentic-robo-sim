import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { ArrowRight, Bot, Camera, CheckCheck, ChevronRight, Copy, ListFilter, Radio, RefreshCw, Wrench } from 'lucide-react';
import type { AgentState, ExchangeEntry, ExchangeFeed as FeedData } from './types';
import { MotionDiagnostics, RouteFailureDetails } from './MotionDiagnostics';

const channels = {
  session: { icon: Radio, from: 'Controller', to: '' },
  feedback: { icon: Camera, from: 'Robot', to: 'LLM' },
  response: { icon: Bot, from: 'LLM', to: 'Controller' },
  tool: { icon: Wrench, from: 'Controller', to: 'Robot' },
  result: { icon: CheckCheck, from: 'Robot', to: 'Controller' },
  policy: { icon: Bot, from: 'Controller', to: 'Mission' },
};
const filters = ['All', 'Inputs', 'LLM', 'Tools', 'Policy', 'Session', 'Failures'] as const;

function hasRecordedFailure(entry: ExchangeEntry) {
  if (entry.kind === 'policy') return !!entry.payload.task?.route_failures?.length;
  if (entry.kind === 'feedback') return !!entry.payload.observation.spatial?.task?.route_failures?.length
    || entry.payload.observation.navigation?.status === 'failed'
    || ['watchdog','collision_monitor'].includes(entry.payload.observation.navigation?.diagnostics?.stop?.initiator ?? '');
  if (entry.kind === 'result') return !['ok','cancelled'].includes(entry.payload.result.status);
  return entry.kind === 'session' && entry.payload.status === 'error';
}

function Payload({ title, value }: { title: string; value: unknown }) {
  return <details className="exchange-payload"><summary>{title}</summary><pre>{typeof value === 'string' ? value : JSON.stringify(value, null, 2)}</pre></details>;
}

function entryPreview(entry: ExchangeEntry): string {
  if (entry.kind === 'feedback') return `Frame ${entry.payload.observation.seq} / ${entry.payload.images_in_request} image(s) / ${entry.payload.observation.simulated_time_s.toFixed(2)} s simulated`;
  if (entry.kind === 'response') return [entry.payload.status, `${entry.payload.latency_s.toFixed(2)} s`,
    entry.payload.refusals.join(' ') || entry.payload.text || entry.payload.calls.map(call => {
      const decision = reportedDecision(call.arguments);
      return decision ? `${decision.action?.replaceAll('_', ' ') ?? call.name}: ${decision.reason}` : call.name;
    }).join('; ')].filter(Boolean).join(' / ');
  if (entry.kind === 'tool') return entry.payload.tool;
  if (entry.kind === 'result') return [entry.payload.result.status, entry.payload.result.error, entry.payload.result.message || entry.payload.tool].filter(Boolean).join(' / ');
  if (entry.kind === 'policy') return entry.payload.task?.reason || ['status', 'action', 'reason', 'message'].map(key => entry.payload[key]).filter(value => typeof value === 'string').join(' / ');
  return entry.payload.reason || entry.payload.message || entry.payload.goal || entry.payload.status || '';
}

function reportedDecision(arguments_: string) {
  let decision: Record<string, unknown>;
  try {
    const parsed: unknown = JSON.parse(arguments_);
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return null;
    decision = parsed as Record<string, unknown>;
  } catch { return null; }
  const reason = typeof decision.reason === 'string' ? decision.reason : null;
  const action = typeof decision.action === 'string' ? decision.action : typeof decision.intent === 'string' ? decision.intent : null;
  if (!reason) return null;
  return {action, reason};
}

function DecisionSummary({ arguments: arguments_ }: { arguments: string }) {
  const decision = reportedDecision(arguments_);
  if (!decision) return null;
  const {action, reason} = decision;
  return <div className="decision-summary">
    <strong>{action ? action.replaceAll('_', ' ') : 'Proposed decision'}</strong>
    <span>Reported reason</span><p>{reason}</p>
  </div>;
}

function ExchangeContent({ entry }: { entry: ExchangeEntry }) {
  if (entry.kind === 'policy') {
    return <>
      <RouteFailureDetails task={entry.payload.task}/>
      {entry.image_url && <a href={entry.image_url} target="_blank" rel="noreferrer">
        <img src={entry.image_url} alt={`Policy input camera, event ${entry.id}`} width={160} height={120} loading="lazy" />
      </a>}
      <Payload title="Policy payload" value={entry.payload} />
    </>;
  }
  if (entry.kind === 'feedback') {
    const { observation } = entry.payload;
    return <>
      <div className="exchange-observation">
        <a href={entry.image_url!} target="_blank" rel="noreferrer" title={`Open input camera frame ${observation.seq}`}>
          <img src={entry.image_url!} alt={`LLM input camera, turn ${entry.turn}, frame ${observation.seq}`} width={160} height={120} loading="lazy" />
        </a>
        <dl className="exchange-sensors">
          <div><dt>Frame</dt><dd>{observation.seq}</dd></div>
          <div><dt>Simulated time</dt><dd>{observation.simulated_time_s.toFixed(2)} s</dd></div>
          <div><dt>Head yaw / pitch</dt><dd>{observation.head_rad.map(value => value.toFixed(2)).join(' / ')} rad</dd></div>
          {observation.joints && <div><dt>Joint readings</dt><dd>{observation.joints.length}</dd></div>}
          <div><dt>Bumpers</dt><dd>{observation.bumpers.join(', ') || 'Clear'}</dd></div>
          {observation.proximity && <div><dt>Distance beams</dt><dd>{observation.proximity.distances.length} / {observation.proximity.max_range_m} m range</dd></div>}
          {observation.battery && <div><dt>Battery</dt><dd>{observation.battery.charge_pct.toFixed(1)}% / {observation.battery.charging ? 'Charging' : observation.battery.low ? 'Low' : 'Ready'}</dd></div>}
          {Object.entries(observation.grippers ?? {}).map(([side, sensor]) => <div key={side}><dt>{side} gripper</dt><dd>{(sensor.aperture_m * 1000).toFixed(0)} mm / {sensor.load_n.toFixed(2)} N</dd></div>)}
        </dl>
      </div>
      {!!entry.image_urls && entry.image_urls.length > 1 && <div className="exchange-camera-batch" aria-label="Additional model input images">
        {entry.image_urls.slice(1).map((url, index) => {
          const map = entry.payload.image_roles?.[index + 1] === 'observed_map';
          const frame = entry.payload.camera_frames?.[index + 1];
          const historyCount = entry.payload.camera_history?.frames.length ?? 0;
          const sheet = historyCount > 0 && index === 0;
          const original = index === (historyCount > 0 ? 1 : 0) ? entry.payload.historical_original : null;
          const label = map ? 'Observed map supplied to Luna' : sheet ? `Motion history / ${historyCount} views` : original ? `Historical original ${original.frame_id}` : `Historical input camera, frame ${frame?.seq ?? index + 1}`;
          return <figure key={url}><a href={url} target="_blank" rel="noreferrer" title={`Open ${label}`}>
            <img src={url} alt={label} width={map ? 256 : 128} height={map ? 276 : sheet ? 128 : 96} style={map ? {aspectRatio:'512/552'} : sheet ? {aspectRatio:'1'} : undefined} loading="lazy" />
          </a><figcaption>{map || sheet ? label : original ? `Original / ${original.simulated_time_s.toFixed(2)} s` : `Frame ${frame?.seq ?? index + 1} / ${frame?.simulated_time_s.toFixed(2) ?? '-'} s`}</figcaption></figure>;
        })}
      </div>}
      <div className="exchange-meta"><span>{entry.payload.context_mode === 'realtime_conversation' ? 'Realtime conversation' : `Replayed turns: ${entry.payload.history_turns.join(', ') || 'None'}`}</span><span>{entry.payload.images_in_request} image(s) in request</span></div>
      {entry.payload.context_tokens !== undefined && <div className="exchange-meta" title={entry.payload.context_estimator}>
        Retained context: {entry.payload.retained_context_tokens_estimate ?? 0} / {entry.payload.context_tokens} tokens (est.)
      </div>}
      {entry.payload.memory_frame_seq != null && <div className="exchange-meta">Remembered initial frame: {entry.payload.memory_frame_seq}</div>}
      {entry.payload.tool_result_call_ids.length > 0 && <div className="exchange-call-ids">Included tool results: {entry.payload.tool_result_call_ids.join(', ')}</div>}
      {entry.payload.collision_feedback && <p className="exchange-text bad">{entry.payload.collision_feedback.guidance}</p>}
      {observation.navigation?.diagnostics && <MotionDiagnostics connected={false} recorded={observation.navigation.diagnostics}/>}
      {observation.spatial?.localization && <dl className="policy-facts">
        <div><dt>Recorded localization</dt><dd>{observation.spatial.localization.status}</dd></div>
        <div><dt>Laser sample age</dt><dd>{observation.spatial.localization.age_s.toFixed(3)} s / {observation.spatial.localization.sample_clock ?? 'clock not recorded'}</dd></div>
        <div><dt>Tracking</dt><dd>{observation.spatial.localization.tracking_method ?? 'Not recorded'}</dd></div>
        <div><dt>Continuous pose correction</dt><dd>{observation.spatial.localization.continuous_pose_correction === false ? 'No' : observation.spatial.localization.continuous_pose_correction === true ? 'Yes' : 'Not recorded'}</dd></div>
      </dl>}
      {entry.payload.observed_map_snapshot && <dl className="policy-facts">
        <div title={entry.payload.observed_map_snapshot.age_basis}><dt>Map capture age</dt><dd>{entry.payload.observed_map_snapshot.age_s.toFixed(3)} s / {entry.payload.observed_map_snapshot.capture_clock ?? 'clock not recorded'}</dd></div>
        <div title={entry.payload.observed_map_snapshot.geometry_age_basis}><dt>Latest geometry integration age</dt><dd>{entry.payload.observed_map_snapshot.geometry_age_s == null ? 'Not recorded' : `${entry.payload.observed_map_snapshot.geometry_age_s.toFixed(3)} s / Unix-derived`}</dd></div>
      </dl>}
      <RouteFailureDetails task={observation.spatial?.task}/>
      <Payload title="Sensor payload" value={observation} />
      {!!entry.payload.recent_actions?.length && <Payload title="Recent actions supplied" value={entry.payload.recent_actions} />}
      <Payload title="Request context" value={{ ...entry.payload, observation: undefined }} />
    </>;
  }
  if (entry.kind === 'response') {
    const output = entry.payload;
    return <>
      {output.text && <p className="exchange-text">{output.text}</p>}
      {output.calls.map(call => <DecisionSummary key={call.call_id} arguments={call.arguments} />)}
      {output.refusals.map((refusal, index) => <p className="exchange-text bad" key={index}>{refusal}</p>)}
      <div className="exchange-meta"><span>{output.status}</span><span>{output.latency_s.toFixed(2)} s inference</span><span>{output.input_tokens ?? '-'} in / {output.output_tokens ?? '-'} out tokens</span>{output.text_truncated && <span>Text truncated</span>}</div>
      {output.calls.length > 0 && <Payload title={`Tool calls (${output.calls.length}${output.calls_truncated ? '+' : ''})`} value={output.calls} />}
    </>;
  }
  if (entry.kind === 'tool') {
    return <><strong className="exchange-tool-name">{entry.payload.tool}</strong><pre className="exchange-arguments">{JSON.stringify(entry.payload.arguments, null, 2)}</pre><span className="exchange-call-ids">{entry.payload.call_id}</span></>;
  }
  if (entry.kind === 'result') {
    const result = entry.payload.result;
    return <>
      <div className="exchange-result"><span className={result.status === 'ok' ? 'ok' : 'bad'}>{result.status}</span><strong className="exchange-tool-name">{entry.payload.tool}</strong><span>{(result.actual_duration_s ?? 0).toFixed(2)} s motion</span>{result.observation && <span>Frame {result.observation.seq}</span>}</div>
      {result.message && <p className="exchange-text">{result.message}</p>}
      {entry.payload.model_result?.collision_feedback && <p className="exchange-text bad">{entry.payload.model_result.collision_feedback.guidance}</p>}
      <Payload title="Result payload" value={{ call_id: entry.payload.call_id, ...result }} />
      {entry.payload.model_result && <Payload title="Reply sent to model" value={entry.payload.model_result} />}
    </>;
  }
  const context = entry.payload;
  return <>
    {context.goal && <p className="exchange-text">{context.goal}</p>}
    {(context.reason || context.message) && <p className={`exchange-text ${context.status === 'error' ? 'bad' : ''}`}>{context.reason || context.message}</p>}
    <div className="exchange-meta">
      {context.model && <span>{context.model} / {context.reasoning}</span>}
      {context.feedback_interval_s !== undefined && <span>{context.feedback_interval_s.toFixed(2)} s feedback interval</span>}
      {context.max_turns && <span>{context.max_turns} turn limit</span>}
      {context.status && <span>{context.status}</span>}
    </div>
    {context.instructions && <Payload title="Controller instructions" value={context.instructions} />}
    {context.tools && <Payload title={`Available tools (${context.tools.length})`} value={context.tools} />}
    <Payload title="Session payload" value={context} />
  </>;
}

export function ExchangeFeed({ agent, visible = true }: { agent: AgentState; visible?: boolean }) {
  const [feed, setFeed] = useState<FeedData>({ session_id: null, revision: 0, first_id: 1, capacity: 200, events: [] });
  const [filter, setFilter] = useState<typeof filters[number]>('All');
  const [follow, setFollow] = useState(true);
  const [error, setError] = useState('');
  const [copying, setCopying] = useState(false);
  const [copyMessage, setCopyMessage] = useState('');
  const [retry, setRetry] = useState(0);
  const cursor = useRef({ sessionId: null as string | null, revision: 0 });
  const viewport = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (cursor.current.sessionId !== agent.session_id) {
      cursor.current = { sessionId: agent.session_id, revision: 0 };
      setFeed({ session_id: agent.session_id, revision: 0, first_id: 1, capacity: 200, events: [] });
      setFollow(true);
      setError('');
      setCopyMessage('');
    }
    if (!agent.session_id || cursor.current.revision >= agent.trace_revision) return;
    const controller = new AbortController();
    const sessionId = agent.session_id;
    const query = new URLSearchParams({ session_id: sessionId, after: String(cursor.current.revision) });
    async function load() {
      try {
        const response = await fetch(`/api/agent/trace?${query}`, { signal: controller.signal });
        if (response.status === 409) return;
        if (!response.ok) throw new Error(`Exchange feed unavailable (${response.status})`);
        const incoming: FeedData = await response.json();
        if (controller.signal.aborted || incoming.session_id !== cursor.current.sessionId) return;
        cursor.current.revision = incoming.revision;
        setFeed(previous => {
          const retained = previous.session_id === sessionId ? previous.events.filter(entry => entry.id >= incoming.first_id) : [];
          const entries = new Map([...retained, ...incoming.events].map(entry => [entry.id, entry]));
          return { ...incoming, events: [...entries.values()].sort((first, second) => first.id - second.id) };
        });
        setError('');
      } catch (failure) {
        if (!controller.signal.aborted) setError(String(failure));
      }
    }
    void load();
    return () => controller.abort();
  }, [agent.session_id, agent.trace_revision, retry]);

  const entries = feed.session_id === agent.session_id ? feed.events : [];
  const shown = entries.filter(entry => filter === 'All' || (filter === 'Inputs' && entry.kind === 'feedback') ||
    (filter === 'LLM' && entry.kind === 'response') || (filter === 'Tools' && ['tool', 'result'].includes(entry.kind)) ||
    (filter === 'Policy' && entry.kind === 'policy') ||
    (filter === 'Failures' && hasRecordedFailure(entry)) ||
    (filter === 'Session' && entry.kind === 'session'));
  useLayoutEffect(() => {
    if (visible && follow && viewport.current) viewport.current.scrollTop = viewport.current.scrollHeight;
  }, [feed.revision, follow, filter, visible]);

  async function copyFeed() {
    const sessionId = agent.session_id;
    const copiedEntries = entries;
    setCopying(true);
    setCopyMessage('');
    try {
      await navigator.clipboard.writeText(JSON.stringify({
        format: 'milo-exchange-feed-v1', exported_at: new Date().toISOString(),
        session_id: sessionId, model_id: agent.model_id, reasoning: agent.reasoning,
        revision: feed.revision, first_id: feed.first_id, capacity: feed.capacity,
        earlier_events_discarded: feed.first_id > 1,
        image_note: 'Camera URLs reference this local session; image pixels are not embedded.',
        events: copiedEntries,
      }, null, 2));
      if (cursor.current.sessionId === sessionId) setCopyMessage(`Copied ${copiedEntries.length} exchanges`);
    } catch {
      if (cursor.current.sessionId === sessionId) setCopyMessage('Copy failed. Allow clipboard access and retry.');
    } finally {
      setCopying(false);
    }
  }

  return <section className="exchange-section" aria-label="Exchange feed">
    <div className="panel-header"><h3><Radio size={16} /> Inputs, decisions & actions</h3><span className="exchange-count">{entries.length} events{feed.first_id > 1 && ` / latest ${feed.capacity}`}</span></div>
    <div className="exchange-toolbar">
      <div className="exchange-filters" role="group" aria-label="Exchange filter"><ListFilter size={14} />{filters.map(option => <button key={option} type="button" aria-pressed={filter === option} onClick={() => setFilter(option)}>{option}</button>)}</div>
      <div className="exchange-actions"><button type="button" className="icon-button" aria-label="Copy exchange feed"
        title="Copy all retained exchanges as JSON, including hidden payloads and other filters"
        disabled={copying || entries.length === 0} onClick={() => void copyFeed()}><Copy size={16} /></button>
      <label className="exchange-follow"><input type="checkbox" checked={follow} onChange={event => setFollow(event.target.checked)} /> Follow latest</label></div>
    </div>
    {copyMessage && <p role="status" className="exchange-meta">{copyMessage}</p>}
    {error && <div className="exchange-error" role="alert"><span>{error}</span><button type="button" className="icon-button" title="Retry loading exchanges" aria-label="Retry loading exchanges" onClick={() => setRetry(value => value + 1)}><RefreshCw size={16} /></button></div>}
    <div className="exchange-feed" role="log" aria-label="Robot and LLM exchanges" aria-live="off" ref={viewport} tabIndex={0} onScroll={() => {
      if (!visible) return;
      const element = viewport.current!;
      if (follow && element.scrollHeight - element.scrollTop - element.clientHeight > 64) setFollow(false);
    }}>
      {shown.length ? shown.map(entry => {
        const channel = entry.title === 'Luna navigation decision'
          ? { icon: Bot, from: 'Luna', to: 'SmolVLA' } : channels[entry.kind];
        const Icon = channel.icon;
        return <article className={`exchange-entry exchange-${entry.kind}`} key={`${feed.session_id}:${entry.id}`} data-kind={entry.kind} data-turn={entry.turn}>
          <details className="exchange-disclosure" onToggle={event => {if (event.currentTarget.open) setFollow(false);}}>
          <summary aria-label={`Event ${entry.id}: ${entry.title}`}>
          <div className="exchange-icon"><Icon size={16} /></div>
          <div className="exchange-body">
            <div className="exchange-heading"><span className="exchange-direction">{channel.from}{channel.to && <><ArrowRight size={12} aria-hidden="true" /><span>{channel.to}</span></>}</span><span className="exchange-turn">Turn {entry.turn}</span><time dateTime={new Date(entry.timestamp * 1000).toISOString()}>{new Date(entry.timestamp * 1000).toLocaleTimeString('en-GB', { hour12: false })}</time></div>
            <strong className="exchange-title">{entry.title}</strong>
            <span className={`exchange-preview ${entry.kind === 'result' && entry.payload.result.status !== 'ok' ? 'bad' : ''}`}>{entryPreview(entry)}</span>
          </div>
          <ChevronRight className="disclosure-chevron" size={15} />
          </summary>
          <div className="exchange-detail"><ExchangeContent entry={entry} /></div>
          </details>
        </article>;
      }) : <p className="empty">{entries.length ? 'No matching exchanges.' : 'No exchanges in this session.'}</p>}
    </div>
  </section>;
}