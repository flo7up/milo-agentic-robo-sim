import { Bot, ChevronRight, LoaderCircle } from 'lucide-react';
import type { AgentState } from './types';

export function ModelActivity({ agent, connected, modelName, onCalls, onDetails }: {
  agent: AgentState; connected: boolean; modelName: string; onCalls: () => void; onDetails: () => void;
}) {
  const budget = agent.inference_budget;
  const supervisor = agent.task_supervision;
  const hasRun = !!agent.session_id;
  const thinking = connected && agent.active && agent.phase === 'thinking';
  const calls = budget ? budget.requests + (supervisor?.requests ?? 0) : null;
  const tokens = agent.input_tokens + agent.output_tokens + (supervisor?.tokens ?? 0);
  const incomplete = budget?.usage_unknown || supervisor?.usage_unknown;
  const limitReached = !!budget && (budget.requests >= budget.max_requests || budget.tokens >= budget.max_tokens);
  const latest = agent.run_messages?.filter(message => message.source === 'model').at(-1);
  const activeModel = agent.configuration.models.find(model => model.id === agent.model_id)?.label ?? agent.model_id;
  const label = hasRun ? `${activeModel}${supervisor ? ' + Luna supervision' : ''}` : modelName;
  const status = !connected ? 'Disconnected — showing last received activity'
    : agent.error ? 'Model needs attention'
    : thinking ? 'Waiting for a model response'
    : agent.active ? 'Robot working' : hasRun ? 'Run ended' : 'Ready when you are';
  return <section className="model-activity" aria-label="Model activity" data-thinking={thinking}>
    <div className="model-activity-heading"><span className="model-activity-icon" aria-hidden="true">
      {thinking ? <LoaderCircle size={19} className="loading-icon" /> : <Bot size={19} />}
    </span><div><h3>Model activity</h3><p>{label || 'No model selected'}</p></div></div>
    <p className="model-status" role="status">{status}</p>
    <div className="model-summary" role="group" aria-label="Run statistics">
      <button type="button" onClick={onCalls} aria-label="Model calls: view details"><strong>{calls ?? (hasRun ? '—' : 0)}</strong><span>Model calls <ChevronRight size={13} /></span></button>
      <button type="button" onClick={onDetails} aria-label="Reported tokens: view usage"><strong>{tokens.toLocaleString('en-US')}</strong><span>Reported tokens <ChevronRight size={13} /></span></button>
      <div><strong>{agent.inference_latency_s == null ? '—' : `${agent.inference_latency_s.toFixed(1)} s`}</strong><span>Latest response</span></div>
    </div>
    {incomplete && <p className="model-notice" role="status">Usage incomplete — some requests have no reported token count.</p>}
    {limitReached && <p className="model-notice" role="status">Run limit reached. Open Details to review usage and limits.</p>}
    {agent.error && <p className="error" role="alert">{agent.error}</p>}
    {latest && <details className="model-latest"><summary>Latest model update</summary><p>{latest.text}</p></details>}
    <button className="model-drilldown" type="button" onClick={onCalls}>View model calls <ChevronRight size={15} /></button>
  </section>;
}
