import { CheckCircle2, CircleStop, Info, TriangleAlert } from 'lucide-react';
import type { LiveState } from './types';

export function OutcomeFeedback({ state, connected }: { state: LiveState; connected: boolean }) {
  const challenge = state.challenge;
  const report = state.agent.outcome;
  const newerAction = !!(report && state.result && state.result.observation.wall_timestamp > report.timestamp);
  let outcome: { kind: string; title: string; message: string; evidence: string } | null = null;
  if (!connected) {
    outcome = { kind: 'warning', title: 'Connection lost', message: 'Current robot state and task outcome are unavailable.', evidence: 'Last received state' };
  } else if (state.agent.error && !newerAction) {
    outcome = { kind: 'error', title: 'Agent error', message: state.agent.error, evidence: 'Controller report' };
  } else if (challenge?.status === 'completed') {
    outcome = { kind: 'success', title: 'Task completed', message: challenge.title,
      evidence: `${challenge.completed_objectives} / ${challenge.progress.length} objectives verified by physics${state.manual_placements ? ' / Operator-assisted episode' : ''}` };
  } else if (challenge?.status === 'failed') {
    outcome = { kind: 'error', title: 'Task failed', message: challenge.progress.find(item => !item.complete)?.detail ?? challenge.title,
      evidence: 'Scenario evaluation' };
  } else if (report && !newerAction && (!state.agent.active || ['voice_ready', 'sleeping'].includes(state.agent.phase))) {
    const labels: Record<string, [string, string]> = {
      completed: ['warning', 'Agent reports completion'], unachievable: ['warning', 'Task cannot be completed'],
      limited: ['warning', 'Turn limit reached'], interrupted: ['stopped', 'Run interrupted'],
      error: ['error', 'Agent error'], ended: ['info', 'Response finished'], failed: ['error', 'Task failed'],
    };
    const [kind, title] = labels[report.kind] ?? ['info', 'Run ended'];
    if (report.source !== 'physics') outcome = { kind, title, message: report.message,
      evidence: report.source === 'agent' ? challenge ? 'Agent report / scenario not verified complete' : 'Agent report / not independently verified' : 'Controller report' };
  }
  if (!outcome && !state.busy && !state.agent.active && state.result) {
    const result = state.result;
    outcome = result.status === 'ok'
      ? { kind: 'info', title: 'Action completed', message: result.message || `Command finished in ${result.actual_duration_s.toFixed(2)} simulated seconds.`, evidence: 'Tool result / not a task-completion claim' }
      : { kind: result.status === 'cancelled' ? 'stopped' : 'warning', title: result.status === 'cancelled' ? 'Action cancelled' : 'Action blocked',
        message: result.message || result.error || 'The command did not complete.', evidence: result.error ?? 'Tool result' };
  }
  if (!outcome && connected && state.stopped) outcome = { kind: 'stopped', title: 'Robot stopped', message: 'Motion is disabled.', evidence: 'Operator control' };
  if (!outcome) return null;
  const Icon = outcome.kind === 'success' ? CheckCircle2 : ['warning', 'error'].includes(outcome.kind) ? TriangleAlert : outcome.kind === 'stopped' ? CircleStop : Info;
  return <div className="outcome-feedback" data-outcome={outcome.kind} role="status" aria-label="Task outcome" aria-live="polite" aria-atomic="true">
    <Icon size={28} aria-hidden="true" />
    <div className="outcome-body"><h3>{outcome.title}</h3><p>{outcome.message}</p><span>{outcome.evidence}</span></div>
  </div>;
}