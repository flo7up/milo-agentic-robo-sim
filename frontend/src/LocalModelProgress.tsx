import { useEffect, useState } from 'react';
import { Check, Cpu, TriangleAlert } from 'lucide-react';
import type { LiveState, LocalModelStatus } from './types';

const phaseLabels: Record<LocalModelStatus['phase'], string> = {
  checking: 'Checking navigation checkpoint',
  loading: 'Loading model on GPU',
  warming: 'Warming up model and preparing scene',
  supervising: 'Luna reviewing camera and planning the next subgoal',
  running: 'Model ready / navigation test running',
  saving: 'Saving test results',
  completed: 'Test finished',
  failed: 'Local test failed / check the test terminal',
  interrupted: 'Test interrupted / no recent heartbeat',
};

export function LocalModelProgress({ live, connected = true, resident }: {
  live?: LocalModelStatus | null; connected?: boolean; resident?: LiveState['local_navigation_model'];
}) {
  const [separateTest, setTest] = useState<LocalModelStatus | null>(null);
  const [statusError, setError] = useState('');
  const hasLive = !!live;
  useEffect(() => {
    if (hasLive) return;
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    let deadline: ReturnType<typeof setTimeout>;
    let controller: AbortController;
    async function poll() {
      controller = new AbortController();
      deadline = setTimeout(() => controller.abort(), 5000);
      try {
        const response = await fetch('/api/local-navigation/status', { signal: controller.signal, cache: 'no-store' });
        if (!response.ok) throw new Error(response.status === 404
          ? 'Restart the app backend to enable local model status.'
          : 'Local model status unavailable / reconnecting');
        const data: { test: LocalModelStatus | null } = await response.json();
        if (active) { setTest(data.test); setError(''); }
      } catch (failure) {
        if (active) setError(failure instanceof Error && failure.name !== 'AbortError'
          ? failure.message : 'Local model status unavailable / reconnecting');
      } finally {
        clearTimeout(deadline);
        if (active) timer = setTimeout(poll, 1000);
      }
    }
    void poll();
    return () => { active = false; clearTimeout(timer); clearTimeout(deadline); controller?.abort(); };
  }, [hasLive]);

  const test = live ?? separateTest;
  const error = live ? connected ? '' : 'Connection lost / local controller status unavailable' : statusError;
  if (!test && !error) return null;
  const failed = !!error || test?.phase === 'failed' || test?.phase === 'interrupted';
  const finished = test?.phase === 'completed';
  const running = test?.phase === 'running';
  const indeterminate = !failed && !finished && !running;
  const maximum = running ? test.request_limit : 1;
  const value = running ? test.requests_completed : finished ? 1 : 0;
  const preparing = live && (test?.phase === 'loading' || test?.phase === 'warming');
  const label = error || (preparing && resident?.phase === 'inferencing' ? 'Model already loaded / waiting for the previous prediction'
    : preparing && resident?.phase === 'ready' ? 'Reusing loaded model / preparing this run'
    : live && test?.phase === 'running' ? 'Model connected / driving browser robot'
    : live && test?.phase === 'interrupted' ? 'Local navigation stopped'
    : live && test?.phase === 'failed' ? 'Local navigation failed / robot stopped'
    : live && finished ? test.success ? 'Selected challenge completed' : 'Run ended / goal completion not verified'
    : finished ? test.success ? 'Test complete / parked successfully' : 'Test complete / parking not achieved'
    : test ? phaseLabels[test.phase] : 'Local model status unavailable');
  const StatusIcon = failed ? TriangleAlert : finished && test.success ? Check : Cpu;
  return <section className="local-model-progress" aria-label="Local model startup" data-phase={error ? 'unavailable' : test?.phase}>
    <div className="local-model-heading"><StatusIcon size={18} aria-hidden="true" /><h3>{live ? 'Luna + SmolVLA navigation' : 'Local navigation test'}</h3><span className="tag">{live ? 'Browser robot' : 'Separate scene'}</span></div>
    <div className="local-model-status"><strong role="status" aria-live="polite">{label}</strong>
      {test && !error && <span>{test.elapsed_s} s elapsed</span>}</div>
    {!failed && <div className={`local-model-meter ${indeterminate ? 'indeterminate' : ''}`} role="progressbar"
      aria-label="Local model progress" aria-valuemin={0} aria-valuemax={maximum}
      aria-valuenow={indeterminate ? undefined : value} aria-valuetext={label}>
      <span style={indeterminate ? undefined : { width: `${100 * value / maximum}%` }} />
    </div>}
    {test && !error && <div className="local-model-details"><span>{test.checkpoint} / {live ? test.challenge_title ?? 'Current scene' : `case ${test.case}`}</span>
      <span>{test.requests_completed} / {test.request_limit} requests</span></div>}
  </section>;
}