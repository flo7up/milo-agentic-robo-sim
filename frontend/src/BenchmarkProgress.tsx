import { useState } from 'react';
import { ArrowRight, ChevronDown, Equal, LockKeyhole, TrendingDown, TrendingUp, TriangleAlert } from 'lucide-react';
import type { Batch, Trial } from './TestResults';

const median = (values: number[]) => {
  const sorted = [...values].sort((first, second) => first - second);
  return sorted.length ? (sorted[Math.floor(sorted.length / 2)] + sorted[Math.floor((sorted.length - 1) / 2)]) / 2 : null;
};
const label = (batch: Batch) => `${new Date(batch.date).toLocaleDateString()} / ${batch.architecture?.version_key ?? batch.design} / ${batch.name.split('/').at(-1)}`;
const eligible = (batch: Batch) => batch.benchmark?.comparison?.eligible === true && !batch.source_changed && !batch.running;
function preferences(): Record<string, string> {
  try {
    const value = JSON.parse(localStorage.getItem('milo-benchmark-baselines') ?? '{}');
    return value && typeof value === 'object' && !Array.isArray(value) ? value : {};
  } catch { return {}; }
}
function pairedTimes(baseline: Batch, candidate: Batch, taskId: string) {
  const before: number[] = [], after: number[] = [];
  for (const trial of candidate.trials) {
    const prior = baseline.trials.find(item => item.case_id === trial.case_id);
    if (trial.benchmark?.task_id === taskId && trial.benchmark.status === 'passed' && prior?.benchmark?.status === 'passed'
      && trial.elapsed_s !== null && prior.elapsed_s !== null) {
      before.push(prior.elapsed_s); after.push(trial.elapsed_s);
    }
  }
  return { before: median(before), after: median(after), count: before.length };
}

export function BenchmarkProgress({ batches, onInspect }: { batches: Batch[]; onInspect: (batch: Batch, trial?: Trial) => void }) {
  const [suiteId, setSuiteId] = useState('');
  const [baselineId, setBaselineId] = useState('');
  const [candidateId, setCandidateId] = useState('');
  const [saved, setSaved] = useState(preferences);
  const [storageError, setStorageError] = useState('');
  const [limit, setLimit] = useState(3);
  const benchmarks = batches.filter(batch => batch.benchmark);
  const suites = [...new Set(benchmarks.map(batch => batch.benchmark!.suite_id))].sort();
  const selectedSuite = suites.includes(suiteId) ? suiteId : suites.find(identity => benchmarks.some(batch => batch.benchmark?.suite_id === identity && eligible(batch))) ?? suites[0];
  const series = benchmarks.filter(batch => batch.benchmark!.suite_id === selectedSuite);
  const choices = series.filter(eligible).sort((first, second) => first.date.localeCompare(second.date));
  const preferred = choices.find(batch => saved[batch.benchmark!.comparison!.cohort_id] === batch.id);
  const baseline = choices.find(batch => batch.id === baselineId) ?? preferred ?? choices[0];
  const cohort = baseline?.benchmark?.comparison?.cohort_id;
  const candidates = series.filter(batch => eligible(batch) && batch.id !== baseline?.id && batch.benchmark!.comparison!.cohort_id === cohort)
    .sort((first, second) => second.date.localeCompare(first.date));
  const candidate = candidates.find(batch => batch.id === candidateId) ?? candidates[0];
  const excluded = series.filter(batch => !eligible(batch) || (cohort && batch.benchmark!.comparison!.cohort_id !== cohort));
  const rows = baseline?.benchmark?.tasks.map(prior => {
    const current = candidate?.benchmark?.tasks.find(task => task.task_id === prior.task_id);
    return { prior, current, delta: current ? current.passed - prior.passed : null,
      timing: candidate ? pairedTimes(baseline, candidate, prior.task_id) : null };
  }) ?? [];
  function chooseBaseline(identity: string) {
    setBaselineId(identity); setCandidateId(''); setLimit(3);
    const batch = choices.find(item => item.id === identity)!;
    const next = { ...saved, [batch.benchmark!.comparison!.cohort_id]: identity };
    setSaved(next);
    try { localStorage.setItem('milo-benchmark-baselines', JSON.stringify(next)); setStorageError(''); }
    catch { setStorageError('Baseline selected for this visit; browser storage unavailable.'); }
  }
  function inspectTask(batch: Batch, taskId: string) {
    const trials = batch.trials.filter(trial => trial.benchmark?.task_id === taskId);
    onInspect(batch, trials.find(trial => trial.benchmark?.status !== 'passed') ?? trials[0]);
  }
  return <section className="benchmark-progress" aria-label="Baseline progress">
    <div className="benchmark-heading"><div><h2>Capability Progress</h2><span>Frozen test series</span></div><span className="benchmark-evidence">{baseline ? baseline.evidence.replaceAll('_', ' ') : 'No comparable series'}</span></div>
    {!benchmarks.length ? <p role="status">No standardized baseline series recorded. Individual trials remain in Run History.</p> : <>
      <div className="benchmark-selectors">
        <label>Test series<select aria-label="Benchmark series" value={selectedSuite} onChange={event => { setSuiteId(event.target.value); setBaselineId(''); setCandidateId(''); setLimit(3); }}>{suites.map(identity => <option key={identity}>{identity}</option>)}</select></label>
        <label><span><LockKeyhole size={14}/> Baseline</span><select aria-label="Baseline run" value={baseline?.id ?? ''} disabled={!choices.length} onChange={event => chooseBaseline(event.target.value)}>{!choices.length && <option value="">No eligible baseline</option>}{choices.map(batch => <option key={batch.id} value={batch.id}>{label(batch)}</option>)}</select></label>
        <label>Compare with<select aria-label="Candidate run" value={candidate?.id ?? ''} disabled={!candidates.length} onChange={event => setCandidateId(event.target.value)}>{!candidates.length && <option value="">No matched candidate</option>}{candidates.map(batch => <option key={batch.id} value={batch.id}>{label(batch)}</option>)}</select></label>
      </div>
      {storageError && <p role="status">{storageError}</p>}
      {baseline && <>
        <dl className="benchmark-summary">
          <div><dt>Improved tasks</dt><dd>{candidate ? rows.filter(row => row.delta! > 0).length : 'Not measured'}</dd></div>
          <div><dt>Regressed tasks</dt><dd>{candidate ? rows.filter(row => row.delta! < 0).length : 'Not measured'}</dd></div>
          <div><dt>Blocked attempts</dt><dd>{(candidate ?? baseline).benchmark!.tasks.reduce((total, task) => total + task.blocked, 0)}</dd></div>
          <div><dt>Matched series runs</dt><dd>{candidates.length + 1}</dd></div>
        </dl>
        {!candidate && <p className="results-warning" role="status"><TriangleAlert size={16}/>Baseline recorded; no complete matched candidate yet.</p>}
        <div className="results-table-scroll"><table className="results-table benchmark-comparison" aria-label="Task changes against baseline">
          <thead><tr><th>Capability</th><th>Baseline pass / planned</th><th>Candidate pass / planned</th><th>Change</th><th>Remaining gaps</th><th>Paired successful time</th></tr></thead>
          <tbody>{rows.map(({prior, current, delta, timing}) => <tr key={prior.task_id} data-change={delta === null ? 'unmeasured' : delta > 0 ? 'improved' : delta < 0 ? 'regressed' : 'unchanged'}>
            <th scope="row"><button onClick={() => inspectTask(candidate ?? baseline, prior.task_id)}>{prior.title}<small>Inspect trials <ArrowRight size={12}/></small></button></th>
            <td>{prior.passed} / {prior.planned}</td><td>{current ? `${current.passed} / ${current.planned}` : 'Not run'}</td>
            <td><span className="benchmark-change">{delta === null ? 'Not measured' : delta > 0 ? <><TrendingUp size={16}/>+{delta} passes</> : delta < 0 ? <><TrendingDown size={16}/>{delta} passes</> : <><Equal size={16}/>Unchanged</>}</span></td>
            <td>{(current ?? prior).failed} failed / {(current ?? prior).blocked} blocked<small>{(current ?? prior).invalid} invalid / {(current ?? prior).not_run} not run</small></td>
            <td>{timing?.count ? <>{timing.before!.toFixed(2)} s <ArrowRight size={12}/> {timing.after!.toFixed(2)} s<small>{timing.count} identical successful cases</small></> : 'No paired successes'}</td>
          </tr>)}</tbody>
        </table></div>
        <div className="benchmark-section-heading"><h3>Series History</h3><span>{selectedSuite}</span></div>
        <div className="results-table-scroll"><table className="results-table benchmark-history" aria-label="Benchmark series history"><thead><tr><th>Run</th>{rows.map(row => <th key={row.prior.task_id}>{row.prior.title}</th>)}</tr></thead><tbody>
          {[baseline, ...candidates].slice(0, limit).map(batch => <tr key={batch.id}><th scope="row"><button onClick={() => onInspect(batch)}>{batch.id === baseline.id ? 'Baseline' : batch.architecture?.version_key ?? batch.design}<small>{new Date(batch.date).toLocaleString()}</small></button></th>{rows.map(({prior}) => {
            const task = batch.benchmark!.tasks.find(item => item.task_id === prior.task_id)!;
            return <td key={prior.task_id} data-result={task.failed || task.invalid ? 'failed' : task.blocked ? 'blocked' : task.passed === task.planned ? 'passed' : 'missing'}><button onClick={() => inspectTask(batch, prior.task_id)}>{task.passed} / {task.planned}<small>{task.blocked ? `${task.blocked} blocked` : task.failed ? `${task.failed} failed` : task.not_run ? `${task.not_run} not run` : 'passed'}</small></button></td>;
          })}</tr>)}
        </tbody></table></div>
        {candidates.length + 1 > limit && <div className="results-more"><button onClick={() => setLimit(value => value + 5)}><ChevronDown size={16}/>Show more</button><button onClick={() => setLimit(candidates.length + 1)}>View all ({candidates.length + 1})</button></div>}
        <details className="results-source"><summary>Comparison provenance</summary><p>Suite {baseline.benchmark!.suite_sha256}</p><p>Map {baseline.benchmark!.map_sha256}</p><p>Matched suite, map, fixtures, evidence and recorded runtime/settings. Shared workload and model randomness remain uncontrolled.</p></details>
      </>}
      {!!excluded.length && <details className="benchmark-excluded" open={!baseline}><summary>Not comparable ({excluded.length})</summary><ul>{excluded.map(batch => <li key={batch.id}><button onClick={() => onInspect(batch)}>{batch.name.split('/').at(-1)}</button><span>{batch.benchmark?.comparison?.reasons.length ? batch.benchmark.comparison.reasons.join('; ') : eligible(batch) ? 'Different map, fixtures, evidence or recorded settings' : 'Comparison provenance not recorded'}</span></li>)}</ul></details>}
    </>}
  </section>;
}