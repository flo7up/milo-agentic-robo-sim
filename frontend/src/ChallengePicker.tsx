import { useEffect, useState } from 'react';
import { CheckCircle2, Circle, FolderOpen, Target } from 'lucide-react';
import type { ChallengeId, ChallengePreset, LiveState } from './types';

export function ChallengePicker({ state, connected, request }: {
  state: LiveState; connected: boolean; request: (path: string, body?: unknown) => Promise<unknown>;
}) {
  const [presets, setPresets] = useState<ChallengePreset[]>([]);
  const [selected, setSelected] = useState<ChallengeId>(state.challenge?.id ?? 'bench');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  useEffect(() => {
    const controller = new AbortController();
    async function loadPresets() {
      try {
        const response = await fetch('/api/challenges', { signal: controller.signal });
        if (!response.ok) throw new Error('Challenge options could not be loaded.');
        setPresets(await response.json());
      } catch (failure) { if (!controller.signal.aborted) setError(String(failure)); }
    }
    void loadPresets();
    return () => controller.abort();
  }, []);
  useEffect(() => { setSelected(state.challenge?.id ?? 'bench'); }, [state.run_id]);

  const preset = presets.find(entry => entry.id === selected);
  const loaded = state.challenge?.id === selected ? state.challenge : null;
  async function load() {
    setLoading(true);
    setError('');
    try { await request('challenges/load', { challenge_id: selected }); }
    catch (failure) { setError(String(failure)); }
    finally { setLoading(false); }
  }

  return <section className="challenge-section" aria-label="Predefined challenges">
    <div className="challenge-toolbar">
      <label><Target size={16} /> Challenge<select aria-label="Predefined challenge" value={selected} disabled={loading} onChange={event => setSelected(event.target.value as ChallengeId)}>
        <option value="bench">Practice bench</option>
        {presets.map(entry => <option value={entry.id} key={entry.id}>{entry.title}</option>)}
      </select></label>
      <span className="tag">{preset?.skill ?? 'Free practice'}</span>
      <button type="button" disabled={!connected || loading || !presets.length} title="Stop current control and load a fresh episode" onClick={() => void load()}><FolderOpen size={16} /> {loading ? 'Loading...' : 'Load challenge'}</button>
      {loaded && <span className={`challenge-status ${loaded.status === 'completed' ? 'ok' : loaded.status === 'failed' ? 'bad' : ''}`} role="status">{loaded.status === 'completed' ? 'Completed' : loaded.status === 'failed' ? 'Failed' : `${loaded.completed_objectives} / ${loaded.progress.length} goals complete`}</span>}
    </div>
    <p className="challenge-goal">{preset?.goal ?? 'Free practice with one cube. No scored objective.'}</p>
    {preset && <ul className="challenge-objectives">{preset.objectives.map((label, index) => {
      const progress = loaded?.progress[index];
      return <li key={label}><span className={progress?.complete ? 'ok' : ''}>{progress?.complete ? <CheckCircle2 size={16} /> : <Circle size={16} />}</span><span>{label}</span><small>{progress?.detail ?? 'Not loaded'}</small></li>;
    })}</ul>}
    {error && <div className="error" role="alert">{error}</div>}
  </section>;
}