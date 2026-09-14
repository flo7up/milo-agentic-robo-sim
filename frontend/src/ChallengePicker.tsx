import { useEffect, useId, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { CheckCircle2, Circle, CircleStop, FolderOpen, ImageOff, LoaderCircle, RotateCcw, Target, X } from 'lucide-react';
import type { ChallengeEnvironment, ChallengeId, ChallengePreset, LiveState } from './types';

const scenarioNotes: Record<ChallengeId, { summary: string; completion: string }> = {
  bench: { summary: 'An open practice area with a cube for driving, camera movement and arm control.', completion: 'Free practice, without scored objectives.' },
  park: { summary: 'A short parking course with two posts and a green destination bay.', completion: 'Stop with the entire base and both wheels inside the bay.' },
  local_park: { summary: 'A simple green-bay course with wide posts, used for local-model parking experiments.', completion: 'Park fully in the green bay, grounded and at rest.' },
  tidy: { summary: 'A single red cube beside a blue drop zone. A focused pick-and-place task.', completion: 'Lift the cube, then release it fully inside the blue zone and let it settle.' },
  sort: { summary: 'Two colored cubes and matching drop zones for a two-object manipulation task.', completion: 'Lift and place each cube in its matching zone, released and settled.' },
  recharge: { summary: 'Leave a cyan charging pad, explore beyond a screen, then return using remembered landmarks.', completion: 'Complete the survey, return after the low-battery warning and recharge to 90%.' },
  apartment: { summary: 'Search connected rooms for a yellow cube on a pedestal, with a red cube as a distractor.', completion: 'Stop within 0.9 m and keep the yellow target visible for one simulated second.' },
  kitchen_bathroom: { summary: 'Identify kitchen fixtures, travel through the hall and find the bathroom.', completion: 'Park the whole base and both wheels inside the bathroom arrival area.' },
  clinic_delivery: { summary: 'Navigate a furnished clinic through reception and diagnostics, avoiding a maintenance barrier.', completion: 'Visit two checkpoints in order, then park in the treatment bay. No carried payload.' },
  warehouse: { summary: 'A large warehouse with offset stock racks and three colored route checkpoints.', completion: 'Visit blue, orange and green in order, then park for one simulated second.' },
  inspection: { summary: 'An equipment gallery with partitions, cabinets and a hidden yellow inspection target.', completion: 'Reject the red decoy and inspect the yellow target from a stationary, unobstructed view.' },
  workshop: { summary: 'Sort two cubes among workbench fixtures, divider blocks and parts racks.', completion: 'Lift both cubes and release them fully inside their matching floor zones.' },
  pedestrian_crossing: { summary: 'A walking person crosses the robot\'s route before a green parking bay.', completion: 'Yield without contact, let the person cross, then park fully in the bay.' },
  flat_kitchen: { summary: 'A furnished five-room flat with a central hallway and multiple doorways. Find the kitchen.', completion: 'Park fully inside the kitchen and remain at rest for half a simulated second.' },
  furniture_circuit: { summary: 'A spacious room with a table, sofa, chair and floor lamp. Choose one object to circle.', completion: 'Complete a lap in the selected direction without contact, then stop near the circuit start.' },
};

function ScenarioImage({ id, title }: { id: string; title: string }) {
  const [status, setStatus] = useState<'loading' | 'ready' | 'error'>('loading');
  const [attempt, setAttempt] = useState(0);
  return <div className="scenario-thumbnail" data-state={status}>
    <img src={`/scenario-previews/${id}.webp${attempt ? `?retry=${attempt}` : ''}`} alt={`Scene preview: ${title}`}
      width={480} height={300} decoding="async" onLoad={() => setStatus('ready')} onError={() => setStatus('error')} />
    {status === 'loading' && <LoaderCircle className="loading-icon scenario-image-status" size={20} aria-label="Loading scene preview" />}
    {status === 'error' && <div className="scenario-image-error"><ImageOff size={20} /><span>Preview unavailable</span>
      <button type="button" aria-label="Retry scene preview" title="Retry scene preview" onClick={() => { setStatus('loading'); setAttempt(value => value + 1); }}><RotateCcw size={16} /></button></div>}
  </div>;
}

export function ChallengePicker({ state, connected, request, onLoadingChange }: {
  state: LiveState; connected: boolean; request: (path: string, body?: unknown) => Promise<unknown>;
  onLoadingChange?: (loading: boolean) => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const menuId = useId();
  const [menuOpen, setMenuOpen] = useState(false);
  const [presets, setPresets] = useState<ChallengePreset[]>([]);
  const [environment, setEnvironment] = useState<ChallengeEnvironment>(state.challenge?.environment ?? 'standalone');
  const [selected, setSelected] = useState<ChallengeId>(state.challenge?.id ?? 'bench');
  const [orbitTarget, setOrbitTarget] = useState(state.challenge?.orbit?.target ?? 'table');
  const [orbitDirection, setOrbitDirection] = useState(state.challenge?.orbit?.direction ?? 'clockwise');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  useEffect(() => {
    const controller = new AbortController();
    setPresets([]);
    async function loadPresets() {
      try {
        const response = await fetch(environment === 'standalone' ? '/api/challenges' : `/api/challenges?environment=${environment}`, { signal: controller.signal });
        if (!response.ok) throw new Error('Challenge options could not be loaded.');
        setPresets(await response.json());
      } catch (failure) { if (!controller.signal.aborted) setError(String(failure)); }
    }
    void loadPresets();
    return () => controller.abort();
  }, [environment]);
  useEffect(() => {
    setSelected(state.challenge?.id ?? 'bench');
    setEnvironment(state.challenge?.environment ?? 'standalone');
    setOrbitTarget(state.challenge?.orbit?.target ?? 'table');
    setOrbitDirection(state.challenge?.orbit?.direction ?? 'clockwise');
  }, [state.run_id]);

  const preset = presets.find(entry => entry.id === selected);
  const shared = environment === 'shared_apartment_v1';
  const loaded = state.challenge?.id === selected && (state.challenge.environment ?? 'standalone') === environment && (selected !== 'furniture_circuit'
    || (state.challenge.orbit?.target === orbitTarget && state.challenge.orbit.direction === orbitDirection)) ? state.challenge : null;
  const objectives = selected === 'furniture_circuit' ? [`Circle the ${orbitTarget} ${orbitDirection} once, then stop`] : preset?.objectives;
  const notes = scenarioNotes[selected];
  const summary = shared ? selected === 'recharge'
    ? 'Leave the living-room charging pad, survey the hall and return to recharge.'
    : selected === 'furniture_circuit' ? 'Circle the living-room table in the shared apartment.'
    : selected === 'apartment' ? 'Search the shared apartment for the yellow cube; reject the red decoy.'
    : 'Find the kitchen among the shared apartment\'s furnished rooms.' : notes.summary;
  const title = preset?.title ?? 'Practice bench';
  async function load() {
    setLoading(true);
    onLoadingChange?.(true);
    setError('');
    try { await request('challenges/load', { challenge_id: selected, environment,
      ...(selected === 'furniture_circuit' ? {orbit_target: orbitTarget, orbit_direction: orbitDirection} : {}) });
      dialog.current?.close(); }
    catch (failure) { setError(String(failure)); }
    finally { setLoading(false); onLoadingChange?.(false); }
  }

  return <div className="challenge-menu" id="setup">
    <button type="button" aria-label="Choose challenge" aria-haspopup="dialog" aria-controls={menuId} aria-expanded={menuOpen}
      onClick={() => { dialog.current?.showModal(); setMenuOpen(true); }}><FolderOpen size={16} />Challenges</button>
    {createPortal(<dialog ref={dialog} id={menuId} className="challenge-menu-dialog" aria-labelledby={`${menuId}-title`}
      onClose={() => setMenuOpen(false)} onCancel={event => { if (loading) event.preventDefault(); }}>
      <div className="challenge-menu-heading">
        <h2 id={`${menuId}-title`}>Load challenge</h2>
        <button type="button" className="stop-button" aria-label="Stop robot" title="Stop robot"
          onClick={() => { void request('stop', {}).catch(failure => setError(String(failure))); }}><CircleStop size={17} />Stop</button>
        <button type="button" className="icon-button" aria-label="Close challenge menu" title="Close challenge menu" disabled={loading}
          onClick={() => dialog.current?.close()}><X size={18} /></button>
      </div>
      <section className="challenge-section scenario-setup" aria-label="Predefined challenges" aria-busy={loading}>
    <div className="challenge-toolbar">
      <label>Environment<select aria-label="Training environment" value={environment} disabled={!connected || loading}
        onChange={event => {
          const next = event.target.value as ChallengeEnvironment;
          setEnvironment(next);
          if (next === 'shared_apartment_v1') {
            if (!['furniture_circuit', 'apartment', 'flat_kitchen', 'recharge'].includes(selected)) setSelected('furniture_circuit');
            setOrbitTarget('table');
          }
        }}><option value="standalone">Standalone scenarios</option><option value="shared_apartment_v1">Shared Apartment V1</option></select></label>
      <label><Target size={16} /> Scenario<select aria-label="Predefined challenge" value={selected} disabled={!connected || loading} onChange={event => setSelected(event.target.value as ChallengeId)}>
        {!shared && <option value="bench">Practice bench</option>}
        {['Navigation', 'Perception', 'Manipulation'].map(category => <optgroup label={category} key={category}>
          {presets.filter(entry => (entry.category ?? 'Navigation') === category).map(entry =>
            <option value={entry.id} key={entry.id}>{entry.title}</option>)}
        </optgroup>)}
      </select></label>
      {selected === 'furniture_circuit' && <>
        <label>Object<select aria-label="Object to circle" value={orbitTarget} disabled={!connected || loading}
          onChange={event => setOrbitTarget(event.target.value as typeof orbitTarget)}>
          {(shared ? ['table'] : ['table','sofa','chair','floor lamp']).map(target => <option key={target} value={target}>{target}</option>)}
        </select></label>
        <label>Direction<select aria-label="Circuit direction" value={orbitDirection} disabled={!connected || loading}
          onChange={event => setOrbitDirection(event.target.value as typeof orbitDirection)}>
          <option value="clockwise">Clockwise</option><option value="counterclockwise">Counterclockwise</option>
        </select></label>
      </>}
      {preset?.difficulty === 'Advanced' && <span className="tag">Advanced</span>}
      <button type="button" disabled={!connected || loading || !presets.length} title="Stop current control and load a fresh episode" onClick={() => void load()}>{loading ? <LoaderCircle size={16} className="loading-icon" /> : <FolderOpen size={16} />} {loading ? 'Loading...' : 'Load challenge'}</button>
      {loaded && <span className={`challenge-status ${loaded.status === 'completed' ? 'ok' : loaded.status === 'failed' ? 'bad' : ''}`} role="status">{loaded.status === 'completed' ? 'Completed' : loaded.status === 'failed' ? 'Failed' : `${loaded.completed_objectives} / ${loaded.progress.length} goals complete`}</span>}
    </div>
    <div className="scenario-overview" aria-label="Selected scenario preview">
      <ScenarioImage key={shared ? environment : selected} id={shared ? environment : selected} title={shared ? 'Shared Apartment V1' : title} />
      <div className="scenario-brief">
        <div className="scenario-brief-heading"><strong>{title}</strong><span className="scenario-selection-state">{selected === 'bench' ? !state.challenge ? 'Loaded' : 'Not loaded' : loaded ? 'Loaded' : 'Not loaded'}</span></div>
        <div className="scenario-facts"><span>{preset?.category ?? 'Practice'}</span>{preset && <><span>{preset.difficulty ?? 'Foundation'}</span><span>{objectives?.length ?? 0} {objectives?.length === 1 ? 'objective' : 'objectives'}</span></>}</div>
        <p className="scenario-summary">{summary}</p>
        <p className="scenario-completion"><span>{selected === 'bench' ? 'Scoring' : 'Complete when'}</span>{selected === 'furniture_circuit'
          ? `Circle the ${orbitTarget} ${orbitDirection} once without contact, then stop near the circuit start for half a simulated second.`
          : notes.completion}</p>
      </div>
    </div>
    <details className="challenge-details">
    <summary>Goal & objectives</summary>
    <p className="challenge-goal">{selected === 'furniture_circuit' && (orbitTarget !== preset?.orbit?.target || orbitDirection !== preset?.orbit?.direction)
      ? `Identify the ${orbitTarget}, drive one complete ${orbitDirection} circuit around it as quickly as safely possible, then stop for half a second. Keep the arms stowed and avoid contact.`
      : preset?.goal ?? 'Free practice with one cube. No scored objective.'}</p>
    {objectives && <ul className="challenge-objectives">{objectives.map((label, index) => {
      const progress = loaded?.progress[index];
      return <li key={label}><span className={progress?.complete ? 'ok' : ''}>{progress?.complete ? <CheckCircle2 size={16} /> : <Circle size={16} />}</span><span>{label}</span><small>{progress?.detail ?? 'Not loaded'}</small></li>;
    })}</ul>}
    </details>
    {error && <div className="error" role="alert">{error}</div>}
      </section>
    </dialog>, document.body)}
  </div>;
}