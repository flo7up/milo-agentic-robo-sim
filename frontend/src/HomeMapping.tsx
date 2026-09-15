import { useEffect, useRef, useState } from 'react';
import { Check, CircleStop, Crosshair, FolderOpen, Map, MapPin, Navigation, Plus, RotateCcw, RotateCw, Save, Scan, Search } from 'lucide-react';

type Place = { place_id: string; name: string; kind: string; pose_m_rad: number[]; reachable: boolean; identity_status?: string };
export type HomeState = {
  environment_id: string; stage: string; map_id: string | null; name: string | null; revision: number; dirty?: boolean;
  localization: { status: string; age_s: number; pose_m_rad: number[] | null; quality: { mean_residual_m?: number } | null };
  places: Place[]; maps: { map_id: string; environment_id: string; name: string; revision: number }[];
  map?: { cells: number[]; width: number; height: number; origin_m: number[]; resolution_m: number };
  edges?: { from: string; to: string }[]; live_obstacles_m?: number[][]; route_m?: number[][];
  coverage: { free_m2: number; visited_cells: number; scan_count: number } | null;
  task: { status: string; reason: string; segments: number; retries: number; visited_frontiers: number } | null;
  error: string | null;
  objects?: { observation_id: string; label: string; position_m: number[]; confidence: number;
    observed_unix_s: number; currently_observed: boolean; supporting_images: string[] }[];
  expansion_allowed?: boolean;
  room_verification?: { status: string; identity_verified: boolean };
  room_workflow?: { status: string; reason: string } | null;
  room_observations?: { observation_id: string; place_id: string; label: string; evidence: string;
    confidence: number; review_status: string; room_matches: boolean | null; observed_unix_s: number; image_url: string }[];
};

export function HomeMapping({ runId, epoch, connected, busy, stopped, request }: {
  runId: string; epoch: number; connected: boolean; busy: boolean; stopped: boolean;
  request: (path: string, body?: unknown) => Promise<unknown>;
}) {
  const [state, setState] = useState<HomeState | null>(null);
  const [open, setOpen] = useState(false);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState('');
  const [mapName, setMapName] = useState('My home');
  const loadedMap = useRef<string | null>(null);
  const [selectedMap, setSelectedMap] = useState('');
  const [placeName, setPlaceName] = useState('');
  const [kind, setKind] = useState('destination');
  const [connects, setConnects] = useState<string[]>([]);
  const [destination, setDestination] = useState('');
  const [seed, setSeed] = useState('');
  const [region, setRegion] = useState('');
  const [budget, setBudget] = useState(60);
  const [point, setPoint] = useState<number[] | null>(null);
  const [heading, setHeading] = useState(0);
  const [zoom, setZoom] = useState(1);
  const [objectLimit, setObjectLimit] = useState(3);
  const [roomLimit, setRoomLimit] = useState(3);
  const canvas = useRef<HTMLCanvasElement>(null);
  const projection = useRef({ left: -5, bottom: -5, width: 10, height: 7.5 });
  const running = state?.task?.status === 'running';
  const disabled = !connected || busy || stopped || pending || running;
  const localized = state?.localization.status === 'localized' && state.localization.age_s < 1;

  useEffect(() => {
    if (state?.map_id && state.revision > 0 && loadedMap.current !== state.map_id) {
      loadedMap.current = state.map_id;
      setMapName(state.name ?? 'My home');
    }
  }, [state?.map_id, state?.revision, state?.name]);

  useEffect(() => {
    if (!connected || !open || pending) return;
    const controller = new AbortController();
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const response = await fetch('/api/home', { signal: controller.signal });
        if (!response.ok) throw new Error('Home map unavailable');
        const value: HomeState = await response.json();
        if (active) setState(value);
      } catch (failure) {
        if (active) setError(failure instanceof Error ? failure.message : String(failure));
      } finally {
        if (active) timer = setTimeout(poll, 1000);
      }
    }
    void poll();
    return () => { active = false; clearTimeout(timer); controller.abort(); };
  }, [connected, open, pending, runId, epoch]);

  async function act(action: string, values: Record<string, unknown> = {}) {
    setPending(true); setError('');
    try {
      if (action === 'start_mapping' || action === 'localize') await request('continuous/scan', { run_id: runId, episode_epoch: epoch, compact_arms: true });
      const result = await request('home', { run_id: runId, episode_epoch: epoch, map_id: state?.map_id ?? null, action, ...values }) as HomeState;
      setState(result);
      if (action === 'load_map') { setMapName(result.name ?? 'My home'); setPoint(null); setSeed(''); setDestination(''); }
      if (action === 'add_place') { setPlaceName(''); setPoint(null); setConnects([]); }
    } catch (failure) { setError(failure instanceof Error ? failure.message : String(failure)); }
    finally { setPending(false); }
  }

  async function drive(linear: number, angular: number) {
    setPending(true); setError('');
    try {
      const current = await request('state') as { run_id: string; episode_epoch: number; observation: { seq: number } };
      if (current.run_id !== runId || current.episode_epoch !== epoch) throw new Error('Scenario changed');
      const result = await request('command', { run_id: runId, episode_epoch: epoch, observation_seq: current.observation.seq,
        action_id: crypto.randomUUID(), tool: 'drive_base', arguments: { linear_mps: linear, angular_radps: angular, duration_s: 1 } }) as { status: string; message?: string; error?: string };
      if (result.status !== 'ok') throw new Error(result.message || result.error || 'Guided rotation was blocked');
    } catch (failure) { setError(failure instanceof Error ? failure.message : String(failure)); }
    finally { setPending(false); }
  }

  useEffect(() => {
    const element = canvas.current, map = state?.map;
    if (!element || !map || !open) return;
    const context = element.getContext('2d')!;
    const style = getComputedStyle(element);
    const color = (name: string) => style.getPropertyValue(name).trim();
    let left = map.width, right = 0, bottom = map.height, top = 0;
    map.cells.forEach((value, index) => {
      if (value === -1) return;
      const column = index % map.width, row = Math.floor(index / map.width);
      left = Math.min(left, column); right = Math.max(right, column + 1);
      bottom = Math.min(bottom, row); top = Math.max(top, row + 1);
    });
    const center = [map.origin_m[0] + (left + right) / 2 * map.resolution_m, map.origin_m[1] + (bottom + top) / 2 * map.resolution_m];
    const width = Math.max(4, (right - left + 8) * map.resolution_m, (top - bottom + 8) * map.resolution_m * 4 / 3) / zoom;
    const height = width * 3 / 4;
    const frame = { left: center[0] - width / 2, bottom: center[1] - height / 2, width, height };
    projection.current = frame;
    const project = (position: number[]) => [(position[0] - frame.left) / width * 640, 480 - (position[1] - frame.bottom) / height * 480];
    context.fillStyle = color('--cp-surface-soft'); context.fillRect(0, 0, 640, 480);
    const cellSize = map.resolution_m / width * 640;
    map.cells.forEach((value, index) => {
      if (value === -1) return;
      const [horizontal, vertical] = project([map.origin_m[0] + index % map.width * map.resolution_m,
        map.origin_m[1] + (Math.floor(index / map.width) + 1) * map.resolution_m]);
      context.fillStyle = color(value === 0 ? '--cp-success' : '--cp-text-muted');
      context.globalAlpha = value === 0 ? .22 : 1;
      context.fillRect(horizontal, vertical, cellSize + .5, cellSize + .5);
    });
    context.globalAlpha = 1;
    context.strokeStyle = color('--cp-border-strong'); context.lineWidth = 1;
    for (const edge of state.edges ?? []) {
      const from = state.places.find(place => place.place_id === edge.from), to = state.places.find(place => place.place_id === edge.to);
      if (!from || !to) continue;
      const begin = project(from.pose_m_rad), end = project(to.pose_m_rad);
      context.beginPath(); context.moveTo(begin[0], begin[1]); context.lineTo(end[0], end[1]); context.stroke();
    }
    context.fillStyle = color('--cp-danger');
    for (const obstacle of state.live_obstacles_m ?? []) {
      const [horizontal, vertical] = project(obstacle); context.fillRect(horizontal - 1.5, vertical - 1.5, 3, 3);
    }
    context.strokeStyle = color('--cp-link'); context.lineWidth = 2; context.beginPath();
    (state.route_m ?? []).forEach((position, index) => {
      const [horizontal, vertical] = project(position);
      if (index) context.lineTo(horizontal, vertical); else context.moveTo(horizontal, vertical);
    }); context.stroke();
    state.places.forEach((place, index) => {
      const [horizontal, vertical] = project(place.pose_m_rad);
      context.fillStyle = color('--cp-accent'); context.beginPath(); context.arc(horizontal, vertical, 8, 0, Math.PI * 2); context.fill();
      context.fillStyle = color('--cp-accent-fg'); context.font = '11px "Segoe UI"'; context.textAlign = 'center'; context.fillText(String(index + 1), horizontal, vertical + 4);
    });
    const pose = state.localization.pose_m_rad;
    if (pose) {
      const [horizontal, vertical] = project(pose);
      context.save(); context.translate(horizontal, vertical); context.rotate(-pose[2]);
      context.fillStyle = color('--cp-link'); context.beginPath(); context.moveTo(10, 0); context.lineTo(-7, -6); context.lineTo(-7, 6); context.closePath(); context.fill(); context.restore();
    }
    if (point) {
      const [horizontal, vertical] = project(point);
      context.strokeStyle = color('--cp-accent'); context.lineWidth = 2; context.strokeRect(horizontal - 6, vertical - 6, 12, 12);
    }
  }, [state, open, point, zoom]);

  const maps = state?.maps.filter(map => map.environment_id === state.environment_id) ?? [];
  return <details className="home-mapping" open={open} onToggle={event => setOpen(event.currentTarget.open)}>
    <summary><Map size={16} />Home map</summary>
    <div className="home-map-toolbar">
      <strong>{state?.name ?? 'No saved map loaded'}</strong>
      <span role="status">{pending ? 'Working...' : stopped ? 'Stopped' : state?.stage.replaceAll('_', ' ') ?? 'Loading'}</span>
      <button className="danger" title="Cancel home task" aria-label="Cancel home task" disabled={!connected} onClick={() => void act('cancel_task')}><CircleStop size={16} /></button>
    </div>
    {error && <p role="alert">{error}</p>}
    {state?.error && <p role="alert">{state.error}</p>}
    <div className="home-map-toolbar">
      <button disabled={disabled || maps.length > 0} onClick={() => void act('start_mapping')}><Scan size={16} />{state?.stage === 'review' && !state.revision ? 'Resume mapping' : 'Start guided mapping'}</button>
      {maps.length > 0 && <><label>Saved home<select aria-label="Saved home" value={selectedMap || maps[0].map_id} onChange={event => setSelectedMap(event.target.value)}>
        {maps.map(map => <option key={map.map_id} value={map.map_id}>{map.name} / v{map.revision}</option>)}</select></label>
        <button title="Load saved home" aria-label="Load saved home" disabled={disabled} onClick={() => void act('load_map', { map_id: selectedMap || maps[0].map_id })}><FolderOpen size={16} /></button></>}
    </div>
    {state?.stage === 'mapping' && <div className="home-map-toolbar" role="group" aria-label="Guided mapping controls">
      <button title="Map: rotate left" aria-label="Map: rotate left" disabled={disabled} onClick={() => void drive(0, .4)}><RotateCcw size={18} /></button>
      <button disabled={disabled || !localized || !point || state.expansion_allowed === false} onClick={() => void act('guided_to', { pose_m_rad: point ? [...point, 0] : null, time_budget: budget })}><Navigation size={18} />Survey to selected point</button>
      <button title="Map: rotate right" aria-label="Map: rotate right" disabled={disabled} onClick={() => void drive(0, -.4)}><RotateCw size={18} /></button>
      <button disabled={disabled} onClick={() => void act('review')}><MapPin size={16} />Review map</button>
    </div>}
    {state?.map && <>
      <div className="home-map-layout">
        <figure>
          <canvas ref={canvas} width={640} height={480} aria-label="Observed home map" title="Select a mapped position" onClick={event => {
            const rect = event.currentTarget.getBoundingClientRect(), frame = projection.current;
            setPoint([frame.left + (event.clientX - rect.left) / rect.width * frame.width, frame.bottom + (1 - (event.clientY - rect.top) / rect.height) * frame.height]);
          }} />
          <figcaption className="spatial-legend"><span><i className="spatial-floor" />Free</span><span><i className="home-occupied" />Occupied</span><span><i className="spatial-unknown" />Unknown</span><span><i className="home-live" />Live obstacle</span></figcaption>
          <label className="home-zoom">Zoom<input aria-label="Map zoom" type="range" min={1} max={4} step={.25} value={zoom} onChange={event => setZoom(Number(event.target.value))} /></label>
          <div className="home-map-metrics"><span>Map frame / m / rad</span><span>v{state.revision}{state.dirty ? ' / unsaved changes' : ''}</span><span>{state.coverage?.free_m2.toFixed(1)} m2 observed free</span><span>{state.coverage?.visited_cells} visited cells</span></div>
        </figure>
        <div className="home-map-fields">
          <h4>Localization</h4>
          <output>{state.localization.status}{state.localization.quality?.mean_residual_m !== undefined ? ` / ${state.localization.quality.mean_residual_m.toFixed(2)} m scan residual` : ''}</output>
          <label>Approximate location<select aria-label="Approximate location" value={seed} onChange={event => setSeed(event.target.value)}><option value="">{point ? 'Selected map position' : 'Find from scan'}</option>{state.places.map(place => <option key={place.place_id} value={place.place_id}>{place.name}</option>)}</select></label>
          <label>Heading (degrees)<input aria-label="Map heading" type="number" min={-180} max={180} step={5} value={heading} onChange={event => setHeading(Number(event.target.value))} /></label>
          <button disabled={disabled} onClick={() => void act('localize', seed ? { place_id: seed } : { pose_m_rad: point ? [...point, heading * Math.PI / 180] : null })}><Crosshair size={16} />Localize</button>
          <h4>Named places</h4>
          <label>Place name<input aria-label="Place name" value={placeName} maxLength={80} onChange={event => setPlaceName(event.target.value)} /></label>
          <label>Place type<select aria-label="Place type" value={kind} onChange={event => setKind(event.target.value)}><option value="destination">Destination</option><option value="room">Room</option><option value="doorway">Doorway</option></select></label>
          {state.places.length > 0 && <label>Connected places<select aria-label="Connected places" multiple value={connects} onChange={event => setConnects(Array.from(event.target.selectedOptions, option => option.value))}>{state.places.map(place => <option key={place.place_id} value={place.place_id}>{place.name}</option>)}</select></label>}
          <output>{point ? `${point[0].toFixed(2)}, ${point[1].toFixed(2)} m` : 'Current robot position'}</output>
          <div className="home-map-toolbar"><button disabled={disabled || !localized || !placeName.trim()} onClick={() => void act('add_place', { name: placeName, kind, connects, pose_m_rad: point ? [...point, heading * Math.PI / 180] : null })}><Plus size={16} />Add place</button>
            {point && <button title="Use current robot position" aria-label="Use current robot position" onClick={() => setPoint(null)}><Crosshair size={16} /></button>}</div>
        </div>
      </div>
      {state.places.length > 0 && <ol className="home-place-list">{state.places.map(place => <li key={place.place_id}><span>{place.name}</span><small>{place.kind} / {place.reachable ? 'reachable' : 'not currently reachable'} / {(place.identity_status ?? 'operator_named_unreviewed').replaceAll('_', ' ')}</small></li>)}</ol>}
      <div className="home-map-toolbar"><label>Map name<input aria-label="Map name" value={mapName} maxLength={80} onChange={event => setMapName(event.target.value)} /></label><button disabled={disabled || !mapName.trim()} onClick={() => void act('save_map', { name: mapName })}><Save size={16} />Save map</button></div>
      {state.revision > 0 && <div className="home-task-fields">
        {state.stage !== 'mapping' && <button disabled={disabled || !localized || state.expansion_allowed === false} onClick={() => void act('continue_mapping')}><Scan size={16} />Continue mapping</button>}
        <label>Destination<select aria-label="Mapped destination" value={destination} onChange={event => setDestination(event.target.value)}><option value="">Select a destination</option>{state.places.map(place => <option key={place.place_id} value={place.place_id} disabled={!place.reachable}>{place.name}{place.reachable ? '' : ' (unreachable)'}</option>)}</select></label>
        <button disabled={disabled || !localized || !destination || !state.places.find(place => place.place_id === destination)?.reachable} onClick={() => void act('navigate_to', { place_id: destination, time_budget: budget })}><Navigation size={16} />Navigate</button>
        <label>Exploration region<select aria-label="Exploration region" value={region} onChange={event => setRegion(event.target.value)}><option value="">All connected space</option>{state.places.filter(place => place.kind === 'room').map(place => <option key={place.place_id} value={place.place_id}>{place.name}</option>)}</select></label>
        <label>Task budget (s)<input aria-label="Map task budget" type="number" min={1} max={300} value={budget} onChange={event => setBudget(Number(event.target.value))} /></label>
        <button disabled={disabled || !localized || state.expansion_allowed === false || budget < 1 || budget > 300} onClick={() => void act('explore', { region_id: region || null, time_budget: budget })}><Search size={16} />Expand map</button>
      </div>}
      {state.task && <div className="home-task-status" role="status"><strong>{state.task.status}</strong><span>{state.task.reason}</span><small>{state.task.segments} route segments / {state.task.retries} retries / {state.task.visited_frontiers} frontiers visited</small></div>}
      {state.room_workflow && <div className="home-task-status" role="status" aria-label="Room round trip"><strong>{state.room_workflow.status.replaceAll('_', ' ')}</strong><span>{state.room_workflow.reason}</span></div>}
      {state.room_verification && <output aria-label="Current room report">{state.room_verification.status.replaceAll('_', ' ')} / independent identity verification unavailable</output>}
      {!!state.room_observations?.length && <details className="home-object-memory"><summary>Room observations ({state.room_observations.length})</summary>
        <ul>{state.room_observations.slice(0, roomLimit).map(observation => <li key={observation.observation_id}>
          <a href={observation.image_url} target="_blank" rel="noreferrer"><img src={observation.image_url} loading="lazy" alt={`Room evidence for ${observation.label}`} /></a>
          <div><strong>{observation.label}</strong><span>{observation.review_status.replaceAll('_', ' ')} / {new Date(observation.observed_unix_s * 1000).toLocaleString()}</span>
            <span>{observation.evidence}</span><small>{(observation.confidence * 100).toFixed(0)}% reported confidence{observation.room_matches === false ? ' / visual mismatch' : observation.room_matches === true ? ' / visual match reported' : ''}</small>
            <button disabled={disabled || observation.review_status === 'operator_confirmed' || observation.room_matches === false} onClick={() => void act('review_room', { evidence_id: observation.observation_id })}><Check size={16} />Confirm {observation.label}</button></div>
        </li>)}</ul>
        {state.room_observations.length > roomLimit && <div className="home-map-toolbar"><button onClick={() => setRoomLimit(value => value + 5)}>Show more</button><button onClick={() => setRoomLimit(state.room_observations!.length)}>View all</button></div>}
      </details>}
      {!!state.objects?.length && <details className="home-object-memory"><summary>Object observations ({state.objects.length})</summary>
        <ul>{state.objects.slice(0, objectLimit).map(observation => <li key={observation.observation_id}>
          <a href={observation.supporting_images[0]} target="_blank" rel="noreferrer"><img src={observation.supporting_images[0]} loading="lazy" alt={`Evidence for ${observation.label}`} /></a>
          <div><strong>{observation.label}</strong><span>{observation.currently_observed ? 'Currently observed' : 'Last seen'} / {new Date(observation.observed_unix_s * 1000).toLocaleString()}</span>
            <small>{observation.position_m.map(value => value.toFixed(2)).join(', ')} m / {(observation.confidence * 100).toFixed(0)}% reported confidence</small></div>
        </li>)}</ul>
        {state.objects.length > objectLimit && <div className="home-map-toolbar"><button onClick={() => setObjectLimit(value => value + 5)}>Show more</button><button onClick={() => setObjectLimit(state.objects!.length)}>View all</button></div>}
      </details>}
    </>}
  </details>;
}