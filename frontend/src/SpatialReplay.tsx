import { useEffect, useRef, useState } from 'react';
import { ChevronDown, Pause, Play, RefreshCw, RotateCcw, TriangleAlert } from 'lucide-react';

type Coverage = { known_cells: number; free_m2: number; visited_cells: number; scan_count: number };
type Task = { task_id?: string; kind?: string; status?: string; reason?: string; segments?: number; retries?: number; completion_verified?: boolean };
type Home = { map?: { map_id: string; revision: number; sequence: number; wall_s: number }; coverage?: Coverage;
  map_snapshot_current?: boolean;
  pose_m_rad?: number[] | null; map_from_odometry_m_rad?: number[] | null; stage?: string; task?: Task;
  localization?: { status: string; quality?: { mean_residual_m?: number } }; depth_age_s?: number | null;
  lidar_age_s?: number; sampled_wall_s?: number; error?: string | null };
type Frame = { wall_s: number; simulated_s: number; index: number; camera_url: string | null; depth_url: string | null;
  segment?: number;
  camera_simulated_s: number | null; depth_simulated_s: number | null; map_url: string | null; telemetry_url: string | null;
  home: Home | null; activity: string; status: string; contact: boolean; assisted: boolean; odometry_m_rad: number[] | null };
type Event = { id: number; wall_s: number; stage: string; localization: string; status: string | null; reason: string; segments: number; retries: number };
type Replay = { frames: Frame[]; events: Event[]; evidence: string; sample_count: number; recording_complete: boolean;
  spatial_progress: { complete?: boolean; snapshots?: number; events?: number; initial?: Coverage | null; final?: Coverage | null } };
type RecordedMap = { map_id: string; revision: number; sha256: string; name: string; width: number; height: number;
  resolution_m: number; origin_m: number[]; cells: number[]; visited_indices: number[];
  places: { place_id: string; name: string; pose_m_rad: number[] }[]; edges: { from: string; to: string }[] };
type Telemetry = { live_obstacles_m: number[][]; route_m: number[][]; task?: Task };

const fixed = (value: number | undefined | null, digits = 2) => value == null ? 'Not recorded' : value.toLocaleString('en-US', { maximumFractionDigits: digits });

export function SpatialReplay({ url }: { url: string }) {
  const [replay, setReplay] = useState<Replay | null>(null);
  const [cursor, setCursor] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [error, setError] = useState('');
  const [revision, setRevision] = useState(0);
  const [media, setMedia] = useState<{ key: string; map: RecordedMap | null; telemetry: Telemetry | null; error: string } | null>(null);
  const [eventLimit, setEventLimit] = useState(3);
  const [imageFailures, setImageFailures] = useState<string[]>([]);
  const canvas = useRef<HTMLCanvasElement>(null);
  const cache = useRef(new Map<string, RecordedMap>());
  useEffect(() => {
    const controller = new AbortController();
    let active = true;
    const timer = setTimeout(() => { controller.abort(); if (active) setError('Replay request timed out'); }, 15000);
    setError(''); setReplay(null); setPlaying(false); setMedia(null); cache.current.clear();
    void fetch(url, { signal: controller.signal }).then(async response => {
      if (!response.ok) throw new Error('Synchronized replay unavailable');
      const data: Replay = await response.json();
      if (!data.frames?.length) throw new Error('No recorded frames');
      if (!controller.signal.aborted) { setReplay(data); setCursor(data.frames[0].wall_s); }
    }).catch(failure => { if (active && !controller.signal.aborted) setError(String(failure.message)); })
      .finally(() => clearTimeout(timer));
    return () => { active = false; clearTimeout(timer); controller.abort(); };
  }, [url, revision]);
  const end = replay?.frames.at(-1)?.wall_s ?? 0;
  const start = replay?.frames[0]?.wall_s ?? 0;
  let frame: Frame | undefined;
  for (const candidate of replay?.frames ?? []) {
    if (candidate.wall_s > cursor) break;
    frame = candidate;
  }
  const frameKey = `${frame?.map_url ?? ''}|${frame?.telemetry_url ?? ''}`;
  useEffect(() => {
    if (!frame) return;
    const controller = new AbortController();
    const timer = setTimeout(() => { controller.abort(); setMedia({ key: frameKey, map: null, telemetry: null, error: 'Recorded media timed out' }); }, 10000);
    const mapUrl = frame.map_url, telemetryUrl = frame.telemetry_url;
    async function read<T>(path: string | null): Promise<T | null> {
      if (!path) return null;
      const response = await fetch(path, { signal: controller.signal });
      if (!response.ok) throw new Error('Recorded map or telemetry missing');
      return response.json();
    }
    void Promise.all([mapUrl && cache.current.has(mapUrl) ? Promise.resolve(cache.current.get(mapUrl)!) : read<RecordedMap>(mapUrl), read<Telemetry>(telemetryUrl)])
      .then(([map, telemetry]) => {
        if (controller.signal.aborted) return;
        if (mapUrl && map) { cache.current.set(mapUrl, map); if (cache.current.size > 8) cache.current.delete(cache.current.keys().next().value!); }
        setMedia({ key: frameKey, map, telemetry, error: '' });
      }).catch(failure => { if (!controller.signal.aborted) setMedia({ key: frameKey, map: null, telemetry: null, error: String(failure.message) }); })
      .finally(() => clearTimeout(timer));
    return () => { clearTimeout(timer); controller.abort(); };
  }, [frameKey, revision]);
  useEffect(() => {
    if (!playing) return;
    let previous = performance.now();
    const timer = setInterval(() => {
      const now = performance.now();
      setCursor(value => Math.min(end, value + (now - previous) / 1000 * speed));
      previous = now;
    }, 100);
    return () => clearInterval(timer);
  }, [playing, speed, end]);
  useEffect(() => { if (cursor >= end) setPlaying(false); }, [cursor, end]);
  const currentMedia = media?.key === frameKey ? media : null;
  useEffect(() => {
    const element = canvas.current, map = currentMedia?.map;
    if (!element) return;
    const context = element.getContext('2d')!;
    const style = getComputedStyle(element);
    const color = (name: string) => style.getPropertyValue(name).trim();
    context.fillStyle = color('--cp-surface-soft'); context.fillRect(0, 0, element.width, element.height);
    if (!map) return;
    let left = map.width, right = 0, bottom = map.height, top = 0;
    map.cells.forEach((cell, index) => {
      if (cell === -1) return;
      const column = index % map.width, row = Math.floor(index / map.width);
      left = Math.min(left, column); right = Math.max(right, column + 1); bottom = Math.min(bottom, row); top = Math.max(top, row + 1);
    });
    if (left >= right) { left = 0; right = map.width; bottom = 0; top = map.height; }
    const scale = Math.min(600 / (right - left + 8), 420 / (top - bottom + 8));
    const offsetX = 320 - (left + right) / 2 * scale, offsetY = 240 + (top + bottom) / 2 * scale;
    const project = (point: number[]) => [(point[0] - map.origin_m[0]) / map.resolution_m * scale + offsetX,
      offsetY - (point[1] - map.origin_m[1]) / map.resolution_m * scale];
    map.cells.forEach((cell, index) => {
      if (cell === -1) return;
      context.fillStyle = color(cell === 0 ? '--cp-success' : '--cp-text-muted'); context.globalAlpha = cell === 0 ? .22 : 1;
      context.fillRect(offsetX + index % map.width * scale, offsetY - (Math.floor(index / map.width) + 1) * scale, scale + .2, scale + .2);
    });
    context.globalAlpha = 1;
    context.fillStyle = color('--cp-link');
    for (const index of map.visited_indices) context.fillRect(offsetX + index % map.width * scale, offsetY - (Math.floor(index / map.width) + 1) * scale, scale, scale);
    context.fillStyle = color('--cp-danger');
    for (const obstacle of currentMedia?.telemetry?.live_obstacles_m ?? []) { const [horizontal, vertical] = project(obstacle); context.fillRect(horizontal - 1, vertical - 1, 3, 3); }
    const drawPath = (points: number[][], stroke: string, dashed = false) => {
      context.strokeStyle = color(stroke); context.lineWidth = 2; context.setLineDash(dashed ? [5, 4] : []); context.beginPath();
      points.forEach((point, index) => { const [horizontal, vertical] = project(point); if (index) context.lineTo(horizontal, vertical); else context.moveTo(horizontal, vertical); });
      context.stroke(); context.setLineDash([]);
    };
    drawPath(currentMedia?.telemetry?.route_m ?? [], '--cp-accent', true);
    const trails: number[][][] = [];
    let previousSegment: number | undefined;
    for (const candidate of replay?.frames ?? []) {
      if (candidate.wall_s > cursor) break;
      if (candidate.home?.map?.map_id !== map.map_id || !candidate.home.pose_m_rad || candidate.home.localization?.status !== 'localized') { previousSegment = undefined; continue; }
      if (previousSegment === undefined || previousSegment !== (candidate.segment ?? 0)) trails.push([]);
      trails.at(-1)!.push(candidate.home.pose_m_rad);
      previousSegment = candidate.segment ?? 0;
    }
    for (const trail of trails) drawPath(trail, '--cp-link');
    map.places.forEach((place, index) => {
      const [horizontal, vertical] = project(place.pose_m_rad);
      context.fillStyle = color('--cp-accent'); context.beginPath(); context.arc(horizontal, vertical, 7, 0, Math.PI * 2); context.fill();
      context.fillStyle = color('--cp-accent-fg'); context.font = '11px "Segoe UI"'; context.textAlign = 'center'; context.fillText(String(index + 1), horizontal, vertical + 4);
    });
    if (frame?.home?.pose_m_rad) {
      const pose = frame.home.pose_m_rad, [horizontal, vertical] = project(pose);
      context.save(); context.translate(horizontal, vertical); context.rotate(-pose[2]); context.fillStyle = color('--cp-link');
      context.beginPath(); context.moveTo(10, 0); context.lineTo(-6, -6); context.lineTo(-6, 6); context.closePath(); context.fill(); context.restore();
    }
  }, [currentMedia, replay, cursor, frame]);
  const selectTime = (value: number) => { setPlaying(false); setCursor(value); };
  if (error) return <section className="spatial-replay"><p role="alert">{error}</p><button title="Retry replay" aria-label="Retry replay" onClick={() => setRevision(value => value + 1)}><RefreshCw size={16} /></button></section>;
  if (!replay || !frame) return <section className="spatial-replay" role="status">Loading synchronized replay...</section>;
  const coverage = frame.home?.coverage, task = frame.home?.task;
  const coverageFrames = replay.frames.filter(candidate => candidate.home?.coverage && candidate.home.map?.map_id === frame!.home?.map?.map_id);
  const maxCoverage = Math.max(1, ...coverageFrames.map(candidate => candidate.home!.coverage!.free_m2));
  const chartX = (seconds: number) => 48 + 548 * (seconds - start) / Math.max(.001, end - start);
  const chartY = (value: number) => 120 - value / maxCoverage * 100;
  return <section className="spatial-replay" aria-label="Synchronized spatial replay">
    <div className="results-route-heading"><h3>Synchronized replay</h3><span>{replay.evidence.replaceAll('_', ' ')} / {replay.sample_count} samples</span></div>
    {(!replay.recording_complete || (replay.spatial_progress.snapshots && !replay.spatial_progress.complete)) && <p className="results-warning"><TriangleAlert size={16} />Recording incomplete; missing intervals cannot establish success.</p>}
    <div className="spatial-replay-controls">
      <button title={playing ? 'Pause replay' : 'Play replay'} aria-label={playing ? 'Pause replay' : 'Play replay'} disabled={end <= start} onClick={() => { if (cursor >= end) setCursor(start); setPlaying(value => !value); }}>{playing ? <Pause size={17} /> : <Play size={17} />}</button>
      <button title="Rewind replay" aria-label="Rewind replay" onClick={() => selectTime(start)}><RotateCcw size={17} /></button>
      <input type="range" aria-label="Replay time" min={start} max={end} step="any" value={cursor} onChange={event => selectTime(Number(event.target.value))} />
      <select aria-label="Replay speed" value={speed} onChange={event => setSpeed(Number(event.target.value))}>{[1, 4, 10].map(value => <option key={value} value={value}>{value}x</option>)}</select>
      <output>{fixed(cursor, 1)} / {fixed(end, 1)} s</output>
    </div>
    <div className="spatial-replay-grid">
      <figure className="spatial-replay-map"><canvas width={640} height={480} ref={canvas} aria-label="Recorded observed map" data-map-id={currentMedia?.map?.map_id ?? ''} data-time={frame.wall_s} />
        {!frame.map_url && <p>Persistent map not recorded at this time</p>}{frame.map_url && !currentMedia && <p role="status">Loading recorded map...</p>}{currentMedia?.error && <p role="alert">{currentMedia.error}<button title="Retry recorded media" aria-label="Retry recorded media" onClick={() => setRevision(value => value + 1)}><RefreshCw size={16}/></button></p>}
        {frame.home?.map_snapshot_current === false && <p className="results-warning">Snapshot limit reached; displayed geometry predates selected telemetry.</p>}
        <figcaption>{currentMedia?.map ? `${currentMedia.map.name} / v${currentMedia.map.revision} / map frame, metres` : 'No map snapshot'}</figcaption>
        <div className="spatial-legend"><span><i className="spatial-floor" />Observed free</span><span><i className="home-occupied" />Occupied</span><span><i className="spatial-obstacle" />Planned route</span><span><i className="replay-visited" />Visited / estimated path</span><span><i className="home-live" />Live obstacle</span><span><i className="spatial-unknown" />Unknown</span></div>
        {currentMedia?.map && <ol className="replay-place-labels">{currentMedia.map.places.map(place => <li key={place.place_id}>{place.name}</li>)}</ol>}
      </figure>
      <div className="spatial-replay-cameras">{([{ label: 'Head camera', url: frame.camera_url, time: frame.camera_simulated_s }, { label: 'Depth', url: frame.depth_url, time: frame.depth_simulated_s }]).map(image => <figure key={image.label}>
        {image.url && !imageFailures.includes(image.url) ? <img key={image.url} src={image.url} width={320} height={240} alt={`Recorded ${image.label.toLowerCase()}`} onError={() => setImageFailures(previous => [...previous, image.url!])} /> : <div className="replay-missing">{image.label} not recorded / unavailable</div>}
        <figcaption>{image.label} / captured {fixed(image.time, 2)} s simulated</figcaption>
      </figure>)}</div>
    </div>
    <dl className="results-facts replay-readouts"><div><dt>At Selected Time</dt><dd>{frame.activity} / {task?.status ?? frame.status}</dd></div>
      <div><dt>Observed Free</dt><dd>{fixed(coverage?.free_m2)} m2</dd></div><div><dt>Visited Cells</dt><dd>{fixed(coverage?.visited_cells, 0)}</dd></div>
      <div><dt>Localization</dt><dd>{frame.home?.localization?.status ?? 'Not recorded'}</dd></div>
      <div><dt>Depth / Lidar Age</dt><dd>{fixed(frame.home?.depth_age_s)} / {fixed(frame.home?.lidar_age_s)} s at capture</dd></div>
      <div><dt>Route Segments / Retries</dt><dd>{fixed(task?.segments, 0)} / {fixed(task?.retries, 0)}</dd></div>
      <div><dt>Assistance</dt><dd>{frame.assisted ? 'Operator-assisted' : 'No recorded intervention'}</dd></div></dl>
    {task?.reason && <p className="replay-task-reason">{task.reason}</p>}
    {!!coverageFrames.length && <figure className="replay-coverage"><figcaption>Observed free area over recorded wall time</figcaption><svg viewBox="0 0 620 155" role="img" aria-label="Recorded map coverage over time">
      <line className="replay-chart-axis" x1={48} y1={120} x2={596} y2={120} /><text x={42} y={25} textAnchor="end">{fixed(maxCoverage, 1)}</text><text x={42} y={124} textAnchor="end">0 m2</text>
      <polyline className="replay-chart-line" points={coverageFrames.map(candidate => `${chartX(candidate.wall_s)},${chartY(candidate.home!.coverage!.free_m2)}`).join(' ')} />
      <line className="replay-chart-cursor" x1={chartX(cursor)} x2={chartX(cursor)} y1={15} y2={120} />
      {coverageFrames.filter((_, index) => index % Math.max(1, Math.ceil(coverageFrames.length / 30)) === 0).map(candidate => <circle key={candidate.index} cx={chartX(candidate.wall_s)} cy={chartY(candidate.home!.coverage!.free_m2)} r={4} tabIndex={0} role="button" aria-label={`${fixed(candidate.wall_s)} seconds: ${fixed(candidate.home!.coverage!.free_m2)} square metres`} onClick={() => selectTime(candidate.wall_s)} onKeyDown={event => { if (event.key === 'Enter' || event.key === ' ') selectTime(candidate.wall_s); }}><title>{fixed(candidate.wall_s)} s / {fixed(candidate.home!.coverage!.free_m2)} m2</title></circle>)}
      <text x={48} y={145}>{fixed(start, 1)} s</text><text x={596} y={145} textAnchor="end">{fixed(end, 1)} s</text>
    </svg></figure>}
    <div className="replay-events"><h4>Mission events</h4>{!replay.events.length ? <p>No mission events recorded</p> : <ol>{replay.events.slice(0, eventLimit).map(event => <li key={`${event.id}-${event.wall_s}`}>
      <button onClick={() => selectTime(event.wall_s)} aria-label={`Jump to event ${event.id}: ${event.reason}`}><span>{fixed(event.wall_s, 1)} s</span><strong>{event.status ?? event.stage}</strong><span>{event.reason}</span></button>
    </li>)}</ol>}{replay.events.length > eventLimit && <div className="results-more"><button onClick={() => setEventLimit(value => value + 5)}><ChevronDown size={16} />Show more</button><button onClick={() => setEventLimit(replay.events.length)}>View all ({replay.events.length})</button></div>}</div>
  </section>;
}