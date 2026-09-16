import { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { Crosshair, Map, Maximize2, Minus, Scan, X } from 'lucide-react';
import { RobotControlSlot } from './RobotControlSurface';
import { usePreference } from './Preferences';
import type { SpatialTelemetry } from './types';

type SpatialState = {
  motion_zones?:SpatialTelemetry['motion_zones'];
  power?: {on: boolean; mode: 'off' | 'idle' | 'working'};
  enabled: boolean; paused: boolean; error: string | null;
  frame: { sequence: number; simulated_time_s: number; rgb_url: string; depth_url: string } | null;
  map: { width: number; height: number; cells: number[]; resolution_m: number; origin_m: number[];
    stale: boolean; age_s: number; observed_floor_cells: number; obstacle_cells: number; robot_odometry_m_rad: number[] } | null;
  footprint: { lower_xy_m: number[]; upper_xy_m: number[]; radius_m: number } | null;
  history?: { trail_retention_s: number; footprints: { polygon_m: number[][]; age_s: number }[];
    labels: {id: string; label: string; evidence: string; position_m: number[]; age_s: number; stale: boolean; observation_seq: number}[] };
  continuous?: {status: string; reason: string; path_m: number[][]; target_m: number[]; remaining_m: number;
    updates: number; buffer_stops: number; elapsed_s: number; minimum_cruise_speed_mps: number;
    replans?: number; handoffs?: number; maximum_update_gap_s?: number} | null;
};

export function SpatialSensing({ runId, epoch, connected, busy, request, hudHost, onTelemetry, showMotionZones = false }: {
  runId: string; epoch: number; connected: boolean; busy: boolean;
  request: (path: string, body?: unknown) => Promise<unknown>;
  hudHost?: HTMLElement | null;
  onTelemetry?: (value:SpatialTelemetry) => void;
  showMotionZones?:boolean;
}) {
  const [state, setState] = useState<SpatialState | null>(null);
  const [pair, setPair] = useState<{ sequence: number; rgb: string; depth: string } | null>(null);
  const [error, setError] = useState('');
  const [sensorError, setSensorError] = useState('');
  const [frameError, setFrameError] = useState('');
  const [pending, setPending] = useState(false);
  const [revision, setRevision] = useState(0);
  const [armed, setArmed] = useState(false);
  const [trailSeconds, setTrailSeconds] = useState(120);
  const [showLabels, setShowLabels] = useState(true);
  const [receivedAt, setReceivedAt] = useState(0);
  const [clock, setClock] = useState(() => performance.now());
  const [selectedLabel, setSelectedLabel] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(false);
  const [minimized, setMinimized] = useState(true);
  const dialog = useRef<HTMLDialogElement>(null);
  const details = useRef<HTMLDetailsElement>(null);
  const labelBoxes = useRef<{id: string; left: number; top: number; width: number; height: number}[]>([]);
  useEffect(() => {
    const timer = setInterval(() => setClock(performance.now()), 1000);
    return () => clearInterval(timer);
  }, []);
  useEffect(() => {
    setState(null); setPair(null); setSensorError(''); setFrameError(''); setError('');
    setSelectedLabel(null); setRevision(value => value + 1);
  }, [runId, epoch]);
  const [compactArms, setCompactArms] = usePreference('compact_arms', true);
  const canvas = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    if (expanded) dialog.current?.showModal(); else dialog.current?.close();
  }, [expanded]);
  const urls = useRef(new Set<string>());
  const navigating = state?.continuous?.status === 'running';
  useEffect(() => { if (!connected || busy || pending) setArmed(false); }, [connected, busy, pending]);

  useEffect(() => {
    if (!connected || pending) return;
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    let sequence = -1;
    let loadingPair = false;
    let pollInterval = 750;
    let telemetry:SpatialTelemetry = {run_id:runId,episode_epoch:epoch,received_at_ms:null,enabled:null,paused:false,error:null,frame:null,map:null};
    const controller = new AbortController();
    async function loadPair(frame: NonNullable<SpatialState['frame']>) {
      loadingPair = true;
      let pendingUrls: string[] = [];
      const signal = AbortSignal.any([controller.signal, AbortSignal.timeout(5000)]);
      try {
        const blobs = await Promise.all([frame.rgb_url, frame.depth_url].map(async path => {
          const response = await fetch(path, { signal });
          if (!response.ok) throw new Error('Spatial paired frame expired; retrying');
          return response.blob();
        }));
        if (!active) return;
        pendingUrls = blobs.map(blob => {
          const url = URL.createObjectURL(blob);
          urls.current.add(url);
          return url;
        });
        await Promise.all(pendingUrls.map(url => new Promise<void>((resolve, reject) => {
          const image = new Image();
          const finish = (failure?: Error) => {
            image.onload = null; image.onerror = null;
            signal.removeEventListener('abort', abort);
            if (failure) { image.src = ''; reject(failure); } else resolve();
          };
          const abort = () => finish(new Error('Spatial paired image timed out; retrying'));
          image.onload = () => finish();
          image.onerror = () => finish(new Error('Spatial paired image unavailable; retrying'));
          signal.addEventListener('abort', abort, { once: true });
          if (signal.aborted) abort(); else image.src = url;
        })));
        if (!active) return;
        setPair({ sequence: frame.sequence, rgb: pendingUrls[0], depth: pendingUrls[1] });
        pendingUrls = [];
        sequence = frame.sequence;
        setFrameError('');
      } catch (failure) {
        if (active) setFrameError(failure instanceof Error ? failure.message : String(failure));
      } finally {
        for (const url of pendingUrls) { URL.revokeObjectURL(url); urls.current.delete(url); }
        loadingPair = false;
      }
    }
    async function poll() {
      const requestedAt = performance.now();
      try {
        const response = await fetch('/api/spatial', { signal: AbortSignal.any([controller.signal, AbortSignal.timeout(5000)]),
          headers:showMotionZones ? {'X-Milo-Motion-Zones':'1'} : undefined });
        if (!response.ok) throw new Error('Spatial sensor unavailable');
        const data: SpatialState = await response.json();
        if (!active) return;
        setState(data);
        const received = performance.now();
        const requestDuration = received - requestedAt;
        pollInterval = data.power?.mode === 'off' ? 5000 : data.power?.mode === 'idle' ? 2000
          : showMotionZones ? Math.max(250, requestDuration * 4) : 750;
        setReceivedAt(received);
        telemetry = {run_id:runId,episode_epoch:epoch,received_at_ms:received,request_duration_ms:requestDuration,
          enabled:data.enabled,paused:data.paused,error:data.error,motion_zones:data.motion_zones,
          frame:data.frame ? {sequence:data.frame.sequence,simulated_time_s:data.frame.simulated_time_s} : null,
          map:data.map ? {age_s:data.map.age_s,stale:data.map.stale,observed_floor_cells:data.map.observed_floor_cells,obstacle_cells:data.map.obstacle_cells} : null};
        onTelemetry?.(telemetry);
        setSensorError('');
        if (!data.enabled) { setPair(null); setFrameError(''); }
        else if (data.power?.on !== false && data.frame && sequence !== data.frame.sequence && !loadingPair) void loadPair(data.frame);
      } catch (failure) {
        if (active) {
          const message = failure instanceof Error ? failure.message : String(failure);
          setSensorError(message);
          onTelemetry?.({...telemetry,error:message});
        }
      } finally {
        if (active) timer = setTimeout(poll, pollInterval);
      }
    }
    void poll();
    return () => { active = false; clearTimeout(timer); controller.abort(); };
  }, [connected, revision, pending, runId, epoch, onTelemetry, showMotionZones]);

  useEffect(() => () => {
    if (pair) for (const url of [pair.rgb, pair.depth]) { URL.revokeObjectURL(url); urls.current.delete(url); }
  }, [pair]);
  useEffect(() => () => { for (const url of urls.current) URL.revokeObjectURL(url); urls.current.clear(); }, []);

  useEffect(() => {
    const map = state?.map;
    const element = canvas.current;
    if (!element) return;
    const context = element.getContext('2d')!;
    context.clearRect(0, 0, element.width, element.height);
    if (!map) return;
    const style = getComputedStyle(element);
    const colors = [-1, 0, 100].map(value => style.getPropertyValue(value === -1 ? '--cp-surface-soft' : value === 0 ? '--cp-success' : '--cp-accent').trim());
    for (let row = 0; row < map.height; row++) for (let column = 0; column < map.width; column++) {
      context.fillStyle = colors[map.cells[row * map.width + column] === -1 ? 0 : map.cells[row * map.width + column] === 0 ? 1 : 2];
      context.fillRect(column * 2, (map.height - row - 1) * 2, 2, 2);
    }
    const pose = map.robot_odometry_m_rad;
    if (!pose) return;
    const elapsed = Math.max(0, (clock - receivedAt) / 1000);
    const project = (point: number[]) => [(point[0] - map.origin_m[0]) / map.resolution_m * 2,
      (map.height - (point[1] - map.origin_m[1]) / map.resolution_m) * 2];
    for (const stamp of state.history?.footprints ?? []) {
      const alpha = Math.max(0, 1 - (stamp.age_s + elapsed) / trailSeconds);
      if (!trailSeconds || !alpha) continue;
      context.save();
      context.globalAlpha = alpha * .6;
      context.strokeStyle = style.getPropertyValue('--cp-text').trim();
      context.lineWidth = 1;
      context.beginPath();
      stamp.polygon_m.forEach((corner, index) => {
        const [left, top] = project(corner);
        if (index === 0) context.moveTo(left, top); else context.lineTo(left, top);
      });
      context.closePath(); context.stroke(); context.restore();
    }
    const route = state.continuous?.path_m;
    if (route?.length) {
      context.strokeStyle = style.getPropertyValue('--cp-link').trim();
      context.lineWidth = 2;
      context.beginPath();
      route.forEach((point, index) => {
        const horizontal = (point[0] - map.origin_m[0]) / map.resolution_m * 2;
        const vertical = (map.height - (point[1] - map.origin_m[1]) / map.resolution_m) * 2;
        if (index === 0) context.moveTo(horizontal, vertical); else context.lineTo(horizontal, vertical);
      });
      context.stroke();
    }
    const horizontal = (pose[0] - map.origin_m[0]) / map.resolution_m * 2;
    const vertical = (map.height - (pose[1] - map.origin_m[1]) / map.resolution_m) * 2;
    context.save();
    context.translate(horizontal, vertical);
    context.rotate(-pose[2]);
    context.strokeStyle = style.getPropertyValue('--cp-text').trim();
    context.lineWidth = 1.5;
    if (state.footprint) {
      const { lower_xy_m: lower, upper_xy_m: upper } = state.footprint;
      const scale = 2 / map.resolution_m;
      context.strokeRect(lower[0] * scale, -upper[1] * scale, (upper[0] - lower[0]) * scale, (upper[1] - lower[1]) * scale);
    }
    context.beginPath(); context.moveTo(0, 0); context.lineTo(12, 0); context.lineTo(7, -4);
    context.moveTo(12, 0); context.lineTo(7, 4); context.stroke();
    context.restore();
    labelBoxes.current = [];
    if (showLabels && (!minimized || expanded)) {
      context.font = '12px "Segoe UI", sans-serif';
      const occupied = [{left: horizontal - 16, top: vertical - 16, width: 32, height: 32}];
      for (const label of [...(state.history?.labels ?? [])].reverse()) {
        const [left, top] = project(label.position_m);
        if (left < 0 || top < 0 || left > 320 || top > 320 || label.age_s + elapsed >= 1800) continue;
        const text = `${label.label.length > 22 ? label.label.slice(0, 21) + '...' : label.label} ?`;
        const width = context.measureText(text).width + 10;
        const options = [-24, 10, -44, 30, -64, 50].map(offset => ({left: Math.max(2, Math.min(318 - width, left + 8)),
          top: Math.max(2, Math.min(298, top + offset)), width, height: 20}));
        const box = options.find(option => !occupied.some(other => option.left < other.left + other.width + 3
          && option.left + option.width + 3 > other.left && option.top < other.top + other.height + 3 && option.top + 23 > other.top));
        context.fillStyle = style.getPropertyValue('--cp-link').trim();
        context.beginPath(); context.arc(left, top, 3, 0, Math.PI * 2); context.fill();
        if (!box) continue;
        occupied.push(box);
        labelBoxes.current.push({id: label.id, ...box});
        context.strokeStyle = style.getPropertyValue('--cp-link').trim();
        context.setLineDash(label.stale || label.age_s + elapsed > 120 ? [3, 3] : []);
        context.beginPath(); context.moveTo(left, top); context.lineTo(box.left, box.top + 10); context.stroke();
        context.fillStyle = style.getPropertyValue('--cp-surface').trim();
        context.fillRect(box.left, box.top, width, 20); context.strokeRect(box.left, box.top, width, 20);
        context.setLineDash([]);
        context.fillStyle = style.getPropertyValue('--cp-text').trim();
        context.fillText(text, box.left + 5, box.top + 14);
      }
    }
  }, [state, trailSeconds, showLabels, clock, receivedAt, hudHost, expanded, minimized]);

  async function enable(enabled: boolean) {
    const previous = state;
    setState(current => ({ enabled, paused: current?.paused ?? false, error: null,
      frame: current?.frame ?? null, map: current?.map ?? null, footprint: current?.footprint ?? null }));
    setPending(true); setError('');
    try {
      const data = await request('spatial', { run_id: runId, episode_epoch: epoch, enabled }) as SpatialState;
      setState(data); setRevision(value => value + 1);
    } catch (failure) { setState(previous); setError(failure instanceof Error ? failure.message : String(failure)); }
    finally { setPending(false); }
  }

  async function run(path: string, body: unknown) {
    setPending(true); setArmed(false); setError('');
    try { setState(await request(path, body) as SpatialState); setRevision(value => value + 1); }
    catch (failure) { setError(failure instanceof Error ? failure.message : String(failure)); }
    finally { setPending(false); }
  }

  const mapStatus = !connected ? 'Disconnected' : state?.power?.on === false ? 'Off' : sensorError || state?.error ? 'Unavailable' : state?.paused ? 'Paused'
    : !state?.enabled ? 'Off' : !state.map ? 'Waiting' : state.power?.mode === 'idle' ? 'Idle' : state.map.stale ? 'Stale' : 'Live';
  const mapFigure = <figure className="observed-map-figure">
    <canvas ref={element => {
      canvas.current = element;
      return () => { if (canvas.current === element) canvas.current = null; };
    }} width={320} height={320} aria-label="Observed floor and obstacle map" data-state={mapStatus.toLowerCase()}
      onClick={event => {
        const bounds = event.currentTarget.getBoundingClientRect();
        const left = (event.clientX - bounds.left) / bounds.width * 320;
        const top = (event.clientY - bounds.top) / bounds.height * 320;
        const selected = labelBoxes.current.find(box => left >= box.left && left <= box.left + box.width && top >= box.top && top <= box.top + box.height);
        setSelectedLabel(selected?.id ?? null);
        if (selected && details.current) details.current.open = true;
      }} />
    {(!state?.map || !state.enabled) && <div className="viewport-map-empty" role="status" aria-label={!connected ? 'Disconnected' : state?.enabled ? 'Waiting for map' : 'Sensing off'}><Map size={24} /><span>{!connected ? 'Disconnected' : state?.enabled ? 'Waiting for map' : 'Sensing off'}</span></div>}
  </figure>;
  return <>
    {hudHost && createPortal(<section className="viewport-widget" aria-label="Spatial map HUD" data-minimized={minimized}>
      <div className="viewport-widget-heading"><h3><Map size={14} /> Spatial map</h3></div>
        <button className={minimized ? 'viewport-widget-restore' : 'icon-button viewport-widget-toggle'} type="button" aria-label={minimized ? 'Restore spatial map' : 'Minimize spatial map'}
          title={minimized ? `${mapStatus} / Restore spatial map` : 'Minimize spatial map'} aria-expanded={!minimized} aria-controls="spatial-map-hud-content"
          onClick={() => setMinimized(value => !value)}>{!minimized && <Minus size={16} />}</button>
      <div id="spatial-map-hud-content">
        <div className="viewport-map-view">
          {expanded ? <div className="viewport-map-placeholder" /> : mapFigure}
          <button className="icon-button viewport-map-expand" type="button" aria-label="Expand spatial map" title="Expand spatial map" onClick={() => setExpanded(true)}><Maximize2 size={16} /></button>
        </div>
        <div className="viewport-map-meta"><span role="status" data-state={mapStatus.toLowerCase()}>{mapStatus}</span><span title="Sensor-observed local map in the wheel-odometry frame">8 x 8 m</span></div>
      </div>
    </section>, hudHost)}
    {createPortal(<dialog ref={dialog} className="spatial-map-dialog" aria-label="Spatial map" onClose={() => setExpanded(false)}>
      <RobotControlSlot active={expanded} />
      <div className="robot-camera-toolbar"><h3>Observed spatial map</h3><span role="status">{mapStatus} / wheel odometry</span>
        <button className="icon-button" type="button" aria-label="Close spatial map" title="Close spatial map" onClick={() => setExpanded(false)}><X size={19} /></button></div>
      {expanded && mapFigure}
      <div className="spatial-legend"><span><i className="spatial-floor" />Observed floor</span><span><i className="spatial-obstacle" />Obstacles</span><span><i className="spatial-unknown" />Unknown</span></div>
    </dialog>, document.body)}
    <details className="spatial-section" ref={details}>
    <summary><Scan size={17} /> Spatial sensing</summary>
    <div className="panel-header">
      <label className="toggle"><input type="checkbox" aria-label="Enable spatial sensing" checked={state?.enabled ?? false}
        title={busy && state?.enabled ? 'Active sensing stays on while a controller owns the robot' : 'Enable paired RGB and metric depth'}
        disabled={!connected || state === null || state.power?.on === false || (busy && !!state.enabled) || pending} onChange={event => void enable(event.target.checked)} /> RGB + depth</label>
      <span className="tag" role="status">{mapStatus === 'Live' ? `Frame ${pair?.sequence ?? '-'} / ${state?.frame?.simulated_time_s.toFixed(2)} s` : mapStatus}</span>
    </div>
    <div className="continuous-controls" aria-label="Continuous local navigation">
      {busy && <span className="tag">Robot under controller control</span>}
      <label className="toggle" hidden={busy || navigating}><input type="checkbox" aria-label="Fold arms during floor scan" checked={compactArms}
        disabled={!connected || pending} onChange={event => setCompactArms(event.target.checked)} />Fold arms during scan</label>
      <button type="button" hidden={busy || navigating} disabled={!connected || state?.power?.on === false || busy || pending} onClick={() => void run('continuous/scan',
        {run_id:runId,episode_epoch:epoch,compact_arms:compactArms})}><Scan size={16} />Scan floor</button>
      <button type="button" hidden={busy || navigating || !pair} aria-pressed={armed} disabled={!connected || state?.power?.on === false || busy || pending || !pair || state?.map?.stale}
        title="Arm selection, then click a visible floor point in the paired RGB image" onClick={() => setArmed(value => !value)}>
        <Crosshair size={16} />{armed ? 'Cancel selection' : 'Select destination'}</button>
    </div>
    {state?.continuous && <div className="continuous-status" role="status" aria-label="Continuous navigation status"
      data-status={state.continuous.status}>
      <strong>{state.continuous.status === 'running' ? 'Following observed path' : state.continuous.status === 'arrived' ? 'Destination reached' : state.continuous.reason}</strong>
      <span>{state.continuous.remaining_m.toFixed(2)} m remaining / {state.continuous.updates} updates / {state.continuous.buffer_stops} buffer stops</span>
      <span>{state.continuous.replans ?? 0} local replans / {state.continuous.handoffs ?? 0} moving handoffs / max update gap {((state.continuous.maximum_update_gap_s ?? 0) * 1000).toFixed(0)} ms</span>
    </div>}
    {state?.enabled && <>
      <div className="continuous-controls" aria-label="Map overlays">
        <label className="toggle"><input type="checkbox" checked={showLabels} onChange={event => setShowLabels(event.target.checked)} />Luna sightings</label>
        <label>Trail timeout <select aria-label="Trail timeout" value={trailSeconds} onChange={event => setTrailSeconds(Number(event.target.value))}>
          <option value={0}>Off</option><option value={30}>30 seconds</option><option value={120}>2 minutes</option><option value={300}>5 minutes</option>
        </select></label>
      </div>
      <div className="spatial-grid">
        <figure><figcaption>Paired RGB / 160 x 120{armed ? ' / Select floor destination' : ''}</figcaption><div className={`spatial-image ${armed ? 'target-selection' : ''}`}>
          {pair && <img alt="Spatial paired RGB" src={pair.rgb} data-sequence={pair.sequence} />}
          {armed && pair && <button type="button" className="spatial-target" aria-label="Drive continuously to floor point" title="Drive continuously to the selected observed floor point"
            onClick={event => {
              const bounds = event.currentTarget.getBoundingClientRect();
              const pixel = event.detail === 0 ? [.5, .5] : [(event.clientX - bounds.left) / bounds.width, (event.clientY - bounds.top) / bounds.height];
              void run('continuous/start', {run_id:runId,episode_epoch:epoch,spatial_sequence:pair.sequence,pixel});
            }}><Crosshair size={28} /></button>}
        </div></figure>
        <figure><figcaption>Depth / 0.015-4 m</figcaption><div className="spatial-image">{pair && <img alt="Spatial metric depth" src={pair.depth} data-sequence={pair.sequence} />}</div></figure>
        {!hudHost && mapFigure}
      </div>
      {!!state.history?.labels.length && <details className="place-sightings" open={selectedLabel !== null}>
        <summary>Unverified place sightings ({state.history.labels.length})</summary>
        <ul>{state.history.labels.map(label => <li key={label.id} aria-current={selectedLabel === label.id ? 'true' : undefined}>
          <strong>{label.label}</strong> <span>Seen from ({label.position_m.map(value => value.toFixed(1)).join(', ')}) m / {Math.floor(label.age_s)} s ago / observation {label.observation_seq}</span>
          <p>{label.evidence}</p>
        </li>)}</ul>
      </details>}
      <div className="spatial-legend"><span><i className="spatial-floor" />Observed floor: {state.map?.observed_floor_cells ?? 0}</span>
        <span><i className="spatial-obstacle" />Obstacles: {state.map?.obstacle_cells ?? 0}</span><span><i className="spatial-unknown" />Unknown</span>
        <span>Footprint radius: {state.footprint?.radius_m.toFixed(2) ?? '-'} m</span><span>Odometry frame / observed local routes only</span></div>
    </>}
    {error && <div className="error" role="alert">{error}</div>}
    {(sensorError || state?.error) && <div className="error" role="alert">{sensorError || state?.error}</div>}
    {frameError && state?.enabled && <div className="error" role="alert">{frameError}</div>}
  </details></>;
}