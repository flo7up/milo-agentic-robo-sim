import { useEffect, useState } from 'react';
import { Box, CheckCircle2, ChevronDown, CircleStop, Download, ExternalLink, FlaskConical, Focus, LoaderCircle, Map, Pause, Play, RefreshCw, RotateCcw, Route, Search, TriangleAlert } from 'lucide-react';
import { RecordedScene, type RecordedRoute } from './RecordedScene';
import { ViewNavigation, type TestView } from './ViewNavigation';
import type { LiveState } from './types';

type Trial = {
  running?: boolean; challenge_sha256?: string; task_sha256?: string; goal?: string; updated_at?: number | null;
  environment?: string;
  mean_translation_speed_mps?: number | null; stationary_wall_s?: number | null; route_arrivals?: number | null;
  case_id: string; challenge: string; title: string; evidence: string; status: string; verified_success: boolean;
  physics_success: boolean | null; recording_complete: boolean; assisted: boolean; completion_s: number | null;
  elapsed_s: number | null; distance_m: number | null; contact_episodes: number | null; input_tokens: number | null;
  output_tokens: number | null; inference_median_s: number | null; turns: number | null; termination: string;
  rendering: string; false_completion_claim: boolean; image_url: string | null; trajectory_url?: string | null;
};
type Batch = {
  session_id?: string; running?: boolean; variant_id?: string | null;
  architecture?: {id:string; name:string; version:string; revision:string; version_key:string} | null;
  model_variant?: {provider:string; deployment:string; revision:string; configuration:Record<string,unknown>} | null;
  id: string; name: string; date: string; date_source: string; design: string; source_sha256: string; mode: string;
  evidence: string; model: string; reasoning: string; budget_s: number | null; history: string; legacy: boolean;
  source_changed: boolean; planned: number; successes: number; trials: Trial[];
};
type Catalog = { batches: Batch[]; skipped: number; truncated: boolean; scope: string };

const evidenceNames: Record<string, string> = {real_model:'Real model',scripted_test:'Scripted test',scripted_reference:'Scripted reference',unknown:'Unknown evidence'};
const numeric = (value: number | null, digits = 1) => value === null ? 'Not recorded' : value.toLocaleString('en-US', {maximumFractionDigits:digits});
const measurement = (value: number | null, unit: string) => value === null ? 'Not recorded' : `${numeric(value)} ${unit}`;
const evidenceName = (value: string) => evidenceNames[value] ?? value.replaceAll('_',' ');
function dateLabel(value: string) {
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? 'Date not recorded' : date.toLocaleString();
}

function TrajectoryView({url, title, updatedAt}: {url?: string | null; title: string; updatedAt?: number | null}) {
  const [route, setRoute] = useState<RecordedRoute | null>(null);
  const [error, setError] = useState('');
  const [revision, setRevision] = useState(0);
  const [mediaRevision, setMediaRevision] = useState(0);
  const [cursor, setCursor] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(4);
  const [view, setView] = useState<'top' | 'perspective' | 'path'>('top');
  const [wholeRoom, setWholeRoom] = useState(false);
  useEffect(() => {
    if (!url) return;
    const controller = new AbortController();
    setError(''); setPlaying(false);
    const timeout = setTimeout(() => {
      controller.abort(); setError('Recorded route timed out.');
    }, 15000);
    void fetch(url, {signal:controller.signal}).then(async response => {
      if (!response.ok) throw new Error('Recorded route unavailable.');
      const data: RecordedRoute = await response.json();
      if (!data.points?.length) throw new Error('Recorded route is empty.');
      if (!controller.signal.aborted) { setRoute(data); setCursor(data.points.at(-1)!.wall_s); }
    }).catch(failure => { if (!controller.signal.aborted) setError(String(failure.message ?? failure)); })
      .finally(() => clearTimeout(timeout));
    return () => { clearTimeout(timeout); controller.abort(); };
  }, [url,revision,updatedAt]);
  const end = route?.points.at(-1)?.wall_s ?? 0;
  useEffect(() => {
    if (!playing) return;
    let previous = performance.now();
    const timer = setInterval(() => {
      const now = performance.now();
      const elapsed = (now-previous)/1000;
      previous = now;
      setCursor(value => Math.min(end,value+elapsed*speed));
    },100);
    return () => clearInterval(timer);
  }, [playing,speed,end]);
  useEffect(() => { if (cursor >= end) setPlaying(false); }, [cursor,end]);
  if (!url) return <div className="results-no-route">Movement map not recorded</div>;
  if (error) return <div className="results-no-route"><span role="status">{error}</span><button type="button" className="icon-button" title="Retry movement map" aria-label="Retry movement map" onClick={()=>setRevision(value=>value+1)}><RefreshCw size={17}/></button></div>;
  if (!route) return <div className="results-no-route" role="status"><LoaderCircle size={20} className="loading-icon"/>Loading movement map...</div>;
  const first = route.points[0];
  const last = route.points.at(-1)!;
  let current = first;
  for (const point of route.points) {
    if (point.wall_s > cursor) break;
    current = point;
  }
  const [minimumX,minimumY,maximumX,maximumY] = route.bounds_m;
  const centerX=(minimumX+maximumX)/2, centerY=(minimumY+maximumY)/2;
  const scale=Math.min(328/Math.max(2,maximumX-minimumX+1),240/Math.max(2,maximumY-minimumY+1));
  const projectX=(value:number)=>220+(value-centerX)*scale;
  const projectY=(value:number)=>144-(value-centerY)*scale;
  const segments: RecordedRoute['points'][]=[];
  for (const point of route.points) {
    if (segments.at(-1)?.at(-1)?.segment!==point.segment) segments.push([]);
    segments.at(-1)!.push(point);
  }
  const coordinates=(points:RecordedRoute['points'])=>points.map(point=>`${projectX(point.x)},${projectY(point.y)}`).join(' ');
  const ticks=[0,1,2,3,4];
  return <figure className="results-route" aria-label={`Movement map: ${title}`}>
    <div className="results-route-heading"><h4>Movement map</h4><span>Recorded world position</span></div>
    {route.scene && <><div className="results-scene-toolbar">
      <div className="results-scene-modes" role="group" aria-label="Movement map view">
        <button type="button" title="Top-down environment" aria-label="Top-down environment" aria-pressed={view==='top'} onClick={()=>setView('top')}><Map size={16}/><span>Top</span></button>
        <button type="button" title="3D environment" aria-label="3D environment" aria-pressed={view==='perspective'} onClick={()=>setView('perspective')}><Box size={16}/><span>3D</span></button>
        <button type="button" title="Path only" aria-label="Path only" aria-pressed={view==='path'} onClick={()=>setView('path')}><Route size={16}/><span>Path</span></button>
      </div>
      {view!=='path' && <button type="button" className="icon-button" title={wholeRoom?'Frame route':'Frame room'} aria-label={wholeRoom?'Frame route':'Frame room'} onClick={()=>setWholeRoom(value=>!value)}><Focus size={17}/></button>}
    </div><p className="results-scene-provenance">{route.scene.source==='recorded_initial'?'Recorded initial environment':'Reconstructed current environment; historical layout unverified'}</p></>}
    {route.scene && view!=='path'?<RecordedScene route={route} cursor={cursor} view={view} wholeRoom={wholeRoom}/>:<>
    <svg viewBox="0 0 400 320" role="img" aria-label={`Recorded robot path: ${title}`}>
      <title>{title}: recorded XY movement in metres, start, end and contact events</title>
      {ticks.map(tick=>{
        const horizontal=56+tick*82, vertical=24+tick*60;
        return <g key={tick} className="results-route-grid"><line x1={horizontal} x2={horizontal} y1={24} y2={264}/><line x1={56} x2={384} y1={vertical} y2={vertical}/>
          <text x={horizontal} y={284} textAnchor="middle">{numeric(centerX+(horizontal-220)/scale)}</text>
          <text x={47} y={vertical+4} textAnchor="end">{numeric(centerY-(vertical-144)/scale)}</text></g>;
      })}
      <text className="results-route-axis" x={220} y={309} textAnchor="middle">X (m)</text>
      <text className="results-route-axis" transform="translate(14 144) rotate(-90)" textAnchor="middle">Y (m)</text>
      {segments.map((points,index)=><g key={index}><polyline className="results-route-full" points={coordinates(points)}/><polyline className="results-route-travelled" points={coordinates(points.filter(point=>point.wall_s<=cursor))}/></g>)}
      {route.contacts.map((point,index)=><circle key={index} className="results-route-contact" cx={projectX(point.x)} cy={projectY(point.y)} r={5}><title>Recorded contact at {numeric(point.wall_s)} s</title></circle>)}
      <circle className="results-route-start" cx={projectX(first.x)} cy={projectY(first.y)} r={7}><title>Start at {numeric(first.wall_s)} s</title></circle>
      <rect className="results-route-end" x={projectX(last.x)-5} y={projectY(last.y)-5} width={10} height={10}><title>End at {numeric(last.wall_s)} s</title></rect>
      <circle className="results-route-cursor" data-time={current.wall_s} cx={projectX(current.x)} cy={projectY(current.y)} r={4}><title>{numeric(current.wall_s)} s: {current.activity}</title></circle>
    </svg>
    </>}
    <div className="results-route-legend"><span><i className="route-start"/>Start</span><span><i className="route-end"/>End</span><span><i className="route-contact"/>Contact</span><span><i className="route-cursor"/>Selected time</span></div>
    <div className="results-route-controls">
      <button type="button" className="icon-button" aria-label={playing?'Pause route':'Play route'} title={playing?'Pause route':'Play route'} disabled={end<=first.wall_s} onClick={()=>{if(cursor>=end)setCursor(first.wall_s);setPlaying(value=>!value);}}>{playing?<Pause size={17}/>:<Play size={17}/>}</button>
      <button type="button" className="icon-button" aria-label="Rewind route" title="Rewind route" onClick={()=>{setPlaying(false);setCursor(first.wall_s);}}><RotateCcw size={17}/></button>
      <input type="range" aria-label="Route time" min={first.wall_s} max={end} step="any" value={cursor} disabled={end<=first.wall_s} aria-valuetext={`${numeric(cursor)} seconds`} onChange={event=>{setPlaying(false);setCursor(Number(event.target.value));}}/>
      <select aria-label="Route playback speed" value={speed} onChange={event=>setSpeed(Number(event.target.value))}>{[1,4,10].map(value=><option key={value} value={value}>{value}x</option>)}</select>
    </div>
    <output className="results-route-readout"><strong>{numeric(cursor)} / {numeric(end)} s</strong><span>X {numeric(current.x,2)} m / Y {numeric(current.y,2)} m</span><span>{current.activity.replaceAll('_',' ')} / {current.status.replaceAll('_',' ')}</span></output>
    <figcaption>{numeric(route.sample_count,0)} recorded samples{route.downsampled?` / ${numeric(route.points.length,0)} displayed`:''}{route.contact_markers_truncated?' / Contact markers capped at 200':''}</figcaption>
  </figure>;
}

export function TestResults({onNavigate, liveState, onStop}: {onNavigate?: (view: TestView) => void; liveState?: LiveState | null; onStop?: () => Promise<unknown>}) {
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [revision, setRevision] = useState(0);
  const [mediaRevision, setMediaRevision] = useState(0);
  const [query, setQuery] = useState('');
  const [evidence, setEvidence] = useState('real_model');
  const [architecture, setArchitecture] = useState('all');
  const [modelVariant, setModelVariant] = useState('all');
  const [followCurrent, setFollowCurrent] = useState(false);
  const [scenario, setScenario] = useState(()=>new URLSearchParams(window.location.search).get('scenario') || 'all');
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [trialIndex, setTrialIndex] = useState(0);
  const [visibleCount, setVisibleCount] = useState(3);
  const [imageFailed, setImageFailed] = useState(false);
  const [downloadError, setDownloadError] = useState('');
  useEffect(() => {
    const refresh = () => { if (!loading && document.visibilityState === 'visible') setRevision(value=>value+1); };
    window.addEventListener('focus', refresh);
    document.addEventListener('visibilitychange', refresh);
    const timer = window.setInterval(refresh, liveState?.agent.active ? 2000 : 15000);
    return () => {
      window.removeEventListener('focus', refresh);
      document.removeEventListener('visibilitychange', refresh);
      window.clearInterval(timer);
    };
  }, [liveState?.agent.active, loading]);
  useEffect(() => { setRevision(value=>value+1); }, [liveState?.agent.active, liveState?.agent.session_id]);
  useEffect(() => {
    const controller = new AbortController();
    setLoading(true); setError('');
    void fetch('/api/test-results', {signal:controller.signal}).then(async response => {
      if (!response.ok) throw new Error(`Saved results unavailable (${response.status})`);
      const data: Catalog = await response.json();
      if (!controller.signal.aborted) setCatalog(data);
    }).catch(failure => { if (!controller.signal.aborted) setError(String(failure)); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [revision]);
  useEffect(() => { setVisibleCount(3); setSelectedId(null); }, [query,evidence,scenario,architecture,modelVariant]);
  const batches = catalog?.batches ?? [];
  const matchingTrials = (batch: Batch) => batch.trials.filter(trial=>scenario==='all' || trial.challenge===scenario);
  const countLabel = (batch: Batch) => scenario==='all'?`${batch.successes} / ${batch.planned}`:
    `${matchingTrials(batch).filter(trial=>trial.verified_success).length} / ${matchingTrials(batch).length}`;
  const filtered = batches.filter(batch => (evidence === 'all' || batch.evidence === evidence)
    && (architecture === 'all' || batch.architecture?.version_key === architecture)
    && (modelVariant === 'all' || batch.model_variant?.revision === modelVariant)
    && (scenario === 'all' || batch.trials.some(trial => trial.challenge === scenario))
    && `${batch.name} ${batch.design} ${batch.model} ${batch.trials.map(trial => `${trial.title} ${trial.case_id}`).join(' ')}`.toLowerCase().includes(query.toLowerCase()));
  const selected = (followCurrent ? filtered.find(batch => batch.session_id === liveState?.agent.session_id) : undefined)
    ?? filtered.find(batch => batch.id === selectedId) ?? filtered[0];
  const trials = selected ? matchingTrials(selected) : [];
  const trial = trials[trialIndex] ?? trials[0];
  const progress = new globalThis.Map<string, {architecture:string;model:string;task:string;settings:string;assistance:string;attempts:number;passes:number;running:number;times:number[]}>();
  for (const batch of filtered) {
    if (!batch.variant_id || !batch.architecture || !batch.model_variant || batch.source_changed || batch.evidence !== 'real_model') continue;
    for (const entry of matchingTrials(batch)) {
      if (entry.evidence !== 'real_model') continue;
      const key = `${batch.variant_id}/${entry.task_sha256 || batch.id}/${entry.environment}/${entry.assisted}/${entry.rendering}/${batch.budget_s}/${batch.history}`;
      const row = progress.get(key) ?? {architecture:batch.architecture.version_key,model:`${batch.model_variant.deployment} / ${batch.model_variant.revision}`,
        task:`${entry.title} / ${entry.goal ?? 'Scenario goal'} / ${entry.environment ?? 'standalone'}`,
        settings:`${measurement(batch.budget_s,'s')} / ${batch.history} / ${entry.rendering}`,
        assistance:entry.assisted?'Operator-assisted':'Unassisted',attempts:0,passes:0,running:0,times:[]};
      row.attempts++; row.passes+=Number(entry.verified_success); row.running+=Number(!!entry.running);
      if (entry.verified_success && entry.completion_s !== null) row.times.push(entry.completion_s);
      progress.set(key,row);
    }
  }
  const median = (values:number[]) => { const ordered=[...values].sort((first,second)=>first-second); const middle=Math.floor(ordered.length/2); return ordered.length ? (ordered[middle]+ordered[Math.floor((ordered.length-1)/2)])/2 : null; };
  useEffect(() => { setTrialIndex(0); setDownloadError(''); }, [selected?.id,scenario]);
  useEffect(() => { setImageFailed(false); }, [trial?.image_url,revision]);
  const scenarios = [...new Set(batches.flatMap(batch => batch.trials.map(trial => trial.challenge)))].sort();
  function download() {
    if (!selected) return;
    try {
      const url = URL.createObjectURL(new Blob([JSON.stringify(selected,null,2)], {type:'application/json'}));
      const link = document.createElement('a'); link.href=url; link.download=`milo-results-${selected.id}.json`;
      link.click(); setTimeout(() => URL.revokeObjectURL(url),1000); setDownloadError('');
    } catch { setDownloadError('Download failed. Please retry.'); }
  }
  return <>
    <header className="topbar results-header"><div className="brand"><span className="brand-mark"><FlaskConical size={24}/></span><h1>Milo <span>/ Test results</span></h1></div>
      <span className="tag">Robot test archive</span></header>
    <ViewNavigation current="archive" onNavigate={onNavigate} />
    <main className="results-page">
      {liveState?.agent.session_id && <div className="results-live-run" role="status">
        <div><strong>{liveState.agent.active ? 'Test running' : 'Latest cockpit test'}</strong><span>{liveState.agent.goal}</span></div>
        <button type="button" onClick={()=>{setQuery('');setScenario('all');setEvidence('all');setArchitecture('all');setModelVariant('all');setFollowCurrent(true);setRevision(value=>value+1);}}>Show current test</button>
        {liveState.agent.active && onStop && <button type="button" className="stop-button" onClick={()=>{void onStop().catch(failure=>setError(String(failure)));}}><CircleStop size={17}/>Stop</button>}
      </div>}
      <div className="results-toolbar">
        <label className="results-search"><Search size={17}/><input type="search" aria-label="Search saved tests" placeholder="Search runs, designs or scenarios" value={query} onChange={event=>setQuery(event.target.value)}/></label>
        <label>Evidence<select aria-label="Filter test evidence" value={evidence} onChange={event=>setEvidence(event.target.value)}><option value="all">All evidence</option>{Object.entries(evidenceNames).map(([value,label])=><option key={value} value={value}>{label}</option>)}</select></label>
        <label>Scenario<select aria-label="Filter test scenario" value={scenario} onChange={event=>setScenario(event.target.value)}><option value="all">All scenarios</option>{scenarios.map(value=><option key={value} value={value}>{value.replaceAll('_',' ')}</option>)}</select></label>
        <label>Architecture<select aria-label="Filter architecture version" value={architecture} onChange={event=>setArchitecture(event.target.value)}><option value="all">All versions</option>{[...new globalThis.Map(batches.filter(batch=>batch.architecture).map(batch=>[batch.architecture!.version_key,batch.architecture!] as const)).values()].map(item=><option key={item.version_key} value={item.version_key}>{item.name} {item.version}+{item.revision}</option>)}</select></label>
        <label>Model variant<select aria-label="Filter model variant" value={modelVariant} onChange={event=>setModelVariant(event.target.value)}><option value="all">All models</option>{[...new globalThis.Map(batches.filter(batch=>batch.model_variant).map(batch=>[batch.model_variant!.revision,batch.model_variant!] as const)).values()].map(item=><option key={item.revision} value={item.revision}>{item.deployment} / {String(item.configuration.reasoning ?? 'default')} / {item.revision}</option>)}</select></label>
        <button type="button" className="icon-button" aria-label="Refresh saved results" title="Refresh saved results" disabled={loading} onClick={()=>{setRevision(value=>value+1);setMediaRevision(value=>value+1);}}>{loading?<LoaderCircle size={18} className="loading-icon"/>:<RefreshCw size={18}/>}</button>
      </div>
      {!!progress.size && <details className="variant-progress"><summary>Model / architecture progress</summary><div className="results-table-scroll"><table className="results-table"><thead><tr><th>Architecture</th><th>Model variant</th><th>Task</th><th>Budget / history / renderer</th><th>Assistance</th><th>Verified / attempts</th><th>Running</th><th>Median completion</th></tr></thead><tbody>
        {[...progress.entries()].map(([key,row])=><tr key={key}><td>{row.architecture}</td><td>{row.model}</td><td>{row.task}</td><td>{row.settings}</td><td>{row.assistance}</td><td>{row.passes} / {row.attempts}</td><td>{row.running}</td><td>{measurement(median(row.times),'s')}</td></tr>)}
      </tbody></table></div></details>}
      {error && <div role="alert" className="error">{error}</div>}
      {downloadError && <div role="alert" className="error">{downloadError}</div>}
      {catalog && (catalog.skipped > 0 || catalog.truncated) && <p className="results-warning"><TriangleAlert size={16}/>{catalog.skipped} unreadable reports skipped.{catalog.truncated?' Archive limit reached.':''}</p>}
      {!catalog && loading ? <div role="status" className="results-empty">Loading saved tests...</div> : !filtered.length ?
        <div role="status" className="results-empty"><FlaskConical size={30}/><h2>{!catalog && error?'Results unavailable':batches.length?'No matching tests':'No saved challenge evaluations'}</h2>{batches.length>0 && <button type="button" onClick={()=>{setQuery('');setEvidence('all');setScenario('all');setArchitecture('all');setModelVariant('all');}}><RefreshCw size={16}/>Clear filters</button>}</div> :
        <div className="results-workspace">
          <aside className="results-list" aria-label="Saved test runs"><div className="results-list-title">Runs <span>{filtered.length}</span></div>
            {filtered.slice(0,visibleCount).map(batch=><button type="button" key={batch.id} className="results-run" aria-pressed={selected?.id===batch.id} onClick={()=>{setFollowCurrent(false);setSelectedId(batch.id);}}>
              <strong>{batch.name}</strong><span>{dateLabel(batch.date)}</span><div><span>{evidenceName(batch.evidence)}</span><span>{batch.running?'Running':`${countLabel(batch)} verified`}</span></div><small>{batch.architecture?.version_key ?? batch.design}</small>
            </button>)}
            {visibleCount < filtered.length && <div className="results-more"><button type="button" onClick={()=>setVisibleCount(value=>value+5)}><ChevronDown size={16}/>Show more</button><button type="button" onClick={()=>setVisibleCount(filtered.length)}>View all ({filtered.length})</button></div>}
          </aside>
          {selected && <section className="results-detail" aria-label="Test run details">
            <div className="results-detail-heading"><div><span className="eyebrow">{evidenceName(selected.evidence)}</span><h2>{selected.name}</h2><p>{dateLabel(selected.date)}{selected.date_source==='file_modified'?' (file date)':''}</p></div>
              <button type="button" aria-label="Download selected test results" title="Download selected test results" onClick={download}><Download size={17}/><span>JSON</span></button></div>
            <dl className="results-facts"><div><dt>Design</dt><dd>{selected.design}</dd></div><div><dt>Controller</dt><dd>{selected.mode.replaceAll('_',' ')}</dd></div><div><dt>Deployment</dt><dd>{selected.model}</dd></div><div><dt>Reasoning</dt><dd>{selected.reasoning}</dd></div><div><dt>Time budget</dt><dd>{measurement(selected.budget_s,'s')}</dd></div><div><dt>Camera history</dt><dd>{selected.history}</dd></div></dl>
            <dl className="results-facts"><div><dt>Architecture version</dt><dd>{selected.architecture?.version_key ?? 'Legacy / not recorded'}</dd></div><div><dt>Model configuration</dt><dd>{selected.model_variant ? `${selected.model_variant.deployment} / ${selected.model_variant.revision}` : 'Legacy / not recorded'}</dd></div></dl>
            {(selected.legacy || selected.source_changed) && <p className="results-warning"><TriangleAlert size={16}/>{selected.source_changed?'Source changed during this run.':'Legacy report: design and configuration may be incomplete.'}</p>}
            <div className="results-trial-heading"><h3>Trials</h3><span>{countLabel(selected)} verified passes</span></div>
            <div className="results-table-scroll"><table className="results-table"><thead><tr><th scope="col">Scenario / trial</th><th scope="col">Result</th><th scope="col">Completion</th><th scope="col">Contacts</th><th scope="col">Tokens</th></tr></thead><tbody>
              {trials.map((entry,index)=><tr key={`${entry.case_id}-${index}`} data-selected={trial===entry}><th scope="row"><button type="button" onClick={()=>setTrialIndex(index)} aria-pressed={trial===entry}>{entry.title}<small>{entry.case_id}</small></button></th>
                <td><span className={entry.verified_success?'ok':''}>{entry.status}</span>{entry.false_completion_claim && <small className="bad">False completion claim</small>}</td><td>{entry.completion_s===null?'Not verified':`${numeric(entry.completion_s)} s`}</td><td>{numeric(entry.contact_episodes,0)}</td><td>{entry.input_tokens===null || entry.output_tokens===null?'Not recorded':numeric(entry.input_tokens+entry.output_tokens,0)}</td></tr>)}
            </tbody></table></div>
            {trial && <section className="results-trial" aria-label="Selected trial details"><div><h3>{trial.title}</h3><p className={trial.verified_success?'ok':'results-trial-status'}>{trial.verified_success && <CheckCircle2 size={16}/>} {trial.status}</p>
              <dl className="results-facts"><div><dt>Environment</dt><dd>{trial.environment==='shared_apartment_v1'?'Shared Apartment V1':trial.environment==='standalone'?'Standalone':trial.environment??'Not recorded'}</dd></div><div><dt>Elapsed</dt><dd>{measurement(trial.elapsed_s,'s')}</dd></div><div><dt>Travel</dt><dd>{measurement(trial.distance_m,'m')}</dd></div><div><dt>Inference median</dt><dd>{measurement(trial.inference_median_s,'s')}</dd></div><div><dt>Input / output tokens</dt><dd>{numeric(trial.input_tokens,0)} / {numeric(trial.output_tokens,0)}</dd></div><div><dt>Termination</dt><dd>{trial.termination}</dd></div><div><dt>Renderer</dt><dd>{trial.rendering}</dd></div></dl>
              <dl className="results-facts results-motion"><div><dt>Moving speed (sim)</dt><dd>{trial.mean_translation_speed_mps==null?'Not recorded':`${numeric(trial.mean_translation_speed_mps,2)} m/s`}</dd></div><div><dt>Stationary time (wall)</dt><dd>{measurement(trial.stationary_wall_s??null,'s')}</dd></div><div><dt>Route arrivals</dt><dd>{numeric(trial.route_arrivals??null,0)}</dd></div></dl>
              <p className="results-provenance">{evidenceName(trial.evidence)} / {trial.assisted?'Operator-assisted':'No recorded operator assistance'} / {trial.recording_complete?'Complete recording':'Recording incomplete or unavailable'}</p></div>
              <div className="results-visuals"><TrajectoryView key={`${selected.id}-${trial.case_id}-${mediaRevision}`} url={trial.trajectory_url} title={trial.title} updatedAt={trial.updated_at}/>
              {trial.image_url && !imageFailed ? <figure><a href={trial.image_url} target="_blank" rel="noopener noreferrer" title="Open saved final camera"><img src={trial.image_url} alt={`Saved final camera: ${trial.title}`} width={320} height={240} onError={()=>setImageFailed(true)}/></a><figcaption>Final head camera <ExternalLink size={12}/></figcaption></figure> : <div className="results-no-image">Final camera not available</div>}
              </div>
            </section>}
            {!!selected.source_sha256 && <details className="results-source"><summary>Source fingerprint</summary><code>{selected.source_sha256}</code></details>}
          </section>}
        </div>}
      <footer>Recorded robot runs, including scripted tests and browser control sessions. Legacy dates use file modification time. Model identity and comparability are not guaranteed.</footer>
    </main>
  </>;
}