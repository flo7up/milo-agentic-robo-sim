import {useEffect, useState} from 'react';
import {CheckCircle2, Circle, CircleStop, Download, ListChecks, LoaderCircle, Play, Route, TriangleAlert} from 'lucide-react';
import type {LiveState, Reasoning, RegressionCase} from './types';

type RegressionRoute = {sample_count:number;distance_m:number;bounds_m:number[];downsampled:boolean;
  points:{x:number;y:number;wall_s:number;segment:number}[];contacts:{x:number;y:number}[]};

function RegressionProgress({entry}:{entry:RegressionCase}) {
  const grade=entry.evaluation;
  if(!grade)return ['passed','failed','invalid','cancelled'].includes(entry.status)?<small className="regression-case-detail">Progress not recorded</small>:null;
  return <div className="regression-progress" role="group" aria-label={`Task progress for ${entry.title}`}>
    <div className="regression-progress-heading"><span>{grade.metric}</span><strong>{grade.progress_pct.toFixed(1)}%</strong>
      <small>Objectives {grade.completed_objectives}/{grade.total_objectives}</small></div>
    <progress max={100} value={grade.progress_pct} aria-label={`${entry.title}: ${grade.metric}`}/>
    <div className="regression-progress-values">
      {grade.kind==='circuit'?<>
        <span>{(grade.valid_degrees??0).toFixed(1)} / 360 degrees</span>
        <span>{(grade.remaining_degrees??360).toFixed(1)} degrees remaining</span>
        <span>Return error {grade.return_error_m==null?'not available':`${grade.return_error_m.toFixed(2)} m`}</span>
      </>:<>
        <span>{(grade.remaining_m??0).toFixed(2)} m remaining{grade.remaining_pct!=null?` (${grade.remaining_pct.toFixed(1)}% of initial gap)`:''}</span>
        <span>Target centre {(grade.center_distance_m??0).toFixed(2)} m away</span>
      </>}
      {grade.dwell_s!==undefined&&<span>Hold {grade.dwell_s.toFixed(2)} / {grade.required_dwell_s?.toFixed(2)} s</span>}
    </div>
    <ul className="regression-checks">{grade.checks.map(check=><li key={check.label} data-complete={check.complete}>
      {check.complete?<CheckCircle2 size={13}/>:<Circle size={13}/>}<span>{check.label}</span>
    </li>)}</ul>
    {grade.detail&&<small>{grade.detail}</small>}
  </div>;
}

function RegressionPath({entry,active}:{entry:RegressionCase;active:boolean}) {
  const [open,setOpen]=useState(false);
  const [route,setRoute]=useState<RegressionRoute|null>(null);
  const [error,setError]=useState('');
  useEffect(()=>{if(active)setOpen(true);},[active]);
  useEffect(()=>{
    if(!open||!entry.trajectory_url)return;
    let mounted=true;
    const load=()=>void fetch(entry.trajectory_url!,{cache:'no-store'}).then(async response=>{
      if(!response.ok)throw new Error('Path is not available yet');
      return response.json();
    }).then(value=>{if(mounted){setRoute(value);setError('');}}).catch(failure=>{if(mounted&&!active)setError(String(failure.message??failure));});
    load();
    const timer=active?window.setInterval(load,1000):undefined;
    return()=>{mounted=false;if(timer!==undefined)window.clearInterval(timer);};
  },[open,active,entry.trajectory_url]);
  const target=entry.evaluation,centre=target?.target_xy_m,radius=target?.target_radius_m??.1;
  const targetBounds=target?.target_bounds_m??(centre?[centre[0]-radius,centre[1]-radius,centre[0]+radius,centre[1]+radius]:null);
  const pathBounds=route?.bounds_m??[0,0,1,1];
  const bounds=targetBounds?[Math.min(pathBounds[0],targetBounds[0]),Math.min(pathBounds[1],targetBounds[1]),
    Math.max(pathBounds[2],targetBounds[2]),Math.max(pathBounds[3],targetBounds[3])]:pathBounds;
  const spanX=Math.max(.5,bounds[2]-bounds[0]),spanY=Math.max(.5,bounds[3]-bounds[1]);
  const scale=Math.min(312/spanX,132/spanY),centerX=(bounds[0]+bounds[2])/2,centerY=(bounds[1]+bounds[3])/2;
  const x=(value:number)=>180+(value-centerX)*scale,y=(value:number)=>82-(value-centerY)*scale;
  const segments:RegressionRoute['points'][]=[];
  for(const point of route?.points??[]){if(segments.at(-1)?.at(-1)?.segment!==point.segment)segments.push([]);segments.at(-1)!.push(point);}
  const first=route?.points[0],last=route?.points.at(-1);
  return <details className="regression-path" open={open} onToggle={event=>setOpen(event.currentTarget.open)}>
    <summary><Route size={14}/><span>Path</span>{route&&<small>{route.distance_m.toFixed(2)} m / {route.sample_count.toLocaleString()} samples</small>}</summary>
    {route?<svg viewBox="0 0 360 164" role="img" aria-label={`Recorded path for ${entry.title}`}>
      <title>{entry.title}: recorded world-position path</title>
      <path className="regression-path-axis" d="M16 82H344 M180 16V148"/>
      {centre&&<g className="regression-path-target"><title>Evaluator target</title>
        {target?.target_bounds_m?<rect x={x(targetBounds![0])} y={y(targetBounds![3])}
          width={(targetBounds![2]-targetBounds![0])*scale} height={(targetBounds![3]-targetBounds![1])*scale}/>
          :<circle cx={x(centre[0])} cy={y(centre[1])} r={radius*scale}/>}
        <path d={`M${x(centre[0])-4} ${y(centre[1])}h8 M${x(centre[0])} ${y(centre[1])-4}v8`}/>
      </g>}
      {segments.map((points,index)=><polyline key={index} className="regression-path-line" points={points.map(point=>`${x(point.x)},${y(point.y)}`).join(' ')}/>) }
      {route.contacts.map((point,index)=><circle key={index} className="regression-path-contact" cx={x(point.x)} cy={y(point.y)} r="4"/>)}
      {first&&<circle className="regression-path-start" cx={x(first.x)} cy={y(first.y)} r="5"/>}
      {last&&<rect className="regression-path-end" x={x(last.x)-5} y={y(last.y)-5} width="10" height="10"/>}
    </svg>:<div className="regression-path-empty" role="status">{error||(active?'Recording path...':'Path unavailable')}</div>}
    {route&&<div className="regression-path-legend"><span>Circle: start</span><span>Square: end</span>
      {centre&&<span>Dashed: evaluator target</span>}</div>}
  </details>;
}

export function RegressionControl({state, connected, request}: {state:LiveState;connected:boolean;
  request:(path:string,body?:unknown)=>Promise<unknown>}) {
  const suite=state.regression;
  const [expanded,setExpanded]=useState(false);
  const [model,setModel]=useState('hybrid');
  const [reasoning,setReasoning]=useState<Reasoning>('none');
  const [pending,setPending]=useState(false);
  const [error,setError]=useState('');
  const selectedModel=suite?.active ? suite.task_supervisor_model_id ? 'hybrid' : suite.model?.id ?? model : model;
  const profile=state.agent.configuration.models.find(entry=>entry.id===(selectedModel==='hybrid'?'qwen':selectedModel));
  const taskSupervisor=state.agent.configuration.models.find(entry=>entry.id==='luna');
  const efforts=profile?.reasoning_efforts ?? ['none'];
  const effectiveReasoning=suite?.active && suite.reasoning ? suite.reasoning : efforts.includes(reasoning) ? reasoning : efforts[0];
  useEffect(()=>{if(suite?.active) setExpanded(true);},[suite?.active]);
  if(!suite) return null;
  const locked=!connected || pending || suite.active || state.agent.active || state.busy || state.power?.on===false || !!state.recording?.active;
  async function start() {
    setPending(true);setError('');
    try {await request('regression/start',{run_id:state.run_id,episode_epoch:state.episode_epoch,
      model_id:model==='hybrid'?'qwen':model,reasoning:model==='hybrid'?'none':effectiveReasoning,
      task_supervisor_model_id:model==='hybrid'?'luna':null});
      document.getElementById('observe')?.scrollIntoView({behavior:'smooth',block:'start'});}
    catch(failure){setError(String(failure));}
    finally{setPending(false);}
  }
  async function download() {
    try {
      const report=await request('regression/report');
      const url=URL.createObjectURL(new Blob([JSON.stringify(report,null,2)],{type:'application/json'}));
      const anchor=document.createElement('a');anchor.href=url;anchor.download=`${suite?.suite_id}-${suite?.sequence_id ?? 'report'}.json`;
      anchor.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
    } catch(failure){setError(String(failure));}
  }
  const finished=suite.cases.filter(entry=>['passed','failed','invalid','cancelled','not_run'].includes(entry.status)).length;
  const missionMinutes=Math.ceil(suite.cases.reduce((total,entry)=>total+entry.budget_s,0)/60);
  const totalRequests=suite.cases.reduce((total,entry)=>total+(entry.max_model_requests??12),0);
  const totalTokens=suite.cases.reduce((total,entry)=>total+(entry.max_model_tokens??80000),0);
  return <details className="regression-panel" open={expanded} onToggle={event=>setExpanded(event.currentTarget.open)}>
    <summary><ListChecks size={17}/><strong>Regression baseline</strong><span>{suite.active ? `${(suite.current_index ?? 0)+1} / ${suite.cases.length} · ${suite.phase.replaceAll('_',' ')}` : suite.phase==='idle' ? `${suite.cases.length} cases · ${missionMinutes} min mission budget` : `${suite.phase} · ${finished} / ${suite.cases.length}`}</span></summary>
    <section aria-label="Observable regression baseline">
      <div className="regression-toolbar">
        <label>Supervisor<select aria-label="Baseline supervisor" value={selectedModel} disabled={locked} onChange={event=>setModel(event.target.value)}>
          <option value="luna">Luna</option><option value="qwen">Local Qwen</option>
          <option value="hybrid">Qwen + Luna task supervision</option></select></label>
        <label>Reasoning<select aria-label="Baseline reasoning" value={effectiveReasoning} disabled={locked} onChange={event=>setReasoning(event.target.value as Reasoning)}>
          {efforts.map(effort=><option key={effort} value={effort}>{effort}</option>)}</select></label>
        <button type="button" disabled={locked || !profile?.configured || (selectedModel==='hybrid' && !taskSupervisor?.configured)} onClick={()=>void start()} title={`Run all ${suite.cases.length} cases in fresh scenes; up to ${totalRequests} primary model requests and ${totalTokens.toLocaleString('en-US')} primary tokens total, plus local preparation and one Luna task review per case in hybrid mode. Existing scene will be replaced.`}>
          {pending ? <LoaderCircle size={16} className="loading-icon"/> : <Play size={16}/>}Start baseline</button>
        {suite.active && <button type="button" className="danger" aria-label="Stop baseline" disabled={!connected} onClick={()=>void request('stop',{}).catch(failure=>setError(String(failure)))}><CircleStop size={16}/>Stop baseline</button>}
        {suite.sequence_id && <button type="button" className="icon-button" aria-label="Download baseline report" title="Download baseline report" onClick={()=>void download()}><Download size={16}/></button>}
        <span className="tag">{suite.evidence ?? 'Real model'}</span>
      </div>
      <ol className="regression-cases">
        {suite.cases.map((entry,index)=>{
          const active=['loading','running'].includes(entry.status);
          const Icon=entry.status==='passed' ? CheckCircle2 : ['failed','invalid'].includes(entry.status) ? TriangleAlert : active ? LoaderCircle : Circle;
          return <li key={entry.id} data-status={entry.status} aria-current={suite.active && index===suite.current_index ? 'step' : undefined}>
            <img src={`/scenario-previews/${entry.challenge_id}.webp`} alt="" width={48} height={30}/>
            <span className="regression-case-title">{index+1}. {entry.title}<small>{entry.budget_s} s / {entry.max_model_requests??12} requests / {(entry.max_model_tokens??80000).toLocaleString('en-US')} tokens{entry.elapsed_s!==undefined ? ` / ${entry.elapsed_s.toFixed(1)} s elapsed` : ''}</small></span>
            <span className="regression-case-status"><Icon size={15} className={active?'loading-icon':''}/>{entry.status.replaceAll('_',' ')}</span>
            {(entry.error || entry.outcome?.message) && <small className="regression-case-detail">{entry.error || entry.outcome?.message}</small>}
            <RegressionProgress entry={entry}/>
            {entry.trajectory_url&&<RegressionPath entry={entry} active={active}/>} 
          </li>;
        })}
      </ol>
      {suite.reason && <div role="status">{suite.reason}</div>}
      {suite.directory && <output className="regression-directory" title={suite.directory}>{suite.directory}</output>}
      {error && <div className="error" role="alert">{error}</div>}
    </section>
  </details>;
}