import {useEffect, useState} from 'react';
import {Brain, GitFork, Plus, Save} from 'lucide-react';

type MemoryEntity = {entity_id:string;kind:string;label?:string;status?:string;requires_revalidation:boolean;
  last_observed_unix_s:number;confidence?:number;success_count?:number;last_distance_m?:number;
  from_entity_id?:string;to_entity_id?:string;evidence_id:string;image_evidence_id?:string};
type MemoryState = {enabled:boolean;scope:{run_id:string;episode_epoch:number;context_id:string;profile_id:string;map_id:string;
  environment_id:string;environment_revision:string};profile:{name:string};profiles:{profile_id:string;name:string}[];
  snapshots:{snapshot_id:string;name:string}[];rooms:MemoryEntity[];objects:MemoryEntity[];connections:MemoryEntity[];
  search_history:{observation_id:string;target:string;result:string;inspection_scope:string;visibility_limits:string}[];
  counts:{rooms:number;objects:number;connections:number};error:string|null;pending_writes:number};

export function MemoryControls({active, locked, runId, epoch, request}: {
  active:boolean;locked:boolean;runId:string;epoch:number;request:(path:string,body?:unknown)=>Promise<unknown>;
}) {
  const [memory,setMemory]=useState<MemoryState|null>(null);
  const [pending,setPending]=useState(false);
  const [error,setError]=useState('');
  const [name,setName]=useState('New knowledge');
  const [snapshot,setSnapshot]=useState('');
  const [limit,setLimit]=useState(3);
  useEffect(()=>{
    if(!active || pending) return;
    const controller=new AbortController();
    let timer:ReturnType<typeof setTimeout>;
    async function refresh() {
      try {
        const response=await fetch('/api/memory',{signal:controller.signal});
        if(!response.ok) throw new Error(response.status===404 ? 'Memory requires a backend update.' : 'Memory is unavailable.');
        const value=await response.json();
        if(!controller.signal.aborted && value.scope?.run_id===runId && value.scope?.episode_epoch===epoch) setMemory(value);
      } catch(failure) {
        if(!controller.signal.aborted) setError(failure instanceof Error ? failure.message : String(failure));
      } finally {
        if(!controller.signal.aborted) timer=setTimeout(refresh,2000);
      }
    }
    void refresh();
    return ()=>{controller.abort();clearTimeout(timer);};
  },[active,pending,runId,epoch]);
  async function act(action:string,values:Record<string,unknown>={}) {
    if(!memory) return;
    setPending(true);setError('');
    try {
      const result=await request('memory',{run_id:runId,episode_epoch:epoch,context_id:memory.scope.context_id,action,name,...values}) as MemoryState;
      setMemory(result);setSnapshot('');setLimit(3);
    } catch(failure) {setError(failure instanceof Error ? failure.message : String(failure));}
    finally {setPending(false);}
  }
  const disabled=locked || pending || !memory?.enabled;
  const entities=[...(memory?.rooms??[]),...(memory?.objects??[])];
  const roomName=(identity?:string)=>memory?.rooms.find(room=>room.entity_id===identity)?.label ?? identity?.slice(0,8) ?? 'Unknown region';
  return <section className="memory-settings" aria-label="Spatial memory">
    <div className="panel-header"><h4><Brain size={16}/>Spatial memory</h4>
      <span role="status">{pending ? 'Saving knowledge' : memory?.pending_writes ? 'Pending observations' : memory?.profile.name ?? 'Loading'}</span></div>
    {(error || memory?.error) && <p className="error" role="alert">{error || memory?.error}</p>}
    {memory && <>
      <dl className="memory-scope"><div><dt>Environment instance</dt><dd title={memory.scope.environment_id}>{memory.scope.environment_id}</dd></div>
        <div><dt>Layout revision</dt><dd title={memory.scope.environment_revision}>{memory.scope.environment_revision.slice(0,12)}</dd></div></dl>
      <label>Knowledge profile<select aria-label="Knowledge profile" value={memory.scope.profile_id} disabled={disabled}
        onChange={event=>void act('select_profile',{profile_id:event.target.value})}>
        {memory.profiles.map(profile=><option value={profile.profile_id} key={profile.profile_id}>{profile.name}</option>)}</select></label>
      <label>Profile or checkpoint name<input aria-label="Knowledge name" value={name} maxLength={80} disabled={disabled}
        onChange={event=>setName(event.target.value)}/></label>
      <div className="memory-actions">
        <button type="button" disabled={disabled || !name.trim()} title="Start an empty knowledge profile; retain the current profile and scene"
          onClick={()=>void act('fresh_profile')}><Plus size={16}/>Start fresh knowledge</button>
        <button type="button" disabled={disabled || !name.trim()} title="Save an immutable map and evidence checkpoint"
          onClick={()=>void act('checkpoint')}><Save size={16}/>Save checkpoint</button>
      </div>
      {!!memory.snapshots.length && <div className="memory-checkpoint">
        <label>Checkpoint<select aria-label="Knowledge checkpoint" value={snapshot || memory.snapshots[0].snapshot_id} disabled={disabled}
          onChange={event=>setSnapshot(event.target.value)}>{memory.snapshots.map(item=><option key={item.snapshot_id} value={item.snapshot_id}>{item.name}</option>)}</select></label>
        <button type="button" className="icon-button" title="Create an independent writable profile from this checkpoint" aria-label="Fork checkpoint"
          disabled={disabled || !name.trim()} onClick={()=>void act('fork_checkpoint',{snapshot_id:snapshot || memory.snapshots[0].snapshot_id})}><GitFork size={18}/></button>
      </div>}
      <p className="exchange-meta">{memory.counts.rooms??0} rooms / {memory.counts.objects??0} objects / {memory.counts.connections??0} pathways</p>
      <details><summary>Remembered rooms and objects</summary>
        <ul className="memory-records">{entities.slice(0,limit).map(item=><li key={item.entity_id}>
          <strong>{item.label}</strong><span>{item.kind} / {item.requires_revalidation ? 'Requires revalidation' : item.status?.replaceAll('_',' ')}</span>
          <small>{new Date(item.last_observed_unix_s*1000).toLocaleString()} / reported confidence {Math.round((item.confidence??0)*100)}%</small>
          {item.image_evidence_id && <a target="_blank" rel="noreferrer" href={`/api/home/${memory.scope.map_id}/${item.kind==='room'?'rooms':'objects'}/${item.evidence_id}/image.png`}>Observation image</a>}
        </li>)}</ul>
        {entities.length>limit && <div className="memory-actions"><button type="button" onClick={()=>setLimit(value=>value+5)}>Show more</button><button type="button" onClick={()=>setLimit(entities.length)}>View all</button></div>}
      </details>
      <details><summary>Successful pathways</summary><ul className="memory-records">{memory.connections.map(item=><li key={item.entity_id}>
        <strong>{roomName(item.from_entity_id)} to {roomName(item.to_entity_id)}</strong>
        <span>{item.success_count} observed transitions / last {(item.last_distance_m??0).toFixed(2)} m</span>
        <small>{item.requires_revalidation ? 'Requires revalidation' : 'Historical travel; current clearance required'}</small>
      </li>)}</ul></details>
      <details><summary>Recent search inspections</summary><ul className="memory-records">{memory.search_history.map(item=><li key={item.observation_id}>
        <strong>{item.target}</strong><span>{item.result.replaceAll('_',' ')} / {item.inspection_scope}</span><small>{item.visibility_limits || 'Visibility scope unverified'}</small>
      </li>)}</ul></details>
    </>}
  </section>;
}