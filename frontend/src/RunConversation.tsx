import { useEffect } from 'react';
import { MessageSquare } from 'lucide-react';
import type { LiveState, ModelCallSelection } from './types';
import { currentModelCalls } from './modelCallMarkers';

export function RunConversation({state,selection,onSelect,visible=true}:{state:LiveState;selection?:ModelCallSelection;onSelect?:(id:string)=>void;visible?:boolean}) {
  const calls=currentModelCalls(state);
  const messages=state.agent.run_messages ?? [];
  const entries=[
    ...calls.map(call=>({id:call.id,timestamp:call.timestamp,call,message:null})),
    ...messages.filter(message=>!message.model_call_id).map(message=>({id:message.id,timestamp:message.timestamp ?? 0,call:null,message})),
  ].sort((a,b)=>a.timestamp-b.timestamp);
  useEffect(()=>{
    if(!visible || selection?.from!=='path') return;
    const frame=requestAnimationFrame(()=>{
      const card=document.getElementById(`model-call-${selection.id}`);
      card?.scrollIntoView({block:'nearest',behavior:'smooth'});
      card?.focus({preventScroll:true});
    });
    return ()=>cancelAnimationFrame(frame);
  },[selection,visible]);
  return <section className="chat-section" aria-label="Run chat">
    <div className="panel-header"><h3><MessageSquare size={17}/>Run chat</h3>{calls.length>0 && <span className="exchange-meta">Numbers match the blue path</span>}</div>
    <div className="chat-transcript" role="log" aria-label="Run conversation" tabIndex={0}>
      {!entries.length && !state.agent.goal && <p className="empty">No messages yet.</p>}
      {state.agent.goal && !messages.some(message=>message.role==='user' && message.text===state.agent.goal) &&
        <article className="chat-message chat-user"><strong>You / mission instruction</strong><p>{state.agent.goal}</p></article>}
      {entries.map(({id,call,message})=>call ? <article key={id} id={`model-call-${id}`} tabIndex={-1}
        className={`chat-message chat-assistant${selection?.id===id?' selected-call':''}`} data-model-call={call.number}>
        <div className="chat-call-heading"><button type="button" className="model-call-badge" aria-label={`Show model call ${call.number} on path`}
          aria-pressed={selection?.id===id} disabled={!call.position_world_m} onClick={()=>onSelect?.(id)}>{call.number}</button>
          <strong>{call.model} / {call.kind==='task_supervision'?'Task review':call.status}</strong>
          <span className="exchange-meta">{call.simulated_time_s?.toFixed(1) ?? '—'} s</span></div>
        <p>{[...new Set(messages.filter(message=>message.model_call_id===id && message.text.trim()).map(message=>message.text))].join('\n') || call.summary}</p>
      </article> : message && <article key={id} className={`chat-message chat-${message.role}`}>
        <strong>{message.role==='user'?'You':message.source==='model'?state.agent.configuration.models.find(model=>model.id===state.agent.model_id)?.label ?? 'Model':'Controller'} / {message.status}</strong>
        <p>{message.text}</p></article>)}
    </div>
  </section>;
}
