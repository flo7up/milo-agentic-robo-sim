import * as THREE from 'three';
import type { LiveState, ModelCall } from './types';

export function currentModelCalls(state:LiveState) {
  return (state.agent.model_calls ?? []).filter(call=>call.run_id===state.run_id && call.episode_epoch===state.episode_epoch
    && call.session_id===state.agent.session_id);
}

/** Operator overlay projected from the same world frame as the travelled path. */
export class ModelCallMarkers {
  readonly element=document.createElement('div');
  private svg=document.createElementNS('http://www.w3.org/2000/svg','svg');
  private nodes=new Map<string,{button:HTMLButtonElement;line:SVGLineElement;dot:SVGCircleElement}>();
  private signature='';
  private lastCalls:ModelCall[] | null=null;

  constructor(private select:(id:string)=>void) {
    this.element.className='model-call-markers';
    this.element.setAttribute('role','group');
    this.element.setAttribute('aria-label','Model call locations');
    this.svg.setAttribute('aria-hidden','true');
    this.element.append(this.svg);
  }

  update(calls:ModelCall[], selected:string | undefined, visible:boolean, camera:THREE.Camera, world:THREE.Object3D, width:number, height:number) {
    this.element.hidden=!visible;
    if(!visible) return;
    const signature=JSON.stringify([selected,camera.matrixWorld.elements,camera.projectionMatrix.elements,width,height]);
    if(calls===this.lastCalls && signature===this.signature) return;
    this.lastCalls=calls;
    this.signature=signature;
    const located=calls.filter(call=>call.position_world_m?.every(Number.isFinite));
    const ids=new Set(located.map(call=>call.id));
    for(const [id,node] of this.nodes) if(!ids.has(id)) {
      node.button.remove();node.line.remove();node.dot.remove();this.nodes.delete(id);
    }
    const occupied:{x:number;y:number}[]=[];
    // Keep the selected marker readable even in a dense group of stationary calls.
    for(const call of [...located].sort((a,b)=>Number(b.id===selected)-Number(a.id===selected))) {
      let node=this.nodes.get(call.id);
      if(!node) {
        const button=document.createElement('button');
        button.type='button';button.className='model-call-marker';
        button.dataset.callId=call.id;
        button.addEventListener('click',()=>this.select(call.id));
        const line=document.createElementNS(this.svg.namespaceURI,'line') as SVGLineElement;
        const dot=document.createElementNS(this.svg.namespaceURI,'circle') as SVGCircleElement;
        dot.setAttribute('r','3.5');
        this.svg.append(line,dot);this.element.append(button);
        node={button,line,dot};this.nodes.set(call.id,node);
      }
      const point=world.localToWorld(new THREE.Vector3(...call.position_world_m!,.04)).project(camera);
      const x=(point.x+1)*width/2, y=(1-point.y)*height/2;
      const shown=point.z>=-1 && point.z<=1 && x>=0 && x<=width && y>=0 && y<=height;
      node.button.hidden=!shown;
      node.line.style.display=node.dot.style.display=shown?'':'none';
      if(!shown) continue;
      let label={x:Math.max(15,Math.min(width-15,x)),y:Math.max(15,Math.min(height-15,y-19))};
      for(let index=0;index<240;index++) {
        const ring=Math.floor(index/12), angle=(index%12)*Math.PI/6-Math.PI/2, radius=19+ring*28;
        const candidate={x:x+Math.cos(angle)*radius,y:y+Math.sin(angle)*radius};
        if(candidate.x<15 || candidate.x>width-15 || candidate.y<15 || candidate.y>height-15) continue;
        if(occupied.every(other=>Math.abs(other.x-candidate.x)>29 || Math.abs(other.y-candidate.y)>27)) {label=candidate;break;}
      }
      occupied.push(label);
      node.button.textContent=String(call.number);
      node.button.setAttribute('aria-label',`Model call ${call.number}: ${call.model}, ${call.simulated_time_s?.toFixed(1) ?? 'unknown'} seconds`);
      node.button.setAttribute('aria-pressed',String(call.id===selected));
      node.button.title=`#${call.number} · ${call.model} · ${call.status}\n${call.summary}`;
      node.button.style.transform=`translate(${label.x}px,${label.y}px) translate(-50%,-50%)`;
      node.dot.setAttribute('cx',String(x));node.dot.setAttribute('cy',String(y));
      for(const [key,value] of Object.entries({x1:x,y1:y,x2:label.x,y2:label.y})) node.line.setAttribute(key,String(value));
      node.dot.classList.toggle('selected',call.id===selected);
      node.line.classList.toggle('selected',call.id===selected);
    }
  }

  dispose() {this.element.remove();this.nodes.clear();}
}
