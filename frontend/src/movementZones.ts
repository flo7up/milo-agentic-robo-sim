import * as THREE from 'three';
import type { LiveState, MotionZones, ProximitySensors, SpatialTelemetry } from './types';

export class MovementZoneOverlay {
  readonly group = new THREE.Group();
  readonly meshes = new Map<string, THREE.Mesh<THREE.RingGeometry, THREE.MeshBasicMaterial>>();
  private footprint:THREE.LineLoop;
  private radius:THREE.LineLoop;
  private beams:THREE.LineSegments;
  private colors:Record<string,THREE.Color>;
  private previousZones?:MotionZones;
  private previousProximity?:ProximitySensors;
  private previousStatus='';

  constructor() {
    const style=getComputedStyle(document.documentElement);
    const color=(name:string)=>new THREE.Color(style.getPropertyValue(name).trim());
    this.colors={clear:color('--cp-success'),restricted:color('--cp-danger'),unknown:color('--cp-border-strong'),
      unavailable:color('--cp-warning'),outline:color('--cp-text'),ray:color('--cp-link')};
    this.footprint=new THREE.LineLoop(new THREE.BufferGeometry(),new THREE.LineBasicMaterial({color:this.colors.outline}));
    this.radius=new THREE.LineLoop(new THREE.BufferGeometry(),new THREE.LineBasicMaterial({color:this.colors.ray}));
    this.beams=new THREE.LineSegments(new THREE.BufferGeometry(),new THREE.LineBasicMaterial({vertexColors:true,transparent:true,opacity:.8}));
    this.group.add(this.footprint,this.radius,this.beams);
  }

  update(state:LiveState, telemetry:SpatialTelemetry | null | undefined, shown:boolean, connected:boolean) {
    const zones=telemetry?.motion_zones;
    const matched=!!zones && zones.run_id===state.run_id && zones.episode_epoch===state.episode_epoch;
    const base=state.snapshot.poses.find(pose=>pose.key===`${state.robot_body_id}:-1`);
    this.group.visible=shown && matched && !!base;
    if(!this.group.visible || !zones || !base) return shown ? 'Unavailable' : 'Hidden';
    const pose=state.observation.odometry_m_rad;
    const delta=pose[2]-zones.odometry_m_rad[2];
    const moved=Math.hypot(pose[0]-zones.odometry_m_rad[0],pose[1]-zones.odometry_m_rad[1])>.05
      || Math.abs(Math.atan2(Math.sin(delta),Math.cos(delta)))>.08;
    const elapsed=telemetry?.received_at_ms==null ? Infinity : Math.max(0,(performance.now()-telemetry.received_at_ms)/1000);
    const stale=zones.stale || zones.sensor_age_s==null || zones.sensor_age_s+elapsed>zones.maximum_sensor_age_s || elapsed>zones.valid_for_s;
    const status=!connected ? 'Disconnected' : state.power?.on===false || telemetry?.enabled===false ? 'Sensing off'
      : telemetry?.error || !zones.sectors.length ? 'Unavailable' : telemetry?.paused ? 'Paused' : moved ? 'Pose changed' : stale ? 'Stale' : 'Observed';
    const forward=new THREE.Vector3(1,0,0).applyQuaternion(new THREE.Quaternion(...base.quaternion));
    this.group.position.set(base.position[0],base.position[1],.025);
    this.group.rotation.z=Math.atan2(forward.y,forward.x);
    if(this.previousZones===zones && this.previousProximity===state.proximity && this.previousStatus===status) return status;
    this.previousZones=zones;
    this.previousProximity=state.proximity;
    this.previousStatus=status;
    const keep=new Set<string>();
    for(const sector of zones.sectors) {
      const key=`${sector.sector}:${sector.outer_m}`;
      keep.add(key);
      let mesh=this.meshes.get(key);
      if(!mesh) {
        mesh=new THREE.Mesh(new THREE.RingGeometry(sector.inner_m,sector.outer_m,8,1,sector.start_rad,sector.end_rad-sector.start_rad),
          new THREE.MeshBasicMaterial({transparent:true,opacity:.25,depthWrite:false,side:THREE.DoubleSide}));
        this.meshes.set(key,mesh);
        this.group.add(mesh);
      }
      const level=status==='Observed' ? sector.status : 'unavailable';
      mesh.material.color.copy(this.colors[level]);
      mesh.material.opacity=level==='clear' ? .22 : level==='restricted' ? .32 : .10;
      mesh.userData.hint=`${sector.inner_m.toFixed(2)}-${sector.outer_m.toFixed(2)} m center travel / ${status==='Observed' ? sector.reason : status}`;
    }
    for(const [key,mesh] of this.meshes) if(!keep.has(key)) {this.group.remove(mesh);mesh.geometry.dispose();mesh.material.dispose();this.meshes.delete(key);}
    const [left,bottom]=zones.footprint.lower_xy_m;
    const [right,top]=zones.footprint.upper_xy_m;
    this.footprint.geometry.setFromPoints([[left,bottom],[right,bottom],[right,top],[left,top]].map(point=>new THREE.Vector3(point[0],point[1],.01)));
    this.radius.geometry.setFromPoints(Array.from({length:64},(_,index)=>{
      const angle=index*Math.PI/32;
      const radius=zones.planning_radius_m+(zones.map_margin_m ?? 0);
      return new THREE.Vector3(Math.cos(angle)*radius,Math.sin(angle)*radius,.012);
    }));
    const vertices:number[]=[], colors:number[]=[];
    const segment=(start:number[],end:number[],color:THREE.Color)=>{vertices.push(...start,...end);colors.push(color.r,color.g,color.b,color.r,color.g,color.b);};
    for(const reading of state.proximity.distances) {
      const origin=reading.origin_base_m;
      if(!origin) continue;
      const cosine=Math.cos(reading.bearing_rad),sine=Math.sin(reading.bearing_rad);
      const distance=reading.distance_m ?? state.proximity.max_range_m;
      const color=status!=='Observed' || reading.status==='occluded' ? this.colors.unknown
        : reading.status==='hit' && zones.beam_checked_directions.includes(reading.direction) && distance<zones.beam_stop_distance_m ? this.colors.restricted : this.colors.ray;
      if(reading.status!=='occluded') segment([origin[0],origin[1],.02],[origin[0]+cosine*distance,origin[1]+sine*distance,.02],color);
      const horizontal=origin[0]+cosine*zones.beam_stop_distance_m,vertical=origin[1]+sine*zones.beam_stop_distance_m;
      if(zones.beam_checked_directions.includes(reading.direction)) segment([horizontal-sine*.04,vertical+cosine*.04,.025],[horizontal+sine*.04,vertical-cosine*.04,.025],this.colors.restricted);
    }
    this.beams.geometry.setAttribute('position',new THREE.Float32BufferAttribute(vertices,3));
    this.beams.geometry.setAttribute('color',new THREE.Float32BufferAttribute(colors,3));
    this.beams.geometry.computeBoundingSphere();
    return status;
  }

  dispose() {
    for(const mesh of this.meshes.values()) {mesh.geometry.dispose();mesh.material.dispose();}
    for(const line of [this.footprint,this.radius,this.beams]) {line.geometry.dispose();(line.material as THREE.Material).dispose();}
  }
}