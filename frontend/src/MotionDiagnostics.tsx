import { useEffect, useState } from 'react';
import { Activity, ChevronRight, Copy } from 'lucide-react';
import type { LiveState, NavigationDiagnostics, NavigationState, SpatialTelemetry, TaskDiagnostics } from './types';

function seconds(value: number | null | undefined) {
  return value == null || !Number.isFinite(value) ? 'Not recorded' : `${value.toFixed(3)} s`;
}

export function MotionDiagnostics({navigation, connected, state, spatial, recorded}: {
  navigation?:NavigationState | null;connected:boolean;state?:LiveState;spatial?:SpatialTelemetry | null;recorded?:NavigationDiagnostics | null;
}) {
  const diagnostics = recorded ?? navigation?.diagnostics;
  const [now, setNow] = useState(() => performance.now());
  const [copyMessage, setCopyMessage] = useState('');
  useEffect(() => {
    if (!state || !connected) return;
    const timer = setInterval(() => setNow(performance.now()), 1000);
    return () => clearInterval(timer);
  }, [!!state, connected]);
  const receiptAge = spatial?.received_at_ms != null ? Math.max(0, (now - spatial.received_at_ms) / 1000) : null;
  const sensorStatus = !connected ? 'Disconnected / last received' : state?.power?.on === false ? 'Power off'
    : spatial?.error ? 'Unavailable / last received' : spatial?.enabled === false ? 'Sensing off'
    : receiptAge != null && receiptAge > 5 ? 'Telemetry delayed' : spatial?.paused ? 'Paused'
    : !spatial?.map ? 'No depth sample' : spatial.map.stale ? 'Stale at receipt' : 'Fresh at receipt';
  const objective = state?.agent.mission?.objective;
  const stop = diagnostics?.stop;
  const renewal = diagnostics?.last_renewal;
  const acceptedAt = diagnostics?.renewed_at_s ?? diagnostics?.issued_at_s;
  const lifetime = diagnostics?.expires_at_s != null && acceptedAt != null ? diagnostics.expires_at_s - acceptedAt : null;
  const stopDelay = stop && diagnostics?.expires_at_s != null ? stop.at_s - diagnostics.expires_at_s : null;
  const tickToStop = stop && diagnostics?.last_controller_tick_at_s != null ? stop.at_s - diagnostics.last_controller_tick_at_s : null;
  async function copy() {
    try {
      await navigator.clipboard.writeText(JSON.stringify({format:'milo-motion-diagnostics-v1',exported_at:new Date().toISOString(),
        run_id:state?.run_id,session_id:state?.agent.session_id,connected,diagnostics,navigation,mission:state?.agent.mission,
        controller_outcome:state?.agent.outcome,spatial,proximity:state?.proximity,camera:state?.camera},null,2));
      setCopyMessage('Diagnostics copied');
    } catch { setCopyMessage('Copy failed. Allow clipboard access and retry.'); }
  }
  return <section className="motion-diagnostics" aria-label={recorded ? 'Recorded motion diagnostics' : 'Motion diagnostics'}>
    <div className="panel-header"><h3><Activity size={16} />{recorded ? 'Recorded motion' : 'Motion diagnostics'}</h3>
      <button type="button" className="icon-button" aria-label="Copy motion diagnostics" title="Copy recorded timing, sensor and authorization evidence" onClick={() => void copy()}><Copy size={15}/></button></div>
    <span className="tag">{recorded ? 'Recorded at this event' : !connected ? 'Disconnected / last received' : stop ? 'Last stopped buffer' : 'Reported buffer'}</span>
    <p className={stop ? 'motion-stop-reason' : 'exchange-text'}>{stop?.reason || navigation?.reason || 'No motion authorization reported'}</p>
    {state && <dl className="diagnostic-readings">
      <div><dt>Depth / map</dt><dd>{sensorStatus}</dd></div>
      <div><dt>Motion buffer</dt><dd>{navigation?.status ?? 'Not reported'} / {seconds(navigation?.expires_in_s)} remaining</dd></div>
      <div><dt>Exploration objective</dt><dd>{objective ? `${objective.action} / ${objective.status}` : 'None reported'}</dd></div>
      <div><dt>Renewal owner</dt><dd>{diagnostics?.renewal_owner ?? 'Not recorded'}</dd></div>
      <div><dt>Last renewal result</dt><dd>{renewal?.result ?? 'Not recorded'}</dd></div>
      <div><dt>Sample age at command</dt><dd>{seconds(diagnostics?.sensor_at_last_command?.age_s)}</dd></div>
    </dl>}
    {copyMessage && <p role="status" className="exchange-meta">{copyMessage}</p>}
    <>
      <details className="compact-disclosure"><summary><ChevronRight className="disclosure-chevron" size={15} /><strong>Sensor and timing details</strong></summary>
      {state && <dl className="policy-facts">
        <div><dt>Depth sample age at receipt</dt><dd>{seconds(spatial?.map?.age_s)} / server monotonic</dd></div>
        <div><dt>Telemetry since receipt</dt><dd>{connected ? seconds(receiptAge) : 'Disconnected'} / browser monotonic</dd></div>
        <div><dt>Depth frame / simulation</dt><dd>{spatial?.frame ? `${spatial.frame.sequence} / ${seconds(spatial.frame.simulated_time_s)}` : 'Not received'}</dd></div>
        <div><dt>Head frame / simulation</dt><dd>{state.camera.seq} / {seconds(state.camera.simulated_time_s)}</dd></div>
        <div><dt>Observed floor / obstacle cells</dt><dd>{spatial?.map ? `${spatial.map.observed_floor_cells} / ${spatial.map.obstacle_cells}` : 'Not received'}</dd></div>
        <div><dt>Proximity / simulation</dt><dd>{seconds(state.proximity.simulated_time_s)} / {state.proximity.collisions.length} contacts</dd></div>
        {state.proximity.distances.map(reading => <div key={reading.direction}><dt>{reading.direction.replaceAll('_',' ')}</dt>
          <dd>{reading.status === 'occluded' ? 'Occluded' : reading.distance_m == null ? `No hit within ${state.proximity.max_range_m} m` : `${reading.distance_m.toFixed(3)} m`}</dd></div>)}
      </dl>}
      {spatial?.error && <p className="motion-stop-reason">{spatial.error}</p>}
      {diagnostics ? <dl className="policy-facts">
        <div><dt>Stop owner</dt><dd>{stop?.initiator ?? 'No stop recorded'}</dd></div>
        <div><dt>Last renewal</dt><dd>{renewal ? `${renewal.result} / ${renewal.rejection_reason ?? 'No rejection recorded'}` : 'No renewal recorded'}</dd></div>
        <div><dt>Authorized window</dt><dd>{seconds(lifetime)} / {diagnostics.clock}</dd></div>
        <div><dt>Sensor age at command</dt><dd>{seconds(diagnostics.sensor_at_last_command?.age_s)} / {diagnostics.sensor_at_last_command?.kind ?? 'No accepted sample'}</dd></div>
        <div><dt>Command sensor clock</dt><dd>{diagnostics.sensor_at_last_command?.clock ?? 'Not recorded'}</dd></div>
        <div><dt>Last tick to stop</dt><dd>{seconds(tickToStop)}</dd></div>
        <div><dt>Stop relative to expiry</dt><dd>{stopDelay == null ? 'Not recorded' : `${seconds(Math.abs(stopDelay))} ${stopDelay >= 0 ? 'after' : 'before'}`}</dd></div>
      </dl> : <p className="exchange-meta">Authorization timing was not recorded.</p>}
      </details>
      {diagnostics && <details className="compact-disclosure"><summary><ChevronRight className="disclosure-chevron" size={15} /><strong>Authorization timeline</strong></summary>
        <dl className="policy-facts">
          <div><dt>Buffer ID</dt><dd>{diagnostics.authorization_id ?? 'Not issued'}</dd></div>
          <div><dt>Task ID</dt><dd>{diagnostics.task_id ?? 'Not linked'}</dd></div>
          <div><dt>Objective ID</dt><dd>{diagnostics.objective_id ?? 'Not linked'}</dd></div>
          <div><dt>Issuer</dt><dd>{diagnostics.issuer}</dd></div>
          <div><dt>Renewal owner</dt><dd>{diagnostics.renewal_owner}</dd></div>
          <div><dt>Issued / {diagnostics.clock}</dt><dd>{seconds(diagnostics.issued_at_s)}</dd></div>
          <div><dt>Last accepted renewal / {diagnostics.clock}</dt><dd>{seconds(diagnostics.renewed_at_s)}</dd></div>
          <div><dt>Expires / {diagnostics.clock}</dt><dd>{seconds(diagnostics.expires_at_s)}</dd></div>
          <div><dt>Last controller tick / {diagnostics.clock}</dt><dd>{seconds(diagnostics.last_controller_tick_at_s)}</dd></div>
          <div><dt>Stop / {diagnostics.clock}</dt><dd>{seconds(stop?.at_s)}</dd></div>
          <div><dt>Recorded max tick gap</dt><dd>{seconds(diagnostics.maximum_recent_tick_gap_s)} / {diagnostics.tick_window_s} s window</dd></div>
        </dl>
        <ol className="authorization-events">{diagnostics.recent_events.map((event,index) => <li key={index}>
          <span>{seconds(event.at_s)} / {diagnostics.clock}</span><strong>{event.event ?? 'renewal'} / {event.result ?? event.initiator}</strong>
          {(event.rejection_reason || event.reason) && <span>{event.rejection_reason || event.reason}</span>}
        </li>)}</ol>
        {objective?.authorization && <>
          <h4>Exploration authorization</h4>
          <dl className="policy-facts">
            <div><dt>Objective status</dt><dd>{objective.status} / {seconds(objective.remaining_s)} remaining</dd></div>
            <div><dt>Scope</dt><dd>{objective.authorization.scope}</dd></div>
            <div><dt>Issuer / renewal owner</dt><dd>{objective.authorization.issuer} / {objective.authorization.renewal_owner}</dd></div>
            <div><dt>Issued / {objective.authorization.clock}</dt><dd>{seconds(objective.authorization.issued_at_s)}</dd></div>
            <div><dt>Expires / {objective.authorization.clock}</dt><dd>{seconds(objective.authorization.expires_at_s)}</dd></div>
            <div><dt>Objective renewal</dt><dd>{objective.authorization.last_renewal?.result ?? 'No renewal recorded'}</dd></div>
            <div><dt>Objective stop</dt><dd>{objective.authorization.stop ? `${objective.authorization.stop.initiator}: ${objective.authorization.stop.reason}` : 'No stop recorded'}</dd></div>
          </dl>
        </>}
      </details>}
    </>
  </section>;
}

export function RouteFailureDetails({task}: {task:TaskDiagnostics | null | undefined}) {
  if (!task?.route_failures?.length) return null;
  return <section className="route-failures" aria-label="Recorded route failures">
    <p className="motion-stop-reason">{task.reason}</p>
    <dl className="policy-facts"><div><dt>Frontier</dt><dd>{task.frontier_id ?? 'Not recorded'}</dd></div>
      <div><dt>Segments / retries</dt><dd>{task.segments ?? '-'} / {task.retries ?? '-'}</dd></div></dl>
    {task.route_failures.map((failure,index) => <details className="compact-disclosure" key={index}>
      <summary><ChevronRight className="disclosure-chevron" size={15} /><strong>Segment {failure.segment}</strong><span className="disclosure-preview">{failure.reason}</span></summary>
      <p className="exchange-meta">Frontier {failure.frontier_id ?? '-'} / {seconds(failure.elapsed_s)} elapsed / {failure.clock ?? 'clock not recorded'}</p>
      {failure.motion_diagnostics ? <MotionDiagnostics connected={false} recorded={failure.motion_diagnostics}/> : <p>Timing not recorded</p>}
    </details>)}
  </section>;
}