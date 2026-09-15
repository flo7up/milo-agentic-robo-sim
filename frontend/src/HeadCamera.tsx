import { useEffect, useRef, useState } from 'react';
import { Camera, Maximize2, RefreshCw, X } from 'lucide-react';
import { RobotControlSlot } from './RobotControlSurface';
import type { CameraFrame } from './types';

export function HeadCamera({ frame, connected = true, compact = false }: { frame: CameraFrame; connected?: boolean; compact?: boolean }) {
  const latest = useRef(frame);
  latest.current = frame;
  const refresh = useRef<(force?: boolean) => void>(() => {});
  const objectUrls = useRef(new Set<string>());
  const [displayed, setDisplayed] = useState<{ frame: CameraFrame; url: string; width: number; height: number } | null>(null);
  const [error, setError] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const dialog = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    if (!connected) return;
    let active = true;
    let loading = false;
    let requestedUrl = '';
    let controller: AbortController | null = null;
    let retry: ReturnType<typeof setTimeout> | undefined;
    async function load(force = false) {
      if (!active || loading || (!force && requestedUrl === latest.current.url)) return;
      clearTimeout(retry);
      loading = true;
      const selected = latest.current;
      requestedUrl = selected.url;
      controller = new AbortController();
      const requestController = controller;
      let objectUrl: string | null = null;
      let timeout: ReturnType<typeof setTimeout> | undefined;
      let failed = false;
      try {
        const image = await Promise.race([
          (async () => {
            const response = await fetch(selected.url, { signal: requestController.signal });
            if (!response.ok) throw new Error('Camera frame unavailable');
            const blob = await response.blob();
            if (!active || requestController.signal.aborted) throw new Error('Camera request cancelled');
            objectUrl = URL.createObjectURL(blob);
            objectUrls.current.add(objectUrl);
            const decoded = new Image();
            await new Promise<void>((resolve, reject) => {
              decoded.onload = () => resolve();
              decoded.onerror = () => reject(new Error('Camera image unavailable'));
              decoded.src = objectUrl!;
            });
            return { url: objectUrl, width: decoded.naturalWidth, height: decoded.naturalHeight };
          })(),
          new Promise<never>((_, reject) => {
            timeout = setTimeout(() => { requestController.abort(); reject(new Error('Camera request timed out')); }, 4000);
          }),
        ]);
        if (active) {
          setDisplayed({ frame: selected, ...image });
          setError(false);
        }
      } catch {
        failed = true;
        if (objectUrl) {
          URL.revokeObjectURL(objectUrl);
          objectUrls.current.delete(objectUrl);
        }
        if (active) setError(true);
      } finally {
        clearTimeout(timeout);
        loading = false;
        if (active && latest.current.url !== requestedUrl) void load();
        else if (active && failed) retry = setTimeout(() => { void load(true); }, 1000);
      }
    }
    refresh.current = force => { void load(force); };
    void load();
    return () => {
      active = false;
      controller?.abort();
      clearTimeout(retry);
      refresh.current = () => {};
    };
  }, [connected]);

  useEffect(() => () => {
    for (const url of objectUrls.current) URL.revokeObjectURL(url);
    objectUrls.current.clear();
  }, []);

  useEffect(() => {
    if (expanded) dialog.current?.showModal();
    else dialog.current?.close();
  }, [expanded]);

  useEffect(() => { refresh.current(); }, [frame.url]);
  useEffect(() => () => {
    if (displayed) {
      URL.revokeObjectURL(displayed.url);
      objectUrls.current.delete(displayed.url);
    }
  }, [displayed]);

  return <>
    <div className="camera-frame">{displayed && <img alt="Authoritative robot head camera" src={displayed.url}
      data-frame={displayed.frame.frame_ref} data-simulated-time={displayed.frame.simulated_time_s} />}
      {!displayed && <div className="camera-empty" role="status" aria-label={!connected ? 'Camera disconnected' : error ? 'Reconnecting camera' : 'Loading robot camera'}>
        {compact ? <Camera size={20} /> : !connected ? 'Camera disconnected' : error ? 'Reconnecting camera...' : 'Loading robot camera...'}</div>}
      <button className="icon-button camera-expand" type="button" aria-label="Expand robot camera" title="Expand robot camera"
        onClick={() => setExpanded(true)}><Maximize2 size={19} /></button>
    </div>
    <div className="camera-meta"><span>{displayed ? `${displayed.width} x ${displayed.height}` : '-'} / 65 deg</span><span>{connected && !error ? 'Live' : 'Last received'} {displayed?.frame.seq ?? '-'} / {displayed?.frame.simulated_time_s.toFixed(2) ?? '0.00'} s</span></div>
    {error && connected && <div className="exchange-error" role="alert"><span>Camera reconnecting</span><button className="icon-button" type="button" aria-label="Retry live camera" title="Retry live camera" onClick={() => refresh.current(true)}><RefreshCw size={16} /></button></div>}
    <dialog ref={dialog} className="robot-camera-dialog" aria-label="Robot camera" onClose={() => setExpanded(false)}>
      <div className="robot-camera-toolbar"><h3>Robot camera</h3><span role="status">{!connected ? 'Disconnected / last frame' : error ? 'Camera reconnecting / last frame' : 'Live head view'}</span>
        <button className="icon-button" type="button" aria-label="Close robot camera" title="Close robot camera" onClick={() => setExpanded(false)}><X size={19} /></button>
      </div>
      <RobotControlSlot active={expanded} />
      <div className="robot-camera-expanded-image">{displayed ? <img alt="Expanded robot head camera" src={displayed.url}
        data-frame={displayed.frame.frame_ref} data-simulated-time={displayed.frame.simulated_time_s} /> : <p role="status">Waiting for the robot camera...</p>}</div>
      <div className="camera-meta"><span>{displayed ? `${displayed.width} x ${displayed.height}` : '-'} / 65 deg</span><span>Frame {displayed?.frame.seq ?? '-'} / {displayed?.frame.simulated_time_s.toFixed(2) ?? '0.00'} s</span></div>
    </dialog>
  </>;
}