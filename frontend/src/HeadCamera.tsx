import { useEffect, useRef, useState } from 'react';
import { RefreshCw } from 'lucide-react';
import type { CameraFrame } from './types';

export function HeadCamera({ frame }: { frame: CameraFrame }) {
  const latest = useRef(frame);
  latest.current = frame;
  const refresh = useRef<(force?: boolean) => void>(() => {});
  const objectUrls = useRef(new Set<string>());
  const [displayed, setDisplayed] = useState<{ frame: CameraFrame; url: string; width: number; height: number } | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    let active = true;
    let loading = false;
    let requestedUrl = '';
    let controller: AbortController | null = null;
    async function load(force = false) {
      if (!active || loading || (!force && requestedUrl === latest.current.url)) return;
      loading = true;
      const selected = latest.current;
      requestedUrl = selected.url;
      controller = new AbortController();
      let objectUrl: string | null = null;
      try {
        const response = await fetch(selected.url, { signal: controller.signal });
        if (!response.ok) throw new Error('Camera frame unavailable');
        const image = await response.blob();
        if (!active) return;
        objectUrl = URL.createObjectURL(image);
        objectUrls.current.add(objectUrl);
        const decoded = new Image();
        decoded.src = objectUrl;
        await decoded.decode();
        if (active) {
          setDisplayed({ frame: selected, url: objectUrl, width: decoded.naturalWidth, height: decoded.naturalHeight });
          setError(false);
        }
      } catch {
        if (objectUrl) {
          URL.revokeObjectURL(objectUrl);
          objectUrls.current.delete(objectUrl);
        }
        if (active) setError(true);
      } finally {
        loading = false;
        if (active && latest.current.url !== requestedUrl) void load();
      }
    }
    refresh.current = force => { void load(force); };
    void load();
    return () => {
      active = false;
      controller?.abort();
      refresh.current = () => {};
      for (const url of objectUrls.current) URL.revokeObjectURL(url);
      objectUrls.current.clear();
    };
  }, []);

  useEffect(() => { refresh.current(); }, [frame.url]);
  useEffect(() => () => {
    if (displayed) {
      URL.revokeObjectURL(displayed.url);
      objectUrls.current.delete(displayed.url);
    }
  }, [displayed]);

  return <>
    <div className="camera-frame">{displayed && <img alt="Authoritative robot head camera" src={displayed.url}
      data-frame={displayed.frame.frame_ref} data-simulated-time={displayed.frame.simulated_time_s} />}</div>
    <div className="camera-meta"><span>{displayed ? `${displayed.width} x ${displayed.height}` : '-'} / 65 deg</span><span>Live {displayed?.frame.seq ?? '-'} / {displayed?.frame.simulated_time_s.toFixed(2) ?? '0.00'} s</span></div>
    {error && <div className="exchange-error" role="alert"><span>Live camera unavailable</span><button className="icon-button" type="button" aria-label="Retry live camera" title="Retry live camera" onClick={() => refresh.current(true)}><RefreshCw size={16} /></button></div>}
  </>;
}