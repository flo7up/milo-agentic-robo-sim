import { useEffect, useRef, useState } from 'react';
import { Check, Mic, PhoneOff, Send, Settings2, Volume2 } from 'lucide-react';
import type { LiveState } from './types';
import { VoiceAudio } from './voiceAudio';

export function VoiceControl({ state, connected, request, interval, maxTurns }: {
  state: LiveState; connected: boolean; request: (path: string, body?: unknown) => Promise<unknown>; interval: number; maxTurns: number;
}) {
  const config = state.realtime;
  const [endpoint, setEndpoint] = useState(config.endpoint);
  const [deployment, setDeployment] = useState(config.deployment);
  const [voice, setVoice] = useState(config.voice);
  const [status, setStatus] = useState('off');
  const [playing, setPlaying] = useState(false);
  const [duration, setDuration] = useState(0);
  const [error, setError] = useState('');
  const [userText, setUserText] = useState('');
  const [robotText, setRobotText] = useState('');
  const socket = useRef<WebSocket | null>(null);
  const audio = useRef<VoiceAudio | null>(null);
  const recording = useRef(false);

  useEffect(() => { setEndpoint(config.endpoint); setDeployment(config.deployment); setVoice(config.voice); }, [config.endpoint, config.deployment, config.voice]);

  function close() {
    const current = socket.current;
    socket.current = null;
    recording.current = false;
    audio.current?.close();
    audio.current = null;
    if (current?.readyState === WebSocket.OPEN) current.send(JSON.stringify({ type: 'close' }));
    current?.close();
    setStatus('off');
  }

  useEffect(() => () => close(), [state.run_id]);
  useEffect(() => { if (!connected) close(); }, [connected]);

  async function finishRecording() {
    if (!recording.current) return;
    recording.current = false;
    setStatus('responding');
    try {
      await audio.current?.stopCapture();
      if (socket.current?.readyState === WebSocket.OPEN) socket.current.send(JSON.stringify({ type: 'commit' }));
    } catch (failure) { setError(String(failure)); close(); }
  }

  async function capture() {
    const current = socket.current;
    if (!current || current.readyState !== WebSocket.OPEN || !audio.current) return;
    setStatus('requesting');
    setDuration(0);
    setRobotText('');
    setUserText('');
    current.send(JSON.stringify({ type: 'listen' }));
    try {
      await audio.current.startCapture(packet => {
        if (socket.current === current && current.readyState === WebSocket.OPEN) {
          if (current.bufferedAmount > 256000) { setError('Microphone connection is too slow.'); close(); return; }
          current.send(packet);
        }
      }, samples => {
        setDuration(samples / 24000);
        if (samples >= 720000) void finishRecording();
      });
      if (socket.current !== current) return;
      recording.current = true;
      setStatus('recording');
    } catch (failure) {
      setError(failure instanceof DOMException && failure.name === 'NotAllowedError' ? 'Microphone permission was denied.' : String(failure));
      close();
    }
  }

  async function microphone() {
    if (recording.current) { await finishRecording(); return; }
    setError('');
    if (socket.current) { await capture(); return; }
    try {
      setStatus('connecting');
      const sessionAudio = new VoiceAudio(setPlaying);
      audio.current = sessionAudio;
      await sessionAudio.resume();
      if (audio.current !== sessionAudio) return;
      const current = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/api/voice`);
      current.binaryType = 'arraybuffer';
      socket.current = current;
      let firstReady = true;
      current.onopen = () => {
        if (socket.current !== current) { current.close(); return; }
        current.send(JSON.stringify({ type: 'start', run_id: state.run_id,
          episode_epoch: state.episode_epoch, feedback_interval_s: interval, max_turns: maxTurns }));
      };
      current.onmessage = event => {
        if (socket.current !== current) return;
        if (event.data instanceof ArrayBuffer) {
          try { audio.current?.play(event.data); }
          catch (failure) { setError(String(failure)); close(); }
          return;
        }
        const message = JSON.parse(event.data);
        if (message.type === 'ready') {
          setStatus('ready');
          if (firstReady) { firstReady = false; void capture(); }
        } else if (message.type === 'waking') {
          setStatus('responding');
          setRobotText('');
        } else if (message.type === 'transcript_delta') {
          setRobotText(previous => (previous + message.text).slice(-16000));
        } else if (message.type === 'transcript') {
          if (message.speaker === 'user') setUserText(message.text);
          else setRobotText(message.text);
        } else if (message.type === 'notice') setError(message.message);
        else if (message.type === 'ended') { if (message.error) setError(message.error); close(); }
      };
      current.onerror = () => { if (socket.current === current) { setError('Voice connection failed.'); close(); } };
      current.onclose = () => { if (socket.current === current) close(); };
    } catch (failure) { setError(String(failure)); close(); }
  }

  const busy = ['connecting', 'requesting', 'responding'].includes(status) || playing;
  const label = status === 'recording' ? 'Send spoken command' : 'Start microphone';
  const configurationStatus = !config.endpoint ? 'Realtime resource endpoint missing'
    : !config.deployment.trim() ? 'GPT Realtime 2 deployment name missing' : 'Microphone off';
  return <section className="voice-section" aria-label="Voice control">
    <div className="voice-row">
      <button className={`voice-mic ${status === 'recording' ? 'recording' : ''}`} type="button" title={label} aria-label={label}
        aria-pressed={status === 'recording'} disabled={!connected || !config.configured || busy || state.busy || interval < .25 || interval > 30 || maxTurns < 1 || maxTurns > 200}
        onClick={() => void microphone()}>{status === 'recording' ? <Send size={23} /> : <Mic size={25} />}</button>
      <div className="voice-status"><h3>Talk to Milo <span className="tag">GPT REALTIME 2 / PREVIEW / AI VOICE</span></h3>
        <span role="status" className="voice-state" data-recorded-seconds={duration}>{playing ? 'Milo speaking' : status === 'ready' && state.agent.phase === 'sleeping' ? 'Idle / microphone off' : ({ off: configurationStatus, connecting: 'Connecting to Foundry', requesting: 'Waiting for microphone', recording: `Recording / ${duration.toFixed(1)} s`, responding: 'Awaiting voice response', ready: 'Microphone off / connected' } as Record<string, string>)[status]}</span></div>
      {status !== 'off' && <button type="button" onClick={close}><PhoneOff size={16} /> End voice</button>}
    </div>
    {error && <div role="alert" className="error">{error}</div>}
    {(userText || robotText) && <div className="voice-transcripts">{userText && <div><strong>You</strong><p>{userText}</p></div>}{robotText && <div><strong><Volume2 size={14} /> Milo</strong><p>{robotText}</p></div>}</div>}
    <details className="voice-connection"><summary><Settings2 size={15} /> Realtime connection</summary>
      <p className="tag">Microsoft Foundry / {config.target_model} / {config.reasoning_effort} reasoning</p>
      <form onSubmit={async event => {
        event.preventDefault(); setError('');
        try { await request('voice/config', { endpoint, deployment, voice }); }
        catch (failure) { setError(String(failure)); }
      }}><fieldset disabled={state.agent.active}>
        <label>Resource endpoint<input aria-label="Realtime resource endpoint" type="url" required value={endpoint} placeholder="https://your-resource.openai.azure.com" onChange={event => setEndpoint(event.target.value)} /></label>
        <label>GPT Realtime 2 deployment<input aria-label="Realtime deployment" required maxLength={120} placeholder="gpt-realtime-2" value={deployment} onChange={event => setDeployment(event.target.value)} /></label>
        <label>Voice<select aria-label="Robot voice" value={voice} onChange={event => setVoice(event.target.value)}>{['alloy', 'echo', 'shimmer', 'ash', 'ballad', 'coral', 'sage', 'verse'].map(name => <option key={name}>{name}</option>)}</select></label>
        <button type="submit"><Check size={16} /> Apply voice connection</button>
      </fieldset></form>
    </details>
  </section>;
}