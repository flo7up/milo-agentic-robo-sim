type Capture = { stream: MediaStream; source: MediaStreamAudioSourceNode; node: AudioWorkletNode | ScriptProcessorNode; gain: GainNode; flushed?: () => void };

export class VoiceAudio {
  private context = new AudioContext({ sampleRate: 24000 });
  private capture: Capture | null = null;
  private closed = false;
  private moduleReady: Promise<void> | null = null;
  private playback = new Set<AudioBufferSourceNode>();
  private nextStart = 0;

  constructor(private onPlayback: (playing: boolean) => void) {}

  async resume() {
    await this.context.resume();
    if (this.context.sampleRate !== 24000) throw new Error('This browser cannot provide 24 kHz voice audio.');
  }

  async startCapture(send: (packet: ArrayBuffer) => void, onSamples: (samples: number) => void) {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true }, video: false });
    try {
      await this.context.resume();
      this.moduleReady ??= this.context.audioWorklet.addModule(new URL('./voice-worklet.js?no-inline', import.meta.url).href);
      let timer: ReturnType<typeof setTimeout>;
      const workletReady = await Promise.race([
        this.moduleReady.then(() => true, () => false),
        new Promise<boolean>(resolve => { timer = setTimeout(() => resolve(false), 2000); }),
      ]);
      clearTimeout(timer!);
      if (this.closed) throw new Error('Voice connection closed.');
      const source = this.context.createMediaStreamSource(stream);
      const node = workletReady ? new AudioWorkletNode(this.context, 'microphone-pcm') : this.context.createScriptProcessor(1024, 1, 1);
      const gain = this.context.createGain();
      gain.gain.value = 0;
      const capture: Capture = { stream, source, node, gain };
      let samples = 0;
      const deliver = (packet: ArrayBuffer) => {
        if (this.closed || samples >= 720000) return;
        const bounded = packet.slice(0, (720000 - samples) * 2);
        send(bounded);
        samples += bounded.byteLength / 2;
        onSamples(samples);
      };
      if (node instanceof AudioWorkletNode) {
        node.port.onmessage = event => {
          if (event.data === 'flushed') { capture.flushed?.(); return; }
          deliver(event.data as ArrayBuffer);
        };
      } else {
        node.onaudioprocess = event => {
          const input = event.inputBuffer.getChannelData(0);
          const samples = new Int16Array(input.length);
          for (let index = 0; index < input.length; index++) {
            const bounded = Math.max(-1, Math.min(1, input[index]));
            samples[index] = Math.round(bounded * (bounded < 0 ? 32768 : 32767));
          }
          deliver(samples.buffer);
        };
      }
      this.capture = capture;
      source.connect(node);
      node.connect(gain);
      gain.connect(this.context.destination);
    } catch (failure) {
      stream.getTracks().forEach(track => track.stop());
      throw failure;
    }
  }

  async stopCapture() {
    const capture = this.capture;
    if (!capture) return;
    capture.stream.getTracks().forEach(track => track.stop());
    capture.source.disconnect();
    let timer: ReturnType<typeof setTimeout>;
    try {
      const node = capture.node;
      if (node instanceof AudioWorkletNode) {
        await new Promise<void>((resolve, reject) => {
          capture.flushed = resolve;
          timer = setTimeout(() => reject(new Error('Microphone flush timed out.')), 1000);
          node.port.postMessage('flush');
        });
      } else node.onaudioprocess = null;
    } finally {
      clearTimeout(timer!);
      capture.node.disconnect();
      capture.gain.disconnect();
      if (capture.node instanceof AudioWorkletNode) capture.node.port.close();
      if (this.capture === capture) this.capture = null;
    }
  }

  play(packet: ArrayBuffer) {
    if (this.closed || !packet.byteLength) return;
    if (packet.byteLength % 2 || this.nextStart - this.context.currentTime > 60) throw new Error('Voice playback buffer exceeded.');
    const source = this.context.createBufferSource();
    const samples = new Int16Array(packet);
    const buffer = this.context.createBuffer(1, samples.length, 24000);
    const output = buffer.getChannelData(0);
    for (let index = 0; index < samples.length; index++) output[index] = samples[index] / 32768;
    source.buffer = buffer;
    source.connect(this.context.destination);
    this.playback.add(source);
    this.onPlayback(true);
    source.onended = () => {
      this.playback.delete(source);
      source.disconnect();
      if (!this.playback.size) this.onPlayback(false);
    };
    const start = Math.max(this.context.currentTime + .02, this.nextStart);
    this.nextStart = start + buffer.duration;
    source.start(start);
  }

  close() {
    if (this.closed) return;
    this.closed = true;
    if (this.capture) {
      this.capture.stream.getTracks().forEach(track => track.stop());
      this.capture.source.disconnect();
      this.capture.node.disconnect();
      this.capture.gain.disconnect();
      this.capture.flushed?.();
      if (this.capture.node instanceof AudioWorkletNode) this.capture.node.port.close();
      else this.capture.node.onaudioprocess = null;
      this.capture = null;
    }
    for (const source of this.playback) source.stop();
    this.playback.clear();
    this.onPlayback(false);
    void this.context.close();
  }
}