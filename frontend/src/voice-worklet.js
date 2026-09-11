class MicrophonePCM extends AudioWorkletProcessor {
  constructor() {
    super();
    this.samples = new Int16Array(2400);
    this.offset = 0;
    this.port.onmessage = event => {
      if (event.data === 'flush') {
        this.flush();
        this.port.postMessage('flushed');
      }
    };
  }

  flush() {
    if (!this.offset) return;
    const packet = this.samples.slice(0, this.offset).buffer;
    this.port.postMessage(packet, [packet]);
    this.offset = 0;
  }

  process(inputs) {
    const input = inputs[0]?.[0];
    if (input) {
      for (const sample of input) {
        const bounded = Math.max(-1, Math.min(1, sample));
        this.samples[this.offset++] = Math.round(bounded * (bounded < 0 ? 32768 : 32767));
        if (this.offset === this.samples.length) this.flush();
      }
    }
    return true;
  }
}

registerProcessor('microphone-pcm', MicrophonePCM);