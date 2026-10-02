class DebateCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = [];
    this.sampleStart = 0;
    this.active = true;
    this.frameSize = Math.round(sampleRate * 0.1);
    this.port.onmessage = ({ data }) => {
      if (data === 'stop') {
        this.active = false;
        this.buffer = [];
      }
      if (data === 'finish') {
        this.active = false;
        this.flush();
        this.port.postMessage({ type: 'flushed' });
      }
    };
  }
  flush() {
    if (!this.buffer.length) return;
    const pcm = new ArrayBuffer(this.buffer.length * 2),
      view = new DataView(pcm);
    for (let i = 0; i < this.buffer.length; i++)
      view.setInt16(
        i * 2,
        Math.round(Math.max(-1, Math.min(1, this.buffer[i])) * 32767),
        true,
      );
    this.port.postMessage(
      {
        type: 'frame',
        pcm,
        sampleStart: this.sampleStart,
        samples: this.buffer.length,
      },
      [pcm],
    );
    this.sampleStart += this.buffer.length;
    this.buffer = [];
  }
  process(inputs) {
    if (!this.active) return true;
    const channels = inputs[0];
    if (!channels?.length) return true;
    for (let i = 0; i < channels[0].length; i++) {
      let sample = 0;
      for (const c of channels) sample += c[i] / channels.length;
      this.buffer.push(sample);
      if (this.buffer.length === this.frameSize) this.flush();
    }
    return true;
  }
}
registerProcessor('debate-capture', DebateCapture);
