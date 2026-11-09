// Resamples the mic (whatever the context rate is, normally 48 kHz) to 16 kHz,
// 16-bit mono, and posts exactly 1,024-byte frames (512 samples = 32 ms).
// The audio clock drives this, so frames come out at real-time cadence.
class Downsampler extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000; // 3 for 48 kHz
    this.phase = 0;
    this.acc = 0;
    this.n = 0;
    this.buf = new Int16Array(512);
    this.len = 0;
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      this.acc += ch[i]; // box filter over the input samples = cheap anti-alias
      this.n++;
      this.phase += 1;
      if (this.phase >= this.ratio) {
        this.phase -= this.ratio;
        const v = Math.max(-1, Math.min(1, this.acc / this.n));
        this.buf[this.len++] = v < 0 ? v * 32768 : v * 32767;
        this.acc = 0;
        this.n = 0;
        if (this.len === 512) {
          this.port.postMessage(this.buf.buffer.slice(0), []);
          this.len = 0;
        }
      }
    }
    return true;
  }
}
registerProcessor('downsampler', Downsampler);
