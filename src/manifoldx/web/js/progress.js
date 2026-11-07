// The loading line under the progress bar: the download's size, how much is in, the speed and
// the time left; and the clock the page's loop steps the app with. Sizes are what travels:
// when the server gzips scene.glb it sends no usable Content-Length, so index.html carries both
// the scene's size and its gzipped size.

const KB = 1024;
const MB = 1024 * 1024;

export function formatBytes(n) {
  return n >= MB ? `${(n / MB).toFixed(1)} MB` : `${Math.round(n / KB)} KB`;
}

function formatWait(seconds) {
  return seconds >= 90 ? `about ${Math.round(seconds / 60)} min left` : `about ${Math.round(seconds)} s left`;
}

// fraction: the share of the scene received (null when unknown); download: bytes on the wire;
// rate: wire bytes per second; received: bytes in, shown when the total is unknown.
export function progressLine({ fraction, download, rate, received = 0 }) {
  if (fraction === null || download === null) return formatBytes(received);
  const done = fraction * download;
  const parts = [download >= MB ? `${(done / MB).toFixed(1)} of ${formatBytes(download)}`
                                : `${Math.round(done / KB)} of ${formatBytes(download)}`];
  if (rate > 0) {
    parts.push(`${formatBytes(rate)}/s`);
    parts.push(formatWait(((1 - fraction) * download) / rate));
  }
  return parts.join(" · ");
}

// The speed over the last `window` seconds, so the estimate follows the link as it changes
// rather than averaging from the start. add(t, bytes): t in seconds, bytes received so far.
export class RateMeter {
  constructor(window = 3.0) {
    this.window = window;
    this.samples = [];
  }

  add(t, bytes) {
    this.samples.push([t, bytes]);
    while (this.samples.length > 2 && this.samples[0][0] < t - this.window) this.samples.shift();
  }

  rate() {
    if (this.samples.length < 2) return 0;
    const [t0, b0] = this.samples[0];
    const [t1, b1] = this.samples[this.samples.length - 1];
    return t1 > t0 ? (b1 - b0) / (t1 - t0) : 0;
  }
}

// The dt of one animation frame in seconds, from two requestAnimationFrame times in ms: at most
// 0.1, so a tab coming back from the background does not step the app across the gap.
export function frameDt(now, last) {
  return Math.min((now - last) / 1000, 0.1);
}
