// The loading line (how much the scene weighs, how much is in, how long is left) and the
// frame clock the page's loop steps with.

import assert from "node:assert/strict";
import test from "node:test";

import { RateMeter, formatBytes, frameDt, progressLine } from "../../../src/manifoldx/web/js/progress.js";

const MB = 1024 * 1024;

test("bytes read as KB or MB", () => {
  assert.equal(formatBytes(850 * 1024), "850 KB");
  assert.equal(formatBytes(2.2 * MB), "2.2 MB");
  assert.equal(formatBytes(0), "0 KB");
});

test("the line gives the download size, the speed and the time left", () => {
  // 1 MB of a 4.4 MB scene in, sent gzipped at 2.2 MB: the line shows the gzipped figures.
  const line = progressLine({ fraction: 1 / 4.4, download: 2.2 * MB, rate: 40 * 1024 });
  assert.equal(line, "0.5 of 2.2 MB · 40 KB/s · about 44 s left");
  assert.equal(progressLine({ fraction: 0.5, download: 600 * 1024, rate: 0 }), "300 of 600 KB");
});

test("long waits read in minutes, and no rate means no estimate", () => {
  assert.match(progressLine({ fraction: 0.1, download: 10 * MB, rate: 20 * 1024 }), /about 8 min left$/);
  assert.match(progressLine({ fraction: 0.5, download: 1 * MB, rate: 6 * 1024 }), /about 85 s left$/);
  assert.equal(progressLine({ fraction: 0, download: 2.2 * MB, rate: 0 }), "0.0 of 2.2 MB");
});

test("an unknown size still shows what has arrived", () => {
  assert.equal(progressLine({ fraction: null, download: null, rate: 0, received: 3 * MB }), "3.0 MB");
});

test("the rate meter averages over the last 3 s, in download bytes per second", () => {
  const m = new RateMeter();
  assert.equal(m.window, 3);
  m.add(0.0, 0);
  m.add(1.0, 50_000);
  m.add(2.0, 100_000);
  assert.ok(Math.abs(m.rate() - 50_000) < 1, m.rate());
  m.add(3.0, 100_000); // a stall pulls the estimate down
  assert.ok(m.rate() < 50_000, m.rate());
  m.add(10.0, 100_000); // the old samples age out
  m.add(11.0, 110_000);
  assert.ok(Math.abs(m.rate() - 10_000) < 1, m.rate());
});

test("frameDt is the time between frames in seconds, at most 0.1", () => {
  assert.equal(frameDt(1016, 1000), 0.016);
  assert.equal(frameDt(6000, 1000), 0.1); // a background tab's 5 s gap
});
