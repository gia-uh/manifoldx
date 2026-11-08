// A bounding volume hierarchy over triangles with binned SAH, as manifoldx/trace/bvh.py builds
// it: 16 bins, the same leaf rules, children of a node adjacent, leaves listing a contiguous run
// of `order`. The tree's shape can differ from numpy's (Python splits a level at a time), the
// closest hits cannot.

export const BINS = 16;
export const LEAF_SIZE = 4;
export const MAX_LEAF = 16;
const TRAVERSAL_COST = 1.0;

/** positions: Float32Array, 9 per triangle. Returns {nodes: Float32Array (8 per node: min.xyz,
 * first bits, max.xyz, count bits), order: Uint32Array, nodeCount}. */
export function build(positions) {
  const T = positions.length / 9;
  if (!T) throw new Error('cannot build a BVH over no triangles');
  const lo = new Float32Array(T * 3);
  const hi = new Float32Array(T * 3);
  const cen = new Float32Array(T * 3);
  for (let t = 0; t < T; t++) {
    for (let c = 0; c < 3; c++) {
      const a = positions[t * 9 + c];
      const b = positions[t * 9 + 3 + c];
      const d = positions[t * 9 + 6 + c];
      lo[t * 3 + c] = Math.min(a, b, d);
      hi[t * 3 + c] = Math.max(a, b, d);
      cen[t * 3 + c] = (lo[t * 3 + c] + hi[t * 3 + c]) * 0.5;
    }
  }
  const order = new Uint32Array(T);
  for (let i = 0; i < T; i++) order[i] = i;
  let cap = Math.max(2 * T, 16);
  let nodes = new Float32Array(cap * 8);
  let bits = new Uint32Array(nodes.buffer);
  let count = 1;
  const binCount = new Int32Array(BINS);
  const binLo = new Float64Array(BINS * 3);
  const binHi = new Float64Array(BINS * 3);
  const scratch = new Uint32Array(T);
  const rightArea = new Float64Array(BINS);
  const rightCount = new Int32Array(BINS);
  const box = new Float64Array(12);  // node lo, hi, centroid lo, hi
  // a stack of (node, start, end)
  let stack = new Uint32Array(3 * 64);
  let sp = 0;
  stack[sp++] = 0; stack[sp++] = 0; stack[sp++] = T;
  while (sp) {
    const end = stack[--sp];
    const start = stack[--sp];
    const id = stack[--sp];
    const n = end - start;
    box.fill(Infinity, 0, 3); box.fill(-Infinity, 3, 6); box.fill(Infinity, 6, 9); box.fill(-Infinity, 9, 12);
    for (let i = start; i < end; i++) {
      const t = order[i] * 3;
      for (let c = 0; c < 3; c++) {
        const l = lo[t + c];
        const h = hi[t + c];
        const m = cen[t + c];
        if (l < box[c]) box[c] = l;
        if (h > box[3 + c]) box[3 + c] = h;
        if (m < box[6 + c]) box[6 + c] = m;
        if (m > box[9 + c]) box[9 + c] = m;
      }
    }
    for (let c = 0; c < 3; c++) { nodes[id * 8 + c] = box[c]; nodes[id * 8 + 4 + c] = box[3 + c]; }
    let bestCost = Infinity;
    let bestAxis = 0;
    let bestSplit = 0;
    if (n > LEAF_SIZE) {
      for (let a = 0; a < 3; a++) {
        const ext = box[9 + a] - box[6 + a];
        const scale = ext > 0 ? BINS / ext : 0;
        const origin = box[6 + a];
        binCount.fill(0);
        binLo.fill(Infinity);
        binHi.fill(-Infinity);
        for (let i = start; i < end; i++) {
          const t = order[i] * 3;
          const b = Math.min(Math.trunc((cen[t + a] - origin) * scale), BINS - 1);
          binCount[b]++;
          const o = b * 3;
          if (lo[t] < binLo[o]) binLo[o] = lo[t];
          if (lo[t + 1] < binLo[o + 1]) binLo[o + 1] = lo[t + 1];
          if (lo[t + 2] < binLo[o + 2]) binLo[o + 2] = lo[t + 2];
          if (hi[t] > binHi[o]) binHi[o] = hi[t];
          if (hi[t + 1] > binHi[o + 1]) binHi[o + 1] = hi[t + 1];
          if (hi[t + 2] > binHi[o + 2]) binHi[o + 2] = hi[t + 2];
        }
        // sweep from the right: area and count of bins b+1 .. BINS-1
        let x0 = Infinity, y0 = Infinity, z0 = Infinity, x1 = -Infinity, y1 = -Infinity, z1 = -Infinity;
        let rc = 0;
        for (let b = BINS - 1; b > 0; b--) {
          rc += binCount[b];
          const o = b * 3;
          x0 = Math.min(x0, binLo[o]); y0 = Math.min(y0, binLo[o + 1]); z0 = Math.min(z0, binLo[o + 2]);
          x1 = Math.max(x1, binHi[o]); y1 = Math.max(y1, binHi[o + 1]); z1 = Math.max(z1, binHi[o + 2]);
          rightCount[b - 1] = rc;
          rightArea[b - 1] = area(x0, y0, z0, x1, y1, z1);
        }
        x0 = y0 = z0 = Infinity;
        x1 = y1 = z1 = -Infinity;
        let lc = 0;
        for (let b = 0; b < BINS - 1; b++) {
          lc += binCount[b];
          const o = b * 3;
          x0 = Math.min(x0, binLo[o]); y0 = Math.min(y0, binLo[o + 1]); z0 = Math.min(z0, binLo[o + 2]);
          x1 = Math.max(x1, binHi[o]); y1 = Math.max(y1, binHi[o + 1]); z1 = Math.max(z1, binHi[o + 2]);
          if (lc === 0 || rightCount[b] === 0) continue;
          const cost = lc * area(x0, y0, z0, x1, y1, z1) + rightCount[b] * rightArea[b];
          if (cost < bestCost) { bestCost = cost; bestAxis = a; bestSplit = b; }
        }
      }
    }
    const sah = TRAVERSAL_COST + bestCost / Math.max(area(box[0], box[1], box[2], box[3], box[4], box[5]), 1e-30);
    if (!(n > LEAF_SIZE && (sah < n || n > MAX_LEAF))) {
      bits[id * 8 + 3] = start;
      bits[id * 8 + 7] = n;
      continue;
    }
    let mid;
    if (Number.isFinite(bestCost)) {  // stable partition: left side first
      const ext = box[9 + bestAxis] - box[6 + bestAxis];
      const scale = ext > 0 ? BINS / ext : 0;
      const origin = box[6 + bestAxis];
      let l = start;
      let r = 0;
      for (let i = start; i < end; i++) {
        const t = order[i];
        const b = Math.min(Math.trunc((cen[t * 3 + bestAxis] - origin) * scale), BINS - 1);
        if (b > bestSplit) scratch[r++] = t;
        else order[l++] = t;
      }
      order.set(scratch.subarray(0, r), l);
      mid = l;
    } else {
      mid = start + (n >> 1);  // every centroid in one point: halve the list
    }
    if (count + 2 > cap) {
      cap *= 2;
      const grown = new Float32Array(cap * 8);
      grown.set(nodes);
      nodes = grown;
      bits = new Uint32Array(nodes.buffer);
    }
    const left = count;
    count += 2;
    bits[id * 8 + 3] = left;
    bits[id * 8 + 7] = 0;
    if (sp + 6 > stack.length) {
      const grown = new Uint32Array(stack.length * 2);
      grown.set(stack);
      stack = grown;
    }
    stack[sp++] = left + 1; stack[sp++] = mid; stack[sp++] = end;
    stack[sp++] = left; stack[sp++] = start; stack[sp++] = mid;
  }
  return { nodes: nodes.slice(0, count * 8), order, nodeCount: count };
}

function area(x0, y0, z0, x1, y1, z1) {
  const fin = (v) => (v > 0 && v < Infinity ? v : 0);
  const dx = fin(x1 - x0);
  const dy = fin(y1 - y0);
  const dz = fin(z1 - z0);
  return 2 * (dx * dy + dy * dz + dz * dx);
}
