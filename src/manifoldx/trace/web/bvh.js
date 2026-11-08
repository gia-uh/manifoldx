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
  const rightArea = new Float64Array(BINS);
  const rightCount = new Int32Array(BINS);
  const scratch = new Uint32Array(T);
  const stack = [[0, 0, T]];
  while (stack.length) {
    const [id, start, end] = stack.pop();
    const n = end - start;
    const nlo = [Infinity, Infinity, Infinity];
    const nhi = [-Infinity, -Infinity, -Infinity];
    const clo = [Infinity, Infinity, Infinity];
    const chi = [-Infinity, -Infinity, -Infinity];
    for (let i = start; i < end; i++) {
      const t = order[i] * 3;
      for (let c = 0; c < 3; c++) {
        if (lo[t + c] < nlo[c]) nlo[c] = lo[t + c];
        if (hi[t + c] > nhi[c]) nhi[c] = hi[t + c];
        if (cen[t + c] < clo[c]) clo[c] = cen[t + c];
        if (cen[t + c] > chi[c]) chi[c] = cen[t + c];
      }
    }
    for (let c = 0; c < 3; c++) { nodes[id * 8 + c] = nlo[c]; nodes[id * 8 + 4 + c] = nhi[c]; }
    let bestCost = Infinity;
    let bestAxis = 0;
    let bestSplit = 0;
    if (n > LEAF_SIZE) {
      for (let a = 0; a < 3; a++) {
        const ext = chi[a] - clo[a];
        const scale = ext > 0 ? BINS / ext : 0;
        binCount.fill(0);
        binLo.fill(Infinity);
        binHi.fill(-Infinity);
        for (let i = start; i < end; i++) {
          const t = order[i] * 3;
          const b = Math.min(Math.trunc((cen[t + a] - clo[a]) * scale), BINS - 1);
          binCount[b]++;
          for (let c = 0; c < 3; c++) {
            if (lo[t + c] < binLo[b * 3 + c]) binLo[b * 3 + c] = lo[t + c];
            if (hi[t + c] > binHi[b * 3 + c]) binHi[b * 3 + c] = hi[t + c];
          }
        }
        // sweep from the right: area and count of bins b+1 .. BINS-1
        const rlo = [Infinity, Infinity, Infinity];
        const rhi = [-Infinity, -Infinity, -Infinity];
        let rc = 0;
        for (let b = BINS - 1; b > 0; b--) {
          rc += binCount[b];
          for (let c = 0; c < 3; c++) {
            rlo[c] = Math.min(rlo[c], binLo[b * 3 + c]);
            rhi[c] = Math.max(rhi[c], binHi[b * 3 + c]);
          }
          rightCount[b - 1] = rc;
          rightArea[b - 1] = area(rlo, rhi);
        }
        const llo = [Infinity, Infinity, Infinity];
        const lhi = [-Infinity, -Infinity, -Infinity];
        let lc = 0;
        for (let b = 0; b < BINS - 1; b++) {
          lc += binCount[b];
          for (let c = 0; c < 3; c++) {
            llo[c] = Math.min(llo[c], binLo[b * 3 + c]);
            lhi[c] = Math.max(lhi[c], binHi[b * 3 + c]);
          }
          if (lc === 0 || rightCount[b] === 0) continue;
          const cost = lc * area(llo, lhi) + rightCount[b] * rightArea[b];
          if (cost < bestCost) { bestCost = cost; bestAxis = a; bestSplit = b; }
        }
      }
    }
    const sah = TRAVERSAL_COST + bestCost / Math.max(area(nlo, nhi), 1e-30);
    const split = n > LEAF_SIZE && (sah < n || n > MAX_LEAF);
    if (!split) {
      bits[id * 8 + 3] = start;
      bits[id * 8 + 7] = n;
      continue;
    }
    let mid;
    if (Number.isFinite(bestCost)) {  // stable partition: left side first
      const ext = chi[bestAxis] - clo[bestAxis];
      const scale = ext > 0 ? BINS / ext : 0;
      let l = start;
      let r = 0;
      for (let i = start; i < end; i++) {
        const t = order[i];
        const b = Math.min(Math.trunc((cen[t * 3 + bestAxis] - clo[bestAxis]) * scale), BINS - 1);
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
    stack.push([left + 1, mid, end], [left, start, mid]);
  }
  return { nodes: nodes.slice(0, count * 8), order, nodeCount: count };
}

function area(lo, hi) {
  const d = [0, 1, 2].map((c) => {
    const v = Math.max(hi[c] - lo[c], 0);
    return Number.isFinite(v) ? v : 0;
  });
  return 2 * (d[0] * d[1] + d[1] * d[2] + d[2] * d[0]);
}
