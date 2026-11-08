"""A bounding volume hierarchy over triangles, built in numpy with binned SAH.

All nodes of one depth are split together: per level, a handful of numpy calls over every
triangle not yet in a leaf. Children of a node are adjacent; leaves list a contiguous run of
triangles in `order`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BINS = 16
LEAF_SIZE = 4  # a node this small or smaller is always a leaf
MAX_LEAF = 16  # a node bigger than this is always split, whatever the SAH says
TRAVERSAL_COST = 1.0  # relative to one triangle test


@dataclass
class BVH:
    bmin: np.ndarray  # (N, 3) float32
    bmax: np.ndarray  # (N, 3) float32
    first: np.ndarray  # (N,) uint32: left child for an inner node, first triangle for a leaf
    count: np.ndarray  # (N,) uint32: 0 for an inner node
    order: np.ndarray  # (T,) int64: triangle index for each leaf slot

    @property
    def node_count(self) -> int:
        return len(self.count)

    def packed(self) -> np.ndarray:
        """(N, 8) float32 rows: min.xyz, first (bits), max.xyz, count (bits); 32 bytes a node."""
        out = np.empty((self.node_count, 8), np.float32)
        out[:, 0:3], out[:, 4:7] = self.bmin, self.bmax
        out.view(np.uint32)[:, 3] = self.first
        out.view(np.uint32)[:, 7] = self.count
        return out

    def depth(self) -> int:
        d = np.zeros(self.node_count, np.int64)
        inner = np.flatnonzero(self.count == 0)
        for i in inner:  # parents come before children
            d[self.first[i]] = d[self.first[i] + 1] = d[i] + 1
        return int(d.max()) + 1


def build(positions: np.ndarray) -> BVH:
    """positions: (T, 3, 3) triangle vertices."""
    p = np.asarray(positions, np.float32)
    T = len(p)
    if T == 0:
        raise ValueError("cannot build a BVH over no triangles")
    lo, hi = p.min(axis=1), p.max(axis=1)
    cen = (lo + hi) * 0.5

    cap = max(2 * T, 16)
    bmin = np.empty((cap, 3), np.float32)
    bmax = np.empty((cap, 3), np.float32)
    first = np.zeros(cap, np.uint32)
    count = np.zeros(cap, np.uint32)
    n_nodes = 1

    order = np.arange(T, dtype=np.int64)
    # active nodes of this level: ids, start, end in `order`
    ids, starts, ends = np.array([0]), np.array([0]), np.array([T])
    while len(ids):
        lengths = ends - starts
        K = len(ids)
        seg = np.repeat(np.arange(K), lengths)
        offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]])
        pos = np.repeat(starts - offsets, lengths) + np.arange(lengths.sum())
        tri = order[pos]
        tlo, thi, tc = lo[tri], hi[tri], cen[tri]

        nlo = np.minimum.reduceat(tlo, offsets)
        nhi = np.maximum.reduceat(thi, offsets)
        bmin[ids], bmax[ids] = nlo, nhi
        clo = np.minimum.reduceat(tc, offsets)
        chi = np.maximum.reduceat(tc, offsets)
        ext = chi - clo

        best_cost = np.full(K, np.inf)
        best_axis = np.zeros(K, np.int64)
        best_split = np.zeros(K, np.int64)
        bins_of = []
        for a in range(3):
            scale = np.where(ext[:, a] > 0, BINS / np.maximum(ext[:, a], 1e-30), 0.0)
            b = np.minimum(((tc[:, a] - clo[seg, a]) * scale[seg]).astype(np.int64), BINS - 1)
            bins_of.append(b)
            key = seg * BINS + b
            cnt = np.bincount(key, minlength=K * BINS).reshape(K, BINS)
            blo = np.full((K * BINS, 3), np.inf, np.float32)
            bhi = np.full((K * BINS, 3), -np.inf, np.float32)
            np.minimum.at(blo, key, tlo)
            np.maximum.at(bhi, key, thi)
            blo, bhi = blo.reshape(K, BINS, 3), bhi.reshape(K, BINS, 3)
            lcnt = np.cumsum(cnt, axis=1)[:, :-1]
            rcnt = np.cumsum(cnt[:, ::-1], axis=1)[:, ::-1][:, 1:]
            larea = _area(np.minimum.accumulate(blo, axis=1), np.maximum.accumulate(bhi, axis=1))[:, :-1]
            rarea = _area(np.minimum.accumulate(blo[:, ::-1], axis=1)[:, ::-1],
                          np.maximum.accumulate(bhi[:, ::-1], axis=1)[:, ::-1])[:, 1:]
            cost = np.where((lcnt > 0) & (rcnt > 0), lcnt * larea + rcnt * rarea, np.inf)
            split = cost.argmin(axis=1)
            c = cost[np.arange(K), split]
            better = c < best_cost
            best_cost[better], best_axis[better], best_split[better] = c[better], a, split[better]

        node_area = np.maximum(_area(nlo, nhi), 1e-30)
        sah = TRAVERSAL_COST + best_cost / node_area
        has_split = np.isfinite(best_cost)
        split_it = (lengths > LEAF_SIZE) & ((sah < lengths) | (lengths > MAX_LEAF))
        median = split_it & ~has_split  # every centroid in one point: halve the list

        leaf = ~split_it
        first[ids[leaf]] = starts[leaf]
        count[ids[leaf]] = lengths[leaf]
        if not split_it.any():
            break

        bins = np.stack(bins_of)  # (3, n)
        side = bins[best_axis[seg], np.arange(len(seg))] > best_split[seg]
        local = np.arange(len(seg)) - offsets[seg]
        side = np.where(median[seg], local >= lengths[seg] // 2, side)
        keep = split_it[seg]
        # stable sort inside each split segment: left side first
        sub = np.flatnonzero(keep)
        perm = np.lexsort((side[sub], seg[sub]))
        order[pos[sub]] = tri[sub][perm]

        sids = ids[split_it]
        n_split = len(sids)
        if n_nodes + 2 * n_split > cap:
            grow = max(cap, n_nodes + 2 * n_split)
            bmin, bmax = np.resize(bmin, (cap + grow, 3)), np.resize(bmax, (cap + grow, 3))
            first, count = np.resize(first, cap + grow), np.resize(count, cap + grow)
            cap += grow
        left = n_nodes + 2 * np.arange(n_split)
        n_nodes += 2 * n_split
        first[sids] = left
        count[sids] = 0
        nleft = np.bincount(seg[keep], weights=~side[keep], minlength=K).astype(np.int64)[split_it]
        s, e = starts[split_it], ends[split_it]
        ids = np.concatenate([left, left + 1])
        starts = np.concatenate([s, s + nleft])
        ends = np.concatenate([s + nleft, e])

    return BVH(bmin[:n_nodes].copy(), bmax[:n_nodes].copy(), first[:n_nodes].copy(), count[:n_nodes].copy(),
               order)


def _area(lo, hi):
    d = np.maximum(hi - lo, 0.0)
    d = np.where(np.isfinite(d), d, 0.0)
    return 2.0 * (d[..., 0] * d[..., 1] + d[..., 1] * d[..., 2] + d[..., 2] * d[..., 0])
