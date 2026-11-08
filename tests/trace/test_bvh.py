import numpy as np

from manifoldx.trace.bvh import build


def _soup(n, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-5, 5, (n, 1, 3))
    return (centers + rng.normal(0, 0.3, (n, 3, 3))).astype(np.float32)


def _hit(p, o, d):
    """Möller-Trumbore over triangles p (k, 3, 3) for one ray; nearest t or inf."""
    e1, e2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    h = np.cross(d, e2)
    a = np.einsum("ij,ij->i", e1, h)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = 1.0 / a
        s = o - p[:, 0]
        u = f * np.einsum("ij,ij->i", s, h)
        q = np.cross(s, e1)
        v = f * (q @ d)
        t = f * np.einsum("ij,ij->i", e2, q)
    ok = (np.abs(a) > 1e-12) & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-6)
    return np.where(ok, t, np.inf).min(initial=np.inf)


def _walk(bvh, p, o, d):
    inv = 1.0 / np.where(d == 0, 1e-30, d)
    best, stack = np.inf, [0]
    while stack:
        i = stack.pop()
        t0, t1 = (bvh.bmin[i] - o) * inv, (bvh.bmax[i] - o) * inv
        near, far = np.minimum(t0, t1).max(), np.maximum(t0, t1).min()
        if far < max(near, 0) or near > best:
            continue
        if bvh.count[i]:
            tris = bvh.order[bvh.first[i]: bvh.first[i] + bvh.count[i]]
            best = min(best, _hit(p[tris], o, d))
        else:
            stack += [int(bvh.first[i]), int(bvh.first[i]) + 1]
    return best


def test_every_triangle_is_in_exactly_one_leaf():
    p = _soup(3000)
    bvh = build(p)
    leaves = bvh.count > 0
    slots = np.concatenate([np.arange(f, f + c) for f, c in zip(bvh.first[leaves], bvh.count[leaves])])
    assert sorted(slots.tolist()) == list(range(len(p)))
    assert sorted(bvh.order.tolist()) == list(range(len(p)))


def test_node_boxes_hold_their_children_and_triangles():
    p = _soup(2000, seed=1)
    bvh = build(p)
    for i in range(bvh.node_count):
        if bvh.count[i]:
            tris = p[bvh.order[bvh.first[i]: bvh.first[i] + bvh.count[i]]]
            assert (tris.min(axis=(0, 1)) >= bvh.bmin[i] - 1e-6).all()
            assert (tris.max(axis=(0, 1)) <= bvh.bmax[i] + 1e-6).all()
        else:
            for c in (bvh.first[i], bvh.first[i] + 1):
                assert (bvh.bmin[c] >= bvh.bmin[i] - 1e-6).all() and (bvh.bmax[c] <= bvh.bmax[i] + 1e-6).all()


def test_closest_hits_match_brute_force():
    p = _soup(1500, seed=2)
    bvh = build(p)
    rng = np.random.default_rng(3)
    for _ in range(300):
        o = rng.uniform(-8, 8, 3)
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        assert _walk(bvh, p, o, d) == _hit(p, o, d)


def test_degenerate_input_still_builds_leaves():
    p = np.zeros((50, 3, 3), np.float32)  # every centroid in one point
    bvh = build(p)
    assert (bvh.count[bvh.count > 0] <= 16).all()
    assert bvh.count.sum() == 50


def test_a_single_triangle_is_a_leaf_root():
    bvh = build(_soup(1))
    assert bvh.node_count == 1 and bvh.count[0] == 1
