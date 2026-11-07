"""The page's Store against the desktop's EntityStore: the same slots for the same spawns and
destroys, capacity that grows as the page spawns, queued writes landing at the flush, and the
store's trace on both sides."""

import json
import random
import subprocess
from pathlib import Path

import numpy as np
import pytest

from manifoldx.components import Transform
from manifoldx.ecs import EntityStore

MANIFOLDX_JS = Path(__file__).parents[2] / "src" / "manifoldx" / "web" / "js" / "manifoldx.js"

# An empty store's header: Transform and Velocity, nothing below the high-water mark.
_EMPTY = """
const empty = (maxEntities) => ({
  max_entities: maxEntities, high_water: 0, free: [], alive: [0, 0],
  components: [
    { name: "Transform", dtype: "f4", cols: 10, fields: { pos: [0, 3], rot: [3, 4], scale: [7, 3] }, rows: [0, 0], members: [0, 0] },
    { name: "Velocity", dtype: "f4", cols: 3, fields: { v: [0, 3] }, rows: [0, 0], members: [0, 0] },
  ],
});
const DEFAULT = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1];
const transforms = (n) => {
  const a = new Float32Array(n * 10);
  for (let i = 0; i < n; i++) a.set(DEFAULT, i * 10);
  return a;
};
const pyError = (e) => e.message.replace(/^RuntimeError: /, "");
"""


def run_js(body: str, data: object = None) -> object:
    """body run as an ES module with Store and traceStore imported and `data` (JSON) bound;
    its `out` printed as JSON and returned."""
    script = (
        f'import {{ Store, traceStore }} from "{MANIFOLDX_JS.as_uri()}";\n'
        f'import {{ readFileSync }} from "node:fs";\n{_EMPTY}\n'
        f"const data = {json.dumps(data)};\nlet out;\n{body}\n"
        "console.log(JSON.stringify(out));"
    )
    done = subprocess.run(
        ["node", "--input-type=module", "-e", script], capture_output=True, text=True, timeout=120
    )
    if done.returncode != 0:
        raise RuntimeError(f"node failed:\n{done.stderr}")
    return json.loads(done.stdout)


def run_store_ops(ops_: list[list], max_entities: int) -> list[dict]:
    """ops_ on an empty page Store: the state after each, as python_states writes it."""
    return run_js(
        """
const store = Store.load(empty(data.max), new ArrayBuffer(0));
out = [];
for (const [op, arg] of data.ops) {
  let got = null;
  try {
    if (op === "spawn") got = Array.from(store.spawn(arg, new Map([["Transform", transforms(arg)]])));
    else store.destroy(arg);
  } catch (e) {
    got = pyError(e);
  }
  const alive = [];
  store.alive.forEach((a, i) => { if (a) alive.push(i); });
  out.push({ got, free: [...store.free], hw: store.highWater, alive });
}
""",
        {"ops": ops_, "max": max_entities},
    )


def ops(seed: int, steps: int = 400) -> list[list]:
    rng = random.Random(seed)
    out = []
    for _ in range(steps):
        if rng.random() < 0.55:
            out.append(["spawn", rng.randint(0, 50)])
        else:  # repeats, dead slots and out-of-order indices
            k = rng.randint(0, 30)
            out.append(["destroy", [rng.randrange(0, 400) for _ in range(k)]])
    return out


def python_states(ops_, max_entities):
    store = EntityStore(max_entities)
    Transform.register(store)
    store.register_component("Velocity", np.dtype("f4"), (3,))
    states = []
    for op, arg in ops_:
        try:
            if op == "spawn":
                got = store.spawn(arg, Transform=Transform.get_default_data(arg)).tolist()
            else:
                store.destroy(np.asarray(arg, dtype=np.int64))
                got = None
        except RuntimeError as e:
            got = str(e)
        states.append({"got": got, "free": list(store._free_list), "hw": store._high_water,
                       "alive": np.flatnonzero(store._alive).tolist()})  # fmt: skip
    return states


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_the_page_picks_slots_as_the_desktop(seed):
    o = ops(seed)
    assert run_store_ops(o, 400) == python_states(o, 400)


def test_spawning_past_max_entities_throws_the_desktops_message():
    o = [["spawn", 30], ["spawn", 30]]
    py, js = python_states(o, 40), run_store_ops(o, 40)
    assert js == py
    assert js[1]["got"] == "Max entities 40 reached: spawning 30 with 10 free slots; raise it with Engine(max_entities=...)"  # fmt: skip


def _exported_store(tmp_path, engine):
    """engine's store as the export writes it: (header path, ecs.bin path)."""
    from manifoldx.web.snapshot import ecs_bytes

    header, data = ecs_bytes(engine)
    (tmp_path / "ecs.json").write_text(json.dumps(header))
    (tmp_path / "ecs.bin").write_bytes(data)
    return str(tmp_path / "ecs.json"), str(tmp_path / "ecs.bin")


_LOAD = """
const buf = readFileSync(data.bin);
const store = Store.load(JSON.parse(readFileSync(data.header, "utf8")),
                         buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
"""


def test_capacity_grows_and_keeps_rows(tmp_path):
    # a store loaded with high_water 6 spawns 500 one at a time, writing pos.x = index;
    # every row and member bit reads back, capacity is a power-of-two multiple of the start
    import manifoldx as mx

    engine = mx.Engine("grow", width=64, height=64)
    engine.spawn(Transform(pos=(7.0, 8.0, 9.0)), n=6)
    header, bin_ = _exported_store(tmp_path, engine)
    got = run_js(
        _LOAD
        + """
const start = store.capacity;
for (let i = 0; i < 500; i++) {
  const [slot] = store.spawn(1, new Map([["Transform", transforms(1)]]));
  store.write("Transform", slot, 0, [slot]);
}
const t = store.column("Transform"), m = store.member("Transform");
out = {
  start, capacity: store.capacity, hw: store.highWater,
  x: Array.from({ length: 506 }, (_, i) => t[i * 10]),
  yz: Array.from({ length: 506 }, (_, i) => [t[i * 10 + 1], t[i * 10 + 2]]),
  members: Array.from(m.slice(0, 506)),
};
""",
        {"header": header, "bin": bin_},
    )
    assert got["start"] == 6 and got["hw"] == 506
    ratio = got["capacity"] // 6
    assert got["capacity"] == 6 * ratio and ratio & (ratio - 1) == 0 and got["capacity"] >= 506
    assert got["x"] == [7.0] * 6 + [float(i) for i in range(6, 506)]
    assert got["yz"] == [[8.0, 9.0]] * 6 + [[0.0, 0.0]] * 500
    assert got["members"] == [1] * 506


def test_queued_updates_land_at_the_flush_in_order():
    # queueUpdate pos of [0, 1] to 1, then queueDestroy [1], then queueUpdate [0] to 2:
    # before flush rows are unchanged; after, row 0 is 2 and slot 1 is dead and in the free list
    got = run_js("""
const store = Store.load(empty(16), new ArrayBuffer(0));
store.spawn(2, new Map([["Transform", transforms(2)]]));
store.queueUpdate("Transform", [0, 1], 0, 3, new Float32Array([1, 1, 1, 1, 1, 1]));
store.queueDestroy([1]);
store.queueUpdate("Transform", [0], 0, 3, new Float32Array([2, 2, 2]));
const t = store.column("Transform");
const before = { rows: [Array.from(t.slice(0, 3)), Array.from(t.slice(10, 13))], alive: Array.from(store.alive.slice(0, 2)) };
const version = store.version;
store.flush();
out = {
  before,
  after: { rows: [Array.from(t.slice(0, 3)), Array.from(t.slice(10, 13))], alive: Array.from(store.alive.slice(0, 2)) },
  free: store.free, bumped: store.version > version, query: Array.from(store.query(["Transform"])),
};
store.flush();
""")
    assert got["before"] == {"rows": [[0.0] * 3, [0.0] * 3], "alive": [1, 1]}
    assert got["after"] == {"rows": [[2.0] * 3, [1.0] * 3], "alive": [1, 0]}
    assert got["free"] == [1] and got["bumped"] and got["query"] == [0]


def test_a_query_takes_alive_members_and_an_unknown_name_matches_nothing():
    got = run_js("""
const store = Store.load(empty(16), new ArrayBuffer(0));
store.spawn(3, new Map([["Transform", transforms(3)], ["Velocity", new Float32Array(9)]]));
store.spawn(2, new Map([["Transform", transforms(2)]]));
store.destroy([1]);
out = [store.query(["Velocity"]), store.query(["Transform"]), store.query(["Velocity", "Nope"]), store.query([])].map((q) => Array.from(q));
""")
    assert got == [[0, 2], [0, 2, 3, 4], [], [0, 2, 3, 4]]


def test_store_trace_matches_and_writes_non_finite_as_strings(tmp_path):
    # a row with nan, inf and -inf: store_trace(py)["rows"]["Transform"][0][:3] == ["nan", "inf", "-inf"]
    # and traceStore in Node gives the same JSON; past 1,000 alive entities rows become
    # {"digest": d, "n": k} on both sides, with the same d
    import manifoldx as mx
    from manifoldx.web.replay import store_trace

    def both(engine, folder):
        folder.mkdir()
        header, bin_ = _exported_store(folder, engine)
        js = run_js(_LOAD + "out = traceStore(store);", {"header": header, "bin": bin_})
        return json.loads(json.dumps(store_trace(engine.store))), js

    engine = mx.Engine("trace", width=64, height=64)
    engine.spawn(Transform(pos=(float("nan"), float("inf"), float("-inf"))))
    engine.spawn(Transform(pos=(1.0, 2.0, 3.0)), n=2)
    engine.store.destroy(np.array([1]))
    py, js = both(engine, tmp_path / "small")
    assert py["rows"]["Transform"][0][:3] == ["nan", "inf", "-inf"]
    assert py["alive"] == [0, 2] and py["free"] == [1] and py["high_water"] == 3
    assert js == py

    big = mx.Engine("big", width=64, height=64)
    rng = np.random.default_rng(4)
    big.spawn(Transform(pos=(0.0, 0.0, 0.0)), n=1200)
    big.store._components["Transform"][:1200, :3] = rng.normal(size=(1200, 3)).astype(np.float32)
    py, js = both(big, tmp_path / "big")
    assert py["rows"]["Transform"]["n"] == 1200
    assert js == py
