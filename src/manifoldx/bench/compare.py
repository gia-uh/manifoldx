"""`manifoldx bench compare`: runs on two or more sides as one Markdown table, and their shots side by side.

A side is one run folder, a folder of run folders (a side of a set), or every side of a set folder. The
first side is the baseline. Each row is one number of one phase: a side's value is the median over its
runs and its spread the minimum and maximum, or, with one run, p25 to p75 of the phase's frame times for
p50_ms. A ratio is the side's value over the baseline's. A timing row whose ratio is within 1.3x either
way needs a quiet re-run, and so does any timing row when a run saw a load average above half its CPUs:
the noise of a shared host is about 10 %. Counts, memory and the app's metrics get no such note.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CLOSE = 1.3  # a timing ratio within this, either way, is not trusted on one run of a shared host
SYSTEMS = 5  # the baseline's slowest systems shown
GAP = 8  # px between shots on a sheet
GAP_RGB = (17, 16, 14)


@dataclass
class SideRuns:
    """One side: its label, the summaries of its runs and their folders."""

    label: str
    runs: list = field(default_factory=list)
    folders: list = field(default_factory=list)


@dataclass
class Row:
    phase: str
    probe: str
    key: str
    values: list
    spreads: list
    ratios: list
    note: str
    timing: bool


def _side_of(folders: list[Path], label: str) -> SideRuns:
    from manifoldx.bench.folder import read_run

    return SideRuns(label, [read_run(f) for f in folders], list(folders))


def load_sides(paths) -> list[SideRuns]:
    """The sides in `paths`: a set folder gives all its sides, a run folder one side of one run, any other
    folder one side of the run folders in it."""
    sides = []
    for path in map(Path, paths):
        if (path / "set.json").exists():
            data = json.loads((path / "set.json").read_text())
            for k, side in enumerate(data["sides"]):
                folders = [path / r["folder"] for r in data["runs"] if r["side"] == k and r["folder"]]
                sides.append(_side_of(folders, side["label"]))
        elif (path / "run.json").exists():
            side = _side_of([path], path.name)
            side.label = side.runs[0].get("label") or path.name
            sides.append(side)
        else:
            folders = sorted(p for p in path.iterdir() if (p / "run.json").exists()) if path.is_dir() else []
            if not folders:
                raise ValueError(f"{path} is not a run folder, a set folder or a folder of runs")
            sides.append(_side_of(folders, path.name))
    return sides


# --- rows ----------------------------------------------------------------------------------------

FRAMES = ("p50_ms", "p95_ms", "draw_p50_ms", "cpu_p50_ms")


def _pick(ph: dict, probe: str, key: str):
    if probe == "frames":
        return ph.get(key)
    if probe in ("split", "systems", "spans"):
        return (ph.get(f"{probe}_ms") or {}).get(key)
    if probe == "counters":
        return ((ph.get("counters") or {}).get(key) or {}).get("mean")
    if probe == "streamer":
        return (ph.get("streamer") or {}).get(key)
    if probe == "workers":
        return ((ph.get("workers") or {}).get(key) or {}).get("p50_ms")
    if probe == "memory":
        return ph.get(key)
    if probe == "app":
        return ((ph.get("app") or {}).get(key) or {}).get("mean")
    return None


def _keys(sides, phase: str) -> list[tuple[str, str, bool]]:
    """(probe, key, timing) for one phase, in the table's order."""
    phs = [r["phases"][phase] for s in sides for r in s.runs if phase in r["phases"]]

    def union(get):
        out: dict = {}
        for ph in phs:
            out.update(dict.fromkeys(get(ph) or {}))
        return list(out)

    base = [r["phases"][phase] for r in sides[0].runs if phase in r["phases"]]
    slow = sorted((base[0].get("systems_ms") or {}).items(), key=lambda kv: -kv[1])[:SYSTEMS] if base else []
    keys = [("frames", k, True) for k in FRAMES]
    keys += [("split", k, True) for k in union(lambda p: p.get("split_ms"))]
    keys += [("systems", k, True) for k, _ in slow]
    keys += [("spans", k, True) for k in union(lambda p: p.get("spans_ms"))]
    keys += [("counters", k, False) for k in union(lambda p: p.get("counters"))]
    keys += [("streamer", k, True) for k in ("first_drawn_s", "settled_s")]
    keys += [("workers", k, True) for k in union(lambda p: p.get("workers"))]
    keys += [("memory", "rss_max_mb", False)]
    keys += [("app", k, False) for k in union(lambda p: p.get("app"))]
    return keys


def _frame_spread(side: SideRuns, phase: str):
    """p25 and p75 of one run's frame times in a phase."""
    run, folder = side.runs[0], side.folders[0]
    try:
        k = run["phase_order"].index(phase)
        with np.load(Path(folder) / "frames.npz") as f:
            wall = f["wall_ms"][f["phase"] == k]
    except (OSError, KeyError, ValueError):
        return None
    if not len(wall):
        return None
    return float(np.percentile(wall, 25)), float(np.percentile(wall, 75))


def loaded(sides) -> tuple[float, int] | None:
    """The highest load and its CPU count over the runs that saw a load above half their CPUs."""
    worst = None
    for s in sides:
        for r in s.runs:
            host = r.get("host") or {}
            load, cpus = host.get("load_max"), host.get("cpus")
            if load is not None and cpus and load > cpus / 2 and (worst is None or load > worst[0]):
                worst = (load, cpus)
    return worst


def rows(sides: list[SideRuns]) -> list[Row]:
    """Every row of the comparison, phase by phase, for the phases every side ran."""
    phases = [p for p in (sides[0].runs[0]["phase_order"] if sides[0].runs else [])
              if all(any(p in r["phases"] for r in s.runs) for s in sides)]
    load = loaded(sides)
    out = []
    for phase in phases:
        for probe, key, timing in _keys(sides, phase):
            values, spreads = [], []
            for s in sides:
                got = [_pick(r["phases"][phase], probe, key) for r in s.runs if phase in r["phases"]]
                got = [float(v) for v in got if isinstance(v, (int, float)) and not isinstance(v, bool)]
                values.append(round(statistics.median(got), 4) if got else None)
                if len(got) > 1:
                    spreads.append((min(got), max(got)))
                elif got and probe == "frames" and key == "p50_ms":
                    spreads.append(_frame_spread(s, phase))
                else:
                    spreads.append(None)
            if all(v is None for v in values):
                continue
            base = values[0]
            ratios = [round(v / base, 4) if v is not None and base else None for v in values]
            notes = []
            if timing and any(r is not None and 1 / CLOSE < r < CLOSE for r in ratios[1:]):
                notes.append("needs a quiet re-run")
            if timing and load:
                notes.append(f"taken under load ({load[0]:.1f} of {load[1]} cpus)")
            out.append(Row(phase, probe, key, values, spreads, ratios, "; ".join(notes), timing))
    return out


def _num(v) -> str:
    if v is None:
        return "–"
    v = float(v)
    return f"{v:.0f}" if abs(v) >= 100 else f"{v:.1f}" if abs(v) >= 1 else f"{v:.3g}"


def table(sides: list[SideRuns], rows_: list[Row], left_out: list[SideRuns] = ()) -> str:
    """The rows as Markdown, then one line per side, and one per side left out."""
    labels = [s.label for s in sides]
    head = ["phase", "probe", "key", *labels, *(f"{x} ×base" for x in labels[1:]), "note"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows_:
        cells = [f"{_num(v)} ({_num(s[0])}–{_num(s[1])})" if s and v is not None else _num(v)
                 for v, s in zip(r.values, r.spreads)]
        ratios = [f"{x:.2f}×" if x is not None else "–" for x in r.ratios[1:]]
        lines.append("| " + " | ".join([r.phase, r.probe, r.key, *cells, *ratios, r.note]) + " |")
    lines.append("")
    for s in sides:
        host = (s.runs[0].get("host") or {}) if s.runs else {}
        params = s.runs[0].get("params") if s.runs else {}
        lines.append(f"- {s.label}: {len(s.runs)} run(s), params {params}, {host.get('cpus')} cpus, "
                     f"load max {max((r.get('host') or {}).get('load_max') or 0 for r in s.runs) if s.runs else '–'}, "
                     f"{host.get('adapter') or 'adapter unknown'}")
    lines += [f"- {s.label}: no run finished, left out" for s in left_out]
    return "\n".join(lines)


def sheets(sides: list[SideRuns], out) -> dict[str, Path]:
    """For each shot every side took, the first run's shots side by side, in the sides' order."""
    from manifoldx.image import read_png, write_png

    def shots(s):
        return {Path(p).stem: Path(p) for p in s.runs[0].get("shots", [])} if s.runs else {}

    per_side = [shots(s) for s in sides]
    names = [n for n in per_side[0] if all(n in x for x in per_side[1:])]
    made = {}
    for name in names:
        imgs = [read_png(x[name])[..., :3] for x in per_side]
        h = max(i.shape[0] for i in imgs)
        w = sum(i.shape[1] for i in imgs) + GAP * (len(imgs) - 1)
        sheet = np.empty((h, w, 3), np.uint8)
        sheet[:] = GAP_RGB
        x = 0
        for i in imgs:
            sheet[: i.shape[0], x: x + i.shape[1]] = i
            x += i.shape[1] + GAP
        path = Path(out) / "sheets" / f"{name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_png(path, sheet)
        made[name] = path
    return made


def write(sides: list[SideRuns], out) -> dict:
    """compare.md and the sheets in `out`; the rows, the table and the sheets' paths. A side with no run
    that finished is left out and named under the table; the first side with runs is the baseline."""
    from dataclasses import asdict

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    kept, left_out = [s for s in sides if s.runs], [s for s in sides if not s.runs]
    rows_ = rows(kept) if kept else []
    text = table(kept, rows_, left_out)
    made = sheets(kept, out) if kept else {}
    sheet_lines = "".join(f"\n- sheet {n}: {p}" for n, p in made.items())
    (out / "compare.md").write_text(text + sheet_lines + "\n")
    return {"table": text, "rows": [asdict(r) for r in rows_], "sheets": {n: str(p) for n, p in made.items()},
            "folder": str(out)}
