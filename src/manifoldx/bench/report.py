"""report.html: one page per run folder that a person opens to see what the run did.

The page holds the run line, the checks, per phase the frame table, a frame-time chart, the split, the
slowest systems and spans, the counters, the streamer, the workers, memory, the app's metrics and the
shots. It is one file: inline CSS, charts as inline SVG, shots as PNG data, no script and no request, so
it opens alone when sent. Every number is in the markup as text. Older run folders get a page too:
whatever a schema lacks is left out.
"""

from __future__ import annotations

import base64
import json
from html import escape
from pathlib import Path

import numpy as np

CSS = """
:root{--bg:#11100e;--surface:#191814;--rule:#2a2721;--ink:#dcd9cf;--strong:#f1ede2;--muted:#8a867b;
--accent:#e0a872;--ok:#9db07e;--err:#d9806f;--comms:#9fb4d0;
--mono:ui-monospace,'JetBrains Mono',Menlo,Consolas,monospace;--ui:system-ui,'Inter',sans-serif}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 var(--ui)}
main{max-width:1040px;margin:0 auto;padding:20px 16px 60px}
h1{color:var(--strong);font-size:22px;margin:0 0 6px}
h2{color:var(--strong);font-size:17px;margin:32px 0 8px;border-bottom:1px solid var(--rule);padding-bottom:4px}
h3{color:var(--muted);font-size:14px;margin:18px 0 6px;font-weight:600}
.kv{display:grid;grid-template-columns:max-content 1fr;gap:2px 14px;font-size:14px}
.kv b{color:var(--muted);font-weight:500}
.t{overflow-x:auto;margin:6px 0}
table{border-collapse:collapse;font:13px/1.4 var(--mono)}
th,td{padding:3px 10px;border-bottom:1px solid var(--rule);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{color:var(--muted);font-weight:500}
td.msg{text-align:left;white-space:normal}
.pass{color:var(--ok)}.fail{color:var(--err)}.skip{color:var(--muted)}
.err{color:var(--err);font-family:var(--mono);font-size:13px;white-space:pre-wrap}
svg{width:100%;height:auto;display:block;background:var(--surface);border-radius:6px;margin:6px 0}
svg text{font:11px var(--mono);fill:var(--muted)}
.shots{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px}
figure{margin:0}figure img{max-width:100%;height:auto;display:block;border-radius:6px}
figcaption{font:12px var(--mono);color:var(--muted)}
.muted{color:var(--muted)}
@media (max-width:640px){main{padding:14px 10px 40px}h1{font-size:19px}.kv{grid-template-columns:1fr}
.kv b{margin-top:6px}}
"""

BAND = ("#e0a872", "#9fb4d0", "#9db07e", "#d9806f", "#b59fd0", "#d0c49f")  # phase colours, in order


def _e(v) -> str:
    return escape("" if v is None else str(v))


def _n(v, fmt=".1f") -> str:
    if v is None:
        return "-"
    if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
        return str(int(v))
    try:
        return format(float(v), fmt)
    except (TypeError, ValueError):
        return _e(v)


def _table(head: list[str], rows: list[list[str]]) -> str:
    """A table of cells already escaped."""
    if not rows:
        return ""
    h = "".join(f"<th>{c}</th>" for c in head)
    body = "".join("<tr>" + "".join(c if c.startswith("<td") else f"<td>{c}</td>" for c in r) + "</tr>"
                   for r in rows)
    return f'<div class="t"><table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table></div>'


def _line(xs, ys, w: int, h: int, bands: list) -> str:
    """A polyline of ys over xs; bands are (x0, x1, name, colour) shaded behind it; dashed lines at the
    median and 1.5 times it. The axis stops at 5 times the median, so a warm-up frame does not flatten the
    rest: a frame above it is drawn at the top and counted in a label."""
    ys = np.asarray(ys, dtype=float)
    if not len(ys):
        return ""
    pad_l, pad_b = 46, 18
    med = float(np.nanmedian(ys))
    peak = float(np.nanmax(ys))
    top = min(peak * 1.08, 5 * med) or 1.0
    above = int((ys > top).sum())
    n = max(len(ys) - 1, 1)

    def X(i):
        return pad_l + (w - pad_l - 6) * i / n

    def Y(v):
        return 6 + (h - pad_b - 6) * (1 - min(v, top) / top)

    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="frame time over frames">']
    for x0, x1, name, colour in bands:
        out.append(f'<rect x="{X(x0):.1f}" y="6" width="{max(X(x1) - X(x0), 1):.1f}" height="{h - pad_b - 6}" '
                   f'fill="{colour}" opacity="0.08"/>')
        if X(x1) - X(x0) > 8 * len(name) + 8:
            out.append(f'<text x="{X(x0) + 4:.1f}" y="30">{_e(name)}</text>')
    for v, label in ((med, "median"), (1.5 * med, "1.5x")):
        if v <= top:
            out.append(f'<line x1="{pad_l}" x2="{w - 6}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="#8a867b" '
                       f'stroke-dasharray="4 4" stroke-width="1"/><text x="{w - 8}" y="{Y(v) - 3:.1f}" '
                       f'text-anchor="end">{label} {v:.1f}</text>')
    pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in zip(xs, ys) if np.isfinite(v))
    out.append(f'<polyline points="{pts}" fill="none" stroke="#e0a872" stroke-width="1.5"/>')
    if above:
        out.append(f'<text x="{w - 8}" y="18" text-anchor="end">{above} above the axis, the slowest {peak:.1f} ms</text>')
    out.append(f'<text x="2" y="16">{top:.1f} ms</text><text x="{pad_l}" y="{h - 4}">frame 0</text>'
               f'<text x="{w - 6}" y="{h - 4}" text-anchor="end">frame {len(ys) - 1}</text></svg>')
    return "".join(out)


def _bars(parts: dict[str, dict[str, float]], w: int = 640) -> str:
    """One stacked bar per row (a phase): its parts' means in ms, each labelled with its value."""
    rows = [(k, {n: v for n, v in p.items() if v and v > 0}) for k, p in parts.items()]
    rows = [(k, p) for k, p in rows if p]
    if not rows:
        return ""
    names = sorted({n for _, p in rows for n in p})
    colour = {n: BAND[i % len(BAND)] for i, n in enumerate(names)}
    top = max(sum(p.values()) for _, p in rows)
    left, bar = 90, 22
    h = 10 + len(rows) * (bar + 10) + 24
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="frame split by phase">']
    for r, (k, p) in enumerate(rows):
        y, x = 10 + r * (bar + 10), float(left)
        out.append(f'<text x="2" y="{y + 15}">{_e(k)}</text>')
        for n in names:
            if n not in p:
                continue
            bw = (w - left - 8) * p[n] / top
            out.append(f'<rect x="{x:.1f}" y="{y}" width="{bw:.1f}" height="{bar}" fill="{colour[n]}" opacity="0.8">'
                       f'<title>{_e(n)} {p[n]:.2f} ms</title></rect>')
            if bw > 34:
                out.append(f'<text x="{x + 3:.1f}" y="{y + 15}" style="fill:#11100e">{p[n]:.1f}</text>')
            x += bw
    lx = 2
    for n in names:
        out.append(f'<rect x="{lx}" y="{h - 16}" width="10" height="10" fill="{colour[n]}"/>'
                   f'<text x="{lx + 14}" y="{h - 7}">{_e(n)}</text>')
        lx += 22 + 7 * len(n)
    out.append("</svg>")
    return "".join(out)


def _dots(points: list[tuple[float, int]], w: int = 640, h: int = 140) -> str:
    """One dot per (time in s, level): when each node was first drawn."""
    if not points:
        return ""
    t1 = max(t for t, _ in points) or 1.0
    levels = max(lv for _, lv in points) + 1
    left = 40

    def Y(lv):
        return 10 + (h - 34) * (lv + 0.5) / levels

    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="nodes drawn over time by level">']
    for lv in range(levels):
        out.append(f'<text x="2" y="{Y(lv) + 4:.1f}">L{lv}</text>')
    for t, lv in points:
        out.append(f'<circle cx="{left + (w - left - 8) * t / t1:.1f}" cy="{Y(lv):.1f}" r="2.5" fill="#9fb4d0"/>')
    out.append(f'<text x="{left}" y="{h - 6}">0 s</text><text x="{w - 6}" y="{h - 6}" text-anchor="end">'
               f'{t1:.2f} s</text></svg>')
    return "".join(out)


def _phase_rows(phases: dict) -> list[list[str]]:
    return [[f"<td>{_e(name)}</td>", _n(ph.get("frames")), _n(ph.get("p50_ms")), _n(ph.get("p95_ms")),
             _n(ph.get("p99_ms")), _n(ph.get("max_ms")), _n(ph.get("draw_p50_ms")), _n(ph.get("cpu_p50_ms")),
             _n(ph.get("over_1_5x")), _n(ph.get("over_3x"))] for name, ph in phases.items()]


def _run_line(s: dict) -> str:
    host = s.get("host", {})
    size = s.get("size") or ["?", "?"]
    adapter = f"{host.get('adapter')}{' (software)' if host.get('software') else ''}"
    items = [("scenario", s.get("scenario")), ("params", json.dumps(s.get("params", {}))),
             ("loop", s.get("loop")), ("size", f"{size[0]}x{size[1]}"), ("smoke", s.get("smoke")),
             ("adapter", adapter), ("load", f"{host.get('load_before')} to {host.get('load_max')} "
                                           f"on {host.get('cpus')} CPUs"),
             ("python", f"{host.get('python')}{' free-threaded' if host.get('free_threaded') else ''}"),
             ("label", s.get("label")), ("versions", ", ".join(f"{k} {v}" for k, v in s.get("versions", {}).items())),
             ("started", s.get("started")), ("seconds", s.get("seconds")), ("schema", s.get("schema", 1))]
    out = '<div class="kv">' + "".join(f"<b>{k}</b><span>{_e(v)}</span>" for k, v in items) + "</div>"
    if s.get("error"):
        out += f'<p class="err">error: {_e(s["error"])}</p>'
    return out


def _checks(s: dict) -> str:
    cs = s.get("checks")
    if cs is None:
        return '<p class="muted">This run is older than checks.</p>'
    if not cs:
        return '<p class="muted">This scenario has no checks.</p>'
    rows = []
    for c in cs:
        st = "skip" if c.get("skipped") else "pass" if c.get("ok") else "fail"
        rows.append([f'<td class="{st}">{st}</td>', f"<td>{_e(c.get('phase'))}</td>",
                     f'<td class="msg">{_e(c.get("message"))}</td>'])
    return _table(["", "phase", "check"], rows)


def _frames(s: dict, frames: dict) -> str:
    phases = s.get("phases", {})
    out = _table(["phase", "frames", "p50 ms", "p95 ms", "p99 ms", "max ms", "draw p50", "cpu p50", "over 1.5x",
                  "over 3x"], _phase_rows(phases))
    wall = frames.get("wall_ms")
    if wall is not None and len(wall):
        order = s.get("phase_order", list(phases))
        ph = np.asarray(frames.get("phase", np.zeros(len(wall))), dtype=int)
        bands, start = [], 0
        for i in range(1, len(ph) + 1):
            if i == len(ph) or ph[i] != ph[start]:
                name = order[ph[start]] if ph[start] < len(order) else str(ph[start])
                bands.append((start, i - 1, name, BAND[ph[start] % len(BAND)]))
                start = i
        out += _line(np.arange(len(wall)), wall, 640, 220, bands)
    return out


def _probes(s: dict) -> str:
    phases = s.get("phases", {})
    out = []
    split = {k: ph.get("split_ms", {}) for k, ph in phases.items()}
    if any(split.values()):
        out.append("<h3>split: mean ms per frame</h3>" + _bars(split))
        out.append(_table(["phase", *sorted({n for p in split.values() for n in p})],
                          [[f"<td>{_e(k)}</td>", *[_n(p.get(n), ".2f") for n in sorted({n for q in split.values()
                                                                                          for n in q})]]
                           for k, p in split.items()]))
    for key, title in (("systems_ms", "slowest systems: mean ms"), ("spans_ms", "spans: mean ms")):
        rows = [[f"<td>{_e(k)}</td>", f"<td>{_e(n)}</td>", _n(v, ".3f")] for k, ph in phases.items()
                for n, v in (ph.get(key) or {}).items()]
        if rows:
            out.append(f"<h3>{title}</h3>" + _table(["phase", "name", "ms"], rows))
    rows = [[f"<td>{_e(k)}</td>", f"<td>{_e(n)}</td>", _n(v.get("mean"), "g"), _n(v.get("max"))]
            for k, ph in phases.items() for n, v in (ph.get("counters") or {}).items()]
    if rows:
        out.append("<h3>counters per frame</h3>" + _table(["phase", "counter", "mean", "max"], rows))
    rows = [[f"<td>{_e(k)}</td>", f"<td>{_e(n)}</td>", _n(v.get("jobs")), _n(v.get("p50_ms")), _n(v.get("max_ms")),
             _n(v.get("processes"))] for k, ph in phases.items() for n, v in (ph.get("workers") or {}).items()]
    if rows:
        out.append("<h3>workers</h3>" + _table(["phase", "job", "jobs", "p50 ms", "max ms", "processes"], rows))
    return "".join(out)


def _streamer(s: dict, events: list) -> str:
    phases = s.get("phases", {})
    out = []
    rows = [[f"<td>{_e(k)}</td>", f"<td>{_e(n)}</td>", f"<td>{_e(v)}</td>"] for k, ph in phases.items()
            for n, v in (ph.get("streamer") or {}).items() if not isinstance(v, (dict, list))]
    if rows:
        out.append(_table(["phase", "", "value"], rows))
    nodes = [e for e in events if e.get("kind") == "node"]
    if nodes:
        by: dict = {}
        for e in nodes:
            by.setdefault(e.get("level", 0), {}).setdefault(e.get("ev"), 0)
            by[e.get("level", 0)][e.get("ev")] += 1
        kinds = sorted({k for d in by.values() for k in d if k})
        out.append("<h3>node events by level</h3>" + _table(
            ["level", *kinds], [[f"<td>L{_e(lv)}</td>", *[_n(by[lv].get(k, 0)) for k in kinds]] for lv in sorted(by)]))
        drawn = [(float(e["t"]), int(e.get("level", 0))) for e in nodes if e.get("ev") == "drawn" and "t" in e]
        if drawn:
            out.append("<h3>nodes drawn over time</h3>" + _dots(drawn))
    return "".join(out)


def _memory(s: dict) -> str:
    m = s.get("memory") or {}
    out = []
    rows = []
    rss, gpu, gc = m.get("rss_mb") or {}, m.get("gpu_mb") or {}, m.get("gc") or {}
    if rss:
        rows.append(["<td>RSS MB</td>", _n(rss.get("start")), _n(rss.get("max")), _n(rss.get("end"))])
    if gpu:
        rows.append(["<td>GPU MB</td>", _n(gpu.get("start"), ".2f"), _n(gpu.get("max"), ".2f"), _n(gpu.get("end"), ".2f")])
    for name, f in (m.get("folders_mb") or {}).items():
        rows.append([f"<td>{_e(name)} MB</td>", _n(f.get("start"), ".3f"), "-", _n(f.get("end"), ".3f")])
    if rows:
        out.append(_table(["", "start", "max", "end"], rows))
    if gc:
        out.append(f'<p>GC: {_e(gc.get("runs"))} runs by generation, {_n(gc.get("ms"), ".2f")} ms in all, the longest '
                   f'{_n(gc.get("max_ms"), ".2f")} ms.</p>')
    rows = [[f"<td>{_e(k)}</td>", _n(ph.get("rss_max_mb")), _n(ph.get("gc_ms"), ".2f"), _n(ph.get("origin_max_m"), ".2f")]
            for k, ph in (s.get("phases") or {}).items() if any(x in ph for x in ("rss_max_mb", "gc_ms", "origin_max_m"))]
    if rows:
        out.append("<h3>per phase</h3>" + _table(["phase", "RSS max MB", "GC ms", "origin max m"], rows))
    return "".join(out)


def _app(s: dict) -> str:
    rows = [[f"<td>{_e(n)}</td>", f"<td>{_e(v)}</td>"] for n, v in (s.get("app") or {}).items()]
    per = [[f"<td>{_e(k)}</td>", f"<td>{_e(n)}</td>", f"<td>{_e(v.get('last'))}</td>", _n(v.get("mean"), "g"),
            _n(v.get("max"), "g")] for k, ph in (s.get("phases") or {}).items() for n, v in (ph.get("app") or {}).items()]
    return (_table(["metric", "last"], rows) + _table(["phase", "metric", "last", "mean", "max"], per)) if rows or per else ""


def _shots(folder: Path) -> str:
    shots = sorted((folder / "shots").glob("*.png")) if (folder / "shots").exists() else []
    figs = [f'<figure><img alt="{_e(p.stem)}" src="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}">'
            f"<figcaption>{_e(p.name)}</figcaption></figure>" for p in shots]
    return f'<div class="shots">{"".join(figs)}</div>' if figs else ""


def write_report(folder) -> Path:
    """Write <folder>/report.html from the run folder and return its path."""
    folder = Path(folder)
    s = json.loads((folder / "run.json").read_text())
    frames = dict(np.load(folder / "frames.npz")) if (folder / "frames.npz").exists() else {}
    events = []
    if (folder / "events.jsonl").exists():
        events = [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines() if line.strip()]
    sections = [("checks", _checks(s)), ("frames", _frames(s, frames)), ("where the frame goes", _probes(s)),
                ("streamer", _streamer(s, events)), ("memory", _memory(s)), ("app metrics", _app(s)),
                ("shots", _shots(folder))]
    body = "".join(f"<h2>{t}</h2>{html}" for t, html in sections if html)
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>{_e(s.get('scenario'))} · manifoldx bench</title><style>{CSS}</style></head><body><main>"
            f"<h1>{_e(s.get('scenario'))}</h1>{_run_line(s)}{body}"
            f'<p class="muted">{_e(folder.name)}</p></main></body></html>')
    out = folder / "report.html"
    out.write_text(page)
    return out
