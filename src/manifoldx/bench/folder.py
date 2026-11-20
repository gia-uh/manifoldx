"""A run on disk: one folder per run and one line per run in the history.

    <runs>/<UTC time>-<scenario>-<4 hex>/
      run.json      the summary
      frames.npz    per-frame arrays: wall_ms, draw_ms, cpu_ms, phase, split_ms, systems_ms, counters,
                    spans_ms and metrics, each matrix with its column names (split_names, ...)
      events.jsonl  marks, steps, shots and errors, with their times
      shots/*.png   the scenario's shots
"""

from __future__ import annotations

import json
import secrets
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from manifoldx.bench.runner import Run


def write_run(run: Run, runs_dir: Path, history: Path | None) -> Path:
    """Write the run's folder under `runs_dir`, append its summary to `history`, return the folder."""
    from manifoldx.image import write_png

    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S")
    folder = Path(runs_dir) / f"{stamp}-{run.summary['scenario']}-{secrets.token_hex(2)}"
    folder.mkdir(parents=True)
    (folder / "run.json").write_text(json.dumps(run.summary, indent=1))
    np.savez(folder / "frames.npz", **run.frames)
    (folder / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in run.events))
    if run.shots:
        (folder / "shots").mkdir()
        for name, img in run.shots.items():
            write_png(folder / "shots" / f"{name}.png", img)
    if history is not None:
        history = Path(history)
        history.parent.mkdir(parents=True, exist_ok=True)
        with history.open("a") as f:
            f.write(json.dumps({**run.summary, "folder": folder.name}) + "\n")
    return folder


def read_run(folder: Path) -> dict:
    """A run folder's summary, with the paths of its shots."""
    folder = Path(folder)
    summary = json.loads((folder / "run.json").read_text())
    summary["shots"] = sorted(str(p) for p in (folder / "shots").glob("*.png")) if (folder / "shots").exists() else []
    return summary
