"""Build a pre-core and a candidate wheel on one CI runner and time the same checks."""

import json
import os
import platform
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path
from time import perf_counter

ROOT = Path.cwd()
EVIDENCE = ROOT / "wheel-evidence"
STARTUP = """
import sys
from pathlib import Path
import manifoldx
from manifoldx import Engine

assert Path(manifoldx.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
engine = Engine("wheel startup", max_entities=16)
assert engine.store.max_entities == 16 and engine._device is None
"""


def measure(source, output, label):
    env = os.environ.copy()
    env["MANIFOLDX_CORE"] = "python"
    # Start each workload cold and keep cache selection independent of runner setup.
    with tempfile.TemporaryDirectory() as tmp, (EVIDENCE / f"{label}.log").open("w") as log:
        work = Path(tmp)
        env["UV_CACHE_DIR"] = str(work / "cache")
        venv = work / "venv"
        scripts = venv / ("Scripts" if sys.platform == "win32" else "bin")
        suffix = ".exe" if sys.platform == "win32" else ""
        python = scripts / f"python{suffix}"

        def run(*command):
            print(subprocess.list2cmdline([str(arg) for arg in command]), file=log, flush=True)
            result = subprocess.run(command, cwd=work, stdout=log, stderr=log, env=env)
            print(f"exit: {result.returncode}", file=log, flush=True)
            result.check_returncode()

        start = perf_counter()
        run("uv", "build", "--python", sys.executable, "--wheel", source, "--out-dir", output)
        build_seconds = perf_counter() - start
        (wheel,) = output.glob("manifold_gfx-*.whl")
        run("uv", "venv", "--python", sys.executable, venv)
        run("uv", "pip", "install", "--python", python, wheel)
        run(python, "-I", "-c", STARTUP)
        run(scripts / f"manifoldx{suffix}", "--help")
        with wheel.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        return {
            "wheel": wheel.name,
            "sha256": digest,
            "bytes": wheel.stat().st_size,
            "build_seconds": build_seconds,
            "workload_seconds": perf_counter() - start,
        }


pre_core = measure(ROOT / "pre-core", EVIDENCE / "pre-core", "pre-core")
candidate = measure(ROOT, ROOT / "dist", "candidate")
candidate["source_sha"] = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
).strip()
candidate["source_tree"] = subprocess.check_output(
    ["git", "rev-parse", "HEAD^{tree}"], cwd=ROOT, text=True,
).strip()
report = {
    "python": sys.version,
    "platform": platform.platform(),
    "runner_image": os.environ.get("ImageVersion"),
    "pre_core": pre_core,
    "candidate": candidate,
    "size_ratio": candidate["bytes"] / pre_core["bytes"],
    "workload_ratio": candidate["workload_seconds"] / pre_core["workload_seconds"],
}
(EVIDENCE / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
