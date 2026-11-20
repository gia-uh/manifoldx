"""A process worker of a streamer ends when its app is killed, instead of living on as an orphan."""

import os
import signal
import subprocess
import sys
import textwrap
import time

APP = textwrap.dedent("""
    import os, sys, time
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    from manifoldx.streaming import _worker_start

    if __name__ == "__main__":
        pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"),
                                   initializer=_worker_start, initargs=(os.getpid(), None))
        print(pool.submit(os.getpid).result(), flush=True)
        time.sleep(60)
""")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    with open(f"/proc/{pid}/stat") as f:
        return f.read().split()[2] != "Z"


def test_a_worker_exits_when_its_app_is_killed(tmp_path):
    script = tmp_path / "app.py"
    script.write_text(APP)
    app = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, text=True)
    worker = int(app.stdout.readline())
    assert _alive(worker)
    app.send_signal(signal.SIGKILL)
    app.wait()
    deadline = time.monotonic() + 5
    while _alive(worker) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not _alive(worker)
