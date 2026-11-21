import multiprocessing as mp

import pytest

from manifoldx.audio.controls import Controls


def _read(spec, q):
    c = Controls.attach(spec)
    q.put(c.get("rpm"))
    c.close()


def test_a_value_set_here_is_read_in_another_process():
    c = Controls.create(["rpm", "throttle"])
    c.set("rpm", 2400.0)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_read, args=(c.spec(), q))
    p.start()
    p.join(30)
    assert q.get(timeout=1) == 2400.0
    c.close()
    c.unlink()


def test_tick_keeps_the_previous_value():
    c = Controls.create(["rpm"])
    c.set("rpm", 1.0)
    c.tick()
    c.set("rpm", 2.0)
    assert (c.prev("rpm"), c.get("rpm")) == (1.0, 2.0)
    c.close()
    c.unlink()


def test_an_unknown_name_raises():
    c = Controls.create(["rpm"])
    with pytest.raises(KeyError):
        c.get("speed")
    c.close()
    c.unlink()


def test_values_start_at_zero_and_names_are_kept():
    c = Controls.create(["a", "b"])
    assert c.get("a") == 0.0 and c.names == ("a", "b") and c.spec()["names"] == ["a", "b"]
    c.close()
    c.unlink()
