"""The opt-in pytest plugin: each scenario is a test, its checks its assertions."""

import textwrap

import manifoldx.bench.plugin  # noqa: F401  imported here once: an in-process pytester run drops what it imports

pytest_plugins = ["pytester"]

SCENARIOS = textwrap.dedent('''
    import numpy as np
    from manifoldx import bench
    from manifoldx.components import Material, Mesh, Transform
    from manifoldx.resources import StandardMaterial, cube


    class Demo:
        def __init__(self, engine):
            engine.camera.fit(10)
            engine.spawn(Mesh(cube(1, 1, 1)), Material(StandardMaterial("#ff8800")),
                         Transform(pos=np.zeros((1, 3), np.float32)), n=1)


    @bench.scenario("ok_one", sizes={"smoke": 2})
    def ok_one(engine, p):
        return Demo(engine), [bench.hold(frames=5)]


    @bench.scenario("tight_one", sizes={"smoke": 2})
    def tight_one(engine, p):
        return Demo(engine), [bench.hold(frames=5)]


    tight_one.check(p95_ms=0.001)
''')


def _project(pytester):
    pytester.makepyprojecttoml(textwrap.dedent('''
        [project]
        name = "plugdemo"
        [tool.manifoldx.bench]
        scenarios = "plug_scenarios"
        [tool.pytest.ini_options]
        testpaths = ["plug_scenarios.py"]
    '''))
    pytester.makepyfile(plug_scenarios=SCENARIOS)
    pytester.syspathinsert()


def test_nothing_is_collected_without_the_plugin(pytester):
    _project(pytester)
    r = pytester.runpytest("--collect-only", "-q")
    assert "test_bench" not in r.stdout.str()


def test_each_scenario_is_a_test(pytester):
    _project(pytester)
    r = pytester.runpytest("-p", "manifoldx.bench.plugin", "-q")
    r.assert_outcomes(passed=1, failed=1)
    assert "check" in r.stdout.str() and "p95_ms" in r.stdout.str()


def test_one_id_runs_one_scenario(pytester):
    _project(pytester)
    r = pytester.runpytest("-p", "manifoldx.bench.plugin", "-q", "plug_scenarios.py::test_bench[ok_one]")
    r.assert_outcomes(passed=1)
