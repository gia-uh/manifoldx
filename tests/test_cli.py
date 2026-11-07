"""manifoldx run's arguments: --web refuses --debug (the tab runs the frames, Python measures
none), and run with neither a script nor a module is a usage error with or without --web."""

import pytest

from manifoldx.cli import main


def test_run_web_with_debug_exits_2(capsys):
    with pytest.raises(SystemExit) as e:
        main(["run", "--web", "--debug", "app.py"])
    assert e.value.code == 2
    assert (
        "--debug measures the window's frames; under --web the tab runs them"
        in capsys.readouterr().err
    )


@pytest.mark.parametrize("extra", [[], ["--web"]], ids=["run", "run --web"])
def test_run_without_script_or_module_is_a_usage_error(capsys, extra):
    with pytest.raises(SystemExit) as e:
        main(["run", *extra])
    assert e.value.code == 2
    assert "give app.py or -m module" in capsys.readouterr().err
