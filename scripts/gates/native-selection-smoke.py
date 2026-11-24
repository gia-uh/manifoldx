"""Run only with an installed wheel, from the verifier's disposable cwd."""

import os
import sys

from manifoldx import Engine
from manifoldx._core import _selection
import manifoldx.engine as engine_module

state = sys.argv[1]
for value in (None, "", "python"):
    if value is None:
        os.environ.pop("MANIFOLDX_CORE", None)
    else:
        os.environ["MANIFOLDX_CORE"] = value
    engine = Engine("selection", max_entities=16)
    assert engine._device is None
    assert engine.store.max_entities == 16
    assert "manifoldx._core_native" not in sys.modules

os.environ["MANIFOLDX_CORE"] = "rust"


def reject_python_initialization(*args, **kwargs):
    raise AssertionError("Rust selection reached Python initialization")


engine_module._warm_wgpu_types = reject_python_initialization
engine_module.EntityStore = reject_python_initialization
if state == "installed":
    name, bootstrap = _selection()
    assert name == "rust" and bootstrap["runtime"] == "not implemented", bootstrap
    expected, message = NotImplementedError, "native Engine runtime is not implemented"
else:
    assert state == "missing", state
    expected, message = ImportError, "requires manifoldx._core_native"

engine = Engine.__new__(Engine)
try:
    engine.__init__("selection")
except expected as exc:
    assert message in str(exc), str(exc)
    if state == "missing":
        assert isinstance(exc.__cause__, ModuleNotFoundError)
    print(f"rust {state}: {type(exc).__name__}: {exc}")
else:
    raise AssertionError("Rust selection constructed a Python Engine")
assert vars(engine) == {}, vars(engine)
print("Python startup usable; Rust rejected before Engine state initialization")
