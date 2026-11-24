# Shared backend parity: `make parity-fast`

`make parity-fast` runs the same three allocator cases against the installed
Python `EntityStore` and the compiled production `manifoldx._core_native`
extension. It is a focused private-slice check, not the full ECS, and it does
not activate the Engine runtime.

The cases are fixed capacity and LIFO reuse, overflow without consuming slots,
and disjoint reservations that stay invisible until commit. The test file is
`tests/core/test_backend.py`; it asserts that the package and extension resolve
inside the selected interpreter's prefix, that the extension is loaded by
`ExtensionFileLoader`, and that `_EntitySlots` is the native class.

## Run it

Use an interpreter that already has the built wheel and pytest installed. The
target builds nothing, installs nothing and runs no `uv sync`.

```bash
make parity-fast \
  PARITY_PYTHON=/var/tmp/my-venv/bin/python \
  TESTS=tests/core/test_backend.py
```

`PARITY_PYTHON` defaults to `python3.14t`; `TESTS` defaults to the suite. The
target unsets `PYTHONPATH` and `PYTHONHOME`, runs the interpreter with `-I` and
pytest's importlib mode, and changes to a disposable directory under `TMPDIR`
(or `/var/tmp`). Absolute test paths and `--confcutdir` keep the checkout's
conftest out of the run.

## Missing extension control

Rename the extension inside the disposable venv, run the same command, and
restore it:

```bash
so=$(/var/tmp/my-venv/bin/python -c 'import manifoldx._core_native as m; print(m.__file__)')
mv "$so" "$so.parity-removed"
make parity-fast PARITY_PYTHON=/var/tmp/my-venv/bin/python TESTS=tests/core/test_backend.py
# expected: failure in test_native_extension_is_installed_compiled_and_native
mv "$so.parity-removed" "$so"
make parity-fast PARITY_PYTHON=/var/tmp/my-venv/bin/python TESTS=tests/core/test_backend.py
```

There is no skip and no Python-module fallback. A `.py` stub also fails the
compiled-loader assertion.

## Test selection

The module carries pytest's existing `slow` marker, so `make test` and the
ordinary Python-only suite deselect it. The `Native` workflow runs it as a
dedicated step against the wheel that job already built, and keeps the
packaging step from collecting it. This is selection by marker, not a skip, and
a missing extension in the dedicated step fails the job.
