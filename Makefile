.PHONY: help test format lint run clean install bench docs docs-test docs-check \
	native-check native-build native-verify

help:
	@echo "ManifoldX Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make install   - Install dependencies with uv"
	@echo "  make test     - Run tests with pytest"
	@echo "  make format   - Format code with ruff"
	@echo "  make lint     - Lint code with ruff"
	@echo "  make run      - Run an example (EXAMPLE=examples.pbr_demo)"
	@echo "  make bench    - Run benchmark suites (ARGS=\"frame --against main\")"
	@echo "  make docs     - Render the guide to docs/site with every block executed"
	@echo "  make docs-check - The docs gate: render, tangle, examples, rift coverage"
	@echo "  make native-check  - fmt, clippy and tests for the workspace crates"
	@echo "  make native-build  - Build the manifold-gfx wheel with maturin"
	@echo "  make native-verify - Build, install in a fresh venv, smoke, negative control"
	@echo "  make clean    - Clean build artifacts"

install:
	uv sync --all-extras

test:
	uv run pytest tests/ -v --tb=short

format:
	uv run ruff format src/

lint:
	uv run ruff check src/

bench:
	uv run python -m benchmarks $(ARGS)

# The native-wheel gate (migration phase 0). NATIVE_PYTHON selects the
# interpreter the wheel is built for and installed against (python3.14 for
# the GIL build, python3.14t for the free-threaded one); NATIVE_DIST is the
# wheel output directory. CARGO_TARGET_DIR defaults in-tree; point it at a
# disk with room when the checkout's filesystem is tight.
NATIVE_PYTHON ?= python3.14
NATIVE_DIST ?= dist

native-check:
	cargo fmt --check
	cargo clippy --workspace --all-targets -- -D warnings
	cargo test --workspace

native-build:
	uv run --group native --locked maturin build --release -i $(NATIVE_PYTHON) --out $(NATIVE_DIST)

# Build the wheel, install it in a fresh venv, run the smoke gate from
# outside the checkout, then the negative control: with the extension
# renamed inside that disposable venv the smoke must fail, and restored it
# must pass again.
native-verify: native-build
	bash scripts/gates/native-verify.sh $(NATIVE_PYTHON) $(NATIVE_DIST)

run:
	@if [ -z "$(EXAMPLE)" ]; then \
		echo "Usage: make run EXAMPLE=examples.pbr_demo"; \
	else \
		uv run python -m $(EXAMPLE); \
	fi

clean:
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

docs:
	uv run --group docs scriptorium render docs/scriptorium.yaml --only site

# docs/examples/ matches the chapters (checked before rendering, which rewrites
# it), every block ran (scriptorium prints a failing block's traceback and
# exits 0), and every example runs on its own.
docs-test:
	for c in docs/chapters/*.md; do uv run --group docs scriptorium tangle --test $$c -d docs || exit 1; done
	for e in docs/examples/*.py; do [ $$(basename $$e) = __init__.py ] || grep -q "export=$${e#docs/}" docs/chapters/*.md || { echo "$$e is not tangled from the docs"; exit 1; }; done
	$(MAKE) docs
	! grep -l 'Traceback (most recent call last)' docs/site/*.html
	for e in docs/examples/*.py; do uv run python $$e > /dev/null || exit 1; done

# The release gate: docs-test, and every public class named in the chapters.
docs-check: docs-test
	uvx --from git+https://github.com/apiad/rift@v0.9.0 rift check .
