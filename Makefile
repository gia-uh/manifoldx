.PHONY: help test format lint run clean install bench docs docs-test docs-check

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
