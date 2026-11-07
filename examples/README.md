# examples/ — focused feature examples

Each file demonstrates **one** engine feature with the least code possible.
These are **documentation**: read one to learn how a single capability works.

Naming: files are named after the feature (`shadows.py`, `fields.py`,
`booleans.py`), not `*_demo.py`.

For multi-system, game-like / sim-like showcases, see [`../demos/`](../demos).

Run one:

```bash
uv run python examples/<name>.py                                   # interactive
uv run python examples/<name>.py --render --duration 3 --output /tmp/<name>.mp4
```

## Exported to the web

Two examples are written for `manifoldx export --web` (`know-how/web-export.md`), and their
replays in `tests/web/` check that the page runs them as Python does:

- `web_walk.py`: walk a yard in first person and pick boxes; plain Python per frame.
- `web_crowd.py`: rings of boxes spawned, drifted, destroyed and picked by query systems;
  the entity store in the page end to end.

```bash
uv run manifoldx export --web examples/web_crowd.py out/ && cd out && python -m http.server 8000
```
