---
when: building or debugging an app, game or simulation together with an agent; using `manifoldx look`, `state`, `set`, `emit`, `events` or `log`; giving an app events an agent can fire; following a person who plays
triggers: ["agent", "run --web", "tab: ", "--debug", "metrics", "fps", "frame rate", "memory", "gc", "dev server", "dev page", "manifoldx look", "manifoldx state", "manifoldx set", "manifoldx emit", "manifoldx events", "manifoldx log", "engine.log", "/api/state", "/api/events", "/api/log", "monitor", "follow the player"]
---

# Working with an agent: the dev server of `manifoldx run`

## When to use this

Read this when you build something on manifoldx with an agent in the loop: the agent edits the code, the app reloads in the open window (`know-how/hot-reload.md`), and the agent needs to see what is on screen, read and change the app's state, fire the app's events, and wait for things to happen, while a person may be playing or watching. The code is `src/manifoldx/devchannel.py`; `tests/test_dev_channel.py` shows every behaviour below.

## The loop

1. The person, or the agent, starts the app: `uv run manifoldx run --debug -m mygame.dev` (or `app.py`; without `--debug` it skips the watchers below).
2. `manifoldx run` prints `dev page and API at http://127.0.0.1:8770`. The page shows the frame on screen, the log as it grows, `engine.state`, and a button for every event the app handles.
3. The agent works from a shell in the same folder:
   - `manifoldx look` saves the frame on screen to `.manifoldx/look.png`. Read the image.
   - `manifoldx log -n 30` shows the last lines: what the person did and what the app said.
   - `manifoldx state player` prints part of `engine.state`; `manifoldx set '{"player": {"hp": 1}}'` changes it.
   - `manifoldx events` lists what the app can be asked to do; `manifoldx emit spawn '{"kind": "slime", "near": "player"}'` asks.
   - `manifoldx log --match 'boss defeated' --wait 300` blocks until that line appears, and exits 1 if it did not.
4. The agent edits code, the app reloads, and the agent checks the result with the same commands.

None of this exists outside `manifoldx run`: the same app run with `python app.py` (the shipped game) has no server, and `engine.log` writes nowhere.

## The API

The server listens on 127.0.0.1 only (port 8770, or the next free one; `manifoldx run --port N` picks another). `.manifoldx/server.json` holds its URL, which the commands read. Add `.manifoldx/` to the project's `.gitignore`.

| Request | Does |
|---|---|
| `GET /` | the dev page |
| `GET /api/state` | `engine.state` as JSON; what JSON cannot hold comes as its `repr` |
| `PATCH /api/state` | apply a JSON merge patch (RFC 7386): objects merge, `null` deletes a key, anything else replaces |
| `GET /api/events` | `{event: [first docstring line of each handler]}` |
| `POST /api/events/<name>` | `engine.emit(name, body)`; 404 with the known events when nothing handles it |
| `GET /api/log?after=N&match=RE&wait=S&last=K` | lines after sequence number `N` (or the last `K`, default 50) matching the regex; `wait` blocks up to `S` seconds (at most 300) for one; the answer's `next` is the `after` of the next request |
| `GET /api/look` | the next frame presented, GUI included, as PNG |
| `GET /api/metrics` | under `--debug`: one entry per second for the last 2 minutes (fps, mean and worst frame ms, resident MB, garbage collections per generation and their ms) |

Every request that reads or changes the app runs on the engine's thread between frames, so it never races a system or a handler. While the app is paused on an error, state, set and emit still answer; look waits for a frame and fails with 504 after 5 s.

## `manifoldx run --debug`: the watchers

Plain `manifoldx run` keeps the dev server light: the log has only the runner's lines and the app's own. `--debug` adds what everyone needs to debug performance, with no code in the app:

- `manifoldx metrics -n 30` (and a panel on the page): each second's fps, mean and worst frame time, resident memory, and garbage collections per generation with the time they took.
- Each frame's time split in four (`split_ms`, mean per second): `update` (events, async handlers, systems), `render` (preparing and submitting the GPU work), `acquire` (waiting for the window's next image: vsync, or the GPU behind), `idle` (the rest: the loop between frames). A frame rate stuck at half the screen's refresh with a large `acquire` is vsync missing a refresh; a large `update` is the app's Python.
- The log gets a line every 10 s with the same numbers, a `slow frame N ms` line for each frame over 100 ms, a `slow gc` line for each collection over 20 ms, and the keys pressed.

A slow frame line says only how long; log where the person was from the app (`engine.log`) so the two lines meet in the log. To find what makes frames slow, watch for them: `manifoldx log --match 'slow frame' --wait 120`, then `manifoldx log -n 20` for what happened around them.

**A look costs a little, so ask for one when you need it.** The frame that takes it only queues a copy; the next frame reads it back and a thread encodes the PNG, but each look still costs a GPU copy and a readback. The page's "live frame" box (off by default) asks for one every 1.5 s.

## Against a tab: `manifoldx run --web`

Under `manifoldx run --web` (`know-how/web-export.md`, "While you edit") the app runs in a browser tab, the app's page is `/` and the dev page moves to `/dev`. The same commands work, and the server forwards them to the tab:

- `look` is the tab's frame with the GUI painted over it (the HTML panels' backgrounds and text, drawn into the PNG at the drawing buffer's size), also while the tab is paused or reloading.
- `state`, `set` and `emit` act on the tab's `engine.state` and handlers; emitted events run at the start of the tab's next frame. `events` answers from the Python engine (the handlers' docstrings are there), without `startup`, `shutdown` and manifoldx's own GUI handlers, which the tab does not run.
- `log` holds the runner's lines, the app's `engine.log` lines from frame code, and the tab's console warnings and errors, uncaught errors and unhandled rejections with the prefix `tab: `, sent every 250 ms.
- `metrics` does not exist (404 `metrics are not measured under --web`), and `--debug` is refused.

With no tab open, the forwarded requests answer 503 `no tab is open: open http://127.0.0.1:8770/`; open the URL first (a Playwright page counts). With several tabs open, the one that connected last answers; when it closes, the one before it takes over within 0.25 s. A tab that takes more than 5 s gives 504 `the tab did not answer`, and a tab that fails gives 500 with its error.

The log lines a reload writes, which `manifoldx log --match` can wait on:

```text
reloaded v4 in 2.41 s (state 0.02, setup 0.08, startup 0.97, reach 0.95, snapshot 0.09, transpile 0.03, gui 0.00, scene kept 0.05, write 0.04)
tab: v4 in 84 ms (scene kept)
reload failed, kept v3: app.py:12: SyntaxError: '(' was never closed
paused until the next edit: app.py:26 in tick: TypeError: Cannot read properties of null (reading 'x')
```

The first line is Python's rebuild, by phase (`scene new` in place of `scene kept` when the scene's hash changed; `, reason tab` when a newly opened tab with a newer state caused it), the second the tab's swap. Save to swap is about the two summed plus up to 0.25 s for the poll. A frame error pauses the tab until the next edit and names the Python file, line and function.

## Writing an app an agent can drive

**Log what the person does and sees, once per change.** `engine.log(text)` writes a line to the page, to `/api/log` and to `.manifoldx/log.txt`. Log a place entered, a menu opened, a choice made, a line spoken, a fight starting and ending, a value crossing a threshold. Never log every frame: compare with what you logged last. The runner logs on its own the start, every reload, failed reloads and pauses with their error, and the keys pressed.

```python
if place != last_place:
    engine.log(f"entered {place}")
```

**Keep the state that matters in `engine.state`, as plain data.** It survives reloads, the agent reads it with `manifoldx state`, and a patch changes it. If the app copies state into objects at startup, a patch will not reach them: read `engine.state` where the value is used, or handle an event that applies the change.

**Give every change worth testing an event, with a docstring.** An event goes through the app's own logic, so the world stays consistent: spawning an enemy through `spawn` places it, registers it and starts its behaviour, where a raw state patch would only add a dict entry. The docstring's first line is what `manifoldx events` and the page show: say what the payload holds.

```python
@engine.on("spawn")
def spawn(payload):
    """Spawn a creature: {"kind": name, "near": "player" | [x, z]}."""
    world.spawn(payload["kind"], near=payload.get("near", "player"))
    engine.log(f"spawned {payload['kind']}")
```

Handlers may be async, as any `engine.on` handler: `await engine.delay(2)` inside one plays out over frames.

**Put debug-only events in the dev entry module.** The module `manifoldx run` starts (say `mygame/dev.py`) registers the events a person would never fire in the real game (teleport, heal, give items, start a fight), and the shipped entry point does not import it.

## Following a person who plays

Let the person play while you watch the log, and act on what they say:

- "this looks wrong": `manifoldx look`, then `manifoldx log -n 20` to know where they are and what they just did.
- "tell me when I reach the cave": `manifoldx log --match 'entered cave' --wait 300`, or in a monitor's done condition `manifoldx log --after <next> --match 'entered cave'` (exits 0 once the line exists).
- "give me a harder enemy": `manifoldx emit spawn '{"kind": "boss"}'`.
- "it broke": the log holds `paused until the next edit: <error>` and the terminal holds the traceback; fix the code and save.

## Gotchas

- **The commands need the folder the app was started from**, where `.manifoldx/server.json` is.
- **`emit` answers before the handlers run**: they run at the start of the next frame. Wait for their log line, or read the state after it.
- **A merge patch replaces lists whole.** To add to a list, read it, change it and patch the whole list, or use an event.
- **Values that are not JSON** (numpy arrays, the app's objects) come back as `repr` and cannot be patched back; keep plain data in the state.
- **No authentication**: anyone on the machine can call the API while `manifoldx run` runs. It never listens beyond 127.0.0.1.
