"""The manifoldx command.

`manifoldx run` runs an app and reloads it in the same window whenever its source changes;
with --web, in a browser tab instead (manifoldx.web.dev).
`manifoldx export` runs an app and writes its scene to glTF instead of a window; with --web, a
folder the browser runs (manifoldx.web.export).
`manifoldx look`, `state`, `set`, `emit`, `events` and `log` talk to the dev server of the app
running under `manifoldx run` in this folder (see manifoldx.devchannel).

    manifoldx run app.py [-- <app args>]
    manifoldx run -m package.module [-- <app args>]
    manifoldx run --web [-m package.module | app.py] [--port N] [-- <app args>]
    manifoldx export app.py scene.glb [--strict] [-- <app args>]
    manifoldx export -m package.module scene.glb [--strict] [-- <app args>]
    manifoldx export --web [-m package.module | app.py] OUTDIR [-- <app args>]
    manifoldx look | events | state [key.path] | set '<merge patch>' | emit <event> ['<payload>']
    manifoldx log [-n N] [--after SEQ] [--match REGEX] [--wait S]
    manifoldx metrics [-n N]
"""

import argparse
import os
import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    app_args = []
    if "--" in argv:
        cut = argv.index("--")
        argv, app_args = argv[:cut], argv[cut + 1:]
    parser = argparse.ArgumentParser(prog="manifoldx")
    sub = parser.add_subparsers(dest="command", required=True)
    ex = sub.add_parser("export", help="run an app and write its scene to a .glb instead of opening a window")
    ex.add_argument("-m", dest="module", help="run this module as __main__, like python -m")
    ex.add_argument("paths", nargs="+", metavar="[app.py] out.glb|OUTDIR")
    ex.add_argument("--strict", action="store_true", help="exit with code 1 and write nothing if anything is left out")
    ex.add_argument("--web", action="store_true",
                    help="write a folder the browser runs: the transpiled app, its state and its scene")
    run = sub.add_parser("run", help="run an app and reload it in the same window when its source changes")
    run.add_argument("-m", dest="module", help="run this module as __main__, like python -m")
    run.add_argument("script", nargs="?", metavar="app.py")
    run.add_argument("--port", type=int, default=None, help="the dev server's port (default 8770 or the next free)")
    run.add_argument("--debug", action="store_true",
                     help="also measure each frame, memory and garbage collection, and log slow ones and keys")
    run.add_argument("--web", action="store_true",
                     help="run the app in a browser tab and swap each save into it, keeping engine.state")
    sub.add_parser("look", help="save the frame on screen to .manifoldx/look.png and print its path")
    st = sub.add_parser("state", help="print engine.state as JSON")
    st.add_argument("path", nargs="?", help="a dotted key path inside the state, e.g. player.x")
    se = sub.add_parser("set", help="apply a JSON merge patch to engine.state (null deletes a key)")
    se.add_argument("patch", help='e.g. \'{"player": {"x": 3}}\'')
    em = sub.add_parser("emit", help="emit an event the app handles, with an optional JSON payload")
    em.add_argument("event")
    em.add_argument("payload", nargs="?", default="null")
    sub.add_parser("events", help="list the events the app handles, with their handlers' docstrings")
    mt = sub.add_parser("metrics", help="print the last seconds of metrics (manifoldx run --debug)")
    mt.add_argument("-n", type=int, default=10, help="how many seconds (default 10)")
    lg = sub.add_parser("log", help="print log lines; with --match or --wait, exit 1 when no line came")
    lg.add_argument("-n", type=int, default=40, help="how many of the last lines (default 40)")
    lg.add_argument("--after", type=int, help="only lines after this sequence number")
    lg.add_argument("--match", help="only lines matching this regex")
    lg.add_argument("--wait", type=float, default=0.0, help="seconds to wait for a line (at most 300)")
    args = parser.parse_args(argv)
    if args.command in ("look", "state", "set", "emit", "events", "log", "metrics"):
        return dev_command(args)
    if args.command == "run":
        if (args.module is None) == (args.script is None):
            parser.error("give app.py or -m module")
        port = {} if args.port is None else {"port": args.port}
        if args.web:
            if args.debug:
                parser.error("--debug measures the window's frames; under --web the tab runs them")
            from manifoldx.web.dev import WebReloader

            WebReloader(script=args.script, module=args.module, argv=app_args, **port).run()
            return 0
        from manifoldx.hot import HotReloader

        HotReloader(script=args.script, module=args.module, argv=app_args, debug=args.debug, **port).run()
        return 0
    out = "OUTDIR" if args.web else "out.glb"
    if len(args.paths) != (1 if args.module else 2):
        parser.error(f"give app.py and {out}, or -m module and {out}")
    if args.web:
        if args.strict:
            parser.error("--strict is for the .glb export; --web always fails on what cannot cross")
        from manifoldx.web.export import export_web

        return export_web(args.module, None if args.module else args.paths[0], args.paths[-1], app_args)
    return export(args.module, None if args.module else args.paths[0], args.paths[-1], args.strict, app_args)


def dev_command(args):
    """look, state, set, emit, events and log: ask the dev server of the app running here."""
    import json
    import urllib.error
    import urllib.request
    from urllib.parse import quote, urlencode

    from manifoldx.devchannel import FOLDER, server_url

    url = server_url()
    if url is None:
        print(f"manifoldx {args.command}: no app runs under manifoldx run here (no {FOLDER}/server.json)",
              file=sys.stderr)
        return 1

    def call(path, method="GET", body=None, timeout=30):
        data = None if body is None else body.encode()
        request = urllib.request.Request(url + path, data=data, method=method)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            raise SystemExit(f"manifoldx {args.command}: {e.code} {e.read().decode(errors='replace')}") from None
        except OSError as e:
            raise SystemExit(f"manifoldx {args.command}: the app does not answer at {url}: {e}") from None

    def show(raw):
        print(json.dumps(json.loads(raw), indent=1, ensure_ascii=False))

    if args.command == "look":
        out = os.path.join(FOLDER, "look.png")
        with open(out, "wb") as f:
            f.write(call("/api/look"))
        print(out)
    elif args.command == "state":
        value = json.loads(call("/api/state"))
        for key in args.path.split(".") if args.path else ():
            try:
                value = value[int(key)] if isinstance(value, list) else value[key]
            except (KeyError, IndexError, ValueError, TypeError):
                print(f"manifoldx state: no {args.path!r} in engine.state", file=sys.stderr)
                return 1
        print(json.dumps(value, indent=1, ensure_ascii=False))
    elif args.command == "set":
        show(call("/api/state", "PATCH", args.patch))
    elif args.command == "emit":
        show(call("/api/events/" + quote(args.event, safe=""), "POST", args.payload))
    elif args.command == "metrics":
        for m in json.loads(call("/api/metrics"))[-args.n:]:
            s = m.get("split_ms", {})
            print(f"{m['time']}  fps {m['fps']:5.1f}  frame {m['frame_ms']:5.1f} ms  worst {m['worst_ms']:6.1f} ms"
                  f"  update {s.get('update', 0):5.1f}  render {s.get('render', 0):5.1f}"
                  f"  acquire {s.get('acquire', 0):5.1f}  idle {s.get('idle', 0):5.1f}"
                  f"  {m['rss_mb']:7.1f} MB  gc {'/'.join(map(str, m['gc']['runs']))} {m['gc']['ms']} ms")
    elif args.command == "events":
        for name, docs in json.loads(call("/api/events")).items():
            print(name + "".join(f"\n    {d}" for d in docs if d))
    else:
        q = {"match": args.match} if args.match else {}
        q |= {"after": args.after} if args.after is not None else {"last": args.n}
        q |= {"wait": args.wait} if args.wait else {}
        got = json.loads(call("/api/log?" + urlencode(q), timeout=args.wait + 30))
        for line in got["lines"]:
            print(f"{line['seq']} {line['time']} {line['text']}")
        if (args.wait or args.match) and not got["lines"]:
            return 1
    return 0


def export(module, script, output, strict, app_args):
    from manifoldx.gltf import ExportIncomplete
    from manifoldx.web.export import capture_engine

    engine, _ = capture_engine(module, script, app_args)
    if engine is None:
        print(f"manifoldx export: {module or script} finished without calling engine.run(); nothing written",
              file=sys.stderr)
        return 2
    try:
        engine.export_gltf(output, strict=strict)
    except ExportIncomplete:
        return 1
    return 0
