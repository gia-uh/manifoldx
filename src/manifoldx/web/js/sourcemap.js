// A JS error in the app's code, told by its Python file and line: the export's source maps
// (one segment per emitted line, sources rewritten to the app's .py paths) read back.

const B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

// The source line of each generated line's first segment, 1-based, 0 where none: an
// Int32Array indexed by the 0-based generated line.
export function lineTable(map) {
  const groups = map.mappings.split(";");
  const out = new Int32Array(groups.length);
  let src = 0;
  groups.forEach((group, i) => {
    let first = 0;
    for (const seg of group.split(",")) {
      if (!seg) continue;
      const values = [];
      let shift = 0, acc = 0;
      for (const ch of seg) {
        const d = B64.indexOf(ch);
        acc += (d & 31) << shift;
        shift += 5;
        if (!(d & 32)) {
          values.push(acc & 1 ? -(acc >> 1) : acc >> 1);
          acc = shift = 0;
        }
      }
      if (values.length >= 4) {
        src += values[2];
        if (!first) first = src + 1;
      }
    }
    out[i] = first;
  });
  return out;
}

// "at fn (url:line:col)", "at url:line:col" (V8, Chromium) or "fn@url:line:col" (Firefox).
const V8 = /^\s*at (?:(.*?) \()?(.+?):(\d+):\d+\)?$/;
const FIREFOX = /^(.*?)@(.+?):(\d+):\d+$/;

// The function's Python name from the stack's: the last dotted part, without V8's "async "
// and " [as x]" or Firefox's "/<"; "<lambda>" when the frame has none.
function pyName(fn) {
  const name = (fn ?? "").replace(/^async /, "").replace(/ \[as [^\]]*\]$/, "").replace(/[/<]+$/, "");
  return name ? name.slice(name.lastIndexOf(".") + 1) : "<lambda>";
}

// The stack's frames in the app's modules, as Python names them: [{file, line, fn}],
// innermost first. maps: {"app/<name>.js": {source, lines}}, lines being lineTable's.
// Frames of other files (the runtime, three.js) and unmapped lines are dropped.
export function pythonFrames(stack, maps) {
  const out = [];
  for (const text of String(stack ?? "").split("\n")) {
    const m = V8.exec(text) ?? FIREFOX.exec(text);
    if (!m) continue;
    const url = m[2].replace(/[?#].*$/, "");
    const key = Object.keys(maps).find((k) => url === k || url.endsWith(`/${k}`));
    if (key === undefined) continue;
    const line = maps[key].lines[Number(m[3]) - 1] ?? 0;
    if (line) out.push({ file: maps[key].source, line, fn: pyName(m[1]) });
  }
  return out;
}

// The error as the page shows it: "app.py:57 in walk: TypeError: ...", then one
// "  called from app.py:121 in tick" line per outer Python frame. With no Python frame,
// "TypeError: ..." alone.
export function describe(error, maps) {
  const head = `${error?.name ?? "Error"}: ${error?.message ?? String(error)}`;
  const [first, ...rest] = pythonFrames(error?.stack, maps);
  if (!first) return head;
  return [
    `${first.file}:${first.line} in ${first.fn}: ${head}`,
    ...rest.map((f) => `  called from ${f.file}:${f.line} in ${f.fn}`),
  ].join("\n");
}
