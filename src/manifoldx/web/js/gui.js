// engine.gui as HTML: state.json's "gui" (manifoldx.web.gui.gui_spec) to elements in a layer
// over the canvas. Sizes are UI pixels times the GUI's scale, as the desktop paints them
// (manifoldx.gui.painter); colours are sRGB hex, which CSS reads as they are. Text goes in
// through textContent only.

const px = (v) => `${v}px`;

// (top, right, bottom, left) from an int or a string of 1, 2 or 4 ints, as
// manifoldx.gui.style.parse_padding reads it.
function padding(value) {
  if (typeof value === "number") return [value, value, value, value];
  const n = String(value).trim().split(/\s+/).map(Number);
  if (n.length === 1) return [n[0], n[0], n[0], n[0]];
  if (n.length === 2) return [n[0], n[1], n[0], n[1]];
  return n;
}

// Where an anchor puts a root panel, per axis: "start", "end" or "center".
const AXES = {
  "top-left": ["start", "start"], "top-center": ["center", "start"], "top-right": ["end", "start"],
  "center-left": ["start", "center"], center: ["center", "center"], "center-right": ["end", "center"],
  "bottom-left": ["start", "end"], "bottom-center": ["center", "end"], "bottom-right": ["end", "end"],
};

// The CSS of one node at a scale: {property: value}. A root panel is placed by its anchor,
// its offset measured inward from the anchored edges (on a centred axis, rightward or
// downward); a nested one is laid out by its parent.
export function panelCss(node, scale, isRoot) {
  const s = node.style;
  const css = {
    "box-sizing": "border-box",
    color: s.fg,
    "font-size": px(s.font_size * scale),
  };
  if (s.width !== null && s.width !== undefined) css.width = px(s.width * scale);
  if (s.height !== null && s.height !== undefined) css.height = px(s.height * scale);
  if (css.width !== undefined || css.height !== undefined) css.overflow = "hidden";
  if (s.flex !== null && s.flex !== undefined) css["flex-grow"] = String(s.flex);
  else css["flex-shrink"] = "0";
  if (node.kind === "Panel") {
    css.display = "flex";
    css["flex-direction"] = s.direction === "h" ? "row" : "column";
    css.padding = padding(s.padding).map((v) => px(v * scale)).join(" ");
    css.gap = px(s.gap * scale);
    css["background-color"] = s.bg;
    css["border-radius"] = px(s.radius * scale);
    if (s.border) {
      css["border-width"] = px(s.border * scale);
      css["border-style"] = "solid";
      css["border-color"] = s.border_color;
    }
  } else if (node.kind === "ValueDisplay" && node.min_width !== null && node.min_width !== undefined) {
    css["min-width"] = px(node.min_width * scale);
  }
  if (isRoot) {
    css.position = "absolute";
    css["pointer-events"] = "auto";
    const [h, v] = AXES[node.anchor];
    const [ox, oy] = node.offset.map((o) => o * scale);
    if (h === "start") css.left = px(ox);
    else if (h === "end") css.right = px(ox);
    else css.left = "50%";
    if (v === "start") css.top = px(oy);
    else if (v === "end") css.bottom = px(oy);
    else css.top = "50%";
    if (h === "center" || v === "center") {
      const shift = `translate(${h === "center" ? "-50%" : "0px"}, ${v === "center" ? "-50%" : "0px"})`;
      const off = `translate(${h === "center" ? px(ox) : "0px"}, ${v === "center" ? px(oy) : "0px"})`;
      css.transform = `${shift} ${off}`;
    }
  }
  return css;
}

// The widgets a pointer acts on (_is_gui_interactive), and those that keep the pointer that went
// down on them (_gui_captures_pointer).
const INTERACTIVE = new Set(["Button", "Slider", "Toggle", "Joystick"]);
const CAPTURES = new Set(["Slider", "Joystick"]);
const WIDGETS = new Set([...INTERACTIVE, "Image"]);

// The SVG of an Image as the src of an <img>, which runs no script and needs no request.
const SVG_URL = "data:image/svg+xml;charset=utf-8,";

// A style size, or the widget's intrinsic one (Slider 160 x 16, Joystick 96 x 96), in UI pixels.
const sized = (v, fallback) => (v !== null && v !== undefined ? v : fallback);

// The Slider's fill fraction t and the Joystick's knob, as the painter computes them.
function sliderT(node) {
  const t = node.max > node.min ? (node.value - node.min) / (node.max - node.min) : 0.0;
  return Math.max(0.0, Math.min(1.0, t));
}

function joystickSize(node, scale) {
  const s = node.style;
  return [sized(s.width, 96) * scale, sized(s.height, 96) * scale];
}

// The CSS of one widget element at a scale: {property: value}, from node.style (bg, fg,
// border, border_color, radius, font_size), the keys the painter reads. Its parts (the Slider's
// fill, the Toggle's box and mark, the Joystick's knob) are partCss's.
export function widgetCss(node, scale) {
  const s = node.style;
  const css = {
    "box-sizing": "border-box",
    "flex-shrink": "0",
    color: s.fg,
    "font-size": px(s.font_size * scale),
    "font-family": "inherit",
    margin: "0",
  };
  if (INTERACTIVE.has(node.kind)) {
    css["touch-action"] = "none";
    css["user-select"] = "none";
  }
  const border = () => {
    css["border-width"] = px(s.border * scale);
    css["border-style"] = s.border ? "solid" : "none";
    css["border-color"] = s.border_color;
  };
  if (node.kind === "Button") {
    css["background-color"] = s.bg;
    css["border-radius"] = px(s.radius * scale);
    border();
    css.padding = `${px(4 * scale)} ${px(12 * scale)}`;
    css["text-align"] = "center";
    if (s.width !== null && s.width !== undefined) css.width = px(s.width * scale);
    if (s.height !== null && s.height !== undefined) css.height = px(s.height * scale);
  } else if (node.kind === "Slider") {
    css.position = "relative";
    css.width = px(sized(s.width, 160) * scale);
    css.height = px(sized(s.height, 16) * scale);
    css["background-color"] = s.bg;
    css["border-radius"] = px(s.radius * scale);
    border();
    css.overflow = "hidden";
  } else if (node.kind === "Toggle") {
    css.display = "flex";
    css["align-items"] = "center";
    css.gap = px(6 * scale);
  } else if (node.kind === "Joystick") {
    const [w, h] = joystickSize(node, scale);
    css.position = "relative";
    css.width = px(w);
    css.height = px(h);
    css["background-color"] = s.bg;
    css["border-radius"] = "50%";
    border();
  } else if (node.kind === "Image") {
    css.display = "block";
    css.width = px(sized(s.width, node.size[0]) * scale);
    css.height = px(sized(s.height, node.size[1]) * scale);
  }
  return css;
}

// The CSS of a widget's parts, by part name, at a scale.
function partCss(node, scale) {
  const s = node.style;
  if (node.kind === "Slider") {
    return {
      fill: {
        position: "absolute", left: "0", top: "0", bottom: "0",
        width: `${sliderT(node) * 100}%`, "background-color": s.fg, "border-radius": px(s.radius * scale),
      },
    };
  }
  if (node.kind === "Toggle") {
    const box = 14 * scale, inset = 3 * scale;
    return {
      box: {
        position: "relative", "box-sizing": "border-box", "flex-shrink": "0", width: px(box), height: px(box),
        "background-color": s.bg, "border-width": px(scale), "border-style": "solid", "border-color": s.fg,
        "border-radius": px(2 * scale),
      },
      mark: {
        position: "absolute", left: px(inset - scale), top: px(inset - scale),
        width: px(box - 2 * inset), height: px(box - 2 * inset),
        "background-color": s.fg, "border-radius": px(scale), display: node.value ? "block" : "none",
      },
      label: { "white-space": "pre" },
    };
  }
  if (node.kind === "Joystick") {
    // As the painter: r half the shorter side, the knob 0.8 r wide, moved by value * (r - knob / 2), y up.
    const [w, h] = joystickSize(node, scale);
    const r = Math.min(w, h) * 0.5, k = 0.8 * r;
    const [vx, vy] = node.value;
    const cx = w * 0.5 + vx * (r - k * 0.5), cy = h * 0.5 - vy * (r - k * 0.5);
    return {
      knob: {
        position: "absolute", left: px(cx - k * 0.5), top: px(cy - k * 0.5), width: px(k), height: px(k),
        "background-color": node.style.fg, "border-radius": "50%", "pointer-events": "none",
      },
    };
  }
  return {};
}

// Write an element's CSS: only the properties that changed, and remove those it lost. had is
// the CSS it has.
function writeCss(el, had, want) {
  for (const k of Object.keys(had)) {
    if (!(k in want)) {
      el.style.removeProperty(k);
      delete had[k];
    }
  }
  for (const [k, v] of Object.entries(want)) {
    if (had[k] !== v) {
      el.style.setProperty(k, v);
      had[k] = v;
    }
  }
}

// Whether a widget's style_overrides (a Map, as every dict of transpiled code) holds what the
// node last applied (a plain object).
function sameOverrides(map, applied) {
  const keys = Object.keys(applied);
  if (map.size !== keys.length) return false;
  for (const k of keys) if (!map.has(k) || map.get(k) !== applied[k]) return false;
  return true;
}

// The fields of each widget kind that frame code writes and the page draws.
const FIELDS = { Button: ["label"], Slider: ["value", "min", "max", "label"], Toggle: ["value", "label"], Joystick: ["value"] };
const same = (a, b) => (Array.isArray(a) ? a.length === b.length && a.every((v, i) => v === b[i]) : a === b);

// buildGui(layer, spec) -> {layer, update(values), sync(widgets), rescale(engineH), destroy(),
// targetOf(domEvent), hold(pointerId, widget), release(pointerId)}: the panels as elements in
// layer, each widget's element marked data-mx-widget with its index. update writes each
// ValueDisplay's text (values in getter order, guiValues') only when it changed; sync applies
// what reached code and the widget code wrote into the widget objects (engine.gui.widgets: a
// Panel's offset, a Text's text, a widget's label, value, min and max with their ARIA values,
// any widget's style_overrides, resolved over its base style), writing the DOM only where a
// value changed; rescale recomputes the sizes when the GUI scales with the window
// (design_height); destroy removes the panels, for the next version's GUI. targetOf names the
// widget a pointer event went to, [widget, fx, fy] with the pointer's place as fractions of
// that widget's box: the widget holding that pointer, else the interactive widget under
// domEvent.target, else null. A pointerdown on a Slider or a Joystick captures the pointer to
// its element and holds it; release ends the hold (attachDom calls it once a pointerup has
// read its target), as a pointercancel does.
export function buildGui(layer, spec) {
  const doc = layer.ownerDocument;
  const all = []; // {el, node, isRoot, had, parts: {name: [element, the CSS it has]}}
  const displays = []; // the ValueDisplays' elements, by getter index
  const shown = [];
  const elements = new Map(); // element -> widget index, the interactive widgets
  const byIndex = new Map(); // widget index -> its element
  const holds = new Map(); // pointerId -> widget index
  let scale = spec.scale;

  function part(entry, name, tag, parent) {
    const el = doc.createElement(tag);
    parent.appendChild(el);
    entry.parts[name] = [el, {}];
    return el;
  }

  function aria(entry) {
    const { el, node } = entry;
    const set = (k, v) => { if (el.getAttribute(k) !== String(v)) el.setAttribute(k, v); };
    if (node.kind === "Slider") {
      set("aria-valuemin", node.min);
      set("aria-valuemax", node.max);
      set("aria-valuenow", node.value);
      set("aria-label", node.label ?? node.name);
    } else if (node.kind === "Toggle") set("aria-checked", node.value ? "true" : "false");
  }

  function make(node, isRoot) {
    const tag = node.kind === "Button" ? "button" : node.kind === "Image" ? "img" : "div";
    const el = doc.createElement(tag);
    const entry = { el, node, isRoot, had: {}, parts: {} };
    all.push(entry);
    if (node.kind === "Panel") {
      el.style.setProperty("white-space", "pre");
      for (const c of node.children) el.appendChild(make(c, false));
    } else if (node.kind === "Text") {
      el.style.setProperty("white-space", "pre");
      el.textContent = node.text;
    } else if (node.kind === "ValueDisplay") {
      el.style.setProperty("white-space", "pre");
      displays[node.value] = el;
    } else if (node.kind === "Button") {
      el.setAttribute("type", "button");
      el.textContent = node.label;
    } else if (node.kind === "Slider") {
      el.setAttribute("role", "slider");
      el.setAttribute("tabindex", "0");
      part(entry, "fill", "div", el);
    } else if (node.kind === "Toggle") {
      el.setAttribute("role", "switch");
      el.setAttribute("tabindex", "0");
      part(entry, "mark", "div", part(entry, "box", "div", el));
      part(entry, "label", "span", el).textContent = node.label;
    } else if (node.kind === "Joystick") {
      part(entry, "knob", "div", el);
    } else if (node.kind === "Image") {
      el.setAttribute("alt", "");
      if (node.svg !== null && node.svg !== undefined) el.setAttribute("src", SVG_URL + encodeURIComponent(node.svg));
    }
    if (node.widget !== undefined && WIDGETS.has(node.kind)) {
      el.setAttribute("data-mx-widget", node.widget);
      byIndex.set(node.widget, el);
    }
    if (INTERACTIVE.has(node.kind)) elements.set(el, node.widget);
    if (CAPTURES.has(node.kind)) {
      el.addEventListener("pointerdown", (ev) => {
        el.setPointerCapture(ev.pointerId);
        holds.set(ev.pointerId, node.widget);
      });
      el.addEventListener("pointercancel", (ev) => holds.delete(ev.pointerId));
    }
    aria(entry);
    return el;
  }

  function css(entry) {
    const { el, node, isRoot, had, parts } = entry;
    writeCss(el, had, WIDGETS.has(node.kind) ? widgetCss(node, scale) : panelCss(node, scale, isRoot));
    const want = partCss(node, scale);
    for (const [name, [pel, phad]] of Object.entries(parts)) writeCss(pel, phad, want[name] ?? {});
  }

  function apply() {
    for (const entry of all) css(entry);
  }

  for (const p of spec.panels) layer.appendChild(make(p, true));
  apply();

  return {
    layer,
    update(values) {
      values.forEach((v, i) => {
        if (shown[i] !== v && displays[i] !== undefined) {
          displays[i].textContent = v;
          shown[i] = v;
        }
      });
    },
    sync(widgets) {
      for (const entry of all) {
        const { el, node } = entry;
        const w = widgets[node.widget];
        if (w === undefined) continue;
        let changed = false;
        if (node.kind === "Panel" && (w.offset[0] !== node.offset[0] || w.offset[1] !== node.offset[1])) {
          node.offset = [w.offset[0], w.offset[1]];
          changed = true;
        }
        if (node.kind === "Text" && w.text !== node.text) {
          node.text = w.text;
          el.textContent = w.text;
        }
        for (const f of FIELDS[node.kind] ?? []) {
          if (w[f] === undefined || same(w[f], node[f])) continue;
          node[f] = Array.isArray(w[f]) ? [...w[f]] : w[f];
          if (f === "label" && node.kind === "Button") el.textContent = w.label;
          else if (f === "label" && node.kind === "Toggle") entry.parts.label[0].textContent = w.label;
          changed = true;
        }
        if (!sameOverrides(w.style_overrides, node.overrides)) {
          node.overrides = Object.fromEntries(w.style_overrides);
          node.style = { ...node.base, ...node.overrides };
          changed = true;
        }
        if (changed) {
          css(entry);
          aria(entry);
        }
      }
    },
    rescale(engineH) {
      if (spec.design_height) {
        scale = engineH / spec.design_height;
        apply();
      }
    },
    destroy() {
      for (const entry of all) if (entry.isRoot) layer.removeChild(entry.el);
      all.length = 0;
      holds.clear();
    },
    targetOf(ev) {
      let index = holds.get(ev.pointerId);
      if (index === undefined) {
        for (let n = ev.target; n !== null && n !== undefined && n !== layer; n = n.parentNode) {
          if (elements.has(n)) {
            index = elements.get(n);
            break;
          }
        }
      }
      if (index === undefined) return null;
      const r = byIndex.get(index).getBoundingClientRect();
      const fx = r.width > 0 ? (ev.clientX - r.left) / r.width : 0.0;
      const fy = r.height > 0 ? (ev.clientY - r.top) / r.height : 0.0;
      return [index, fx, fy];
    },
    hold(pointerId, widget) {
      holds.set(pointerId, widget);
    },
    release(pointerId) {
      holds.delete(pointerId);
    },
  };
}
