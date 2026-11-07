// engine.gui as HTML: state.json's "gui" (manifoldx.web.gui.gui_spec) to divs in a layer
// over the canvas. Sizes are UI pixels times the GUI's scale, as the desktop paints them;
// colours are sRGB hex, which CSS reads as they are. Text goes in through textContent only.

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

// Whether a widget's style_overrides (a Map, as every dict of transpiled code) holds what the
// node last applied (a plain object).
function sameOverrides(map, applied) {
  const keys = Object.keys(applied);
  if (map.size !== keys.length) return false;
  for (const k of keys) if (!map.has(k) || map.get(k) !== applied[k]) return false;
  return true;
}

// buildGui(layer, spec) -> {update(values), sync(widgets), rescale(engineH), pointerOver}: the
// panels as divs in layer. update writes each ValueDisplay's text (values in getter order,
// guiValues') only when it changed; sync applies what reached code wrote into the widget
// objects (engine.gui.widgets: a Panel's offset, a Text's text, any widget's style_overrides,
// resolved over its base style), writing the DOM only where a value changed; rescale
// recomputes the sizes when the GUI scales with the window (design_height). pointerOver is
// whether the pointer is over a root panel; browser.js copies it into
// engine.gui.pointer_over_gui each frame.
export function buildGui(layer, spec) {
  const doc = layer.ownerDocument;
  const all = []; // [element, node, isRoot, the CSS it has]
  const displays = []; // the ValueDisplays' elements, by getter index
  const shown = [];
  let over = 0;
  let scale = spec.scale;

  function make(node, isRoot) {
    const el = doc.createElement("div");
    all.push([el, node, isRoot, {}]);
    el.style.setProperty("white-space", "pre");
    if (node.kind === "Panel") for (const c of node.children) el.appendChild(make(c, false));
    else if (node.kind === "Text") el.textContent = node.text;
    else if (node.kind === "ValueDisplay") displays[node.value] = el;
    if (isRoot) {
      el.addEventListener("pointerenter", () => { over++; });
      el.addEventListener("pointerleave", () => { over = Math.max(0, over - 1); });
    }
    return el;
  }

  // Write one element's CSS: only the properties that changed, and remove those it lost.
  function css(entry) {
    const [el, node, isRoot, had] = entry;
    const want = panelCss(node, scale, isRoot);
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

  function apply() {
    for (const entry of all) css(entry);
  }

  for (const p of spec.panels) layer.appendChild(make(p, true));
  apply();

  return {
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
        const [el, node] = entry;
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
        if (!sameOverrides(w.style_overrides, node.overrides)) {
          node.overrides = Object.fromEntries(w.style_overrides);
          node.style = { ...node.base, ...node.overrides };
          changed = true;
        }
        if (changed) css(entry);
      }
    },
    rescale(engineH) {
      if (spec.design_height) {
        scale = engineH / spec.design_height;
        apply();
      }
    },
    get pointerOver() { return over > 0; },
  };
}
