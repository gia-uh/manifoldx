// The HTML GUI without a browser: panelCss as a pure function, buildGui over a fake document.

import assert from "node:assert/strict";
import test from "node:test";

import { buildGui, panelCss } from "../../../src/manifoldx/web/js/gui.js";

const THEME = {
  bg: "#222", fg: "#ddd", font_size: 14, padding: 4, gap: 4, border: 0, border_color: "#444",
  radius: 0, width: null, height: null, flex: null, direction: "v",
};
const panel = (style = {}, extra = {}) => ({
  kind: "Panel", anchor: "top-left", offset: [0, 0], style: { ...THEME, ...style }, children: [], ...extra,
});

test("direction v is a column, h a row", () => {
  assert.equal(panelCss(panel(), 1, true)["flex-direction"], "column");
  assert.equal(panelCss(panel({ direction: "h" }), 1, true)["flex-direction"], "row");
});

test("padding '4 8' becomes four pixel values, scaled", () => {
  assert.equal(panelCss(panel({ padding: "4 8" }), 1, true).padding, "4px 8px 4px 8px");
  assert.equal(panelCss(panel({ padding: "1 2 3 4" }), 2, true).padding, "2px 4px 6px 8px");
  assert.equal(panelCss(panel({ padding: 3 }), 1, true).padding, "3px 3px 3px 3px");
});

test("every size, padding, gap, border, radius, font size and offset is scaled", () => {
  const css = panelCss(
    panel({ width: 240, height: 76, padding: 8, gap: 4, border: 2, radius: 6, font_size: 14 }, { offset: [16, 10] }),
    1.5, true,
  );
  assert.equal(css.width, "360px");
  assert.equal(css.height, "114px");
  assert.equal(css.padding, "12px 12px 12px 12px");
  assert.equal(css.gap, "6px");
  assert.equal(css["border-width"], "3px");
  assert.equal(css["border-radius"], "9px");
  assert.equal(css["font-size"], "21px");
  assert.equal(css.left, "24px");
  assert.equal(css.top, "15px");
});

test("the nine anchors place a root panel", () => {
  const at = (anchor) => {
    const css = panelCss(panel({}, { anchor, offset: [16, 16] }), 1, true);
    const pick = {};
    for (const k of ["left", "right", "top", "bottom", "transform"]) if (k in css) pick[k] = css[k];
    return pick;
  };
  assert.deepEqual(at("top-left"), { left: "16px", top: "16px" });
  assert.deepEqual(at("top-right"), { right: "16px", top: "16px" });
  assert.deepEqual(at("bottom-left"), { left: "16px", bottom: "16px" });
  assert.deepEqual(at("bottom-right"), { right: "16px", bottom: "16px" });
  assert.deepEqual(at("center"), { left: "50%", top: "50%", transform: "translate(-50%, -50%) translate(16px, 16px)" });
  assert.deepEqual(at("top-center"), { left: "50%", top: "16px", transform: "translate(-50%, 0px) translate(16px, 0px)" });
  assert.deepEqual(at("bottom-center"), { left: "50%", bottom: "16px", transform: "translate(-50%, 0px) translate(16px, 0px)" });
  assert.deepEqual(at("center-left"), { left: "16px", top: "50%", transform: "translate(0px, -50%) translate(0px, 16px)" });
  assert.deepEqual(at("center-right"), { right: "16px", top: "50%", transform: "translate(0px, -50%) translate(0px, 16px)" });
});

test("a nested panel is not placed", () => {
  const css = panelCss(panel({}, { anchor: "center", offset: [5, 5] }), 1, false);
  for (const k of ["position", "left", "right", "top", "bottom", "transform"]) assert.equal(k in css, false, k);
});

test("bg passes through as CSS hex; colours and border colour too", () => {
  const css = panelCss(panel({ bg: "#1a1a1aC0", fg: "#ffcc00", border: 1, border_color: "#123" }), 1, true);
  assert.equal(css["background-color"], "#1a1a1aC0");
  assert.equal(css.color, "#ffcc00");
  assert.equal(css["border-color"], "#123");
  assert.equal(css["border-style"], "solid");
});

test("width null leaves the width to content; a flex value becomes flex-grow", () => {
  const css = panelCss(panel({ flex: 2 }), 1, false);
  assert.equal("width" in css, false);
  assert.equal("height" in css, false);
  assert.equal(css["flex-grow"], "2");
  assert.equal("flex-grow" in panelCss(panel(), 1, false), false);
});

test("sized panels clip what overflows", () => {
  assert.equal(panelCss(panel({ width: 100 }), 1, true).overflow, "hidden");
  assert.equal("overflow" in panelCss(panel(), 1, true), false);
});

// ---------------------------------------------------------------- buildGui over a fake DOM

class FakeElement {
  constructor(tag) {
    this.tagName = tag;
    this.children = [];
    this.props = {};
    this.listeners = {};
    this.writes = 0;
    this._text = "";
    this.style = { setProperty: (k, v) => { this.props[k] = v; }, removeProperty: (k) => { delete this.props[k]; } };
  }
  get textContent() { return this._text; }
  set textContent(v) { this._text = String(v); this.writes++; }
  appendChild(c) { this.children.push(c); return c; }
  addEventListener(type, fn) { (this.listeners[type] ??= []).push(fn); }
  fire(type) { for (const fn of this.listeners[type] ?? []) fn({ type }); }
}
const fakeLayer = () => {
  const doc = { createElement: (tag) => new FakeElement(tag) };
  const layer = new FakeElement("div");
  layer.ownerDocument = doc;
  return layer;
};

const WALK = {
  scale: 1, design_height: null,
  panels: [panel({ width: 240, height: 76, padding: 8, bg: "#1a1a1aC0" }, {
    offset: [16, 16],
    children: [0, 1, 2].map((value) => ({ kind: "ValueDisplay", value, min_width: null, style: THEME })),
  })],
};

test("buildGui makes one div per widget and writes text only when it changed", () => {
  const layer = fakeLayer();
  const gui = buildGui(layer, WALK);
  assert.equal(layer.children.length, 1);
  const root = layer.children[0];
  assert.equal(root.props.left, "16px");
  assert.equal(root.props["background-color"], "#1a1a1aC0");
  assert.equal(root.children.length, 3);
  gui.update(["picked: nothing", "mode: walk", "<b>at</b>"]);
  assert.deepEqual(root.children.map((c) => c.textContent), ["picked: nothing", "mode: walk", "<b>at</b>"]);
  const writes = root.children.map((c) => c.writes);
  gui.update(["picked: nothing", "mode: fly", "<b>at</b>"]);
  assert.deepEqual(root.children.map((c) => c.writes), [writes[0], writes[1] + 1, writes[2]]);
});

test("a Text shows its text; a ValueDisplay's min_width is a scaled min-width", () => {
  const layer = fakeLayer();
  buildGui(layer, {
    scale: 2, design_height: null,
    panels: [panel({}, { children: [
      { kind: "Text", text: "hello", style: THEME },
      { kind: "ValueDisplay", value: 0, min_width: 80, style: THEME },
    ] })],
  });
  const [text, vd] = layer.children[0].children;
  assert.equal(text.textContent, "hello");
  assert.equal(vd.props["min-width"], "160px");
  assert.equal(text.props["font-size"], "28px");
});

test("pointerOver follows the root panels' pointerenter and pointerleave", () => {
  const layer = fakeLayer();
  const gui = buildGui(layer, WALK);
  assert.equal(gui.pointerOver, false);
  layer.children[0].fire("pointerenter");
  assert.equal(gui.pointerOver, true);
  layer.children[0].fire("pointerleave");
  assert.equal(gui.pointerOver, false);
});

test("rescale follows engine.h when design_height is set, and does nothing otherwise", () => {
  const layer = fakeLayer();
  const gui = buildGui(layer, { ...WALK, design_height: 500 });
  gui.rescale(1000);
  assert.equal(layer.children[0].props.width, "480px");
  assert.equal(layer.children[0].children[0].props["font-size"], "28px");
  const fixed = fakeLayer();
  const g2 = buildGui(fixed, WALK);
  g2.rescale(1000);
  assert.equal(fixed.children[0].props.width, "240px");
});

// ---------------------------------------------------------------- widget writes (#147)

// A root panel (widget 0) holding a Text (widget 1), as gui_spec writes them, and the widget
// objects reached code writes, as loadApp builds them from the spec.
const MOVING = () => ({
  scale: 1, design_height: null,
  panels: [panel({}, {
    widget: 0, base: THEME, overrides: {}, offset: [10, 20],
    children: [{ kind: "Text", widget: 1, text: "hi", style: THEME, base: THEME, overrides: {} }],
  })],
});
const widgetsOf = () => [
  { index: 0, offset: [10, 20], style_overrides: new Map() },
  { index: 1, text: "hi", style_overrides: new Map() },
];
const countWrites = (el) => {
  let n = 0;
  const { setProperty, removeProperty } = el.style;
  el.style.setProperty = (k, v) => { n++; setProperty(k, v); };
  el.style.removeProperty = (k) => { n++; removeProperty(k); };
  return () => n;
};

test("sync moves a root panel when reached code writes its offset", () => {
  const layer = fakeLayer();
  const gui = buildGui(layer, MOVING());
  const widgets = widgetsOf();
  widgets[0].offset = [55, 30];
  gui.sync(widgets);
  assert.equal(layer.children[0].props.left, "55px");
  assert.equal(layer.children[0].props.top, "30px");
});

test("sync writes a Text's new text and a style override, and drops a removed one", () => {
  const layer = fakeLayer();
  const gui = buildGui(layer, MOVING());
  const widgets = widgetsOf();
  widgets[1].text = "bye";
  widgets[0].style_overrides.set("width", 100);
  widgets[0].style_overrides.set("bg", "#ff0000");
  gui.sync(widgets);
  const root = layer.children[0];
  assert.equal(root.children[0].textContent, "bye");
  assert.equal(root.props.width, "100px");
  assert.equal(root.props["background-color"], "#ff0000");
  widgets[0].style_overrides.delete("width");
  gui.sync(widgets);
  assert.equal("width" in root.props, false);
  assert.equal(root.props["background-color"], "#ff0000");
});

test("sync writes nothing to the DOM when no widget field changed", () => {
  const layer = fakeLayer();
  const gui = buildGui(layer, MOVING());
  const widgets = widgetsOf();
  const root = layer.children[0], text = root.children[0];
  const writes = countWrites(root), textWrites = text.writes;
  gui.sync(widgets);
  widgets[0].offset = [10, 20]; // a new tuple, the same value
  gui.sync(widgets);
  assert.equal(writes(), 0);
  assert.equal(text.writes, textWrites);
  widgets[0].offset = [11, 20];
  gui.sync(widgets);
  assert.ok(writes() > 0);
});

test("a rescale keeps the offset reached code wrote", () => {
  const layer = fakeLayer();
  const gui = buildGui(layer, { ...MOVING(), design_height: 100 });
  const widgets = widgetsOf();
  widgets[0].offset = [40, 20];
  gui.sync(widgets);
  gui.rescale(200);
  assert.equal(layer.children[0].props.left, "80px");
});
