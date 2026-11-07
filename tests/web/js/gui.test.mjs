// The HTML GUI without a browser: panelCss as a pure function, buildGui over a fake document.

import assert from "node:assert/strict";
import test from "node:test";

import { buildGui, panelCss, widgetCss } from "../../../src/manifoldx/web/js/gui.js";

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
    this.tagName = tag.toUpperCase();
    this.children = [];
    this.parentNode = null;
    this.props = {};
    this.attrs = {};
    this.listeners = {};
    this.captured = [];
    this.writes = 0;
    this.attrWrites = 0;
    this._text = "";
    this.rect = { left: 0, top: 0, width: 0, height: 0 };
    this.style = { setProperty: (k, v) => { this.props[k] = v; }, removeProperty: (k) => { delete this.props[k]; } };
  }
  get textContent() { return this._text; }
  set textContent(v) { this._text = String(v); this.writes++; }
  setAttribute(k, v) { this.attrs[k] = String(v); this.attrWrites++; }
  getAttribute(k) { return this.attrs[k] ?? null; }
  getBoundingClientRect() { return this.rect; }
  setPointerCapture(id) { this.captured.push(id); }
  appendChild(c) { this.children.push(c); c.parentNode = this; return c; }
  removeChild(c) { this.children.splice(this.children.indexOf(c), 1); c.parentNode = null; return c; }
  addEventListener(type, fn) { (this.listeners[type] ??= []).push(fn); }
  removeEventListener(type, fn) { this.listeners[type] = (this.listeners[type] ?? []).filter((f) => f !== fn); }
  fire(type, ev = {}) { for (const fn of this.listeners[type] ?? []) fn({ type, target: this, ...ev }); }
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

test("destroy leaves the layer empty", () => {
  const layer = fakeLayer();
  const gui = buildGui(layer, { scale: 1, design_height: null, panels: [panel(), panel()] });
  gui.destroy();
  assert.equal(layer.children.length, 0);
});

// ---------------------------------------------------------------- the five widgets (5b)

const STYLE = { ...THEME, bg: "#303030", fg: "#e0e0e0", border: 1, border_color: "#555", radius: 4 };
const node = (kind, widget, extra = {}) => ({ kind, widget, style: STYLE, base: STYLE, overrides: {}, ...extra });
const WIDGETS = () => ({
  scale: 2, design_height: null,
  panels: [panel({}, {
    widget: 0, base: THEME, overrides: {},
    children: [
      node("Button", 1, { name: "go", label: "Go" }),
      node("Slider", 2, { name: "speed", min: 0, max: 4, value: 1, label: "speed" }),
      node("Toggle", 3, { name: "lit", value: false, label: "lit" }),
      node("Joystick", 4, { name: "pad", value: [0.5, 0.25] }),
      node("Image", 5, { svg: '<svg xmlns="http://www.w3.org/2000/svg" width="8" height="4"/>', size: [8, 4] }),
    ],
  })],
});
const built = () => {
  const layer = fakeLayer();
  const gui = buildGui(layer, WIDGETS());
  const [button, slider, toggle, joystick, image] = layer.children[0].children;
  return { layer, gui, button, slider, toggle, joystick, image };
};
const widgetObjects = () => [
  { index: 0, kind: "Panel", offset: [0, 0], style_overrides: new Map() },
  { index: 1, kind: "Button", name: "go", label: "Go", style_overrides: new Map() },
  { index: 2, kind: "Slider", name: "speed", min: 0, max: 4, value: 1, label: "speed", style_overrides: new Map() },
  { index: 3, kind: "Toggle", name: "lit", value: false, label: "lit", style_overrides: new Map() },
  { index: 4, kind: "Joystick", name: "pad", value: [0.5, 0.25], style_overrides: new Map() },
  { index: 5, kind: "Image", style_overrides: new Map() },
];

test("a Button is a button element with its label and the resolved style", () => {
  const { button } = built();
  assert.equal(button.tagName, "BUTTON");
  assert.equal(button.attrs.type, "button");
  assert.equal(button.textContent, "Go");
  assert.equal(button.props["background-color"], "#303030");
  assert.equal(button.props.color, "#e0e0e0");
  assert.equal(button.props["border-radius"], "8px");
  assert.equal(button.props["border-width"], "2px");
  assert.equal(button.props["border-color"], "#555");
  assert.equal(button.props["text-align"], "center");
  assert.equal(button.attrs["data-mx-widget"], "1");
  assert.equal(widgetCss(WIDGETS().panels[0].children[0], 2)["font-size"], "28px");
});

test("a Slider has role slider, its ARIA range and a knob at its value", () => {
  const { slider } = built();
  assert.equal(slider.attrs.role, "slider");
  assert.equal(slider.attrs["aria-valuemin"], "0");
  assert.equal(slider.attrs["aria-valuemax"], "4");
  assert.equal(slider.attrs["aria-valuenow"], "1");
  assert.equal(slider.attrs["aria-label"], "speed");
  assert.equal(slider.attrs.tabindex, "0");
  assert.equal(slider.props.width, "320px"); // the intrinsic 160 x 16, times the scale
  assert.equal(slider.props.height, "32px");
  assert.equal(slider.children[0].props.width, "25%");
  assert.equal(slider.children[0].props["background-color"], "#e0e0e0");
  assert.equal(slider.props["touch-action"], "none");
});

test("a Toggle has role switch and aria-checked follows its value", () => {
  const { gui, toggle } = built();
  assert.equal(toggle.attrs.role, "switch");
  assert.equal(toggle.attrs["aria-checked"], "false");
  const [box, label] = toggle.children;
  assert.equal(box.props.width, "28px"); // 14 px, times the scale
  assert.equal(box.children[0].props.display, "none");
  assert.equal(label.textContent, "lit");
  const w = widgetObjects();
  w[3].value = true;
  gui.sync(w);
  assert.equal(toggle.attrs["aria-checked"], "true");
  assert.equal(box.children[0].props.display, "block");
});

test("a Joystick is a round pad with a knob and touch-action none", () => {
  const { joystick } = built();
  assert.equal(joystick.props["border-radius"], "50%");
  assert.equal(joystick.props.width, "192px"); // 96 px, times the scale
  assert.equal(joystick.props["touch-action"], "none");
  const knob = joystick.children[0];
  // r 96, knob 0.4 * 192 = 76.8; centre 96 + 0.5 * (96 - 38.4), 96 - 0.25 * (96 - 38.4), y up
  const k = 0.8 * 96;
  assert.equal(knob.props.width, `${k}px`);
  assert.equal(knob.props.left, `${96 + 0.5 * (96 - k * 0.5) - k * 0.5}px`);
  assert.equal(knob.props.top, `${96 - 0.25 * (96 - k * 0.5) - k * 0.5}px`);
  assert.equal(knob.props["border-radius"], "50%");
});

test("an Image shows its SVG as an img at svg_size times the scale", () => {
  const { image } = built();
  assert.equal(image.tagName, "IMG");
  assert.ok(image.attrs.src.startsWith("data:image/svg+xml;charset=utf-8,"));
  assert.equal(decodeURIComponent(image.attrs.src.slice("data:image/svg+xml;charset=utf-8,".length)), WIDGETS().panels[0].children[4].svg);
  assert.equal(image.props.width, "16px");
  assert.equal(image.props.height, "8px");
});

test("targetOf gives the widget and the fractions of its box", () => {
  const { gui, slider } = built();
  slider.rect = { left: 100, top: 50, width: 200, height: 20 };
  const knob = slider.children[0];
  assert.deepEqual(gui.targetOf({ target: knob, pointerId: 1, clientX: 150, clientY: 55 }), [2, 0.25, 0.25]);
  assert.equal(gui.targetOf({ target: { parentNode: null }, pointerId: 1, clientX: 150, clientY: 55 }), null);
});

test("a non-interactive widget is no target", () => {
  const { gui, image, layer } = built();
  assert.equal(gui.targetOf({ target: image, pointerId: 1, clientX: 0, clientY: 0 }), null);
  assert.equal(gui.targetOf({ target: layer.children[0], pointerId: 1, clientX: 0, clientY: 0 }), null);
});

test("a held pointer keeps its widget outside the box", () => {
  const { gui, slider } = built();
  slider.rect = { left: 100, top: 50, width: 200, height: 20 };
  gui.hold(1, 2);
  const canvas = { parentNode: null };
  assert.deepEqual(gui.targetOf({ target: canvas, pointerId: 1, clientX: 340, clientY: 60 }), [2, 1.2, 0.5]);
  assert.equal(gui.targetOf({ target: canvas, pointerId: 2, clientX: 340, clientY: 60 }), null);
  gui.release(1);
  assert.equal(gui.targetOf({ target: canvas, pointerId: 1, clientX: 340, clientY: 60 }), null);
});

test("a pointerdown on a Slider or a Joystick captures and holds it; a cancel releases", () => {
  const { gui, slider, joystick, button } = built();
  slider.fire("pointerdown", { pointerId: 5 });
  joystick.fire("pointerdown", { pointerId: 6 });
  button.fire("pointerdown", { pointerId: 7 });
  assert.deepEqual([slider.captured, joystick.captured, button.captured], [[5], [6], []]);
  const away = { target: { parentNode: null }, clientX: 0, clientY: 0 };
  assert.equal(gui.targetOf({ ...away, pointerId: 5 })[0], 2);
  assert.equal(gui.targetOf({ ...away, pointerId: 6 })[0], 4);
  assert.equal(gui.targetOf({ ...away, pointerId: 7 }), null);
  joystick.fire("pointercancel", { pointerId: 6 });
  assert.equal(gui.targetOf({ ...away, pointerId: 6 }), null);
});

test("a frame-code write updates the label, the knob and aria-valuenow, and only what changed", () => {
  const { gui, button, slider, joystick } = built();
  const w = widgetObjects();
  const fill = slider.children[0], knob = joystick.children[0];
  const before = [button.writes, slider.attrWrites, countWrites(fill), countWrites(knob)];
  gui.sync(w);
  assert.deepEqual([button.writes, slider.attrWrites, before[2](), before[3]()], [before[0], before[1], 0, 0]);
  w[1].label = "Again";
  w[2].value = 3;
  w[4].value = [0, 0];
  gui.sync(w);
  assert.equal(button.textContent, "Again");
  assert.equal(slider.attrs["aria-valuenow"], "3");
  assert.equal(fill.props.width, "75%");
  assert.equal(knob.props.left, `${96 - 0.8 * 96 * 0.5}px`);
  assert.equal(button.writes, before[0] + 1);
  assert.ok(before[2]() > 0 && before[3]() > 0);
  w[2].max = 6;
  gui.sync(w);
  assert.equal(slider.attrs["aria-valuemax"], "6");
  assert.equal(fill.props.width, "50%");
});
