// Browser input: DOM events to one Frame per animation frame (the form ScriptInput.setFrame
// reads, as the replay feeds it), and pointer lock as the engine's capture hook.

// rendercanvas's KEYMAP, then manifoldx.input._key_name: a letter lowercase, the space bar
// "Space".
const KEYMAP = { Ctrl: "Control", Del: "Delete", Esc: "Escape" };
const keyName = (key) => {
  const k = KEYMAP[key] ?? key;
  return k === " " ? "Space" : k.length === 1 ? k.toLowerCase() : k;
};

// rendercanvas's MOUSE_BUTTON_MAP: DOM button to manifoldx button.
const BUTTONS = { 0: 1, 1: 3, 2: 2, 3: 4, 4: 5 };
const buttonOf = (b) => BUTTONS[b] ?? b;

const MODS = [["altKey", "Alt"], ["ctrlKey", "Control"], ["metaKey", "Meta"], ["shiftKey", "Shift"]];

// Wheel deltaMode to pixels: pixel (divided by the device pixel ratio), line, page, as
// rendercanvas's browser backend scales them.
const WHEEL_LINE = 16, WHEEL_PAGE = 600;

const sortedNames = (set) => [...set].sort();
const sortedNumbers = (set) => [...set].sort((a, b) => a - b);

// Events in, one Frame out per take(): keys and buttons held at the end of the frame, those
// pressed (just, clicks) and released (up, released) during it, the motion (d) and wheel
// summed over it, the last pointer position and the modifiers of the latest event that
// carries them. A key is tracked by its code and named from its key at keydown, so Shift
// released before "2" still releases the "@" it pressed.
export class FrameBuilder {
  constructor({ devicePixelRatio = 1 } = {}) {
    this.devicePixelRatio = devicePixelRatio;
    this.lastPointerType = null;
    this._held = new Map(); // code -> name
    this._buttons = new Set();
    this._pos = [0, 0];
    this._mods = [];
    this._reset();
  }

  _reset() {
    this._just = new Set();
    this._up = new Set();
    this._clicks = new Set();
    this._released = new Set();
    this._d = [0, 0];
    this._wheel = [0, 0];
    this._lockEnded = false;
  }

  _modsFrom(ev) {
    if (!MODS.some(([flag]) => typeof ev[flag] === "boolean")) return;
    this._mods = MODS.filter(([flag]) => ev[flag]).map(([, name]) => name);
  }

  keyDown(ev) {
    this._modsFrom(ev);
    if (ev.repeat || this._held.has(ev.code)) return;
    const name = keyName(ev.key);
    this._held.set(ev.code, name);
    this._just.add(name);
  }

  keyUp(ev) {
    this._modsFrom(ev);
    const name = this._held.get(ev.code);
    if (name === undefined) return;
    this._held.delete(ev.code);
    this._up.add(name);
  }

  pointerDown(ev) {
    this._modsFrom(ev);
    this.lastPointerType = ev.pointerType ?? null;
    const b = buttonOf(ev.button);
    if (!this._buttons.has(b)) this._clicks.add(b);
    this._buttons.add(b);
    this._pos = [ev.x, ev.y];
  }

  pointerUp(ev) {
    this._modsFrom(ev);
    const b = buttonOf(ev.button);
    if (this._buttons.delete(b)) this._released.add(b);
    this._pos = [ev.x, ev.y];
  }

  pointerMove(ev) {
    this._modsFrom(ev);
    if (ev.pointerType) this.lastPointerType = ev.pointerType;
    this._d[0] += ev.movementX;
    this._d[1] += ev.movementY;
    this._pos = [ev.x, ev.y];
  }

  wheel(ev) {
    this._modsFrom(ev);
    const scale = [1 / this.devicePixelRatio, WHEEL_LINE, WHEEL_PAGE][ev.deltaMode] ?? 1;
    this._wheel[0] += ev.deltaX * scale;
    this._wheel[1] += ev.deltaY * scale;
  }

  // The page lost focus: nothing is held any more, and the next frame says so.
  blur() {
    for (const name of this._held.values()) this._up.add(name);
    for (const b of this._buttons) this._released.add(b);
    this._held.clear();
    this._buttons.clear();
    this._mods = [];
  }

  // The browser ended the pointer lock (Escape, which Chromium keeps from the page): the next
  // frame's just holds "Escape", as the desktop's Escape keydown would.
  lockEnded() {
    this._lockEnded = true;
  }

  take(viewport) {
    if (this._lockEnded) this._just.add("Escape");
    const frame = {
      keys: sortedNames(new Set(this._held.values())),
      just: sortedNames(this._just),
      d: [...this._d],
      buttons: sortedNumbers(this._buttons),
      clicks: sortedNumbers(this._clicks),
      vp: [viewport[0], viewport[1]],
      up: sortedNames(this._up),
      released: sortedNumbers(this._released),
      pos: [...this._pos],
      wheel: [...this._wheel],
      mods: [...this._mods],
    };
    this._reset();
    return frame;
  }
}

// Feed builder from the DOM: keys and blur on the window, pointerdown on the canvas (captured,
// so a drag that leaves it keeps reporting), pointermove/pointerup on the window, the wheel
// and the context menu on the canvas. Positions are CSS pixels from the canvas's top-left.
// Tab and Space are kept from the page (no focus move, no scroll); browser shortcuts such as
// Control+W are left alone. Returns detach().
export function attachDom(canvas, builder, { devicePixelRatio } = {}) {
  const win = canvas.ownerDocument.defaultView;
  const at = (ev) => {
    const r = canvas.getBoundingClientRect();
    return { x: ev.clientX - r.left, y: ev.clientY - r.top };
  };
  const listeners = [
    [win, "keydown", (ev) => {
      if (ev.code === "Tab" || ev.code === "Space") ev.preventDefault();
      builder.keyDown(ev);
    }],
    [win, "keyup", (ev) => builder.keyUp(ev)],
    [win, "blur", () => builder.blur()],
    [canvas, "pointerdown", (ev) => {
      canvas.setPointerCapture(ev.pointerId);
      builder.pointerDown({ ...pick(ev), button: ev.button, pointerType: ev.pointerType, ...at(ev) });
    }],
    [win, "pointermove", (ev) => {
      builder.pointerMove({ ...pick(ev), movementX: ev.movementX, movementY: ev.movementY, pointerType: ev.pointerType, ...at(ev) });
    }],
    [win, "pointerup", (ev) => builder.pointerUp({ ...pick(ev), button: ev.button, ...at(ev) })],
    [canvas, "wheel", (ev) => {
      ev.preventDefault();
      builder.devicePixelRatio = devicePixelRatio ?? win.devicePixelRatio ?? 1;
      builder.wheel(ev);
    }, { passive: false }],
    [canvas, "contextmenu", (ev) => ev.preventDefault()],
  ];
  for (const [target, type, fn, opts] of listeners) target.addEventListener(type, fn, opts);
  return () => {
    for (const [target, type, fn, opts] of listeners) target.removeEventListener(type, fn, opts);
  };
}

// The modifier flags of a DOM event, which spreading an Event does not copy.
const pick = (ev) => Object.fromEntries(MODS.map(([flag]) => [flag, ev[flag]]));

// Pointer lock as the engine's capture hook: request(on) is hooks.capture. A lock needs a user
// gesture: with an active one it is requested at once, otherwise at the next pointerdown on
// the canvas. Where the canvas has no pointer lock, or the last pointer was a touch, it refuses
// (false), as the desktop does with no window. When the lock ends without the app asking
// (Escape, an error), the builder's next frame holds Escape and input.lockLost() runs; set
// input to the engine's once loadApp made it.
export class PointerLock {
  constructor(canvas, doc, builder) {
    this.canvas = canvas;
    this.doc = doc;
    this.builder = builder;
    this.input = null;
    this.locked = false;
    this._releasing = false;
    this._armed = null;
    doc.addEventListener("pointerlockchange", () => {
      this.locked = doc.pointerLockElement === canvas;
      if (this.locked) return;
      if (this._releasing) this._releasing = false;
      else this._lost();
    });
    doc.addEventListener("pointerlockerror", () => this._lost());
  }

  _lost() {
    this.locked = false;
    this.builder.lockEnded();
    if (this.input) this.input.lockLost();
  }

  _activation() {
    return this.doc.userActivation ?? this.doc.defaultView?.navigator?.userActivation;
  }

  _lock() {
    const done = this.canvas.requestPointerLock();
    // Recent browsers return a promise; the refusal also fires pointerlockerror.
    if (done && typeof done.catch === "function") done.catch(() => {});
  }

  _disarm() {
    if (this._armed) this.canvas.removeEventListener("pointerdown", this._armed);
    this._armed = null;
  }

  request(on) {
    if (!on) {
      this._disarm();
      if (this.locked) {
        this._releasing = true;
        this.doc.exitPointerLock();
      }
      return true;
    }
    if (typeof this.canvas.requestPointerLock !== "function" || this.builder.lastPointerType === "touch") return false;
    if (this.locked) return true;
    const activation = this._activation();
    if (!activation || activation.isActive) {
      this._disarm();
      this._lock();
    } else if (!this._armed) {
      this._armed = () => {
        this._armed = null;
        this._lock();
      };
      this.canvas.addEventListener("pointerdown", this._armed, { once: true });
    }
    return true;
  }
}
