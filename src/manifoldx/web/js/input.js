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
// released before "2" still releases the "@" it pressed. events lists the desktop's input
// events in the order the DOM gave them, as manifoldx.input._InputBridge builds them:
// ["key_down" | "key_up", {key, modifiers, is_down}], ["pointer_down" | "pointer_up" |
// "pointer_move", {x, y, dx, dy, button, buttons, modifiers, phase, pointer_id}], ["wheel",
// {dx, dy, x, y, buttons, modifiers}] and, at take(), ["resize", {width, height, pixel_ratio}];
// an outside event (emit: a host page's message, an agent's manifoldx emit) goes in among them
// in arrival order, its payload plain JSON.
// clock is the wall clock at take(), Date.now() / 1000, never less than the last frame's, so
// time.time() in frame code does not run backwards when the system clock steps back.
export class FrameBuilder {
  constructor({ devicePixelRatio = 1 } = {}) {
    this.devicePixelRatio = devicePixelRatio;
    this.lastPointerType = null;
    this._held = new Map(); // code -> name
    this._buttons = new Set();
    this._pos = [0, 0];
    this._last = null; // the last pointer event's position, which a move's dx and dy are from
    this._mods = [];
    this._viewport = null;
    this._clock = 0;
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
    this._events = [];
  }

  _modsFrom(ev) {
    if (!MODS.some(([flag]) => typeof ev[flag] === "boolean")) return;
    this._mods = MODS.filter(([flag]) => ev[flag]).map(([, name]) => name);
  }

  _key(name, key) {
    this._events.push([name, { key, modifiers: [...this._mods], is_down: name === "key_down" }]);
  }

  // A pointer event as _InputBridge._build_pointer_event makes it: dx and dy from the last
  // pointer event's position on a move, 0 on a down or up; the position re-anchored on each.
  // target, the GUI's [widget, fx, fy] for the event, goes in as its "gui" when there is one.
  _pointer(phase, ev, button, target = null) {
    const [x, y] = this._pos;
    const moved = phase === "move" && this._last !== null;
    const dx = moved ? x - this._last[0] : 0, dy = moved ? y - this._last[1] : 0;
    this._last = [x, y];
    const pointer_id = ev.isPrimary === false ? ev.pointerId : 0;
    const payload = {
      x, y, dx, dy, button, buttons: sortedNumbers(this._buttons), modifiers: [...this._mods], phase, pointer_id,
    };
    if (target !== null) payload.gui = target;
    this._events.push([`pointer_${phase}`, payload]);
  }

  keyDown(ev) {
    this._modsFrom(ev);
    if (ev.repeat || this._held.has(ev.code)) return;
    const name = keyName(ev.key);
    this._held.set(ev.code, name);
    this._just.add(name);
    this._key("key_down", name);
  }

  keyUp(ev) {
    this._modsFrom(ev);
    const name = this._held.get(ev.code);
    if (name === undefined) return;
    this._held.delete(ev.code);
    this._up.add(name);
    this._key("key_up", name);
  }

  pointerDown(ev, target = null) {
    this._modsFrom(ev);
    this.lastPointerType = ev.pointerType ?? null;
    const b = buttonOf(ev.button);
    if (!this._buttons.has(b)) this._clicks.add(b);
    this._buttons.add(b);
    this._pos = [ev.x, ev.y];
    this._pointer("down", ev, b, target);
  }

  pointerUp(ev, target = null) {
    this._modsFrom(ev);
    const b = buttonOf(ev.button);
    if (this._buttons.delete(b)) this._released.add(b);
    this._pos = [ev.x, ev.y];
    this._pointer("up", ev, b, target);
  }

  pointerMove(ev, target = null) {
    this._modsFrom(ev);
    if (ev.pointerType) this.lastPointerType = ev.pointerType;
    this._d[0] += ev.movementX;
    this._d[1] += ev.movementY;
    this._pos = [ev.x, ev.y];
    this._pointer("move", ev, 0, target);
  }

  wheel(ev) {
    this._modsFrom(ev);
    const scale = [1 / this.devicePixelRatio, WHEEL_LINE, WHEEL_PAGE][ev.deltaMode] ?? 1;
    const dx = ev.deltaX * scale, dy = ev.deltaY * scale;
    this._wheel[0] += dx;
    this._wheel[1] += dy;
    const [x, y] = this._pos;
    this._events.push(["wheel", { dx, dy, x, y, buttons: sortedNumbers(this._buttons), modifiers: [...this._mods] }]);
  }

  // An event from outside the page's input, recorded as it arrives; advance's eventOf reads
  // its JSON payload.
  emit(name, payload = null) {
    this._events.push([name, payload]);
  }

  // The page lost focus: nothing is held any more, and the next frame says so, with a key_up
  // per held key and a pointer_up per held button.
  blur() {
    this._mods = [];
    for (const name of this._held.values()) {
      this._up.add(name);
      this._key("key_up", name);
    }
    this._held.clear();
    for (const b of sortedNumbers(this._buttons)) {
      this._released.add(b);
      this._buttons.delete(b);
      this._pointer("up", {}, b);
    }
  }

  // The browser ended the pointer lock (Escape, which Chromium keeps from the page): the next
  // frame's just holds "Escape", as the desktop's Escape keydown would.
  lockEnded() {
    this._lockEnded = true;
  }

  take(viewport) {
    if (this._lockEnded) this._just.add("Escape");
    const [w, h] = viewport;
    if (this._viewport !== null && (w !== this._viewport[0] || h !== this._viewport[1])) {
      this._events.push(["resize", { width: w, height: h, pixel_ratio: this.devicePixelRatio }]);
    }
    this._viewport = [w, h];
    this._clock = Math.max(Date.now() / 1000, this._clock);
    const frame = {
      keys: sortedNames(new Set(this._held.values())),
      just: sortedNames(this._just),
      d: [...this._d],
      buttons: sortedNumbers(this._buttons),
      clicks: sortedNumbers(this._clicks),
      vp: [w, h],
      up: sortedNames(this._up),
      released: sortedNumbers(this._released),
      pos: [...this._pos],
      wheel: [...this._wheel],
      mods: [...this._mods],
      events: this._events,
      clock: this._clock,
    };
    this._reset();
    return frame;
  }
}

// Feed builder from the DOM: keys and blur on the window, pointerdown on the canvas (captured,
// so a drag that leaves it keeps reporting) and on gui's layer, pointermove/pointerup on the
// window, the wheel and the context menu on the canvas. With a gui (buildGui's, or anything
// with its layer, targetOf and release), each pointer event carries gui.targetOf(ev), the
// widget it went to, and a pointerup releases the widget's hold once it has read it. A
// pointer down on the GUI is a click in the frame too, as the desktop's input layer records
// every pointer event before the GUI sees it. Positions are CSS pixels from the canvas's
// top-left. Tab and Space are kept from the page (no focus move, no scroll); browser shortcuts
// such as Control+W are left alone. Returns detach().
export function attachDom(canvas, builder, { devicePixelRatio, gui = null } = {}) {
  const win = canvas.ownerDocument.defaultView;
  const at = (ev) => {
    const r = canvas.getBoundingClientRect();
    return { x: ev.clientX - r.left, y: ev.clientY - r.top };
  };
  const targetOf = (ev) => (gui === null ? null : gui.targetOf(ev));
  const down = (ev) => builder.pointerDown({ ...pick(ev), button: ev.button, pointerType: ev.pointerType, ...at(ev) }, targetOf(ev));
  const listeners = [
    [win, "keydown", (ev) => {
      if (ev.code === "Tab" || ev.code === "Space") ev.preventDefault();
      builder.keyDown(ev);
    }],
    [win, "keyup", (ev) => builder.keyUp(ev)],
    [win, "blur", () => builder.blur()],
    [canvas, "pointerdown", (ev) => {
      down(ev);
      // A locked pointer cannot be captured (Chromium throws InvalidStateError), nor needs to be.
      if (!canvas.ownerDocument.pointerLockElement) canvas.setPointerCapture(ev.pointerId);
    }],
    [win, "pointermove", (ev) => {
      builder.pointerMove({ ...pick(ev), movementX: ev.movementX, movementY: ev.movementY, pointerType: ev.pointerType, ...at(ev) }, targetOf(ev));
    }],
    [win, "pointerup", (ev) => {
      builder.pointerUp({ ...pick(ev), button: ev.button, ...at(ev) }, targetOf(ev));
      gui?.release(ev.pointerId);
    }],
    [canvas, "wheel", (ev) => {
      ev.preventDefault();
      builder.devicePixelRatio = devicePixelRatio ?? win.devicePixelRatio ?? 1;
      builder.wheel(ev);
    }, { passive: false }],
    [canvas, "contextmenu", (ev) => ev.preventDefault()],
  ];
  if (gui !== null) listeners.push([gui.layer, "pointerdown", down]);
  for (const [target, type, fn, opts] of listeners) target.addEventListener(type, fn, opts);
  return () => {
    for (const [target, type, fn, opts] of listeners) target.removeEventListener(type, fn, opts);
  };
}

// The modifier flags and the pointer's identity of a DOM event, which spreading an Event does not copy.
const pick = (ev) => ({
  ...Object.fromEntries(MODS.map(([flag]) => [flag, ev[flag]])),
  pointerId: ev.pointerId,
  isPrimary: ev.isPrimary,
});

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
