// The runtime the transpiled modules import as $rt: the Python semantics JavaScript lacks.

const isArr = (x) => Array.isArray(x) || ArrayBuffer.isView(x);
// A dict is an object with a null prototype (or a plain literal): prototype names are plain keys.
const isDict = (x) => typeof x === "object" && x !== null
  && [Object.prototype, null].includes(Object.getPrototypeOf(x));

export function dict(pairs) {
  const d = Object.create(null);
  for (const [k, v] of pairs) d[k] = v;
  return d;
}

// Python's truth: empty containers and strings are false, NaN is true.
export function truthy(x) {
  if (x === null || x === undefined) return false;
  if (typeof x === "number") return x !== 0;
  if (typeof x === "boolean") return x;
  if (typeof x === "string" || isArr(x)) return x.length > 0;
  if (x instanceof Set || x instanceof Map) return x.size > 0;
  if (isDict(x)) return Object.keys(x).length > 0;
  return true;
}

// `a or b or c` returns an operand; the operands after the first are thunks, evaluated lazily.
export const or_ = (a, ...rest) => { for (const f of rest) { if (truthy(a)) return a; a = f(); } return a; };
export const and_ = (a, ...rest) => { for (const f of rest) { if (!truthy(a)) return a; a = f(); } return a; };

// CPython's float divmod (Objects/floatobject.c, _float_div_mod): the remainder takes the
// divisor's sign and the quotient is floored, exact where floor(a / b) rounds wrong
// (1 // 0.1 is 9.0). Exact for integers below 2**53 too.
function divmod(vx, wx) {
  if (wx === 0) throw new Error("ZeroDivisionError: division or modulo by zero");
  let mod = vx % wx;
  let div = (vx - mod) / wx;
  if (mod) {
    if ((wx < 0) !== (mod < 0)) { mod += wx; div -= 1; }
  } else {
    mod = wx < 0 ? -0 : 0;
  }
  let floordiv;
  if (div) {
    floordiv = Math.floor(div);
    if (div - floordiv > 0.5) floordiv += 1;
  } else {
    const q = vx / wx;
    floordiv = q < 0 || Object.is(q, -0) ? -0 : 0;
  }
  return [floordiv, mod];
}
export const mod = (a, b) => divmod(a, b)[1];
export const floordiv = (a, b) => divmod(a, b)[0];

export function eq(a, b) {
  if (a === undefined) a = null;
  if (b === undefined) b = null;
  if (typeof a === "boolean") a = +a;   // True == 1
  if (typeof b === "boolean") b = +b;
  if (a === b) return true;
  if (isArr(a) && isArr(b)) return a.length === b.length && Array.prototype.every.call(a, (x, i) => eq(x, b[i]));
  if (a instanceof Set && b instanceof Set) return a.size === b.size && [...a].every((x) => b.has(x));
  if (isDict(a) && isDict(b)) {
    const ka = Object.keys(a);
    return ka.length === Object.keys(b).length && ka.every((k) => Object.hasOwn(b, k) && eq(a[k], b[k]));
  }
  return false;
}

export function contains(c, x) {
  if (c instanceof Set) return c.has(x);
  if (isArr(c)) return Array.prototype.some.call(c, (y) => eq(x, y));
  if (typeof c === "string") return c.includes(x);
  if (isDict(c)) return Object.hasOwn(c, x);
  throw new TypeError("argument of type is not iterable");
}

// len counts code points in a string, as Python does; JS's length counts UTF-16 units.
export function len(x) {
  if (typeof x === "string") { let n = 0; for (const _ of x) n++; return n; }
  if (isArr(x)) return x.length;
  if (x instanceof Set || x instanceof Map) return x.size;
  if (isDict(x)) return Object.keys(x).length;
  throw new TypeError("object has no len()");
}

export function range(start, stop, step = 1) {
  if (stop === undefined) { stop = start; start = 0; }
  if (step === 0) throw new Error("ValueError: range() arg 3 must not be zero");
  const out = [];
  for (let i = start; step > 0 ? i < stop : i > stop; i += step) out.push(i);
  return out;
}

// What a for loop walks: a dict's keys, anything else as it is.
export const iter = (x) => (isDict(x) ? Object.keys(x) : x);
export const list = (x) => Array.from(iter(x));
export const any_ = (xs) => { for (const x of iter(xs)) if (truthy(x)) return true; return false; };
export const all_ = (xs) => { for (const x of iter(xs)) if (!truthy(x)) return false; return true; };

function pick(xs, key, better) {
  const items = [...iter(xs)];
  if (!items.length) throw new Error("ValueError: min() or max() arg is an empty sequence");
  let best = items[0], bk = key ? key(best) : best;
  for (const x of items.slice(1)) {
    const k = key ? key(x) : x;
    if (better(k, bk)) { best = x; bk = k; }
  }
  return best;   // the first of equals, as Python
}
export const min_ = (xs, key = null) => pick(xs, key, (a, b) => a < b);
export const max_ = (xs, key = null) => pick(xs, key, (a, b) => a > b);

export const lrep = (xs, n) => Array.from({ length: Math.max(0, n) }, () => xs).flat();

function index(c, k) {
  const n = isArr(c) ? c.length : len(c);
  const i = k < 0 ? k + n : k;
  if (!Number.isInteger(i) || i < 0 || i >= n) throw new Error("IndexError: index out of range");
  return i;
}

export function getitem(c, k) {
  if (isArr(c)) return c[index(c, k)];
  if (typeof c === "string") return [...c][index(c, k)];
  if (isDict(c)) {
    if (!Object.hasOwn(c, k)) throw new Error(`KeyError: ${k}`);
    return c[k];
  }
  throw new TypeError("object is not subscriptable");
}

export function setitem(c, k, v) {
  if (isArr(c)) c[index(c, k)] = v;
  else if (isDict(c)) c[k] = v;
  else throw new TypeError("object does not support item assignment");
}
