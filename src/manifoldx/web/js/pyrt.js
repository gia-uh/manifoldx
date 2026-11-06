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

// ---------------------------------------------------------------- math and small numpy vectors

// CPython multiplies by a constant computed once (Modules/mathmodule.c); x * Math.PI / 180
// rounds differently.
const DEG_TO_RAD = Math.PI / 180, RAD_TO_DEG = 180 / Math.PI;
export const radians = (x) => x * DEG_TO_RAD;
export const degrees = (x) => x * RAD_TO_DEG;

// A float64 array is a Float64Array, a float32 one a Float32Array. Arithmetic broadcasts a
// number against an array; the "32" variants hold float32 results and round a Python number
// to float32 before using it, as numpy does with a float32 array and a Python scalar.
function vop(f, Out) {
  const cast = Out === Float32Array ? Math.fround : (x) => x;
  return (a, b) => {
    const A = isArr(a), B = isArr(b);
    if (A && B && a.length !== b.length) {
      throw new Error(`ValueError: operands could not be broadcast together (${a.length}) (${b.length})`);
    }
    const n = A ? a.length : b.length, out = new Out(n);
    for (let i = 0; i < n; i++) out[i] = f(cast(A ? a[i] : a), cast(B ? b[i] : b));
    return out;
  };
}
const add = (x, y) => x + y, sub = (x, y) => x - y, mul = (x, y) => x * y, div = (x, y) => x / y;
export const vadd = vop(add, Float64Array), vsub = vop(sub, Float64Array);
export const vmul = vop(mul, Float64Array), vdiv = vop(div, Float64Array);
export const vadd32 = vop(add, Float32Array), vsub32 = vop(sub, Float32Array);
export const vmul32 = vop(mul, Float32Array), vdiv32 = vop(div, Float32Array);
export const vneg = (a) => a.map((x) => -x);

// numpy's a += b writes into a, so an alias of a sees the change. A typed array rounds what
// it stores to its own precision.
function viop(f, cast) {
  return (a, b) => {
    const B = isArr(b);
    if (B && a.length !== b.length) {
      throw new Error(`ValueError: operands could not be broadcast together (${a.length}) (${b.length})`);
    }
    for (let i = 0; i < a.length; i++) a[i] = f(a[i], cast(B ? b[i] : b));
    return a;
  };
}
const same = (x) => x;
export const viadd = viop(add, same), visub = viop(sub, same);
export const vimul = viop(mul, same), vidiv = viop(div, same);
export const viadd32 = viop(add, Math.fround), visub32 = viop(sub, Math.fround);
export const vimul32 = viop(mul, Math.fround), vidiv32 = viop(div, Math.fround);

// np.linalg.norm is sqrt(dot(a, a)) through BLAS. OpenBLAS's ddot sums the products in order
// below 16 items; its sdot sums float32 products in a double below 32 items. Longer arrays
// go through SIMD kernels whose order this does not follow.
export function vnorm(a) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * a[i];
  return Math.sqrt(s);
}
export function vnorm32(a) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += Math.fround(a[i] * a[i]);
  return Math.fround(Math.sqrt(Math.fround(s)));
}

// np.arange fills start, start + step, then start + i * delta with delta the difference of
// those two, and has ceil((stop - start) / step) items.
export function arange(start, stop, step = 1) {
  if (stop === undefined) { stop = start; start = 0; }
  if (step === 0) throw new Error("ZeroDivisionError: arange() step is zero");
  const n = Math.max(0, Math.ceil((stop - start) / step)), out = new Float64Array(n);
  if (n > 0) out[0] = start;
  if (n > 1) out[1] = start + step;
  const delta = out[1] - out[0];
  for (let i = 2; i < n; i++) out[i] = start + i * delta;
  return out;
}
