// The runtime the transpiled modules import as $rt: the Python semantics JavaScript lacks.

const isArr = (x) => Array.isArray(x) || ArrayBuffer.isView(x);
// A dict is a Map: it keeps Python's insertion order (an object puts "2" and "10" before
// "a"), and a key named like a prototype member ("constructor", "__proto__") is a plain key.
const isDict = (x) => x instanceof Map;

export const dict = (pairs) => new Map(pairs);

// Python's truth: empty containers and strings are false, NaN is true.
export function truthy(x) {
  if (x === null || x === undefined) return false;
  if (typeof x === "number") return x !== 0;
  if (typeof x === "boolean") return x;
  if (typeof x === "string" || isArr(x)) return x.length > 0;
  if (x instanceof Set || x instanceof Map) return x.size > 0;
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
    return a.size === b.size && [...a].every(([k, v]) => b.has(k) && eq(v, b.get(k)));
  }
  return false;
}

export function contains(c, x) {
  if (c instanceof Set) return c.has(x);
  if (isArr(c)) return Array.prototype.some.call(c, (y) => eq(x, y));
  if (typeof c === "string") return c.includes(x);
  if (isDict(c)) return c.has(x);
  throw new TypeError("argument of type is not iterable");
}

// len counts code points in a string, as Python does; JS's length counts UTF-16 units.
export function len(x) {
  if (typeof x === "string") return cplen(x);
  if (isArr(x)) return x.length;
  if (x instanceof Set || x instanceof Map) return x.size;
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
export const iter = (x) => (isDict(x) ? x.keys() : x);
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
    if (!c.has(k)) throw new Error(`KeyError: '${k}'`);
    return c.get(k);
  }
  throw new TypeError("object is not subscriptable");
}

export function setitem(c, k, v) {
  if (isArr(c)) c[index(c, k)] = v;
  else if (isDict(c)) c.set(k, v);
  else throw new TypeError("object does not support item assignment");
}

// ---------------------------------------------------------------- math and small numpy vectors

// CPython multiplies by a constant computed once (Modules/mathmodule.c); x * Math.PI / 180
// rounds differently.
const DEG_TO_RAD = Math.PI / 180, RAD_TO_DEG = 180 / Math.PI;
export const radians = (x) => x * DEG_TO_RAD;
export const degrees = (x) => x * RAD_TO_DEG;

// A component's constructor (Transform(pos=...), Mesh(g), Velocity(v=...)) as a record of
// its values by field, which engine.spawn turns into rows as the component's get_data does.
export const component = (name, values) => ({ component: name, values });

// Transform.rotation(x, y, z) of Python numbers: numpy computes the half angles' cos and sin
// and the products in float64, then stacks them as float32 (x, y, z, w).
export function quat_euler(x, y, z) {
  const cx = Math.cos(x / 2), sx = Math.sin(x / 2);
  const cy = Math.cos(y / 2), sy = Math.sin(y / 2);
  const cz = Math.cos(z / 2), sz = Math.sin(z / 2);
  return Float32Array.of(
    sx * cy * cz - cx * sy * sz,
    cx * sy * cz + sx * cy * sz,
    cx * cy * sz - sx * sy * cz,
    cx * cy * cz + sx * sy * sz,
  );
}

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

// ---------------------------------------------------------------- numpy arrays of any rank

// An NdArray is a C-contiguous numpy array: data a typed array of its dtype ("f64", "f32",
// "i32" or "bool", held as 0 and 1), shape its dimensions. A bare Float32Array or
// Float64Array (the small vectors above) is a rank-1 array wherever $rt.nd takes an array.
// Every operation writes a new typed array of its result's dtype, so each element rounds
// as numpy's does; the transpiler passes the dtype numpy 2 computes in, which the page
// cannot tell from the values (a Python 2 and 2.0 are one number here).
const TYPED = { f64: Float64Array, f32: Float32Array, i32: Int32Array, bool: Uint8Array };
// A value cast to a dtype, as numpy casts each operand before it computes: a Python int
// against int32 wraps, as numpy's does.
const CAST = { f64: Number, f32: Math.fround, i32: (x) => x | 0, bool: (x) => (x ? 1 : 0) };

export class NdArray {
  constructor(data, shape, dtype) {
    this.data = data;
    this.shape = shape;
    this.dtype = dtype;
  }

  // The trace writes an array as its nested lists, as Python's json writes tolist().
  toJSON() {
    return tolist(this);
  }
}

const asnd = (x) =>
  x instanceof NdArray ? x
  : x instanceof Float32Array ? new NdArray(x, [x.length], "f32")
  : x instanceof Float64Array ? new NdArray(x, [x.length], "f64")
  : x;

const shapeText = (s) => `(${s.join(",")}${s.length === 1 ? "," : ""})`;
const sizeOf = (shape) => shape.reduce((n, d) => n * d, 1);

// The shape of a op b: a scalar against anything, equal shapes, or a shape against one that
// ends with it (a (c,) row against (k, c) rows). Other shapes numpy may broadcast (a 1
// dimension) are outside the subset and throw numpy's message too.
function broadcast(a, b) {
  if (!(a instanceof NdArray)) return b.shape;
  if (!(b instanceof NdArray)) return a.shape;
  const [long, short] = a.shape.length >= b.shape.length ? [a.shape, b.shape] : [b.shape, a.shape];
  const tail = long.slice(long.length - short.length);
  if (tail.some((d, i) => d !== short[i])) {
    throw new Error(`ValueError: operands could not be broadcast together with shapes ${shapeText(a.shape)} ${shapeText(b.shape)}`);
  }
  return long;
}

// f over a and b broadcast, both cast to dtype; the result of dtype `out` (dtype's own when
// null). An operand that is shorter than the result repeats, as a trailing row does.
function binary(f, out = null, fi32 = null) {
  return (x, y, dtype) => {
    const a = asnd(x), b = asnd(y), cast = CAST[dtype];
    const g = dtype === "i32" && fi32 ? fi32 : f;
    const shape = broadcast(a, b), n = sizeOf(shape);
    const res = new TYPED[out ?? dtype](n);
    const A = a instanceof NdArray ? a.data : null, B = b instanceof NdArray ? b.data : null;
    const sa = A ? null : cast(a), sb = B ? null : cast(b);
    const na = A ? A.length : 1, nb = B ? B.length : 1;
    for (let i = 0; i < n; i++) {
      res[i] = g(A ? cast(A[i % na]) : sa, B ? cast(B[i % nb]) : sb);
    }
    return new NdArray(res, [...shape], out ?? dtype);
  };
}

// f over each element of a, cast to dtype, into an array of dtype.
function unary(f) {
  return (x, dtype) => {
    const a = asnd(x), d = dtype ?? a.dtype, cast = CAST[d];
    const res = new TYPED[d](a.data.length);
    for (let i = 0; i < res.length; i++) res[i] = f(cast(a.data[i]));
    return new NdArray(res, [...a.shape], d);
  };
}

const full = (shape, value, dtype) => {
  const dims = typeof shape === "number" ? [shape] : [...shape];
  return new NdArray(new TYPED[dtype](sizeOf(dims)).fill(CAST[dtype](value)), dims, dtype);
};

// np.arange with a dtype: numpy writes start and start + step cast to it, then fills
// start + i * delta in the dtype, delta being the difference of those two.
function ndArange(start, stop, step, dtype) {
  if (step === 0) throw new Error("ZeroDivisionError: arange() step is zero");
  const n = Math.max(0, Math.ceil((stop - start) / step)), cast = CAST[dtype];
  const out = new TYPED[dtype](n);
  if (n > 0) out[0] = cast(start);
  if (n > 1) out[1] = cast(start + step);
  const delta = cast(out[1] - out[0]);
  for (let i = 2; i < n; i++) out[i] = dtype === "i32" ? out[0] + i * delta : cast(out[0] + cast(i * delta));
  return new NdArray(out, [n], dtype);
}

// a.tolist(): nested lists of numbers, booleans for a bool array.
function tolist(x) {
  const a = asnd(x);
  const item = a.dtype === "bool" ? (v) => v === 1 : Number;
  const rows = (start, dims) => {
    if (dims.length === 1) return Array.from(a.data.subarray(start, start + dims[0]), item);
    const step = sizeOf(dims.slice(1));
    return Array.from({ length: dims[0] }, (_, i) => rows(start + i * step, dims.slice(1)));
  };
  return rows(0, a.shape);
}

// numpy's pairwise sum (numpy/_core/src/umath/loops_utils.h.src, pairwise_sum) over
// data[start, start + n), every addition cast to the dtype: below 8 items left to right, up
// to 128 in eight accumulators, past that split at n / 2 rounded down to a multiple of 8.
const BLOCK = 8;
function pairwise(data, start, n, cast) {
  if (n < BLOCK) {
    let res = -0;
    for (let i = 0; i < n; i++) res = cast(res + cast(data[start + i]));
    return res;
  }
  if (n <= 128) {
    const r = Array.from({ length: BLOCK }, (_, j) => cast(data[start + j]));
    let i = BLOCK;
    for (; i < n - (n % BLOCK); i += BLOCK) {
      for (let j = 0; j < BLOCK; j++) r[j] = cast(r[j] + cast(data[start + i + j]));
    }
    let res = cast(cast(cast(r[0] + r[1]) + cast(r[2] + r[3])) + cast(cast(r[4] + r[5]) + cast(r[6] + r[7])));
    for (; i < n; i++) res = cast(res + cast(data[start + i]));
    return res;
  }
  let n2 = Math.floor(n / 2);
  n2 -= n2 % 8;
  return cast(pairwise(data, start, n2, cast) + pairwise(data, start + n2, n - n2, cast));
}

// A reduction of a over axis (null: every element, flat), as numpy runs it on a C-contiguous
// array: axis 1 of (k, c) reduces each row with `row`, axis 0 folds row after row with `step`.
function reduce(row, step) {
  return (x, axis, dtype) => {
    const a = asnd(x), cast = CAST[dtype], data = a.data;
    if (axis === null || a.shape.length === 1) return row(data, 0, data.length, cast);
    const [k, c] = a.shape;
    const out = new TYPED[dtype](axis === 1 ? k : c);
    if (axis === 1) for (let i = 0; i < k; i++) out[i] = row(data, i * c, c, cast);
    else {
      if (k === 0) throw new Error("ValueError: zero-size array to reduction operation which has no identity");
      for (let j = 0; j < c; j++) out[j] = cast(data[j]);
      for (let i = 1; i < k; i++) for (let j = 0; j < c; j++) out[j] = step(out[j], cast(data[i * c + j]), cast);
    }
    return new NdArray(out, [out.length], dtype);
  };
}

const sumRow = (data, start, n, cast) => (n === 0 ? 0 : pairwise(data, start, n, cast));
// min and max propagate NaN, as numpy's do.
const nanPick = (better) => (x, y) => (Number.isNaN(x) || Number.isNaN(y) ? NaN : better(x, y) ? x : y);
const lower = nanPick((x, y) => x <= y), higher = nanPick((x, y) => x >= y);
function extreme(f) {
  return (data, start, n, cast) => {
    if (n === 0) throw new Error("ValueError: zero-size array to reduction operation which has no identity");
    let res = cast(data[start]);
    for (let i = 1; i < n; i++) res = f(res, cast(data[start + i]));
    return res;
  };
}

const ndSum = reduce(sumRow, (s, v, cast) => cast(s + v));
const ndMean = (x, axis, dtype) => {
  const a = asnd(x), cast = CAST[dtype];
  const s = ndSum(a, axis, dtype);
  const count = axis === null || a.shape.length === 1 ? a.data.length : a.shape[axis];
  if (!(s instanceof NdArray)) return cast(s / count);
  return new NdArray(s.data.map((v) => cast(v / count)), s.shape, dtype);
};

// The flat index of the first minimum (maximum), or of the first NaN, as numpy's.
function argExtreme(better) {
  return (x) => {
    const { data } = asnd(x);
    if (!data.length) throw new Error("ValueError: attempt to get argmin of an empty sequence");
    let at = 0;
    for (let i = 0; i < data.length; i++) {
      if (Number.isNaN(data[i])) return i;
      if (better(data[i], data[at])) at = i;
    }
    return at;
  };
}

// np.where(cond): the indices of the true elements, one Int32Array per axis; with x and y,
// x where cond holds and y elsewhere, broadcast and cast to dtype.
function where(c, x, y, dtype) {
  const cond = asnd(c);
  if (x === undefined) {
    const axes = cond.shape.map(() => []);
    cond.data.forEach((v, i) => {
      if (!v) return;
      for (let d = cond.shape.length - 1; d >= 0; d--) {
        axes[d].push(i % cond.shape[d]);
        i = Math.floor(i / cond.shape[d]);
      }
    });
    return axes.map((ix) => new NdArray(Int32Array.from(ix), [ix.length], "i32"));
  }
  const a = asnd(x), b = asnd(y), cast = CAST[dtype];
  const arrays = [cond, a, b].filter((v) => v instanceof NdArray);
  const shape = arrays.reduce((s, v) => broadcast(new NdArray(null, s, null), v), arrays[0].shape);
  const n = sizeOf(shape);
  const at = (v, i) => (v instanceof NdArray ? v.data[i % v.data.length] : v);
  const out = new TYPED[dtype](n);
  for (let i = 0; i < n; i++) out[i] = cast(at(cond, i) ? at(a, i) : at(b, i));
  return new NdArray(out, [...shape], dtype);
}

// np.stack of rank-1 arrays of one length: axis 0 puts them as rows, axis 1 as columns.
function stack(list, axis, dtype) {
  const items = list.map(asnd), k = items.length ? items[0].data.length : 0, cast = CAST[dtype];
  if (items.some((v) => v.data.length !== k)) throw new Error("ValueError: all input arrays must have the same shape");
  const out = new TYPED[dtype](k * items.length);
  items.forEach((v, j) => {
    for (let i = 0; i < k; i++) out[axis === 0 ? j * k + i : i * items.length + j] = cast(v.data[i]);
  });
  return new NdArray(out, axis === 0 ? [items.length, k] : [k, items.length], dtype);
}

const outOfBounds = (i, axis, size) => new Error(`IndexError: index ${i} is out of bounds for axis ${axis} with size ${size}`);

// a[key], a copy: an int (a number from a rank-1 array, else a row), [null, j] (a[:, j]),
// a bool mask over the whole array (the elements, flat) or over its rows, or an index array
// of rows.
function ndGetitem(x, key) {
  const a = asnd(x), [k] = a.shape, w = sizeOf(a.shape.slice(1));
  const rows = (ix) => {
    const out = new TYPED[a.dtype](ix.length * w);
    ix.forEach((r, j) => out.set(a.data.subarray(r * w, (r + 1) * w), j * w));
    return new NdArray(out, [ix.length, ...a.shape.slice(1)], a.dtype);
  };
  const row = (i) => {
    const r = i < 0 ? i + k : i;
    if (!(r >= 0 && r < k)) throw outOfBounds(i, 0, k);
    return r;
  };
  if (typeof key === "number") {
    const r = row(key);
    return a.shape.length === 1 ? a.data[r] : new NdArray(a.data.slice(r * w, (r + 1) * w), a.shape.slice(1), a.dtype);
  }
  if (Array.isArray(key)) {
    const [, j] = key, c = a.shape[1], col = j < 0 ? j + c : j;
    if (!(col >= 0 && col < c)) throw outOfBounds(j, 1, c);
    return new NdArray(TYPED[a.dtype].from({ length: k }, (_, i) => a.data[i * c + col]), [k], a.dtype);
  }
  const m = asnd(key);
  if (m.dtype !== "bool") return rows(Array.from(m.data, row));
  if (m.shape.length === a.shape.length) {
    if (m.shape.some((d, i) => d !== a.shape[i])) {
      throw new Error(`IndexError: boolean index did not match indexed array along axis 0; size of axis is ${k} but size of corresponding boolean axis is ${m.shape[0]}`);
    }
    return new NdArray(a.data.filter((_, i) => m.data[i]), [m.data.reduce((s, v) => s + v, 0)], a.dtype);
  }
  if (m.data.length !== k) {
    throw new Error(`IndexError: boolean index did not match indexed array along axis 0; size of axis is ${k} but size of corresponding boolean axis is ${m.data.length}`);
  }
  const ix = [];
  m.data.forEach((v, i) => { if (v) ix.push(i); });
  return rows(ix);
}

const plus = (x, y) => x + y, minus = (x, y) => x - y, times = (x, y) => x * y;
export const nd = {
  asnd,
  add: binary(plus),
  sub: binary(minus),
  mul: binary(times, null, Math.imul),
  div: binary((x, y) => x / y),
  square: (a, dtype) => nd.mul(a, a, dtype),
  sqrt: unary(Math.sqrt),
  neg: unary((x) => -x),
  abs: unary(Math.abs),
  not: unary((x) => 1 - x),
  and: binary((x, y) => x & y),
  or: binary((x, y) => x | y),
  lt: binary((x, y) => x < y, "bool"),
  le: binary((x, y) => x <= y, "bool"),
  gt: binary((x, y) => x > y, "bool"),
  ge: binary((x, y) => x >= y, "bool"),
  eq: binary((x, y) => x === y, "bool"),
  ne: binary((x, y) => x !== y, "bool"),
  zeros: (shape, dtype) => full(shape, 0, dtype),
  ones: (shape, dtype) => full(shape, 1, dtype),
  full,
  arange: ndArange,
  tolist,
  array: (items, dtype) => {
    const v = Array.from(items);
    return new NdArray(TYPED[dtype].from(v, CAST[dtype]), [v.length], dtype);
  },
  sin: unary(Math.sin),
  cos: unary(Math.cos),
  exp: unary(Math.exp),
  floor: unary(Math.floor),
  clip: (a, lo, hi, dtype) => nd.minimum(nd.maximum(a, lo, dtype), hi, dtype),
  minimum: binary(lower),
  maximum: binary(higher),
  sum: ndSum,
  mean: ndMean,
  min: reduce(extreme(lower), (s, v) => lower(s, v)),
  max: reduce(extreme(higher), (s, v) => higher(s, v)),
  norm: (a, axis, dtype) => {
    const s = ndSum(nd.mul(a, a, dtype), axis, dtype);
    return s instanceof NdArray ? nd.sqrt(s, dtype) : CAST[dtype](Math.sqrt(s));
  },
  argmin: argExtreme((x, y) => x < y),
  argmax: argExtreme((x, y) => x > y),
  where,
  stack,
  column_stack: (list, dtype) => stack(list, 1, dtype),
  getitem: ndGetitem,
};

// ---------------------------------------------------------------- objects and collections

// An instance of a class with __call__: a function that carries the object, so that
// obj(x) calls obj.__call__(x) and obj.field, methods and instanceof still work.
export function callable_(obj) {
  const f = (...args) => f.__call__(...args);
  delete f.name;     // a field may have these names; a function's own ones are read-only
  delete f.length;
  Object.setPrototypeOf(f, Object.getPrototypeOf(obj));
  return Object.assign(f, obj);
}

// obj.method as a value keeps obj, as a Python bound method does.
export const bound = (obj, name) => obj[name].bind(obj);

// hasattr: the object's fields, its class's methods and properties, not what every JS
// object inherits (toString) or a class's prototype holds for JS (constructor).
export function hasattr_(x, name) {
  for (let o = x; o != null && o !== Object.prototype && o !== Function.prototype;
    o = Object.getPrototypeOf(o)) {
    if (Object.hasOwn(o, name) && !(o !== x && name === "constructor")) return true;
  }
  return false;
}

export const isinstance_ = (x, classes) => classes.some((c) => x instanceof c);

export const dget = (d, k, dflt = null) => (d.has(k) ? d.get(k) : dflt);
// set.remove and set.update, None as in Python; remove of a missing item is a KeyError.
export function sremove(s, x) {
  if (!s.has(x)) throw new Error(`KeyError: '${x}'`);
  s.delete(x);
  return null;
}
export function supdate(s, ...iterables) {
  for (const it of iterables) for (const x of it) s.add(x);
  return null;
}

export const dsetdefault = (d, k, dflt = null) => (d.has(k) ? d.get(k) : (d.set(k, dflt), dflt));

// What next() of an exhausted generator gives: its default, or StopIteration.
export function next_(...dflt) {
  if (dflt.length) return dflt[0];
  throw new Error("StopIteration");
}

// ---------------------------------------------------------------- text

// A str's length in code points, as Python counts: JS's .length counts UTF-16 units.
export function cplen(s) {
  let n = 0;
  for (const _ of s) n++;
  return n;
}

// The characters str.isspace() accepts, which str.strip() removes.
const PY_SPACE = "[\\t-\\r\\x1c-\\x20\\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000]";
const STRIP = new RegExp(`^${PY_SPACE}+|${PY_SPACE}+$`, "gu");
export const strip = (s) => s.replace(STRIP, "");

// The double x is exactly mant * 2 ** exp.
function decompose(x) {
  const view = new DataView(new ArrayBuffer(8));
  view.setFloat64(0, x);
  const bits = view.getBigUint64(0);
  const e = Number((bits >> 52n) & 0x7ffn), m = bits & ((1n << 52n) - 1n);
  return e === 0 ? [m, -1074] : [m | (1n << 52n), e - 1075];
}

// format(x, ".Nf") as Python: rounded from the double's exact value, a tie to even. JS's
// toFixed breaks exact ties upward and switches to exponents at 1e21.
export function fmt(x, spec) {
  const m = /^\.(\d+)f$/.exec(spec);
  if (!m) throw new Error(`the format spec :${spec} is outside the subset`);
  const n = Number(m[1]);
  if (typeof x === "boolean") x = Number(x);
  if (Number.isNaN(x)) return "nan";
  if (!Number.isFinite(x)) return x > 0 ? "inf" : "-inf";
  const neg = x < 0 || Object.is(x, -0);
  const [mant, exp] = decompose(Math.abs(x));
  const scaled = mant * 10n ** BigInt(n);
  let q;
  if (exp >= 0) {
    q = scaled << BigInt(exp);
  } else {
    const den = 1n << BigInt(-exp);
    q = scaled / den;
    const twice = 2n * (scaled - q * den);
    if (twice > den || (twice === den && (q & 1n) === 1n)) q += 1n;
  }
  let digits = q.toString();
  if (n > 0) {
    digits = digits.padStart(n + 1, "0");
    digits = `${digits.slice(0, -n)}.${digits.slice(-n)}`;
  }
  return (neg ? "-" : "") + digits;
}

// repr of a float: the shortest digits that round-trip (JS finds the same ones), laid out
// as Python does, positional from 1e-4 up to 1e16, "3.0" for an integral value.
function floatRepr(x) {
  if (Number.isNaN(x)) return "nan";
  if (!Number.isFinite(x)) return x > 0 ? "inf" : "-inf";
  if (x === 0) return Object.is(x, -0) ? "-0.0" : "0.0";
  const [mantissa, e] = Math.abs(x).toExponential().split("e");
  const digits = mantissa.replace(".", ""), exp = Number(e), sign = x < 0 ? "-" : "";
  if (exp < -4 || exp >= 16) {
    const head = digits.length > 1 ? `${digits[0]}.${digits.slice(1)}` : digits;
    return `${sign}${head}e${exp < 0 ? "-" : "+"}${String(Math.abs(exp)).padStart(2, "0")}`;
  }
  if (exp < 0) return `${sign}0.${"0".repeat(-exp - 1)}${digits}`;
  const whole = digits.slice(0, exp + 1).padEnd(exp + 1, "0");
  return `${sign}${whole}.${digits.slice(exp + 1) || "0"}`;
}

// str(x) for a scalar. The transpiler passes how a number is written, "float" or "int",
// from the type: 3.0 and 3 are the same JS number.
export function str_(x, mode = "int") {
  if (x === null || x === undefined) return "None";
  if (typeof x === "boolean") return x ? "True" : "False";
  if (typeof x === "string") return x;
  if (typeof x === "number") {
    if (mode === "float") return floatRepr(x);
    return Math.abs(x) < 1e21 ? String(x) : BigInt(x).toString();
  }
  throw new TypeError("str() of this value is outside the subset");
}

// textwrap.wrap(text, width) with TextWrapper's defaults, ported from CPython's textwrap:
// the chunks of _split (whitespace, words, and the parts of hyphenated words) and
// _wrap_chunks, counting code points. \w and the letters are Unicode, as in Python's re.
const WS = "[\\t\\n\\x0b\\x0c\\r ]", NWS = "[^\\t\\n\\x0b\\x0c\\r ]";
const WORD = "[\\p{L}\\p{N}_]";
const WORD_PUNCT = "[\\p{L}\\p{N}_!\"'&.,?]";
const LETTER = "[\\p{L}\\p{Nl}\\p{No}_]";   // \w but not a decimal digit
const WORDSEP = new RegExp(
  `(${WS}+`
  + `|(?<=${WORD_PUNCT})-{2,}(?=${WORD})`
  + `|${NWS}+?(?:-(?:(?<=${LETTER}{2}-)|(?<=${LETTER}-${LETTER}-))(?=${LETTER}-?${LETTER})`
  + `|(?=${WS}|$)`
  + `|(?<=${WORD_PUNCT})(?=-{2,}${WORD})))`,
  "u",
);

function expandtabs(s, size) {
  let out = "", col = 0;
  for (const ch of s) {
    if (ch === "\t") {
      const k = size - (col % size);
      out += " ".repeat(k);
      col += k;
    } else {
      out += ch;
      col = ch === "\n" || ch === "\r" ? 0 : col + 1;
    }
  }
  return out;
}

export function wrap(text, width = 70) {
  if (width <= 0) throw new Error(`ValueError: invalid width ${width} (must be > 0)`);
  const munged = expandtabs(text, 8).replace(/[\t\n\x0b\x0c\r]/g, " ");
  // Each chunk as an array of code points; reversed, so the next chunk is the last.
  const chunks = munged.split(WORDSEP).filter((c) => c).map((c) => Array.from(c)).reverse();
  const blank = (c) => strip(c.join("")) === "";
  const lines = [];
  while (chunks.length) {
    const cur = [];
    let curLen = 0;
    if (blank(chunks.at(-1)) && lines.length) chunks.pop();
    while (chunks.length && curLen + chunks.at(-1).length <= width) {
      const c = chunks.pop();
      cur.push(c);
      curLen += c.length;
    }
    if (chunks.length && chunks.at(-1).length > width) {
      // _handle_long_word: break the word, after its last hyphen when one fits.
      const spaceLeft = width < 1 ? 1 : width - curLen, chunk = chunks.at(-1);
      let end = spaceLeft;
      if (chunk.length > spaceLeft) {
        const hyphen = chunk.slice(0, spaceLeft).lastIndexOf("-");
        if (hyphen > 0 && chunk.slice(0, hyphen).some((ch) => ch !== "-")) end = hyphen + 1;
      }
      cur.push(chunk.slice(0, end));
      chunks[chunks.length - 1] = chunk.slice(end);
      curLen = cur.reduce((n, c) => n + c.length, 0);
    }
    if (cur.length && blank(cur.at(-1))) cur.pop();
    if (cur.length) lines.push(cur.map((c) => c.join("")).join(""));
  }
  return lines;
}
