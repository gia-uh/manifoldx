"""Minimal Wavefront OBJ parser.

Supports the four face-line forms (1-indexed):
    f v ...
    f v/vt ...
    f v/vt/vn ...
    f v//vn ...

All face lines in a file must use the same form. Polygon faces with >3
vertices are fan-triangulated. Materials (`mtllib`, `usemtl`) and
grouping directives (`o`, `g`, `s`) are silently ignored — v1 carries
material info through Python kwargs, not MTL.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import numpy as np


class ObjParseError(ValueError):
    """The OBJ file is malformed or uses a feature v1 doesn't support."""


def load_obj(path: str | Path) -> dict:
    """Parse a Wavefront .obj file into a manifoldx geometry dict.

    Returns:
        {"name": str,
         "positions": (N, 3) float32,
         "normals":   (N, 3) float32,   # the file's, or area-weighted vertex normals
         "uvs":       (N, 2) float32,   # present if file has UVs; v flipped to top-left origin
         "indices":   (M,)   uint32}
    """
    geo = _load_obj(Path(path))
    if "normals" not in geo:  # the mesh pipeline needs them
        geo["normals"] = _vertex_normals(geo["positions"], geo["indices"])
    return geo


def _vertex_normals(positions, indices):
    """Area-weighted vertex normals of an indexed triangle list."""
    tri = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
    a, b, c = (positions[tri[:, k]] for k in range(3))
    face = np.cross(b - a, c - a)
    normals = np.zeros_like(positions, dtype=np.float32)
    for k in range(3):
        np.add.at(normals, tri[:, k], face)
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    return (normals / np.where(length > 0, length, 1)).astype(np.float32)


def _load_obj(p: Path) -> dict:
    text = p.read_text()
    geo = _load_triangles(text)
    if geo is not None:
        return {"name": p.stem, **geo}

    raw_positions: list[list[float]] = []
    raw_normals: list[list[float]] = []
    raw_uvs: list[list[float]] = []
    face_triples: list[tuple] = []
    face_form: Optional[str] = None

    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        head, *rest = line.split()
        if head == "v":
            raw_positions.append([float(x) for x in rest[:3]])
        elif head == "vn":
            raw_normals.append([float(x) for x in rest[:3]])
        elif head == "vt":
            raw_uvs.append([float(x) for x in rest[:2]])
        elif head == "f":
            parsed, this_form = _parse_face(lineno, rest)
            if face_form is None:
                face_form = this_form
            elif face_form != this_form:
                raise ObjParseError(
                    f"line {lineno}: face-line form changed from "
                    f"'{face_form}' to '{this_form}'; pick one"
                )
            # Fan-triangulate.
            for i in range(1, len(parsed) - 1):
                face_triples.append(parsed[0])
                face_triples.append(parsed[i])
                face_triples.append(parsed[i + 1])
        # Silently ignored: o, g, s, mtllib, usemtl.

    return _build_geometry(p.stem, raw_positions, raw_normals, raw_uvs,
                           face_triples, face_form)


# A face line with every number replaced by "d", for each form.
_FORMS = {b"d": "v", b"d/d": "v/vt", b"d/d/d": "v/vt/vn", b"d//d": "v//vn"}
# Bytes str.split() splits on that the byte scan below does not; ASCII only.
_ODD_SPACE = np.array([11, 12, 13, 28, 29, 30, 31], np.uint8)
_CLASS = np.full(256, ord("x"), np.uint8)
_CLASS[ord("0"):ord("9") + 1] = ord("d")
for _c in b" /\n":
    _CLASS[_c] = _c


def _floats(body: np.ndarray, rows: int, cols: int) -> Optional[np.ndarray]:
    if rows == 0:
        return np.zeros((0, cols))
    try:
        a = np.loadtxt(io.StringIO(body.tobytes().decode()), comments=None, ndmin=2)
    except ValueError:
        return None
    return a[:, :cols] if a.shape == (rows, a.shape[1]) and a.shape[1] >= cols else None


def _load_triangles(text: str) -> Optional[dict]:
    """Vectorised parse of the common case: unindented lines, every face a triangle of
    one form, every index in range. Returns None for anything else, which the line
    parser handles with its own results and error messages."""
    if not text.isascii():
        return None
    data = np.frombuffer((text + "\n").encode(), np.uint8).copy()
    data[data == 9] = 32  # a tab splits like a space
    if np.isin(data, _ODD_SPACE).any():
        return None
    ends = np.flatnonzero(data == 10)
    starts = np.r_[0, ends[:-1] + 1]
    pad = np.append(data, [10, 10])
    c0, c1, c2 = pad[starts], pad[starts + 1], pad[starts + 2]
    v, f, t, n, sp, nl = (ord(c) for c in "vftn \n")
    if (c0 == sp).any() or (((c0 == v) | (c0 == f)) & (c1 == nl)).any() or (
        (c0 == v) & ((c1 == t) | (c1 == n)) & (c2 == nl)
    ).any():
        return None  # indented lines, or a bare "v"/"f": the line parser's business

    # Label every byte with its line's kind (1 v, 2 vt, 3 vn, 4 f, 0 other), heads unlabelled.
    is_v = c0 == v
    kinds = [is_v & (c1 == sp), is_v & (c1 == t) & (c2 == sp), is_v & (c1 == n) & (c2 == sp), (c0 == f) & (c1 == sp)]
    line_kind = np.zeros(len(starts), np.uint8)
    for k, sel in enumerate(kinds, start=1):
        line_kind[sel] = k
    byte_kind = np.repeat(line_kind, ends - starts + 1)
    for sel, head in zip(kinds, (2, 3, 3, 2)):
        for i in range(head):
            byte_kind[starts[sel] + i] = 0

    def body(k):
        """The bytes of the lines of kind k after their head, newlines kept, and their count."""
        return data[byte_kind == k], int(kinds[k - 1].sum())

    face_bytes, n_faces = body(4)
    if n_faces == 0:
        return None
    # Collapse each number to "d" and each blank run to one space, drop blanks at the
    # line ends: a file of triangles of one form is then the same line n_faces times.
    shape = _CLASS[face_bytes]
    shape = shape[~np.r_[False, (shape[1:] == shape[:-1]) & ((shape[1:] == ord("d")) | (shape[1:] == sp))]]
    shape = shape[~((shape == sp) & (np.r_[True, shape[:-1] == nl] | np.r_[shape[1:] == nl, True]))].tobytes()
    corner = shape.split(b" ", 1)[0]
    form = _FORMS.get(corner)
    if form is None or shape != (b" ".join([corner] * 3) + b"\n") * n_faces:
        return None
    has_uv = form in ("v/vt", "v/vt/vn")
    has_normal = form in ("v/vt/vn", "v//vn")

    pools = [_floats(*body(1), 3)]
    uvs = _floats(*body(2), 2)
    normals = _floats(*body(3), 3)
    if pools[0] is None or uvs is None or normals is None:
        return None
    if has_uv:
        pools.append(uvs)
    if has_normal:
        pools.append(normals)

    face_bytes[face_bytes == ord("/")] = sp
    corners = np.fromstring(face_bytes.tobytes(), dtype=np.int64, sep=" ").reshape(-1, len(pools)) - 1
    sizes = [len(a) for a in pools]
    if corners.min() < 0 or (corners >= sizes).any():
        return None  # 0 or out of range: the line parser reproduces what that does

    # Deduplicate corners in first-occurrence order, as the line parser's dict does.
    if np.prod(sizes, dtype=np.float64) < 2**62:
        key = corners[:, 0]
        for j in range(1, len(pools)):
            key = key * sizes[j] + corners[:, j]
        _, first, inverse = np.unique(key, return_index=True, return_inverse=True)
    else:
        _, first, inverse = np.unique(corners, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    unique = corners[first[order]]

    geo = {
        "positions": pools[0][unique[:, 0]].astype(np.float32),
        "indices": rank[inverse.ravel()].astype(np.uint32),
    }
    if has_normal:
        geo["normals"] = normals[unique[:, -1]].astype(np.float32)
    if has_uv:
        uv = uvs[unique[:, 1]]
        # OBJ puts v=0 at the bottom of the image: flip to the engine's top-left origin.
        geo["uvs"] = np.stack([uv[:, 0], 1.0 - uv[:, 1]], 1).astype(np.float32)
    return geo


def _parse_face(lineno: int, tokens: list[str]) -> tuple[list[tuple], str]:
    """Parse one face line's tokens into a list of (pi, ti|None, ni|None)
    triples and detect the face-line form."""
    if len(tokens) < 3:
        raise ObjParseError(
            f"line {lineno}: face needs at least 3 vertices, got {len(tokens)}"
        )

    parsed = []
    forms = set()
    for tok in tokens:
        if "//" in tok:
            pi_s, ni_s = tok.split("//")
            pi = _idx(lineno, pi_s)
            ni = _idx(lineno, ni_s)
            parsed.append((pi, None, ni))
            forms.add("v//vn")
        elif "/" in tok:
            parts = tok.split("/")
            if len(parts) == 2:
                pi = _idx(lineno, parts[0])
                ti = _idx(lineno, parts[1])
                parsed.append((pi, ti, None))
                forms.add("v/vt")
            elif len(parts) == 3:
                pi = _idx(lineno, parts[0])
                ti = _idx(lineno, parts[1])
                ni = _idx(lineno, parts[2])
                parsed.append((pi, ti, ni))
                forms.add("v/vt/vn")
            else:
                raise ObjParseError(
                    f"line {lineno}: malformed face token '{tok}'"
                )
        else:
            pi = _idx(lineno, tok)
            parsed.append((pi, None, None))
            forms.add("v")

    if len(forms) > 1:
        raise ObjParseError(
            f"line {lineno}: mixed face-line forms within a single face: "
            f"{sorted(forms)}"
        )
    (this_form,) = forms
    return parsed, this_form


def _idx(lineno: int, s: str) -> int:
    """Parse a 1-indexed OBJ index. Negative (relative) indices unsupported."""
    n = int(s)
    if n < 0:
        raise ObjParseError(
            f"line {lineno}: negative face indices not supported in v1; "
            f"re-export with absolute indices"
        )
    return n - 1


def _build_geometry(name, raw_positions, raw_normals, raw_uvs,
                    face_triples, face_form):
    has_uv = face_form in ("v/vt", "v/vt/vn")
    has_normal = face_form in ("v/vt/vn", "v//vn")

    dedup: dict[tuple, int] = {}
    positions_out: list[list[float]] = []
    normals_out: list[list[float]] = []
    uvs_out: list[list[float]] = []
    indices_out: list[int] = []

    for (pi, ti, ni) in face_triples:
        key = (pi, ti, ni)
        if key in dedup:
            indices_out.append(dedup[key])
            continue
        vi = len(positions_out)
        dedup[key] = vi
        positions_out.append(raw_positions[pi])
        if has_normal and ni is not None:
            normals_out.append(raw_normals[ni])
        if has_uv and ti is not None:
            # OBJ puts v=0 at the bottom of the image; textures are uploaded
            # top row first, so flip to the engine's top-left origin.
            u, v = raw_uvs[ti]
            uvs_out.append([u, 1.0 - v])
        indices_out.append(vi)

    geo = {
        "name": name,
        "positions": np.asarray(positions_out, dtype=np.float32),
        "indices": np.asarray(indices_out, dtype=np.uint32),
    }
    if has_normal:
        geo["normals"] = np.asarray(normals_out, dtype=np.float32)
    if has_uv:
        geo["uvs"] = np.asarray(uvs_out, dtype=np.float32)
    return geo
