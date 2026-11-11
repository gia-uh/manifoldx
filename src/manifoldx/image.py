"""Small, dependency-free image tools: PNG in and out, resizing and padding.

Images are numpy arrays: ``uint8`` with shape ``(H, W, 3)`` (RGB) or
``(H, W, 4)`` (RGBA). Row 0 is the top of the picture, so image coordinates
``(x, y)`` are (column, row). Only numpy and the standard library are used.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

__all__ = ["ASPECTS", "read_png", "write_png", "resize", "pad_to_aspect"]

# Named aspect ratios: width:height.
ASPECTS = {
    "1:1": (1, 1),
    "4:3": (4, 3),
    "3:4": (3, 4),
    "3:2": (3, 2),
    "2:3": (2, 3),
    "16:9": (16, 9),
    "9:16": (9, 16),
    "21:9": (21, 9),
}

_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _unfilter(data: bytes, width: int, height: int, bpp: int, depth: int = 8) -> np.ndarray:
    """Undo the five PNG row filters; return (height, stride) uint8.

    ``bpp`` is the samples per pixel and ``depth`` their bit depth, so a 4-bit
    palette image has a stride of ``ceil(width / 2)`` bytes. The filters work
    on whole bytes, so ``bpp`` stays 1 for the sub-byte depths.
    """
    stride = (width * bpp * depth + 7) // 8
    if len(data) < height * (stride + 1):
        raise ValueError("PNG image data is truncated")
    rows = []
    prev = bytearray(stride)
    pos = 0
    for _ in range(height):
        ftype = data[pos]
        pos += 1
        row = bytearray(data[pos:pos + stride])
        pos += stride
        if ftype == 1:
            for i in range(bpp, stride):
                row[i] = (row[i] + row[i - bpp]) & 0xFF
        elif ftype == 2:
            for i in range(stride):
                row[i] = (row[i] + prev[i]) & 0xFF
        elif ftype == 3:
            for i in range(stride):
                left = row[i - bpp] if i >= bpp else 0
                row[i] = (row[i] + ((left + prev[i]) >> 1)) & 0xFF
        elif ftype == 4:
            for i in range(stride):
                left = row[i - bpp] if i >= bpp else 0
                upper_left = prev[i - bpp] if i >= bpp else 0
                row[i] = (row[i] + _paeth(left, prev[i], upper_left)) & 0xFF
        elif ftype != 0:
            raise ValueError(f"unknown PNG filter type {ftype}")
        rows.append(row)
        prev = row
    return np.frombuffer(b"".join(rows), dtype=np.uint8).reshape(height, stride)


def _unpack_bits(raw: np.ndarray, width: int, depth: int) -> np.ndarray:
    """Unpack a sub-byte PNG scanline into one integer per pixel, most significant bit first."""
    height = raw.shape[0]
    bits = np.unpackbits(raw, axis=1)[:, :width * depth]
    weights = (1 << np.arange(depth - 1, -1, -1)).astype(np.uint16)
    return (bits.reshape(height, width, depth) * weights).sum(axis=2).astype(np.uint8)


def read_png(path) -> np.ndarray:
    """Decode an 8-bit or sub-byte, non-interlaced PNG. Missing alpha becomes opaque.

    RGB and RGBA stay as they are; grey becomes RGB and grey+alpha becomes
    RGBA; a palette image becomes RGB, or RGBA when it carries a tRNS chunk.
    Grey and palette images may be 1, 2, 4 or 8 bits per pixel (a palette
    drawing is often stored at 4). Any other PNG raises ``ValueError``.
    """
    data = Path(path).read_bytes()
    if not data.startswith(_SIGNATURE):
        raise ValueError("not a PNG file")
    pos = len(_SIGNATURE)
    header = None
    palette = None
    trns = None
    idat = []
    while pos + 8 <= len(data):
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        kind = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        pos += 12 + length
        if kind == b"IHDR":
            width, height, depth, color_type, compression, filter_method, interlace = (
                struct.unpack(">IIBBBBB", body))
            header = (width, height, depth, color_type, compression, filter_method, interlace)
        elif kind == b"PLTE":
            palette = np.frombuffer(body, dtype=np.uint8).reshape(-1, 3)
        elif kind == b"tRNS":
            trns = np.frombuffer(body, dtype=np.uint8)
        elif kind == b"IDAT":
            idat.append(body)
        elif kind == b"IEND":
            break
    if header is None:
        raise ValueError("PNG without an IHDR chunk")
    width, height, depth, color_type, compression, filter_method, interlace = header
    if depth not in (1, 2, 4, 8):
        raise ValueError(f"only 1, 2, 4 and 8-bit PNGs are supported, got {depth}-bit")
    if depth != 8 and color_type not in (0, 3):
        raise ValueError(f"a {depth}-bit PNG must be grey or paletted, not colour type {color_type}")
    if compression != 0 or filter_method != 0:
        raise ValueError("unsupported PNG compression or filter method")
    if interlace != 0:
        raise ValueError("interlaced PNGs are not supported")
    if color_type not in _CHANNELS:
        raise ValueError(f"unsupported PNG colour type {color_type}")
    if not idat:
        raise ValueError("PNG without image data")

    channels = _CHANNELS[color_type]
    raw = _unfilter(zlib.decompress(b"".join(idat)), width, height, channels, depth)
    if depth < 8:
        values = _unpack_bits(raw, width, depth)
        if color_type == 0:
            return np.ascontiguousarray(np.repeat(
                (values.astype(np.uint16) * 255 // ((1 << depth) - 1)).astype(np.uint8)[..., None], 3, axis=2))
        pixels = values[..., None]
    else:
        pixels = raw.reshape(height, width, channels)

    if color_type == 0:
        return np.ascontiguousarray(np.repeat(pixels, 3, axis=2))
    if color_type == 4:
        grey = np.repeat(pixels[..., :1], 3, axis=2)
        return np.ascontiguousarray(np.concatenate([grey, pixels[..., 1:]], axis=2))
    if color_type == 2:
        return np.ascontiguousarray(pixels)
    if color_type == 6:
        return np.ascontiguousarray(pixels)
    # Palette image: indices into PLTE, with optional per-entry alpha.
    if palette is None:
        raise ValueError("palette PNG without a PLTE chunk")
    indices = pixels[..., 0]
    if indices.max(initial=0) >= len(palette):
        raise ValueError("palette PNG index past the end of PLTE")
    rgb = palette[indices]
    if trns is None:
        return np.ascontiguousarray(rgb)
    alpha = np.full(len(palette), 255, dtype=np.uint8)
    alpha[:len(trns)] = trns
    return np.ascontiguousarray(np.dstack([rgb, alpha[indices]]))


def _encode_png(img) -> bytes:
    """PNG bytes of an (H, W, 3) or (H, W, 4) uint8 image (filter 0 rows)."""
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        raise ValueError("images must be uint8")
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError("images must have shape (H, W, 3) or (H, W, 4)")
    height, width, channels = arr.shape
    color_type = 2 if channels == 3 else 6
    raw = np.ascontiguousarray(arr).reshape(height, width * channels)
    rows = b"".join(b"\x00" + bytes(raw[y]) for y in range(height))

    def chunk(kind, body):
        return (struct.pack(">I", len(body)) + kind + body
                + struct.pack(">I", zlib.crc32(kind + body)))

    return (_SIGNATURE
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows, 6))
            + chunk(b"IEND", b""))


def write_png(path, img) -> None:
    """Write an RGB or RGBA uint8 image as a PNG."""
    Path(path).write_bytes(_encode_png(img))


def resize(img, size, nearest: bool = False) -> np.ndarray:
    """Resize to ``size=(w, h)``; bilinear for colour, ``nearest`` for labels."""
    arr = np.asarray(img)
    if arr.ndim not in (2, 3):
        raise ValueError("images must be (H, W) or (H, W, C)")
    height, width = arr.shape[:2]
    out_w, out_h = int(size[0]), int(size[1])
    if out_w < 1 or out_h < 1:
        raise ValueError("size must be positive")

    def axis(n_out, n_in):
        if nearest:
            idx = np.floor((np.arange(n_out) + 0.5) * n_in / n_out).astype(np.intp)
            return np.clip(idx, 0, n_in - 1), None, None
        src = (np.arange(n_out) + 0.5) * n_in / n_out - 0.5
        i0 = np.floor(src).astype(np.intp)
        i0 = np.clip(i0, 0, n_in - 1)
        weight = np.clip(src - i0, 0.0, 1.0)
        i1 = np.clip(i0 + 1, 0, n_in - 1)
        return i0, i1, weight

    y0, y1, wy = axis(out_h, height)
    x0, x1, wx = axis(out_w, width)
    if nearest:
        out = arr[y0][:, x0]
        return out.astype(arr.dtype, copy=False)

    work = arr.astype(np.float32)
    extra = (1,) * (work.ndim - 2)
    wx = wx.reshape((1, out_w) + extra)
    wy = wy.reshape((out_h, 1) + extra)
    top = work[y0][:, x0] * (1.0 - wx) + work[y0][:, x1] * wx
    bottom = work[y1][:, x0] * (1.0 - wx) + work[y1][:, x1] * wx
    out = top * (1.0 - wy) + bottom * wy
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        out = np.clip(np.rint(out), info.min, info.max)
    return out.astype(arr.dtype)


def _aspect_ratio(aspect) -> float:
    if isinstance(aspect, str):
        if aspect in ASPECTS:
            w, h = ASPECTS[aspect]
        else:
            try:
                w, h = (int(part) for part in aspect.split(":"))
            except ValueError as exc:
                raise ValueError(f"unknown aspect ratio {aspect!r}") from exc
    else:
        try:
            w, h = aspect
        except (TypeError, ValueError) as exc:
            raise ValueError(f"unknown aspect ratio {aspect!r}") from exc
    if w <= 0 or h <= 0:
        raise ValueError("aspect ratio terms must be positive")
    return w / h


def nearest_aspect(width, height):
    """The name in ``ASPECTS`` whose ratio is closest to ``width / height``."""
    r = np.log(width / height)
    return min(ASPECTS, key=lambda n: abs(np.log(ASPECTS[n][0] / ASPECTS[n][1]) - r))


def pad_to_aspect(img, aspect=None, fill=255):
    """Pad an image (centred) to ``aspect`` and return it with the offset.

    ``aspect`` is a name in ``ASPECTS`` such as ``"16:9"`` or a ``(w, h)``
    pair; ``None`` takes the nearest one (:func:`nearest_aspect`). The result is ``(padded, (x0, y0))``, where ``(x0, y0)`` is the
    position of the original image inside the padded one.
    """
    arr = np.asarray(img)
    if arr.ndim not in (2, 3):
        raise ValueError("images must be (H, W) or (H, W, C)")
    height, width = arr.shape[:2]
    ratio = _aspect_ratio(nearest_aspect(width, height) if aspect is None else aspect)
    if width / height > ratio:
        new_height, new_width = max(height, round(width / ratio)), width
    else:
        new_height, new_width = height, max(width, round(height * ratio))
    x0 = (new_width - width) // 2
    y0 = (new_height - height) // 2
    out = np.empty((new_height, new_width) + arr.shape[2:], dtype=arr.dtype)
    out[...] = fill
    out[y0:y0 + height, x0:x0 + width] = arr
    return out, (x0, y0)
