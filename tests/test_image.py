"""Tests for manifoldx.image: PNG in and out, resize and padding.

The PNGs are built here with zlib and struct, so the tests need nothing but
numpy and the standard library (the plan's rule for every asset tool).
"""

import struct
import zlib

import numpy as np
import pytest

from manifoldx.image import ASPECTS, pad_to_aspect, read_png, resize, write_png


def _chunk(kind, body):
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))


def _png_bytes(width, height, color_type, rows, plte=None, trns=None, depth=8, interlace=0):
    """Assemble a PNG from already-filtered scanlines (filter byte + data)."""
    ihdr = struct.pack(">IIBBBBB", width, height, depth, color_type, 0, 0, interlace)
    out = b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr)
    if plte is not None:
        out += _chunk(b"PLTE", bytes(plte))
    if trns is not None:
        out += _chunk(b"tRNS", bytes(trns))
    out += _chunk(b"IDAT", zlib.compress(b"".join(rows), 6))
    out += _chunk(b"IEND", b"")
    return out


def _paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _filter_row(row, prev, ftype, bpp):
    """Apply one PNG filter to a raw row of bytes; return filter byte + data."""
    out = bytearray()
    for i, value in enumerate(row):
        a = row[i - bpp] if i >= bpp else 0
        b = prev[i]
        c = prev[i - bpp] if i >= bpp else 0
        if ftype == 0:
            predictor = 0
        elif ftype == 1:
            predictor = a
        elif ftype == 2:
            predictor = b
        elif ftype == 3:
            predictor = (a + b) // 2
        else:
            predictor = _paeth(a, b, c)
        out.append((value - predictor) & 0xFF)
    return bytes([ftype]) + bytes(out)


def _encode(img, filters=(0,)):
    """Encode an (H, W, C) uint8 image, cycling through the given filter types."""
    h, w, c = img.shape
    color_type = {1: 0, 2: 4, 3: 2, 4: 6}[c]
    raw = np.ascontiguousarray(img).reshape(h, w * c)
    rows, prev = [], bytes(w * c)
    for y in range(h):
        row = bytes(raw[y])
        ftype = filters[y % len(filters)]
        rows.append(_filter_row(row, prev, ftype, c))
        prev = row
    return _png_bytes(w, h, color_type, rows)


def test_rgb_round_trip(tmp_path):
    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, size=(9, 13, 3), dtype=np.uint8)
    path = tmp_path / "rgb.png"
    write_png(path, img)
    back = read_png(path)
    assert back.dtype == np.uint8 and back.shape == img.shape
    assert (back == img).all()


def test_rgba_round_trip(tmp_path):
    rng = np.random.default_rng(4)
    img = rng.integers(0, 256, size=(7, 5, 4), dtype=np.uint8)
    path = tmp_path / "rgba.png"
    write_png(path, img)
    back = read_png(path)
    assert back.shape == img.shape and (back == img).all()


def test_grey_becomes_rgb(tmp_path):
    grey = np.array([[0, 40, 255], [7, 8, 9]], dtype=np.uint8)
    path = tmp_path / "grey.png"
    path.write_bytes(_encode(grey[..., None]))
    back = read_png(path)
    assert back.shape == (2, 3, 3)
    assert (back[..., 0] == grey).all()
    assert (back[..., 0] == back[..., 1]).all() and (back[..., 1] == back[..., 2]).all()


def test_grey_alpha_stays_rgba(tmp_path):
    grey = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    alpha = np.array([[0, 64], [128, 255]], dtype=np.uint8)
    path = tmp_path / "ga.png"
    path.write_bytes(_encode(np.stack([grey, alpha], axis=-1)))
    back = read_png(path)
    assert back.shape == (2, 2, 4)
    assert (back[..., 0] == grey).all() and (back[..., 3] == alpha).all()


def test_palette_reads_plte(tmp_path):
    palette = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    indices = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.uint8)
    path = tmp_path / "pal.png"
    path.write_bytes(_png_bytes(3, 2, 3, [b"\x00" + bytes(row) for row in indices],
                                plte=palette.reshape(-1)))
    back = read_png(path)
    assert back.shape == (2, 3, 3)
    assert (back == palette[indices]).all()


def test_palette_with_trns_gives_alpha(tmp_path):
    palette = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
    indices = np.array([[0, 1, 0]], dtype=np.uint8)
    path = tmp_path / "palt.png"
    path.write_bytes(_png_bytes(3, 1, 3, [b"\x00" + bytes(indices[0])],
                                plte=palette.reshape(-1), trns=[40, 255]))
    back = read_png(path)
    assert back.shape == (1, 3, 4)
    assert (back[0, :, 3] == [40, 255, 40]).all()
    assert (back[0, 0, :3] == [255, 0, 0]).all()


def test_all_five_filters_decode(tmp_path):
    rng = np.random.default_rng(5)
    img = rng.integers(0, 256, size=(10, 11, 2), dtype=np.uint8)
    path = tmp_path / "filters.png"
    path.write_bytes(_encode(img, filters=(0, 1, 2, 3, 4)))
    back = read_png(path)
    # grey+alpha reads back as RGBA with grey repeated in the colour channels.
    assert back.shape == (10, 11, 4)
    assert (back[..., 0] == img[..., 0]).all()
    assert (back[..., 1] == img[..., 0]).all()
    assert (back[..., 2] == img[..., 0]).all()
    assert (back[..., 3] == img[..., 1]).all()


def test_rejects_unsupported_pngs(tmp_path):
    path = tmp_path / "deep.png"
    path.write_bytes(_png_bytes(1, 1, 0, [b"\x00\x00\x00"], depth=16))
    with pytest.raises(ValueError):
        read_png(path)

    path = tmp_path / "interlaced.png"
    path.write_bytes(_png_bytes(1, 1, 0, [b"\x00\x00"], interlace=1))
    with pytest.raises(ValueError):
        read_png(path)


def test_resize_constant_stays_constant():
    img = np.full((4, 6, 3), (10, 20, 30), dtype=np.uint8)
    for nearest in (False, True):
        out = resize(img, (9, 5), nearest=nearest)
        assert out.shape == (5, 9, 3)
        assert (out == img[0, 0]).all()


def test_resize_nearest_keeps_labels():
    img = np.array([[0, 255]], dtype=np.uint8)
    out = resize(img, (8, 1), nearest=True)
    assert out[0, :4].tolist() == [0, 0, 0, 0]
    assert out[0, 4:].tolist() == [255, 255, 255, 255]


def test_resize_bilinear_midpoint():
    img = np.array([[0], [255]], dtype=np.uint8)
    out = resize(img, (1, 4))
    assert out.shape == (4, 1)
    assert out[0, 0] == 0 and out[3, 0] == 255
    assert out[1, 0] == pytest.approx(64, abs=2)
    assert out[2, 0] == pytest.approx(191, abs=2)


def test_pad_to_aspect_square():
    img = np.zeros((10, 20, 3), dtype=np.uint8)
    out, (x0, y0) = pad_to_aspect(img, "1:1", fill=255)
    assert out.shape == (20, 20, 3)
    assert (x0, y0) == (0, 5)
    assert (out[y0:y0 + 10, x0:x0 + 20] == img).all()
    assert (out[:y0] == 255).all() and (out[y0 + 10:] == 255).all()


def test_pad_to_aspect_given_and_custom_fill():
    img = np.zeros((10, 20, 3), dtype=np.uint8)
    out, (x0, y0) = pad_to_aspect(img, (4, 3), fill=7)
    assert out.shape == (15, 20, 3) and (x0, y0) == (0, 2)
    assert (out[0] == 7).all()
    assert out.shape[0] > img.shape[0]
    tall, offset = pad_to_aspect(img[:10, :4], "9:16", fill=0)
    assert tall.shape == (10, 6, 3) and offset == (1, 0)
    assert (tall[:, 0] == 0).all() and (tall[:, 1:5] == img[:10, :4]).all()


def test_pad_to_aspect_presets_and_errors():
    assert {"1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16", "21:9"} <= set(ASPECTS)
    with pytest.raises(ValueError):
        pad_to_aspect(np.zeros((4, 4, 3), np.uint8), "5:0")


def test_pad_to_aspect_takes_the_nearest_aspect_by_default():
    from manifoldx.image import nearest_aspect, pad_to_aspect
    assert nearest_aspect(1920, 1080) == "16:9" and nearest_aspect(300, 410) == "3:4"
    padded, _ = pad_to_aspect(np.zeros((410, 300, 3), np.uint8))
    assert padded.shape[:2] == (410, round(410 * 3 / 4))
