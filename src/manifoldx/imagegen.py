"""Image models, for drawing the pictures a model or a texture is built from.

An `ImageModel` takes a prompt and a few reference pictures and gives back a
numpy image (`uint8`, `(H, W, 3)`). `OpenRouterImage` reaches any image model
OpenRouter serves; `HttpImage` reaches the FLUX.2 klein service on this machine.
Both answer the same way, so a pipeline does not care which one drew a view.

Money is tracked in a CSV log, one line per call: the time, the model, the
output's name and what the call cost in dollars. With `budget`, a call whose log
total has already reached the budget is refused before anything is sent.

Prompts for the pictures a textured solid needs are the functions at the end: a
flat blueprint per view, an albedo, a normal map, roughness and emission maps,
and an edit that keeps the canvas pixel for pixel.
"""

from __future__ import annotations

import base64
import csv
import json
import os
import struct
import threading
import time
import urllib.error
import urllib.request
import zlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from manifoldx.devchannel import png

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
TIMEOUT = 180.0  # seconds an image model may take

_LONG_SIDE = {"1K": 1024, "2K": 2048, "4K": 4096}


@runtime_checkable
class ImageModel(Protocol):
    """What a picture model has to offer: a `name`, the `cost` of its last call, and `generate`.

    `generate` returns an image as a `uint8` array of shape `(H, W, 3)` or
    `(H, W, 4)`; `images` are reference pictures of the same kind (a blueprint, a
    previous answer) and `aspect` and `size` ask for a canvas (`"16:9"`, `"2K"`)."""

    name: str
    cost: float | None

    def generate(self, prompt: str, images: Sequence[np.ndarray] = (), aspect: str = "1:1",
                 size: str = "1K") -> np.ndarray:
        ...


class BudgetExceeded(RuntimeError):
    """A call was refused because the cost log has already reached the budget."""


def _as_rgb(img: np.ndarray) -> np.ndarray:
    """An image as contiguous RGB bytes; an alpha channel is composited over white."""
    a = np.asarray(img)
    if a.dtype != np.uint8:
        raise ValueError("an image must be uint8")
    if a.ndim != 3 or a.shape[2] not in (3, 4):
        raise ValueError("an image must have shape (H, W, 3) or (H, W, 4)")
    if a.shape[2] == 3:
        return np.ascontiguousarray(a)
    alpha = a[..., 3:4].astype(np.uint16)
    rgb = a[..., :3].astype(np.uint16)
    out = (rgb * alpha + 255 * (255 - alpha) + 127) // 255
    return np.ascontiguousarray(out.astype(np.uint8))


def _data_url(img: np.ndarray) -> str:
    """A PNG data URL for an image, which is how both services take a reference."""
    return "data:image/png;base64," + base64.b64encode(png(_as_rgb(img))).decode("ascii")


def _pixels(aspect: str, size: str) -> tuple[int, int]:
    """The pixel size of a canvas asked for as an aspect ratio and a size name."""
    try:
        aw, ah = (int(part) for part in aspect.split(":"))
    except (ValueError, AttributeError):
        raise ValueError(f"bad aspect ratio {aspect!r}: expected W:H") from None
    if aw <= 0 or ah <= 0:
        raise ValueError(f"bad aspect ratio {aspect!r}: expected W:H")
    name = str(size).upper()
    long_side = _LONG_SIDE.get(name)
    if long_side is None:
        try:
            long_side = int(name)
        except ValueError:
            raise ValueError(f"bad size {size!r}: expected 1K, 2K, 4K or pixels") from None
        if long_side <= 0:
            raise ValueError(f"bad size {size!r}: expected 1K, 2K, 4K or pixels")
    if aw >= ah:
        return long_side, max(1, round(long_side * ah / aw))
    return max(1, round(long_side * aw / ah)), long_side


# ----------------------------------------------------------------------------- PNG reading

def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def _decode_png(data: bytes) -> np.ndarray:
    """Decode an 8-bit RGB or RGBA, non-interlaced PNG into a `(H, W, 3|4)` uint8 array.

    This is a small reader for what an image model sends back; `manifoldx.image`
    replaces it. Anything else (palette, grey, 16-bit, interlaced) raises
    `ValueError`."""
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    pos, header, idat = 8, None, []
    while pos + 8 <= len(data):
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        kind = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        if len(body) != length:
            raise ValueError("truncated PNG")
        if kind == b"IHDR":
            header = struct.unpack(">IIBBBBB", body)
        elif kind == b"IDAT":
            idat.append(body)
        elif kind == b"IEND":
            break
        pos += 12 + length
    if header is None or not idat:
        raise ValueError("PNG without IHDR or IDAT")
    w, h, depth, colour, _, _, interlace = header
    if depth != 8:
        raise ValueError(f"unsupported PNG bit depth {depth}")
    if colour not in (2, 6):
        raise ValueError(f"unsupported PNG colour type {colour}")
    if interlace:
        raise ValueError("interlaced PNG")
    bpp = 3 if colour == 2 else 4
    stride = w * bpp
    raw = zlib.decompress(b"".join(idat))
    if len(raw) != h * (stride + 1):
        raise ValueError("truncated PNG data")
    out = np.empty((h, stride), np.uint8)
    previous = np.zeros(stride, np.uint8)
    for y in range(h):
        kind = raw[y * (stride + 1)]
        line = np.frombuffer(raw, np.uint8, stride, y * (stride + 1) + 1).astype(np.uint16)
        if kind == 2:  # up, before the left-hand pass so a filtered byte is never read twice
            line += previous
            line &= 0xFF
        elif kind == 1:  # sub
            for x in range(bpp, stride):
                line[x] = (line[x] + line[x - bpp]) & 0xFF
        elif kind == 3:  # average
            for x in range(stride):
                left = line[x - bpp] if x >= bpp else 0
                line[x] = (line[x] + ((int(left) + int(previous[x])) >> 1)) & 0xFF
        elif kind == 4:  # paeth
            for x in range(stride):
                left = int(line[x - bpp]) if x >= bpp else 0
                up = int(previous[x])
                corner = int(previous[x - bpp]) if x >= bpp else 0
                line[x] = (line[x] + _paeth(left, up, corner)) & 0xFF
        elif kind != 0:
            raise ValueError(f"unknown PNG filter {kind}")
        row = line.astype(np.uint8)
        out[y] = row
        previous = row
    return out.reshape(h, w, bpp)


def _decode_data_url(url: str) -> np.ndarray:
    """The image in a `data:image/png;base64,...` URL (or in bare base64)."""
    if url.startswith("data:"):
        head, _, body = url.partition(",")
        if not head.lower().startswith("data:image/png"):
            raise ValueError(f"unsupported image type {head.split(';')[0]!r}")
        url = body
    try:
        return _decode_png(base64.b64decode(url, validate=True))
    except (ValueError, TypeError) as e:
        raise ValueError(f"the answer is not a PNG data URL: {e}") from None


# ----------------------------------------------------------------------------- OpenRouter

class OpenRouterImage:
    """An image model behind OpenRouter's chat-completions API.

    `model` is the model's id there (`"google/gemini-3-pro-image-preview"`). The
    key comes from `key`, else the text of `key_file`, else `OPENROUTER_API_KEY`;
    it is never logged or put in an error. `log` is a CSV file that gets one line
    per call, `time,model,out_name,cost`; `budget` is dollars, and a call is
    refused with `BudgetExceeded` once the log's total has reached it. `out_name`
    is recorded in the log and can be passed to `generate` instead."""

    def __init__(self, model: str, key: str | None = None, key_file: str | Path | None = None,
                 log: str | Path | None = None, budget: float | None = None, *,
                 url: str | None = None, out_name: str = "", timeout: float = TIMEOUT) -> None:
        if budget is not None and log is None:
            raise ValueError("a budget needs a log file to count the spending in")
        self.model = model
        self.name = model
        self.cost: float | None = None
        self.log = Path(log) if log is not None else None
        self.budget = budget
        self.out_name = out_name
        self.url = url or OPENROUTER_URL
        self.timeout = timeout
        self._key = self._read_key(key, key_file)
        self._lock = threading.Lock()

    @staticmethod
    def _read_key(key: str | None, key_file: str | Path | None) -> str:
        if key:
            return key
        if key_file is not None:
            try:
                text = Path(key_file).read_text(encoding="utf-8").strip()
            except OSError as e:
                raise ValueError(f"cannot read the key file {key_file}: {e.strerror}") from None
            if not text:
                raise ValueError(f"the key file {key_file} is empty")
            return text
        text = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if text:
            return text
        raise ValueError("no API key: pass key or key_file, or set OPENROUTER_API_KEY")

    def _redact(self, text: str) -> str:
        return text.replace(self._key, "***") if self._key else text

    def _fail(self, error: urllib.error.HTTPError) -> RuntimeError:
        try:
            detail = error.read().decode("utf-8", "replace")[:500]
        except OSError:
            detail = ""
        return RuntimeError(f"OpenRouter request failed: HTTP {error.code}: {self._redact(detail)}")

    def _post(self, body: dict) -> dict:
        request = urllib.request.Request(
            self.url, data=json.dumps(body).encode("utf-8"), method="POST",
            headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json",
                     "Accept": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as answer:
                return json.loads(answer.read().decode("utf-8"))
        except urllib.error.HTTPError:
            raise
        except (urllib.error.URLError, OSError) as e:
            raise RuntimeError(f"OpenRouter request failed: {self._redact(str(e))}") from None

    def _spent(self) -> float:
        """The sum of the costs in the log, which is what the budget counts."""
        if self.log is None or not self.log.exists():
            return 0.0
        total = 0.0
        with self._lock, self.log.open(newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if not row:
                    continue
                try:
                    total += float(row[-1])
                except ValueError:
                    pass
        return total

    def _record(self, out_name: str) -> None:
        if self.log is None:
            return
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        cost = "" if self.cost is None else f"{self.cost:.6f}"
        with self._lock, self.log.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f, lineterminator="\n").writerow([stamp, self.model, out_name, cost])

    def generate(self, prompt: str, images: Sequence[np.ndarray] = (), aspect: str = "1:1",
                 size: str = "1K", *, out_name: str | None = None) -> np.ndarray:
        """Draw one picture for `prompt`; references go in `images`; returns an RGB uint8 array."""
        if self.budget is not None:
            spent = self._spent()
            if spent >= self.budget:
                raise BudgetExceeded(
                    f"the log has spent ${spent:.4f} of the ${self.budget:.4f} budget")
        content: str | list[dict]
        if images:
            content = [{"type": "image_url", "image_url": {"url": _data_url(img)}} for img in images]
            content.append({"type": "text", "text": prompt})
        else:
            content = prompt
        body: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "modalities": ["image", "text"],
            "usage": {"include": True},
        }
        # Models that do not know image_config reject the whole request; try once more without it.
        body["image_config"] = {"aspect_ratio": aspect, "image_size": size}
        try:
            answer = self._post(body)
        except urllib.error.HTTPError as e:
            if "image_config" not in body:
                raise self._fail(e) from None
            del body["image_config"]
            try:
                answer = self._post(body)
            except urllib.error.HTTPError as e2:
                raise self._fail(e2) from None

        usage = answer.get("usage") or {}
        value = usage.get("cost")
        self.cost = float(value) if value is not None else None
        image = _decode_data_url(_answer_url(answer))
        self._record(self.out_name if out_name is None else out_name)
        return image


def _answer_url(answer: dict) -> str:
    """The data URL of the first image in a chat-completion answer."""
    choices = answer.get("choices") or []
    if not choices:
        raise RuntimeError("the answer has no choices")
    message = choices[0].get("message") or {}
    for item in message.get("images") or []:
        url = (item.get("image_url") or {}).get("url")
        if url:
            return url
    content = message.get("content")
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") in (
                    "image_url", "image", "output_image"):
                url = ((part.get("image_url") or {}).get("url") or part.get("url")
                       or part.get("data"))
                if url:
                    return url
    raise RuntimeError("the answer has no image")


# ----------------------------------------------------------------------------- local service

class HttpImage:
    """The FLUX.2 klein service on this machine: `POST /generate`.

    It takes `{prompt, refs, width, height, seed}` with references as PNG data
    URLs and answers `{image: data-url}`. `token_file`, when given, holds the
    bearer token; it is read once and never logged or put in an error."""

    name = "klein"

    def __init__(self, url: str, token_file: str | Path | None = None, *,
                 timeout: float = TIMEOUT) -> None:
        self.url = url.rstrip("/")
        self.endpoint = self.url if self.url.endswith("/generate") else self.url + "/generate"
        self.timeout = timeout
        self.cost: float | None = None
        self._token = ""
        if token_file is not None:
            try:
                self._token = Path(token_file).read_text(encoding="utf-8").strip()
            except OSError as e:
                raise ValueError(f"cannot read the token file {token_file}: {e.strerror}") from None

    def _redact(self, text: str) -> str:
        return text.replace(self._token, "***") if self._token else text

    def generate(self, prompt: str, images: Sequence[np.ndarray] = (), aspect: str = "1:1",
                 size: str = "1K", *, seed: int | None = None) -> np.ndarray:
        """Ask the service for one picture; `seed` makes a redraw repeatable."""
        width, height = _pixels(aspect, size)
        body = {"prompt": prompt, "refs": [_data_url(img) for img in images],
                "width": width, "height": height, "seed": seed}
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        request = urllib.request.Request(self.endpoint, data=json.dumps(body).encode("utf-8"),
                                         method="POST", headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as answer:
                data = json.loads(answer.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                detail = e.read().decode("utf-8", "replace")[:500]
            except OSError:
                detail = ""
            raise RuntimeError(f"the image service failed: HTTP {e.code}: {self._redact(detail)}"
                               ) from None
        except (urllib.error.URLError, OSError) as e:
            raise RuntimeError(f"the image service failed: {self._redact(str(e))}") from None
        url = data.get("image")
        if not url:
            raise RuntimeError("the image service did not return an image")
        self.cost = 0.0
        return _decode_data_url(url)


# ----------------------------------------------------------------------------- prompts

def _swatch(name: str, value) -> str:
    """One palette entry as `name #rrggbb`, or as the description it already is."""
    if isinstance(value, str):
        return f"{name} {value}"
    rgb = list(np.asarray(value).reshape(-1))[:3]
    if len(rgb) != 3:
        return f"{name} {value}"
    return f"{name} #{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}"


def _palette_text(palette: Mapping) -> str:
    return ", ".join(_swatch(name, value) for name, value in palette.items())


def blueprint(part: str, view: str, palette: Mapping, notes: str = "") -> str:
    """The prompt for one flat, orthographic view of a part in a fixed colour code."""
    text = (
        f"Orthographic {view} view of the {part}, one view per image, from straight on: no "
        "perspective, no shading, no highlights, no shadow, no ground. Draw every region as one "
        f"flat colour with hard closed edges: {_palette_text(palette)}. White background, the "
        "whole part inside the canvas, no text, labels, arrows, dimensions or parts list. Keep "
        "the outlines clean and thin.")
    return f"{text} {notes}".strip()


def albedo(view: str, palette_meaning: Mapping, notes: str = "") -> str:
    """The prompt for the albedo (flat colour) texture of one view."""
    meanings = ", ".join(f"{name}: {value}" if isinstance(value, str)
                         else _swatch(name, value) for name, value in palette_meaning.items())
    text = (
        f"Orthographic {view} view as an albedo texture: flat, even light, no shading, no "
        "highlights, no ambient occlusion, no baked lighting, hard edges between colours, white "
        f"background, no text. Keep the canvas and every outline of the reference. Colours mean: "
        f"{meanings}.")
    return f"{text} {notes}".strip()


def normal_map() -> str:
    """The prompt for a tangent-space normal map of the same canvas (OpenGL convention)."""
    return (
        "Tangent-space normal map of the same canvas in OpenGL convention: a flat surface is "
        "(128, 128, 255), raised details and grooves only, no colour, no lighting, no shadows, "
        "no baked light, hard edges. Match the outlines and details of the reference exactly, "
        "one view, same canvas, no text.")


def roughness_map(materials) -> str:
    """The prompt for a grayscale roughness map, white rough and black smooth."""
    if isinstance(materials, Mapping):
        by_material = ", ".join(f"{name} {float(value):.2f}"
                                for name, value in materials.items())
    else:
        by_material = ", ".join(str(name) for name in materials)
    return (
        "Grayscale roughness map of the same view: white is rough and matte, black is smooth "
        "and glossy, grey in between, no colour, no lighting, no text. Keep the canvas and the "
        f"outlines of the reference. Roughness by material: {by_material}.")


def emission_map(lit) -> str:
    """The prompt for an emission mask: the lit parts white, everything else black."""
    if isinstance(lit, Mapping):
        parts = ", ".join(f"{name} {'lit' if value else 'off'}" for name, value in lit.items())
    else:
        parts = ", ".join(str(name) for name in lit)
    return (
        "Emission map of the same canvas: the lit parts pure white and everything else pure "
        "black, hard edges, no glow, no blur, no colour, no text. Keep the outlines and position "
        f"of the reference. Lit: {parts}.")


def edit(change: str, keep: str) -> str:
    """The prompt for changing one thing in a picture and keeping the rest pixel for pixel."""
    return (
        f"Edit this picture: {change}. Keep {keep} pixel for pixel: same canvas size, the same "
        "outlines, position and colours of everything you do not change. No text, no new "
        "objects, no restyling, no extra views.")
