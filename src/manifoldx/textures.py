"""Texture loading and GPU upload for the mesh-PBR path.

A TextureHandle is the public reference users hand to materials
(e.g. StandardMaterial(albedo_map=handle)). The engine's
TextureRegistry owns the underlying GPU resources.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import wgpu

from manifoldx import _deferred


class TextureSizeError(ValueError):
    """Image is larger than the device's max_texture_dimension_2d limit."""


@dataclass
class TextureHandle:
    id: int
    texture: Any            # wgpu.Texture
    view: Any               # wgpu.TextureView
    sampler: Any            # wgpu.Sampler
    size: tuple[int, int]   # (width, height) in pixels
    source: Path | None = None  # the image file load_texture read; glTF export embeds it
    layers: int = 0  # a texture_array's layers; 0 for a plain 2D texture
    make: Any = None  # a texture_array made before the engine had a device: makes its GPU texture when first used


def ready(handle: TextureHandle) -> TextureHandle:
    """The handle with its GPU texture made, for a texture_array asked for before the engine had a device."""
    if handle.texture is None and handle.make is not None:
        handle.make()
    return handle


class TextureRegistry:
    """Owns GPU texture + sampler resources for the engine lifetime."""

    def __init__(self) -> None:
        self._handles: Dict[int, TextureHandle] = {}
        self._next_id = 1

    def add(self, handle: TextureHandle) -> None:
        self._handles[handle.id] = handle

    def alloc_id(self) -> int:
        new_id = self._next_id
        self._next_id += 1
        return new_id


MAX_ANISOTROPY = 8


def _mip_chain(img):
    """The image and its successive halvings down to 1x1, as RGBA uint8 arrays.

    Each level is a box filter of the previous one, done on the sRGB bytes; averaging
    in linear light would be more exact and costs a float pass per level for a
    difference that does not show on ground textures."""
    from PIL import Image

    levels = [np.asarray(img, dtype=np.uint8)]
    w, h = img.size
    while w > 1 or h > 1:
        w, h = max(1, w // 2), max(1, h // 2)
        img = img.resize((w, h), Image.Resampling.BOX)
        levels.append(np.asarray(img, dtype=np.uint8))
    return levels


def load_texture(engine, path: str | Path) -> TextureHandle:
    """Decode an image file with Pillow, upload to GPU as Rgba8UnormSrgb with a full
    mip chain, sampled trilinearly with anisotropic filtering and repeat addressing.

    Args:
        engine: the manifoldx Engine (must have a device initialized).
        path: filesystem path to a PNG / JPEG / any Pillow-supported format.

    Returns:
        A TextureHandle the caller passes to material constructors.
    """
    from PIL import Image

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    rgba = np.asarray(Image.open(p).convert("RGBA"), dtype=np.uint8)
    return texture_from_array(engine, rgba, _source=p)


def texture_from_array(engine, rgba: np.ndarray, *, _source: Path | None = None) -> TextureHandle:
    """Upload a (height, width, 4) or (height, width, 3) uint8 image, sRGB colour
    with straight alpha, as a texture like load_texture's: Rgba8UnormSrgb, full mip
    chain, trilinear and anisotropic, repeat addressing."""
    from PIL import Image

    rgba = np.asarray(rgba)
    if rgba.dtype != np.uint8 or rgba.ndim != 3 or rgba.shape[2] not in (3, 4):
        raise ValueError(
            f"texture_from_array expects a (h, w, 4|3) uint8 array, got {rgba.dtype} {rgba.shape}"
        )
    img = Image.fromarray(np.ascontiguousarray(rgba)).convert("RGBA")
    w, h = img.size

    upload_native = getattr(engine, "_upload_native_albedo", None)
    if upload_native is not None:
        return upload_native(img, _source)

    device = getattr(engine, "_device", None)
    if device is None:
        raise RuntimeError(
            "engine has no device yet; initialize the canvas before creating textures"
        )

    limits = getattr(device, "limits", {}) or {}
    max_dim = limits.get("max-texture-dimension-2d", 8192) if isinstance(limits, dict) else 8192
    if w > max_dim or h > max_dim:
        raise TextureSizeError(
            f"image is {w}x{h}, device max_texture_dimension_2d is {max_dim}"
        )

    levels = _mip_chain(img)
    texture = device.create_texture(
        size=(w, h, 1),
        format=wgpu.TextureFormat.rgba8unorm_srgb,
        usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        mip_level_count=len(levels),
        sample_count=1,
    )
    for i, level in enumerate(levels):
        lh, lw = level.shape[:2]
        device.queue.write_texture(
            {"texture": texture, "mip_level": i, "origin": (0, 0, 0)},
            level.tobytes(),
            {"offset": 0, "bytes_per_row": lw * 4, "rows_per_image": lh},
            (lw, lh, 1),
        )
    view = texture.create_view()
    sampler = device.create_sampler(
        mag_filter=wgpu.FilterMode.linear,
        min_filter=wgpu.FilterMode.linear,
        mipmap_filter=wgpu.MipmapFilterMode.linear,
        address_mode_u=wgpu.AddressMode.repeat,
        address_mode_v=wgpu.AddressMode.repeat,
        max_anisotropy=MAX_ANISOTROPY,
    )

    registry = engine._texture_registry
    handle = TextureHandle(
        id=registry.alloc_id(), texture=texture, view=view, sampler=sampler,
        size=(w, h), source=_source,
    )
    registry.add(handle)
    return handle


def texture_from_svg(engine, svg, size_px: int | tuple[int, int]) -> TextureHandle:
    """Rasterize an SVG (path, markup or bytes) and upload it with texture_from_array.

    `size_px` is the longest side in pixels, the other following the SVG's aspect
    ratio, or an exact (width, height). Transparent parts stay transparent: a
    StandardMaterial with this albedo_map cuts them out."""
    from manifoldx import svg as _svg

    data = _svg.svg_bytes(svg)
    if isinstance(size_px, int):
        sw, sh = _svg.svg_size(data)
        k = size_px / max(sw, sh)
        w, h = max(1, round(sw * k)), max(1, round(sh * k))
    else:
        w, h = int(size_px[0]), int(size_px[1])
    return texture_from_array(engine, _svg.rasterize_svg(data, w, h))


@_deferred.deferred_in_pipeline
def write_texture(engine, handle: TextureHandle, rgba: np.ndarray, origin=(0, 0), *, layer=0) -> None:
    """Overwrite part of a texture in place: `rgba` is a (h, w, 4) uint8 sRGB patch with
    straight alpha, written with its top-left corner at `origin` = (x, y) pixels.
    For a texture array, `layer` selects the destination slice.

    Only the full-size level changes. The GUI samples that level of any texture, so a
    map in an Image shows the patch at once; a material's minified view of a texture
    with mips keeps the old image in the smaller levels."""
    rgba = np.ascontiguousarray(rgba, dtype=np.uint8)
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError(f"write_texture expects a (h, w, 4) uint8 array, got {rgba.shape}")
    h, w = rgba.shape[:2]
    x, y = int(origin[0]), int(origin[1])
    if x < 0 or y < 0 or x + w > handle.size[0] or y + h > handle.size[1]:
        raise ValueError(f"a {w}x{h} patch at {(x, y)} does not fit a {handle.size} texture")
    if not 0 <= layer < max(1, handle.layers):
        raise ValueError("texture patch layer is outside the array")
    write_native = getattr(engine, "_write_native_texture", None)
    if write_native is not None:
        return write_native(handle, rgba, x, y, layer)
    engine._device.queue.write_texture(
        {"texture": handle.texture, "mip_level": 0, "origin": (x, y, layer)},
        rgba.tobytes(),
        {"offset": 0, "bytes_per_row": w * 4, "rows_per_image": h},
        (w, h, 1),
    )


def texture_array(engine, layers: int, size: int) -> TextureHandle:
    """An empty array of `layers` square images of `size` pixels a side, sRGB with a full mip chain, for
    write_layer to fill one layer at a time; sampled trilinearly, clamped at the edges. A material reads it
    as a texture_2d_array (StandardMaterial's baked=). Asked for before the engine has a device (an app
    building its world before its window opens), it is made on first use."""
    upload_native = getattr(engine, "_upload_native_baked_array", None)
    if upload_native is not None:
        if not 1 <= layers <= 2**32 - 1 or not 1 <= size <= 2**32 - 1:
            raise ValueError("baked texture size and layer count must be positive")
        return upload_native(int(layers), int(size))

    registry = engine._texture_registry
    handle = TextureHandle(id=registry.alloc_id(), texture=None, view=None, sampler=None, size=(size, size),
                           layers=int(layers))

    def make():
        device = getattr(engine, "_device", None)
        if device is None:
            raise RuntimeError("engine has no device yet; a texture_array is made once it has one")
        handle.texture = device.create_texture(
            size=(size, size, layers),
            format=wgpu.TextureFormat.rgba8unorm_srgb,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
            mip_level_count=int(size).bit_length(),
            sample_count=1,
        )
        handle.view = handle.texture.create_view(dimension=wgpu.TextureViewDimension.d2_array)
        handle.sampler = device.create_sampler(
            mag_filter=wgpu.FilterMode.linear,
            min_filter=wgpu.FilterMode.linear,
            mipmap_filter=wgpu.MipmapFilterMode.linear,
            address_mode_u=wgpu.AddressMode.clamp_to_edge,
            address_mode_v=wgpu.AddressMode.clamp_to_edge,
            max_anisotropy=MAX_ANISOTROPY,
        )
        handle.make = None

    handle.make = make
    if getattr(engine, "_device", None) is not None:
        make()
    registry.add(handle)
    return handle


def layer_mips(rgb: np.ndarray) -> list[np.ndarray]:
    """An (n, n, 3|4) uint8 sRGB image as the RGBA levels write_layer takes: the image and its halvings to 1x1,
    each a 2x2 box filter of the last (an odd side drops its last row or column). Pure numpy, so a
    streamer's worker makes them off the frame."""
    rgb = np.asarray(rgb, np.uint8)
    if rgb.ndim != 3 or rgb.shape[0] != rgb.shape[1] or rgb.shape[2] not in (3, 4):
        raise ValueError(f"layer_mips expects a square (n, n, 3|4) uint8 image, got {rgb.shape}")
    level = np.full(rgb.shape[:2] + (4,), 255, np.uint8)
    level[..., : rgb.shape[2]] = rgb
    levels = [level]
    while level.shape[0] > 1:
        n = level.shape[0] // 2
        q = level[: 2 * n, : 2 * n].astype(np.uint16).reshape(n, 2, n, 2, 4).sum(axis=(1, 3))
        level = ((q + 2) // 4).astype(np.uint8)
        levels.append(level)
    return levels


@_deferred.deferred_in_pipeline
def write_layer(engine, handle: TextureHandle, layer: int, levels: list[np.ndarray]) -> None:
    """Fill layer `layer` of a texture_array with `levels` (layer_mips of an image of the array's size)."""
    write_native = getattr(engine, "_write_native_baked_layer", None)
    if write_native is not None:
        return write_native(handle, int(layer), levels)

    ready(handle)
    if levels[0].shape[:2] != (handle.size[1], handle.size[0]):
        raise ValueError(f"a {levels[0].shape[:2]} image does not fit a {handle.size} array")
    for i, level in enumerate(levels[: handle.texture.mip_level_count]):
        lh, lw = level.shape[:2]
        engine._device.queue.write_texture(
            {"texture": handle.texture, "mip_level": i, "origin": (0, 0, int(layer))},
            np.ascontiguousarray(level, np.uint8).tobytes(),
            {"offset": 0, "bytes_per_row": lw * 4, "rows_per_image": lh},
            (lw, lh, 1),
        )
