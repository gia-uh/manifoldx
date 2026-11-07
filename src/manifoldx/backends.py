"""Lazy backend imports for rendering canvases.

This module provides functions to create render canvases for different backends.
Each function performs lazy imports to provide clear error messages when
dependencies are missing.

Usage:
    from manifoldx.backends import get_desktop_canvas, get_offscreen_canvas

    # Desktop (requires: pip install manifold-gfx[desktop])
    canvas = get_desktop_canvas(width=800, height=600, fullscreen=False, title="MyApp")

    # Offscreen (requires: pip install manifold-gfx[offline])
    canvas = get_offscreen_canvas(width=1920, height=1080)
"""

from __future__ import annotations

import ctypes
import os


def get_desktop_canvas(
    width: int,
    height: int,
    fullscreen: bool,
    title: str,
    monitor: int | str | None = None,
    max_fps: float = 240.0,
) -> "BaseRenderCanvas":
    """Create a desktop canvas using GLFW.

    Requires: pip install manifold-gfx[desktop]

    Parameters
    ----------
    width : int
        Window width in pixels.
    height : int
        Window height in pixels.
    fullscreen : bool
        Whether to create a fullscreen window.
    title : str
        Window title.
    monitor : int | str | None
        Fullscreen only: index into the monitor list, or a case-insensitive
        substring of a monitor's name. None reads ``MANIFOLDX_MONITOR`` (same
        rules), then falls back to the primary monitor.
    max_fps : float
        Frame-rate cap. rendercanvas defaults to 30; the default here sits
        above common refresh rates, so vsync sets the pace.

    Returns
    -------
    BaseRenderCanvas
        The GLFW render canvas.

    Raises
    ------
    ImportError
        If glfw is not installed.
    RuntimeError
        If not running in a desktop environment.
    """
    # Lazy import - raises clear error if glfw not installed
    try:
        from rendercanvas.glfw import GlfwRenderCanvas
    except ImportError:
        raise ImportError(
            "Desktop backend requires glfw.\nInstall with: pip install manifold-gfx[desktop]"
        )

    # Create the canvas
    canvas = GlfwRenderCanvas(size=(width, height), title=title, max_fps=max_fps)

    # Handle fullscreen via GLFW
    if fullscreen:
        import glfw

        monitors = glfw.get_monitors()
        names = [_monitor_name(glfw, m) for m in monitors]
        primary = _primary_index(glfw, monitors)
        target = monitors[_resolve_monitor(monitor, names, primary)]
        mode = glfw.get_video_mode(target)  # its current mode: no mode switch
        # Stay up when focus moves to another monitor; GLFW would minimize it.
        glfw.set_window_attrib(canvas._window, glfw.AUTO_ICONIFY, False)
        glfw.set_window_monitor(
            canvas._window,
            target,
            0,
            0,
            mode.size[0],
            mode.size[1],
            mode.refresh_rate,
        )

    return canvas


def _monitor_name(glfw, m) -> str:
    name = glfw.get_monitor_name(m)
    return name.decode() if isinstance(name, bytes) else str(name)


def _addr(m) -> int:
    # GLFW monitors are ctypes pointers; two pointer objects to the same
    # monitor compare unequal, so compare their addresses.
    return ctypes.cast(m, ctypes.c_void_p).value


def _primary_index(glfw, monitors) -> int:
    primary = _addr(glfw.get_primary_monitor())
    return next((i for i, m in enumerate(monitors) if _addr(m) == primary), 0)


def _resolve_monitor(spec: int | str | None, names: list[str], primary: int) -> int:
    """Index of the monitor `spec` selects among `names`.

    `spec` is an index, or a case-insensitive substring of a name (a string of
    digits counts as an index). None reads MANIFOLDX_MONITOR, then falls back
    to `primary`. Raises ValueError listing the monitors when nothing matches.
    """
    if spec is None:
        spec = os.environ.get("MANIFOLDX_MONITOR") or None
        if spec is None:
            return primary
    if isinstance(spec, str) and spec.strip().isdigit():
        spec = int(spec)
    if isinstance(spec, int):
        if 0 <= spec < len(names):
            return spec
    else:
        for i, name in enumerate(names):
            if spec.lower() in name.lower():
                return i
    available = ", ".join(f"{i}: {name}" for i, name in enumerate(names))
    raise ValueError(f"no monitor matches {spec!r}; available monitors: {available}")


def list_monitors() -> list[tuple[int, str, tuple[int, int], int, bool]]:
    """Connected monitors as (index, name, (width, height), refresh_hz, is_primary).

    The index and name are what `Engine(monitor=...)` and MANIFOLDX_MONITOR
    accept. Requires: pip install manifold-gfx[desktop]
    """
    import glfw

    if not glfw.init():
        raise RuntimeError("GLFW could not initialize, so no monitors can be listed (no display?)")
    monitors = glfw.get_monitors()
    primary = _primary_index(glfw, monitors)
    out = []
    for i, m in enumerate(monitors):
        mode = glfw.get_video_mode(m)
        size = (int(mode.size[0]), int(mode.size[1]))
        out.append((i, _monitor_name(glfw, m), size, int(mode.refresh_rate), i == primary))
    return out


def get_offscreen_canvas(width: int, height: int) -> "BaseRenderCanvas":
    """Create an offscreen canvas for headless rendering.

    Requires: pip install manifold-gfx[offline]

    Parameters
    ----------
    width : int
        Canvas width in pixels.
    height : int
        Canvas height in pixels.

    Returns
    -------
    BaseRenderCanvas
        The offscreen render canvas.

    Raises
    ------
    ImportError
        If imageio-ffmpeg is not installed.
    """
    # First check for imageio_ffmpeg - raises clear error if not installed
    try:
        import imageio_ffmpeg  # noqa: F401 - just checking availability
    except ImportError:
        raise ImportError(
            "Offline rendering requires imageio-ffmpeg.\n"
            "Install with: pip install manifold-gfx[offline]"
        ) from None

    return get_headless_canvas(width, height)


def get_headless_canvas(width: int, height: int) -> "BaseRenderCanvas":
    """An offscreen canvas for a wgpu device and still frames, with no video writer: what
    render_frame and the exports need, so it does not require the offline extra."""
    from rendercanvas.offscreen import OffscreenRenderCanvas

    return OffscreenRenderCanvas(size=(width, height), format="rgba-u8")


# Type hint for the canvas return type
BaseRenderCanvas = None  # Will be set when rendercanvas is imported
