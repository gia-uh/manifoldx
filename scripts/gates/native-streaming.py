"""Verify the Rust wheel's streamed publications and sampled resource ownership."""

import argparse
import json
import sys
import time
from pathlib import Path

import manifoldx as mx
import numpy as np
from PIL import Image

from manifoldx.camera import Camera
from manifoldx.components import Layer, Material, Mesh, Transform
from manifoldx.resources import DirectionalLight, FlatMaterial, StandardMaterial, plane
from manifoldx.streaming import MeshData, Streamer, _pack_native_vertices


def rejected(call, message):
    try:
        call()
    except (ValueError, RuntimeError, TypeError) as error:
        assert message in str(error), str(error)
        return str(error)
    raise AssertionError(f"expected rejection containing {message!r}")


def save(output, name, pixels):
    Image.fromarray(pixels).save(output / f"{name}.png")
    return np.median(pixels[50:70, 70:90], axis=(0, 1)).tolist()


def engine_for(name):
    engine = mx.Engine(name, width=160, height=120, max_entities=32)
    assert type(engine).__name__ == "_NativeEngine"
    engine.camera.set_pose((0, 0, 4), (0, 0, 0))
    engine.set_sun(DirectionalLight("#ffffff", 3.0, (0, 0, -1)))
    return engine


def solid(pixels, channel):
    rgb = np.median(pixels, axis=(0, 1))
    assert rgb[channel] > max(np.delete(rgb, channel)) + 40, rgb


def views(output, fault):
    engine = engine_for("Sampled native secondary camera")
    other = engine_for("Foreign renderer")
    camera = Camera()
    camera.layers = {1}
    camera.set_pose((0, 0, 4), (0, 0, 0))
    engine.camera.layers = {2}
    red = FlatMaterial("#e83728")
    source = engine.spawn(Mesh(plane(3, 3)), Material(red), Transform(), Layer(layer=1))
    texture = engine.render_to_texture(camera, (40, 40), live=False, background="#28dc44")
    assert rejected(lambda: other.read_texture(texture), "different")
    material = StandardMaterial("#ffffff", roughness=0.8, albedo_map=texture, vertex_colors=True)
    engine.spawn(Mesh(plane(3, 3)), Material(material), Transform(), Layer(layer=2))
    read = engine.read_texture(texture)
    first = engine.render_frame()
    first_order = engine.last_frame["frame_order"]
    solid(read.result(timeout=1), 0)
    first_rgb = save(output, "snapshot", first)
    assert first_rgb[0] > first_rgb[1] + 40, first_rgb
    assert first_order[0] == "view", first_order

    camera.set_pose((20, 0, 4), (20, 0, 0))
    assert np.array_equal(first, engine.render_frame())
    assert "view" not in engine.last_frame["frame_order"]
    if fault != "refresh":
        engine.refresh_texture(texture)
    read = engine.read_texture(texture)
    refreshed = engine.render_frame()
    refreshed_rgb = save(output, "refreshed", refreshed)
    assert refreshed_rgb[1] > refreshed_rgb[0] + 40, refreshed_rgb
    solid(read.result(timeout=1), 1)

    engine.set_texture_live(texture, True)
    camera.set_pose((0, 0, 4), (0, 0, 0))
    assert np.array_equal(engine.render_frame(), first)
    engine.drop_texture(texture)
    assert rejected(lambda: engine.refresh_texture(texture), "dropped")
    camera.set_pose((20, 0, 4), (20, 0, 0))
    last = engine.read_texture(texture)
    assert np.array_equal(engine.render_frame(), first)
    solid(last.result(timeout=1), 0)
    assert engine.last_frame["view_resources"] == [0, 1]
    del texture, material
    assert np.array_equal(engine.render_frame(), first)
    assert engine.last_frame["view_resources"] == [0, 1]
    engine.clear_scene()
    assert rejected(lambda: source.transform.pos, "stale")
    drained = engine.render_frame()
    save(output, "drained", drained)
    assert engine.last_frame["view_resources"] == [0, 0]
    reopened = engine.render_to_texture(camera, (24, 18), live=False)
    new_read = engine.read_texture(reopened)
    engine.render_frame()
    assert new_read.result(timeout=1).shape == (18, 24, 3)
    result = {"first_order": first_order, "first_rgb": first_rgb, "refreshed_rgb": refreshed_rgb,
              "retired_views": [0, 0], "reopened_views": engine.last_frame["view_resources"]}
    engine.close()
    other.close()
    return result


def albedo(output, fault):
    engine = engine_for("Same-ID albedo replacement")
    old = Image.new("RGBA", (8, 8), (225, 20, 30, 255))
    handle = engine._upload_native_albedo(old, None)
    material = StandardMaterial("#ffffff", roughness=0.8, albedo_map=handle, vertex_colors=True)
    engine.spawn(Mesh(plane(3, 3)), Material(material), Transform())
    index = engine._material(material)
    engine._graphics.surface_patterns(index, [material.surface_patterns[i] for i in range(1, 10)], True)
    before = save(output, "original-albedo", engine.render_frame())
    if fault != "albedo":
        engine._graphics.albedo(handle.id, 8, 8,
                               [bytes((20, 225, 30, 255)) * (size * size) for size in (8, 4, 2, 1)])
    after = save(output, "replaced-albedo", engine.render_frame())
    assert before[0] > before[1] + 40, before
    assert after[1] > after[0] + 40, after
    engine.close()
    return {"before": before, "after": after}


def install_mesh(engine, color):
    geometry = plane(2.5, 2.5)
    geometry["colors"] = np.tile(np.asarray(color, np.float32), (4, 1))
    vb = _pack_native_vertices(geometry)
    ib = np.asarray(geometry["indices"], np.uint32).tobytes()
    mesh = next(engine._mesh_ids)
    engine._graphics.stream_begin(mesh, [[-1.25, -1.25, 0], [1.25, 1.25, 0]], len(vb), len(ib))
    engine._graphics.stream_write(mesh, True, vb)
    engine._graphics.stream_write(mesh, False, ib)
    return mesh


def transaction(output, fault):
    engine = engine_for("Atomic streamed cell")
    engine.configure_stream_collision(4)
    engine.world.register("FixtureRows", kind="u32", width=1)
    graphics = engine._graphics
    material = engine._material(StandardMaterial("#ffffff", roughness=0.8, vertex_colors=True))
    red, green = install_mesh(engine, (0.9, 0.05, 0.05)), install_mesh(engine, (0.05, 0.9, 0.05))
    key = (1, 0, 0, 0)

    def commit(version, mesh, height, invalid=False):
        grids = [((0, 0), (4, 4), (2, 2), [height] * 4, 0.8, 42)]
        tiles = [((0, 0), version, grids, [], [])]
        rows = [(1, [("FixtureRows", None, [int(height), 99] if invalid else [int(height)])])]
        graphics.stream_commit(key, version, [(mesh, material, None, None)], (0, 0, 0), rows, tiles, True)
        return engine.render_frame()

    before = commit(1, red, 1.0)
    assert graphics.stream_result(1)
    assert engine.sample_stream_collision(2, 2, 1)[0] == 1
    assert save(output, "version-1", before)[0] > 100
    bad = commit(2, green, 2.0, invalid=fault != "atomic")
    assert rejected(lambda: graphics.stream_result(2), "payload")
    assert np.array_equal(before, bad)
    assert engine.sample_stream_collision(2, 2, 1)[0] == 1
    after = commit(3, green, 3.0)
    assert graphics.stream_result(3)
    assert engine.sample_stream_collision(2, 2, 3)[-1] == 3
    assert save(output, "version-3", after)[1] > 100
    stale_version = 4 if fault == "late" else 2
    stale = commit(stale_version, red, 2.0)
    assert rejected(lambda: graphics.stream_result(stale_version), "stale")
    assert np.array_equal(after, stale)
    assert engine.sample_stream_collision(2, 2, 3)[0] == 3
    assert engine.world.query("FixtureRows").read().numpy().tolist() == [[3]]
    graphics.stream_hide(key, 1)
    assert np.array_equal(engine.render_frame(), after)
    graphics.stream_hide(key, 3)
    graphics.stream_retire(red)
    graphics.stream_retire(green)
    save(output, "retired", engine.render_frame())
    assert engine.sample_stream_collision(2, 2, 3) is None
    assert engine.last_frame["stream_resources"] == [0, 0]
    assert engine.last_frame["streamed_cells"] == 0
    engine.close()
    return {"versions_observed": [1, 1, 3, 3], "retired_meshes": [0, 0], "remaining_cells": 0}


async def async_builder(request):
    geometry = plane(2.5, 2.5)
    return MeshData(geometry["positions"], geometry["normals"],
                    np.tile(np.array((0.85, 0.22, 0.12), np.float32), (4, 1)), geometry["indices"])


def async_stream(output):
    engine = engine_for("Async Python builder")
    stream = Streamer(engine, 4, async_builder, radius=0, ready_radius=0, request=True,
                      upload_bytes_per_frame=64)
    for _ in range(90):
        stream.focus(0, 0)
        image = engine.render_frame()
        if stream.settled:
            break
        time.sleep(0.002)
    assert stream.settled and stream.stats()["uploaded"] == 1, stream.stats()
    assert save(output, "async-cell", image)[0] > 100
    stream.close()
    engine.render_frame()
    assert engine.last_frame["stream_resources"] == [0, 0]
    engine.close()
    return stream.stats()


def capsule():
    import native
    from manifoldx._core_native import _StreamBuilder

    assert Path(native.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    good = _StreamBuilder(native.build_cell)
    report = {"native_module": Path(native.__file__).name}
    report["wrong_version"] = rejected(lambda: _StreamBuilder(native.build_cell_v2), "v2")
    report["invalid_request"] = rejected(lambda: good.build(0, 0, float("nan"), 0, 0), "positive size")
    malformed = _StreamBuilder(native.build_cell_malformed)
    report["invalid_output"] = rejected(lambda: malformed.build(0, 0, 100, 0, 0), "invalid native mesh")
    panicking = _StreamBuilder(native.build_cell_panics)
    report["panic"] = rejected(lambda: panicking.build(0, 0, 100, 0, 0), "boom from the native builder")
    meshes, rows = good.build(0, 0, 100, 0, 0)
    assert len(meshes) == 2 and rows and portable_rows(rows)
    first = bytes(meshes[0][1])
    good.build(100, 0, 125, 1, 10)
    assert meshes[0][1] == first
    report["owned_vertex_bytes"] = len(first)
    report["copied_row_bytes"] = sum(len(data) for _, _, data in rows)
    return report


def portable_rows(rows):
    return all(width == 12 and len(payload) % width == 0 for _, width, payload in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--case", required=True, choices=("views", "albedo", "transaction", "async", "capsule"))
    parser.add_argument("--fault", choices=("refresh", "albedo", "atomic", "late"))
    args = parser.parse_args()
    assert sys.flags.isolated
    assert Path(mx.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    if args.case == "async":
        result = async_stream(args.output)
    elif args.case == "capsule":
        result = capsule()
    else:
        case = {"views": views, "albedo": albedo, "transaction": transaction}[args.case]
        result = case(args.output, args.fault)
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
