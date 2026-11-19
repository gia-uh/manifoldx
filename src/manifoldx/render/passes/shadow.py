"""Depth-only shadow pass — renders mesh geometry from the sun's POV.

Runs before the main render pass (encoded at the command-encoder level in
Engine._draw_frame). Writes a depth-only shadow map that the StandardMaterial
fragment shader later samples. No-op unless both a sun (set_sun) and a shadow
config (enable_shadows) are present.
"""

import numpy as np
import wgpu

from manifoldx.skinning import BLOCKS_BINDING, JOINT_LOCATION, JOINTS_BINDING, skin_shadow_shader


def _ensure_shadow_map(rp, resolution):
    if rp._shadow_map is not None and rp._shadow_map_size == resolution:
        return
    rp._shadow_map = rp._device.create_texture(
        size=(resolution, resolution, 1),
        format=wgpu.TextureFormat.depth24plus,
        usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING,
    )
    rp._shadow_map_view = rp._shadow_map.create_view()
    rp._shadow_map_size = resolution


def _storage_entry(binding):
    return {
        "binding": binding,
        "visibility": wgpu.ShaderStage.VERTEX,
        "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
    }


def _ensure_vs_layout(rp):
    if rp._shadow_bind_layout_vs is not None:
        return
    globals_entry = {
        "binding": 0,
        "visibility": wgpu.ShaderStage.VERTEX,
        "buffer": {"type": wgpu.BufferBindingType.uniform},
    }
    layout = rp._device.create_bind_group_layout(entries=[globals_entry, _storage_entry(1)])
    rp._shadow_bind_layout_vs = layout
    rp._shadow_pipeline_layout = rp._device.create_pipeline_layout(bind_group_layouts=[layout])
    # Skinned draws also read the joint blocks and the block index per instance.
    skinned = rp._device.create_bind_group_layout(
        entries=[globals_entry, _storage_entry(1), _storage_entry(JOINTS_BINDING),
                 _storage_entry(BLOCKS_BINDING)]
    )
    rp._shadow_bind_layout_skinned = skinned
    rp._shadow_pipeline_layout_skinned = rp._device.create_pipeline_layout(
        bind_group_layouts=[skinned]
    )


def _get_pipeline(rp, stride, joints_offset=None):
    """One depth-only pipeline per (vertex stride, skinned).

    Skinned geometry (joints_offset set) poses position by the entity's joint
    block; the unskinned pipeline runs _SHADOW_SHADER as it is.
    """
    cache = getattr(rp, "_shadow_pipelines", None)
    if cache is None:
        cache = {}
        rp._shadow_pipelines = cache
    key = (stride, joints_offset is not None)
    if key in cache:
        return cache[key]
    from manifoldx.renderer import _SHADOW_SHADER

    code = skin_shadow_shader(_SHADOW_SHADER) if key[1] else _SHADOW_SHADER
    rp._shadow_shader_sources[key] = code
    attributes = [{"format": wgpu.VertexFormat.float32x3, "offset": 0, "shader_location": 0}]
    if key[1]:
        attributes.append(
            {"format": wgpu.VertexFormat.uint32, "offset": joints_offset,
             "shader_location": JOINT_LOCATION}
        )
    module = rp._device.create_shader_module(code=code)
    pipeline = rp._device.create_render_pipeline(
        layout=rp._shadow_pipeline_layout_skinned if key[1] else rp._shadow_pipeline_layout,
        vertex={
            "module": module,
            "entry_point": "vs_main",
            "buffers": [
                {
                    "array_stride": stride,
                    "step_mode": wgpu.VertexStepMode.vertex,
                    "attributes": attributes,
                }
            ],
        },
        primitive={
            "topology": wgpu.PrimitiveTopology.triangle_list,
            "front_face": wgpu.FrontFace.ccw,
            "cull_mode": wgpu.CullMode.back,
        },
        depth_stencil={
            "format": wgpu.TextureFormat.depth24plus,
            "depth_write_enabled": True,
            "depth_compare": wgpu.CompareFunction.less,
        },
        fragment=None,
    )
    cache[key] = pipeline
    return pipeline


_ARGS = 5  # draw_indexed_indirect's u32s: index count, instance count, first index, base vertex, first instance


class _Casters:
    """The mesh entities that cast, for one live set: `ents` sorted by geometry (entity order kept within
    one), their `geoms` and `mats`, each geometry's `starts` in them and the geometries `uniq`. Kept while
    the live set, its geometries and materials and which materials cast stay, so a still city skips the
    sort; `entries` are its draws as resolved for the geometry registry's `gpu_version` and bind `groups`,
    and `bundle` replays them, recorded against the indirect-args buffer `recorded`."""

    def __init__(self, alive, geoms, mats, mat_ids, casts, ents, geoms_s, mats_s):
        self.alive, self.all_geoms, self.all_mats, self.mat_ids, self.casts = alive, geoms, mats, mat_ids, casts
        self.ents, self.geoms, self.mats = ents, geoms_s, mats_s
        self.starts = np.flatnonzero(np.r_[True, np.diff(geoms_s) != 0]) if len(geoms_s) else np.zeros(0, np.int64)
        self.uniq = geoms_s[self.starts]
        self.bundle, self.recorded, self.index_counts = None, None, np.zeros(len(self.uniq), np.uint32)
        self.gpu_version, self.gpus, self.skinned, self.entries, self.groups = None, [], False, None, None
        self.culled = None


def _casters(rp, engine):
    """The frame's _Casters, the kept one when nothing it was made of changed; None without meshes."""
    store = rp._store
    if "Mesh" not in store._components:
        return None
    alive = np.flatnonzero(store._alive)
    geoms = store.get_component_data("Mesh", alive)[:, 0].astype(np.int64)
    mats = (store.get_component_data("Material", alive)[:, 0].astype(np.int64)
            if "Material" in store._components else np.zeros(len(alive), np.int64))
    kept = rp.__dict__.get("_shadow_casters")
    if (kept is not None and np.array_equal(kept.alive, alive) and np.array_equal(kept.all_geoms, geoms)
            and np.array_equal(kept.all_mats, mats)):
        mat_ids = kept.mat_ids
    else:
        kept, mat_ids = None, np.unique(mats[geoms != 0])
    # A material's opacity may change while the live set stays: which materials cast is asked every frame,
    # once per material.
    casts = tuple(bool(getattr(engine._material_registry.get(int(m)), "casts_shadow", True)) for m in mat_ids)
    if kept is not None and kept.casts == casts:
        return kept
    keep = geoms != 0
    for m, c in zip(mat_ids, casts):
        if not c:
            keep &= mats != m
    idx = np.flatnonzero(keep)
    order = idx[np.argsort(geoms[idx], kind="stable")]
    kept = rp._shadow_casters = _Casters(alive, geoms, mats, mat_ids, casts, alive[order], geoms[order], mats[order])
    return kept


def render_shadow_map(rp, engine, command_encoder):
    cfg = getattr(engine, "_shadow_config", None)
    sun = getattr(engine, "_sun", None)
    spot = getattr(engine, "_spot", None)
    if cfg is None or (sun is None and spot is None) or not rp._initialized:
        return

    _ensure_shadow_map(rp, cfg["resolution"])
    _ensure_vs_layout(rp)

    casters = _casters(rp, engine)
    if casters is None or not len(casters.ents):
        return
    store, cache = rp._store, rp._transform_cache
    if not rp._in_frame:
        cache.refresh(store)
    ents = casters.ents
    dirty = cache._dirty[ents]
    if dirty.any():
        cache._recompute_matrices(store, ents[dirty])
    if not cfg.get("auto_fit", True):
        # Only what the shadow caster's box holds casts; its box is fixed, not fitted to these. The spheres
        # are kept per entity, so a still city costs a compare.
        clip = rp._caster_light_view_proj(engine, 2 if spot is not None else 1)
        box = np.asarray(clip, np.float64).tobytes()
        kept = casters.culled  # (box, transform cache version, drawn casters, counts per geometry)
        if kept is not None and kept[0] == box and not cache.changed_since(kept[1], ents):
            local, counts = kept[2], kept[3]  # a box that stays put over still casters culls the same
        else:
            from manifoldx.renderer import _Rows

            vis = rp._in_frustum(engine, casters.geoms, _Rows(cache._matrix_cache, ents), casters.mats, clip, ents)
            local = ents[vis]
            counts = np.add.reduceat(vis.astype(np.int64), casters.starts)
            casters.culled = (box, cache.version, local, counts)
    else:
        local = ents
        counts = np.diff(np.r_[casters.starts, len(ents)])
    if not len(local):
        return
    first = np.cumsum(counts) - counts

    # Upload the drawn casters' matrices (column-major), geometry after geometry, unless the buffer holds them.
    rp._shadow_batch_buffers.upload_transforms_for(
        local, cache, lambda: cache._matrix_cache[local].reshape(-1, 4, 4).transpose(0, 2, 1).reshape(-1, 16))
    buffers = rp._shadow_batch_buffers
    geometry = engine._geometry_registry
    if casters.gpu_version != geometry._version_value:  # the geometries' buffers, once per geometry change
        casters.gpus = []
        for geom_id in casters.uniq:
            gpu = geometry.get_gpu_buffers(int(geom_id))
            if gpu is None:
                geom = geometry.get(int(geom_id))
                gpu = geometry.create_buffers(int(geom_id), geom, rp._device.queue) if geom else None
            casters.gpus.append(gpu)
        casters.skinned = any(g is not None and g.get("has_joints") for g in casters.gpus)
        casters.gpu_version, casters.entries = geometry._version_value, None
    gpus = casters.gpus
    skinned = _skinned_bind_group(rp, local) if casters.skinned else None
    plain = rp.__dict__.get("_shadow_vs_group")
    if plain is None or plain[0] is not buffers.transforms_buf:
        plain = rp._shadow_vs_group = (buffers.transforms_buf, rp._device.create_bind_group(
            layout=rp._shadow_bind_layout_vs,
            entries=[
                {"binding": 0, "resource": {"buffer": rp._globals_buffer, "offset": 0, "size": 416}},
                {"binding": 1, "resource": {"buffer": buffers.transforms_buf, "offset": 0,
                                            "size": buffers.transforms_capacity}},
            ],
        ))

    groups = (plain[1], skinned)
    if casters.entries is None or casters.groups != groups:
        casters.entries, casters.groups = [], groups  # (slot, pipeline, bind group, vertex buffer, index buffer)
        for b, gpu in enumerate(gpus):
            casters.index_counts[b] = 0 if gpu is None else gpu["index_count"]
            if gpu is None:
                continue
            stride = gpu.get("stride", 6 * 4)
            if gpu.get("has_joints"):
                casters.entries.append((b, _get_pipeline(rp, stride, gpu["joints_offset"]), skinned,
                                        gpu["vertex_buffer"], gpu["index_buffer"]))
            else:
                casters.entries.append((b, _get_pipeline(rp, stride), plain[1], gpu["vertex_buffer"],
                                        gpu["index_buffer"]))
        casters.bundle = None
    entries = casters.entries

    shadow_pass = command_encoder.begin_render_pass(
        color_attachments=[],
        depth_stencil_attachment={
            "view": rp._shadow_map_view,
            "depth_clear_value": 1.0,
            "depth_load_op": wgpu.LoadOp.clear,
            "depth_store_op": wgpu.StoreOp.store,
        },
    )
    if rp._use_bundles:
        # The draws are recorded once into a bundle of indirect draws; a frame writes only their counts.
        args = np.zeros((len(gpus), _ARGS), np.uint32)
        args[:, 0], args[:, 1], args[:, 4] = casters.index_counts, counts, first
        args_buf = _args_buffer(rp, len(gpus))
        data = args.tobytes()
        if rp.__dict__.get("_shadow_args_bytes") != data:  # a still view writes nothing
            rp._device.queue.write_buffer(args_buf, 0, data)
            rp._shadow_args_bytes = data
        if casters.bundle is None or casters.recorded is not args_buf:
            encoder = rp._device.create_render_bundle_encoder(
                color_formats=[], depth_stencil_format=wgpu.TextureFormat.depth24plus)
            _encode(encoder, entries, lambda e, s: e.draw_indexed_indirect(args_buf, 4 * _ARGS * s))
            casters.bundle, casters.recorded = encoder.finish(), args_buf
        shadow_pass.execute_bundles([casters.bundle])
    else:
        _encode(shadow_pass, [e for e in entries if counts[e[0]]],
                lambda p, s: p.draw_indexed(int(casters.index_counts[s]), int(counts[s]), first_index=0,
                                            base_vertex=0, first_instance=int(first[s])))
    shadow_pass.end()


def _args_buffer(rp, draws):
    """The shadow pass's indirect-args buffer, grown to hold `draws` draws."""
    buf = rp.__dict__.get("_shadow_args_buf")
    size = max(4 * _ARGS * draws, 4 * _ARGS)
    if buf is None or buf.size < size:
        buf = rp._shadow_args_buf = rp._device.create_buffer(
            size=size, usage=wgpu.BufferUsage.INDIRECT | wgpu.BufferUsage.COPY_DST)
        rp._shadow_args_bytes = None
    return buf


def _encode(target, entries, draw):
    """Set each entry's state on `target` (a render pass or bundle encoder), none twice in a row, and
    call draw(target, slot) for it."""
    bound = [None] * 4
    for slot, *state in entries:
        pipeline, group, vertex, index = state
        if pipeline is not bound[0]:
            target.set_pipeline(pipeline)
        if group is not bound[1]:
            target.set_bind_group(0, group)
        if vertex is not bound[2]:
            target.set_vertex_buffer(0, vertex)
        if index is not bound[3]:
            target.set_index_buffer(index, wgpu.IndexFormat.uint32)
        bound = state
        draw(target, slot)


def _skinned_bind_group(rp, entities):
    """Bindings 0, 1, 8, 9: the shared ones plus every joint block and, per
    instance in transform order, the entity's block index. Made again only when one of those buffers is new."""
    rp._shadow_batch_buffers.upload_joint_blocks(rp._joints.block_of[entities])
    joints_buf = rp._joints.ensure_buffer(rp._device)
    buffers = rp._shadow_batch_buffers
    key = (buffers.transforms_buf, joints_buf, buffers.joint_blocks_buf)
    kept = rp.__dict__.get("_shadow_skinned_group")
    if kept is not None and kept[0] == key:
        return kept[1]
    group = rp._device.create_bind_group(
        layout=rp._shadow_bind_layout_skinned,
        entries=[
            {"binding": 0, "resource": {"buffer": rp._globals_buffer, "offset": 0, "size": 416}},
            {"binding": 1, "resource": {"buffer": buffers.transforms_buf, "offset": 0,
                                        "size": buffers.transforms_capacity}},
            {"binding": JOINTS_BINDING,
             "resource": {"buffer": joints_buf, "offset": 0, "size": joints_buf.size}},
            {"binding": BLOCKS_BINDING,
             "resource": {"buffer": buffers.joint_blocks_buf, "offset": 0,
                          "size": buffers.joint_blocks_capacity}},
        ],
    )
    rp._shadow_skinned_group = (key, group)
    return group
