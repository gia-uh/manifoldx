"""Depth-only shadow pass — renders mesh geometry from the sun's POV.

Runs before the main render pass (encoded at the command-encoder level in
Engine._draw_frame). Writes a depth-only shadow map that the StandardMaterial
fragment shader later samples. No-op unless both a sun (set_sun) and a shadow
config (enable_shadows) are present.
"""

import numpy as np
import wgpu

from manifoldx.resources import FADE_WGSL, TINTS_BINDING
from manifoldx.skinning import BLOCKS_BINDING, JOINT_LOCATION, JOINTS_BINDING, skin_shadow_shader
from manifoldx import counters


def _ensure_shadow_map(rp, resolution, layers):
    """The shadow map: `layers` depth layers of `resolution` squared, a view of each to draw into and
    one array view the Standard shader samples."""
    if rp._shadow_map is not None and rp._shadow_map_size == resolution and len(rp._shadow_layer_views) == layers:
        return
    rp._shadow_map = rp._device.create_texture(
        size=(resolution, resolution, layers),
        format=wgpu.TextureFormat.depth24plus,
        usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING,
    )
    rp._shadow_map_view = rp._shadow_map.create_view(dimension=wgpu.TextureViewDimension.d2_array)
    rp._shadow_layer_views = [
        rp._shadow_map.create_view(dimension=wgpu.TextureViewDimension.d2, base_array_layer=k, array_layer_count=1)
        for k in range(layers)
    ]
    rp._shadow_map_size = resolution


_SLOT = 256  # bytes between the layers' light matrices in their uniform buffer (the offset alignment)


class _LayerGpu:
    """One shadow-map layer's buffers and bind groups, kept between frames: the drawn casters' matrices
    (`buffers`), its bind groups, the fades and indirect args last written."""

    def __init__(self, buffers):
        self.buffers = buffers
        self.vs_group = self.skinned_group = self.faded_group = None
        self.fade_bytes = self.args_buf = self.args_bytes = None


def _layer_gpu(rp, k):
    layers = rp.__dict__.setdefault("_shadow_layers", [])
    while len(layers) <= k:
        from manifoldx.renderer import _BatchBuffers

        layers.append(_LayerGpu(rp._shadow_batch_buffers if not layers else _BatchBuffers(rp._device)))
    return layers[k]


def _write_light_matrices(rp, matrices):
    """Each layer's light view-projection (column-major) into its slot of the layers' uniform buffer."""
    from manifoldx.shadow import MAX_CASCADES

    buf = rp.__dict__.get("_shadow_lvp_buf")
    if buf is None:
        buf = rp._shadow_lvp_buf = rp._device.create_buffer(
            size=_SLOT * MAX_CASCADES, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        rp._shadow_lvp_bytes = None
    data = np.zeros((len(matrices), _SLOT // 4), np.float32)
    for k, m in enumerate(matrices):
        data[k, :16] = np.asarray(m, np.float32).T.reshape(-1)
    data = data.tobytes()
    if rp.__dict__.get("_shadow_lvp_bytes") != data:
        counters.write_buffer(rp._device.queue, buf, 0, data)
        rp._shadow_lvp_bytes = data


def _write_cascades(rp, fitted):
    """The Standard shader's Cascades uniform: count 0 when one map covers the caster's box."""
    data = np.zeros(80, np.float32)
    if fitted is not None:
        cascades, splits = fitted
        for k, (m, texel, depth) in enumerate(cascades):
            data[16 * k:16 * k + 16] = np.asarray(m, np.float32).T.reshape(-1)
            data[64 + k], data[68 + k], data[72 + k] = splits[k], texel, depth
        data[76:77] = np.frombuffer(np.uint32(len(cascades)).tobytes(), np.float32)
    data = data.tobytes()
    if rp.__dict__.get("_cascades_bytes") != data:
        counters.write_buffer(rp._device.queue, rp._cascades_buffer, 0, data)
        rp._cascades_bytes = data


def _lvp_entry(rp, k):
    return {"binding": 0, "resource": {"buffer": rp._shadow_lvp_buf, "offset": _SLOT * k, "size": 64}}


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
    # Faded casters also read each instance's fade, the w of its row in the tints buffer.
    faded = rp._device.create_bind_group_layout(
        entries=[globals_entry, _storage_entry(1), _storage_entry(TINTS_BINDING)]
    )
    rp._shadow_bind_layout_faded = faded
    rp._shadow_pipeline_layout_faded = rp._device.create_pipeline_layout(bind_group_layouts=[faded])


def _fade_shadow_shader(source):
    """_SHADOW_SHADER with a fragment stage that drops the pixels a caster's Fade drops (the same 4x4
    pattern, on the shadow map's texels), so a fading caster's shadow fades under PCF."""
    vs = source[source.index("@vertex"):]
    out = source.replace(vs, (
        f"@group(0) @binding({TINTS_BINDING}) var<storage, read> tints: array<vec4<f32>>;\n" + FADE_WGSL + """
struct FadeOut {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) fade: f32,
};

@vertex
fn vs_main(@location(0) position: vec3<f32>,
           @builtin(instance_index) instance: u32) -> FadeOut {
    var out: FadeOut;
    let model = transforms.models[instance];
    let world = (model * vec4<f32>(position, 1.0)).xyz;
    out.position = globals.light_view_proj * vec4<f32>(world, 1.0);
    out.fade = tints[instance].w;
    return out;
}

@fragment
fn fs_fade(in: FadeOut) {
    if (fade_dropped(in.position.xy, in.fade)) {
        discard;
    }
}
"""))
    assert "fs_fade" in out and "return globals.light_view_proj" not in out
    return out


def _get_pipeline(rp, stride, joints_offset=None, faded=False):
    """One depth-only pipeline per (vertex stride, skinned), and per stride a faded one.

    Skinned geometry (joints_offset set) poses position by the entity's joint
    block; the unskinned pipeline runs _SHADOW_SHADER as it is, the faded one
    drops what each caster's Fade drops.
    """
    cache = getattr(rp, "_shadow_pipelines", None)
    if cache is None:
        cache = {}
        rp._shadow_pipelines = cache
    key = (stride, False, "fade") if faded else (stride, joints_offset is not None)
    if key in cache:
        return cache[key]
    from manifoldx.renderer import _SHADOW_SHADER

    if faded:
        code = _fade_shadow_shader(_SHADOW_SHADER)
    else:
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
        layout=(rp._shadow_pipeline_layout_faded if faded
                else rp._shadow_pipeline_layout_skinned if key[1] else rp._shadow_pipeline_layout),
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
        fragment={"module": module, "entry_point": "fs_fade", "targets": []} if faded else None,
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
        self.index_counts = np.zeros(len(self.uniq), np.uint32)
        self.gpu_version, self.gpus, self.skinned = None, [], False
        self.layers = []

    def layer(self, k):
        while len(self.layers) <= k:
            self.layers.append(_Layer())
        return self.layers[k]

    @property
    def bundle(self):
        return self.layer(0).bundle

    @property
    def culled(self):
        return self.layer(0).culled


class _Layer:
    """One layer's draws of a _Casters: `culled` (box, transform cache version, drawn casters, counts per
    geometry), `entries` as resolved for bind `groups`, and the `bundle` replaying them against `recorded`."""

    def __init__(self):
        self.culled = self.entries = self.groups = self.bundle = self.recorded = None


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

    fitted = rp._sun_cascades(engine)
    if fitted is None:  # one map: the caster's box, or the whole scene when auto-fit
        boxes = [None if cfg.get("auto_fit", True) else rp._caster_light_view_proj(engine, 2 if spot is not None else 1)]
        matrices = [rp._caster_light_view_proj(engine, 2 if spot is not None else 1)]
    else:
        boxes = matrices = [c[0] for c in fitted[0]]
    _ensure_shadow_map(rp, cfg["resolution"], len(boxes))
    _write_cascades(rp, fitted)
    _write_light_matrices(rp, matrices)
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
    geometry = engine._geometry_registry
    if casters.gpu_version != geometry._version_value:  # the geometries' buffers, once per geometry change
        casters.gpus = []
        for geom_id in casters.uniq:
            gpu = geometry.get_gpu_buffers(int(geom_id))
            if gpu is None:
                geom = geometry.get(int(geom_id))
                gpu = geometry.create_buffers(int(geom_id), geom, rp._device.queue) if geom else None
            casters.gpus.append(gpu)
            casters.index_counts[len(casters.gpus) - 1] = 0 if gpu is None else gpu["index_count"]
        casters.skinned = any(g is not None and g.get("has_joints") for g in casters.gpus)
        casters.gpu_version = geometry._version_value
        for layer in casters.layers:
            layer.entries = None
    timer = engine._gpu_timer()
    last = len(boxes) - 1
    for k, box in enumerate(boxes):
        writes = None if timer is None else timer.writes("shadow", begin=k == 0, end=k == last)
        _render_layer(rp, engine, casters, k, box, command_encoder, writes)


def _render_layer(rp, engine, casters, k, clip, command_encoder, timestamp_writes):
    """Draw into layer k what its box `clip` holds (every caster when None) from its light matrix."""
    cache, ents, layer, g = rp._transform_cache, casters.ents, casters.layer(k), _layer_gpu(rp, k)
    if clip is not None:
        # Only what the layer's box holds casts. The spheres are kept per entity, so a still city costs a
        # compare, and a box that stays put over still casters culls the same.
        box = np.asarray(clip, np.float64).tobytes()
        kept = layer.culled
        if kept is not None and kept[0] == box and not cache.changed_since(kept[1], ents):
            local, counts = kept[2], kept[3]
        else:
            from manifoldx.renderer import _Rows

            vis = rp._in_frustum(engine, casters.geoms, _Rows(cache._matrix_cache, ents), casters.mats, clip, ents)
            local = ents[vis]
            counts = np.add.reduceat(vis.astype(np.int64), casters.starts)
            layer.culled = (box, cache.version, local, counts)
    else:
        local = ents
        counts = np.diff(np.r_[casters.starts, len(ents)])

    shadow_pass = command_encoder.begin_render_pass(
        timestamp_writes=timestamp_writes,
        color_attachments=[],
        depth_stencil_attachment={
            "view": rp._shadow_layer_views[k],
            "depth_clear_value": 1.0,
            "depth_load_op": wgpu.LoadOp.clear,
            "depth_store_op": wgpu.StoreOp.store,
        },
    )
    if not len(local):
        shadow_pass.end()  # cleared: last frame's casters must not linger in this layer
        return
    first = np.cumsum(counts) - counts

    # Upload the drawn casters' matrices (column-major), geometry after geometry, unless the buffer holds them.
    buffers = g.buffers
    buffers.upload_transforms_for(
        local, cache, lambda: cache._matrix_cache[local].reshape(-1, 4, 4).transpose(0, 2, 1).reshape(-1, 16))
    gpus = casters.gpus
    skinned = _skinned_bind_group(rp, g, k, local) if casters.skinned else None
    if g.vs_group is None or g.vs_group[0] is not buffers.transforms_buf:
        g.vs_group = (buffers.transforms_buf, rp._device.create_bind_group(
            layout=rp._shadow_bind_layout_vs,
            entries=[
                _lvp_entry(rp, k),
                {"binding": 1, "resource": {"buffer": buffers.transforms_buf, "offset": 0,
                                            "size": buffers.transforms_capacity}},
            ],
        ))
    plain = g.vs_group[1]

    fades = _fades(rp, engine, casters, local)
    faded = None if fades is None else _faded_bind_group(rp, g, k, fades)
    groups = (plain, skinned, faded)
    if layer.entries is None or layer.groups != groups:
        layer.entries, layer.groups = [], groups  # (slot, pipeline, bind group, vertex buffer, index buffer)
        for b, gpu in enumerate(gpus):
            if gpu is None:
                continue
            stride = gpu.get("stride", 6 * 4)
            if gpu.get("has_joints"):
                layer.entries.append((b, _get_pipeline(rp, stride, gpu["joints_offset"]), skinned,
                                      gpu["vertex_buffer"], gpu["index_buffer"]))
            elif faded is not None:
                layer.entries.append((b, _get_pipeline(rp, stride, faded=True), faded, gpu["vertex_buffer"],
                                      gpu["index_buffer"]))
            else:
                layer.entries.append((b, _get_pipeline(rp, stride), plain, gpu["vertex_buffer"],
                                      gpu["index_buffer"]))
        layer.bundle = None
    entries = layer.entries

    if rp._use_bundles:
        # The draws are recorded once into a bundle of indirect draws; a frame writes only their counts.
        args = np.zeros((len(gpus), _ARGS), np.uint32)
        args[:, 0], args[:, 1], args[:, 4] = casters.index_counts, counts, first
        args_buf = _args_buffer(g, rp._device, len(gpus))
        data = args.tobytes()
        if g.args_bytes != data:  # a still view writes nothing
            counters.write_buffer(rp._device.queue, args_buf, 0, data)
            g.args_bytes = data
        if layer.bundle is None or layer.recorded is not args_buf:
            encoder = rp._device.create_render_bundle_encoder(
                color_formats=[], depth_stencil_format=wgpu.TextureFormat.depth24plus)
            _encode(encoder, entries, lambda e, s: e.draw_indexed_indirect(args_buf, 4 * _ARGS * s))
            layer.bundle, layer.recorded = encoder.finish(), args_buf
            counters.add("bundles_recorded")
        counters.add("draws", len(entries))
        shadow_pass.execute_bundles([layer.bundle])
    else:
        drawn = [e for e in entries if counts[e[0]]]
        counters.add("draws", len(drawn))
        _encode(shadow_pass, drawn,
                lambda p, s: p.draw_indexed(int(casters.index_counts[s]), int(counts[s]), first_index=0,
                                            base_vertex=0, first_instance=int(first[s])))
    shadow_pass.end()


def _args_buffer(g, device, draws):
    """A layer's indirect-args buffer, grown to hold `draws` draws."""
    size = max(4 * _ARGS * draws, 4 * _ARGS)
    if g.args_buf is None or g.args_buf.size < size:
        g.args_buf = device.create_buffer(size=size, usage=wgpu.BufferUsage.INDIRECT | wgpu.BufferUsage.COPY_DST)
        g.args_bytes = None
    return g.args_buf


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


def _fades(rp, engine, casters, local):
    """Each drawn caster's row for the tints buffer (w its Fade), or None while no caster of a faded
    material is partly faded, so the plain pipelines draw and nothing is gathered."""
    store = rp._store
    values = store._components.get("Fade")
    if values is None:
        return None
    faded_ids = casters.__dict__.get("faded_ids")
    if faded_ids is None:
        registry = engine._material_registry
        faded_ids = casters.faded_ids = np.array(
            [m for m in casters.mat_ids if getattr(registry.get(int(m)), "faded", False)], np.int64)
    if not len(faded_ids):
        return None
    has = store._members["Fade"][local]
    if not has.any():
        return None
    has &= np.isin(store.get_component_data("Material", local)[:, 0].astype(np.int64), faded_ids)
    v = values[local[has]].reshape(-1)
    if not (np.abs(v) < 1.0).any():
        return None
    out = np.ones((len(local), 4), np.float32)
    out[has, 3] = v
    return out


def _faded_bind_group(rp, g, k, fades):
    """Bindings 0, 1, 10: layer k's light matrix and casters plus their fades, written when they changed."""
    buffers = g.buffers
    data = fades.tobytes()
    if buffers.tints_buf is None or buffers.tints_capacity < len(data):
        buffers.upload_tints(fades)
    elif g.fade_bytes != data:
        counters.write_buffer(rp._device.queue, buffers.tints_buf, 0, data)
    g.fade_bytes = data
    key = (buffers.transforms_buf, buffers.tints_buf)
    if g.faded_group is not None and g.faded_group[0] == key:
        return g.faded_group[1]
    group = rp._device.create_bind_group(
        layout=rp._shadow_bind_layout_faded,
        entries=[
            _lvp_entry(rp, k),
            {"binding": 1, "resource": {"buffer": buffers.transforms_buf, "offset": 0,
                                        "size": buffers.transforms_capacity}},
            {"binding": TINTS_BINDING, "resource": {"buffer": buffers.tints_buf, "offset": 0,
                                                    "size": buffers.tints_capacity}},
        ],
    )
    g.faded_group = (key, group)
    return group


def _skinned_bind_group(rp, g, k, entities):
    """Bindings 0, 1, 8, 9: layer k's light matrix and casters plus every joint block and, per instance in
    transform order, the entity's block index. Made again only when one of those buffers is new."""
    buffers = g.buffers
    buffers.upload_joint_blocks(rp._joints.block_of[entities])
    joints_buf = rp._joints.ensure_buffer(rp._device)
    key = (buffers.transforms_buf, joints_buf, buffers.joint_blocks_buf)
    if g.skinned_group is not None and g.skinned_group[0] == key:
        return g.skinned_group[1]
    group = rp._device.create_bind_group(
        layout=rp._shadow_bind_layout_skinned,
        entries=[
            _lvp_entry(rp, k),
            {"binding": 1, "resource": {"buffer": buffers.transforms_buf, "offset": 0,
                                        "size": buffers.transforms_capacity}},
            {"binding": JOINTS_BINDING,
             "resource": {"buffer": joints_buf, "offset": 0, "size": joints_buf.size}},
            {"binding": BLOCKS_BINDING,
             "resource": {"buffer": buffers.joint_blocks_buf, "offset": 0,
                          "size": buffers.joint_blocks_capacity}},
        ],
    )
    g.skinned_group = (key, group)
    return group
