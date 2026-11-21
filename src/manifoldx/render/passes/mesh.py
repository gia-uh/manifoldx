"""Mesh render pass — instanced indexed draws batched by (geom_id, material_type).

Each batch's state (pipeline, bind groups, buffers) is resolved once into a draw list, kept while the
batches, the registries, the materials and the buffers they bind stay the same. With the device's
indirect-first-instance feature the draw list is recorded once into a render bundle of indirect
draws, whose instance counts and first rows a frame writes into one small buffer: a still scene, a
moving camera and moving instances replay it with one call. Without it the pass encodes the draw list.
"""

import numpy as np
import wgpu

from manifoldx.resources import TINTS_BINDING
from manifoldx.skinning import BLOCKS_BINDING, JOINTS_BINDING
from manifoldx import counters

_ARGS = 5  # draw_indexed_indirect's u32s: index count, instance count, first index, base vertex, first instance


_CHUNK = 32  # opaque batches per render bundle: a batch that comes or goes re-records one bundle


class _Chunk:
    """Batches recorded together: `keys`, their indirect-args `slots`, the geometry buffers they were
    resolved with (`bufs`), their materials (`mids`), the resolved `entries` (slot, pipeline, group 0,
    group 1, group 2, vertex buffer, index buffer) and the `bundle` replaying them."""

    def __init__(self, rank):
        self.rank, self.keys, self.dirty = rank, [], True
        self.slots, self.bufs, self.mids, self.entries, self.bundle, self.skinned = [], [], set(), [], None, False


class _Draw:
    """The mesh pass's kept draws: every batch has a slot in the indirect-args buffer and sits in one
    chunk. Opaque batches fill chunks of _CHUNK in the order they come, since the depth test makes their
    order free; every other rank keeps the layout's order in one chunk. Valid while `glob` (the transform,
    joint and light buffers) stays the same."""

    def __init__(self, glob):
        self.glob, self.geom_version, self.layout, self.args_buf = glob, None, None, None
        self.slot, self.free, self.chunk_of = {}, [], {}  # key -> slot; free slots; key -> _Chunk
        self.index_counts = np.zeros(0, np.uint32)  # per slot
        self.chunks: list[_Chunk] = []
        self.mat_sigs = {}  # mat_id -> what its bind group and pipeline were made of
        self.batch_slots = np.zeros(0, np.int64)  # layout batch -> slot

    def take_slot(self, key):
        s = self.free.pop() if self.free else len(self.slot) + len(self.free)
        if s >= len(self.index_counts):  # capacity doubles, so the args buffer grows rarely
            self.index_counts = np.concatenate([self.index_counts, np.zeros(max(64, len(self.index_counts)),
                                                                            np.uint32)])
        self.slot[key] = s
        return s

    def drop(self, key):
        chunk = self.chunk_of.pop(key)
        i = chunk.keys.index(key)
        del chunk.keys[i]
        chunk.dirty = True
        s = self.slot.pop(key)
        self.free.append(s)
        self.index_counts[s] = 0

    def place(self, key, rank):
        """Put a new opaque batch in a chunk recorded again anyway (one a batch left), else the last with
        room, else a new one."""
        self.take_slot(key)
        room = [c for c in self.chunks if c.rank == rank and len(c.keys) < _CHUNK]
        chunk = next((c for c in room if c.dirty), room[-1] if room else None)
        if chunk is None:
            chunk = _Chunk(rank)
            self.chunks.append(chunk)
        chunk.keys.append(key)
        chunk.dirty = True
        self.chunk_of[key] = chunk

    def update(self, layout):
        """Follow the layout's batches: drop the gone ones, place the new ones, re-order other ranks."""
        ranks = dict(zip(layout.keys, layout.ranks))
        for key in [k for k in self.slot if ranks.get(k) != self.chunk_of[k].rank]:
            self.drop(key)
        opaque = (False, False)
        for key, rank in zip(layout.keys, layout.ranks):
            if key not in self.slot and rank == opaque:
                self.place(key, rank)
        for rank in sorted(set(layout.ranks) - {opaque}):
            keys = [k for k, r in zip(layout.keys, layout.ranks) if r == rank]
            chunk = next((c for c in self.chunks if c.rank == rank), None)
            if chunk is None:
                chunk = _Chunk(rank)
                self.chunks.append(chunk)
            if chunk.keys != keys:
                for key in keys:
                    if key not in self.slot:
                        self.take_slot(key)
                        self.chunk_of[key] = chunk
                chunk.keys, chunk.dirty = keys, True
        self.chunks = sorted((c for c in self.chunks if c.keys), key=lambda c: c.rank)  # stable: opaque first
        for c in self.chunks:
            c.slots = [self.slot[k] for k in c.keys]
        self.batch_slots = np.array([self.slot[k] for k in layout.keys], np.int64)
        self.layout = layout


def render_mesh_batches(rp, engine, render_pass, mesh_batches, model_matrices, material_data,
                        entity_indices):
    """Render the frame's mesh batches (a renderer._MeshBatches) with one shared transform buffer.

    `entity_indices[i]` is the entity behind local index i; skinned batches read
    each instance's joint block through it.
    """
    layout, local = mesh_batches.layout, mesh_batches.local
    if not len(local):
        return
    entities = np.asarray(entity_indices)[local]

    # Upload ALL visible transforms at once, batch after batch (queue.write_buffer happens before the
    # GPU runs the command buffer, so per-batch writes would be overwritten by the last batch);
    # transposed for WGSL's column-major layout, unless the buffer holds them already.
    rp._batch_buffers.upload_transforms_for(
        entities,
        rp._transform_cache,
        lambda: model_matrices[local].reshape(-1, 4, 4).transpose(0, 2, 1).reshape(-1, 16)
        .astype(np.float32),
    )

    # Per material, once a frame: its uniform buffer, one per material instance, allocated lazily on
    # first sight. Sizing follows the existing convention (32B for materials that bind lights, 16B
    # otherwise), or the material's own data when it holds more. Written only when its bytes changed.
    mats = {}  # mat_id -> (material, needs_lights, buffer, buffer size)
    for mat_id in layout.mat_ids:
        mat_obj = engine._material_registry.get(mat_id) if mat_id > 0 else None
        if mat_obj is None:
            continue
        mat_data = mat_obj.get_data(1, engine._material_registry)
        counters.add("material_reads")
        first_row = mat_data[0] if mat_data.ndim > 1 else mat_data
        row = first_row.astype(np.float32).tobytes()
        needs_lights = "@binding(3)" in type(mat_obj)._compile()
        mat_buffer_size = max(32 if needs_lights else 16, -(-len(row) // 16) * 16)
        mat_buffer = rp._material_buffers_by_mat_id.get(mat_id)
        if mat_buffer is None:
            mat_buffer = rp._device.create_buffer(
                size=mat_buffer_size,
                usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            )
            rp._material_buffers_by_mat_id[mat_id] = mat_buffer
        if rp._material_uniforms_by_mat_id.get(mat_id) != row:
            counters.add("material_uploads")
            counters.write_buffer(rp._device.queue, mat_buffer, 0, row)
            rp._material_uniforms_by_mat_id[mat_id] = row
        mats[mat_id] = (mat_obj, needs_lights, mat_buffer, mat_buffer_size)

    lit = _lit_groups(rp, engine) if any(m[1] for m in mats.values()) else (None, None)
    joints = {}  # "bufs" -> (joints, joint blocks) once uploaded this frame

    def joint_buffers():
        # Block index per instance, in the same order as the transforms, uploaded once per frame.
        if "bufs" not in joints:
            rp._batch_buffers.upload_joint_blocks(rp._joints.block_of[entities])
            joints["bufs"] = (rp._joints.ensure_buffer(rp._device), rp._batch_buffers.joint_blocks_buf)
        return joints["bufs"]

    draw = rp.__dict__.get("_mesh_draw")
    if draw is not None and any(c.skinned for c in draw.chunks):
        joint_buffers()
    tints = _upload_tints(rp, mesh_batches, entities)
    bundles = rp._use_bundles
    geometry = engine._geometry_registry
    glob = (rp._batch_buffers.transforms_buf, joints.get("bufs"), lit, bundles, tints)
    if draw is None or draw.glob != glob:
        draw = rp._mesh_draw = _Draw(glob)
    if draw.layout is not layout:
        draw.update(layout)
        live = draw.slot
        draw_list = rp.__dict__.setdefault("_draw_list", {})
        for key in [k for k in draw_list if k not in live][: max(0, len(draw_list) - 2 * len(live))]:
            del draw_list[key]  # batches gone for a while: keep at most twice the live count
    args_buf = _args_buffer(rp, len(draw.index_counts)) if bundles else None
    if args_buf is not draw.args_buf:  # the bundles read the old one
        draw.args_buf = args_buf
        for c in draw.chunks:
            c.dirty = True

    # A chunk is recorded again when a geometry it draws got new buffers or a material it binds changed.
    if draw.geom_version != geometry._version_value:
        for c in draw.chunks:
            if not c.dirty and any(geometry.get_gpu_buffers(k[0]) is not b for k, b in zip(c.keys, c.bufs)):
                c.dirty = True
        draw.geom_version = geometry._version_value
    sigs = {mid: (m[0], getattr(m[0], "pipeline_state", None), m[2],
                  tuple((b, h.sampler, h.view) for b, h in m[0].get_texture_bindings().items()))
            for mid, m in mats.items()}
    changed = {mid for mid in layout.mat_ids if sigs.get(mid) != draw.mat_sigs.get(mid)}
    draw.mat_sigs = sigs
    for c in draw.chunks:
        if c.dirty or (changed and not changed.isdisjoint(c.mids)):
            c.entries, c.skinned = _resolve(rp, engine, c.keys, c.slots, mats, lit, joint_buffers, tints,
                                            draw.index_counts)
            c.bufs = [geometry.get_gpu_buffers(k[0]) for k in c.keys]
            c.mids = {k[3] for k in c.keys}
            c.bundle = None
            if bundles:
                encoder = rp._device.create_render_bundle_encoder(
                    color_formats=[engine._texture_format],
                    depth_stencil_format=wgpu.TextureFormat.depth32float,
                )
                _encode(encoder, c.entries, lambda e, s: e.draw_indexed_indirect(args_buf, 4 * _ARGS * s))
                c.bundle = encoder.finish()
                counters.add("bundles_recorded")
            c.dirty = False

    counts = np.zeros(len(draw.index_counts), np.int64)  # per slot: gone and hidden batches draw none
    first = np.zeros(len(draw.index_counts), np.int64)
    counts[draw.batch_slots], first[draw.batch_slots] = mesh_batches.counts, mesh_batches.first
    if bundles:
        args = np.zeros((len(draw.index_counts), _ARGS), np.uint32)
        args[:, 0], args[:, 1], args[:, 4] = draw.index_counts, counts, first
        data = args.tobytes()
        if rp.__dict__.get("_mesh_args_bytes") != data:  # a still view writes nothing
            counters.write_buffer(rp._device.queue, args_buf, 0, data)
            rp._mesh_args_bytes = data
        counters.add("draws", sum(len(c.entries) for c in draw.chunks))
        render_pass.execute_bundles([c.bundle for c in draw.chunks if c.entries])
    else:
        drawn = [e for c in draw.chunks for e in c.entries if counts[e[0]]]
        counters.add("draws", len(drawn))
        _encode(render_pass, drawn,
                lambda p, s: p.draw_indexed(int(draw.index_counts[s]), int(counts[s]), first_index=0,
                                            base_vertex=0, first_instance=int(first[s])))


def _args_buffer(rp, batches):
    """The indirect-args buffer, grown to hold `batches` draws."""
    buf = rp.__dict__.get("_mesh_args_buf")
    size = max(4 * _ARGS * batches, 4 * _ARGS)
    if buf is None or buf.size < size:
        buf = rp._mesh_args_buf = rp._device.create_buffer(
            size=size, usage=wgpu.BufferUsage.INDIRECT | wgpu.BufferUsage.COPY_DST)
        rp._mesh_args_bytes = None
    return buf


def _lit_groups(rp, engine):
    """Groups 1 (IBL) and 2 (shadow map) for materials that bind lights, uploaded or made as needed."""
    ibl_env = engine._ready_environment()
    if ibl_env is not None and id(ibl_env) != rp._ibl_env_id:
        rp._upload_ibl_env(rp._device, ibl_env)
    ibl_bg = (
        rp._ibl_active_bind_group
        if ibl_env is not None and rp._ibl_active_bind_group is not None
        else rp._ibl_placeholder_bind_group
    )
    # group(2): shadow map (active when a sun + enable_shadows() are set,
    # else a 1x1 placeholder so the pipeline layout is always satisfied).
    shadow_on = (
        getattr(engine, "_shadow_config", None) is not None
        and (
            getattr(engine, "_sun", None) is not None
            or getattr(engine, "_spot", None) is not None
        )
        and rp._shadow_map_view is not None
    )
    shadow_view = rp._shadow_map_view if shadow_on else rp._shadow_placeholder_view
    if rp._shadow_bind_group is None or rp._shadow_bind_group[0] is not shadow_view:
        rp._shadow_bind_group = (shadow_view, rp._device.create_bind_group(
            layout=rp._shadow_bind_group_layout,
            entries=[
                {"binding": 0, "resource": shadow_view},
                {"binding": 1, "resource": rp._shadow_sampler},
                {"binding": 2, "resource": {"buffer": rp._cascades_buffer, "offset": 0,
                                            "size": rp._cascades_buffer.size}},
            ],
        ))
    return ibl_bg, rp._shadow_bind_group[1]


def _resolve(rp, engine, keys, slots, mats, lit, joint_buffers, tints, index_counts):
    """Each batch's pipeline, bind groups and buffers: ([entries], any skinned). Writes each slot's
    index count into `index_counts` (0 for a batch that cannot draw)."""
    draw_list = rp.__dict__.setdefault("_draw_list", {})  # batch key -> (buffers, material, state, pipeline, layout)
    entries = []
    skinned = False
    for b, key in zip(slots, keys):
        index_counts[b] = 0
        geom_id, mat_type, mat_subtype, mat_id = key

        # Get GPU buffers for geometry
        gpu_buffers = engine._geometry_registry.get_gpu_buffers(geom_id)
        if gpu_buffers is None:
            geom_obj = engine._geometry_registry.get(geom_id)
            if geom_obj is None:
                continue
            gpu_buffers = engine._geometry_registry.create_buffers(
                geom_id, geom_obj, rp._device.queue
            )
            if gpu_buffers is None:
                continue

        if mat_id not in mats:
            continue
        mat_obj, needs_lights, mat_buffer, mat_buffer_size = mats[mat_id]

        # The batch's pipeline is kept while its geometry buffers, material object and pipeline
        # state stay the same.
        state = getattr(mat_obj, "pipeline_state", None)
        entry = draw_list.get(key)
        if entry is None or entry[0] is not gpu_buffers or entry[1] is not mat_obj or entry[2] != state:
            entry = (gpu_buffers, mat_obj, state, *rp._get_or_create_pipeline(
                rp._device,
                engine._texture_format,
                geom_id,
                mat_obj,
                engine._material_registry,
                geometry_buffers=gpu_buffers,
            ))
            draw_list[key] = entry
        pipeline, bind_group_layout = entry[3], entry[4]

        has_joints = bool(gpu_buffers.get("has_joints"))
        if has_joints:
            joints_buf, blocks_buf = joint_buffers()
            skinned = True
        tinted = _reads_tints(mat_subtype)

        # Every object the bind group points at: it is rebuilt only when one
        # of them changed (a grown buffer, a swapped texture).
        texture_bindings = mat_obj.get_texture_bindings()
        resources = (
            bind_group_layout,
            rp._globals_buffer,
            rp._batch_buffers.transforms_buf,
            mat_buffer,
            joints_buf if has_joints else None,
            blocks_buf if has_joints else None,
            rp._lights_buffer if needs_lights else None,
            tints if tinted else None,
            *((b_, h.sampler, h.view) for b_, h in texture_bindings.items()),
        )
        cached = rp._mesh_bind_groups.get((mat_id, has_joints))
        if cached is not None and cached[0] == resources:
            bind_group = cached[1]
        else:
            bind_group_entries = [
                {
                    "binding": 0,
                    "resource": {
                        "buffer": rp._globals_buffer,
                        "offset": 0,
                        "size": 560,
                    },
                },
                {
                    "binding": 1,
                    "resource": {
                        "buffer": rp._batch_buffers.transforms_buf,
                        "offset": 0,
                        "size": rp._batch_buffers.transforms_capacity,
                    },
                },
                {
                    "binding": 2,
                    "resource": {
                        "buffer": mat_buffer,
                        "offset": 0,
                        "size": mat_buffer_size,
                    },
                },
            ]

            if has_joints:
                bind_group_entries.append(
                    {
                        "binding": JOINTS_BINDING,
                        "resource": {"buffer": joints_buf, "offset": 0, "size": joints_buf.size},
                    }
                )
                bind_group_entries.append(
                    {
                        "binding": BLOCKS_BINDING,
                        "resource": {
                            "buffer": blocks_buf,
                            "offset": 0,
                            "size": rp._batch_buffers.joint_blocks_capacity,
                        },
                    }
                )

            if tinted:
                bind_group_entries.append(
                    {"binding": TINTS_BINDING, "resource": {"buffer": tints, "offset": 0, "size": tints.size}}
                )

            if needs_lights:
                bind_group_entries.append(
                    {
                        "binding": 3,
                        "resource": {
                            "buffer": rp._lights_buffer,
                            "offset": 0,
                            "size": 128,
                        },
                    }
                )

            # Append texture bindings declared by the material
            # (sampler at N, view at N+1 for each entry).
            for binding, handle in texture_bindings.items():
                bind_group_entries.append(
                    {
                        "binding": binding,
                        "resource": handle.sampler,
                    }
                )
                bind_group_entries.append(
                    {
                        "binding": binding + 1,
                        "resource": handle.view,
                    }
                )

            bind_group = rp._device.create_bind_group(
                layout=bind_group_layout,
                entries=bind_group_entries,
            )
            rp._mesh_bind_groups[(mat_id, has_joints)] = (resources, bind_group)

        ibl_bg, shadow_bg = lit if needs_lights else (None, None)
        entries.append((b, pipeline, bind_group, ibl_bg, shadow_bg,
                        gpu_buffers["vertex_buffer"], gpu_buffers["index_buffer"]))
        index_counts[b] = gpu_buffers["index_count"]
    return entries, skinned


def _is_tinted(subtype) -> bool:
    return isinstance(subtype, str) and subtype.startswith("vcolor-tint")


def _is_faded(subtype) -> bool:
    return isinstance(subtype, str) and (subtype == "fade" or subtype.endswith("-fade"))


def _is_baked(subtype) -> bool:
    return isinstance(subtype, str) and "-baked" in subtype


def _reads_tints(subtype) -> bool:
    return _is_tinted(subtype) or _is_faded(subtype) or _is_baked(subtype)


def _upload_tints(rp, mesh_batches, entities):
    """The tints buffer, one (4,) row per drawn instance in the transforms' order (rgb the Tint, w the
    Fade; x the Baked layer for a baked material, which takes no Tint), written when its bytes changed;
    None when the layout holds no tinted, faded or baked batch. Only those batches' rows are gathered (an
    entity without a Tint keeps its colours, one without a Fade is drawn whole, one without a Baked layer
    draws its own colours); every other row is ones. A batch culled this frame still binds the buffer, so
    it is kept, unwritten."""
    layout = mesh_batches.layout
    kinds = layout.__dict__.get("_tinted")
    if kinds is None:
        kinds = layout._tinted = (np.array([_is_tinted(k[2]) for k in layout.keys], bool),
                                  np.array([_is_faded(k[2]) for k in layout.keys], bool),
                                  np.array([_is_baked(k[2]) for k in layout.keys], bool))
    tinted, faded, baked = kinds
    flags = tinted | faded | baked
    if not flags.any():
        return None
    bufs = rp._batch_buffers
    if not mesh_batches.counts[flags].any():
        if bufs.tints_buf is None:
            bufs.upload_tints(np.ones((1, 4), np.float32))
        return bufs.tints_buf
    out = np.ones((len(entities), 4), np.float32)
    if baked.any():
        out[np.repeat(baked, mesh_batches.counts), 0] = -1.0  # no layer until a Baked gives one
    for name, kind, cols in (("Tint", tinted, slice(0, 3)), ("Fade", faded, 3), ("Baked", baked, 0)):
        values = rp._store._components.get(name)
        if values is None or not kind.any():
            continue
        rows = np.flatnonzero(np.repeat(kind, mesh_batches.counts))
        ents = entities[rows]
        has = rp._store._members[name][ents]
        out[rows[has], cols] = values[ents[has]] if name == "Tint" else values[ents[has]].reshape(-1)
    data = out.tobytes()
    if bufs.tints_buf is None or bufs.tints_capacity < len(data):
        bufs.upload_tints(out)
    elif rp.__dict__.get("_mesh_tint_bytes") != data:  # a still view writes nothing
        counters.write_buffer(rp._device.queue, bufs.tints_buf, 0, data)
    rp._mesh_tint_bytes = data
    return bufs.tints_buf


def _encode(target, entries, draw):
    """Set each entry's state on `target` (a render pass or bundle encoder), none twice in a row, and
    call draw(target, batch) for it."""
    bound = [None] * 6  # pipeline, groups 0-2, vertex buffer, index buffer: what the target holds now
    for b, pipeline, bind_group, ibl_bg, shadow_bg, vertex_buffer, index_buffer in entries:
        if pipeline is not bound[0]:
            target.set_pipeline(pipeline)
            bound[:4] = [pipeline, None, None, None]  # another layout may not keep the groups
        if bind_group is not bound[1]:
            target.set_bind_group(0, bind_group)
            bound[1] = bind_group
        if ibl_bg is not None and ibl_bg is not bound[2]:
            target.set_bind_group(1, ibl_bg)
            bound[2] = ibl_bg
        if shadow_bg is not None and shadow_bg is not bound[3]:
            target.set_bind_group(2, shadow_bg)
            bound[3] = shadow_bg
        if vertex_buffer is not bound[4]:
            target.set_vertex_buffer(0, vertex_buffer)
            bound[4] = vertex_buffer
        if index_buffer is not bound[5]:
            target.set_index_buffer(index_buffer, wgpu.IndexFormat.uint32)
            bound[5] = index_buffer
        draw(target, b)
