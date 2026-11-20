"""Label render pass — camera-facing billboards textured from a shared atlas."""

import numpy as np
import wgpu
from manifoldx import counters


def render_label_pass(
    rp, engine, render_pass, label_batches, model_matrices, material_data
):
    """Draw all label batches with alpha-blend on, depth-write off.

    Mirrors `render_sprite_batches` but with the label-specific bindings:
    atlas texture array + slice index per instance instead of LUT + scalar.
    """
    from manifoldx.renderer import _BatchBuffers
    from manifoldx.viz.materials import LabelMaterial

    all_local_indices = []  # one index array per batch, concatenated below
    batch_draw_info = {}
    instance_offset = 0
    for mat_id, local_indices in label_batches.items():
        count = len(local_indices)
        batch_draw_info[mat_id] = (instance_offset, count)
        all_local_indices.append(np.asarray(local_indices, dtype=np.int64))
        instance_offset += count

    if not instance_offset:
        return

    ent_arr = np.concatenate(all_local_indices)

    # Transforms for label entities (column-major upload, like sprites).
    # A dedicated label batch buffer is required: reusing the sprite buffer
    # would clobber sprite transforms within the same render pass.
    all_matrices = model_matrices[ent_arr]
    all_matrices_t = all_matrices.reshape(-1, 4, 4).transpose(0, 2, 1).reshape(-1, 16)
    if rp._label_batch_buffers is None:
        rp._label_batch_buffers = _BatchBuffers(rp._device)
    rp._label_batch_buffers.upload_transforms(all_matrices_t.astype(np.float32))

    # Per-instance label slice indices.
    if "TextLabel" in rp._store._components:
        alive_indices = np.where(rp._store._alive)[0]
        entity_indices = alive_indices[ent_arr]
        label_data = rp._store.get_component_data("TextLabel", entity_indices)
        rp._label_batch_buffers.upload_label_indices(
            label_data[:, 0].astype(np.float32)
        )
    else:
        rp._label_batch_buffers.upload_label_indices(
            np.zeros(len(ent_arr), dtype=np.float32)
        )

    # Ensure the atlas's GPU texture is current.
    atlas = engine.get_label_atlas()
    atlas.upload_dirty(rp._device, rp._device.queue)
    if atlas.gpu_texture is None:
        return  # nothing to draw — no labels were registered

    # Sprite quad geometry (shared with the sprite path).
    sprite_geom_id = engine._geometry_registry.get_id("sprite_quad")
    gpu_buffers = engine._geometry_registry.get_gpu_buffers(sprite_geom_id)
    if gpu_buffers is None:
        geom_obj = engine._geometry_registry.get(sprite_geom_id)
        if geom_obj is not None:
            gpu_buffers = engine._geometry_registry.create_buffers(
                sprite_geom_id, geom_obj, rp._device.queue
            )
    if gpu_buffers is None:
        return

    for mat_id, local_indices in label_batches.items():
        mat_obj = engine._material_registry.get(mat_id) if mat_id > 0 else None
        if not isinstance(mat_obj, LabelMaterial):
            continue
        first_instance, instance_count = batch_draw_info[mat_id]

        pipeline, bind_group_layout = rp._get_or_create_pipeline(
            rp._device,
            engine._texture_format,
            sprite_geom_id,
            mat_obj,
            engine._material_registry,
            label=True,
        )

        mat_data = mat_obj.get_data(1, engine._material_registry)  # only row 0 is uploaded
        counters.add("material_reads")
        material_type = type(mat_obj).__name__
        material_subtype = getattr(mat_obj, "pipeline_subtype", None)
        bkey = (sprite_geom_id, material_type, material_subtype, "label")
        mat_buffer = rp._material_buffers.get(bkey)
        if mat_buffer is not None:
            first_row = mat_data[0] if mat_data.ndim > 1 else mat_data
            counters.add("material_uploads")
            counters.write_buffer(rp._device.queue,
                mat_buffer, 0, first_row.astype(np.float32).tobytes()
            )

        bind_group = make_label_bind_group(
            rp, bind_group_layout, atlas, mat_buffer
        )

        render_pass.set_pipeline(pipeline)
        render_pass.set_bind_group(0, bind_group, [], 0, 0)
        render_pass.set_vertex_buffer(0, gpu_buffers["vertex_buffer"])
        render_pass.set_index_buffer(gpu_buffers["index_buffer"], wgpu.IndexFormat.uint32)
        counters.add("draws")
        render_pass.draw_indexed(
            gpu_buffers["index_count"],
            instance_count,
            first_index=0,
            base_vertex=0,
            first_instance=first_instance,
        )


def make_label_bind_group(rp, bind_group_layout, atlas, mat_buffer):
    """Bindings 0-5 for label rendering.

    0: globals uniform (208 bytes)
    1: transforms storage
    2: material uniform (16 bytes)
    3: label_indices storage
    4: atlas_texture (texture_2d_array)
    5: atlas_sampler
    """
    return rp._device.create_bind_group(
        layout=bind_group_layout,
        entries=[
            {
                "binding": 0,
                "resource": {"buffer": rp._globals_buffer, "offset": 0, "size": 240},
            },
            {
                "binding": 1,
                "resource": {
                    "buffer": rp._label_batch_buffers.transforms_buf,
                    "offset": 0,
                    "size": rp._label_batch_buffers.transforms_capacity,
                },
            },
            {
                "binding": 2,
                "resource": {"buffer": mat_buffer, "offset": 0, "size": 16},
            },
            {
                "binding": 3,
                "resource": {
                    "buffer": rp._label_batch_buffers.label_indices_buf,
                    "offset": 0,
                    "size": rp._label_batch_buffers.label_indices_capacity,
                },
            },
            {
                "binding": 4,
                "resource": atlas.gpu_texture.create_view(
                    dimension=wgpu.TextureViewDimension.d2_array,
                ),
            },
            {
                "binding": 5,
                "resource": atlas.gpu_sampler,
            },
        ],
    )
