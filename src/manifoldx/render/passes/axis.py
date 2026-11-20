"""Axis render pass — LineList primitives for world- and screen-anchored axes."""

import numpy as np
import wgpu
from manifoldx import counters


def render_axis_pass(
    rp, engine, render_pass, axis_batches, model_matrices, material_data
):
    """Draw all axis batches as LineList primitives.

    Each batch is keyed by (geom_id, mat_id) and gets its own pipeline
    (LineList) + per-batch material color uniform. Axes share a dedicated
    _axis_batch_buffers (separate from sprite/label) so their transform
    uploads don't clobber other passes.
    """
    from manifoldx.renderer import _BatchBuffers
    from manifoldx.viz.materials import AxisMaterial

    if rp._axis_batch_buffers is None:
        rp._axis_batch_buffers = _BatchBuffers(rp._device)

    # Pack all axis transforms once. instance_offset/count per batch.
    all_local_indices = []  # one index array per batch, concatenated below
    batch_draw_info = {}  # (geom_id, mat_id) -> (offset, count)
    instance_offset = 0
    for key, local_indices in axis_batches.items():
        count = len(local_indices)
        batch_draw_info[key] = (instance_offset, count)
        all_local_indices.append(np.asarray(local_indices, dtype=np.int64))
        instance_offset += count

    if not instance_offset:
        return

    ent_arr = np.concatenate(all_local_indices)
    all_matrices = model_matrices[ent_arr]
    all_matrices_t = all_matrices.reshape(-1, 4, 4).transpose(0, 2, 1).reshape(-1, 16)
    rp._axis_batch_buffers.upload_transforms(all_matrices_t.astype(np.float32))

    for (geom_id, mat_id), local_indices in axis_batches.items():
        mat_obj = engine._material_registry.get(mat_id) if mat_id > 0 else None
        if not isinstance(mat_obj, AxisMaterial):
            continue
        first_instance, instance_count = batch_draw_info[(geom_id, mat_id)]

        gpu_buffers = engine._geometry_registry.get_gpu_buffers(geom_id)
        if gpu_buffers is None:
            geom_obj = engine._geometry_registry.get(geom_id)
            if geom_obj is not None:
                gpu_buffers = engine._geometry_registry.create_buffers(
                    geom_id, geom_obj, rp._device.queue
                )
        if gpu_buffers is None:
            continue

        pipeline, bind_group_layout = rp._get_or_create_pipeline(
            rp._device,
            engine._texture_format,
            geom_id,
            mat_obj,
            engine._material_registry,
            line=True,
        )

        mat_data = mat_obj.get_data(1, engine._material_registry)  # only row 0 is uploaded
        counters.add("material_reads")
        material_type = type(mat_obj).__name__
        material_subtype = getattr(mat_obj, "pipeline_subtype", None)
        bkey = (geom_id, material_type, material_subtype, "line")
        mat_buffer = rp._material_buffers.get(bkey)
        if mat_buffer is not None:
            first_row = mat_data[0] if mat_data.ndim > 1 else mat_data
            counters.add("material_uploads")
            counters.write_buffer(rp._device.queue,
                mat_buffer, 0, first_row.astype(np.float32).tobytes()
            )

        bind_group = rp._device.create_bind_group(
            layout=bind_group_layout,
            entries=[
                {
                    "binding": 0,
                    "resource": {"buffer": rp._globals_buffer, "offset": 0, "size": 240},
                },
                {
                    "binding": 1,
                    "resource": {
                        "buffer": rp._axis_batch_buffers.transforms_buf,
                        "offset": 0,
                        "size": rp._axis_batch_buffers.transforms_capacity,
                    },
                },
                {
                    "binding": 2,
                    "resource": {"buffer": mat_buffer, "offset": 0, "size": 32},
                },
            ],
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
