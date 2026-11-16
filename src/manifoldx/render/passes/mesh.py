"""Mesh render pass — instanced indexed draws batched by (geom_id, material_type)."""

import numpy as np
import wgpu

from manifoldx.skinning import BLOCKS_BINDING, JOINTS_BINDING


def render_mesh_batches(rp, engine, render_pass, mesh_batches, model_matrices, material_data,
                        entity_indices):
    """Render all mesh batches using instanced draw with shared transform buffer.

    `entity_indices[i]` is the entity behind local index i; skinned batches read
    each instance's joint block through it.
    """
    # ---------------------------------------------------------------
    # Upload ALL transforms at once (queue.write_buffer happens
    # before GPU processes the command buffer, so per-batch writes
    # would be overwritten by the last batch).
    # ---------------------------------------------------------------

    # Flatten batch order and record (first_instance, instance_count) per batch
    all_local_indices = []  # one index array per batch, concatenated below
    batch_draw_info = {}  # key -> (first_instance, instance_count)
    instance_offset = 0

    for key, local_indices in mesh_batches.items():
        count = len(local_indices)
        batch_draw_info[key] = (instance_offset, count)
        all_local_indices.append(np.asarray(local_indices, dtype=np.int64))
        instance_offset += count

    if not instance_offset:
        return
    all_local_indices = np.concatenate(all_local_indices)

    # Transpose all matrices for WGSL column-major layout and upload once, unless the buffer holds them
    rp._batch_buffers.upload_transforms_for(
        np.asarray(entity_indices)[all_local_indices],
        rp._transform_cache,
        lambda: model_matrices[all_local_indices].reshape(-1, 4, 4).transpose(0, 2, 1).reshape(-1, 16)
        .astype(np.float32),
    )
    joint_blocks_uploaded = False
    draw_list = rp.__dict__.setdefault("_draw_list", {})  # batch key -> (buffers, material, state, pipeline, layout)
    seen = set()
    bound = [None] * 6  # pipeline, groups 0-2, vertex buffer, index buffer: what the pass holds now
    mat_states = {}  # mat_id -> (needs_lights, buffer, buffer size), built once this frame

    # ---------------------------------------------------------------
    # Draw each batch using first_instance to index into the
    # shared transform buffer.
    # ---------------------------------------------------------------
    for (geom_id, mat_type, mat_subtype, mat_id), local_indices in mesh_batches.items():
        first_instance, instance_count = batch_draw_info[(geom_id, mat_type, mat_subtype, mat_id)]

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

        # Get material and create/fetch pipeline
        mat_obj = engine._material_registry.get(mat_id) if mat_id > 0 else None
        if mat_obj is None:
            continue

        # The batch's pipeline is kept between frames while its geometry buffers, material object and
        # pipeline state stay the same, so a still scene resolves none.
        state = getattr(mat_obj, "pipeline_state", None)
        entry = draw_list.get((geom_id, mat_type, mat_subtype, mat_id))
        if entry is None or entry[0] is not gpu_buffers or entry[1] is not mat_obj or entry[2] != state:
            entry = (gpu_buffers, mat_obj, state, *rp._get_or_create_pipeline(
                rp._device,
                engine._texture_format,
                geom_id,
                mat_obj,
                engine._material_registry,
                geometry_buffers=gpu_buffers,
            ))
            draw_list[(geom_id, mat_type, mat_subtype, mat_id)] = entry
        seen.add((geom_id, mat_type, mat_subtype, mat_id))
        pipeline, bind_group_layout = entry[3], entry[4]

        # Per-mat_id uniform buffer: one per material instance, allocated
        # lazily on first sight. Sizing follows the existing convention
        # (32B for materials that bind lights, 16B otherwise), or the
        # material's own data when it holds more. Built once per material
        # per frame and written only when its bytes changed.
        mat_state = mat_states.get(mat_id)
        if mat_state is None:
            mat_data = mat_obj.get_data(1, engine._material_registry)
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
                rp._device.queue.write_buffer(mat_buffer, 0, row)
                rp._material_uniforms_by_mat_id[mat_id] = row
            mat_state = mat_states[mat_id] = (needs_lights, mat_buffer, mat_buffer_size)
        needs_lights, mat_buffer, mat_buffer_size = mat_state

        has_joints = bool(gpu_buffers.get("has_joints"))
        if has_joints:
            # Block index per instance, in the same order as the transforms,
            # uploaded once per frame on the first skinned batch.
            if not joint_blocks_uploaded:
                blocks = rp._joints.block_of[np.asarray(entity_indices)[all_local_indices]]
                rp._batch_buffers.upload_joint_blocks(blocks)
                joint_blocks_uploaded = True
            joints_buf = rp._joints.ensure_buffer(rp._device)

        # Every object the bind group points at: it is rebuilt only when one
        # of them changed (a grown buffer, a swapped texture).
        texture_bindings = mat_obj.get_texture_bindings()
        resources = (
            bind_group_layout,
            rp._globals_buffer,
            rp._batch_buffers.transforms_buf,
            mat_buffer,
            joints_buf if has_joints else None,
            rp._batch_buffers.joint_blocks_buf if has_joints else None,
            rp._lights_buffer if needs_lights else None,
            *((b, h.sampler, h.view) for b, h in texture_bindings.items()),
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
                            "buffer": rp._batch_buffers.joint_blocks_buf,
                            "offset": 0,
                            "size": rp._batch_buffers.joint_blocks_capacity,
                        },
                    }
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

        if pipeline is not bound[0]:
            render_pass.set_pipeline(pipeline)
            bound[:4] = [pipeline, None, None, None]  # another layout may not keep the groups
        if bind_group is not bound[1]:
            render_pass.set_bind_group(0, bind_group)
            bound[1] = bind_group

        if needs_lights:
            ibl_env = engine._ready_environment()
            if ibl_env is not None and id(ibl_env) != rp._ibl_env_id:
                rp._upload_ibl_env(rp._device, ibl_env)
            ibl_bg = (
                rp._ibl_active_bind_group
                if ibl_env is not None and rp._ibl_active_bind_group is not None
                else rp._ibl_placeholder_bind_group
            )
            if ibl_bg is not bound[2]:
                render_pass.set_bind_group(1, ibl_bg)
                bound[2] = ibl_bg

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
                    ],
                ))
            if rp._shadow_bind_group[1] is not bound[3]:
                render_pass.set_bind_group(2, rp._shadow_bind_group[1])
                bound[3] = rp._shadow_bind_group[1]

        if gpu_buffers["vertex_buffer"] is not bound[4]:
            render_pass.set_vertex_buffer(0, gpu_buffers["vertex_buffer"])
            bound[4] = gpu_buffers["vertex_buffer"]
        if gpu_buffers["index_buffer"] is not bound[5]:
            render_pass.set_index_buffer(gpu_buffers["index_buffer"], wgpu.IndexFormat.uint32)
            bound[5] = gpu_buffers["index_buffer"]

        # first_instance offsets into the shared transform buffer
        render_pass.draw_indexed(
            gpu_buffers["index_count"],
            instance_count,
            first_index=0,
            base_vertex=0,
            first_instance=first_instance,
        )
    for key in [k for k in draw_list if k not in seen][: max(0, len(draw_list) - 2 * len(seen))]:
        del draw_list[key]  # batches gone for a while: keep at most twice the live count
