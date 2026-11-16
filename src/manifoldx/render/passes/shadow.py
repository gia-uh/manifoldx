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


def render_shadow_map(rp, engine, command_encoder):
    cfg = getattr(engine, "_shadow_config", None)
    sun = getattr(engine, "_sun", None)
    spot = getattr(engine, "_spot", None)
    if cfg is None or (sun is None and spot is None) or not rp._initialized:
        return

    _ensure_shadow_map(rp, cfg["resolution"])
    _ensure_vs_layout(rp)

    instances = rp._collect_mesh_instances(engine, with_entities=True, cull=True)
    if not instances:
        return

    # Upload all model matrices (column-major) once; record per-geometry offsets.
    all_mats, all_entities, draw_info, offset = [], [], {}, 0
    for geom_id, (mats, entities) in instances.items():
        draw_info[geom_id] = (offset, len(mats))
        all_mats.append(mats)
        all_entities.append(entities)
        offset += len(mats)
    rp._shadow_batch_buffers.upload_transforms_for(
        np.concatenate(all_entities),
        rp._transform_cache,
        lambda: np.concatenate(all_mats).reshape(-1, 4, 4).transpose(0, 2, 1).reshape(-1, 16).astype(np.float32),
    )
    skinned_bind_group = None

    bind_group = rp._device.create_bind_group(
        layout=rp._shadow_bind_layout_vs,
        entries=[
            {"binding": 0, "resource": {"buffer": rp._globals_buffer, "offset": 0, "size": 416}},
            {
                "binding": 1,
                "resource": {
                    "buffer": rp._shadow_batch_buffers.transforms_buf,
                    "offset": 0,
                    "size": rp._shadow_batch_buffers.transforms_capacity,
                },
            },
        ],
    )

    shadow_pass = command_encoder.begin_render_pass(
        color_attachments=[],
        depth_stencil_attachment={
            "view": rp._shadow_map_view,
            "depth_clear_value": 1.0,
            "depth_load_op": wgpu.LoadOp.clear,
            "depth_store_op": wgpu.StoreOp.store,
        },
    )
    for geom_id, (first_instance, count) in draw_info.items():
        gpu = engine._geometry_registry.get_gpu_buffers(geom_id)
        if gpu is None:
            geom = engine._geometry_registry.get(geom_id)
            gpu = (
                engine._geometry_registry.create_buffers(geom_id, geom, rp._device.queue)
                if geom
                else None
            )
        if gpu is None:
            continue
        stride = gpu.get("stride", 6 * 4)
        if gpu.get("has_joints"):
            if skinned_bind_group is None:
                skinned_bind_group = _skinned_bind_group(rp, np.concatenate(all_entities))
            shadow_pass.set_bind_group(0, skinned_bind_group)
            shadow_pass.set_pipeline(_get_pipeline(rp, stride, gpu["joints_offset"]))
        else:
            shadow_pass.set_bind_group(0, bind_group)
            shadow_pass.set_pipeline(_get_pipeline(rp, stride))
        shadow_pass.set_vertex_buffer(0, gpu["vertex_buffer"])
        shadow_pass.set_index_buffer(gpu["index_buffer"], wgpu.IndexFormat.uint32)
        shadow_pass.draw_indexed(
            gpu["index_count"], count, first_index=0, base_vertex=0, first_instance=first_instance
        )
    shadow_pass.end()


def _skinned_bind_group(rp, entities):
    """Bindings 0, 1, 8, 9: the shared ones plus every joint block and, per
    instance in transform order, the entity's block index."""
    rp._shadow_batch_buffers.upload_joint_blocks(rp._joints.block_of[entities])
    joints_buf = rp._joints.ensure_buffer(rp._device)
    buffers = rp._shadow_batch_buffers
    return rp._device.create_bind_group(
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
