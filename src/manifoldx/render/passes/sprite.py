"""Sprite render pass — camera-facing point sprites with sphere imposters."""

import numpy as np
import wgpu

from manifoldx.viz.materials import ColormapMaterial
from manifoldx import counters


_MAX_RUNS = 64  # more runs of changed rows than this: one full upload beats many small writes


def render_sprite_batches(
    rp, engine, render_pass, sprite_batches, model_matrices, material_data
):
    """Render all sprite batches (PointCloud entities).

    Each batch is one (mat_id) group; geometry is always SPRITE_QUAD.
    Per-instance buffers: transforms, scalar_values, radii.
    """
    # Flatten all sprite local indices for single buffer upload
    all_local_indices = []  # one index array per batch, concatenated below
    batch_draw_info = {}  # mat_id -> (first_instance, instance_count)
    instance_offset = 0

    for mat_id, local_indices in sprite_batches.items():
        count = len(local_indices)
        batch_draw_info[mat_id] = (instance_offset, count)
        all_local_indices.append(np.asarray(local_indices, dtype=np.int64))
        instance_offset += count

    if not instance_offset:
        return

    # Per-instance arrays: uploaded again only when the sprites, their transforms, values or radii changed
    # since the last upload, so a static cloud costs a few comparisons a frame.
    ent_arr = np.concatenate(all_local_indices)
    entity_indices = np.flatnonzero(rp._store._alive)[ent_arr]
    if "ScalarValue" in rp._store._components:
        scalar_data = rp._store.get_component_data("ScalarValue", entity_indices)[:, 0].astype(np.float32)
    else:
        scalar_data = np.zeros((len(ent_arr),), dtype=np.float32)
    if "Radius" in rp._store._components:
        radius_data = rp._store.get_component_data("Radius", entity_indices)[:, 0].astype(np.float32)
    else:
        radius_data = np.ones((len(ent_arr),), dtype=np.float32)
    cache, last = rp._transform_cache, getattr(rp, "_sprite_uploaded", None)
    same = last is not None and np.array_equal(last["entities"], entity_indices)
    moved = cache._row_version[entity_indices] > last["version"] if same else None
    changed = (moved | (last["scalars"] != scalar_data) | (last["radii"] != radius_data)) if same else None
    if same and not changed.any():
        pass  # a still cloud: nothing to send
    elif same and len(np.flatnonzero(np.diff(np.flatnonzero(changed)) != 1)) < _MAX_RUNS:
        rows = np.flatnonzero(changed)  # few runs of rows moved or changed: send those alone
        matrices = model_matrices[ent_arr[rows]].reshape(-1, 4, 4).transpose(0, 2, 1).reshape(-1, 16)
        rp._sprite_batch_buffers.write_rows("transforms", rows, matrices)
        rp._sprite_batch_buffers.write_rows("scalar_values", rows, scalar_data[rows])
        rp._sprite_batch_buffers.write_rows("radii", rows, radius_data[rows])
        rp._sprite_uploaded = {"entities": entity_indices, "version": cache.version,
                               "scalars": scalar_data, "radii": radius_data}
    else:
        all_matrices = model_matrices[ent_arr]  # (total_instances, 16)
        all_matrices_t = all_matrices.reshape(-1, 4, 4).transpose(0, 2, 1).reshape(-1, 16)
        rp._sprite_batch_buffers.upload_transforms(all_matrices_t.astype(np.float32))
        rp._sprite_batch_buffers.upload_scalar_values(scalar_data)
        rp._sprite_batch_buffers.upload_radii(radius_data)
        rp._sprite_uploaded = {"entities": entity_indices, "version": cache.version,
                               "scalars": scalar_data, "radii": radius_data}

    # Get sprite quad geometry and create buffers if needed
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

    # ---------------------------------------------------------------
    # Draw each sprite batch
    # ---------------------------------------------------------------
    for mat_id, local_indices in sprite_batches.items():
        mat_obj = engine._material_registry.get(mat_id) if mat_id > 0 else None
        if mat_obj is None:
            continue

        if not isinstance(mat_obj, ColormapMaterial):
            raise TypeError(
                f"sprite batch material must be ColormapMaterial; got {type(mat_obj).__name__}"
            )

        first_instance, instance_count = batch_draw_info[mat_id]

        # Create or fetch sprite pipeline
        pipeline, bind_group_layout = rp._get_or_create_pipeline(
            rp._device,
            engine._texture_format,
            sprite_geom_id,
            mat_obj,
            engine._material_registry,
            sprite=True,
        )

        # Upload material uniforms
        mat_data = mat_obj.get_data(1, engine._material_registry)  # only row 0 is uploaded
        counters.add("material_reads")
        material_type = type(mat_obj).__name__
        material_subtype = getattr(mat_obj, "pipeline_subtype", None)
        bkey = (sprite_geom_id, material_type, material_subtype, True)
        mat_buffer = rp._material_buffers.get(bkey)
        if mat_buffer is not None:
            first_row = mat_data[0] if mat_data.ndim > 1 else mat_data
            counters.add("material_uploads")
            counters.write_buffer(rp._device.queue,
                mat_buffer, 0, first_row.astype(np.float32).tobytes()
            )

        # Build sprite bind group
        bind_group = make_sprite_bind_group(rp, bind_group_layout, mat_obj, mat_buffer)

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


def get_or_create_lut_texture(rp, device, material):
    """Create or retrieve a cached 1D LUT texture + sampler for a colormap."""
    cmap_name = material.cmap
    if cmap_name in rp._lut_textures:
        return rp._lut_textures[cmap_name]

    lut = material.get_lut()  # (256, 4) uint8 — matplotlib-encoded sRGB
    # Use rgba8unorm-srgb so the GPU sRGB-decodes on sample. The framebuffer
    # is also sRGB-encoded on write, so the round trip preserves the
    # author-intended display colors. Without -srgb the gamma curve gets
    # applied twice and colors come out brighter than matplotlib's swatch.
    texture = device.create_texture(
        size=(256, 1, 1),
        format=wgpu.TextureFormat.rgba8unorm_srgb,
        usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        dimension=wgpu.TextureDimension.d1,
    )
    device.queue.write_texture(
        {"texture": texture},
        lut.tobytes(),
        {"bytes_per_row": 256 * 4, "rows_per_image": 1},
        (256, 1, 1),
    )
    sampler = device.create_sampler(
        address_mode_u=wgpu.AddressMode.clamp_to_edge,
        mag_filter=wgpu.FilterMode.linear,
        min_filter=wgpu.FilterMode.linear,
    )
    rp._lut_textures[cmap_name] = (texture, sampler)
    return rp._lut_textures[cmap_name]


def make_sprite_bind_group(rp, bind_group_layout, material, mat_buffer):
    """Create bind group with bindings 0-6 for sprite rendering.

    Bindings:
        0: globals uniform (vp, view, camera_pos)
        1: transforms storage
        2: material uniform (vmin, vmax, lit_flag, _pad)
        3: scalar_values storage
        4: radii storage
        5: lut_texture (1D RGBA8)
        6: lut_sampler
    """
    lut_texture, lut_sampler = get_or_create_lut_texture(rp, rp._device, material)

    mat_buffer_size = 16  # 4 floats: vmin, vmax, lit_flag, _pad

    bind_group_entries = [
        {
            "binding": 0,
            "resource": {
                "buffer": rp._globals_buffer,
                "offset": 0,
                # Bind the full globals block (not just the compact 240 prefix) so
                # sprite/point shaders may also read the lighting fields (sun, etc.).
                # Shaders declaring only the short struct still read a valid prefix.
                "size": rp._globals_buffer.size,
            },
        },
        {
            "binding": 1,
            "resource": {
                "buffer": rp._sprite_batch_buffers.transforms_buf,
                "offset": 0,
                "size": rp._sprite_batch_buffers.transforms_capacity,
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
        {
            "binding": 3,
            "resource": {
                "buffer": rp._sprite_batch_buffers.scalar_values_buf,
                "offset": 0,
                "size": rp._sprite_batch_buffers.scalar_values_capacity,
            },
        },
        {
            "binding": 4,
            "resource": {
                "buffer": rp._sprite_batch_buffers.radii_buf,
                "offset": 0,
                "size": rp._sprite_batch_buffers.radii_capacity,
            },
        },
        {
            "binding": 5,
            "resource": lut_texture.create_view(),
        },
        {
            "binding": 6,
            "resource": lut_sampler,
        },
    ]

    return rp._device.create_bind_group(
        layout=bind_group_layout,
        entries=bind_group_entries,
    )
