# Pipelined frames — shared-state audit (manifoldx, main @ 2026-11-16)

Scope: `Engine._draw_frame` steps 3–8, `_render_only`, `_draw_passes`, and everything
`RenderPipeline.run()` reaches, versus everything user code in a system / event
handler / async task can mutate. Read-only audit; no repo changes.

Repo: manifoldx, branch `main` at commit 43057af.
Spec: section 08 "Pipelined frames" of `.knowledge/specs/2026-11-16-parallel-systems-design.html`
(as it stood before the deferred-calls revision).
There is no `pipelined` flag in the code yet; every path below is the current serial code.

Type legend: **NP** = numpy array, **PY** = Python container (dict/list/set/object),
**GPU** = wgpu object (buffer/texture/sampler/pipeline/bind group), **S** = scalar/str/tuple.

---

## 1. RENDER READS

Entry points:
- `Engine._render_only` — `src/manifoldx/engine.py:1312-1319`
- `Engine._draw_passes` — `src/manifoldx/engine.py:1321-1391`
- `RenderPipeline.run` — `src/manifoldx/renderer.py:1837-1843`
- `RenderPipeline.render` → `_render_scene_passes` + GUI pass — `src/manifoldx/renderer.py:1508-1523`, `1569-1830`
- `RenderPipeline.render_shadow_map` — `src/manifoldx/renderer.py:1502-1506` → `src/manifoldx/render/passes/shadow.py:110-187`

### 1.1 ECS store (the core of the snapshot)

| State | Attribute path | Read at | Type |
|---|---|---|---|
| Alive mask | `engine.store._alive` | `renderer.py:1469`, `1572`; `render/passes/label.py:43`; `render/passes/sprite.py:37`; `render/passes/gui.py` (via GUI getters) | NP (bool, max_entities) |
| High-water mark | `engine.store._high_water` | `renderer.py:324` (`TransformCache.refresh`) | S (int) |
| Transform rows | `engine.store._components["Transform"]` | `renderer.py:327-328`, `353`; existence check `1584`; writes by cache only | NP (max_entities,10) f32 |
| Transform → matrix cache | `rp._transform_cache._matrix_cache` | `renderer.py:1616`, `1835`; `render/passes/mesh.py:41`, `sprite.py:54,61`, `label.py:35`, `axis.py:37` | NP (max_entities,16) f32 |
| Dirty rows | `rp._transform_cache._dirty` | `renderer.py:1613`, `342` | NP (bool) |
| Seen Transform rows | `rp._transform_cache._seen` | `renderer.py:328`, `407` | NP (max_entities,10) f32 |
| Row version | `rp._transform_cache._row_version` / `version` | `renderer.py:320`, `48` (`sprite.py:48`); `renderer.py:408-409` | NP int64 / S int |
| Mesh geometry ids | `engine.store._components["Mesh"]` | `renderer.py:1473-1479`, `1587`, `1621`; existence `1473` | NP (max_entities,1) u32 |
| Material ids | `engine.store._components["Material"]` | `renderer.py:1481-1482`, `1623-1624`, `1630` | NP (max_entities,1) u32 |
| Volume ids | `engine.store._components["Volume"]` | `renderer.py:1591`, `1626-1627`, `1632` | NP (max_entities,1) u32 (or int) |
| Layer ids | `engine.store._components["Layer"]` | `renderer.py:1575-1579` | NP (max_entities,1) |
| Component presence flags | `engine.store._components` keys `"Mesh"`, `"PointCloud"`, `"TextLabel"`, `"AxisFrame"`, `"Volume"` | `renderer.py:1587-1591` | PY (dict) |
| Scalar channel (sprites) | `engine.store._components["ScalarValue"]` | `render/passes/sprite.py:38-39` | NP |
| Radius channel (sprites) | `engine.store._components["Radius"]` | `render/passes/sprite.py:42-43` | NP |
| Label slice ids | `engine.store._components["TextLabel"]` | `render/passes/label.py:42-47` | NP |
| Spawned/destroyed sets | `engine.store._free_list`, `_members` | not read by render; read/written by `spawn`/`destroy` | PY |

### 1.2 Camera

| State | Path | Read at | Type |
|---|---|---|---|
| Camera object | `engine._camera` | `renderer.py:1573`, `1602`; `engine.py:1467` (swapped for views) | PY |
| Camera layers | `camera.layers` | `renderer.py:1573` | PY (`set[int]`/None) |
| Position | `camera.position` | `renderer.py:1650-1651` | NP (3,) f32 |
| Near / far | `camera.near`, `camera.far` | `renderer.py:1661`; `camera.py:75`, `97-98` | S (float) |
| FOV | `camera.fov` | `camera.py:89` | S (float) |
| View matrix | `camera.get_view_matrix()` | `renderer.py:1607`; `camera.py:33-61` | NP (4,4) f32 (computed) |
| View-projection | `camera.get_view_projection_matrix(aspect)` | `renderer.py:1604`; `camera.py:73-76` | NP (4,4) |
| Projection | `camera.get_projection_matrix(aspect, near, far)` | `renderer.py:1661-1663` | NP (4,4) |
| Camera target/up | `camera.target`, `camera.up` | `camera.py:41`, `44` (inside view matrix) | NP (3,) |
| Spherical cache | `camera._distance`, `_azimuth`, `_elevation` | `camera.py:26-31`, `152-164` | S |

`camera.set_pose` is `src/manifoldx/camera.py:137-141`; all field writes are in §2.

### 1.3 Lights, shadow, fog, environment, clock, viewport

| State | Path | Read at | Type |
|---|---|---|---|
| Point-light list | `engine._lights` | `renderer.py:1770-1776`; `mesh.py:136` (buffer) | PY (list) |
| Light fields | `light.color`, `.intensity`, `.position`, `.direction`, `.inner_angle`, `.outer_angle`, `.distance` | `resources.py:1895-1906` (Point), `1943-1956` (Spot), `1976-1987` (Directional), called from `renderer.py:1773` | S/str/tuple |
| Sun | `engine._sun` | `renderer.py:1383`, `1689`, `1704-1707`; `mesh.py:247` | PY |
| Sun fields | `sun.direction`, `sun.color`, `sun.intensity` | `renderer.py:1388`, `resources.py:1976-1987` | S/tuple |
| Spot | `engine._spot` | `renderer.py:1366`, `1690`, `1722-1747`; `mesh.py:248` | PY |
| Spot fields | `.position`, `.direction`, `.color`, `.inner_angle`, `.outer_angle`, `.intensity`, `.distance` | `renderer.py:1723-1746` | S/tuple |
| Shadow config | `engine._shadow_config` | `renderer.py:1367`, `1382`, `1487`, `1688-1717`; `shadow.py:111-117` | PY (dict) |
| Shadow config keys | `["target"]`, `["extent"]`, `["near"]`, `["far"]`, `["bias"]`, `["resolution"]`, `["pcf_radius"]`, `["auto_fit"]` | `renderer.py:1407-1410`, `1712-1717`; `shadow.py:117` | S/tuple |
| Fog | `engine.fog_enabled`, `.fog_start`, `.fog_end`, `.fog_color` | `renderer.py:1753-1758` | S/tuple |
| Benders | `engine.benders` | `renderer.py:1763` | PY (list of tuples) |
| Environment object | `engine._environment` via `_ready_environment()` | `engine.py:866-869`; `renderer.py:1679`, `1781`; `mesh.py:230`; `skybox.py:78` | PY |
| Environment fields | `env._computed`, `.intensity`, `.show_skybox` | `engine.py:868-869`; `renderer.py:1682`; `skybox.py:79` | S (bool/float) |
| Environment precomputed data | `env._irradiance`, `env._prefiltered` | `renderer.py:691-732` (`_upload_ibl_env`) | NP (float16) |
| Clock | `engine.elapsed` | `renderer.py:1670` | S (float) |
| Floating origin | `engine._origin` | `renderer.py:1675-1677` | S (tuple f64) |
| Viewport | `engine.w`, `engine.h` | `renderer.py:1603`, `1672`; `gui.py:299-300`, `373`, `395` | S (int) |
| Clear colour | `engine.background_color` | `engine.py:1365`, `1471` | S (tuple) |
| Texture format | `engine._texture_format` | `mesh.py:81`; `label.py:80`; `axis.py:59`; `volume.py:383`; `skybox.py:109`; `gui.py:409`, `589`, `810` | S (wgpu enum) |

### 1.4 Registries

| State | Path | Read at | Type |
|---|---|---|---|
| Geometry dicts | `engine._geometry_registry._geometries` via `.get(id)` | `renderer.py:1319`, `1417`; `mesh.py:60`; `sprite.py:73`; `label.py:64`; `axis.py:49`; `shadow.py:165` | PY (dict id→dict) |
| Geometry GPU buffers | `engine._geometry_registry._gpu_buffers` via `.get_gpu_buffers(id)` | `mesh.py:58`; `sprite.py:71`; `label.py:62`; `axis.py:47`; `shadow.py:163` | PY (dict id→dict of GPU) |
| Named geometry ids | `engine._geometry_registry._name_to_id` via `.get_id("sprite_quad")` | `sprite.py:70`; `label.py:61` | PY (dict str→int) |
| Geometry vertex/index count | `buffers["vertex_buffer"]`, `["index_buffer"]`, `["index_count"]`, `["stride"]`, `["has_joints"]`, `["joints_offset"]`, `["has_colors"]`, `["has_uvs"]` | `mesh.py:116`, `265-274`; `sprite.py:123-130`; `label.py:104-111`; `axis.py:101-108`; `shadow.py:173-185` | GPU objects + S |
| Materials | `engine._material_registry._materials` via `.get(id)` | `renderer.py:1447`, `1485`, `1545`, `1560`, `1793`; `mesh.py:70`; `sprite.py:85`; `label.py:73`; `axis.py:42`; `volume.py:407` | PY (dict id→material) |
| Material type / subtype | `type(mat_obj).__name__`, `mat_obj.pipeline_subtype` | `renderer.py:1561-1562`; `mesh.py:101`; `sprite.py:109`; `label.py:89`; `axis.py:68` | PY/S |
| Material pipeline state | `mat_obj.pipeline_state` | `renderer.py:1793`; `mesh.py:76` | S (tuple) |
| Material mirror | `mat_obj.mirror` | `renderer.py:1447-1450` | S (float/None) |
| Material casts_shadow | `mat_obj.casts_shadow` | `renderer.py:1485` | S (bool) |
| Material uniform data | `mat_obj.get_data(1, registry)` → material fields | `mesh.py:98`; `sprite.py:107`; `label.py:87`; `axis.py:66` | NP (16+ floats) |
| Material texture handles | `mat_obj.get_texture_bindings()` → `handle.sampler`, `handle.view` | `mesh.py:128`, `202-214`; `resources.py:1345-1348` | GPU |
| Volume resources | `engine._volume_registry._volumes` via `.get(vol_id)` | `volume.py:408` | PY (dict) |
| Volume data / dirty / texture | `res.data`, `res.dirty`, `res.texture` | `volume.py:409-410`; `resources.py:2055-2090` | NP / S / GPU |
| Volume material fields | `mat_obj.vmin`, `.vmax`, `.density_scale`, `.max_steps`, `.cmap`, `.opacity_lut` | `volume.py:365-370`, `412-413` | S / str / NP |
| Texture handles | `engine._texture_registry._handles` (indirectly, via handles held by materials/GUI) | not iterated by render; handles read through `mat_obj`/widgets | PY |

### 1.5 GUI

| State | Path | Read at | Type |
|---|---|---|---|
| GUI root | `engine.gui` (`_GuiRoot._panels`) | `gui.py:236`, `240`, `256`, `299`, `302`, `309` | PY (list) |
| GUI scale | `engine.gui.effective_scale(engine.h)` | `gui.py:299`; `bridge.py:96` | S |
| Exact colours flag | `engine.gui.exact_colors` | `gui.py:256`, `309` | S (bool) |
| Panels / children | `panel.children`, `panel.anchor`, `panel.offset` | `gui.py:302-306`; `painter.py:141-155` | PY |
| Widget text | `Text._text`, `Button.label`, `Slider.label`, `Toggle.label` | `painter.py:156`, `168`, `174`, `207`, `211`, `226`, `235`, `262`, `270` | S (str) |
| Widget values | `Toggle.value`, `Slider.value/min/max`, `Joystick.value` | `painter.py:250-306` | S/tuple |
| Widget style | `widget.effective_style()` → `gui.style._theme`, `_classes`, `style_overrides` | `painter.py:141-306`; `widgets.py:40-50`; `style.py:54-66` | PY (dict) |
| `ValueDisplay` text (user getter) | `ValueDisplay.refresh()` → `self._getter()` | `gui.py:225-231`, `240-241`; `value_display.py:40-47` | S (str); getter is arbitrary user code |
| Label atlas | `engine.get_label_atlas()` → `_label_atlas` | `gui.py:264`, `702`; `label.py:55` | PY |
| Atlas GPU state | `atlas._gpu_texture`, `_gpu_sampler`, `_dirty_slices`, `_slices` | `label.py:56-57`, `154-160`; `gui.py:266`; `viz/text.py:191-230` | GPU / NP / PY |
| SVG image data | `Image.svg`, `svg_data`, `svg_digest` | `painter.py:189-198` | S/NP |
| Texture handles in GUI images | `Image.texture` / `TextureHandle.view` | `gui.py:260-262`, `851-947` | GPU |

### 1.6 Readback / dev channel, views, render-pipeline internals

| State | Path | Read at | Type |
|---|---|---|---|
| Pending look futures | `engine._looks` | `engine.py:1387-1391`; `devchannel.py:248-251` | PY (list) |
| Pending readback | `engine._look_copy` | `engine.py:1326-1328` | PY (Readback) |
| Readback buffer/texture | `Readback.buffer`, `.looks`, `.w`, `.h`, `.row`, `.bgra` | `devchannel.py:528-561` | GPU / S |
| Second-camera views | `engine._views` (dict `id → _View`) | `engine.py:1459-1471` | PY (dict) |
| `_View` fields | `.camera`, `.handle`, `.depth_view`, `.live`, `.pending`, `.background`, `.fog` | `engine.py:1460-1488` | PY / GPU |
| Depth texture cache | `engine._depth_texture`, `_depth_texture_view`, `_depth_texture_size` | `engine.py:1337-1345` | GPU / tuple |
| Pipeline caches | `rp._pipelines`, `_pipeline_layouts`, `_bind_group_layouts`, `_material_buffers*`, `_mesh_bind_groups`, `_shadow_*`, `_lut_textures`, `_ibl_*`, `_draw_list`, `_routing_cache`, `_sprite_uploaded`, `_label_batch_buffers`, `_axis_batch_buffers`, `_globals_buffer`, `_lights_buffer`, `_joints` | `renderer.py:472-499`, `mesh.py:45-48`, `sprite.py:46-59`, `volume.py`, `gui.py`; written and read by the renderer only | PY / GPU |
| Shared buffers | `rp._globals_buffer`, `rp._lights_buffer`, `rp._batch_buffers*`, `rp._material_buffers_by_mat_id` | `mesh.py:131-137`, `216-219`; `sprite.py:186-199`; `label.py:130-150`; `axis.py:82-94`; `volume.py:401`; `shadow.py:141-149` | GPU |

Note: the render pipeline's own caches are written by the render thread only, but the
objects they point at (buffers/textures) are also written from user-triggered paths
(§4). That is the structural hazard: the renderer caches *identity*, not *content*.

---

## 2. SYSTEM-SIDE WRITERS

Everything below runs in user code: a system (worker thread under pipelining), a sync
event handler (main, step 3), an async handler / task callback (main, `_pump_aio_loop`
step 4 or `_drain_tasks` step 1), or a GUI pointer callback (main, between frames).

| # | Writer | file:line | What it mutates | GPU calls at call time? |
|---|---|---|---|---|
| 1 | `Engine.spawn` | `engine.py:1063-1113` (store write `1111`) | `store._alive` (`ecs.py:84`), `_high_water` (`86`), `_members` (`90-91`), `_free_list` (`83`), every component array row (`106`); runs `_release_hooks` (`87-88`) | **Yes, indirectly**: the joint-block release hook can `queue.write_buffer` (`skinning.py:117`); `Mesh.get_data`/`Material.get_data` mutate registries (`components.py:316-323`, `358-362`) but do not touch the device |
| 2 | `EntityHandle.transform.pos/rot/scale` setters | `engine.py:55-96` (`77-80`, `86-88`, `94-96`) | `store._components["Transform"]` row, in place | No |
| 3 | `EntityHandle.set_geometry` | `engine.py:120-169` | Geometry dict `orig["positions"]` (`146`), `rp._geom_local_bounds` (`147-149`), vertex buffer via `write_buffer` (`169`) | **Yes** — `queue.write_buffer` (169) |
| 4 | `EntityHandle.set_material` | `engine.py:171-178` | `_material_registry.register` (`177`), `store._components["Material"][i,0]` (`178`) | No (registry only) |
| 5 | `EntityHandle.set_joints` | `engine.py:180-201` → `skinning.py:78-87` | `JointStore.mirror`, `.block_of`; `queue.write_buffer` (`87`) | **Yes** — `write_buffer` (87) |
| 6 | `EntityHandle.destroy` | `engine.py:108-118` | `engine.destroy` (queues a DESTROY command, `116`); `_geometries_to_free.add` (`118`, Python set) | No (command) |
| 7 | `Engine.destroy` | `engine.py:1156-1167` | Command buffer only (DESTROY) | No |
| 8 | `Engine.set_joints` | `engine.py:1115-1144` → `skinning.py:89-100` | `JointStore.mirror`, `.block_of`; `queue.write_buffer` (`100`) | **Yes** — `write_buffer` (100) |
| 9 | `Engine.set_material` | `engine.py:1146-1154` | `_material_registry.register` (`1154`), `store._components["Material"][i,0]` (`1154`) | No |
| 10 | `Engine.set_origin` | `engine.py:742-769` | `engine._origin` (`753`); **every live Transform row** (`759-760`); `camera.position`/`target` (`761-763`); `_spot.position` (`764-765`); `_shadow_config["target"]` (`766-769`) | No |
| 11 | `Camera` mutators: `move_to` `122-125`, `move_by` `127-130`, `look_at` `132-135`, `set_pose` `137-141`, `orbit` `143-164`, `pan` `166-186`, `zoom` `192-204`, `dolly` `206-216`, `fit` `218-243`, `fit_bounds` `245-269`; direct field writes `camera.py:14-21` | `src/manifoldx/camera.py` | `position`, `target`, `up`, `fov`, `near`, `far`, `layers`, `_distance`, `_azimuth`, `_elevation` | No |
| 12 | `Engine.set_lights` / `add_light` | `engine.py:684-686`, `688-690` | `engine._lights` list (rebind / append) | No |
| 13 | Light field writes (`light.position = ...`, `.intensity`, `.direction`, …) | `resources.py:1879-1884`, `1912-1930`, `1962-1965` (plain attributes) | Python object fields | No |
| 14 | `engine.background_color = (...)` | plain attribute, default `engine.py:433`; read `1365`, `1471` | tuple | No |
| 15 | `Engine.enable_fog` / direct `fog_*` writes | `engine.py:782-787` (defaults `437-445`) | `fog_enabled`, `fog_start`, `fog_end`, `fog_color` | No |
| 16 | `Engine.set_sun` / `set_spot` / `enable_shadows` | `engine.py:789-796`, `798-803`, `805-833` | `_sun`, `_spot`, `_shadow_config` (whole dict rebuild) | No |
| 17 | `Engine.set_environment` | `engine.py:835-853` | `_environment`, `_env_task`; constructs a preset `EnvironmentMap` | No (precompute is on a background worker, `_precompute` is pure numpy, `ibl.py:101-133`) |
| 18 | Environment field writes | `ibl.py:36-37` (`intensity`, `show_skybox` are plain attributes) | Python scalars | No |
| 19 | `GeometryRegistry.register` | `resources.py:1563-1573` | `_object_to_id`, `_geometries`, `_next_id` | No |
| 20 | `GeometryRegistry.register_by_name` | `resources.py:1581-1583` | same + `_name_to_id` | No |
| 21 | `GeometryRegistry.create_buffers` | `resources.py:1506-1538` | `_gpu_buffers` dict; creates buffers + `write_buffer`/`write_texture` | **Yes** — `create_buffer` (1517, 1529), `write_buffer` (1521, 1533) |
| 22 | `GeometryRegistry.adopt_buffers` | `resources.py:1540-1544` | `_gpu_buffers` dict (used by the streamer) | No (buffers made by caller) |
| 23 | `GeometryRegistry.release` | `resources.py:1550-1561` | `_gpu_buffers`, `_geometries`, `_object_to_id`; `buf.destroy()` (1557) | **Yes** — `destroy` (1557) |
| 24 | `MaterialRegistry.register` | `resources.py:1617-1627` | `_materials`, `_object_to_id`, `_next_id` | No |
| 25 | `VolumeRegistry.register` / `update` | `resources.py:2018-2034`, `2036-2048` | `_volumes` dict; `res.data` rebind, `res.dirty = True` | No |
| 26 | Material field writes (`material.roughness = ...`, `.color`, `.opacity`, `.mirror`, `.cut`, `.sway`, `.albedo_map`, `.cmap`, `.vmin`, `.vmax`, `.density_scale`, `.opacity_lut`, atmosphere/cloud fields) | plain attributes in `resources.py:1305-1326`, `455-484`, `584-641`; `viz/materials.py:231-269`, `517-523` | Python fields / numpy LUTs | No |
| 27 | `entity.emit`-style direct array writes: `engine.store._components[...] = ...` | user code; canonical internal example `viz/shims.py:223-229` (`_sync` system) | Transform/ScalarValue/Radius rows | No |
| 28 | `ComponentAccessor` / `_FieldView` assignment | `ecs.py:474-503`, `209-259` | **Command buffer** (UPDATE_COMPONENT), not the store | No |
| 29 | `Engine.component(cls)` decorator | `engine.py:607-613` → `ecs.py:46-57` | `store._components`, `_members` (allocates a new numpy array) | No |
| 30 | `engine.state` | `engine.py:377` | arbitrary Python object | No |
| 31 | `QuadStreamer.focus` | `streaming.py:467-529` | `_ready`, `_pending`, `_drawn`, `_gpu`, `_selected`, `_still`, `_quiet`; calls `_upload`, `_draw`, `_undraw`, `_unload` | **Yes** — see #32-35 |
| 32 | `_Uploads._upload` | `streaming.py:140-186` | `GeometryRegistry._gpu_buffers` via `register`/`adopt_buffers` (149, 175); `create_buffer` (155, 159); `write_buffer` (170) | **Yes** |
| 33 | `_Uploads._draw` | `streaming.py:197-211` | `engine.spawn` (204) → store rows; `_drawn` dict | Indirectly (joint hook, as #1) |
| 34 | `_Uploads._undraw` | `streaming.py:213-216` | `EntityHandle.destroy` → DESTROY command; `_drawn` dict | No |
| 35 | `_Uploads._unload` | `streaming.py:188-195` | `_gpu` dict; `buf.destroy()` (194); `GeometryRegistry.release` (195) | **Yes** — `destroy` (194) |
| 36 | GUI widget mutation: `Text.text` setter `gui/widgets.py:138-142`; `Slider.value` `gui/slider.py:42,57,74`; `Toggle.value` `gui/toggle.py:27,36`; `Joystick.value` `gui/joystick.py:50,61`; `Button.label` `gui/button.py:26`; `Panel.children` `gui/widgets.py:77`; `_GuiRoot.append/remove/clear` `gui/widgets.py:243-249` | as cited | Python widget fields / panel list | No |
| 37 | GUI bridge (pointer/keyboard) | `gui/bridge.py:55-118` (`_layout_box` 80, `_captured` 90, dispatch 114-118) | widget fields + `_captured`, `_focused`, `gui.pointer_over_gui` | No |
| 38 | `gui.style.set_theme` / `define` | `gui/style.py:43-51` | module-global `_theme`, `_classes` dicts | No |
| 39 | `ValueDisplay` getter | user-supplied callable, invoked by render (`gui.py:225-231`) | arbitrary user state (read) | No |
| 40 | `Engine.render_to_texture` / `refresh_texture` / `set_texture_live` / `drop_texture` | `engine.py:1401-1457` | `engine._views` dict, `TextureRegistry`; creates textures/sampler (`1420`, `1426`, `1431`) | **Yes** |
| 41 | `textures.load_texture` / `texture_from_array` / `texture_from_svg` | `textures.py:68-165` (`create_texture` 116, `write_texture` 125, `create_sampler` 132) | `TextureRegistry._handles`, `_next_id` | **Yes** |
| 42 | `textures.write_texture` | `textures.py:168-187` (`write_texture` 182) | GPU texture level 0 | **Yes** |
| 43 | `Engine.register_volume` / `update_volume` | `engine.py:615-639` | `VolumeRegistry._volumes`; `res.data`, `res.dirty` | No (upload is render-side, `volume.py:409`) |
| 44 | `Engine.compute(cls)` registration | `engine.py:651-682` (`create_shader_module` 669) | `ComputeRunner._registered` | **Yes** — shader module at registration |
| 45 | `engine.get_label_atlas()` + `LabelTextureAtlas.get_or_create` | `engine.py:488-498`; `viz/text.py:110-181` | `_label_atlas`, `_slices`, `_index`, `_dirty_slices` | No at registration; upload is render-side (`viz/text.py:191-230`) |
| 46 | `engine.benders` append/replace | `engine.py:443`; user list | Python list | No |

---

## 3. COMMAND BUFFER (safe writes applied at step 6)

Applied at `engine.py:1292` (`self.commands.execute(self.store)`) then cleared (`1293`);
executor `commands.py:57-85`.

| Path | file:line | Command | Applied mutation |
|---|---|---|---|
| `Engine.destroy` / `EntityHandle.destroy` | `engine.py:1167`, `116` | `DESTROY` | `store.destroy` (`commands.py:69-71`; `ecs.py:110-127`) |
| `ComponentAccessor.__setattr__` | `ecs.py:474-503` | `UPDATE_COMPONENT` | whole row or slice (`commands.py:73-82`) |
| `_FieldView._set_data` (all `=`, `+=`, `-=`, `*=`, `/=` on a field view) | `ecs.py:209-259`, operators `261-281` | `UPDATE_COMPONENT` with `col_start`/`col_end` | columns of one component (`commands.py:79-80`) |
| `CommandType.SPAWN` | `commands.py:64-67` | declared but **no API path uses it** | n/a |

Important gaps (why the command buffer is not enough):
- `Engine.spawn` writes the store **immediately**, not via `commands` (`engine.py:1111`).
- `EntityHandle.transform.*` and `EntityHandle.set_material` / `Engine.set_material` write component rows **directly** (`engine.py:80,88,96,178,1154`).
- `Engine.set_joints` / `EntityHandle.set_joints` write the joint mirror and GPU buffer directly (`skinning.py:87,100`).
- `Engine.set_origin` rewrites every Transform row directly (`engine.py:760`).
- `viz/shims.py:225-229` writes Transform/ScalarValue/Radius directly in a system.
- Everything in §1.2–§1.6 (camera, lights, registries, GUI, environment, origin, w/h) never passes through the command buffer at all.

So the command buffer already makes *field-view updates and destroys* safe to enqueue from
a worker, provided `execute()` is only called after the worker joins.

---

## 4. GPU CALLS OUTSIDE THE RENDER

Calls to `device`/`queue` reachable from a system, handler, task callback, streamer or
the compute runner. (Render-internal calls are listed only where they are also reachable
from user code, because the renderer itself is expected to own the device on the main
thread.)

| # | Call | file:line | Reachable from |
|---|---|---|---|
| 1 | `device.create_buffer` / `queue.write_buffer` (vertex, index) | `resources.py:1517`, `1521`, `1529`, `1533` (`GeometryRegistry.create_buffers`) | render first-use (`mesh.py:63`), streamer (`streaming.py` via `adopt_buffers` no; `_upload` makes its own), user code |
| 2 | `buf.destroy()` | `resources.py:1557` (`GeometryRegistry.release`) | step 6 `_free_geometries` (`engine.py:1294`), `streaming.py:195` |
| 3 | `queue.write_buffer` (geometry in-place update) | `engine.py:169` (`EntityHandle.set_geometry`) | system/handler |
| 4 | `queue.write_buffer` (joint block, one) | `skinning.py:87` (`JointStore.set`) | `EntityHandle.set_joints` |
| 5 | `queue.write_buffer` (joint blocks, range) | `skinning.py:100` (`JointStore.set_many`) | `Engine.set_joints` |
| 6 | `queue.write_buffer` (joint block reset) | `skinning.py:117` (`JointStore.release`) | store release hook: any `spawn`/`destroy` (`ecs.py:87-88,124-125`; hook registered at `renderer.py:498`) |
| 7 | `device.create_buffer` + `queue.write_buffer` (joint buffer) | `skinning.py:126`, `130` (`ensure_buffer`) | render mesh/shadow pass; also reachable if a system calls it |
| 8 | `device.create_buffer` / `queue.write_buffer` (streamer upload) | `streaming.py:155`, `159`, `170` | `QuadStreamer.focus` / `Streamer.focus` |
| 9 | `buf.destroy()` (streamer unload) | `streaming.py:194` | `focus` / `close` |
| 10 | `device.create_texture` / `write_texture` / `create_sampler` | `textures.py:116`, `125`, `132` | `load_texture`, `texture_from_array`, `texture_from_svg` |
| 11 | `queue.write_texture` (patch) | `textures.py:182` | `write_texture` |
| 12 | `device.create_texture` ×2 / `create_sampler` | `engine.py:1420`, `1426`, `1431` | `Engine.render_to_texture` |
| 13 | `device.create_shader_module` | `engine.py:669` | `Engine.compute(cls)` registration |
| 14 | `create_bind_group_layout`, `create_pipeline_layout`, `create_shader_module`, `create_buffer`, `write_buffer`, `create_bind_group`, `create_command_encoder`, `queue.submit`, `queue.read_buffer` | `compute/_core.py:343-344`, `347`, `356`, `368`, `409`, `436`, `462`, `484`, `503`, `510` | `ComputeRunner.run_all` — **step 7** (`engine.py:1297`); step 7 also writes `engine._last_dt` (`412`) and store rows (`509-511`) |
| 15 | `device.create_texture` / `create_sampler` / `queue.write_texture` (label atlas) | `viz/text.py:203`, `209`, `218` | render label/GUI pass (`label.py:56`, `gui.py:266`); a system calling `atlas.upload_dirty` would hit it too |
| 16 | `device.create_buffer` / `create_command_encoder` / `queue.submit` (look readback) | `devchannel.py:535`, `537`, `541` | render (`engine.py:1391`), on a look request |
| 17 | Full trace pipeline: `create_buffer_with_data`, `create_buffer`, `create_shader_module`, `create_bind_group_layout`, `create_pipeline_layout`, `create_bind_group`, `write_buffer`, `create_command_encoder`, `queue.submit` | `trace/tracer.py:86`, `122`, `170`, `179`, `187`, `189`, `197`, `225`, `243`, `246`, `254`, `256`, `278`, `280`, `285`, `287`, `295`, `297`, `319`, `321` | `manifoldx.trace.Tracer` — user-invoked from a handler/system (**UNKNOWN** whether any production path calls it per frame) |
| 18 | `device.create_texture` / `write_texture` (IBL upload) | `renderer.py:694`, `702`, `717`, `727` | render only (`mesh.py:232`, `skybox.py:83`) |
| 19 | `device.create_texture` / `create_sampler` / `create_shader_module` / `create_render_pipeline` / `create_bind_group` | `renderer.py:512-570`, `590-687`, `747+`; `volume.py:292`, `317`, `388`, `396`, `418`; `gui.py:410-947`; `shadow.py:18-23`, `43-54`, `80-106`, `138-151`, `196-208` | render only, first use / cache miss |
| 20 | `device.create_texture` / `create_command_encoder` / `queue.submit` (depth, frame) | `engine.py:1339`, `1352`, `1385`; `1469-1489` | render only (`_draw_passes`, `_render_views`) |
| 21 | `queue.write_buffer` (globals, lights) | `renderer.py:1767`, `1776` | render only |

---

## 5. CLASSIFICATION

Legend: **(a)** safe as is while the render runs on another thread · **(b)** needs a copy
in the snapshot · **(c)** defer to the apply step on the main thread · **(d)** needs a lock.

### 5.1 §2 writers

| # | Item | Class | Reason |
|---|---|---|---|
| 1 | `Engine.spawn` | **c** (+b) | Writes `_alive`, `_high_water`, `_members`, component rows and fires GPU-writing release hooks; must run on the main thread at the apply step, and the new rows only become visible to the render through the next snapshot. |
| 2 | `EntityHandle.transform.*` | **c** | Direct in-place row writes race the transform gather and can tear a matrix between position, rotation and scale; route through a command or the apply step. |
| 3 | `EntityHandle.set_geometry` | **c** | `queue.write_buffer` (169) can land after the render's submit (half-updated mesh) and mutates the geometry dict the render reads; defer. |
| 4 | `EntityHandle.set_material` | **c** | Mutates the material registry and the `Material` id row the routing cache keys on; defer with the other registry writes. |
| 5 | `EntityHandle.set_joints` | **d** | `write_buffer` (87) into a buffer the in-flight frame binds; a lock (or a deferred queue) is needed, and the mirror is read by `_compute_scene_bounds` (`renderer.py:1346`). |
| 6 | `EntityHandle.destroy` | **c** | The command part is safe; `_geometries_to_free.add` (118) is a Python set mutated while step 6 iterates it, so do it at apply. |
| 7 | `Engine.destroy` | **a** | Only appends a DESTROY command; safe if commands are executed after the worker joins. |
| 8 | `Engine.set_joints` | **d** | Same buffer as #5, larger range; lock the joint mirror against the render's block reads. |
| 9 | `Engine.set_material` | **c** | Registry + `Material` row write; defer. |
| 10 | `Engine.set_origin` | **c** | Rewrites every live Transform row, the camera, the spot and the shadow target; shifting all of them under a running render is a guaranteed torn frame. |
| 11 | Camera mutators/fields | **b** | Read only through `_camera` at three call sites; a scalar snapshot (position, target, up, fov, near, far, layers) taken at the snapshot step is enough and cheap. |
| 12 | `Engine.set_lights` / `add_light` | **b** | Render reads `engine._lights` and each object's fields up to 4 times; copy the first 4 lights' `get_data()` (or the list) into the snapshot. |
| 13 | Light field writes | **b** | Same as #12; the snapshot must hold the packed 8-float rows, not the mutable objects. |
| 14 | `engine.background_color` | **b** | A single tuple read during pass encoding; copy it into the snapshot (or read it once before the worker starts). |
| 15 | `enable_fog` / `fog_*` | **b** | Four scalars read at `renderer.py:1753-1758`; snapshot them. |
| 16 | `set_sun` / `set_spot` / `enable_shadows` | **b** | The render packs sun/spot/shadow config into globals; copy the packed values (direction, colour, intensity, angles, config keys) into the snapshot. |
| 17 | `Engine.set_environment` | **b** | Render reads `_environment` identity, `intensity`, `show_skybox` and `_irradiance/_prefiltered`; snapshot the identity + two scalars, and keep the precomputed arrays alive until the frame is done. |
| 18 | Environment field writes | **b** | `intensity` and `show_skybox` are read directly; snapshot the scalars. |
| 19 | `GeometryRegistry.register` | **d** | `_next_id += 1` and three dict writes; a render `get()` during a resize is safe under the GIL, but the id counter is not atomic across threads. |
| 20 | `GeometryRegistry.register_by_name` | **d** | Same dicts plus `_name_to_id`. |
| 21 | `GeometryRegistry.create_buffers` | **c** | Device calls and `_gpu_buffers` insertion; first-use creation must happen on the render/apply thread, never concurrently with a pass binding the same id. |
| 22 | `GeometryRegistry.adopt_buffers` | **d** | Only a dict write, but it publishes a buffer the render may bind in the same frame; lock the registry or do it at apply. |
| 23 | `GeometryRegistry.release` | **c** | `buf.destroy()` (1557) while a submitted frame still references the buffer is invalid; defer and keep the buffer alive for the in-flight frame. |
| 24 | `MaterialRegistry.register` | **d** | `_next_id += 1` is not atomic; two threads registering at once can hand out the same id. |
| 25 | `VolumeRegistry.register` / `update` | **b** | `update` swaps `res.data` while `upload_to_gpu` reads it (shape, `tobytes`); the snapshot must hold the array reference (or a copy) and the dirty flag. |
| 26 | Material field writes | **b** | The render calls `get_data()`/`get_texture_bindings()` and reads `pipeline_state`/`mirror` every frame; snapshot the packed uniform rows and the state tuple, not the live objects. |
| 27 | Direct array writes (e.g. `viz/shims.py:225-229`) | **c** | Exactly the rows the snapshot is meant to protect; they must happen in the worker *before* the snapshot, or be routed through commands. |
| 28 | `ComponentAccessor`/`_FieldView` writes | **a** | Already deferred to `commands`, applied at step 6 after the worker joins. |
| 29 | `Engine.component` decorator | **c** | Allocates a new component array and inserts into `_components` mid-frame; registering components is setup work, not per-frame work. |
| 30 | `engine.state` | **a** | Not read by the render path. (But see #39: user getters can read it during render.) |
| 31 | `QuadStreamer.focus` | **c** | It spawns, creates and frees GPU buffers and rewrites the geometry registry; run it at the apply step (or make it enqueue work for that step). |
| 32 | `_Uploads._upload` | **c** | `create_buffer`/`write_buffer` (155, 159, 170) plus `_gpu_buffers` publication; defer. |
| 33 | `_Uploads._draw` | **c** | `engine.spawn` (204) → see #1. |
| 34 | `_Uploads._undraw` | **a** | Only queues DESTROY commands (`216` → `engine.py:116`). |
| 35 | `_Uploads._unload` | **c** | `destroy()` (194) + registry release (195) while a frame may still draw the cell; defer a frame. |
| 36 | GUI widget mutation | **d** | The render iterates `_panels` and reads widget fields on the main thread; a worker mutating them needs a lock (or all GUI writes move to the main thread). |
| 37 | GUI bridge | **a** | Runs on the main thread from pointer events, between frames; it does not overlap the render. |
| 38 | `gui.style.set_theme`/`define` | **d** | Module-global dicts read by every `resolve()` during layout; a write from a system would race. |
| 39 | `ValueDisplay` getter | **d** | The render calls arbitrary user code (`gui.py:225-231`); whatever it reads (often `engine.state`) must be stable while the worker runs. |
| 40 | `render_to_texture` family | **c** | Texture creation and `_views` mutation; setup for the next frame, not a per-frame mutation. |
| 41 | `load_texture` family | **c** | Device calls plus `TextureRegistry` writes; do it on the main thread (startup or apply step). |
| 42 | `write_texture` | **c** | `queue.write_texture` (182) may land after the render's submit; defer. |
| 43 | `register_volume`/`update_volume` | **b** | CPU-only, but the array the render uploads must be the snapshot's copy. |
| 44 | `Engine.compute(cls)` registration | **c** | `create_shader_module` (669) on the calling thread; registration belongs to setup, not a system. |
| 45 | `get_label_atlas` + `get_or_create` | **d** | Python dict/set mutation while the render iterates `_dirty_slices` and builds bind groups; lock or restrict to setup. |
| 46 | `engine.benders` | **b** | A list read at `renderer.py:1763`; snapshot the first `MAX_BENDERS` tuples. |

### 5.2 §4 GPU calls

| # | Call site | Class | Reason |
|---|---|---|---|
| 1 | `GeometryRegistry.create_buffers` | **c** | Buffer creation is a first-use render/apply action; never from a worker. |
| 2 | `GeometryRegistry.release` | **c** | `destroy()` invalidates a buffer a submitted frame may still use. |
| 3 | `EntityHandle.set_geometry` `write_buffer` | **c** | Ordering against the render's `queue.submit` is undefined from a worker. |
| 4 | `JointStore.set` | **d** | In-place range write into a storage buffer the render binds; lock the mirror. |
| 5 | `JointStore.set_many` | **d** | Same, wider range. |
| 6 | `JointStore.release` | **d** | Fired by `spawn`/`destroy` hooks, including ones on the worker; same buffer. |
| 7 | `JointStore.ensure_buffer` | **c** | Creates and zero-fills the whole joint buffer; must happen on the render thread before binding. |
| 8 | Streamer `create_buffer`/`write_buffer` | **c** | See §5.1 #32. |
| 9 | Streamer `destroy()` | **c** | See §5.1 #35. |
| 10 | `textures.load_texture` device calls | **c** | Setup/apply step only. |
| 11 | `write_texture` | **c** | Same ordering problem as #3. |
| 12 | `render_to_texture` device calls | **c** | Setup/apply step only. |
| 13 | `Engine.compute` `create_shader_module` | **c** | Registration-time work. |
| 14 | Compute `_dispatch` (step 7) | **a** | The spec runs it on the main thread after the systems join and before the snapshot; it writes store rows (`_core.py:509-511`) that the next snapshot must capture, and `_last_dt` (`412`). Safe only if step 7 truly happens after the worker is joined. |
| 15 | Label atlas `upload_dirty` | **a** (render) / **c** (from a system) | Render-side it is the only writer; a system calling it would touch the same texture. |
| 16 | `devchannel.Readback` | **a** | Created by the render on the main thread; the only cross-thread part is the PNG encode thread (`devchannel.py:561`), which reads a copied buffer. |
| 17 | `trace.Tracer` | **c** | A second renderer with its own device objects; must not run concurrently with the frame render. |
| 18-21 | Render-internal IBL / pipelines / depth / globals | **a** | Owned by the render thread; they are the render's own state and only become a problem through the objects they alias (§1.4/§1.6). |

### 5.3 Snapshot contents this implies (no code, just the list)

Minimum per-frame snapshot for step 8 to read a stable frame:
- ECS rows: `Transform` (pos/rot/scale), `Mesh` ids, `Material` ids, `Volume` ids, `Layer`,
  `PointCloud`/`TextLabel`/`AxisFrame` marker rows, `ScalarValue`, `Radius`, `_alive`,
  `_high_water`.
- Packed per-material uniform rows and the material state tuple
  (`type name`, `pipeline_subtype`, `pipeline_state`, `mirror`, `casts_shadow`,
  `get_data()` bytes, texture handles).
- Packed lights (`_lights[0:4].get_data()`), sun row, spot block, shadow config,
  fog block, benders, `elapsed`, `_origin`, `w/h`, `background_color`.
- Camera scalars (position, target, up, fov, near, far, layers).
- Environment identity + `intensity` + `show_skybox` (and a keep-alive on
  `_irradiance`/`_prefiltered`).
- Geometry ids → (vertex buffer, index buffer, index_count, stride/flags) and the
  local-bounds cache generation; a geometry rewritten after the snapshot must not be
  bound by the in-flight frame.
- GUI: a deep-enough copy of the panel tree (or a lock around GUI mutation) plus the
  label atlas slice set.
- Views: a copy of `_views` (camera + handle + live/pending/background/fog).

---

## 6. UNKNOWNS

- **wgpu-py / wgpu-native thread-safety guarantees** for concurrent `device` calls from a
  worker and a render thread: not asserted anywhere in the repo; treat as UNKNOWN and
  assume no.
- Whether `trace.Tracer` is ever invoked per frame by a system in this repo or downstream
  (only the public API exists; no call site found in `src/`).
- Whether downstream apps (havana-driver, mosaico) mutate `engine._lights[i]` fields or
  `engine.gui` widgets from systems rather than handlers: UNKNOWN; the audit lists the
  API, not its users.
- `engine._id_materials` (`engine.py:469`) and the humanoid/glTF helpers were not traced
  for per-frame writes; they appear to be setup-only.
- The web target (`src/manifoldx/web/`) has its own replay/snapshot path and is out of
  scope for the threaded engine; it mutates `engine.w/h` at `web/replay.py:116` but is not
  reachable from a desktop system.

---

## 7. Highest-risk summary

1. `engine.spawn` writes the store immediately (`engine.py:1111`) and can fire a GPU
   `write_buffer` through the joint release hook (`skinning.py:117`).
2. Direct Transform writes from handles (`engine.py:80/88/96`) and `set_origin`
   (`engine.py:760`) bypass the command buffer and can tear matrices.
3. `EntityHandle.set_geometry` (`engine.py:169`) and all joint uploads
   (`skinning.py:87/100`) write GPU buffers the in-flight frame binds.
4. `QuadStreamer.focus` creates, writes and destroys GPU buffers and spawns entities from
   whatever thread calls it (`streaming.py:155-204`, `194`).
5. `GeometryRegistry.release`/`destroy` (`resources.py:1557`) while a submitted frame still
   references the buffer.
6. Material/lights/camera/fog/environment are read live, not snapshotted
   (`renderer.py:1573-1776`), so every scalar write in §2 is a torn-frame candidate.
7. `ValueDisplay` getters run arbitrary user code inside the render pass
   (`gui.py:225-241`).
8. GUI panels/widgets are iterated live (`gui.py:240-302`) with no lock; `_GuiRoot.append`
   from a system can resize the list mid-iteration.
9. `MaterialRegistry.register` / `GeometryRegistry.register` bump `_next_id` without a
   lock (`resources.py:1570`, `1624`).
10. `Engine.destroy` is safe (command), but `_free_geometries` runs at step 6
    (`engine.py:1294`) and frees buffers the previous frame may still be using.
