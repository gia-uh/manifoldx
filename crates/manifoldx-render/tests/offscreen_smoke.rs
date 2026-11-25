//! A real offscreen render when the host exposes a Vulkan adapter. CI hosts
//! without one skip instead of failing the unit suite; the standalone binary
//! is the hardware evidence path.

use manifoldx_core::ecs::{ColumnData, ElementType, World};
use manifoldx_render::camera::Camera;
use manifoldx_render::renderer::{NativeRenderer, RendererOptions};
use manifoldx_render::scene::{resolve_source, spawn_instances, MeshData};

fn quad() -> MeshData {
    MeshData {
        positions: vec![
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ],
        normals: vec![[0.0, 0.0, 1.0]; 4],
        indices: vec![0, 1, 2, 0, 2, 3],
    }
}

#[test]
fn offscreen_render_reads_world_and_changes_with_a_transform() {
    let mut renderer = match NativeRenderer::new(RendererOptions {
        allow_software: true,
        ..Default::default()
    }) {
        Ok(renderer) => renderer,
        Err(error) => {
            eprintln!("skip offscreen test: {error}");
            return;
        }
    };
    renderer.upload_mesh(1, &quad(), [0.2, 0.7, 0.3]).unwrap();
    renderer.register_material(1, [0.8, 0.2, 0.2]);

    let mut world = World::new(4).unwrap();
    let transform = world.register("Transform", ElementType::F32, 10).unwrap();
    world.register("Mesh", ElementType::U32, 1).unwrap();
    world.register("Material", ElementType::U32, 1).unwrap();
    let ids = spawn_instances(
        &mut world,
        &[
            [-0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.5],
            [0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.5],
        ],
        &[1, 1],
        &[1, 1],
    )
    .unwrap();
    let source = resolve_source(&world, "Transform", "Mesh", Some("Material"), 8).unwrap();
    let camera = Camera::new([0.0, 0.0, 3.0], [0.0; 3]);

    let first = renderer
        .render(&world, source, &camera, 96, 72, false)
        .unwrap();
    assert_eq!((first.width, first.height), (96, 72));
    assert_eq!(first.rgba.len(), 96 * 72 * 4);
    let lit = first
        .rgba
        .as_chunks::<4>()
        .0
        .iter()
        .filter(|pixel| pixel[0] > 10 || pixel[1] > 10 || pixel[2] > 10)
        .count();
    assert!(lit > 100, "only {lit} lit pixels");

    let rows = [
        [0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.5],
        [0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.5],
    ]
    .concat();
    world
        .update_rows(transform, &ids, ColumnData::F32(rows))
        .unwrap();
    let second = renderer
        .render(&world, source, &camera, 96, 72, false)
        .unwrap();
    let changed = first
        .rgba
        .as_chunks::<4>()
        .0
        .iter()
        .zip(second.rgba.as_chunks::<4>().0)
        .filter(|(a, b)| a != b)
        .count();
    assert!(changed > 50, "only {changed} pixels changed");
}
