//! The renderer must read exactly the live World rows through the public
//! owned-read API, and a native row update must change the instance matrix.

use manifoldx_core::ecs::{ColumnData, ComponentPayload, ElementType, EntityId, World};
use manifoldx_render::math::Mat4;
use manifoldx_render::scene::{
    read_instances, resolve_source, spawn_instances, MeshData, SceneError,
};

fn world_with_rows() -> (World, Vec<EntityId>, usize) {
    let mut world = World::new(8).unwrap();
    let transform = world.register("Transform", ElementType::F32, 10).unwrap();
    world.register("Mesh", ElementType::U32, 1).unwrap();
    world.register("Material", ElementType::U32, 1).unwrap();
    let ids = spawn_instances(
        &mut world,
        &[
            [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            [4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0],
        ],
        &[7, 9],
        &[3, 4],
    )
    .unwrap();
    (world, ids, transform)
}

#[test]
fn resolves_the_production_schemas() {
    let (world, _, transform) = world_with_rows();
    let source = resolve_source(&world, "Transform", "Mesh", Some("Material"), 32).unwrap();
    assert_eq!(source.transform, transform);
    assert_eq!(source.max_instances, 32);
}

#[test]
fn reads_only_member_rows_and_keeps_request_order() {
    let (mut world, ids, _) = world_with_rows();
    // An entity without Mesh must not produce an instance.
    let transform = world.component_index("Transform").unwrap();
    world
        .spawn(
            1,
            &[ComponentPayload {
                component: transform,
                data: ColumnData::F32(vec![9.0; 10]),
            }],
        )
        .unwrap();
    let source = resolve_source(&world, "Transform", "Mesh", Some("Material"), 32).unwrap();
    let batch = read_instances(&world, source, false).unwrap();
    assert_eq!(batch.entities_examined, 3);
    assert_eq!(batch.instances.len(), 2);
    assert_eq!(batch.instances[0].geometry_id, 7);
    assert_eq!(batch.instances[0].material_id, Some(3));
    assert_eq!(batch.instances[1].geometry_id, 9);
    assert_eq!(
        batch.instances[0].model.transform_point([0.0; 3])[..3],
        [1.0, 2.0, 3.0]
    );
    assert_eq!(ids.len(), 2);
}

#[test]
fn a_native_row_update_changes_the_instance_matrix() {
    let (mut world, ids, transform) = world_with_rows();
    let source = resolve_source(&world, "Transform", "Mesh", Some("Material"), 32).unwrap();
    let before = read_instances(&world, source, false).unwrap();
    let before_position = before.instances[0].model.transform_point([0.0; 3]);

    let mut rows = Vec::new();
    for entity in [ids[0], ids[1]] {
        let read = world.read_component(transform, Some(&[entity])).unwrap();
        match read.data {
            ColumnData::F32(values) => rows.extend_from_slice(&values),
            _ => panic!("wrong column kind"),
        }
    }
    rows[0] = 11.0;
    world
        .update_rows(transform, &ids, ColumnData::F32(rows))
        .unwrap();

    let after = read_instances(&world, source, false).unwrap();
    let after_position = after.instances[0].model.transform_point([0.0; 3]);
    assert_eq!(before_position[0], 1.0);
    assert_eq!(after_position[0], 11.0);
    assert_ne!(before.instances[0].model, after.instances[0].model);
}

#[test]
fn ignore_transform_forces_identity_models() {
    let (world, _, _) = world_with_rows();
    let source = resolve_source(&world, "Transform", "Mesh", Some("Material"), 32).unwrap();
    let batch = read_instances(&world, source, true).unwrap();
    assert_eq!(batch.instances[0].model, Mat4::IDENTITY);
    assert_eq!(batch.instances[1].model, Mat4::IDENTITY);
}

#[test]
fn the_instance_bound_rejects_the_whole_read() {
    let (world, _, _) = world_with_rows();
    let source = resolve_source(&world, "Transform", "Mesh", Some("Material"), 1).unwrap();
    match read_instances(&world, source, false) {
        Err(SceneError::TooManyInstances { requested, max }) => {
            assert_eq!((requested, max), (2, 1));
        }
        other => panic!("expected a bound error, got {other:?}"),
    }
}

#[test]
fn wrong_schema_is_rejected_before_a_read() {
    let mut world = World::new(4).unwrap();
    world.register("Transform", ElementType::F32, 9).unwrap();
    world.register("Mesh", ElementType::U32, 1).unwrap();
    match resolve_source(&world, "Transform", "Mesh", None, 4) {
        Err(SceneError::Schema { name, .. }) => assert_eq!(name, "Transform"),
        other => panic!("expected a schema error, got {other:?}"),
    }
}

#[test]
fn mesh_validation_rejects_bad_indices_and_mismatched_normals() {
    let mut mesh = MeshData {
        positions: vec![[0.0; 3]; 3],
        normals: vec![[0.0, 0.0, 1.0]; 3],
        indices: vec![0, 1, 2],
    };
    assert_eq!(mesh.validate(), Ok(3));
    mesh.indices = vec![0, 1, 3];
    assert!(matches!(
        mesh.validate(),
        Err(SceneError::IndexOutOfRange { .. })
    ));
    mesh.indices = vec![0, 1, 2];
    mesh.normals.pop();
    assert!(matches!(
        mesh.validate(),
        Err(SceneError::NormalCount { .. })
    ));
}
