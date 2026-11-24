//! Mesh input and the bounded read boundary from the existing native World.
//!
//! Geometry is prepared outside the renderer (Python/NumPy or a model loader)
//! and handed over once as plain slices. Per frame, the renderer reads only
//! the selected Transform, Mesh and optional Material rows through the World's
//! documented owned-read API. There is no pointer into Rust storage and no
//! Python readback.

use std::collections::HashSet;

use manifoldx_core::ecs::{ColumnData, ComponentPayload, ElementType, EntityId, World, WorldError};

use crate::math::{model_from_transform_row, Mat4};

/// Default hard cap on instances one read may produce.
pub const DEFAULT_MAX_INSTANCES: usize = 65_536;

/// CPU-side indexed geometry handed to the renderer.
#[derive(Clone, Debug, PartialEq)]
pub struct MeshData {
    /// Vertex positions.
    pub positions: Vec<[f32; 3]>,
    /// Vertex normals, parallel to `positions`.
    pub normals: Vec<[f32; 3]>,
    /// Triangle indices, three per triangle, counter-clockwise outside.
    pub indices: Vec<u32>,
}

impl MeshData {
    /// Check the data fits the indexed pipeline and return the vertex count.
    pub fn validate(&self) -> Result<usize, SceneError> {
        if self.positions.is_empty() {
            return Err(SceneError::EmptyMesh);
        }
        if self.positions.len() != self.normals.len() {
            return Err(SceneError::NormalCount {
                positions: self.positions.len(),
                normals: self.normals.len(),
            });
        }
        if self.indices.is_empty() || !self.indices.len().is_multiple_of(3) {
            return Err(SceneError::IndexCount(self.indices.len()));
        }
        if let Some(&bad) = self
            .indices
            .iter()
            .find(|&&index| index as usize >= self.positions.len())
        {
            return Err(SceneError::IndexOutOfRange {
                index: bad,
                vertices: self.positions.len(),
            });
        }
        Ok(self.positions.len())
    }

    /// Interleave position and normal into the vertex buffer layout.
    pub fn interleaved(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.positions.len() * 6);
        for (position, normal) in self.positions.iter().zip(&self.normals) {
            out.extend_from_slice(position);
            out.extend_from_slice(normal);
        }
        out
    }
}

/// Which World columns feed the instance stream.
#[derive(Clone, Copy, Debug)]
pub struct InstanceSource {
    /// Index of the `Transform` column (`f32`, width 10).
    pub transform: usize,
    /// Index of the `Mesh` column (`u32`, width 1).
    pub mesh: usize,
    /// Optional `Material` column (`u32`, width 1).
    pub material: Option<usize>,
    /// Maximum instances this read may return.
    pub max_instances: usize,
}

/// One instance extracted from the World, before GPU upload.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Instance {
    /// Registered geometry id from the Mesh row.
    pub geometry_id: u32,
    /// Optional material id from the Material row.
    pub material_id: Option<u32>,
    /// Local-to-world matrix.
    pub model: Mat4,
}

/// An owned instance list plus the number of live entities examined.
#[derive(Clone, Debug, PartialEq)]
pub struct InstanceBatch {
    /// Instances in ascending entity-slot order, as the World read returns.
    pub instances: Vec<Instance>,
    /// Live entities before membership or geometry filtering.
    pub entities_examined: usize,
}

/// Read and validate the columns that feed the instance stream once.
///
/// The schema check happens before any read, so a wrong column type or width
/// is a clear error instead of a reinterpreted buffer.
pub fn resolve_source(
    world: &World,
    transform: &str,
    mesh: &str,
    material: Option<&str>,
    max_instances: usize,
) -> Result<InstanceSource, SceneError> {
    if max_instances == 0 {
        return Err(SceneError::BadLimit);
    }
    let transform = world
        .component_index(transform)
        .ok_or_else(|| SceneError::MissingComponent(transform.to_string()))?;
    let mesh = world
        .component_index(mesh)
        .ok_or_else(|| SceneError::MissingComponent(mesh.to_string()))?;
    let material = material
        .map(|name| {
            world
                .component_index(name)
                .ok_or_else(|| SceneError::MissingComponent(name.to_string()))
        })
        .transpose()?;

    expect_schema(world, transform, ElementType::F32, 10, "Transform")?;
    expect_schema(world, mesh, ElementType::U32, 1, "Mesh")?;
    if let Some(material) = material {
        expect_schema(world, material, ElementType::U32, 1, "Material")?;
    }
    Ok(InstanceSource {
        transform,
        mesh,
        material,
        max_instances,
    })
}

fn expect_schema(
    world: &World,
    component: usize,
    element: ElementType,
    width: u32,
    name: &str,
) -> Result<(), SceneError> {
    let schema = world
        .schema(component)
        .ok_or_else(|| SceneError::MissingComponent(name.to_string()))?;
    if schema.element != element || schema.width != width {
        return Err(SceneError::Schema {
            name: name.to_string(),
            expected: format!("{element:?} width {width}"),
            actual: format!("{:?} width {}", schema.element, schema.width),
        });
    }
    Ok(())
}

/// Read the selected World rows and build owned instances.
///
/// `ignore_transform` exists only for the negative control: it proves that a
/// changed World transform is what changes the image, because forcing identity
/// makes the two scene states identical.
pub fn read_instances(
    world: &World,
    source: InstanceSource,
    ignore_transform: bool,
) -> Result<InstanceBatch, SceneError> {
    // Reject before snapshotting: a scene above the bound must not allocate an
    // entity list or any component rows.
    let alive = world.alive_count() as usize;
    if alive > source.max_instances {
        return Err(SceneError::TooManyInstances {
            requested: alive,
            max: source.max_instances,
        });
    }
    let snapshot = world.snapshot();
    debug_assert_eq!(snapshot.entities.len(), alive);
    let entities = &snapshot.entities;

    let transforms = world.read_component(source.transform, Some(entities))?;
    let meshes = world.read_component(source.mesh, Some(entities))?;
    let materials = source
        .material
        .map(|component| world.read_component(component, Some(entities)))
        .transpose()?;

    let transforms = f32_rows(&transforms)?;
    let meshes = u32_rows(&meshes)?;
    let materials = materials.as_ref().map(u32_rows).transpose()?;

    let mut instances = Vec::with_capacity(entities.len());
    for index in 0..entities.len() {
        if !transforms.membership[index] || !meshes.membership[index] {
            continue;
        }
        let geometry_id = meshes.data[index];
        if geometry_id == 0 {
            // The existing Mesh component uses 0 for "no geometry".
            continue;
        }
        let transform_row = &transforms.data[index * 10..(index + 1) * 10];
        let model = if ignore_transform {
            Mat4::IDENTITY
        } else {
            model_from_transform_row(transform_row)
        };
        let material_id = match &materials {
            Some(readout) if readout.membership[index] => Some(readout.data[index]),
            _ => None,
        };
        instances.push(Instance {
            geometry_id,
            material_id,
            model,
        });
    }

    Ok(InstanceBatch {
        instances,
        entities_examined: entities.len(),
    })
}

struct F32Rows<'a> {
    data: &'a [f32],
    membership: &'a [bool],
}
struct U32Rows<'a> {
    data: &'a [u32],
    membership: &'a [bool],
}

fn f32_rows(readout: &manifoldx_core::ecs::ComponentReadout) -> Result<F32Rows<'_>, SceneError> {
    match &readout.data {
        ColumnData::F32(data) => Ok(F32Rows {
            data,
            membership: &readout.membership,
        }),
        other => Err(SceneError::Schema {
            name: readout.schema.name.clone(),
            expected: "f32 rows".into(),
            actual: format!("{:?} rows", other.element_type()),
        }),
    }
}

fn u32_rows(readout: &manifoldx_core::ecs::ComponentReadout) -> Result<U32Rows<'_>, SceneError> {
    match &readout.data {
        ColumnData::U32(data) => Ok(U32Rows {
            data,
            membership: &readout.membership,
        }),
        other => Err(SceneError::Schema {
            name: readout.schema.name.clone(),
            expected: "u32 rows".into(),
            actual: format!("{:?} rows", other.element_type()),
        }),
    }
}

/// Build the World the sample binary and tests use: the production schemas
/// (`Transform` 10×f32, `Mesh` 1×u32, `Material` 1×u32) and one spawn batch.
pub fn spawn_instances(
    world: &mut World,
    transforms: &[[f32; 10]],
    geometry_ids: &[u32],
    material_ids: &[u32],
) -> Result<Vec<EntityId>, SceneError> {
    if transforms.len() != geometry_ids.len() || transforms.len() != material_ids.len() {
        return Err(SceneError::BatchLength);
    }
    let transform = world
        .component_index("Transform")
        .ok_or_else(|| SceneError::MissingComponent("Transform".into()))?;
    let mesh = world
        .component_index("Mesh")
        .ok_or_else(|| SceneError::MissingComponent("Mesh".into()))?;
    let material = world
        .component_index("Material")
        .ok_or_else(|| SceneError::MissingComponent("Material".into()))?;
    let count = transforms.len() as u32;
    let mut transform_values = Vec::with_capacity(transforms.len() * 10);
    for row in transforms {
        transform_values.extend_from_slice(row);
    }
    let mut mesh_values = Vec::with_capacity(geometry_ids.len());
    mesh_values.extend_from_slice(geometry_ids);
    let mut material_values = Vec::with_capacity(material_ids.len());
    material_values.extend_from_slice(material_ids);
    let ids = world.spawn(
        count,
        &[
            ComponentPayload {
                component: transform,
                data: ColumnData::F32(transform_values),
            },
            ComponentPayload {
                component: mesh,
                data: ColumnData::U32(mesh_values),
            },
            ComponentPayload {
                component: material,
                data: ColumnData::U32(material_values),
            },
        ],
    )?;
    Ok(ids)
}

/// Select registered geometry ids referenced by a batch, preserving first use.
pub fn used_geometry_ids(batch: &InstanceBatch) -> Vec<u32> {
    let mut seen = HashSet::new();
    let mut ids = Vec::new();
    for instance in &batch.instances {
        if seen.insert(instance.geometry_id) {
            ids.push(instance.geometry_id);
        }
    }
    ids
}

/// A renderer-side failure while preparing a scene.
#[derive(Debug)]
pub enum SceneError {
    /// The World rejected a read or spawn.
    World(WorldError),
    /// A required component is not registered.
    MissingComponent(String),
    /// A registered component does not match the expected schema.
    Schema {
        /// Component name.
        name: String,
        /// Expected layout.
        expected: String,
        /// Actual layout.
        actual: String,
    },
    /// The instance limit was zero or the live set exceeded it.
    BadLimit,
    /// The live set exceeded the configured bound.
    TooManyInstances {
        /// Entities requested.
        requested: usize,
        /// Configured maximum.
        max: usize,
    },
    /// Mesh vertex positions were empty.
    EmptyMesh,
    /// Normals were not parallel to positions.
    NormalCount {
        /// Position count.
        positions: usize,
        /// Normal count.
        normals: usize,
    },
    /// Indices were empty or not a triangle list.
    IndexCount(usize),
    /// An index pointed past the vertex array.
    IndexOutOfRange {
        /// The offending index.
        index: u32,
        /// Vertex count.
        vertices: usize,
    },
    /// Spawn arrays had different lengths.
    BatchLength,
}

impl std::fmt::Display for SceneError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::World(error) => write!(f, "world: {error}"),
            Self::MissingComponent(name) => write!(f, "missing component {name}"),
            Self::Schema {
                name,
                expected,
                actual,
            } => write!(f, "component {name}: expected {expected}, got {actual}"),
            Self::BadLimit => write!(f, "instance limit must be positive"),
            Self::TooManyInstances { requested, max } => {
                write!(
                    f,
                    "{requested} live entities exceed the {max} instance bound"
                )
            }
            Self::EmptyMesh => write!(f, "mesh has no vertices"),
            Self::NormalCount { positions, normals } => {
                write!(f, "{positions} positions but {normals} normals")
            }
            Self::IndexCount(count) => write!(f, "{count} indices are not a triangle list"),
            Self::IndexOutOfRange { index, vertices } => {
                write!(f, "index {index} is outside {vertices} vertices")
            }
            Self::BatchLength => write!(f, "spawn arrays have different lengths"),
        }
    }
}

impl std::error::Error for SceneError {}

impl From<WorldError> for SceneError {
    fn from(error: WorldError) -> Self {
        Self::World(error)
    }
}
