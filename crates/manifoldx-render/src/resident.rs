//! One versioned float32 World column on the desktop renderer's device.

use std::collections::BTreeMap;

use manifoldx_core::ecs::{ColumnData, ComponentReadout, ElementType, EntityId, Scalar, World};
use manifoldx_core::runtime::{apply, Command};

use crate::renderer::ResidentPositions;
use crate::{NativeRenderer, RenderError};

const WORKGROUP: u32 = 64;
const TRANSFER_LIMIT: u64 = 16 * 1024 * 1024;

pub struct PositionSpec {
    pub column: usize,
    pub width: u32,
    pub offset: u32,
    pub kernel_version: u32,
    pub writes: Vec<(u32, u32)>,
    pub wgsl: String,
}

#[derive(Default)]
pub struct ResidencyStats {
    pub row_uploads: u64,
    pub row_upload_bytes: u64,
    pub field_uploads: u64,
    pub field_upload_bytes: u64,
    pub mask_uploads: u64,
    pub mask_upload_bytes: u64,
    pub uniform_uploads: u64,
    pub dispatches: u64,
    pub version: u64,
    pub cpu_version: u64,
    pub generations_seeded: u64,
    pub generations_retired: u64,
}

pub struct ResidentKernel {
    spec: PositionSpec,
    device: wgpu::Device,
    queue: wgpu::Queue,
    rows: wgpu::Buffer,
    mask: wgpu::Buffer,
    uniforms: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    bindings: wgpu::BindGroup,
    members: Vec<Option<EntityId>>,
    entity_version: Option<u64>,
    field_patches: BTreeMap<(u32, u32), (EntityId, f32)>,
    span: u32,
    pub stats: ResidencyStats,
}

impl ResidentKernel {
    pub fn new(
        renderer: &NativeRenderer,
        world: &World,
        spec: PositionSpec,
    ) -> Result<Self, RenderError> {
        let schema = world
            .schema(spec.column)
            .ok_or_else(|| invalid("column is unregistered"))?;
        let size = u64::from(world.capacity()) * u64::from(spec.width) * 4;
        let device = renderer.device();
        let limits = device.limits();
        if schema.element != ElementType::F32
            || schema.width != spec.width
            || spec
                .offset
                .checked_add(3)
                .is_none_or(|end| end > spec.width)
            || !spec.writes.contains(&(spec.offset, 3))
            || spec.writes.iter().any(|(offset, width)| {
                !matches!(width, 1 | 3)
                    || offset
                        .checked_add(*width)
                        .is_none_or(|end| end > spec.width)
            })
            || spec.kernel_version != 1
        {
            return Err(invalid(
                "expected a v1 kernel writing the bound vec3<f32> field",
            ));
        }
        if size
            > TRANSFER_LIMIT
                .min(limits.max_buffer_size)
                .min(u64::from(limits.max_storage_buffer_binding_size))
            || world.capacity().div_ceil(WORKGROUP) > limits.max_compute_workgroups_per_dimension
            || limits.max_compute_workgroup_size_x < WORKGROUP
            || limits.max_compute_invocations_per_workgroup < WORKGROUP
        {
            return Err(invalid(
                "graphics device cannot dispatch the resident column",
            ));
        }
        let make = |label, size, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: usage | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let rows = make(
            "resident component rows",
            size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let mask = make(
            "resident component membership",
            u64::from(world.capacity()) * 4,
            wgpu::BufferUsages::STORAGE,
        );
        let uniforms = make("resident kernel arguments", 16, wgpu::BufferUsages::UNIFORM);
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("resident Python kernel"),
            source: wgpu::ShaderSource::Wgsl(spec.wgsl.as_str().into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("resident Python kernel"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        if let Some(error) = pollster::block_on(scope.pop()) {
            return Err(invalid(error));
        }
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("resident Python kernel"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: rows.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: mask.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniforms.as_entire_binding(),
                },
            ],
        });
        Ok(Self {
            spec,
            device: device.clone(),
            queue: renderer.queue().clone(),
            rows,
            mask,
            uniforms,
            pipeline,
            bindings,
            members: vec![None; world.capacity() as usize],
            entity_version: None,
            field_patches: BTreeMap::new(),
            span: 0,
            stats: ResidencyStats::default(),
        })
    }

    pub fn apply(
        &mut self,
        world: &mut World,
        command: Command,
    ) -> Result<Option<ComponentReadout>, String> {
        match command {
            Command::Update { component, .. } | Command::Read { component, .. }
                if component == self.spec.column =>
            {
                Err("whole-row updates and CPU reads of a resident column are unavailable".into())
            }
            Command::Set {
                selection,
                component,
                field,
                value,
            } => {
                if selection
                    .predicates
                    .iter()
                    .any(|item| item.0 == self.spec.column)
                {
                    return Err("CPU predicates cannot read a resident column".into());
                }
                if component != self.spec.column {
                    return apply(
                        world,
                        Command::Set {
                            selection,
                            component,
                            field,
                            value,
                        },
                    );
                }
                let Scalar::F32(value) = value else {
                    return Err("resident field requires f32".into());
                };
                if self.spec.writes.iter().any(|&(offset, width)| {
                    (offset as usize..(offset + width) as usize).contains(&field)
                }) {
                    return Err("CPU cannot write a field owned by the resident kernel".into());
                }
                let ids = world.select(&selection)?;
                world.set_selected(&selection, component, field, Scalar::F32(value))?;
                for id in ids {
                    self.field_patches
                        .insert((id.slot(), field as u32), (id, value));
                }
                Ok(None)
            }
            command => apply(world, command),
        }
    }

    fn synchronize(&mut self, world: &World) -> Result<(), RenderError> {
        let entity_version = world.entity_version();
        let cpu_version = world.column_version(self.spec.column).map_err(invalid)?;
        if self.entity_version != Some(entity_version) || self.stats.cpu_version != cpu_version {
            let alive = world.snapshot().entities;
            let membership = world
                .memberships(self.spec.column, &alive)
                .map_err(invalid)?;
            let mut members = vec![None; self.members.len()];
            for (id, present) in alive.into_iter().zip(membership) {
                if present {
                    members[id.slot() as usize] = Some(id);
                }
            }
            let mut seeded = Vec::new();
            let mut mask_changed = self.entity_version.is_none();
            for (old, current) in self.members.iter().zip(&members) {
                mask_changed |= old.is_some() != current.is_some();
                if old != current {
                    self.stats.generations_retired += u64::from(old.is_some());
                    if let Some(id) = current {
                        seeded.push(*id);
                    }
                }
            }
            if !seeded.is_empty() {
                let readout = world
                    .read_component(self.spec.column, Some(&seeded))
                    .map_err(invalid)?;
                let ColumnData::F32(data) = readout.data else {
                    return Err(invalid("resident schema changed"));
                };
                let width = self.spec.width as usize;
                let mut start = 0;
                while start < seeded.len() {
                    let mut end = start + 1;
                    while end < seeded.len() && seeded[end].slot() == seeded[end - 1].slot() + 1 {
                        end += 1;
                    }
                    let bytes = bytemuck::cast_slice(&data[start * width..end * width]);
                    self.queue.write_buffer(
                        &self.rows,
                        u64::from(seeded[start].slot()) * u64::from(self.spec.width) * 4,
                        bytes,
                    );
                    self.stats.row_uploads += 1;
                    self.stats.row_upload_bytes += bytes.len() as u64;
                    start = end;
                }
                self.stats.generations_seeded += seeded.len() as u64;
            }
            for ((slot, field), (id, value)) in std::mem::take(&mut self.field_patches) {
                let index = slot as usize;
                if self.members[index] == Some(id) && members[index] == Some(id) {
                    let offset =
                        (u64::from(slot) * u64::from(self.spec.width) + u64::from(field)) * 4;
                    self.queue
                        .write_buffer(&self.rows, offset, &value.to_le_bytes());
                    self.stats.field_uploads += 1;
                    self.stats.field_upload_bytes += 4;
                }
            }
            if mask_changed {
                let mask: Vec<_> = members
                    .iter()
                    .map(|member| u32::from(member.is_some()))
                    .collect();
                self.queue
                    .write_buffer(&self.mask, 0, bytemuck::cast_slice(&mask));
                self.stats.mask_uploads += 1;
                self.stats.mask_upload_bytes += mask.len() as u64 * 4;
            }
            self.span = members
                .iter()
                .rposition(Option::is_some)
                .map_or(0, |index| index as u32 + 1);
            self.members = members;
            self.entity_version = Some(entity_version);
            self.stats.cpu_version = cpu_version;
        }
        Ok(())
    }

    pub fn dispatch(
        &mut self,
        world: &World,
        encoder: &mut wgpu::CommandEncoder,
        dt: f32,
        skip_dispatch: bool,
    ) -> Result<(), RenderError> {
        self.synchronize(world)?;
        if self.span == 0 || skip_dispatch {
            return Ok(());
        }
        self.queue.write_buffer(
            &self.uniforms,
            0,
            bytemuck::cast_slice(&[self.spec.kernel_version, self.span, dt.to_bits(), 0]),
        );
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bindings, &[]);
        pass.dispatch_workgroups(self.span.div_ceil(WORKGROUP), 1, 1);
        self.stats.uniform_uploads += 1;
        self.stats.dispatches += 1;
        self.stats.version += 1;
        Ok(())
    }

    pub fn positions(&self) -> ResidentPositions<'_> {
        ResidentPositions {
            buffer: &self.rows,
            members: &self.members,
            row_width: self.spec.width,
            offset: self.spec.offset,
        }
    }

    pub fn diagnostic_readback(&self) -> Result<(u64, Vec<ResidentDiagnosticRow>), RenderError> {
        if self.span == 0 {
            return Ok((0, Vec::new()));
        }
        let bytes = u64::from(self.span) * u64::from(self.spec.width) * 4;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("resident rows diagnostic"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&self.rows, 0, &buffer, 0, bytes);
        self.queue.submit([encoder.finish()]);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer.map_async(wgpu::MapMode::Read, .., move |result| {
            let _ = sender.send(result);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(invalid)?;
        receiver.recv().map_err(invalid)?.map_err(invalid)?;
        let mapped = buffer.get_mapped_range(..);
        let data: &[f32] = bytemuck::cast_slice(&mapped);
        let rows = self
            .members
            .iter()
            .flatten()
            .map(|id| {
                let start = id.slot() as usize * self.spec.width as usize;
                (
                    id.slot(),
                    id.generation(),
                    data[start..start + self.spec.width as usize].to_vec(),
                )
            })
            .collect();
        drop(mapped);
        buffer.unmap();
        Ok((bytes, rows))
    }
}

pub type ResidentDiagnosticRow = (u32, u64, Vec<f32>);

fn invalid(reason: impl std::fmt::Display) -> RenderError {
    RenderError::Window(format!("resident position binding: {reason}"))
}
