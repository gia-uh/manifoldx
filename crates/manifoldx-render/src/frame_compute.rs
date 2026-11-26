//! Native execution for declared or hand-written Compute WGSL.

use std::collections::HashMap;

use manifoldx_core::ecs::{ColumnData, ComponentReadout, EntityId, World};

use crate::manual::gpu_buffer;
use crate::{NativeRenderer, RenderError};

pub struct ComputeBinding {
    pub column: Option<usize>,
    pub name: String,
    pub width: u32,
    pub write: bool,
}

pub struct ComputeProgram {
    pub key: u32,
    pub source: String,
    pub workgroup_size: u32,
    pub uniforms: Vec<(u8, f32)>,
    pub threads: (u8, u32),
    pub bindings: Vec<ComputeBinding>,
}

struct Storage {
    descriptor: ComputeBinding,
    buffer: wgpu::Buffer,
    readback: Option<wgpu::Buffer>,
}

struct GpuColumn {
    width: u32,
    buffer: wgpu::Buffer,
}

struct Timestamp {
    queries: wgpu::QuerySet,
    resolved: wgpu::Buffer,
    readback: wgpu::Buffer,
    available: bool,
}

struct Pipeline {
    program: ComputeProgram,
    pipeline: wgpu::ComputePipeline,
    group: wgpu::BindGroup,
    uniforms: Option<wgpu::Buffer>,
    columns: Vec<Storage>,
    timing: Option<Timestamp>,
}

#[derive(Default)]
pub struct FrameCompute {
    pipelines: Vec<Pipeline>,
    resident: HashMap<String, GpuColumn>,
    live: Vec<Option<EntityId>>,
    frame: u64,
}

impl FrameCompute {
    pub fn register(
        &mut self,
        renderer: &NativeRenderer,
        world: &World,
        program: ComputeProgram,
    ) -> Result<(), RenderError> {
        if self
            .pipelines
            .iter()
            .any(|pipe| pipe.program.key == program.key)
        {
            return Ok(());
        }
        let device = renderer.device();
        let limits = device.limits();
        if program.workgroup_size == 0
            || program.workgroup_size > limits.max_compute_workgroup_size_x
            || program.workgroup_size > limits.max_compute_invocations_per_workgroup
            || program.threads.0 > 3
            || program
                .uniforms
                .iter()
                .any(|&(tag, value)| tag > 3 || !value.is_finite())
        {
            return Err(problem("invalid workgroup or uniform declaration"));
        }
        let mut columns = Vec::new();
        let mut entries = Vec::new();
        let uniform_bytes = (program.uniforms.len() as u64 * 4).next_multiple_of(16);
        let uniforms = (uniform_bytes > 0).then(|| {
            entries.push(layout_buffer(0, wgpu::BufferBindingType::Uniform));
            gpu_buffer(
                device,
                "native Compute uniforms",
                uniform_bytes,
                wgpu::BufferUsages::UNIFORM,
            )
        });
        for (i, descriptor) in program.bindings.iter().enumerate() {
            let size = u64::from(world.capacity()) * u64::from(descriptor.width) * 4;
            if size == 0 || size > limits.max_storage_buffer_binding_size {
                return Err(problem(format!(
                    "{} exceeds storage limits",
                    descriptor.name
                )));
            }
            if let Some(column) = descriptor.column {
                if world.schema(column).is_none_or(|schema| {
                    schema.name != descriptor.name || schema.width != descriptor.width
                }) {
                    return Err(problem(format!(
                        "invalid declared column {}",
                        descriptor.name
                    )));
                }
            }
            let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
            let buffer = if descriptor.column.is_some() {
                gpu_buffer(device, "native Compute mirrored column", size, usage)
            } else {
                let resident = self
                    .resident
                    .entry(descriptor.name.clone())
                    .or_insert_with(|| GpuColumn {
                        width: descriptor.width,
                        buffer: gpu_buffer(device, "native Compute resident column", size, usage),
                    });
                if resident.width != descriptor.width {
                    return Err(problem("conflicting resident column widths"));
                }
                resident.buffer.clone()
            };
            let readback = (descriptor.column.is_some() && descriptor.write).then(|| {
                gpu_buffer(
                    device,
                    "native Compute mirrored writeback",
                    size,
                    wgpu::BufferUsages::MAP_READ,
                )
            });
            entries.push(layout_buffer(
                i as u32 + u32::from(uniforms.is_some()),
                wgpu::BufferBindingType::Storage {
                    read_only: !descriptor.write,
                },
            ));
            columns.push((buffer, readback));
        }

        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("native Compute bindings"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("native declared Compute layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("native authored Compute WGSL"),
            source: wgpu::ShaderSource::Wgsl(program.source.as_str().into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("native declared Compute"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bindings: Vec<_> = uniforms
            .iter()
            .chain(columns.iter().map(|(buffer, _)| buffer))
            .enumerate()
            .map(|(i, buffer)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("native declared Compute buffers"),
            layout: &layout,
            entries: &bindings,
        });
        if let Some(error) = pollster::block_on(scope.pop()) {
            return Err(problem(error));
        }
        let columns = program
            .bindings
            .iter()
            .zip(columns)
            .map(|(descriptor, (buffer, readback))| Storage {
                descriptor: ComputeBinding {
                    column: descriptor.column,
                    name: descriptor.name.clone(),
                    width: descriptor.width,
                    write: descriptor.write,
                },
                buffer,
                readback,
            })
            .collect();
        let timing = Timestamp::new(device);
        self.pipelines.push(Pipeline {
            program,
            pipeline,
            group,
            uniforms,
            columns,
            timing,
        });
        Ok(())
    }

    pub fn run(
        &mut self,
        renderer: &NativeRenderer,
        world: &mut World,
        dt: f32,
        overrides: &HashMap<u32, u32>,
    ) -> Result<(), RenderError> {
        let entities = world.snapshot().entities;
        let span = world.high_water() as usize;
        let device = renderer.device();
        let queue = renderer.queue();
        if !self.resident.is_empty() {
            let mut current = vec![None; span];
            for &id in &entities {
                current[id.slot() as usize] = Some(id);
            }
            for (i, slot) in current.iter().enumerate() {
                if self.live.get(i).is_some_and(|old| old != slot) {
                    for column in self.resident.values() {
                        let zero = vec![0u32; column.width as usize];
                        queue.write_buffer(
                            &column.buffer,
                            i as u64 * u64::from(column.width) * 4,
                            bytemuck::cast_slice(&zero),
                        );
                    }
                }
            }
            self.live = current;
        }
        for pipeline in &mut self.pipelines {
            let (dispatch, value) = pipeline.program.threads;
            let threads = match dispatch {
                0 => value,
                1 => world.alive_count(),
                2 => world.capacity(),
                _ => *overrides
                    .get(&pipeline.program.key)
                    .ok_or_else(|| problem("missing callable dispatch size"))?,
            };
            let groups = threads.div_ceil(pipeline.program.workgroup_size);
            if groups == 0 {
                if let Some(timing) = &mut pipeline.timing {
                    timing.available = false;
                }
                continue;
            }
            if groups > device.limits().max_compute_workgroups_per_dimension {
                return Err(problem("dispatch exceeds the device workgroup limit"));
            }
            let inputs: Vec<_> = pipeline
                .columns
                .iter()
                .map(|binding| {
                    binding
                        .descriptor
                        .column
                        .map(|column| world.read_component(column, Some(&entities)))
                        .transpose()
                        .map_err(problem)
                })
                .collect::<Result<_, _>>()?;

            for (column, input) in pipeline.columns.iter().zip(&inputs) {
                if let Some(input) = input.as_ref().filter(|_| span > 0) {
                    let words = mirrored_input(input, &entities, span);
                    queue.write_buffer(&column.buffer, 0, bytemuck::cast_slice(&words));
                }
            }
            if let Some(buffer) = &pipeline.uniforms {
                let values: Vec<f32> = pipeline
                    .program
                    .uniforms
                    .iter()
                    .map(|&(tag, value)| match tag {
                        0 => value,
                        1 => dt,
                        2 => world.alive_count() as f32,
                        _ => self.frame as f32,
                    })
                    .collect();
                queue.write_buffer(buffer, 0, bytemuck::cast_slice(&values));
            }
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("native Compute dispatch"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("native Compute dispatch"),
                    timestamp_writes: pipeline.timing.as_ref().map(|timing| {
                        wgpu::ComputePassTimestampWrites {
                            query_set: &timing.queries,
                            beginning_of_pass_write_index: Some(0),
                            end_of_pass_write_index: Some(1),
                        }
                    }),
                });
                pass.set_pipeline(&pipeline.pipeline);
                pass.set_bind_group(0, &pipeline.group, &[]);
                pass.dispatch_workgroups(groups, 1, 1);
            }
            if let Some(timing) = &mut pipeline.timing {
                encoder.resolve_query_set(&timing.queries, 0..2, &timing.resolved, 0);
                encoder.copy_buffer_to_buffer(&timing.resolved, 0, &timing.readback, 0, 16);
                timing.available = true;
            }
            if span > 0 {
                for column in &pipeline.columns {
                    if let Some(readback) = &column.readback {
                        encoder.copy_buffer_to_buffer(
                            &column.buffer,
                            0,
                            readback,
                            0,
                            span as u64 * u64::from(column.descriptor.width) * 4,
                        );
                    }
                }
            }
            queue.submit([encoder.finish()]);
            for (column, input) in pipeline.columns.iter().zip(&inputs) {
                if let (Some(readback), Some(input)) =
                    (&column.readback, input.as_ref().filter(|_| span > 0))
                {
                    let width = input.schema.width as usize;
                    let bytes = read_buffer(device, readback, span as u64 * width as u64 * 4)?;
                    let words: Vec<u32> = bytes
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .copied()
                        .map(u32::from_ne_bytes)
                        .collect();
                    let ids: Vec<_> = entities
                        .iter()
                        .copied()
                        .zip(&input.membership)
                        .filter_map(|(id, member)| member.then_some(id))
                        .collect();
                    let updates = ids.iter().flat_map(|id| {
                        let offset = id.slot() as usize * width;
                        words[offset..offset + width].iter().copied()
                    });
                    let data = match input.data {
                        ColumnData::F32(_) => {
                            ColumnData::F32(updates.map(f32::from_bits).collect())
                        }
                        ColumnData::U32(_) => ColumnData::U32(updates.collect()),
                    };
                    world
                        .update_rows(
                            column.descriptor.column.expect("mirrored column"),
                            &ids,
                            data,
                        )
                        .map_err(problem)?;
                }
            }
        }
        self.frame += 1;
        Ok(())
    }

    pub fn timings(&self, renderer: &NativeRenderer) -> Result<Vec<Option<u64>>, RenderError> {
        let period = f64::from(renderer.queue().get_timestamp_period());
        self.pipelines
            .iter()
            .map(|pipeline| {
                let Some(timing) = pipeline.timing.as_ref().filter(|timing| timing.available)
                else {
                    return Ok(None);
                };
                let bytes = read_buffer(renderer.device(), &timing.readback, 16)?;
                let [start, end] =
                    bytemuck::try_pod_read_unaligned::<[u64; 2]>(&bytes).map_err(problem)?;
                Ok(Some((end.saturating_sub(start) as f64 * period) as u64))
            })
            .collect()
    }
}

impl Timestamp {
    fn new(device: &wgpu::Device) -> Option<Self> {
        device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
            .then(|| Self {
                queries: device.create_query_set(&wgpu::QuerySetDescriptor {
                    label: Some("native Compute pass timestamps"),
                    ty: wgpu::QueryType::Timestamp,
                    count: 2,
                }),
                resolved: gpu_buffer(
                    device,
                    "native Compute resolved timestamps",
                    16,
                    wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                ),
                readback: gpu_buffer(
                    device,
                    "native Compute diagnostic timestamps",
                    16,
                    wgpu::BufferUsages::MAP_READ,
                ),
                available: false,
            })
    }
}

fn mirrored_input(readout: &ComponentReadout, entities: &[EntityId], span: usize) -> Vec<u32> {
    let width = readout.schema.width as usize;
    let source: &[u32] = match &readout.data {
        ColumnData::F32(values) => bytemuck::cast_slice(values),
        ColumnData::U32(values) => values,
    };
    let mut target = vec![0; span * width];
    for (i, id) in entities
        .iter()
        .enumerate()
        .filter(|(i, _)| readout.membership[*i])
    {
        let offset = id.slot() as usize * width;
        target[offset..offset + width].copy_from_slice(&source[i * width..(i + 1) * width]);
    }
    target
}

fn read_buffer(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
    size: u64,
) -> Result<Vec<u8>, RenderError> {
    let (send, receive) = std::sync::mpsc::channel();
    buffer.map_async(wgpu::MapMode::Read, 0..size, move |result| {
        let _ = send.send(result);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(problem)?;
    receive.recv().map_err(problem)?.map_err(problem)?;
    let bytes = buffer.get_mapped_range(0..size).to_vec();
    buffer.unmap();
    Ok(bytes)
}

fn layout_buffer(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        count: None,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
    }
}

fn problem(reason: impl std::fmt::Display) -> RenderError {
    RenderError::Resource(format!("native Compute: {reason}"))
}
