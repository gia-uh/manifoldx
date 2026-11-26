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
}

struct GpuColumn {
    component: Option<usize>,
    width: u32,
    slots: wgpu::Buffer,
    readback: Option<wgpu::Buffer>,
}

struct Mirror {
    input: ComponentReadout,
    written: Vec<bool>,
}

#[derive(Clone, Copy)]
struct Transfer {
    slot: u32,
    invocation: u32,
    count: u32,
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
    columns: HashMap<String, GpuColumn>,
    live: Vec<Option<EntityId>>,
    frame: u64,
    transfers: [usize; 3],
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
            let column = self
                .columns
                .entry(descriptor.name.clone())
                .or_insert_with(|| GpuColumn {
                    component: descriptor.column,
                    width: descriptor.width,
                    slots: gpu_buffer(
                        device,
                        "native Compute physical slots",
                        size,
                        wgpu::BufferUsages::COPY_SRC,
                    ),
                    readback: None,
                });
            if column.component != descriptor.column || column.width != descriptor.width {
                return Err(problem("conflicting declarations of a Compute column"));
            }
            if descriptor.column.is_some() && descriptor.write && column.readback.is_none() {
                column.readback = Some(gpu_buffer(
                    device,
                    "native Compute mirrored writeback",
                    size,
                    wgpu::BufferUsages::MAP_READ,
                ));
            }
            let buffer = gpu_buffer(
                device,
                "native Compute invocation rows",
                size,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            );
            entries.push(layout_buffer(
                i as u32 + u32::from(uniforms.is_some()),
                wgpu::BufferBindingType::Storage {
                    read_only: !descriptor.write,
                },
            ));
            columns.push(buffer);
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
            .chain(&columns)
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
            .map(|(descriptor, buffer)| Storage {
                descriptor: ComputeBinding {
                    column: descriptor.column,
                    name: descriptor.name.clone(),
                    width: descriptor.width,
                    write: descriptor.write,
                },
                buffer,
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
        let dispatches: Vec<_> = self
            .pipelines
            .iter()
            .map(|pipeline| {
                let (mode, value) = pipeline.program.threads;
                let threads = match mode {
                    0 => value,
                    1 => world.alive_count(),
                    2 => world.capacity(),
                    _ => *overrides
                        .get(&pipeline.program.key)
                        .ok_or_else(|| problem("missing callable dispatch size"))?,
                };
                let groups = threads.div_ceil(pipeline.program.workgroup_size);
                if groups > device.limits().max_compute_workgroups_per_dimension {
                    return Err(problem("dispatch exceeds the device workgroup limit"));
                }
                Ok((mode == 1, threads, groups))
            })
            .collect::<Result<_, _>>()?;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("native Compute frame"),
        });
        let mut current = vec![None; span];
        for &id in &entities {
            current[id.slot() as usize] = Some(id);
        }
        for column in self.columns.values() {
            if column.component.is_none() {
                let stride = u64::from(column.width) * 4;
                for (slot, id) in current.iter().enumerate() {
                    if self.live.get(slot).is_some_and(|previous| previous != id) {
                        encoder.clear_buffer(&column.slots, slot as u64 * stride, Some(stride));
                    }
                }
            }
        }
        let mut mirrors = HashMap::new();
        for (pipeline, &(_, _, groups)) in self.pipelines.iter().zip(&dispatches) {
            if groups == 0 {
                continue;
            }
            for storage in &pipeline.columns {
                let name = &storage.descriptor.name;
                let Some(component) = storage.descriptor.column else {
                    continue;
                };
                if mirrors.contains_key(name) {
                    continue;
                }
                let input = world
                    .read_component(component, Some(&entities))
                    .map_err(problem)?;
                if span > 0 {
                    let words = mirrored_input(&input, &entities, span);
                    queue.write_buffer(&self.columns[name].slots, 0, bytemuck::cast_slice(&words));
                }
                mirrors.insert(
                    name.clone(),
                    Mirror {
                        input,
                        written: vec![false; entities.len()],
                    },
                );
            }
        }

        let passes = dispatches
            .iter()
            .filter(|(_, _, groups)| *groups > 0)
            .count();
        for (pipeline, (dense, threads, groups)) in self.pipelines.iter_mut().zip(dispatches) {
            if groups == 0 {
                if let Some(timing) = &mut pipeline.timing {
                    timing.available = false;
                }
                continue;
            }

            // All bindings use the same invocation domain. Prepare a fresh view of
            // the physical slots for this pass so holes and padding read as zero.
            for storage in &pipeline.columns {
                let name = &storage.descriptor.name;
                let column = &self.columns[name];
                let membership = mirrors
                    .get(name)
                    .map(|mirror| mirror.input.membership.as_slice());
                encoder.clear_buffer(&storage.buffer, 0, None);
                let runs = transfers(&entities, membership, dense, u32::MAX);
                copy_rows(
                    &mut encoder,
                    &column.slots,
                    &storage.buffer,
                    column.width,
                    &runs,
                    true,
                );
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
            // Copy back only live, member rows inside the declared thread count.
            // Later passes see the result through the same buffers, across modes.
            for storage in pipeline.columns.iter().filter(|s| s.descriptor.write) {
                let name = &storage.descriptor.name;
                let column = &self.columns[name];
                let mirror = mirrors.get_mut(name);
                let membership = mirror.as_ref().map(|m| m.input.membership.as_slice());
                let runs = transfers(&entities, membership, dense, threads);
                copy_rows(
                    &mut encoder,
                    &storage.buffer,
                    &column.slots,
                    column.width,
                    &runs,
                    false,
                );
                if let Some(mirror) = mirror {
                    for (row, id) in entities.iter().enumerate() {
                        if invocation(row, *id, dense) >= threads {
                            break;
                        }
                        if mirror.input.membership[row] {
                            mirror.written[row] = true;
                        }
                    }
                }
            }
        }

        let written: Vec<_> = mirrors
            .iter()
            .filter(|(_, mirror)| mirror.written.contains(&true))
            .collect();
        for (name, _) in &written {
            let column = &self.columns[*name];
            encoder.copy_buffer_to_buffer(
                &column.slots,
                0,
                column.readback.as_ref().expect("writable mirrored column"),
                0,
                span as u64 * u64::from(column.width) * 4,
            );
        }
        let transfers = [
            passes,
            if span > 0 { mirrors.len() } else { 0 },
            written.len(),
        ];
        queue.submit([encoder.finish()]);
        self.live = current;
        for (name, mirror) in written {
            let column = &self.columns[name];
            let width = column.width as usize;
            let bytes = read_buffer(
                device,
                column.readback.as_ref().expect("writable mirrored column"),
                span as u64 * width as u64 * 4,
            )?;
            let words: Vec<_> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .copied()
                .map(u32::from_ne_bytes)
                .collect();
            let ids: Vec<_> = entities
                .iter()
                .copied()
                .zip(&mirror.written)
                .filter_map(|(id, written)| written.then_some(id))
                .collect();
            let updates = ids.iter().flat_map(|id| {
                let offset = id.slot() as usize * width;
                words[offset..offset + width].iter().copied()
            });
            let data = match mirror.input.data {
                ColumnData::F32(_) => ColumnData::F32(updates.map(f32::from_bits).collect()),
                ColumnData::U32(_) => ColumnData::U32(updates.collect()),
            };
            world
                .update_rows(column.component.expect("mirrored column"), &ids, data)
                .map_err(problem)?;
        }
        self.transfers = transfers;
        self.frame += 1;
        Ok(())
    }

    /// Dispatched programs, mirrored uploads and final mirrored publications.
    pub fn transfers(&self) -> [usize; 3] {
        self.transfers
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

fn invocation(row: usize, id: EntityId, dense: bool) -> u32 {
    if dense {
        row as u32
    } else {
        id.slot()
    }
}

fn transfers(
    entities: &[EntityId],
    membership: Option<&[bool]>,
    dense: bool,
    threads: u32,
) -> Vec<Transfer> {
    let mut runs: Vec<Transfer> = Vec::new();
    for (row, &id) in entities.iter().enumerate() {
        let invocation = invocation(row, id, dense);
        if invocation >= threads {
            break;
        }
        if membership.is_some_and(|flags| !flags[row]) {
            continue;
        }
        let slot = id.slot();
        if let Some(last) = runs
            .last_mut()
            .filter(|run| run.slot + run.count == slot && run.invocation + run.count == invocation)
        {
            last.count += 1;
        } else {
            runs.push(Transfer {
                slot,
                invocation,
                count: 1,
            });
        }
    }
    runs
}

fn copy_rows(
    encoder: &mut wgpu::CommandEncoder,
    source: &wgpu::Buffer,
    target: &wgpu::Buffer,
    width: u32,
    runs: &[Transfer],
    gather: bool,
) {
    let stride = u64::from(width) * 4;
    for run in runs {
        let slot = u64::from(run.slot) * stride;
        let invocation = u64::from(run.invocation) * stride;
        let (source_offset, target_offset) = if gather {
            (slot, invocation)
        } else {
            (invocation, slot)
        };
        encoder.copy_buffer_to_buffer(
            source,
            source_offset,
            target,
            target_offset,
            u64::from(run.count) * stride,
        );
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
