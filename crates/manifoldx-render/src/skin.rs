//! Packed per-vertex joint selection and generational pose blocks.

use std::collections::{BTreeSet, HashMap};

use manifoldx_core::ecs::EntityId;

use crate::math::Mat4;
use crate::renderer::{Gpu, RenderError};

pub const MAX_JOINTS: usize = 32;
const FRAME_MASK: u32 = 0x000f_ffff;
const BLOCK_BYTES: u64 = (MAX_JOINTS * 16 * size_of::<f32>()) as u64;

pub type JointMatrices = Vec<[f32; 16]>;
type Block = [[f32; 16]; MAX_JOINTS];
const IDENTITY_BLOCK: Block = [Mat4::IDENTITY.0; MAX_JOINTS];

struct Pose {
    identity: EntityId,
    own_block: u32,
    frame: u32,
}

pub(crate) struct Skin {
    pub(crate) layout: wgpu::BindGroupLayout,
    pub(crate) group: wgpu::BindGroup,
    buffer: wgpu::Buffer,
    blocks: Vec<Block>,
    dirty: BTreeSet<usize>,
    poses: HashMap<u32, Pose>,
    clips: HashMap<u32, (u32, usize)>,
    max_blocks: usize,
}

impl Skin {
    pub(crate) fn new(gpu: &Gpu) -> Self {
        let device = gpu.device();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("native joint blocks"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(BLOCK_BYTES),
                },
                count: None,
            }],
        });
        let (buffer, group) = storage(device, &layout, 1);
        let limits = device.limits();
        let max_blocks = (limits
            .max_storage_buffer_binding_size
            .min(limits.max_buffer_size)
            / BLOCK_BYTES)
            .min(u64::from(FRAME_MASK) + 1) as usize;
        gpu.queue()
            .write_buffer(&buffer, 0, bytemuck::cast_slice(&IDENTITY_BLOCK));
        Self {
            layout,
            group,
            buffer,
            blocks: vec![IDENTITY_BLOCK],
            dirty: BTreeSet::new(),
            poses: HashMap::new(),
            clips: HashMap::new(),
            max_blocks,
        }
    }

    fn entry(&mut self, identity: EntityId) -> &mut Pose {
        let pose = self.poses.entry(identity.slot()).or_insert(Pose {
            identity,
            own_block: 0,
            frame: 0,
        });
        if pose.identity != identity {
            if pose.own_block != 0 {
                self.blocks[pose.own_block as usize] = IDENTITY_BLOCK;
                self.dirty.insert(pose.own_block as usize);
            }
            pose.identity = identity;
            pose.frame = 0;
        }
        pose
    }

    pub(crate) fn set(
        &mut self,
        identity: EntityId,
        matrices: &[[f32; 16]],
    ) -> Result<(), RenderError> {
        validate_matrices(matrices)?;
        let own_block = self.entry(identity).own_block;
        let index = if own_block == 0 {
            let index = self.allocate(1)?;
            self.poses
                .get_mut(&identity.slot())
                .expect("joint pose entry")
                .own_block = index;
            index as usize
        } else {
            own_block as usize
        };
        self.poses
            .get_mut(&identity.slot())
            .expect("joint pose entry")
            .frame = 0;
        self.write(index, matrices);
        Ok(())
    }

    pub(crate) fn bake(&mut self, clip: u32, frames: &[JointMatrices]) -> Result<(), RenderError> {
        if frames.is_empty() {
            return Err(invalid("a clip must have at least one frame"));
        }
        for frame in frames {
            validate_matrices(frame)?;
        }
        let start = if let Some((start, count)) = self.clips.get(&clip) {
            if *count != frames.len() {
                return Err(invalid("a replacement clip must keep its frame count"));
            }
            *start
        } else {
            let start = self.allocate(frames.len())?;
            self.clips.insert(clip, (start, frames.len()));
            start
        };
        for (i, matrices) in frames.iter().enumerate() {
            self.write(start as usize + i, matrices);
        }
        Ok(())
    }

    pub(crate) fn play(
        &mut self,
        identity: EntityId,
        clip: u32,
        position: f64,
    ) -> Result<(), RenderError> {
        let &(first, count) = self
            .clips
            .get(&clip)
            .ok_or_else(|| invalid("unknown clip"))?;
        let last = count - 1;
        if !(0.0..=last as f64).contains(&position) {
            return Err(invalid("position is outside the clip's frames"));
        }
        let whole = (position.floor() as usize).min(last.saturating_sub(1));
        let fraction = ((position - whole as f64) * 4095.0).round_ties_even() as u32;
        self.entry(identity).frame = (first + whole as u32) | (fraction << 20);
        Ok(())
    }

    pub(crate) fn frame(&self, identity: EntityId) -> u32 {
        self.poses
            .get(&identity.slot())
            .filter(|pose| pose.identity == identity)
            .map_or(0, |pose| {
                if pose.frame == 0 {
                    pose.own_block
                } else {
                    pose.frame
                }
            })
    }

    fn allocate(&mut self, count: usize) -> Result<u32, RenderError> {
        let start = self.blocks.len();
        if count > self.max_blocks - start {
            return Err(invalid("pose blocks exceed the storage-binding limit"));
        }
        self.blocks.resize(start + count, IDENTITY_BLOCK);
        Ok(start as u32)
    }

    fn write(&mut self, index: usize, matrices: &[[f32; 16]]) {
        for (dst, src) in self.blocks[index].iter_mut().zip(matrices) {
            *dst = Mat4(*src).to_columns();
        }
        self.dirty.insert(index);
    }

    pub(crate) fn upload(&mut self, gpu: &Gpu) {
        if self.dirty.is_empty() {
            return;
        }
        if self.blocks.len() as u64 * BLOCK_BYTES > self.buffer.size() {
            let capacity = self.blocks.len().next_power_of_two().min(self.max_blocks);
            (self.buffer, self.group) = storage(gpu.device(), &self.layout, capacity);
            gpu.queue()
                .write_buffer(&self.buffer, 0, bytemuck::cast_slice(&self.blocks));
            self.dirty.clear();
            return;
        }
        let dirty = std::mem::take(&mut self.dirty);
        let mut indices = dirty.into_iter().peekable();
        while let Some(start) = indices.next() {
            let mut end = start + 1;
            while indices.next_if_eq(&end).is_some() {
                end += 1;
            }
            gpu.queue().write_buffer(
                &self.buffer,
                start as u64 * BLOCK_BYTES,
                bytemuck::cast_slice(&self.blocks[start..end]),
            );
        }
    }
}

pub fn validate_matrices(matrices: &[[f32; 16]]) -> Result<(), RenderError> {
    if !(1..=MAX_JOINTS).contains(&matrices.len())
        || matrices
            .iter()
            .flatten()
            .any(|component| !component.is_finite())
    {
        return Err(invalid("need 1 to 32 finite 4x4 joint matrices"));
    }
    Ok(())
}

fn invalid(message: &str) -> RenderError {
    RenderError::Resource(format!("skinning: {message}"))
}

fn storage(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    capacity: usize,
) -> (wgpu::Buffer, wgpu::BindGroup) {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("native joint blocks"),
        size: capacity as u64 * BLOCK_BYTES,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("native joint blocks"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });
    (buffer, group)
}
