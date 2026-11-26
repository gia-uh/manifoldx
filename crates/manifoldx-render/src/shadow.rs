//! Directional and spot shadow projections and the depth-map array.

use crate::camera::Camera;
use crate::math::Mat4;
use crate::renderer::{DirectionalLight, RenderError, SpotLight};

pub(crate) const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth24Plus;
const UNIFORM_FLOATS: usize = 88;

#[derive(Clone, Copy, Debug)]
pub struct ShadowOptions {
    pub target: [f32; 3],
    pub extent: f32,
    pub resolution: u32,
    pub near: f32,
    pub far: f32,
    pub bias: f32,
    pub pcf_radius: u32,
    pub auto_fit: bool,
    pub cascades: u32,
    pub distance: Option<f32>,
}

impl ShadowOptions {
    pub fn validate(self) -> Result<(), RenderError> {
        if self.target.iter().any(|x| !x.is_finite())
            || !self.extent.is_finite()
            || self.extent <= 0.0
            || !self.near.is_finite()
            || self.near < 0.0
            || !self.far.is_finite()
            || self.far <= self.near
            || self.distance.is_some_and(|d| !d.is_finite() || d <= 0.0)
            || !self.bias.is_finite()
            || self.bias < 0.0
            || !matches!(self.cascades, 1..=4)
            || !matches!(self.resolution, 1..=8192)
            || self.pcf_radius > 8
        {
            return Err(RenderError::Resource("invalid shadow options".into()));
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub(crate) struct ShadowFrame {
    pub(crate) matrices: [Mat4; 4],
    pub(crate) layers: usize,
    pub(crate) resolution: u32,
    view: Mat4,
    splits: [f32; 4],
    texels: [f32; 4],
    ranges: [f32; 4],
    caster: f32,
    bias: f32,
    pcf_radius: f32,
}

impl ShadowFrame {
    pub(crate) fn new(
        options: ShadowOptions,
        camera: &Camera,
        aspect: f32,
        sun: Option<DirectionalLight>,
        spot: Option<SpotLight>,
        bounds: Option<([f32; 3], f32)>,
    ) -> Option<Self> {
        let mut frame = Self {
            matrices: [Mat4::IDENTITY; 4],
            layers: 1,
            resolution: options.resolution,
            view: camera.view(),
            splits: [0.0; 4],
            texels: [0.0; 4],
            ranges: [0.0; 4],
            caster: 1.0,
            bias: options.bias,
            pcf_radius: options.pcf_radius as f32,
        };
        if let Some(spot) = spot {
            let direction = unit(spot.direction)?;
            let near = options.near.max(0.001);
            let far = spot.range;
            if far <= near || !far.is_finite() {
                return None;
            }
            let tan = spot.cos_outer.acos().tan();
            if !tan.is_finite() || tan <= 0.0 {
                return None;
            }
            let f = tan.recip();
            let projection = Mat4([
                f,
                0.0,
                0.0,
                0.0,
                0.0,
                f,
                0.0,
                0.0,
                0.0,
                0.0,
                far / (near - far),
                near * far / (near - far),
                0.0,
                0.0,
                -1.0,
                0.0,
            ]);
            frame.matrices[0] = projection.multiply(&light_view(spot.position, direction));
            frame.caster = 2.0;
            return Some(frame);
        }
        let direction = unit(sun?.direction)?;
        if options.cascades > 1 {
            let far = options.distance.unwrap_or(camera.far.min(300.0));
            let origin = camera.position;
            let forward = [-frame.view.0[8], -frame.view.0[9], -frame.view.0[10]];
            let tan = (camera.fov_y_degrees.to_radians() * 0.5).tan();
            let a2 = tan * tan * (1.0 + aspect * aspect);
            let light = light_view([0.0; 3], direction);
            let right = [light.0[0], light.0[1], light.0[2]];
            let up = [light.0[4], light.0[5], light.0[6]];
            let near = camera.near.max(0.001);
            if far <= near {
                return None;
            }
            frame.layers = options.cascades as usize;
            let mut previous = near;
            let mut previous_start = near;
            for k in 0..frame.layers {
                let fraction = (k + 1) as f32 / options.cascades as f32;
                let end = 0.75 * near * (far / near).powf(fraction)
                    + 0.25 * (near + (far - near) * fraction);
                let start = if k == 0 {
                    camera.near
                } else {
                    previous - 0.1 * (previous - previous_start)
                };
                let offset = (0.5 * (start + end) * (1.0 + a2)).min(end);
                let r = ((offset - start).powi(2) + a2 * start * start)
                    .max((end - offset).powi(2) + a2 * end * end)
                    .sqrt();
                let mut center = add_scaled(origin, forward, offset);
                let texel = 2.0 * r / options.resolution as f32;
                for axis in [right, up] {
                    let d = dot(axis, center);
                    center = add_scaled(center, axis, (d / texel).round_ties_even() * texel - d);
                }
                let back = r + far;
                frame.matrices[k] = ortho(direction, center, r, 0.0, back + r, back);
                frame.splits[k] = end;
                frame.texels[k] = texel;
                frame.ranges[k] = back + r;
                previous_start = previous;
                previous = end;
            }
        } else if let Some((center, radius)) = bounds.filter(|_| options.auto_fit) {
            let pad = (radius * 0.05).max(0.1);
            let r = radius + pad;
            frame.matrices[0] = ortho(direction, center, r, 0.01, 2.0 * r + pad, r + pad);
        } else {
            frame.matrices[0] = ortho(
                direction,
                options.target,
                options.extent,
                options.near,
                options.far,
                0.5 * options.far,
            );
        }
        Some(frame)
    }

    fn data(&self) -> [f32; UNIFORM_FLOATS] {
        let mut data = [0.0; UNIFORM_FLOATS];
        for (index, matrix) in self.matrices.iter().enumerate() {
            data[index * 16..(index + 1) * 16].copy_from_slice(&matrix.to_columns());
        }
        data[64..68].copy_from_slice(&[
            self.caster,
            self.resolution as f32,
            self.pcf_radius,
            self.bias,
        ]);
        data[68..72].copy_from_slice(&self.splits);
        data[72..76].copy_from_slice(&self.texels);
        data[76..80].copy_from_slice(&self.ranges);
        data[80..84].copy_from_slice(&[
            self.view.0[8],
            self.view.0[9],
            self.view.0[10],
            self.view.0[11],
        ]);
        data[84] = self.layers as f32;
        data
    }
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a.into_iter().zip(b).map(|(a, b)| a * b).sum()
}

fn unit(v: [f32; 3]) -> Option<[f32; 3]> {
    let length = dot(v, v).sqrt();
    (length.is_finite() && length > 0.0).then(|| v.map(|component| component / length))
}

fn add_scaled(a: [f32; 3], b: [f32; 3], factor: f32) -> [f32; 3] {
    std::array::from_fn(|i| a[i] + b[i] * factor)
}

fn light_view(eye: [f32; 3], direction: [f32; 3]) -> Mat4 {
    let mut camera = Camera::new(eye, add_scaled(eye, direction, 1.0));
    if direction[1].abs() > 0.999 {
        camera.up = [0.0, 0.0, 1.0];
    }
    camera.view()
}

fn ortho(
    direction: [f32; 3],
    target: [f32; 3],
    extent: f32,
    near: f32,
    far: f32,
    back: f32,
) -> Mat4 {
    let inv = (far - near).recip();
    let scale = extent.recip();
    let proj = Mat4([
        scale,
        0.0,
        0.0,
        0.0,
        0.0,
        scale,
        0.0,
        0.0,
        0.0,
        0.0,
        -inv,
        -near * inv,
        0.0,
        0.0,
        0.0,
        1.0,
    ]);
    proj.multiply(&light_view(add_scaled(target, direction, -back), direction))
}

pub(crate) struct ShadowMap {
    pub(crate) layout: wgpu::BindGroupLayout,
    pub(crate) group: wgpu::BindGroup,
    pub(crate) depth: Vec<wgpu::TextureView>,
    size: u32,
    sampler: wgpu::Sampler,
    uniform: wgpu::Buffer,
    active: bool,
}

impl ShadowMap {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("native shadow sample layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    count: None,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    count: None,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    count: None,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                },
            ],
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            compare: Some(wgpu::CompareFunction::Less),
            ..Default::default()
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("native shadow sample parameters"),
            size: (UNIFORM_FLOATS * size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (group, depth) = allocate_map(device, &layout, &sampler, &uniform, 1, 1);
        Self {
            layout,
            group,
            depth,
            size: 1,
            sampler,
            uniform,
            active: false,
        }
    }

    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame: Option<&ShadowFrame>,
    ) -> Result<(), RenderError> {
        if let Some(frame) = frame {
            if frame.resolution > device.limits().max_texture_dimension_2d {
                return Err(RenderError::Resource(
                    "shadow map exceeds the device dimensions".into(),
                ));
            }
            if self.size != frame.resolution || self.depth.len() != frame.layers {
                (self.group, self.depth) = allocate_map(
                    device,
                    &self.layout,
                    &self.sampler,
                    &self.uniform,
                    frame.resolution,
                    frame.layers,
                );
                self.size = frame.resolution;
            }
            queue.write_buffer(&self.uniform, 0, bytemuck::cast_slice(&frame.data()));
        } else if self.active {
            queue.write_buffer(
                &self.uniform,
                0,
                bytemuck::cast_slice(&[0.0f32; UNIFORM_FLOATS]),
            );
        }
        self.active = frame.is_some();
        Ok(())
    }
}

fn allocate_map(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    uniform: &wgpu::Buffer,
    size: u32,
    layers: usize,
) -> (wgpu::BindGroup, Vec<wgpu::TextureView>) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("native shadow-map array"),
        size: wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: layers as u32,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("native shadow map"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: uniform.as_entire_binding(),
            },
        ],
    });
    let layers = (0..layers)
        .map(|i| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: i as u32,
                array_layer_count: Some(1),
                ..Default::default()
            })
        })
        .collect();
    (group, layers)
}
