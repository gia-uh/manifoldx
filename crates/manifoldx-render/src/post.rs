//! Screen-space passes over the native scene colour and reversed depth.

use std::mem::size_of;

use crate::camera::Camera;
use crate::math::{transform_direction, Mat4};
use crate::renderer::{srgb_to_linear, DirectionalLight};

const HDR: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const AO: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;
const UNIFORM_FLOATS: usize = 220;

#[derive(Clone, Debug)]
pub struct PostOptions {
    /// Radius, intensity, sample count, bias and power.
    pub ssao: Option<[f32; 5]>,
    /// Threshold, knee and intensity.
    pub bloom: Option<[f32; 3]>,
    /// Intensity, decay, density and sample count.
    pub shafts: Option<[f32; 4]>,
    /// Saturation, contrast and vignette.
    pub grade: Option<[f32; 3]>,
    pub sky: Option<Sky>,
    pub fxaa: bool,
    pub kernel: [[f32; 4]; 32],
}

#[derive(Clone, Copy, Debug)]
pub struct Sky {
    pub zenith: [f32; 3],
    pub horizon: Option<[f32; 3]>,
    pub sun_size_degrees: f32,
    pub sun_intensity: f32,
    pub glow: f32,
}

impl PostOptions {
    fn uniforms(
        &self,
        [width, height]: [u32; 2],
        camera: &Camera,
        sun: Option<DirectionalLight>,
        background: [f32; 3],
    ) -> ([f32; UNIFORM_FLOATS], bool) {
        let aspect = width as f32 / height as f32;
        let projection = camera.projection(aspect);
        let view = camera.view();
        // Invert the diagonal projection and its lower 2x2 reverse-depth block.
        let mut inv_proj = Mat4::ZERO;
        inv_proj.0[0] = projection.0[0].recip();
        inv_proj.0[5] = projection.0[5].recip();
        inv_proj.0[11] = -1.0;
        inv_proj.0[14] = projection.0[11].recip();
        inv_proj.0[15] = projection.0[10] / projection.0[11];
        // Only the inverse rotation is used by the shader's world-space rays.
        let mut inv_rotation = Mat4::IDENTITY;
        for row in 0..3 {
            for column in 0..3 {
                inv_rotation.0[row * 4 + column] = view.at(column, row);
            }
        }

        let mut data = [0.0; UNIFORM_FLOATS];
        data[..16].copy_from_slice(&projection.to_columns());
        data[16..32].copy_from_slice(&inv_proj.to_columns());
        data[60..76].copy_from_slice(&inv_rotation.to_columns());
        let [mut sun_u, mut sun_v, mut visible] = [0.5, 0.5, 0.0];
        if let Some(sun) = sun.filter(|sun| sun.intensity > 0.0) {
            let norm = sun.direction.iter().map(|x| x * x).sum::<f32>().sqrt();
            let to_sun = sun
                .direction
                .map(|x| -x / if norm > 0.0 { norm } else { 1.0 });
            data[76..79].copy_from_slice(&to_sun);
            data[79] = 1.0;
            let [r, g, b, _] = srgb_to_linear(sun.color)
                .map(|value| (value * sun.intensity.min(1.0)).clamp(0.0, 1.0));
            let max = r.max(g).max(b);
            if max > 0.0 {
                data[56..59].copy_from_slice(&[r / max, g / max, b / max]);
            }
            let [x, y, z] = transform_direction(&view, to_sun);
            if self.shafts.is_some() && to_sun[1] > -0.02 && z < -0.001 {
                let x = projection.0[0] * x / -z;
                let y = projection.0[5] * y / -z;
                sun_u = 0.5 + 0.5 * x;
                sun_v = 0.5 - 0.5 * y;
                visible = ((1.6 - x.abs().max(y.abs())) / 0.6).clamp(0.0, 1.0)
                    * (to_sun[1] / 0.08 + 0.25).clamp(0.0, 1.0);
            }
        }
        let shafts_on = visible > 0.0;
        data[32..36].copy_from_slice(&[width as f32, height as f32, sun_u, sun_v]);
        let [radius, intensity, samples, bias, power] =
            self.ssao.unwrap_or([1.0, 1.0, 16.0, 0.03, 2.0]);
        data[36..40].copy_from_slice(&[radius, intensity, samples, bias]);
        data[40..43].copy_from_slice(&self.bloom.unwrap_or([0.85, 0.1, 0.5]));
        let [intensity, decay, density, shaft_samples] =
            self.shafts.unwrap_or([0.5, 0.96, 0.9, 48.0]);
        data[44..48].copy_from_slice(&[intensity, decay, density, visible]);
        let [saturation, contrast, vignette] = self.grade.unwrap_or([1.0, 1.0, 0.0]);
        data[48..52].copy_from_slice(&[saturation, contrast, vignette, power]);
        data[52..56].copy_from_slice(
            &[
                self.ssao.is_some() as u32,
                self.bloom.is_some() as u32,
                shafts_on as u32,
                shaft_samples as u32,
            ]
            .map(f32::from_bits),
        );
        let sky = self.sky.unwrap_or(Sky {
            zenith: [0.18, 0.36, 0.78],
            horizon: None,
            sun_size_degrees: 0.6,
            sun_intensity: 24.0,
            glow: 1.0,
        });
        data[80..83].copy_from_slice(&sky.zenith);
        data[84..87].copy_from_slice(&sky.horizon.unwrap_or(background));
        data[88..92].copy_from_slice(&[
            sky.sun_size_degrees.to_radians().cos(),
            sky.sun_intensity,
            sky.glow,
            u8::from(self.sky.is_some()) as f32,
        ]);
        data[92..].copy_from_slice(self.kernel.as_flattened());
        (data, shafts_on)
    }
}

#[repr(usize)]
#[derive(Clone, Copy)]
enum Program {
    Ssao,
    SsaoBlur,
    Bright,
    Down,
    Up,
    Shafts,
    Composite,
    Fxaa,
}

struct Targets {
    size: [u32; 2],
    scene: wgpu::TextureView,
    ldr: wgpu::TextureView,
    ao: wgpu::TextureView,
    ao_blur: wgpu::TextureView,
    down: [wgpu::TextureView; 4],
    up: [wgpu::TextureView; 3],
    shafts: wgpu::TextureView,
}

struct Groups {
    depth: wgpu::TextureView,
    ssao: wgpu::BindGroup,
    ssao_blur: wgpu::BindGroup,
    bright: wgpu::BindGroup,
    down: [wgpu::BindGroup; 3],
    up: [wgpu::BindGroup; 3],
    shafts: wgpu::BindGroup,
    composite: wgpu::BindGroup,
    fxaa: wgpu::BindGroup,
}

pub(crate) struct PostPasses {
    format: wgpu::TextureFormat,
    pipelines: [wgpu::RenderPipeline; 8],
    uniforms: wgpu::Buffer,
    sampler: wgpu::Sampler,
    targets: Option<Targets>,
    groups: Option<Groups>,
    generation: u64,
}

impl PostPasses {
    pub(crate) fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let vs = include_str!("post/vs.wgsl");
        let uniform = include_str!("post/uniforms.wgsl");
        let sources = [
            ("ssao", uniform, include_str!("post/ssao.wgsl"), AO),
            (
                "ssao-blur",
                uniform,
                include_str!("post/ssao_blur.wgsl"),
                AO,
            ),
            (
                "bloom-bright",
                uniform,
                include_str!("post/bright.wgsl"),
                HDR,
            ),
            ("bloom-down", "", include_str!("post/down.wgsl"), HDR),
            ("bloom-up", "", include_str!("post/up.wgsl"), HDR),
            ("shafts", uniform, include_str!("post/shafts.wgsl"), HDR),
            (
                "composite",
                uniform,
                include_str!("post/composite.wgsl"),
                format,
            ),
            ("fxaa", "", include_str!("post/fxaa.wgsl"), format),
        ];
        let pipelines = sources.map(|(name, uniforms, fragment, target)| {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(name),
                source: wgpu::ShaderSource::Wgsl(format!("{vs}{uniforms}{fragment}").into()),
            });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(name),
                layout: None,
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: target,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        });
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("manifoldx-post-uniforms"),
            size: (UNIFORM_FLOATS * size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Self {
            format,
            pipelines,
            uniforms,
            sampler,
            targets: None,
            groups: None,
            generation: 0,
        }
    }

    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        [w, h]: [u32; 2],
        depth: &wgpu::TextureView,
    ) {
        if self
            .targets
            .as_ref()
            .is_none_or(|targets| targets.size != [w, h])
        {
            let texture = |width: u32, height: u32, format| {
                device
                    .create_texture(&wgpu::TextureDescriptor {
                        label: Some("manifoldx-post-target"),
                        size: wgpu::Extent3d {
                            width: width.max(1),
                            height: height.max(1),
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format,
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                            | wgpu::TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    })
                    .create_view(&wgpu::TextureViewDescriptor::default())
            };
            self.targets = Some(Targets {
                size: [w, h],
                scene: texture(w, h, self.format),
                ldr: texture(w, h, self.format),
                ao: texture(w.div_ceil(2), h.div_ceil(2), AO),
                ao_blur: texture(w.div_ceil(2), h.div_ceil(2), AO),
                down: std::array::from_fn(|k| texture(w >> (k + 1), h >> (k + 1), HDR)),
                up: std::array::from_fn(|k| texture(w >> (k + 1), h >> (k + 1), HDR)),
                shafts: texture(w >> 2, h >> 2, HDR),
            });
            self.groups = None;
            self.generation += 1;
        }
        if self
            .groups
            .as_ref()
            .is_some_and(|groups| &groups.depth == depth)
        {
            return;
        }
        let t = self.targets.as_ref().expect("post targets prepared");
        let u = || self.uniforms.as_entire_binding();
        let s = || wgpu::BindingResource::Sampler(&self.sampler);
        fn v(view: &wgpu::TextureView) -> wgpu::BindingResource<'_> {
            wgpu::BindingResource::TextureView(view)
        }
        let group = |program: Program, first: u32, resources: &[wgpu::BindingResource<'_>]| {
            let entries: Vec<_> = resources
                .iter()
                .cloned()
                .enumerate()
                .map(|(i, resource)| wgpu::BindGroupEntry {
                    binding: first + i as u32,
                    resource,
                })
                .collect();
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("manifoldx-post-group"),
                layout: &self.pipelines[program as usize].get_bind_group_layout(0),
                entries: &entries,
            })
        };
        self.groups = Some(Groups {
            depth: depth.clone(),
            ssao: group(Program::Ssao, 0, &[u(), v(depth)]),
            ssao_blur: group(Program::SsaoBlur, 0, &[u(), v(depth), v(&t.ao)]),
            bright: group(Program::Bright, 0, &[u(), v(&t.scene), s(), v(depth)]),
            down: std::array::from_fn(|k| group(Program::Down, 1, &[v(&t.down[k]), s()])),
            up: std::array::from_fn(|k| {
                let low = if k == 2 { &t.down[3] } else { &t.up[k + 1] };
                group(Program::Up, 1, &[v(low), s(), v(&t.down[k])])
            }),
            shafts: group(Program::Shafts, 0, &[u(), v(depth)]),
            composite: group(
                Program::Composite,
                0,
                &[
                    u(),
                    v(&t.scene),
                    s(),
                    v(&t.ao_blur),
                    v(&t.up[0]),
                    v(&t.shafts),
                    v(depth),
                ],
            ),
            fxaa: group(Program::Fxaa, 1, &[v(&t.ldr), s()]),
        });
    }

    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }

    pub(crate) fn scene(&self) -> &wgpu::TextureView {
        &self.targets.as_ref().expect("post targets prepared").scene
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        output: &wgpu::TextureView,
        options: &PostOptions,
        camera: &Camera,
        sun: Option<DirectionalLight>,
        background: [f32; 3],
    ) -> Vec<&'static str> {
        let t = self.targets.as_ref().expect("post targets prepared");
        let g = self.groups.as_ref().expect("post groups prepared");
        let (uniforms, shafts_on) = options.uniforms(t.size, camera, sun, background);
        queue.write_buffer(&self.uniforms, 0, bytemuck::cast_slice(&uniforms));
        let mut order = Vec::with_capacity(12);
        let mut draw = |name: &'static str, program: Program, view, group| {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(name),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.pipelines[program as usize]);
            pass.set_bind_group(0, group, &[]);
            pass.draw(0..3, 0..1);
            order.push(name);
        };
        if options.ssao.is_some() {
            draw("ssao", Program::Ssao, &t.ao, &g.ssao);
            draw("ssao-blur", Program::SsaoBlur, &t.ao_blur, &g.ssao_blur);
        }
        if options.bloom.is_some() {
            draw("bloom-bright", Program::Bright, &t.down[0], &g.bright);
            for k in 0..3 {
                draw("bloom-down", Program::Down, &t.down[k + 1], &g.down[k]);
            }
            for k in (0..3).rev() {
                draw("bloom-up", Program::Up, &t.up[k], &g.up[k]);
            }
        }
        if shafts_on {
            draw("shafts", Program::Shafts, &t.shafts, &g.shafts);
        }
        let composite = if options.fxaa { &t.ldr } else { output };
        draw("composite", Program::Composite, composite, &g.composite);
        if options.fxaa {
            draw("fxaa", Program::Fxaa, output, &g.fxaa);
        }
        order
    }
}
