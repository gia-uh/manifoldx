//! GPU pipelines and material bindings for restricted Python fragments.

use crate::renderer::{Gpu, RenderError, DEPTH_FORMAT};

pub struct FragmentData {
    pub source: String,
    pub uniforms: Vec<f32>,
    pub textures: Vec<Vec<u8>>,
}

pub(crate) struct FragmentPipeline {
    pub(crate) group: wgpu::BindGroup,
    pub(crate) pipelines: [wgpu::RenderPipeline; 2],
}

impl FragmentPipeline {
    pub(crate) fn new(
        gpu: &Gpu,
        format: wgpu::TextureFormat,
        vertex: &[wgpu::VertexBufferLayout<'_>],
        base_source: &str,
        vertex_layouts: [&wgpu::BindGroupLayout; 2],
        data: &FragmentData,
    ) -> Result<Self, RenderError> {
        let device = gpu.device();
        if data.uniforms.iter().any(|x| !x.is_finite())
            || data.uniforms.len() as u64 * 4 > device.limits().max_uniform_buffer_binding_size
            || data.textures.len() > device.limits().max_sampled_textures_per_shader_stage as usize
            || data.textures.len() > device.limits().max_samplers_per_shader_stage as usize
            || data.textures.iter().any(|bytes| {
                bytes.is_empty()
                    || bytes.len() % 4 != 0
                    || bytes.len() / 4 > device.limits().max_texture_dimension_1d as usize
            })
        {
            return Err(RenderError::Resource(
                "invalid fragment material resource size".into(),
            ));
        }
        let mut entries = Vec::new();
        if !data.uniforms.is_empty() {
            entries.push(binding(
                2,
                wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            ));
        }
        for index in 0..data.textures.len() as u32 {
            entries.push(binding(
                3 + index * 2,
                wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            ));
            entries.push(binding(
                4 + index * 2,
                wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D1,
                    multisampled: false,
                },
            ));
        }
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("native fragment material bindings"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("native fragment layout"),
            bind_group_layouts: &[
                Some(vertex_layouts[0]),
                Some(&layout),
                Some(vertex_layouts[1]),
            ],
            immediate_size: 0,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("native generated fragment"),
            source: wgpu::ShaderSource::Wgsl(format!("{base_source}\n{}", data.source).into()),
        });
        let pipelines = [wgpu::FrontFace::Ccw, wgpu::FrontFace::Cw].map(|front_face| {
            make_pipeline(
                device,
                &module,
                &pipeline_layout,
                vertex,
                format,
                "fs_custom",
                (front_face, None),
            )
        });
        if let Some(error) = pollster::block_on(scope.pop()) {
            return Err(RenderError::Resource(format!(
                "fragment material WGSL: {error}"
            )));
        }

        let uniform = (!data.uniforms.is_empty()).then(|| {
            let size = (data.uniforms.len() as u64 * 4).next_multiple_of(16);
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("native fragment scalar values"),
                size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            gpu.queue()
                .write_buffer(&buffer, 0, bytemuck::cast_slice(&data.uniforms));
            buffer
        });
        let mut entries = Vec::new();
        if let Some(uniform) = &uniform {
            entries.push(wgpu::BindGroupEntry {
                binding: 2,
                resource: uniform.as_entire_binding(),
            });
        }
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("native fragment linear clamp sampler"),
            min_filter: wgpu::FilterMode::Linear,
            mag_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let views: Vec<_> = data
            .textures
            .iter()
            .map(|bytes| upload_1d(gpu, bytes))
            .collect();
        for (index, view) in views.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: 3 + index as u32 * 2,
                resource: wgpu::BindingResource::Sampler(&sampler),
            });
            entries.push(wgpu::BindGroupEntry {
                binding: 4 + index as u32 * 2,
                resource: wgpu::BindingResource::TextureView(view),
            });
        }
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("native fragment material"),
            layout: &layout,
            entries: &entries,
        });
        Ok(Self { group, pipelines })
    }
}

fn binding(binding: u32, ty: wgpu::BindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        ty,
        visibility: wgpu::ShaderStages::FRAGMENT,
        count: None,
    }
}

pub(crate) fn upload_1d(gpu: &Gpu, bytes: &[u8]) -> wgpu::TextureView {
    let size = wgpu::Extent3d {
        width: (bytes.len() / 4) as u32,
        height: 1,
        depth_or_array_layers: 1,
    };
    let texture = gpu.device().create_texture(&wgpu::TextureDescriptor {
        label: Some("native fragment 1D texels"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D1,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    gpu.queue().write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(size.width * 4),
            rows_per_image: Some(1),
        },
        size,
    );
    texture.create_view(&Default::default())
}

pub(crate) fn make_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
    vertex: &[wgpu::VertexBufferLayout<'_>],
    format: wgpu::TextureFormat,
    fragment: &str,
    (front_face, blend): (wgpu::FrontFace, Option<wgpu::BlendState>),
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("native fragment pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            buffers: vertex,
            compilation_options: Default::default(),
        },
        primitive: wgpu::PrimitiveState {
            front_face,
            cull_mode: blend.is_none().then_some(wgpu::Face::Back),
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(blend.is_none()),
            depth_compare: Some(wgpu::CompareFunction::Greater),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some(fragment),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}
