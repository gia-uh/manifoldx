//! Native GPU ownership for prefiltered environment maps.

use crate::renderer::RenderError;

pub(crate) struct EnvironmentGpu {
    sampler: wgpu::Sampler,
    irradiance: wgpu::TextureView,
    prefiltered: wgpu::TextureView,
    brdf: wgpu::TextureView,
    has_lut: bool,
}

pub struct EnvironmentData {
    pub irradiance: Vec<u8>,
    pub prefiltered: Vec<Vec<u8>>,
    pub brdf: Option<Vec<u8>>,
}

impl EnvironmentData {
    pub fn validate(&self) -> Result<(), RenderError> {
        if self.irradiance.len() != 6 * 64 * 64 * 8
            || self.prefiltered.len() != 8
            || self.prefiltered.iter().enumerate().any(|(level, bytes)| {
                let side = 128 >> level;
                bytes.len() != 6 * side * side * 8
            })
            || self
                .brdf
                .as_ref()
                .is_some_and(|bytes| bytes.len() != 512 * 512 * 4)
        {
            return Err(RenderError::Resource(
                "invalid environment cube or BRDF extents".into(),
            ));
        }
        Ok(())
    }
}

impl EnvironmentGpu {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            min_filter: wgpu::FilterMode::Linear,
            mag_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        let black = texture(
            device,
            "inactive environment cube",
            1,
            6,
            1,
            wgpu::TextureFormat::Rgba16Float,
        );
        let black = cube_view(&black);
        let brdf = texture(
            device,
            "inactive environment BRDF",
            1,
            1,
            1,
            wgpu::TextureFormat::Rg16Float,
        )
        .create_view(&Default::default());
        Self {
            sampler,
            irradiance: black.clone(),
            prefiltered: black,
            brdf,
            has_lut: false,
        }
    }

    pub(crate) fn load(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: &EnvironmentData,
    ) -> Result<(), RenderError> {
        data.validate()?;
        if !self.has_lut && data.brdf.is_none() {
            return Err(RenderError::Resource(
                "the first environment needs a GGX BRDF LUT".into(),
            ));
        }
        let format = wgpu::TextureFormat::Rgba16Float;
        let irr = texture(device, "environment irradiance", 64, 6, 1, format);
        upload(queue, &irr, 0, 64, 6, 8, &data.irradiance);
        let prefiltered = texture(device, "environment roughness mips", 128, 6, 8, format);
        for (level, bytes) in data.prefiltered.iter().enumerate() {
            upload(queue, &prefiltered, level as u32, 128 >> level, 6, 8, bytes);
        }
        if let Some(brdf) = &data.brdf {
            let tex = texture(
                device,
                "environment BRDF",
                512,
                1,
                1,
                wgpu::TextureFormat::Rg16Float,
            );
            upload(queue, &tex, 0, 512, 1, 4, brdf);
            self.brdf = tex.create_view(&Default::default());
            self.has_lut = true;
        }
        self.irradiance = cube_view(&irr);
        self.prefiltered = cube_view(&prefiltered);
        Ok(())
    }

    pub(crate) fn layout() -> [wgpu::BindGroupLayoutEntry; 4] {
        let texture = |binding, view_dimension| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            count: None,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension,
                multisampled: false,
            },
        };
        [
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                count: None,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            },
            texture(4, wgpu::TextureViewDimension::Cube),
            texture(5, wgpu::TextureViewDimension::Cube),
            texture(6, wgpu::TextureViewDimension::D2),
        ]
    }

    pub(crate) fn bindings(&self) -> [wgpu::BindGroupEntry<'_>; 4] {
        [
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(&self.sampler),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(&self.irradiance),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::TextureView(&self.prefiltered),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::TextureView(&self.brdf),
            },
        ]
    }
}

fn texture(
    device: &wgpu::Device,
    label: &str,
    side: u32,
    layers: u32,
    mips: u32,
    format: wgpu::TextureFormat,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: side,
            height: side,
            depth_or_array_layers: layers,
        },
        mip_level_count: mips,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

fn cube_view(texture: &wgpu::Texture) -> wgpu::TextureView {
    texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::Cube),
        ..Default::default()
    })
}

fn upload(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    level: u32,
    side: u32,
    layers: u32,
    texel_bytes: u32,
    bytes: &[u8],
) {
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: level,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(side * texel_bytes),
            rows_per_image: Some(side),
        },
        wgpu::Extent3d {
            width: side,
            height: side,
            depth_or_array_layers: layers,
        },
    );
}
