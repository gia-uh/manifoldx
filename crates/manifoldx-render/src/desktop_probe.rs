use manifoldx_render::desktop::{run_desktop, DesktopApplication, DesktopOptions};
use manifoldx_render::winit::{dpi::LogicalSize, window::Window};
use manifoldx_render::{
    wgpu, FrameTarget, Gpu, NativeRenderer, RenderError, RenderStats, RendererOptions,
};

#[derive(Clone, Debug)]
pub enum Probe {
    Positive,
    ForeignDevice,
    LogicalAsPhysical,
}

impl Probe {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "positive" => Ok(Self::Positive),
            "foreign-device" => Ok(Self::ForeignDevice),
            "logical-as-physical" => Ok(Self::LogicalAsPhysical),
            _ => Err(format!("unknown desktop control {value:?}")),
        }
    }

    pub fn run(self) -> Result<(), String> {
        let logical = LogicalSize::new(400_u32, 300_u32);
        let scale = if matches!(self, Self::LogicalAsPhysical) {
            1.0
        } else {
            2.0
        };
        let options = DesktopOptions {
            title: "Native surface control".into(),
            physical_px: logical.to_physical(scale),
            fullscreen: false,
            monitor: None,
            present_mode: wgpu::PresentMode::Fifo,
            renderer: RendererOptions::default(),
            sample_limit: Some(1),
            suppress_present: false,
            resize_after_three: None,
        };
        let (presents, _) =
            run_desktop(options, &mut SurfaceProbe(self)).map_err(|error| error.to_string())?;
        if presents != 1 {
            return Err(format!(
                "surface probe expected one present, observed {presents}"
            ));
        }
        println!("{{\"physical_px\":[800,600],\"scale\":2,\"presented\":{presents}}}");
        Ok(())
    }
}

struct SurfaceProbe(Probe);

impl DesktopApplication for SurfaceProbe {
    fn start(
        &mut self,
        _: &Window,
        _: &mut NativeRenderer,
        _: &manifoldx_render::desktop::DesktopInfo,
    ) -> Result<bool, RenderError> {
        Ok(true)
    }

    fn draw(
        &mut self,
        window: &Window,
        _: &mut NativeRenderer,
        encoder: &mut wgpu::CommandEncoder,
        target: FrameTarget<'_>,
    ) -> Result<Option<RenderStats>, RenderError> {
        if window.scale_factor() != 2.0 {
            return Err(RenderError::Window(format!(
                "expected scale 2, got {}",
                window.scale_factor()
            )));
        }
        if (target.width, target.height) != (800, 600) {
            return Err(RenderError::Window(format!(
                "physical extent mismatch: expected 800x600, got {}x{}",
                target.width, target.height
            )));
        }
        if matches!(self.0, Probe::ForeignDevice) {
            let foreign = Gpu::new(false)?;
            let scope = foreign
                .device()
                .push_error_scope(wgpu::ErrorFilter::Validation);
            let mut foreign_encoder = foreign.device().create_command_encoder(&Default::default());
            clear(&mut foreign_encoder, target.color);
            foreign.queue().submit([foreign_encoder.finish()]);
            if let Some(error) = pollster::block_on(scope.pop()) {
                return Err(RenderError::Device(format!(
                    "foreign surface device rejected: {error}"
                )));
            }
        }
        clear(encoder, target.color);
        Ok(Some(RenderStats::default()))
    }
}

fn clear(encoder: &mut wgpu::CommandEncoder, color: &wgpu::TextureView) {
    let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: color,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLUE),
                store: wgpu::StoreOp::Store,
            },
        })],
        ..Default::default()
    });
}
