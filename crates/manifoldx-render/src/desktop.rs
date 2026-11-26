//! Calling-thread desktop event loop. Window, surface and device stay together.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    monitor::MonitorHandle,
    window::{Fullscreen, Window, WindowId},
};

use crate::renderer::{depth_target, FrameTarget, Gpu, NativeRenderer, RendererOptions};
use crate::{RenderError, RenderStats};

#[derive(Clone, Debug)]
pub struct Monitor {
    pub name: String,
    pub physical_px: PhysicalSize<u32>,
    pub scale_factor: f64,
    pub refresh_millihertz: Option<u32>,
    pub primary: bool,
}

pub struct DesktopOptions {
    pub title: String,
    pub physical_px: PhysicalSize<u32>,
    pub fullscreen: bool,
    pub monitor: Option<String>,
    pub present_mode: wgpu::PresentMode,
    pub renderer: RendererOptions,
    pub sample_limit: Option<u64>,
    pub suppress_present: bool,
    pub resize_after_three: Option<PhysicalSize<u32>>,
}

#[derive(Clone)]
pub struct DesktopInfo {
    pub monitors: Vec<Monitor>,
    pub selected_monitor: Option<usize>,
    pub physical_px: PhysicalSize<u32>,
    pub scale_factor: f64,
    pub format: wgpu::TextureFormat,
    pub present_mode: wgpu::PresentMode,
}

#[derive(Debug)]
pub struct PresentedFrame {
    pub sequence: u64,
    pub physical_px: PhysicalSize<u32>,
    pub scale_factor: f64,
    pub wall_interval: Option<Duration>,
    pub acquire: Duration,
    pub submit: Duration,
    pub present: Duration,
    pub render: RenderStats,
}

pub trait DesktopApplication {
    fn start(
        &mut self,
        window: &Window,
        renderer: &mut NativeRenderer,
        info: &DesktopInfo,
    ) -> Result<bool, RenderError>;

    // None exits without submitting the pending encoder or presenting the texture.
    fn draw(
        &mut self,
        window: &Window,
        renderer: &mut NativeRenderer,
        encoder: &mut wgpu::CommandEncoder,
        target: FrameTarget<'_>,
    ) -> Result<Option<RenderStats>, RenderError>;

    fn event(&mut self, _window: &Window, _event: &WindowEvent) -> Result<(), RenderError> {
        Ok(())
    }

    fn presented(&mut self, _frame: PresentedFrame) -> Result<(), RenderError> {
        Ok(())
    }
}

pub fn run_desktop(
    options: DesktopOptions,
    app: &mut impl DesktopApplication,
) -> Result<(u64, Vec<u64>), RenderError> {
    let event_loop = EventLoop::new().map_err(window_error)?;
    let mut handler = Desktop {
        options,
        app,
        context: None,
        redraw_after: None,
        error: None,
        presents: 0,
        last_present: None,
        gpu_ns: Vec::new(),
    };
    event_loop.run_app(&mut handler).map_err(window_error)?;
    match handler.error {
        Some(error) => Err(error),
        None => Ok((handler.presents, handler.gpu_ns)),
    }
}

struct Context {
    surface: wgpu::Surface<'static>,
    renderer: NativeRenderer,
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    window: Arc<Window>,
    config: wgpu::SurfaceConfiguration,
    depth: Option<wgpu::TextureView>,
    size: PhysicalSize<u32>,
    occluded: bool,
    device_loss: Arc<Mutex<Option<String>>>,
    timings: Option<GpuTimings>,
}

impl Context {
    fn configure(&mut self, size: PhysicalSize<u32>) -> Result<(), RenderError> {
        self.size = size;
        if size.width == 0 || size.height == 0 {
            self.depth = None;
            return Ok(());
        }
        let device = self.renderer.device();
        let max_side = device.limits().max_texture_dimension_2d;
        if size.width > max_side || size.height > max_side {
            return Err(RenderError::Window(format!(
                "physical size {}x{} exceeds device limit {max_side}",
                size.width, size.height
            )));
        }
        self.config.width = size.width;
        self.config.height = size.height;
        self.surface.configure(device, &self.config);
        self.depth = Some(depth_target(device, size.width, size.height));
        Ok(())
    }

    fn replace_surface(&mut self) -> Result<(), RenderError> {
        let surface = self
            .instance
            .create_surface(self.window.clone())
            .map_err(window_error)?;
        let caps = surface.get_capabilities(&self.adapter);
        if !caps.formats.contains(&self.config.format)
            || !caps.present_modes.contains(&self.config.present_mode)
            || !caps.alpha_modes.contains(&self.config.alpha_mode)
        {
            return Err(RenderError::Window(
                "replacement surface is incompatible with the active GPU configuration".into(),
            ));
        }
        self.surface = surface;
        self.configure(self.window.inner_size())
    }
}

// Reuse two query slots in queue order and map the compact results once
// after presentation ends, outside the interval being measured.
struct GpuTimings {
    queries: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
    limit: u64,
    period: f64,
}

impl GpuTimings {
    const FRAME_BYTES: u64 = 16;

    fn new(renderer: &NativeRenderer, limit: u64) -> Option<Self> {
        let device = renderer.device();
        if !device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return None;
        }
        let queries = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("desktop render queries"),
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        });
        let resolve = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("desktop render query resolve"),
            size: Self::FRAME_BYTES,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("desktop render query readback"),
            size: limit * Self::FRAME_BYTES,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Some(Self {
            queries,
            resolve,
            readback,
            limit,
            period: renderer.queue().get_timestamp_period().into(),
        })
    }

    fn finish(&self, device: &wgpu::Device, count: u64) -> Result<Vec<u64>, RenderError> {
        let count = count.min(self.limit);
        if count == 0 {
            return Ok(Vec::new());
        }
        let slice = self.readback.slice(..count * Self::FRAME_BYTES);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|error| RenderError::Poll(error.to_string()))?;
        receiver
            .recv()
            .map_err(|_| RenderError::ReadbackChannel)?
            .map_err(|error| RenderError::Readback(error.to_string()))?;
        let mapped = slice.get_mapped_range();
        let stamps: &[[u64; 2]] = bytemuck::cast_slice(&mapped);
        let samples = stamps
            .iter()
            .map(|pair| (pair[1].wrapping_sub(pair[0]) as f64 * self.period).round() as u64)
            .collect();
        drop(mapped);
        self.readback.unmap();
        Ok(samples)
    }
}

struct Desktop<'a, A> {
    options: DesktopOptions,
    app: &'a mut A,
    context: Option<Context>,
    redraw_after: Option<Instant>,
    error: Option<RenderError>,
    presents: u64,
    last_present: Option<Instant>,
    gpu_ns: Vec<u64>,
}

impl<A: DesktopApplication> Desktop<'_, A> {
    fn open(&mut self, loop_: &ActiveEventLoop) -> Result<bool, RenderError> {
        let primary = loop_.primary_monitor();
        let handles: Vec<_> = loop_.available_monitors().collect();
        let monitors: Vec<_> = handles
            .iter()
            .map(|handle| describe(handle, primary.as_ref()))
            .collect();
        let selected_monitor = select_monitor(&monitors, self.options.monitor.as_deref())?;
        let fullscreen = self
            .options
            .fullscreen
            .then(|| Fullscreen::Borderless(selected_monitor.map(|index| handles[index].clone())));
        let window = Arc::new(
            loop_
                .create_window(
                    Window::default_attributes()
                        .with_title(&self.options.title)
                        .with_inner_size(self.options.physical_px)
                        .with_fullscreen(fullscreen),
                )
                .map_err(window_error)?,
        );
        let instance = Gpu::instance();
        let surface = instance
            .create_surface(window.clone())
            .map_err(window_error)?;
        let adapter = Gpu::compatible_adapter(&instance, Some(&surface))?;
        let caps = surface.get_capabilities(&adapter);
        let Some(format) = caps
            .formats
            .iter()
            .copied()
            .find(wgpu::TextureFormat::is_srgb)
        else {
            return Err(RenderError::Window(
                "display has no sRGB surface format".into(),
            ));
        };
        if !caps.present_modes.contains(&self.options.present_mode) {
            return Err(RenderError::Window(format!(
                "surface does not support {:?} presentation",
                self.options.present_mode
            )));
        }
        let mut config = surface
            .get_default_config(&adapter, 1, 1)
            .ok_or_else(|| RenderError::Window("display has no surface configuration".into()))?;
        config.format = format;
        config.present_mode = self.options.present_mode;
        let gpu = Gpu::from_adapter(&adapter, self.options.renderer.allow_software)?;
        let renderer = NativeRenderer::with_gpu(gpu, format, self.options.renderer.max_instances);
        let device_loss = Arc::new(Mutex::new(None));
        renderer.device().set_device_lost_callback({
            let device_loss = Arc::clone(&device_loss);
            move |reason, message| {
                *device_loss.lock().expect("device loss lock") =
                    Some(format!("{reason:?}: {message}"));
            }
        });
        let timings = self
            .options
            .sample_limit
            .and_then(|limit| GpuTimings::new(&renderer, limit));
        let size = window.inner_size();
        let mut context = Context {
            surface,
            renderer,
            instance,
            adapter,
            window,
            config,
            depth: None,
            size,
            occluded: false,
            device_loss,
            timings,
        };
        context.configure(size)?;
        let info = DesktopInfo {
            monitors,
            selected_monitor,
            physical_px: size,
            scale_factor: context.window.scale_factor(),
            format,
            present_mode: context.config.present_mode,
        };
        let keep_running = self
            .app
            .start(&context.window, &mut context.renderer, &info)?;
        self.context = Some(context);
        Ok(keep_running)
    }

    fn redraw(&mut self) -> Result<bool, RenderError> {
        let Some(context) = self.context.as_mut() else {
            return Ok(true);
        };
        if let Some(reason) = context.device_loss.lock().expect("device loss lock").take() {
            return Err(RenderError::Device(format!("active device lost: {reason}")));
        }
        let Some(depth) = context.depth.as_ref() else {
            return Ok(true);
        };
        let acquired_at = Instant::now();
        let (texture, suboptimal) = match context.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture) => (texture, false),
            wgpu::CurrentSurfaceTexture::Suboptimal(texture) => (texture, true),
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                self.redraw_after = Some(Instant::now() + Duration::from_millis(25));
                return Ok(true);
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                context.configure(context.window.inner_size())?;
                return Ok(true);
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                context.replace_surface()?;
                return Ok(true);
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                return Err(RenderError::Window(
                    "surface texture acquisition failed validation".into(),
                ));
            }
        };
        let acquire = acquired_at.elapsed();
        let color = texture.texture.create_view(&Default::default());
        let timing = context
            .timings
            .as_ref()
            .filter(|timing| self.presents < timing.limit);
        let target = FrameTarget {
            color: &color,
            depth,
            width: context.size.width,
            height: context.size.height,
            clear: context.renderer.background(),
            positions: None,
            timestamps: timing.map(|timing| wgpu::RenderPassTimestampWrites {
                query_set: &timing.queries,
                beginning_of_pass_write_index: Some(0),
                end_of_pass_write_index: Some(1),
            }),
        };
        let mut encoder = context
            .renderer
            .device()
            .create_command_encoder(&Default::default());
        let Some(render) =
            self.app
                .draw(&context.window, &mut context.renderer, &mut encoder, target)?
        else {
            return Ok(false);
        };
        if let Some(timing) = timing {
            encoder.resolve_query_set(&timing.queries, 0..2, &timing.resolve, 0);
            encoder.copy_buffer_to_buffer(
                &timing.resolve,
                0,
                &timing.readback,
                self.presents * GpuTimings::FRAME_BYTES,
                GpuTimings::FRAME_BYTES,
            );
        }
        let submitted_at = Instant::now();
        context.renderer.queue().submit([encoder.finish()]);
        let submit = submitted_at.elapsed();
        drop(color);
        if self.options.suppress_present {
            drop(texture);
            return Ok(true);
        }
        context.window.pre_present_notify();
        let presenting_at = Instant::now();
        texture.present();
        let present = presenting_at.elapsed();
        self.presents += 1;
        let now = Instant::now();
        let wall_interval = self.last_present.replace(now).map(|last| now - last);
        self.app.presented(PresentedFrame {
            sequence: self.presents,
            physical_px: context.size,
            scale_factor: context.window.scale_factor(),
            wall_interval,
            acquire,
            submit,
            present,
            render,
        })?;
        if self.presents == 3 {
            if let Some(size) = self.options.resize_after_three {
                if let Some(applied_size) = context.window.request_inner_size(size) {
                    context.configure(applied_size)?;
                }
            }
        }
        if suboptimal {
            context.configure(context.window.inner_size())?;
        }
        Ok(true)
    }

    fn close(&mut self, loop_: &ActiveEventLoop, error: Option<RenderError>) {
        self.error = error;
        if let Some(context) = self.context.take() {
            if let (None, Some(timings)) = (&self.error, context.timings) {
                match timings.finish(context.renderer.device(), self.presents) {
                    Ok(samples) => self.gpu_ns = samples,
                    Err(error) => self.error = Some(error),
                }
            }
        }
        loop_.exit();
    }
}

impl<A: DesktopApplication> ApplicationHandler for Desktop<'_, A> {
    fn resumed(&mut self, loop_: &ActiveEventLoop) {
        if self.context.is_none() {
            match self.open(loop_) {
                Ok(true) => {}
                Ok(false) => self.close(loop_, None),
                Err(error) => self.close(loop_, Some(error)),
            }
        }
    }

    fn window_event(&mut self, loop_: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        let Some(context) = self.context.as_mut().filter(|ctx| ctx.window.id() == id) else {
            return;
        };
        let result = self.app.event(&context.window, &event);
        let result = result.and_then(|()| match event {
            WindowEvent::CloseRequested | WindowEvent::Destroyed => Ok(false),
            WindowEvent::Resized(size) => {
                if size != context.size {
                    context.configure(size)?;
                }
                Ok(true)
            }
            WindowEvent::ScaleFactorChanged { .. } => {
                let size = context.window.inner_size();
                if size != context.size {
                    context.configure(size)?;
                }
                Ok(true)
            }
            WindowEvent::Occluded(occluded) => {
                context.occluded = occluded;
                Ok(true)
            }
            _ => Ok(true),
        });
        let result = if matches!(event, WindowEvent::RedrawRequested) {
            result.and_then(|_| self.redraw())
        } else {
            result
        };
        match result {
            Ok(true) => {}
            Ok(false) => self.close(loop_, None),
            Err(error) => self.close(loop_, Some(error)),
        }
    }

    fn about_to_wait(&mut self, loop_: &ActiveEventLoop) {
        loop_.set_control_flow(ControlFlow::Wait);
        let Some(context) = self.context.as_ref() else {
            return;
        };
        if context.occluded || context.depth.is_none() {
            return;
        }
        if let Some(deadline) = self
            .redraw_after
            .take()
            .filter(|&until| Instant::now() < until)
        {
            self.redraw_after = Some(deadline);
            loop_.set_control_flow(ControlFlow::WaitUntil(deadline));
            return;
        }
        context.window.request_redraw();
    }
}

fn describe(handle: &MonitorHandle, primary: Option<&MonitorHandle>) -> Monitor {
    Monitor {
        name: handle.name().unwrap_or_default(),
        physical_px: handle.size(),
        scale_factor: handle.scale_factor(),
        refresh_millihertz: handle.refresh_rate_millihertz(),
        primary: primary == Some(handle),
    }
}

fn select_monitor(
    monitors: &[Monitor],
    requested: Option<&str>,
) -> Result<Option<usize>, RenderError> {
    let Some(requested) = requested.filter(|name| !name.is_empty()) else {
        return Ok(monitors
            .iter()
            .position(|m| m.primary)
            .or((!monitors.is_empty()).then_some(0)));
    };
    let name = requested.trim().to_lowercase();
    let selected = match name.parse::<usize>() {
        Ok(index) => (index < monitors.len()).then_some(index),
        Err(_) => monitors
            .iter()
            .position(|monitor| monitor.name.to_lowercase().contains(&name)),
    };
    selected.map(Some).ok_or_else(|| {
        let names = monitors
            .iter()
            .enumerate()
            .map(|(index, monitor)| format!("{index}: {}", monitor.name))
            .collect::<Vec<_>>();
        RenderError::Window(format!(
            "no monitor matches {requested:?}; available monitors: {}",
            names.join(", ")
        ))
    })
}

fn window_error(error: impl std::fmt::Display) -> RenderError {
    RenderError::Window(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn monitor_selection_uses_primary_index_or_partial_name() {
        let screen = |name: &str, primary| Monitor {
            name: name.into(),
            physical_px: PhysicalSize::new(1920, 1080),
            scale_factor: 1.0,
            refresh_millihertz: Some(60_000),
            primary,
        };
        let screens = [screen("Desk", false), screen("Internal panel", true)];
        assert_eq!(select_monitor(&screens, None).unwrap(), Some(1));
        assert_eq!(select_monitor(&screens, Some("0")).unwrap(), Some(0));
        assert_eq!(select_monitor(&screens, Some("PANEL")).unwrap(), Some(1));
        assert!(select_monitor(&screens, Some("7")).is_err());
        assert!(select_monitor(&screens, Some("Unknown")).is_err());
    }
}
