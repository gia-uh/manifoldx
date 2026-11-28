//! Calling-thread desktop event loop. Window, surface and device stay together.

use std::sync::atomic::{AtomicBool, Ordering};
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

/// One presented frame, copied after every pass (post and GUI included) and
/// before it reaches the surface. Rows are tightly packed.
#[derive(Clone, Debug)]
pub struct PresentedImage {
    pub width: u32,
    pub height: u32,
    pub bgra: bool,
    pub pixels: Vec<u8>,
}

/// The dev channel's request/answer slot for `manifoldx look`.
///
/// A normal frame never touches the GPU for it: the desktop loop only copies
/// the acquired surface texture when `request` was set, and hands the pixels
/// back through this slot after the frame that was presented.
#[derive(Default)]
pub struct PresentedCapture {
    requested: AtomicBool,
    supported: AtomicBool,
    ready: Mutex<Option<Result<PresentedImage, String>>>,
}

impl PresentedCapture {
    pub fn request(&self) {
        self.requested.store(true, Ordering::Release);
    }

    pub fn supported(&self) -> bool {
        self.supported.load(Ordering::Acquire)
    }

    pub fn set_supported(&self, supported: bool) {
        self.supported.store(supported, Ordering::Release);
    }

    /// The image of the last requested frame, once its copy came back.
    pub fn take(&self) -> Option<Result<PresentedImage, String>> {
        self.ready.lock().expect("presented capture lock").take()
    }

    fn take_request(&self) -> bool {
        self.requested.swap(false, Ordering::AcqRel)
    }

    fn deliver(&self, image: PresentedImage) {
        *self.ready.lock().expect("presented capture lock") = Some(Ok(image));
    }

    fn fail(&self, error: String) {
        *self.ready.lock().expect("presented capture lock") = Some(Err(error));
    }
}

pub struct DesktopOptions {
    pub title: String,
    pub physical_px: PhysicalSize<u32>,
    pub fullscreen: bool,
    pub monitor: Option<String>,
    pub present_mode: wgpu::PresentMode,
    pub renderer: RendererOptions,
    pub sample_limit: Option<u64>,
    pub gpu_timings: bool,
    pub suppress_present: bool,
    pub resize_after_three: Option<PhysicalSize<u32>>,
    pub presented: Option<Arc<PresentedCapture>>,
}

/// Why the bounded desktop diagnostic did or did not write GPU timestamps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuTimingState {
    /// A normal run: the frame loop never bounds itself, so it never measures.
    NotBounded,
    /// The caller turned timings off (`MANIFOLDX_GPU_TIMINGS=0`).
    Disabled,
    /// The bounded run asked for timings, but the adapter has no timestamp queries.
    Unavailable,
    /// Every presented frame wrote one timestamp pair.
    Measured,
}

impl GpuTimingState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NotBounded => "not_bounded",
            Self::Disabled => "disabled",
            Self::Unavailable => "unavailable",
            Self::Measured => "measured",
        }
    }
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
        adapter: &wgpu::Adapter,
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
) -> Result<(u64, Vec<u64>, GpuTimingState), RenderError> {
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
        gpu_state: GpuTimingState::NotBounded,
        look: None,
    };
    event_loop.run_app(&mut handler).map_err(window_error)?;
    match handler.error {
        Some(error) => Err(error),
        None => Ok((handler.presents, handler.gpu_ns, handler.gpu_state)),
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
    copy_src: bool,
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
        self.config.usage = if self.copy_src {
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC
        } else {
            wgpu::TextureUsages::RENDER_ATTACHMENT
        };
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
        // A replacement surface may not allow copying its images back.
        self.copy_src = caps.usages.contains(wgpu::TextureUsages::COPY_SRC);
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
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(5)),
            })
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

/// The copy of one presented surface texture, submitted with its frame and
/// mapped at the start of the next redraw so no frame waits on the GPU.
struct LookReadback {
    buffer: wgpu::Buffer,
    width: u32,
    height: u32,
    row: u32,
    bgra: bool,
}

impl LookReadback {
    fn finish(self, device: &wgpu::Device) -> Result<PresentedImage, String> {
        let slice = self.buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(5)),
            })
            .map_err(|error| error.to_string())?;
        receiver
            .recv()
            .map_err(|_| "the frame readback channel closed".to_string())?
            .map_err(|error| error.to_string())?;
        let mapped = slice.get_mapped_range();
        let mut pixels = Vec::with_capacity(self.width as usize * self.height as usize * 4);
        for line in mapped.chunks_exact(self.row as usize) {
            pixels.extend_from_slice(&line[..self.width as usize * 4]);
        }
        drop(mapped);
        self.buffer.unmap();
        Ok(PresentedImage {
            width: self.width,
            height: self.height,
            bgra: self.bgra,
            pixels,
        })
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
    gpu_state: GpuTimingState,
    look: Option<LookReadback>,
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
        // Copying the presented image is what `manifoldx look` reads. Only
        // request it when the surface advertises it, so a surface that cannot
        // still draws and the channel answers 501 instead.
        let copy_src = caps.usages.contains(wgpu::TextureUsages::COPY_SRC);
        config.usage = if copy_src {
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC
        } else {
            wgpu::TextureUsages::RENDER_ATTACHMENT
        };
        if let Some(presented) = &self.options.presented {
            presented.set_supported(copy_src);
        }
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
        let timings = match self.options.sample_limit {
            None => {
                self.gpu_state = GpuTimingState::NotBounded;
                None
            }
            Some(_) if !self.options.gpu_timings => {
                self.gpu_state = GpuTimingState::Disabled;
                None
            }
            Some(limit) => match GpuTimings::new(&renderer, limit) {
                Some(timings) => {
                    self.gpu_state = GpuTimingState::Measured;
                    Some(timings)
                }
                None => {
                    self.gpu_state = GpuTimingState::Unavailable;
                    None
                }
            },
        };
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
            copy_src,
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
        let keep_running = self.app.start(
            &context.window,
            &mut context.renderer,
            &info,
            &context.adapter,
        )?;
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
        // The copy submitted with the last presented frame is ready: map it
        // here, before this frame's barrier, so `manifoldx look` answers.
        if let Some(look) = self.look.take() {
            let result = look.finish(context.renderer.device());
            if let Some(presented) = &self.options.presented {
                match result {
                    Ok(image) => presented.deliver(image),
                    Err(error) => presented.fail(error),
                }
            }
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
                if let Some(presented) = &self.options.presented {
                    presented.set_supported(context.copy_src);
                    if !context.copy_src {
                        presented
                            .fail("the replacement surface cannot copy the presented frame".into());
                    }
                }
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
        // Only the frame a look asked for pays for the copy; every other frame
        // goes straight to the surface.
        if context.copy_src
            && self
                .options
                .presented
                .as_ref()
                .is_some_and(|presented| presented.take_request())
        {
            let width = context.size.width;
            let height = context.size.height;
            let row = (width * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
                * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
            let buffer = context
                .renderer
                .device()
                .create_buffer(&wgpu::BufferDescriptor {
                    label: Some("manifoldx presented look"),
                    size: u64::from(row) * u64::from(height),
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(row),
                        rows_per_image: Some(height),
                    },
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
            self.look = Some(LookReadback {
                buffer,
                width,
                height,
                row,
                bgra: matches!(
                    context.config.format,
                    wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
                ),
            });
        }
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
        if self.options.suppress_present && self.options.sample_limit == Some(self.presents + 1) {
            // The diagnostic submits the final frame but never presents it.
            // Exit without another acquisition on the discarded swapchain image.
            drop(texture);
            return Ok(false);
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
        // A bounded session has completed. Do not acquire a texture the
        // scheduler would immediately discard on a follow-up redraw.
        if self.options.sample_limit == Some(self.presents) {
            return Ok(false);
        }
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
