use std::sync::Arc;
use wgpu;

#[cfg(target_arch = "wasm32")]
use raw_window_handle::{
    HasRawDisplayHandle, HasRawWindowHandle, RawDisplayHandle, RawWindowHandle, WebDisplayHandle,
    WebWindowHandle,
};

pub struct WgpuContext {
    #[cfg(all(not(target_arch = "wasm32"), feature = "winit"))]
    pub event_loop: Option<winit::event_loop::EventLoop<()>>,
    #[cfg(all(not(target_arch = "wasm32"), feature = "winit"))]
    pub window: winit::window::Window,
    pub device: Arc<wgpu::Device>,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface<'static>,
    pub surface_config: wgpu::SurfaceConfiguration,
}

#[cfg(target_arch = "wasm32")]
struct CanvasWindow {
    id: u32,
}

#[cfg(target_arch = "wasm32")]
unsafe impl HasRawWindowHandle for CanvasWindow {
    fn raw_window_handle(&self) -> RawWindowHandle {
        let mut window_handle = WebWindowHandle::empty();
        window_handle.id = self.id;
        RawWindowHandle::Web(window_handle)
    }
}

#[cfg(target_arch = "wasm32")]
unsafe impl HasRawDisplayHandle for CanvasWindow {
    fn raw_display_handle(&self) -> RawDisplayHandle {
        RawDisplayHandle::Web(WebDisplayHandle::empty())
    }
}

#[cfg(target_arch = "wasm32")]
fn init_window(bind_id: &str) -> Result<CanvasWindow, Box<dyn std::error::Error>> {
    use crate::utils::set_panic_hook;
    console_log::init(); // FIXME only do this once
    set_panic_hook();
    let win = web_sys::window().ok_or("window is None")?;
    let doc = win.document().ok_or("document is None")?;
    let element = doc
        .get_element_by_id(bind_id)
        .ok_or(format!("cannot find element {bind_id}"))?;
    use wasm_bindgen::JsCast;
    let canvas = element
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .or(Err("cannot cast to canvas"))?;
    canvas
        .get_context("webgpu")
        .or(Err("no webgpu"))?
        .ok_or("no webgpu")?;
    canvas
        .set_attribute("data-raw-handle", "42")
        .or(Err("cannot set attribute"))?;
    Ok(CanvasWindow { id: 42 })
}

#[cfg(all(not(target_arch = "wasm32"), feature = "winit"))]
fn init_window(
    size: winit::dpi::Size,
    event_loop: &winit::event_loop::EventLoop<()>,
) -> Result<winit::window::Window, Box<dyn std::error::Error>> {
    env_logger::init_from_env(
        env_logger::Env::default().filter_or(env_logger::DEFAULT_FILTER_ENV, "info"),
    );
    let window = winit::window::WindowBuilder::new()
        .with_inner_size(size)
        .build(event_loop)?;
    Ok(window)
}

#[cfg(feature = "winit")]
pub async fn init_wgpu(width: u32, height: u32, bind_id: &str) -> Result<WgpuContext, String> {
    #[cfg(not(target_arch = "wasm32"))]
    let event_loop = winit::event_loop::EventLoop::new().map_err(|e| e.to_string())?;
    #[cfg(not(target_arch = "wasm32"))]
    let window = init_window(
        winit::dpi::Size::Physical(winit::dpi::PhysicalSize::new(width, height)),
        &event_loop,
    )
    .map_err(|e| e.to_string())?;

    #[cfg(target_arch = "wasm32")]
    let window = init_window(bind_id).map_err(|e| e.to_string())?;

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        dx12_shader_compiler: wgpu::Dx12Compiler::Fxc,
        flags: wgpu::InstanceFlags::default(),
        gles_minor_version: wgpu::Gles3MinorVersion::Automatic,
    });

    let surface = unsafe {
        instance.create_surface_unsafe(wgpu::SurfaceTargetUnsafe::from_window(&window).unwrap())
    }.map_err(|e| e.to_string())?;

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
        })
        .await
        .ok_or("unable to create adapter")?;
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("GPU Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        )
        .await
        .map_err(|e| e.to_string())?;

    let surface_format = preferred_framebuffer_format(&surface.get_capabilities(&adapter).formats);
    let surface_config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: surface_format,
        width,
        height,
        present_mode: wgpu::PresentMode::Fifo, // vsync
        alpha_mode: wgpu::CompositeAlphaMode::Opaque,
        view_formats: vec![
            surface_format.add_srgb_suffix(),
            surface_format.remove_srgb_suffix(),
        ],
        desired_maximum_frame_latency: 1,
    };
    surface.configure(&device, &surface_config);

    log::info!("adapter.limits = {:#?}", adapter.limits());
    Ok(WgpuContext {
        #[cfg(all(not(target_arch = "wasm32"), feature = "winit"))]
        event_loop: Some(event_loop),
        #[cfg(all(not(target_arch = "wasm32"), feature = "winit"))]
        window,
        device: Arc::new(device),
        queue,
        surface,
        surface_config,
    })
}

fn preferred_framebuffer_format(formats: &[wgpu::TextureFormat]) -> wgpu::TextureFormat {
    for &format in formats {
        if matches!(
            format,
            wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Bgra8Unorm
        ) {
            return format;
        }
    }
    formats[0]
}
