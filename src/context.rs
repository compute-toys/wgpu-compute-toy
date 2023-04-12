use std::sync::Arc;

use crate::utils::set_panic_hook;

pub struct WgpuContext {
    pub event_loop: Option<winit::event_loop::EventLoop<()>>,
    pub window: winit::window::Window,
    pub device: Arc<wgpu::Device>,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface,
    pub surface_format: wgpu::TextureFormat,
}

#[cfg(target_arch = "wasm32")]
fn init_window(
    size: winit::dpi::Size,
    event_loop: &winit::event_loop::EventLoop<()>,
    bind_id: &str,
) -> Result<winit::window::Window, Box<dyn std::error::Error>> {
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
    use winit::platform::web::WindowBuilderExtWebSys;
    let window = winit::window::WindowBuilder::new()
        .with_inner_size(size)
        .with_canvas(Some(canvas))
        .build(event_loop)?;
    Ok(window)
}

#[cfg(not(target_arch = "wasm32"))]
fn init_window(
    size: winit::dpi::Size,
    event_loop: &winit::event_loop::EventLoop<()>,
    _: &str,
) -> Result<winit::window::Window, Box<dyn std::error::Error>> {
    env_logger::init_from_env(
        env_logger::Env::default().filter_or(env_logger::DEFAULT_FILTER_ENV, "info"),
    );
    let window = winit::window::WindowBuilder::new()
        .with_inner_size(size)
        .build(event_loop)?;
    Ok(window)
}

pub async fn init_wgpu(width: u32, height: u32, bind_id: &str) -> Result<WgpuContext, String> {
    let size = winit::dpi::Size::Physical(winit::dpi::PhysicalSize::new(width, height));
    let event_loop = winit::event_loop::EventLoop::new();
    let window = init_window(size, &event_loop, bind_id).map_err(|e| e.to_string())?;

    let instance = wgpu::Instance::default();
    let surface = unsafe { instance.create_surface(&window) }.map_err(|e| e.to_string())?;

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: Default::default(),
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
        })
        .await
        .ok_or("unable to create adapter")?;
    let (device, queue) = adapter
        .request_device(&Default::default(), None)
        .await
        .map_err(|e| e.to_string())?;

    let surface_format = surface.get_capabilities(&adapter).formats[0];

    surface.configure(
        &device,
        &wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo, // vsync
            alpha_mode: wgpu::CompositeAlphaMode::Opaque,
            view_formats: vec![],
        },
    );
    log::info!("adapter.limits = {:#?}", adapter.limits());
    Ok(WgpuContext {
        event_loop: Some(event_loop),
        window,
        device: Arc::new(device),
        queue,
        surface,
        surface_format,
    })
}
