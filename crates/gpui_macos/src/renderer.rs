use crate::metal_renderer;
use anyhow::{Context as _, Result, ensure};
use cocoa::base::{YES, id, nil};
use gpui::{
    CachedHardwareRendererInitializationError, DevicePixels, HardwareRendererInitializationError,
    PlatformAtlas, RendererFallbackReason, RendererPreference, RendererSelection, RenderingInfo,
    Rgba, Scene, Size, select_renderer,
};
use gpui_software::{SoftwareAtlas, SoftwarePresenter, SoftwareRenderer};
#[cfg(any(test, feature = "test-support"))]
use image::RgbaImage;
use objc::{msg_send, sel, sel_impl};
use parking_lot::Mutex;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::{ffi::c_void, ptr::NonNull, sync::Arc};

#[derive(Clone, Default)]
pub(crate) struct Context {
    metal: metal_renderer::Context,
    hardware_device: Arc<Mutex<Option<metal::Device>>>,
    hardware_failure: Arc<Mutex<Option<CachedHardwareRendererInitializationError>>>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct AppKitWindow {
    view: NonNull<c_void>,
}

impl HasWindowHandle for AppKitWindow {
    fn window_handle(
        &self,
    ) -> std::result::Result<raw_window_handle::WindowHandle<'_>, raw_window_handle::HandleError>
    {
        let handle = raw_window_handle::AppKitWindowHandle::new(self.view);
        // MacWindow retains the NSView until after the presenter is destroyed.
        Ok(unsafe { raw_window_handle::WindowHandle::borrow_raw(handle.into()) })
    }
}

impl HasDisplayHandle for AppKitWindow {
    fn display_handle(
        &self,
    ) -> std::result::Result<raw_window_handle::DisplayHandle<'_>, raw_window_handle::HandleError>
    {
        Ok(raw_window_handle::DisplayHandle::appkit())
    }
}

pub(crate) enum Renderer {
    Hardware {
        renderer: metal_renderer::MetalRenderer,
        rendering_info: RenderingInfo,
    },
    Software {
        renderer: SoftwareRenderer,
        presenter: Option<SoftwarePresenter<AppKitWindow, AppKitWindow>>,
        atlas: Arc<SoftwareAtlas>,
        backing_layer: id,
        rendering_info: RenderingInfo,
    },
}

pub(crate) unsafe fn new_renderer(
    context: Context,
    _native_window: *mut c_void,
    native_view: *mut c_void,
    _bounds: Size<f32>,
    transparent: bool,
    preference: RendererPreference,
) -> Result<Renderer> {
    let metal_context = context.metal;
    let hardware_device = context.hardware_device;
    let hardware_failure = context.hardware_failure;
    let selection = select_renderer(
        preference,
        || {
            if let Some(error) = hardware_failure.lock().as_ref() {
                return Err(error.to_error());
            }
            let renderer = match (|| {
                let device = if let Some(device) = hardware_device.lock().clone() {
                    device
                } else {
                    let device =
                        metal_renderer::MetalRenderer::create_device().map_err(|error| {
                            HardwareRendererInitializationError::new(
                                RendererFallbackReason::NoHardwareAdapter,
                                error,
                            )
                        })?;
                    hardware_device.lock().replace(device.clone());
                    device
                };
                metal_renderer::MetalRenderer::new_with_device(device, metal_context, transparent)
                    .map_err(|error| {
                        HardwareRendererInitializationError::new(
                            RendererFallbackReason::DeviceInitialization,
                            error.context("creating Metal renderer"),
                        )
                    })
            })() {
                Ok(renderer) => renderer,
                Err(error) => {
                    if matches!(
                        error.reason,
                        RendererFallbackReason::NoHardwareAdapter
                            | RendererFallbackReason::DeviceInitialization
                    ) {
                        hardware_failure
                            .lock()
                            .replace(CachedHardwareRendererInitializationError::new(&error));
                    }
                    return Err(error);
                }
            };
            let gpu_specs = renderer.gpu_specs();
            Ok((renderer, Some(gpu_specs)))
        },
        || {
            unsafe {
                let _: () = msg_send![native_view as id, setWantsLayer: YES];
            }
            let view = NonNull::new(native_view).context("AppKit software view is null")?;
            let window = AppKitWindow { view };
            let presenter = SoftwarePresenter::new(window, window)
                .context("creating AppKit software presenter")?;
            let backing_layer: id = unsafe { msg_send![native_view as id, layer] };
            ensure!(
                backing_layer != nil,
                "AppKit software view has no backing layer"
            );
            let atlas = Arc::new(SoftwareAtlas::new());
            let renderer = SoftwareRenderer::new(atlas.clone());
            Ok((renderer, presenter, atlas, backing_layer))
        },
    )?;

    match selection {
        RendererSelection::Hardware {
            renderer,
            rendering_info,
        } => Ok(Renderer::Hardware {
            renderer,
            rendering_info,
        }),
        RendererSelection::Software {
            renderer: (renderer, presenter, atlas, backing_layer),
            rendering_info,
        } => Ok(Renderer::Software {
            renderer,
            presenter: Some(presenter),
            atlas,
            backing_layer,
            rendering_info,
        }),
    }
}

impl Renderer {
    pub(crate) fn layer(&self) -> Option<id> {
        match self {
            Self::Hardware { renderer, .. } => {
                let layer = renderer.layer_ptr() as id;
                (layer != nil).then_some(layer)
            }
            Self::Software { backing_layer, .. } => Some(*backing_layer),
        }
    }

    pub(crate) fn layer_ptr(&self) -> id {
        self.layer().unwrap_or(nil)
    }

    pub(crate) fn sprite_atlas(&self) -> Arc<dyn PlatformAtlas> {
        match self {
            Self::Hardware { renderer, .. } => renderer.sprite_atlas().clone(),
            Self::Software { atlas, .. } => atlas.clone(),
        }
    }

    pub(crate) fn set_presents_with_transaction(&mut self, value: bool) {
        if let Self::Hardware { renderer, .. } = self {
            renderer.set_presents_with_transaction(value);
        }
    }

    pub(crate) fn update_drawable_size(&mut self, size: Size<DevicePixels>) {
        match self {
            Self::Hardware { renderer, .. } => renderer.update_drawable_size(size),
            Self::Software { renderer, .. } => {
                if renderer.size() != size {
                    renderer.invalidate();
                }
            }
        }
    }

    pub(crate) fn update_transparency(&mut self, transparent: bool) {
        if let Self::Hardware { renderer, .. } = self {
            renderer.update_transparency(transparent);
        }
    }

    pub(crate) fn destroy(&mut self) {
        match self {
            Self::Hardware { renderer, .. } => renderer.destroy(),
            Self::Software { presenter, .. } => {
                presenter.take();
            }
        }
    }

    pub(crate) fn draw(
        &mut self,
        scene: &Scene,
        size: Size<DevicePixels>,
        scale_factor: f32,
    ) -> Result<()> {
        match self {
            Self::Hardware { renderer, .. } => {
                renderer.draw(scene);
                Ok(())
            }
            Self::Software {
                renderer,
                presenter,
                ..
            } => {
                let frame =
                    renderer.render_frame(scene, size, scale_factor, opaque_background())?;
                presenter
                    .as_mut()
                    .context("software presenter was released before window destruction")?
                    .present(frame)?;
                Ok(())
            }
        }
    }

    #[cfg(any(test, feature = "test-support"))]
    pub(crate) fn render_to_image(
        &mut self,
        scene: &Scene,
        size: Size<DevicePixels>,
        scale_factor: f32,
    ) -> Result<RgbaImage> {
        match self {
            Self::Hardware { renderer, .. } => renderer.render_to_image(scene),
            Self::Software { renderer, .. } => {
                let frame =
                    renderer.render_frame(scene, size, scale_factor, opaque_background())?;
                let byte_count = frame
                    .framebuffer
                    .len()
                    .checked_mul(4)
                    .context("software screenshot byte length overflowed")?;
                let mut bytes = Vec::new();
                bytes
                    .try_reserve_exact(byte_count)
                    .context("allocating software screenshot")?;
                for pixel in frame.framebuffer {
                    bytes.extend_from_slice(&[
                        ((pixel >> 16) & 0xff) as u8,
                        ((pixel >> 8) & 0xff) as u8,
                        (pixel & 0xff) as u8,
                        0xff,
                    ]);
                }
                let image = RgbaImage::from_raw(frame.size[0], frame.size[1], bytes)
                    .context("software screenshot dimensions do not match its pixels")?;
                renderer.invalidate();
                Ok(image)
            }
        }
    }

    pub(crate) fn rendering_info(&self) -> RenderingInfo {
        match self {
            Self::Hardware { rendering_info, .. } | Self::Software { rendering_info, .. } => {
                rendering_info.clone()
            }
        }
    }

    pub(crate) fn is_software(&self) -> bool {
        matches!(self, Self::Software { .. })
    }
}

fn opaque_background() -> Rgba {
    Rgba {
        r: 1.0,
        g: 1.0,
        b: 1.0,
        a: 1.0,
    }
}
