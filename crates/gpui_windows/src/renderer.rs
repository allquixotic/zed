use crate::{DirectXDevices, DirectXRenderer, SafeHwnd};
use anyhow::{Context as _, Result};
use gpui::{
    DevicePixels, HardwareAvailability, PlatformAtlas, RendererPreference, RenderingInfo, Rgba,
    Scene, Size, WindowBackgroundAppearance,
};
use gpui_software::{SoftwareAtlas, SoftwarePresenter, SoftwareRenderer};
use gpui_util::ResultExt as _;
use std::sync::Arc;
use windows::Win32::Foundation::HWND;

#[derive(Clone)]
pub(crate) enum WindowsRendererConfig {
    Hardware(DirectXDevices),
    Software,
}

pub(crate) enum WindowsRenderer {
    Hardware {
        renderer: DirectXRenderer,
        rendering_info: RenderingInfo,
    },
    Software {
        renderer: SoftwareRenderer,
        presenter: Option<SoftwarePresenter<SafeHwnd, SafeHwnd>>,
        atlas: Arc<SoftwareAtlas>,
        rendering_info: RenderingInfo,
    },
}

impl WindowsRenderer {
    pub(crate) fn new(
        hwnd: HWND,
        config: WindowsRendererConfig,
        disable_direct_composition: bool,
    ) -> Result<Self> {
        match config {
            WindowsRendererConfig::Hardware(devices) => {
                let renderer = DirectXRenderer::new(hwnd, &devices, disable_direct_composition)
                    .context("creating DirectX renderer")?;
                let rendering_info = RenderingInfo::hardware(renderer.gpu_specs().log_err());
                Ok(Self::Hardware {
                    renderer,
                    rendering_info,
                })
            }
            WindowsRendererConfig::Software => {
                let atlas = Arc::new(SoftwareAtlas::new());
                let renderer = SoftwareRenderer::new(atlas.clone());
                let handle = SafeHwnd::from(hwnd);
                let presenter = SoftwarePresenter::new(handle, handle)
                    .context("creating Win32 software presenter")?;
                Ok(Self::Software {
                    renderer,
                    presenter: Some(presenter),
                    atlas,
                    rendering_info: RenderingInfo::software(
                        RendererPreference::Software,
                        HardwareAvailability::NotProbed,
                        None,
                    ),
                })
            }
        }
    }

    pub(crate) fn draw(
        &mut self,
        scene: &Scene,
        background_appearance: WindowBackgroundAppearance,
        size: Size<DevicePixels>,
        scale_factor: f32,
    ) -> Result<()> {
        match self {
            Self::Hardware { renderer, .. } => renderer.draw(scene, background_appearance),
            Self::Software {
                renderer,
                presenter,
                ..
            } => {
                let presenter = presenter
                    .as_mut()
                    .context("software presenter was released before window destruction")?;
                let frame = renderer.render_frame(
                    scene,
                    size,
                    scale_factor,
                    Rgba {
                        r: 1.0,
                        g: 1.0,
                        b: 1.0,
                        a: 1.0,
                    },
                )?;
                presenter.present(frame)?;
                Ok(())
            }
        }
    }

    pub(crate) fn resize(&mut self, size: Size<DevicePixels>) -> Result<()> {
        match self {
            Self::Hardware { renderer, .. } => renderer.resize(size),
            Self::Software { renderer, .. } => {
                renderer.invalidate();
                Ok(())
            }
        }
    }

    pub(crate) fn sprite_atlas(&self) -> Arc<dyn PlatformAtlas> {
        match self {
            Self::Hardware { renderer, .. } => renderer.sprite_atlas(),
            Self::Software { atlas, .. } => atlas.clone(),
        }
    }

    pub(crate) fn rendering_info(&self) -> RenderingInfo {
        match self {
            Self::Hardware { rendering_info, .. } | Self::Software { rendering_info, .. } => {
                rendering_info.clone()
            }
        }
    }

    pub(crate) fn handle_device_lost(&mut self, devices: &DirectXDevices) -> Result<()> {
        match self {
            Self::Hardware { renderer, .. } => renderer.handle_device_lost(devices),
            Self::Software { .. } => Ok(()),
        }
    }

    pub(crate) fn mark_drawable(&mut self) {
        if let Self::Hardware { renderer, .. } = self {
            renderer.mark_drawable();
        }
    }

    pub(crate) fn is_hardware(&self) -> bool {
        matches!(self, Self::Hardware { .. })
    }

    pub(crate) fn is_software(&self) -> bool {
        matches!(self, Self::Software { .. })
    }

    pub(crate) fn prepare_window_destroy(&mut self) {
        if let Self::Software { presenter, .. } = self {
            // softbuffer's Win32 backend releases a window DC and must be dropped before DestroyWindow.
            presenter.take();
        }
    }
}
