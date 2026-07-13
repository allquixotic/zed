use anyhow::{Context as _, Result};
use gpui::{
    DevicePixels, HardwareRendererInitializationError, PlatformAtlas, RendererPreference,
    RendererSelection, RenderingInfo, Rgba, Scene, Size, select_renderer,
};
use gpui_software::{SoftwareAtlas, SoftwarePresenter, SoftwareRenderer};
use gpui_wgpu::{CompositorGpuHint, GpuContext, WgpuRenderer, WgpuSurfaceConfig};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::{fmt::Debug, sync::Arc};

pub(crate) enum LinuxRenderer<W> {
    Hardware {
        renderer: WgpuRenderer,
        rendering_info: RenderingInfo,
    },
    Software {
        renderer: SoftwareRenderer,
        presenter: Option<SoftwarePresenter<W, W>>,
        atlas: Arc<SoftwareAtlas>,
        rendering_info: RenderingInfo,
    },
}

impl<W> LinuxRenderer<W>
where
    W: HasDisplayHandle + HasWindowHandle + Debug + Send + Sync + Clone + 'static,
{
    pub(crate) fn new(
        gpu_context: GpuContext,
        window: W,
        config: WgpuSurfaceConfig,
        compositor_gpu: Option<CompositorGpuHint>,
        preference: RendererPreference,
    ) -> Result<Self> {
        let software_window = window.clone();
        let selection = select_renderer(
            preference,
            || {
                let renderer =
                    WgpuRenderer::new_categorized(gpu_context, &window, config, compositor_gpu)
                        .map_err(|error| {
                            HardwareRendererInitializationError::new(
                                error.reason,
                                error.error.context("creating wgpu renderer"),
                            )
                        })?;
                let gpu_specs = renderer.gpu_specs();
                Ok((renderer, Some(gpu_specs)))
            },
            || {
                let atlas = Arc::new(SoftwareAtlas::new());
                let renderer = SoftwareRenderer::new(atlas.clone());
                let presenter = SoftwarePresenter::new(software_window.clone(), software_window)
                    .context("creating Linux software presenter")?;
                Ok((renderer, presenter, atlas))
            },
        )?;

        match selection {
            RendererSelection::Hardware {
                renderer,
                rendering_info,
            } => Ok(Self::Hardware {
                renderer,
                rendering_info,
            }),
            RendererSelection::Software {
                renderer: (renderer, presenter, atlas),
                rendering_info,
            } => Ok(Self::Software {
                renderer,
                presenter: Some(presenter),
                atlas,
                rendering_info,
            }),
        }
    }

    pub(crate) fn draw(
        &mut self,
        scene: &Scene,
        size: Size<DevicePixels>,
        scale_factor: f32,
        background: Rgba,
    ) -> Result<bool> {
        match self {
            Self::Hardware { renderer, .. } => Ok(renderer.draw(scene)),
            Self::Software {
                renderer,
                presenter,
                ..
            } => {
                let frame = renderer.render_frame(scene, size, scale_factor, background)?;
                presenter
                    .as_mut()
                    .context("software presenter was released before window destruction")?
                    .present(frame)
            }
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

    pub(crate) fn set_subpixel_layout(&mut self, is_bgr: bool) {
        match self {
            Self::Hardware { renderer, .. } => renderer.set_subpixel_layout(is_bgr),
            Self::Software { renderer, .. } => {
                let mut parameters = renderer.text_rendering_params();
                parameters.is_bgr = is_bgr;
                renderer.set_text_rendering_params(parameters);
            }
        }
    }

    pub(crate) fn supports_subpixel_rendering(&self) -> bool {
        match self {
            Self::Hardware { renderer, .. } => renderer.supports_dual_source_blending(),
            Self::Software { .. } => true,
        }
    }

    pub(crate) fn max_texture_size(&self) -> u32 {
        match self {
            Self::Hardware { renderer, .. } => renderer.max_texture_size(),
            Self::Software { .. } => i32::MAX as u32,
        }
    }

    pub(crate) fn device_lost(&self) -> bool {
        match self {
            Self::Hardware { renderer, .. } => renderer.device_lost(),
            Self::Software { .. } => false,
        }
    }

    pub(crate) fn recover(&mut self, window: &W) -> Result<()> {
        match self {
            Self::Hardware { renderer, .. } => renderer.recover(window),
            Self::Software { .. } => Ok(()),
        }
    }

    pub(crate) fn needs_redraw(&mut self) -> bool {
        match self {
            Self::Hardware { renderer, .. } => renderer.needs_redraw(),
            Self::Software { .. } => false,
        }
    }

    pub(crate) fn sprite_atlas(&self) -> Arc<dyn PlatformAtlas> {
        match self {
            Self::Hardware { renderer, .. } => renderer.sprite_atlas().clone(),
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

    pub(crate) fn is_software(&self) -> bool {
        matches!(self, Self::Software { .. })
    }
}
