mod dispatcher;
mod headless;
mod keyboard;
mod platform;
#[cfg(any(feature = "wayland", feature = "x11"))]
mod renderer;
#[cfg(any(feature = "wayland", feature = "x11"))]
mod text_system;
#[cfg(feature = "wayland")]
mod wayland;
#[cfg(feature = "x11")]
mod x11;

#[cfg(any(feature = "wayland", feature = "x11"))]
mod xdg_desktop_portal;

pub use dispatcher::*;
pub(crate) use headless::*;
pub(crate) use keyboard::*;
pub(crate) use platform::*;
#[cfg(any(feature = "wayland", feature = "x11"))]
pub(crate) use renderer::*;
#[cfg(any(feature = "wayland", feature = "x11"))]
pub(crate) use text_system::*;
#[cfg(feature = "wayland")]
pub(crate) use wayland::*;
#[cfg(feature = "x11")]
pub(crate) use x11::*;

use std::rc::Rc;

use gpui::PlatformOptions;

/// Returns the default platform implementation for the current OS.
pub fn current_platform(headless: bool) -> Rc<dyn gpui::Platform> {
    current_platform_with_options(PlatformOptions::auto(headless))
        .expect("failed to initialize Linux platform")
}

pub fn current_platform_with_options(
    options: PlatformOptions,
) -> anyhow::Result<Rc<dyn gpui::Platform>> {
    try_current_platform(options)
}

pub fn try_current_platform(options: PlatformOptions) -> anyhow::Result<Rc<dyn gpui::Platform>> {
    #[cfg(feature = "x11")]
    use anyhow::Context as _;

    if options.headless {
        return Ok(Rc::new(LinuxPlatform {
            inner: HeadlessClient::new(),
        }));
    }

    match gpui::guess_compositor() {
        #[cfg(feature = "wayland")]
        "Wayland" => Ok(Rc::new(LinuxPlatform {
            inner: WaylandClient::new(options.renderer_preference),
        })),

        #[cfg(feature = "x11")]
        "X11" => Ok(Rc::new(LinuxPlatform {
            inner: X11Client::new(options.renderer_preference)
                .context("Failed to initialize X11 client.")?,
        })),

        "Headless" => Ok(Rc::new(LinuxPlatform {
            inner: HeadlessClient::new(),
        })),
        compositor => anyhow::bail!(
            "unsupported Linux compositor {compositor:?}; enable the matching gpui_linux feature"
        ),
    }
}
