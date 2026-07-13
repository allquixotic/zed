//! Convenience crate that re-exports GPUI's platform traits and the
//! `current_platform` constructor so consumers don't need `#[cfg]` gating.

pub use gpui::{Platform, PlatformOptions};

use std::rc::Rc;

/// Returns a background executor for the current platform.
pub fn background_executor() -> gpui::BackgroundExecutor {
    current_platform(true).background_executor()
}

pub fn application() -> gpui::Application {
    gpui::Application::with_platform(current_platform(false))
}

pub fn headless() -> gpui::Application {
    gpui::Application::with_platform(current_platform(true))
}

/// Unlike `application`, this function returns a single-threaded web application.
#[cfg(target_family = "wasm")]
pub fn single_threaded_web() -> gpui::Application {
    gpui::Application::with_platform(Rc::new(gpui_web::WebPlatform::new(false)))
}

/// Initializes panic hooks and logging for the web platform.
/// Call this before running the application in a wasm_bindgen entrypoint.
#[cfg(target_family = "wasm")]
pub fn web_init() {
    console_error_panic_hook::set_once();
    gpui_web::init_logging();
}

/// Returns the default [`Platform`] for the current OS.
pub fn current_platform(headless: bool) -> Rc<dyn Platform> {
    #[cfg(target_os = "macos")]
    {
        Rc::new(gpui_macos::MacPlatform::new(headless))
    }

    #[cfg(target_os = "windows")]
    {
        Rc::new(
            gpui_windows::WindowsPlatform::new(headless)
                .expect("failed to initialize Windows platform"),
        )
    }

    #[cfg(any(target_os = "linux", target_os = "freebsd"))]
    {
        gpui_linux::current_platform(headless)
    }

    #[cfg(target_family = "wasm")]
    {
        let _ = headless;
        Rc::new(gpui_web::WebPlatform::new(true))
    }
}

/// Returns the default [`Platform`] for the current OS using the supplied options.
pub fn current_platform_with_options(options: PlatformOptions) -> gpui::Result<Rc<dyn Platform>> {
    try_current_platform(options)
}

/// Attempts to construct the default [`Platform`] for the current OS.
pub fn try_current_platform(options: PlatformOptions) -> gpui::Result<Rc<dyn Platform>> {
    #[cfg(target_os = "macos")]
    {
        Ok(Rc::new(gpui_macos::MacPlatform::new_with_options(options)))
    }

    #[cfg(target_os = "windows")]
    {
        Ok(Rc::new(gpui_windows::WindowsPlatform::new_with_options(
            options,
        )?))
    }

    #[cfg(any(target_os = "linux", target_os = "freebsd"))]
    {
        gpui_linux::try_current_platform(options)
    }

    #[cfg(target_family = "wasm")]
    {
        let _ = options;
        Ok(Rc::new(gpui_web::WebPlatform::new(true)))
    }
}

/// Returns a new [`HeadlessRenderer`] for the current platform, if available.
#[cfg(feature = "test-support")]
pub fn current_headless_renderer() -> Option<Box<dyn gpui::PlatformHeadlessRenderer>> {
    #[cfg(target_os = "macos")]
    {
        gpui_macos::metal_renderer::MetalHeadlessRenderer::new()
            .ok()
            .map(|renderer| Box::new(renderer) as Box<dyn gpui::PlatformHeadlessRenderer>)
    }

    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

#[cfg(test)]
mod renderer_contract_tests {
    use super::*;
    use gpui::{
        AnimationPolicy, AtlasTextureId, AtlasTextureKind, AtlasTile, BackgroundKind, GpuSpecs,
        HardwareAvailability, PolychromeSprite, RendererBackend, RendererFallbackReason,
        RendererPreference, RenderingCapabilities, RenderingInfo, Shadow, TileId, Underline,
        checkerboard, hsla, linear_color_stop, linear_gradient, pattern_slash, rgba,
    };

    #[test]
    fn renderer_contract_reports_hardware_and_software_status() {
        let gpu_specs = GpuSpecs {
            device_name: "Test GPU".to_string(),
            ..GpuSpecs::default()
        };
        let hardware = RenderingInfo::hardware(Some(gpu_specs.clone()));
        assert_eq!(hardware.requested_preference, RendererPreference::Auto);
        assert_eq!(hardware.active_backend, RendererBackend::Hardware);
        assert_eq!(
            hardware.hardware_availability,
            HardwareAvailability::Available
        );
        assert_eq!(hardware.gpu_specs, Some(gpu_specs));
        assert_eq!(hardware.fallback_reason, None);
        assert_eq!(hardware.capabilities, RenderingCapabilities::hardware());

        let software = RenderingInfo::software(
            RendererPreference::Auto,
            HardwareAvailability::Unavailable,
            Some(RendererFallbackReason::NoHardwareAdapter),
        );
        assert_eq!(software.active_backend, RendererBackend::Software);
        assert_eq!(software.gpu_specs, None);
        assert_eq!(software.capabilities.transparency, false);
        assert_eq!(
            software.capabilities.animation_policy,
            AnimationPolicy::Reduced
        );
    }

    #[test]
    fn renderer_contract_exposes_semantic_scene_values() {
        assert!(gpui::PaddedBool32::from(true).get());
        assert!(!gpui::PaddedBool32::from(false).get());

        let from = linear_color_stop(rgba(0xff0099ff), 0.0);
        let to = linear_color_stop(rgba(0x00ff99ff), 1.0);
        assert_eq!(
            linear_gradient(90.0, from, to).kind(),
            BackgroundKind::LinearGradient {
                angle: 90.0,
                color_space: gpui::ColorSpace::Srgb,
                stops: [from, to],
            }
        );

        let color = hsla(0.5, 0.6, 0.7, 0.8);
        assert!(matches!(
            pattern_slash(color, 2.0, 3.0).kind(),
            BackgroundKind::PatternSlash { .. }
        ));
        assert_eq!(
            checkerboard(color, 8.0).kind(),
            BackgroundKind::Checkerboard { color, size: 8.0 }
        );

        let underline = Underline {
            order: 0,
            pad: 0,
            bounds: Default::default(),
            content_mask: Default::default(),
            color: Default::default(),
            thickness: Default::default(),
            wavy: true.into(),
        };
        assert!(underline.is_wavy());

        let shadow = Shadow {
            order: 0,
            blur_radius: Default::default(),
            bounds: Default::default(),
            corner_radii: Default::default(),
            content_mask: Default::default(),
            color: Default::default(),
            element_bounds: Default::default(),
            element_corner_radii: Default::default(),
            inset: 1,
            pad: 0,
        };
        assert!(shadow.is_inset());

        let sprite = PolychromeSprite {
            order: 0,
            pad: 0,
            grayscale: true.into(),
            opacity: 1.0,
            bounds: Default::default(),
            content_mask: Default::default(),
            corner_radii: Default::default(),
            tile: AtlasTile {
                texture_id: AtlasTextureId {
                    index: 0,
                    kind: AtlasTextureKind::Polychrome,
                },
                tile_id: TileId(0),
                padding: 0,
                bounds: Default::default(),
            },
        };
        assert!(sprite.is_grayscale());
    }

    #[test]
    fn renderer_contract_keeps_legacy_and_fallible_factories() {
        let _: fn(bool) -> Rc<dyn Platform> = current_platform;
        let _: fn(PlatformOptions) -> gpui::Result<Rc<dyn Platform>> =
            current_platform_with_options;
        let _: fn(PlatformOptions) -> gpui::Result<Rc<dyn Platform>> = try_current_platform;
        assert_eq!(PlatformOptions::default(), PlatformOptions::auto(false));
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    use gpui::{AppContext, Empty, VisualTestAppContext};
    use std::cell::RefCell;
    use std::time::Duration;

    // Note: All VisualTestAppContext tests are ignored by default because they require
    // the macOS main thread. Standard Rust tests run on worker threads, which causes
    // SIGABRT when interacting with macOS AppKit/Cocoa APIs.
    //
    // To run these tests, use:
    // cargo test -p gpui visual_test_context -- --ignored --test-threads=1

    #[test]
    #[ignore] // Requires macOS main thread
    fn test_foreground_tasks_run_with_run_until_parked() {
        let mut cx = VisualTestAppContext::new(current_platform(false));

        let task_ran = Rc::new(RefCell::new(false));

        // Spawn a foreground task via the App's spawn method
        // This should use our TestDispatcher, not the MacDispatcher
        {
            let task_ran = task_ran.clone();
            cx.update(|cx| {
                cx.spawn(async move |_| {
                    *task_ran.borrow_mut() = true;
                })
                .detach();
            });
        }

        // The task should not have run yet
        assert!(!*task_ran.borrow());

        // Run until parked should execute the foreground task
        cx.run_until_parked();

        // Now the task should have run
        assert!(*task_ran.borrow());
    }

    #[test]
    #[ignore] // Requires macOS main thread
    fn test_advance_clock_triggers_delayed_tasks() {
        let mut cx = VisualTestAppContext::new(current_platform(false));

        let task_ran = Rc::new(RefCell::new(false));

        // Spawn a task that waits for a timer
        {
            let task_ran = task_ran.clone();
            let executor = cx.background_executor.clone();
            cx.update(|cx| {
                cx.spawn(async move |_| {
                    executor.timer(Duration::from_millis(500)).await;
                    *task_ran.borrow_mut() = true;
                })
                .detach();
            });
        }

        // Run until parked - the task should be waiting on the timer
        cx.run_until_parked();
        assert!(!*task_ran.borrow());

        // Advance clock past the timer duration
        cx.advance_clock(Duration::from_millis(600));

        // Now the task should have completed
        assert!(*task_ran.borrow());
    }

    #[test]
    #[ignore] // Requires macOS main thread - window creation fails on test threads
    fn test_window_spawn_uses_test_dispatcher() {
        let mut cx = VisualTestAppContext::new(current_platform(false));

        let task_ran = Rc::new(RefCell::new(false));

        let window = cx
            .open_offscreen_window_default(|_, cx| cx.new(|_| Empty))
            .expect("Failed to open window");

        // Spawn a task via window.spawn - this is the critical test case
        // for tooltip behavior, as tooltips use window.spawn for delayed show
        {
            let task_ran = task_ran.clone();
            cx.update_window(window.into(), |_, window, cx| {
                window
                    .spawn(cx, async move |_| {
                        *task_ran.borrow_mut() = true;
                    })
                    .detach();
            })
            .ok();
        }

        // The task should not have run yet
        assert!(!*task_ran.borrow());

        // Run until parked should execute the foreground task spawned via window
        cx.run_until_parked();

        // Now the task should have run
        assert!(*task_ran.borrow());
    }
}
