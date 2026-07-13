mod wgpu_atlas;
mod wgpu_context;
mod wgpu_renderer;

pub use gpui_cosmic_text::CosmicTextSystem;
pub use wgpu;
pub use wgpu_atlas::*;
pub use wgpu_context::*;
pub use wgpu_renderer::{GpuContext, WgpuRenderer, WgpuSurfaceConfig};

#[cfg(test)]
mod tests {
    use super::CosmicTextSystem;

    #[test]
    fn cosmic_text_system_remains_reexported() {
        let _: fn(&str) -> CosmicTextSystem = CosmicTextSystem::new;
    }
}
