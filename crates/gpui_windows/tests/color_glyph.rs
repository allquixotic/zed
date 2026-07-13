#[path = "../src/color_glyph.rs"]
mod color_glyph;

use color_glyph::{ColorGlyphLayer, ColorGlyphRenderingParams, rasterize_color_glyph};

const UNCORRECTED: ColorGlyphRenderingParams = ColorGlyphRenderingParams {
    gamma_ratios: [0.0; 4],
    enhanced_contrast: 0.0,
};

#[test]
fn color_glyph_layers_source_over_on_cpu() {
    let rasterized = rasterize_color_glyph(
        [1, 1],
        [
            ColorGlyphLayer {
                origin: [0, 0],
                size: [1, 1],
                color: [1.0, 0.0, 0.0, 1.0],
                alpha_mask: vec![128],
            },
            ColorGlyphLayer {
                origin: [0, 0],
                size: [1, 1],
                color: [0.0, 0.0, 1.0, 0.5],
                alpha_mask: vec![255],
            },
        ],
        UNCORRECTED,
    )
    .expect("CPU color glyph composition should succeed");

    assert_eq!(rasterized, [170, 0, 85, 192]);
}

#[test]
fn color_glyph_cpu_compositor_clips_and_validates() {
    let rasterized = rasterize_color_glyph(
        [1, 1],
        [ColorGlyphLayer {
            origin: [-1, 0],
            size: [2, 1],
            color: [0.0, 1.0, 0.0, 1.0],
            alpha_mask: vec![255, 128],
        }],
        UNCORRECTED,
    )
    .expect("clipped CPU color glyph composition should succeed");
    assert_eq!(rasterized, [0, 255, 0, 128]);

    let error = rasterize_color_glyph(
        [1, 1],
        [ColorGlyphLayer {
            origin: [0, 0],
            size: [1, 1],
            color: [1.0; 4],
            alpha_mask: Vec::new(),
        }],
        UNCORRECTED,
    )
    .expect_err("malformed color glyph masks must be rejected");
    assert!(error.to_string().contains("mask length"));
}

#[test]
fn direct_write_text_system_has_no_d3d_dependencies() {
    let source = include_str!("../src/direct_write.rs");

    assert!(!source.contains("Direct3D11"));
    assert!(!source.contains("ID3D11"));
    assert!(!source.contains("GPUState"));
    assert!(!source.contains("handle_gpu_lost"));
    assert!(source.contains("pub(crate) fn new() -> Result<Self>"));
}
