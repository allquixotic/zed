mod software_atlas;
mod software_rasterizer;

use anyhow::{Context as _, Result, bail, ensure};
use gpui::{
    BackgroundKind, Bounds, DevicePixels, PrimitiveBatch, Quad, Rgba, ScaledPixels, Scene, Size,
};
use std::sync::Arc;

pub use software_atlas::{SoftwareAtlas, SoftwareAtlasStats, SoftwareAtlasTile};
pub use software_rasterizer::{SOFTWARE_TILE_SIZE, SoftwareRasterizer};

pub struct SoftwareHeadlessRenderer {
    atlas: Arc<SoftwareAtlas>,
    rasterizer: SoftwareRasterizer,
    framebuffer: Vec<u32>,
    size: Size<DevicePixels>,
}

pub struct SoftwareRenderer {
    headless: SoftwareHeadlessRenderer,
}

impl SoftwareHeadlessRenderer {
    pub fn new(atlas: Arc<SoftwareAtlas>) -> Self {
        Self {
            atlas,
            rasterizer: SoftwareRasterizer::new(),
            framebuffer: Vec::new(),
            size: Size::default(),
        }
    }

    pub fn atlas(&self) -> &Arc<SoftwareAtlas> {
        &self.atlas
    }

    pub fn size(&self) -> Size<DevicePixels> {
        self.size
    }

    pub fn framebuffer(&self) -> &[u32] {
        &self.framebuffer
    }

    pub fn render(
        &mut self,
        scene: &Scene,
        size: Size<DevicePixels>,
        background: Rgba,
    ) -> Result<&[u32]> {
        let width = usize::try_from(size.width.0).context("software viewport width is negative")?;
        let height =
            usize::try_from(size.height.0).context("software viewport height is negative")?;
        let pixel_count = width
            .checked_mul(height)
            .context("software viewport dimensions overflowed")?;
        if self.framebuffer.len() != pixel_count {
            self.framebuffer.clear();
            self.framebuffer
                .try_reserve_exact(pixel_count)
                .context("allocating software framebuffer")?;
            self.framebuffer.resize(pixel_count, 0);
        }
        self.size = size;
        if pixel_count == 0 {
            return Ok(&self.framebuffer);
        }

        let mut tile_y = 0usize;
        while tile_y < height {
            let tile_height = height.saturating_sub(tile_y).min(SOFTWARE_TILE_SIZE);
            let mut tile_x = 0usize;
            while tile_x < width {
                let tile_width = width.saturating_sub(tile_x).min(SOFTWARE_TILE_SIZE);
                self.render_tile(
                    scene,
                    background,
                    [tile_x, tile_y],
                    [tile_width, tile_height],
                    width,
                )?;
                tile_x = tile_x
                    .checked_add(tile_width)
                    .context("software tile x coordinate overflowed")?;
            }
            tile_y = tile_y
                .checked_add(tile_height)
                .context("software tile y coordinate overflowed")?;
        }
        Ok(&self.framebuffer)
    }

    fn render_tile(
        &mut self,
        scene: &Scene,
        background: Rgba,
        tile_origin: [usize; 2],
        tile_size: [usize; 2],
        framebuffer_width: usize,
    ) -> Result<()> {
        self.rasterizer.begin(
            [
                u16::try_from(tile_size[0]).context("software tile width exceeds u16")?,
                u16::try_from(tile_size[1]).context("software tile height exceeds u16")?,
            ],
            background,
        )?;
        for batch in scene.batches() {
            match batch {
                PrimitiveBatch::Quads(range) => {
                    let quads = scene
                        .quads
                        .get(range)
                        .context("scene quad batch is out of bounds")?;
                    for quad in quads {
                        render_quad(&mut self.rasterizer, quad, tile_origin, tile_size)?;
                    }
                }
                PrimitiveBatch::Shadows(range)
                | PrimitiveBatch::Paths(range)
                | PrimitiveBatch::Underlines(range)
                | PrimitiveBatch::Surfaces(range) => {
                    ensure!(range.is_empty(), "scene primitive is not implemented yet");
                }
                PrimitiveBatch::MonochromeSprites { range, .. }
                | PrimitiveBatch::SubpixelSprites { range, .. }
                | PrimitiveBatch::PolychromeSprites { range, .. } => {
                    ensure!(range.is_empty(), "scene sprite is not implemented yet");
                }
            }
        }
        let tile_pixels = self.rasterizer.finish()?;
        copy_tile(
            &mut self.framebuffer,
            framebuffer_width,
            tile_origin,
            tile_size,
            tile_pixels,
        )
    }
}

impl SoftwareRenderer {
    pub fn new(atlas: Arc<SoftwareAtlas>) -> Self {
        Self {
            headless: SoftwareHeadlessRenderer::new(atlas),
        }
    }

    pub fn atlas(&self) -> &Arc<SoftwareAtlas> {
        self.headless.atlas()
    }

    pub fn render(
        &mut self,
        scene: &Scene,
        size: Size<DevicePixels>,
        background: Rgba,
    ) -> Result<&[u32]> {
        self.headless.render(scene, size, background)
    }
}

fn render_quad(
    rasterizer: &mut SoftwareRasterizer,
    quad: &Quad,
    tile_origin: [usize; 2],
    tile_size: [usize; 2],
) -> Result<()> {
    let BackgroundKind::Solid(color) = quad.background.kind() else {
        bail!("non-solid software quad is not implemented yet");
    };
    ensure!(
        [
            quad.border_widths.top.0,
            quad.border_widths.right.0,
            quad.border_widths.bottom.0,
            quad.border_widths.left.0,
            quad.corner_radii.top_left.0,
            quad.corner_radii.top_right.0,
            quad.corner_radii.bottom_right.0,
            quad.corner_radii.bottom_left.0,
        ]
        .into_iter()
        .all(|value| value == 0.0),
        "bordered or rounded software quad is not implemented yet"
    );
    let clipped = intersect_bounds(quad.bounds, quad.content_mask.bounds)?;
    let tile_left = tile_origin[0] as f64;
    let tile_top = tile_origin[1] as f64;
    let tile_right = tile_left + tile_size[0] as f64;
    let tile_bottom = tile_top + tile_size[1] as f64;
    let rect = [
        clipped[0].max(tile_left) - tile_left,
        clipped[1].max(tile_top) - tile_top,
        clipped[2].min(tile_right) - tile_left,
        clipped[3].min(tile_bottom) - tile_top,
    ];
    rasterizer.fill_rectangle(rect, color.to_rgb())
}

fn intersect_bounds(first: Bounds<ScaledPixels>, second: Bounds<ScaledPixels>) -> Result<[f64; 4]> {
    let first = bounds_edges(first)?;
    let second = bounds_edges(second)?;
    Ok([
        first[0].max(second[0]),
        first[1].max(second[1]),
        first[2].min(second[2]),
        first[3].min(second[3]),
    ])
}

fn bounds_edges(bounds: Bounds<ScaledPixels>) -> Result<[f64; 4]> {
    let left = f64::from(bounds.origin.x.0);
    let top = f64::from(bounds.origin.y.0);
    let width = f64::from(bounds.size.width.0);
    let height = f64::from(bounds.size.height.0);
    ensure!(
        [left, top, width, height].into_iter().all(f64::is_finite),
        "scene bounds contain a non-finite coordinate"
    );
    Ok([left, top, left + width, top + height])
}

fn copy_tile(
    framebuffer: &mut [u32],
    framebuffer_width: usize,
    tile_origin: [usize; 2],
    tile_size: [usize; 2],
    tile_pixels: &[u32],
) -> Result<()> {
    let expected_tile_pixels = tile_size[0]
        .checked_mul(tile_size[1])
        .context("software tile dimensions overflowed")?;
    ensure!(
        tile_pixels.len() == expected_tile_pixels,
        "software tile has {} pixels; expected {expected_tile_pixels}",
        tile_pixels.len()
    );
    for row in 0..tile_size[1] {
        let source_start = row
            .checked_mul(tile_size[0])
            .context("software tile source row overflowed")?;
        let source_end = source_start
            .checked_add(tile_size[0])
            .context("software tile source range overflowed")?;
        let destination_row = tile_origin[1]
            .checked_add(row)
            .context("software tile destination row overflowed")?;
        let destination_start = destination_row
            .checked_mul(framebuffer_width)
            .and_then(|row_start| row_start.checked_add(tile_origin[0]))
            .context("software tile destination offset overflowed")?;
        let destination_end = destination_start
            .checked_add(tile_size[0])
            .context("software tile destination range overflowed")?;
        let source = tile_pixels
            .get(source_start..source_end)
            .context("software tile source range is out of bounds")?;
        let destination = framebuffer
            .get_mut(destination_start..destination_end)
            .context("software tile destination range is out of bounds")?;
        destination.copy_from_slice(source);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::{BorderStyle, ContentMask, Corners, Edges, Hsla, Point, solid_background};

    fn scaled_bounds(left: f32, top: f32, width: f32, height: f32) -> Bounds<ScaledPixels> {
        Bounds {
            origin: Point {
                x: ScaledPixels(left),
                y: ScaledPixels(top),
            },
            size: Size {
                width: ScaledPixels(width),
                height: ScaledPixels(height),
            },
        }
    }

    fn quad(order: u32, bounds: Bounds<ScaledPixels>, color: Hsla) -> Quad {
        Quad {
            order,
            border_style: BorderStyle::Solid,
            bounds,
            content_mask: ContentMask {
                bounds: scaled_bounds(0.0, 0.0, 130.0, 70.0),
            },
            background: solid_background(color),
            border_color: Hsla::transparent_black(),
            corner_radii: Corners::default(),
            border_widths: Edges::default(),
        }
    }

    #[test]
    fn headless_renderer_preserves_order_across_tiles() -> Result<()> {
        let mut scene = Scene::default();
        scene
            .quads
            .push(quad(0, scaled_bounds(1.0, 1.0, 128.0, 68.0), Hsla::red()));
        scene
            .quads
            .push(quad(1, scaled_bounds(63.0, 2.0, 2.0, 66.0), Hsla::blue()));
        scene.finish();
        let mut renderer = SoftwareHeadlessRenderer::new(Arc::new(SoftwareAtlas::new()));
        let framebuffer = renderer.render(
            &scene,
            Size {
                width: DevicePixels(130),
                height: DevicePixels(70),
            },
            Rgba {
                r: 1.0,
                g: 1.0,
                b: 1.0,
                a: 0.25,
            },
        )?;

        assert_eq!(framebuffer[0], 0x00ff_ffff);
        assert_eq!(framebuffer[1 + 130], 0x00ff_0000);
        assert_eq!(framebuffer[63 + 2 * 130], 0x0000_00ff);
        assert_eq!(framebuffer[64 + 67 * 130], 0x0000_00ff);
        assert_eq!(framebuffer[129 + 69 * 130], 0x00ff_ffff);
        Ok(())
    }

    #[test]
    fn zero_sized_and_invalid_viewports_do_not_panic() -> Result<()> {
        let scene = Scene::default();
        let mut renderer = SoftwareHeadlessRenderer::new(Arc::new(SoftwareAtlas::new()));
        assert!(
            renderer
                .render(
                    &scene,
                    Size {
                        width: DevicePixels(-1),
                        height: DevicePixels(1),
                    },
                    Rgba::default(),
                )
                .is_err()
        );
        let framebuffer = renderer.render(
            &scene,
            Size {
                width: DevicePixels(0),
                height: DevicePixels(0),
            },
            Rgba::default(),
        )?;
        assert!(framebuffer.is_empty());
        Ok(())
    }
}
