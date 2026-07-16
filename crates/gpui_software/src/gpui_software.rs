mod software_atlas;
mod software_presenter;
mod software_rasterizer;
mod software_scene;
mod software_surface;

use anyhow::{Context as _, Result, ensure};
use gpui::{DevicePixels, Rgba, Scene, Size};
use rayon::prelude::*;
use software_scene::SoftwareScene;
use std::sync::{Arc, OnceLock};

pub use software_atlas::{SoftwareAtlas, SoftwareAtlasStats, SoftwareAtlasTile};
pub use software_presenter::SoftwarePresenter;
pub use software_rasterizer::{
    SOFTWARE_TILE_SIZE, SoftwareRasterizer, SoftwareTextRenderingParams,
};

static SOFTWARE_WORKER_POOL: OnceLock<Result<rayon::ThreadPool, String>> = OnceLock::new();

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SoftwareDamageRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

pub struct SoftwareFrame<'a> {
    pub size: [u32; 2],
    pub framebuffer: &'a [u32],
    pub damage: &'a [SoftwareDamageRect],
    pub rasterized_tiles: usize,
}

impl SoftwareFrame<'_> {
    pub fn changed(&self) -> bool {
        !self.damage.is_empty()
    }
}

pub struct SoftwareRenderer {
    atlas: Arc<SoftwareAtlas>,
    framebuffer: Vec<u32>,
    size: Size<DevicePixels>,
    previous_scene: Option<SoftwareScene>,
    damage: Vec<SoftwareDamageRect>,
    rasterized_tiles: usize,
    text_rendering: SoftwareTextRenderingParams,
}

pub struct SoftwareHeadlessRenderer {
    renderer: SoftwareRenderer,
}

impl SoftwareRenderer {
    pub fn new(atlas: Arc<SoftwareAtlas>) -> Self {
        Self {
            atlas,
            framebuffer: Vec::new(),
            size: Size::default(),
            previous_scene: None,
            damage: Vec::new(),
            rasterized_tiles: 0,
            text_rendering: SoftwareTextRenderingParams::default(),
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

    pub fn damage(&self) -> &[SoftwareDamageRect] {
        &self.damage
    }

    pub fn text_rendering_params(&self) -> SoftwareTextRenderingParams {
        self.text_rendering
    }

    pub fn set_text_rendering_params(&mut self, params: SoftwareTextRenderingParams) {
        if !text_params_equal(self.text_rendering, params) {
            self.text_rendering = params;
            self.previous_scene = None;
        }
    }

    pub fn invalidate(&mut self) {
        self.previous_scene = None;
    }

    pub fn render(
        &mut self,
        scene: &Scene,
        size: Size<DevicePixels>,
        background: Rgba,
    ) -> Result<&[u32]> {
        self.render_frame(scene, size, 1.0, background)?;
        Ok(&self.framebuffer)
    }

    pub fn render_frame(
        &mut self,
        scene: &Scene,
        size: Size<DevicePixels>,
        scale_factor: f32,
        background: Rgba,
    ) -> Result<SoftwareFrame<'_>> {
        let next_scene =
            SoftwareScene::new(scene, size, scale_factor, background, self.atlas.as_ref())?;
        let frame_size = [
            u32::try_from(next_scene.width).context("software viewport width exceeds u32")?,
            u32::try_from(next_scene.height).context("software viewport height exceeds u32")?,
        ];
        let pixel_count = next_scene
            .width
            .checked_mul(next_scene.height)
            .context("software viewport dimensions overflowed")?;
        if self.framebuffer.len() != pixel_count {
            let mut framebuffer = Vec::new();
            framebuffer
                .try_reserve_exact(pixel_count)
                .context("allocating software framebuffer")?;
            framebuffer.resize(pixel_count, 0);
            self.framebuffer = framebuffer;
        }

        let dirty_tiles = dirty_tiles(self.previous_scene.as_ref(), &next_scene);
        if dirty_tiles.is_empty() {
            self.size = size;
            self.previous_scene = Some(next_scene);
            self.damage.clear();
            self.rasterized_tiles = 0;
            return Ok(SoftwareFrame {
                size: frame_size,
                framebuffer: &self.framebuffer,
                damage: &self.damage,
                rasterized_tiles: 0,
            });
        }

        let text_rendering = self.text_rendering;
        let background = next_scene.background.get();
        let rendered_tiles = worker_pool()?.install(|| {
            dirty_tiles
                .par_iter()
                .map_init(SoftwareRasterizer::new, |rasterizer, &index| {
                    let tile_origin = next_scene.tile_origin(index)?;
                    let tile_size = next_scene.tile_size(index)?;
                    rasterizer.begin_tile(
                        [
                            u16::try_from(tile_size[0])
                                .context("software tile width exceeds u16")?,
                            u16::try_from(tile_size[1])
                                .context("software tile height exceeds u16")?,
                        ],
                        tile_origin,
                        background,
                        text_rendering,
                    )?;
                    let commands = next_scene
                        .tiles
                        .get(index)
                        .context("software tile command list is missing")?;
                    rasterizer.render_commands(commands)?;
                    Ok::<_, anyhow::Error>(RenderedTile {
                        origin: tile_origin,
                        size: tile_size,
                        pixels: rasterizer.finish()?.to_vec(),
                    })
                })
                .collect::<Result<Vec<_>>>()
        })?;

        for tile in &rendered_tiles {
            copy_tile(
                &mut self.framebuffer,
                next_scene.width,
                tile.origin,
                tile.size,
                &tile.pixels,
            )?;
        }
        self.damage = coalesce_damage(&dirty_tiles, &next_scene)?;
        self.rasterized_tiles = rendered_tiles.len();
        self.size = size;
        self.previous_scene = Some(next_scene);
        Ok(SoftwareFrame {
            size: frame_size,
            framebuffer: &self.framebuffer,
            damage: &self.damage,
            rasterized_tiles: self.rasterized_tiles,
        })
    }
}

impl SoftwareHeadlessRenderer {
    pub fn new(atlas: Arc<SoftwareAtlas>) -> Self {
        Self {
            renderer: SoftwareRenderer::new(atlas),
        }
    }

    pub fn atlas(&self) -> &Arc<SoftwareAtlas> {
        self.renderer.atlas()
    }

    pub fn size(&self) -> Size<DevicePixels> {
        self.renderer.size()
    }

    pub fn framebuffer(&self) -> &[u32] {
        self.renderer.framebuffer()
    }

    pub fn damage(&self) -> &[SoftwareDamageRect] {
        self.renderer.damage()
    }

    pub fn render(
        &mut self,
        scene: &Scene,
        size: Size<DevicePixels>,
        background: Rgba,
    ) -> Result<&[u32]> {
        self.renderer.render(scene, size, background)
    }

    pub fn render_frame(
        &mut self,
        scene: &Scene,
        size: Size<DevicePixels>,
        scale_factor: f32,
        background: Rgba,
    ) -> Result<SoftwareFrame<'_>> {
        self.renderer
            .render_frame(scene, size, scale_factor, background)
    }
}

struct RenderedTile {
    origin: [usize; 2],
    size: [usize; 2],
    pixels: Vec<u32>,
}

fn worker_pool() -> Result<&'static rayon::ThreadPool> {
    match SOFTWARE_WORKER_POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .thread_name(|index| format!("gpui-software-{index}"))
            .build()
            .map_err(|error| error.to_string())
    }) {
        Ok(pool) => Ok(pool),
        Err(error) => anyhow::bail!("creating software renderer worker pool: {error}"),
    }
}

fn dirty_tiles(previous: Option<&SoftwareScene>, next: &SoftwareScene) -> Vec<usize> {
    let Some(previous) = previous.filter(|previous| {
        previous.width == next.width
            && previous.height == next.height
            && previous.columns == next.columns
            && previous.rows == next.rows
            && previous.scale_factor == next.scale_factor
            && previous.background == next.background
            && previous.tiles.len() == next.tiles.len()
    }) else {
        return (0..next.tiles.len()).collect();
    };
    previous
        .tiles
        .iter()
        .zip(&next.tiles)
        .enumerate()
        .filter_map(|(index, (previous, next))| (previous != next).then_some(index))
        .collect()
}

fn coalesce_damage(
    dirty_tiles: &[usize],
    scene: &SoftwareScene,
) -> Result<Vec<SoftwareDamageRect>> {
    let mut horizontal = Vec::new();
    let mut cursor = 0;
    while cursor < dirty_tiles.len() {
        let first_index = dirty_tiles[cursor];
        let row = first_index / scene.columns;
        let first_column = first_index % scene.columns;
        let mut last_column = first_column;
        cursor = cursor.saturating_add(1);
        while let Some(&index) = dirty_tiles.get(cursor) {
            let next_row = index / scene.columns;
            let next_column = index % scene.columns;
            if next_row != row || next_column != last_column.saturating_add(1) {
                break;
            }
            last_column = next_column;
            cursor = cursor.saturating_add(1);
        }
        let x = first_column
            .checked_mul(SOFTWARE_TILE_SIZE)
            .context("software damage x coordinate overflowed")?;
        let y = row
            .checked_mul(SOFTWARE_TILE_SIZE)
            .context("software damage y coordinate overflowed")?;
        let right = last_column
            .checked_add(1)
            .and_then(|column| column.checked_mul(SOFTWARE_TILE_SIZE))
            .context("software damage right coordinate overflowed")?
            .min(scene.width);
        let bottom = y.saturating_add(SOFTWARE_TILE_SIZE).min(scene.height);
        horizontal.push(SoftwareDamageRect {
            x: u32::try_from(x).context("software damage x exceeds u32")?,
            y: u32::try_from(y).context("software damage y exceeds u32")?,
            width: u32::try_from(right.saturating_sub(x))
                .context("software damage width exceeds u32")?,
            height: u32::try_from(bottom.saturating_sub(y))
                .context("software damage height exceeds u32")?,
        });
    }

    let mut coalesced: Vec<SoftwareDamageRect> = Vec::new();
    for rect in horizontal {
        if let Some(previous) = coalesced.iter_mut().rev().find(|previous| {
            previous.x == rect.x
                && previous.width == rect.width
                && previous.y.saturating_add(previous.height) == rect.y
        }) {
            previous.height = previous.height.saturating_add(rect.height);
        } else {
            coalesced.push(rect);
        }
    }
    Ok(coalesced)
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

fn text_params_equal(
    first: SoftwareTextRenderingParams,
    second: SoftwareTextRenderingParams,
) -> bool {
    first
        .gamma_ratios
        .into_iter()
        .zip(second.gamma_ratios)
        .all(|(first, second)| first.to_bits() == second.to_bits())
        && first.grayscale_enhanced_contrast.to_bits()
            == second.grayscale_enhanced_contrast.to_bits()
        && first.subpixel_enhanced_contrast.to_bits() == second.subpixel_enhanced_contrast.to_bits()
        && first.is_bgr == second.is_bgr
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::{
        AtlasKey, BorderStyle, ColorSpace, ContentMask, Corners, Edges, FontId, GlyphId, Hsla,
        ImageId, MonochromeSprite, PaintSurface, Path, PathStroke, PlatformAtlas, Point,
        PolychromeSprite, Quad, RenderGlyphParams, RenderImageParams, RenderSvgParams,
        ScaledPixels, Shadow, SubpixelSprite, TransformationMatrix, Underline, checkerboard,
        linear_color_stop, linear_gradient, pattern_slash, solid_background,
    };
    use std::borrow::Cow;

    fn scaled_bounds(left: f32, top: f32, width: f32, height: f32) -> gpui::Bounds<ScaledPixels> {
        gpui::Bounds {
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

    fn quad(order: u32, bounds: gpui::Bounds<ScaledPixels>, color: Hsla) -> Quad {
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

    fn content_mask(width: f32, height: f32) -> ContentMask<ScaledPixels> {
        ContentMask {
            bounds: scaled_bounds(0.0, 0.0, width, height),
        }
    }

    fn insert_tile(
        atlas: &SoftwareAtlas,
        key: AtlasKey,
        width: i32,
        height: i32,
        bytes: Vec<u8>,
    ) -> Result<gpui::AtlasTile> {
        let mut bytes = Some(bytes);
        atlas
            .get_or_insert_with(&key, &mut || {
                Ok(Some((
                    Size {
                        width: DevicePixels(width),
                        height: DevicePixels(height),
                    },
                    Cow::Owned(bytes.take().context("atlas test tile was built twice")?),
                )))
            })?
            .context("atlas test tile was not inserted")
    }

    fn nonwhite_pixels(framebuffer: &[u32], framebuffer_width: usize, bounds: [usize; 4]) -> usize {
        (bounds[1]..bounds[3])
            .flat_map(|y| (bounds[0]..bounds[2]).map(move |x| x + y * framebuffer_width))
            .filter(|&index| {
                framebuffer
                    .get(index)
                    .is_some_and(|pixel| *pixel != 0x00ff_ffff)
            })
            .count()
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
    fn unchanged_scene_does_no_raster_work() -> Result<()> {
        let mut scene = Scene::default();
        scene
            .quads
            .push(quad(0, scaled_bounds(1.0, 1.0, 4.0, 4.0), Hsla::red()));
        scene.finish();
        let mut renderer = SoftwareRenderer::new(Arc::new(SoftwareAtlas::new()));
        let size = Size {
            width: DevicePixels(128),
            height: DevicePixels(64),
        };
        assert_eq!(
            renderer
                .render_frame(&scene, size, 1.0, Rgba::default())?
                .rasterized_tiles,
            2
        );
        let frame = renderer.render_frame(&scene, size, 1.0, Rgba::default())?;
        assert_eq!(frame.rasterized_tiles, 0);
        assert!(!frame.changed());
        Ok(())
    }

    #[test]
    fn order_and_background_changes_invalidate_affected_tiles() -> Result<()> {
        let size = Size {
            width: DevicePixels(128),
            height: DevicePixels(64),
        };
        let mut renderer = SoftwareRenderer::new(Arc::new(SoftwareAtlas::new()));
        let mut first = Scene::default();
        first
            .quads
            .push(quad(0, scaled_bounds(2.0, 2.0, 12.0, 12.0), Hsla::red()));
        first
            .quads
            .push(quad(1, scaled_bounds(2.0, 2.0, 12.0, 12.0), Hsla::blue()));
        first.finish();
        renderer.render_frame(&first, size, 1.0, Rgba::default())?;

        let mut reordered = Scene::default();
        reordered
            .quads
            .push(quad(1, scaled_bounds(2.0, 2.0, 12.0, 12.0), Hsla::red()));
        reordered
            .quads
            .push(quad(0, scaled_bounds(2.0, 2.0, 12.0, 12.0), Hsla::blue()));
        reordered.finish();
        let frame = renderer.render_frame(&reordered, size, 1.0, Rgba::default())?;
        assert_eq!(frame.rasterized_tiles, 1);
        assert_eq!(frame.framebuffer[2 + 2 * 128], 0x00ff_0000);

        let frame = renderer.render_frame(
            &reordered,
            size,
            1.0,
            Rgba {
                r: 1.0,
                g: 1.0,
                b: 1.0,
                a: 1.0,
            },
        )?;
        assert_eq!(frame.rasterized_tiles, 2);
        Ok(())
    }

    fn paint_surface(frame_revision: u64) -> Result<PaintSurface> {
        Ok(PaintSurface {
            order: 0,
            bounds: scaled_bounds(2.0, 2.0, 8.0, 8.0),
            content_mask: content_mask(64.0, 64.0),
            frame_revision,
            #[cfg(target_os = "macos")]
            image_buffer: core_video::pixel_buffer::CVPixelBuffer::new(
                core_video::pixel_buffer::kCVPixelFormatType_32BGRA,
                8,
                8,
                None,
            )
            .map_err(|status| anyhow::anyhow!("creating test CVPixelBuffer failed: {status}"))?,
        })
    }

    #[test]
    fn surface_frame_revision_invalidates_reused_buffers() -> Result<()> {
        let size = Size {
            width: DevicePixels(64),
            height: DevicePixels(64),
        };
        let mut renderer = SoftwareRenderer::new(Arc::new(SoftwareAtlas::new()));
        let mut first = Scene::default();
        first.surfaces.push(paint_surface(1)?);
        first.finish();
        renderer.render_frame(&first, size, 1.0, Rgba::default())?;

        let mut next = Scene::default();
        next.surfaces.push(paint_surface(2)?);
        next.finish();
        let frame = renderer.render_frame(&next, size, 1.0, Rgba::default())?;
        assert_eq!(frame.rasterized_tiles, 1);
        assert!(frame.changed());
        Ok(())
    }

    #[test]
    fn renders_quad_backgrounds_rounding_and_borders() -> Result<()> {
        let mut scene = Scene::default();
        let mut rounded = quad(0, scaled_bounds(2.0, 2.0, 26.0, 26.0), Hsla::red());
        rounded.corner_radii = Corners::all(ScaledPixels(6.0));
        rounded.border_widths = Edges::all(ScaledPixels(2.0));
        rounded.border_color = Hsla::blue();
        scene.quads.push(rounded);

        let mut gradient = quad(1, scaled_bounds(34.0, 2.0, 26.0, 26.0), Hsla::default());
        gradient.background = linear_gradient(
            90.0,
            linear_color_stop(Hsla::red(), 0.0),
            linear_color_stop(Hsla::blue(), 1.0),
        )
        .color_space(ColorSpace::Oklab);
        scene.quads.push(gradient);

        let mut slashes = quad(2, scaled_bounds(66.0, 2.0, 26.0, 26.0), Hsla::default());
        slashes.background = pattern_slash(Hsla::red(), 3.0, 3.0);
        scene.quads.push(slashes);

        let mut checks = quad(3, scaled_bounds(98.0, 2.0, 26.0, 26.0), Hsla::default());
        checks.background = checkerboard(Hsla::blue(), 4.0);
        checks.border_widths = Edges::all(ScaledPixels(2.0));
        checks.border_color = Hsla::red();
        checks.border_style = BorderStyle::Dashed;
        scene.quads.push(checks);

        let mut per_edge = quad(4, scaled_bounds(2.0, 34.0, 26.0, 26.0), Hsla::white());
        per_edge.border_widths = Edges {
            top: ScaledPixels(1.0),
            right: ScaledPixels(2.0),
            bottom: ScaledPixels(3.0),
            left: ScaledPixels(4.0),
        };
        per_edge.border_color = Hsla::red();
        scene.quads.push(per_edge);
        scene.finish();

        let mut renderer = SoftwareHeadlessRenderer::new(Arc::new(SoftwareAtlas::new()));
        let framebuffer = renderer.render(
            &scene,
            Size {
                width: DevicePixels(128),
                height: DevicePixels(64),
            },
            Rgba {
                r: 1.0,
                g: 1.0,
                b: 1.0,
                a: 1.0,
            },
        )?;
        for bounds in [
            [2, 2, 28, 28],
            [34, 2, 60, 28],
            [66, 2, 92, 28],
            [98, 2, 124, 28],
            [2, 34, 28, 60],
        ] {
            assert!(nonwhite_pixels(framebuffer, 128, bounds) > 20);
        }
        assert_eq!(framebuffer[2 + 2 * 128], 0x00ff_ffff);
        Ok(())
    }

    #[test]
    fn renders_srgb_and_oklab_gradients_in_their_selected_spaces() -> Result<()> {
        let mut center_pixels = Vec::new();
        for color_space in [ColorSpace::Srgb, ColorSpace::Oklab] {
            let mut scene = Scene::default();
            let mut gradient = quad(0, scaled_bounds(0.0, 0.0, 64.0, 16.0), Hsla::default());
            gradient.background = linear_gradient(
                90.0,
                linear_color_stop(Hsla::red(), 0.0),
                linear_color_stop(Hsla::blue(), 1.0),
            )
            .color_space(color_space);
            scene.quads.push(gradient);
            scene.finish();
            let mut renderer = SoftwareHeadlessRenderer::new(Arc::new(SoftwareAtlas::new()));
            let framebuffer = renderer.render(
                &scene,
                Size {
                    width: DevicePixels(64),
                    height: DevicePixels(16),
                },
                Rgba {
                    r: 1.0,
                    g: 1.0,
                    b: 1.0,
                    a: 1.0,
                },
            )?;
            center_pixels.push(framebuffer[32 + 8 * 64]);
        }
        assert_ne!(center_pixels[0], center_pixels[1]);
        assert!(center_pixels.iter().all(|pixel| *pixel != 0x00ff_ffff));
        Ok(())
    }

    #[test]
    fn renders_monochrome_subpixel_and_polychrome_sprites() -> Result<()> {
        let atlas = Arc::new(SoftwareAtlas::new());
        let monochrome = insert_tile(
            &atlas,
            AtlasKey::Svg(RenderSvgParams {
                path: "test.svg".into(),
                size: Size {
                    width: DevicePixels(2),
                    height: DevicePixels(2),
                },
            }),
            2,
            2,
            vec![255; 4],
        )?;
        let subpixel = insert_tile(
            &atlas,
            AtlasKey::Glyph(RenderGlyphParams {
                font_id: FontId(0),
                glyph_id: GlyphId(1),
                font_size: gpui::px(12.0),
                subpixel_variant: Point::new(0, 0),
                scale_factor: 1.0,
                is_emoji: false,
                subpixel_rendering: true,
                dilation: 0,
            }),
            2,
            2,
            vec![255; 16],
        )?;
        let polychrome = insert_tile(
            &atlas,
            AtlasKey::Image(RenderImageParams {
                image_id: ImageId(1),
                frame_index: 0,
            }),
            2,
            2,
            [255, 0, 0, 255].repeat(4),
        )?;

        let mut scene = Scene::default();
        scene.monochrome_sprites.push(MonochromeSprite {
            order: 0,
            pad: 0,
            bounds: scaled_bounds(4.0, 4.0, 12.0, 12.0),
            content_mask: content_mask(64.0, 24.0),
            color: Hsla::red(),
            tile: monochrome,
            transformation: TransformationMatrix::unit()
                .translate(Point::new(ScaledPixels(4.0), ScaledPixels(0.0))),
        });
        scene.subpixel_sprites.push(SubpixelSprite {
            order: 1,
            pad: 0,
            bounds: scaled_bounds(24.0, 4.0, 12.0, 12.0),
            content_mask: content_mask(64.0, 24.0),
            color: Hsla::red(),
            tile: subpixel,
            transformation: TransformationMatrix::unit(),
        });
        scene.polychrome_sprites.push(PolychromeSprite {
            order: 2,
            pad: 0,
            grayscale: false.into(),
            opacity: 1.0,
            bounds: scaled_bounds(44.0, 4.0, 12.0, 12.0),
            content_mask: content_mask(64.0, 24.0),
            corner_radii: Corners::all(ScaledPixels(2.0)),
            tile: polychrome,
        });
        scene.finish();
        let mut renderer = SoftwareHeadlessRenderer::new(atlas);
        let framebuffer = renderer.render(
            &scene,
            Size {
                width: DevicePixels(64),
                height: DevicePixels(24),
            },
            Rgba {
                r: 1.0,
                g: 1.0,
                b: 1.0,
                a: 1.0,
            },
        )?;
        assert_eq!(framebuffer[5 + 8 * 64], 0x00ff_ffff);
        assert_eq!(framebuffer[12 + 8 * 64], 0x00ff_0000);
        assert_eq!(framebuffer[28 + 8 * 64], 0x00ff_0000);
        assert_eq!(framebuffer[48 + 8 * 64], 0x0000_00ff);
        Ok(())
    }

    #[test]
    fn atlas_replacement_and_scale_changes_invalidate_tiles() -> Result<()> {
        let atlas = Arc::new(SoftwareAtlas::new());
        let key = AtlasKey::Svg(RenderSvgParams {
            path: "revision.svg".into(),
            size: Size {
                width: DevicePixels(1),
                height: DevicePixels(1),
            },
        });
        let first_tile = insert_tile(&atlas, key.clone(), 1, 1, vec![255])?;
        let mut scene = Scene::default();
        scene.monochrome_sprites.push(MonochromeSprite {
            order: 0,
            pad: 0,
            bounds: scaled_bounds(2.0, 2.0, 4.0, 4.0),
            content_mask: content_mask(64.0, 64.0),
            color: Hsla::red(),
            tile: first_tile,
            transformation: TransformationMatrix::unit(),
        });
        scene.finish();
        let size = Size {
            width: DevicePixels(64),
            height: DevicePixels(64),
        };
        let mut renderer = SoftwareRenderer::new(atlas.clone());
        renderer.render_frame(&scene, size, 1.0, Rgba::default())?;
        assert_eq!(
            renderer
                .render_frame(&scene, size, 1.0, Rgba::default())?
                .rasterized_tiles,
            0
        );

        atlas.remove(&key);
        let replacement = insert_tile(&atlas, key, 1, 1, vec![128])?;
        assert_eq!(replacement, first_tile);
        assert_eq!(
            renderer
                .render_frame(&scene, size, 1.0, Rgba::default())?
                .rasterized_tiles,
            1
        );
        assert_eq!(
            renderer
                .render_frame(&scene, size, 2.0, Rgba::default())?
                .rasterized_tiles,
            1
        );
        Ok(())
    }

    #[test]
    fn renders_shadows_underlines_and_path_forms() -> Result<()> {
        let mut scene = Scene::default();
        scene.shadows.push(Shadow {
            order: 0,
            blur_radius: ScaledPixels(3.0),
            bounds: scaled_bounds(8.0, 8.0, 20.0, 20.0),
            corner_radii: Corners::all(ScaledPixels(4.0)),
            content_mask: content_mask(128.0, 64.0),
            color: Hsla::red(),
            element_bounds: scaled_bounds(8.0, 8.0, 20.0, 20.0),
            element_corner_radii: Corners::all(ScaledPixels(4.0)),
            inset: 0,
            pad: 0,
        });
        scene.shadows.push(Shadow {
            order: 1,
            blur_radius: ScaledPixels(2.0),
            bounds: scaled_bounds(38.0, 12.0, 12.0, 12.0),
            corner_radii: Corners::all(ScaledPixels(2.0)),
            content_mask: content_mask(128.0, 64.0),
            color: Hsla::blue(),
            element_bounds: scaled_bounds(34.0, 8.0, 20.0, 20.0),
            element_corner_radii: Corners::all(ScaledPixels(4.0)),
            inset: 1,
            pad: 0,
        });

        let path_mask = ContentMask {
            bounds: gpui::Bounds {
                origin: Point::new(gpui::px(0.0), gpui::px(0.0)),
                size: Size {
                    width: gpui::px(128.0),
                    height: gpui::px(64.0),
                },
            },
        };
        let mut canonical = Path::new(Point::new(gpui::px(64.0), gpui::px(8.0)));
        canonical.line_to(Point::new(gpui::px(88.0), gpui::px(8.0)));
        canonical.curve_to(
            Point::new(gpui::px(76.0), gpui::px(28.0)),
            Point::new(gpui::px(90.0), gpui::px(24.0)),
        );
        canonical.close();
        canonical.color = solid_background(Hsla::red());
        canonical.content_mask = path_mask;
        let mut canonical = canonical.scale(1.0)?;
        canonical.order = 2;
        scene.paths.push(canonical);

        let mut stroke = Path::new(Point::new(gpui::px(94.0), gpui::px(8.0)));
        stroke.cubic_curve_to(
            Point::new(gpui::px(120.0), gpui::px(26.0)),
            Point::new(gpui::px(104.0), gpui::px(2.0)),
            Point::new(gpui::px(112.0), gpui::px(30.0)),
        );
        stroke.set_stroke(PathStroke {
            width: 2.0,
            ..PathStroke::default()
        })?;
        stroke.color = solid_background(Hsla::blue());
        stroke.content_mask = path_mask;
        let mut stroke = stroke.scale(1.0)?;
        stroke.order = 3;
        scene.paths.push(stroke);

        let mut triangles = Path::new(Point::new(gpui::px(60.0), gpui::px(36.0)));
        triangles.push_triangle(
            (
                Point::new(gpui::px(60.0), gpui::px(36.0)),
                Point::new(gpui::px(76.0), gpui::px(60.0)),
                Point::new(gpui::px(44.0), gpui::px(60.0)),
            ),
            (
                Point::new(0.0, 1.0),
                Point::new(0.0, 1.0),
                Point::new(0.0, 1.0),
            ),
        );
        triangles.color = solid_background(Hsla::blue());
        triangles.content_mask = path_mask;
        let mut triangles = triangles.scale(1.0)?;
        triangles.order = 4;
        scene.paths.push(triangles);

        scene.underlines.push(Underline {
            order: 5,
            pad: 0,
            bounds: scaled_bounds(4.0, 42.0, 28.0, 2.0),
            content_mask: content_mask(128.0, 64.0),
            color: Hsla::red(),
            thickness: ScaledPixels(2.0),
            wavy: false.into(),
        });
        scene.underlines.push(Underline {
            order: 6,
            pad: 0,
            bounds: scaled_bounds(92.0, 42.0, 30.0, 6.0),
            content_mask: content_mask(128.0, 64.0),
            color: Hsla::blue(),
            thickness: ScaledPixels(1.5),
            wavy: true.into(),
        });
        scene.finish();

        let mut renderer = SoftwareHeadlessRenderer::new(Arc::new(SoftwareAtlas::new()));
        let framebuffer = renderer.render(
            &scene,
            Size {
                width: DevicePixels(128),
                height: DevicePixels(64),
            },
            Rgba {
                r: 1.0,
                g: 1.0,
                b: 1.0,
                a: 1.0,
            },
        )?;
        for bounds in [
            [4, 4, 32, 32],
            [34, 8, 54, 28],
            [64, 8, 90, 29],
            [94, 2, 122, 31],
            [44, 36, 77, 61],
            [4, 40, 32, 46],
            [92, 38, 123, 50],
        ] {
            assert!(nonwhite_pixels(framebuffer, 128, bounds) > 0);
        }
        Ok(())
    }

    #[test]
    fn movement_and_removal_dirty_old_and_new_tiles() -> Result<()> {
        let size = Size {
            width: DevicePixels(128),
            height: DevicePixels(64),
        };
        let mut renderer = SoftwareRenderer::new(Arc::new(SoftwareAtlas::new()));
        let mut first = Scene::default();
        first
            .quads
            .push(quad(0, scaled_bounds(2.0, 2.0, 4.0, 4.0), Hsla::red()));
        first.finish();
        renderer.render_frame(&first, size, 1.0, Rgba::default())?;

        let mut moved = Scene::default();
        moved
            .quads
            .push(quad(0, scaled_bounds(66.0, 2.0, 4.0, 4.0), Hsla::red()));
        moved.finish();
        let frame = renderer.render_frame(&moved, size, 1.0, Rgba::default())?;
        assert_eq!(frame.rasterized_tiles, 2);
        assert_eq!(
            frame.damage,
            &[SoftwareDamageRect {
                x: 0,
                y: 0,
                width: 128,
                height: 64,
            }]
        );

        let empty = Scene::default();
        let frame = renderer.render_frame(&empty, size, 1.0, Rgba::default())?;
        assert_eq!(frame.rasterized_tiles, 1);
        assert_eq!(frame.damage[0].x, 64);
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
        assert!(
            renderer
                .render(
                    &scene,
                    Size {
                        width: DevicePixels(i32::MAX),
                        height: DevicePixels(i32::MAX),
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

    #[test]
    fn extreme_radii_and_offscreen_primitives_are_bounded() -> Result<()> {
        let mut scene = Scene::default();
        let mut rounded = quad(0, scaled_bounds(2.0, 2.0, 12.0, 12.0), Hsla::red());
        rounded.corner_radii = Corners::all(ScaledPixels(f32::MAX));
        scene.quads.push(rounded);
        scene.quads.push(quad(
            1,
            scaled_bounds(-1.0e20, -1.0e20, 10.0, 10.0),
            Hsla::blue(),
        ));
        scene.finish();
        let mut renderer = SoftwareHeadlessRenderer::new(Arc::new(SoftwareAtlas::new()));
        let framebuffer = renderer.render(
            &scene,
            Size {
                width: DevicePixels(16),
                height: DevicePixels(16),
            },
            Rgba::default(),
        )?;
        assert_eq!(framebuffer.len(), 256);
        Ok(())
    }
}
