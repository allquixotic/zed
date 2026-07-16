use crate::software_scene::{
    AtlasImageData, BackgroundData, ColorData, CommandKind, CornersData, EdgesData,
    PathElementData, PathGeometryData, PathPaintData, PointData, RectData, SoftwareCommand,
    StrokeData, TransformData,
};
use crate::software_surface::{SurfaceImageData, SurfaceImageKind};
use anyhow::{Context as _, Result, ensure};
use gpui::{
    ColorSpace, PathFillRule, PathLineCap, PathLineJoin, Rgba, get_gamma_correction_ratios,
};
use vello_cpu::{
    Level, Pixmap, RenderContext, RenderMode, RenderSettings, Resources,
    color::{AlphaColor, ColorSpaceTag, PremulRgba8, Srgb},
    kurbo::{Affine, BezPath, Cap, Join, Rect, RoundedRect, Shape, Stroke},
    peniko::{BlendMode, Compose, Extend, Gradient, InterpolationAlphaSpace},
};

pub const SOFTWARE_TILE_SIZE: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoftwareTextRenderingParams {
    pub gamma_ratios: [f32; 4],
    pub grayscale_enhanced_contrast: f32,
    pub subpixel_enhanced_contrast: f32,
    pub is_bgr: bool,
}

impl Default for SoftwareTextRenderingParams {
    fn default() -> Self {
        Self {
            gamma_ratios: get_gamma_correction_ratios(1.8),
            grayscale_enhanced_contrast: 1.0,
            subpixel_enhanced_contrast: 0.5,
            is_bgr: false,
        }
    }
}

#[derive(Default)]
pub struct SoftwareRasterizer {
    target: Option<RasterTarget>,
    size: [u16; 2],
    tile_origin: [usize; 2],
    text_rendering: SoftwareTextRenderingParams,
    xrgb: Vec<u32>,
}

struct RasterTarget {
    context: RenderContext,
    resources: Resources,
    pixmap: Pixmap,
    has_pending_commands: bool,
}

impl SoftwareRasterizer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn begin(&mut self, size: [u16; 2], background: Rgba) -> Result<()> {
        self.begin_tile(
            size,
            [0, 0],
            background,
            SoftwareTextRenderingParams::default(),
        )
    }

    pub fn begin_tile(
        &mut self,
        size: [u16; 2],
        tile_origin: [usize; 2],
        background: Rgba,
        text_rendering: SoftwareTextRenderingParams,
    ) -> Result<()> {
        ensure!(
            usize::from(size[0]) <= SOFTWARE_TILE_SIZE
                && usize::from(size[1]) <= SOFTWARE_TILE_SIZE,
            "software rasterizer target exceeds {SOFTWARE_TILE_SIZE}x{SOFTWARE_TILE_SIZE}"
        );
        validate_color(background)?;
        validate_text_rendering(text_rendering)?;
        self.size = size;
        self.tile_origin = tile_origin;
        self.text_rendering = text_rendering;
        if size[0] == 0 || size[1] == 0 {
            self.target = None;
            self.xrgb.clear();
            return Ok(());
        }

        let target_size_is_unchanged = self.target.as_ref().is_some_and(|target| {
            target.pixmap.width() == size[0] && target.pixmap.height() == size[1]
        });
        if !target_size_is_unchanged {
            let settings = RenderSettings {
                level: Level::try_detect().unwrap_or_else(Level::baseline),
                num_threads: 0,
                render_mode: RenderMode::OptimizeSpeed,
            };
            self.target = Some(RasterTarget {
                context: RenderContext::new_with(size[0], size[1], settings),
                resources: Resources::new(),
                pixmap: Pixmap::new(size[0], size[1]),
                has_pending_commands: false,
            });
            let pixel_count = usize::from(size[0])
                .checked_mul(usize::from(size[1]))
                .context("software raster tile dimensions overflowed")?;
            self.xrgb.clear();
            self.xrgb
                .try_reserve_exact(pixel_count)
                .context("allocating software raster tile")?;
            self.xrgb.resize(pixel_count, 0);
        }

        let transform = tile_transform(tile_origin)?;
        let target = self
            .target
            .as_mut()
            .context("software raster target was not initialized")?;
        target.context.reset();
        target.context.set_transform(transform);
        target.has_pending_commands = false;
        let background = Rgba {
            a: 1.0,
            ..background
        };
        let background = premultiplied_color(background);
        target.pixmap.data_mut().fill(background);
        target.pixmap.set_may_have_transparency(false);
        Ok(())
    }

    pub(crate) fn render_commands(&mut self, commands: &[SoftwareCommand]) -> Result<()> {
        for command in commands {
            match &command.kind {
                CommandKind::Quad {
                    border_style,
                    bounds,
                    background,
                    border_color,
                    corner_radii,
                    border_widths,
                } => self.render_quad(
                    command.clip,
                    *border_style,
                    *bounds,
                    *background,
                    *border_color,
                    *corner_radii,
                    *border_widths,
                )?,
                CommandKind::Shadow {
                    blur_radius,
                    bounds,
                    corner_radii,
                    color,
                    element_bounds,
                    element_corner_radii,
                    inset,
                } => self.render_shadow(
                    command.clip,
                    f64::from(blur_radius.get()),
                    *bounds,
                    *corner_radii,
                    *color,
                    *element_bounds,
                    *element_corner_radii,
                    *inset,
                )?,
                CommandKind::Path {
                    color,
                    paint,
                    geometry,
                    ..
                } => self.render_path(command.clip, *color, *paint, geometry)?,
                CommandKind::Underline {
                    bounds,
                    color,
                    thickness,
                    wavy,
                } => self.render_underline(
                    command.clip,
                    *bounds,
                    *color,
                    f64::from(thickness.get()),
                    *wavy,
                )?,
                CommandKind::MonochromeSprite {
                    bounds,
                    color,
                    image,
                    transform,
                } => {
                    self.flush_segment()?;
                    self.render_monochrome_sprite(
                        command.clip,
                        *bounds,
                        *color,
                        image,
                        *transform,
                    )?;
                }
                CommandKind::SubpixelSprite {
                    bounds,
                    color,
                    image,
                    transform,
                } => {
                    self.flush_segment()?;
                    self.render_subpixel_sprite(command.clip, *bounds, *color, image, *transform)?;
                }
                CommandKind::PolychromeSprite {
                    bounds,
                    grayscale,
                    opacity,
                    corner_radii,
                    image,
                } => {
                    self.flush_segment()?;
                    self.render_polychrome_sprite(
                        command.clip,
                        *bounds,
                        *grayscale,
                        opacity.get(),
                        *corner_radii,
                        image,
                    )?;
                }
                CommandKind::Surface {
                    bounds,
                    frame_revision,
                    image,
                } => {
                    self.flush_segment()?;
                    self.render_surface(command.clip, *bounds, *frame_revision, image)?;
                }
            }
        }
        Ok(())
    }

    pub fn fill_rectangle(&mut self, rect: [f64; 4], color: Rgba) -> Result<()> {
        ensure!(
            rect.into_iter().all(f64::is_finite),
            "software raster rectangle contains a non-finite coordinate"
        );
        validate_color(color)?;
        if rect[2] <= rect[0] || rect[3] <= rect[1] || color.a <= 0.0 {
            return Ok(());
        }
        let target = self.target_mut()?;
        target.context.set_paint(vello_color(color));
        target
            .context
            .fill_rect(&Rect::new(rect[0], rect[1], rect[2], rect[3]));
        target.has_pending_commands = true;
        Ok(())
    }

    pub fn finish(&mut self) -> Result<&[u32]> {
        self.flush_segment()?;
        let Some(target) = self.target.as_ref() else {
            ensure!(
                self.size[0] == 0 || self.size[1] == 0,
                "software raster target disappeared"
            );
            return Ok(&self.xrgb);
        };
        let rgba = target.pixmap.data();
        ensure!(
            rgba.len() == self.xrgb.len(),
            "Vello returned {} pixels for a {}-pixel tile",
            rgba.len(),
            self.xrgb.len()
        );
        for (destination, source) in self.xrgb.iter_mut().zip(rgba) {
            *destination =
                (u32::from(source.r) << 16) | (u32::from(source.g) << 8) | u32::from(source.b);
        }
        Ok(&self.xrgb)
    }

    fn render_quad(
        &mut self,
        clip: RectData,
        border_style: gpui::BorderStyle,
        bounds: RectData,
        background: BackgroundData,
        border_color: ColorData,
        corner_radii: CornersData,
        border_widths: EdgesData,
    ) -> Result<()> {
        let Some(_) = bounds.intersect(clip) else {
            return Ok(());
        };
        let bounds_rect = rect(bounds);
        if bounds_rect.width() <= 0.0 || bounds_rect.height() <= 0.0 {
            return Ok(());
        }
        let outer_path = rounded_path(bounds, corner_radii);
        let clip_path = rect(clip).to_path(0.1);
        self.push_clip(&clip_path)?;
        self.fill_background(&outer_path, bounds, background)?;

        let widths = border_widths.get().map(|width| width.max(0.0));
        if widths.iter().all(|width| *width <= 0.0) || border_color.alpha.get() <= 0.0 {
            self.pop_clip()?;
            return Ok(());
        }
        self.push_clip(&outer_path)?;
        let uniform = widths
            .iter()
            .all(|width| (*width - widths[0]).abs() <= f64::EPSILON);
        if uniform {
            self.stroke_uniform_border(
                bounds,
                corner_radii,
                widths[0],
                border_style,
                border_color,
            )?;
        } else {
            self.fill_edge_borders(bounds_rect, widths, border_style, border_color)?;
        }
        self.pop_clip()?;
        self.pop_clip()
    }

    fn stroke_uniform_border(
        &mut self,
        bounds: RectData,
        corners: CornersData,
        width: f64,
        border_style: gpui::BorderStyle,
        color: ColorData,
    ) -> Result<()> {
        if width <= 0.0 {
            return Ok(());
        }
        let edges = bounds.edges();
        let inset = width * 0.5;
        let inset_bounds = RectData::new(
            (edges[0] + inset) as f32,
            (edges[1] + inset) as f32,
            (edges[2] - edges[0] - width).max(0.0) as f32,
            (edges[3] - edges[1] - width).max(0.0) as f32,
            "software inset border bounds",
        )?;
        let radii = corners.get().map(|radius| (radius - inset).max(0.0));
        let path = rounded_path_from_radii(inset_bounds, radii);
        let mut stroke = Stroke::new(width);
        if border_style == gpui::BorderStyle::Dashed {
            stroke.dash_pattern.extend([width * 2.0, width]);
        }
        let target = self.target_mut()?;
        target.context.set_paint(vello_color(color.get()));
        target.context.set_stroke(stroke);
        target.context.stroke_path(&path);
        target.has_pending_commands = true;
        Ok(())
    }

    fn fill_edge_borders(
        &mut self,
        bounds: Rect,
        widths: [f64; 4],
        border_style: gpui::BorderStyle,
        color: ColorData,
    ) -> Result<()> {
        let edges = [
            Rect::new(bounds.x0, bounds.y0, bounds.x1, bounds.y0 + widths[0]),
            Rect::new(bounds.x1 - widths[1], bounds.y0, bounds.x1, bounds.y1),
            Rect::new(bounds.x0, bounds.y1 - widths[2], bounds.x1, bounds.y1),
            Rect::new(bounds.x0, bounds.y0, bounds.x0 + widths[3], bounds.y1),
        ];
        for (index, edge) in edges.into_iter().enumerate() {
            if widths[index] <= 0.0 {
                continue;
            }
            if border_style == gpui::BorderStyle::Solid {
                self.fill_vello_rect(edge, color.get())?;
            } else {
                self.fill_dashed_edge(edge, widths[index], index % 2 == 0, color.get())?;
            }
        }
        Ok(())
    }

    fn fill_dashed_edge(
        &mut self,
        edge: Rect,
        width: f64,
        horizontal: bool,
        color: Rgba,
    ) -> Result<()> {
        let start = if horizontal { edge.x0 } else { edge.y0 };
        let end = if horizontal { edge.x1 } else { edge.y1 };
        let dash = width * 2.0;
        let period = width * 3.0;
        if period <= 0.0 {
            return Ok(());
        }
        let mut position = start;
        while position < end {
            let dash_end = (position + dash).min(end);
            let rect = if horizontal {
                Rect::new(position, edge.y0, dash_end, edge.y1)
            } else {
                Rect::new(edge.x0, position, edge.x1, dash_end)
            };
            self.fill_vello_rect(rect, color)?;
            position += period;
        }
        Ok(())
    }

    fn render_shadow(
        &mut self,
        clip: RectData,
        blur_radius: f64,
        bounds: RectData,
        corner_radii: CornersData,
        color: ColorData,
        element_bounds: RectData,
        element_corner_radii: CornersData,
        inset: bool,
    ) -> Result<()> {
        let clip_path = rect(clip).to_path(0.1);
        self.push_clip(&clip_path)?;
        if inset {
            let element_path = rounded_path(element_bounds, element_corner_radii);
            {
                let target = self.target_mut()?;
                target
                    .context
                    .push_layer(Some(&element_path), None, None, None, None);
                target.context.set_paint(vello_color(color.get()));
                target.context.fill_path(&element_path);
                target
                    .context
                    .set_blend_mode(BlendMode::from(Compose::DestOut));
                target.context.set_paint(vello_color(Rgba {
                    r: 1.0,
                    g: 1.0,
                    b: 1.0,
                    a: 1.0,
                }));
                if blur_radius > 0.0 {
                    target.context.fill_blurred_rounded_rect(
                        &rect(bounds),
                        average_radius(corner_radii) as f32,
                        blur_radius as f32,
                    );
                } else {
                    target
                        .context
                        .fill_path(&rounded_path(bounds, corner_radii));
                }
                target.context.set_blend_mode(BlendMode::default());
                target.context.pop_layer();
                target.has_pending_commands = true;
            }
        } else {
            let target = self.target_mut()?;
            target.context.set_paint(vello_color(color.get()));
            if blur_radius > 0.0 {
                target.context.fill_blurred_rounded_rect(
                    &rect(bounds),
                    average_radius(corner_radii) as f32,
                    blur_radius as f32,
                );
            } else {
                target
                    .context
                    .fill_path(&rounded_path(bounds, corner_radii));
            }
            target.has_pending_commands = true;
        }
        self.pop_clip()
    }

    fn render_path(
        &mut self,
        clip: RectData,
        background: BackgroundData,
        paint: PathPaintData,
        geometry: &PathGeometryData,
    ) -> Result<()> {
        let path = match geometry {
            PathGeometryData::Canonical(elements) => canonical_path(elements),
            PathGeometryData::Triangles(vertices) => triangle_path(vertices),
        };
        if path.is_empty() {
            return Ok(());
        }
        let clip_path = rect(clip).to_path(0.1);
        self.push_clip(&clip_path)?;
        match paint {
            PathPaintData::Fill(fill_rule) => {
                let target = self.target_mut()?;
                target.context.set_fill_rule(match fill_rule {
                    PathFillRule::NonZero => vello_cpu::peniko::Fill::NonZero,
                    PathFillRule::EvenOdd => vello_cpu::peniko::Fill::EvenOdd,
                });
                self.fill_background(&path, path_bounds(&path)?, background)?;
            }
            PathPaintData::Stroke(stroke) => {
                self.set_background_paint(path_bounds(&path)?, background)?;
                let stroke = vello_stroke(stroke);
                let target = self.target_mut()?;
                target.context.set_stroke(stroke);
                target.context.stroke_path(&path);
                target.has_pending_commands = true;
            }
        }
        self.pop_clip()
    }

    fn render_underline(
        &mut self,
        clip: RectData,
        bounds: RectData,
        color: ColorData,
        thickness: f64,
        wavy: bool,
    ) -> Result<()> {
        if thickness <= 0.0 {
            return Ok(());
        }
        let clip_path = rect(clip).to_path(0.1);
        self.push_clip(&clip_path)?;
        if !wavy {
            self.fill_vello_rect(rect(bounds), color.get())?;
        } else {
            let edges = bounds.edges();
            let amplitude = ((edges[3] - edges[1]) * 0.5).max(thickness);
            let wavelength = (amplitude * 4.0).max(thickness * 4.0);
            let center_y = (edges[1] + edges[3]) * 0.5;
            let mut path = BezPath::new();
            path.move_to((edges[0], center_y));
            let mut x = edges[0];
            while x < edges[2] {
                let quarter = wavelength * 0.25;
                path.curve_to(
                    (x + quarter * 0.5, center_y - amplitude),
                    (x + quarter * 0.5, center_y - amplitude),
                    (x + quarter, center_y - amplitude),
                );
                path.curve_to(
                    (x + quarter * 1.5, center_y - amplitude),
                    (x + quarter * 1.5, center_y),
                    (x + quarter * 2.0, center_y),
                );
                path.curve_to(
                    (x + quarter * 2.5, center_y + amplitude),
                    (x + quarter * 2.5, center_y + amplitude),
                    (x + quarter * 3.0, center_y + amplitude),
                );
                path.curve_to(
                    (x + quarter * 3.5, center_y + amplitude),
                    (x + quarter * 3.5, center_y),
                    (x + wavelength, center_y),
                );
                x += wavelength;
            }
            let target = self.target_mut()?;
            target.context.set_paint(vello_color(color.get()));
            target.context.set_stroke(Stroke::new(thickness));
            target.context.stroke_path(&path);
            target.has_pending_commands = true;
        }
        self.pop_clip()
    }

    fn fill_background(
        &mut self,
        path: &BezPath,
        bounds: RectData,
        background: BackgroundData,
    ) -> Result<()> {
        match background {
            BackgroundData::Solid(_) | BackgroundData::LinearGradient { .. } => {
                self.set_background_paint(bounds, background)?;
                let target = self.target_mut()?;
                target.context.fill_path(path);
                target.has_pending_commands = true;
            }
            BackgroundData::PatternSlash {
                color,
                width,
                interval,
            } => {
                self.push_clip(path)?;
                self.fill_slash_pattern(bounds, color, width.get(), interval.get())?;
                self.pop_clip()?;
            }
            BackgroundData::Checkerboard { color, size } => {
                self.push_clip(path)?;
                self.fill_checkerboard(bounds, color, size.get())?;
                self.pop_clip()?;
            }
        }
        Ok(())
    }

    fn set_background_paint(&mut self, bounds: RectData, background: BackgroundData) -> Result<()> {
        let target = self.target_mut()?;
        match background {
            BackgroundData::Solid(color) => target.context.set_paint(vello_color(color.get())),
            BackgroundData::LinearGradient {
                angle,
                color_space,
                stops,
            } => {
                let edges = bounds.edges();
                let width = edges[2] - edges[0];
                let height = edges[3] - edges[1];
                let center = [(edges[0] + edges[2]) * 0.5, (edges[1] + edges[3]) * 0.5];
                let radians = f64::from(angle.get()).rem_euclid(360.0).to_radians()
                    - std::f64::consts::FRAC_PI_2;
                let mut direction = [radians.cos(), radians.sin()];
                if width > height && width > 0.0 {
                    direction[1] *= height / width;
                } else if height > 0.0 {
                    direction[0] *= width / height;
                }
                let magnitude = direction[0].hypot(direction[1]);
                if magnitude <= f64::EPSILON {
                    target.context.set_paint(vello_color(stops[1].color.get()));
                    return Ok(());
                }
                direction[0] /= magnitude;
                direction[1] /= magnitude;
                let half_length = if direction[0].abs() > direction[1].abs() {
                    width * 0.5
                } else {
                    height * 0.5
                };
                let start = (
                    center[0] - direction[0] * half_length,
                    center[1] - direction[1] * half_length,
                );
                let end = (
                    center[0] + direction[0] * half_length,
                    center[1] + direction[1] * half_length,
                );
                let stop_values = [
                    (stops[0].percentage.get(), vello_color(stops[0].color.get())),
                    (stops[1].percentage.get(), vello_color(stops[1].color.get())),
                ];
                let gradient = Gradient::new_linear(start, end)
                    .with_extend(Extend::Pad)
                    .with_interpolation_cs(match color_space {
                        ColorSpace::Srgb => ColorSpaceTag::Srgb,
                        ColorSpace::Oklab => ColorSpaceTag::Oklab,
                    })
                    .with_interpolation_alpha_space(InterpolationAlphaSpace::Unpremultiplied)
                    .with_stops(stop_values.as_slice());
                target.context.set_paint(gradient);
            }
            BackgroundData::PatternSlash { .. } | BackgroundData::Checkerboard { .. } => {
                anyhow::bail!("pattern paint requires a clip shape")
            }
        }
        Ok(())
    }

    fn fill_slash_pattern(
        &mut self,
        bounds: RectData,
        color: ColorData,
        width: f32,
        interval: f32,
    ) -> Result<()> {
        let width = f64::from(width).max(0.0);
        let interval = f64::from(interval).max(0.0);
        let period = width + interval;
        if width <= 0.0 || period <= 0.0 {
            return Ok(());
        }
        let edges = bounds.edges();
        let height = edges[3] - edges[1];
        let mut x = edges[0] - height - period;
        while x < edges[2] + period {
            let mut path = BezPath::new();
            path.move_to((x, edges[3]));
            path.line_to((x + height, edges[1]));
            let target = self.target_mut()?;
            target.context.set_paint(vello_color(color.get()));
            target.context.set_stroke(Stroke::new(width));
            target.context.stroke_path(&path);
            target.has_pending_commands = true;
            x += period * std::f64::consts::SQRT_2;
        }
        Ok(())
    }

    fn fill_checkerboard(&mut self, bounds: RectData, color: ColorData, size: f32) -> Result<()> {
        let size = f64::from(size);
        if !size.is_finite() || size <= 0.0 {
            return Ok(());
        }
        let edges = bounds.edges();
        let tile_left = self.tile_origin[0] as f64;
        let tile_top = self.tile_origin[1] as f64;
        let tile_right = tile_left + f64::from(self.size[0]);
        let tile_bottom = tile_top + f64::from(self.size[1]);
        let left = edges[0].max(tile_left);
        let top = edges[1].max(tile_top);
        let right = edges[2].min(tile_right);
        let bottom = edges[3].min(tile_bottom);
        if left >= right || top >= bottom {
            return Ok(());
        }
        if size < 1.0 {
            for local_y in 0..self.size[1] {
                let y = tile_top + f64::from(local_y);
                let center_y = y + 0.5;
                if center_y < top || center_y >= bottom {
                    continue;
                }
                let row = checker_index_parity(center_y, edges[1], size);
                for local_x in 0..self.size[0] {
                    let x = tile_left + f64::from(local_x);
                    let center_x = x + 0.5;
                    if center_x < left || center_x >= right {
                        continue;
                    }
                    let column = checker_index_parity(center_x, edges[0], size);
                    if (row + column) % 2 == 1 {
                        self.fill_vello_rect(
                            Rect::new(
                                x.max(left),
                                y.max(top),
                                (x + 1.0).min(right),
                                (y + 1.0).min(bottom),
                            ),
                            color.get(),
                        )?;
                    }
                }
            }
            return Ok(());
        }

        let column_offset = ((left - edges[0]) / size).floor();
        let row_offset = ((top - edges[1]) / size).floor();
        let mut row = row_offset.rem_euclid(2.0) as usize;
        let mut y = left_aligned_pattern_origin(top, edges[1], size);
        while y < bottom {
            let mut column = column_offset.rem_euclid(2.0) as usize;
            let mut x = left_aligned_pattern_origin(left, edges[0], size);
            while x < right {
                if (row + column) % 2 == 1 {
                    self.fill_vello_rect(
                        Rect::new(
                            x.max(left),
                            y.max(top),
                            (x + size).min(right),
                            (y + size).min(bottom),
                        ),
                        color.get(),
                    )?;
                }
                column = column.saturating_add(1);
                x += size;
            }
            row = row.saturating_add(1);
            y += size;
        }
        Ok(())
    }

    fn render_monochrome_sprite(
        &mut self,
        clip: RectData,
        bounds: RectData,
        color: ColorData,
        image: &AtlasImageData,
        transform: TransformData,
    ) -> Result<()> {
        ensure!(
            image.bytes_per_pixel == 1,
            "monochrome atlas tile is not one byte per pixel"
        );
        let text_rendering = self.text_rendering;
        self.for_each_sprite_pixel(clip, bounds, transform, None, |target, index, source| {
            let coverage = sample_monochrome(image, source)?;
            let color = color.get();
            let corrected = apply_contrast_and_gamma(
                coverage,
                [color.r, color.g, color.b],
                text_rendering.grayscale_enhanced_contrast,
                text_rendering.gamma_ratios,
            );
            blend_straight(
                target,
                index,
                [color.r, color.g, color.b],
                color.a * corrected,
            )
        })
    }

    fn render_subpixel_sprite(
        &mut self,
        clip: RectData,
        bounds: RectData,
        color: ColorData,
        image: &AtlasImageData,
        transform: TransformData,
    ) -> Result<()> {
        ensure!(
            image.bytes_per_pixel == 4,
            "subpixel atlas tile is not BGRA"
        );
        let text_rendering = self.text_rendering;
        self.for_each_sprite_pixel(clip, bounds, transform, None, |target, index, source| {
            let sample = sample_bgra(image, source)?;
            let mut coverage = [sample[0], sample[1], sample[2]];
            if text_rendering.is_bgr {
                coverage.reverse();
            }
            let color = color.get();
            let corrected = apply_subpixel_contrast_and_gamma(
                coverage,
                [color.r, color.g, color.b],
                text_rendering.subpixel_enhanced_contrast,
                text_rendering.gamma_ratios,
            );
            blend_subpixel(
                target,
                index,
                [color.r, color.g, color.b],
                corrected.map(|alpha| alpha * color.a),
            )
        })
    }

    fn render_polychrome_sprite(
        &mut self,
        clip: RectData,
        bounds: RectData,
        grayscale: bool,
        opacity: f32,
        corners: CornersData,
        image: &AtlasImageData,
    ) -> Result<()> {
        ensure!(
            image.bytes_per_pixel == 4,
            "polychrome atlas tile is not BGRA"
        );
        ensure!(opacity.is_finite(), "software image opacity is not finite");
        self.for_each_sprite_pixel(
            clip,
            bounds,
            TransformData::identity(),
            Some(corners),
            |target, index, source| {
                let mut sample = sample_bgra(image, source)?;
                if grayscale {
                    let gray = sample[0] * 0.2126 + sample[1] * 0.7152 + sample[2] * 0.0722;
                    sample[0] = gray;
                    sample[1] = gray;
                    sample[2] = gray;
                }
                blend_straight(
                    target,
                    index,
                    [sample[0], sample[1], sample[2]],
                    sample[3] * opacity.clamp(0.0, 1.0),
                )
            },
        )
    }

    fn for_each_sprite_pixel(
        &mut self,
        clip: RectData,
        bounds: RectData,
        transform: TransformData,
        corners: Option<CornersData>,
        mut paint: impl FnMut(&mut RasterTarget, usize, [f64; 2]) -> Result<()>,
    ) -> Result<()> {
        let inverse = transform
            .inverse()
            .context("software sprite transform is singular")?;
        let destination = bounds.edges();
        if destination[2] <= destination[0] || destination[3] <= destination[1] {
            return Ok(());
        }
        let clip = clip.edges();
        let tile_origin = self.tile_origin;
        let tile_width = usize::from(self.size[0]);
        let tile_height = usize::from(self.size[1]);
        let target = self.target_mut()?;
        for local_y in 0..tile_height {
            for local_x in 0..tile_width {
                let global = [
                    tile_origin[0] as f64 + local_x as f64 + 0.5,
                    tile_origin[1] as f64 + local_y as f64 + 0.5,
                ];
                if global[0] < clip[0]
                    || global[0] >= clip[2]
                    || global[1] < clip[1]
                    || global[1] >= clip[3]
                {
                    continue;
                }
                let point = inverse.apply(global);
                if point[0] < destination[0]
                    || point[0] >= destination[2]
                    || point[1] < destination[1]
                    || point[1] >= destination[3]
                {
                    continue;
                }
                if corners.is_some_and(|corners| !inside_rounded_rect(point, bounds, corners)) {
                    continue;
                }
                let source = [
                    (point[0] - destination[0]) / (destination[2] - destination[0]),
                    (point[1] - destination[1]) / (destination[3] - destination[1]),
                ];
                let index = local_y
                    .checked_mul(tile_width)
                    .and_then(|row| row.checked_add(local_x))
                    .context("software sprite destination index overflowed")?;
                paint(target, index, source)?;
            }
        }
        Ok(())
    }

    fn render_surface(
        &mut self,
        clip: RectData,
        bounds: RectData,
        frame_revision: u64,
        image: &SurfaceImageData,
    ) -> Result<()> {
        let tile_origin = self.tile_origin;
        let tile_width = usize::from(self.size[0]);
        let tile_height = usize::from(self.size[1]);
        let clipped = bounds.intersect(clip);
        let Some(clipped) = clipped else {
            return Ok(());
        };
        let target = self.target_mut()?;
        for local_y in 0..tile_height {
            for local_x in 0..tile_width {
                let global_x = tile_origin[0] as f64 + local_x as f64 + 0.5;
                let global_y = tile_origin[1] as f64 + local_y as f64 + 0.5;
                if global_x < clipped[0]
                    || global_x >= clipped[2]
                    || global_y < clipped[1]
                    || global_y >= clipped[3]
                {
                    continue;
                }
                let color = if image.kind == SurfaceImageKind::Rgba {
                    let bounds = bounds.edges();
                    let coordinates = [
                        (global_x - bounds[0]) / (bounds[2] - bounds[0]),
                        (global_y - bounds[1]) / (bounds[3] - bounds[1]),
                    ];
                    sample_surface(image, coordinates)?
                } else {
                    let checker = (((global_x / 8.0).floor() as i64
                        + (global_y / 8.0).floor() as i64
                        + (frame_revision & 1) as i64)
                        & 1)
                        == 0;
                    if checker {
                        [0.35, 0.0, 0.35, 1.0]
                    } else {
                        [0.1, 0.1, 0.1, 1.0]
                    }
                };
                let index = local_y
                    .checked_mul(tile_width)
                    .and_then(|row| row.checked_add(local_x))
                    .context("software surface destination index overflowed")?;
                blend_straight(target, index, [color[0], color[1], color[2]], color[3])?;
            }
        }
        Ok(())
    }

    fn fill_vello_rect(&mut self, rect: Rect, color: Rgba) -> Result<()> {
        if rect.width() <= 0.0 || rect.height() <= 0.0 || color.a <= 0.0 {
            return Ok(());
        }
        let target = self.target_mut()?;
        target.context.set_paint(vello_color(color));
        target.context.fill_rect(&rect);
        target.has_pending_commands = true;
        Ok(())
    }

    fn push_clip(&mut self, path: &BezPath) -> Result<()> {
        let target = self.target_mut()?;
        target.context.push_clip_path(path);
        Ok(())
    }

    fn pop_clip(&mut self) -> Result<()> {
        let target = self.target_mut()?;
        target.context.pop_clip_path();
        Ok(())
    }

    fn flush_segment(&mut self) -> Result<()> {
        let transform = tile_transform(self.tile_origin)?;
        let Some(target) = self.target.as_mut() else {
            return Ok(());
        };
        if !target.has_pending_commands {
            return Ok(());
        }
        target.context.flush();
        target
            .context
            .composite_to_pixmap_at_offset(&target.resources, &mut target.pixmap, 0, 0);
        target.context.reset();
        target.context.set_transform(transform);
        target.has_pending_commands = false;
        Ok(())
    }

    fn target_mut(&mut self) -> Result<&mut RasterTarget> {
        self.target
            .as_mut()
            .context("software rasterizer begin was not called")
    }
}

impl TransformData {
    fn identity() -> Self {
        Self {
            matrix: [
                [float_bits(1.0), float_bits(0.0)],
                [float_bits(0.0), float_bits(1.0)],
            ],
            translation: [float_bits(0.0), float_bits(0.0)],
        }
    }

    fn inverse(self) -> Option<InverseTransform> {
        let a = f64::from(self.matrix[0][0].get());
        let b = f64::from(self.matrix[0][1].get());
        let c = f64::from(self.matrix[1][0].get());
        let d = f64::from(self.matrix[1][1].get());
        let determinant = a * d - b * c;
        if determinant.abs() <= f64::EPSILON {
            return None;
        }
        Some(InverseTransform {
            matrix: [
                [d / determinant, -b / determinant],
                [-c / determinant, a / determinant],
            ],
            translation: [
                f64::from(self.translation[0].get()),
                f64::from(self.translation[1].get()),
            ],
        })
    }
}

struct InverseTransform {
    matrix: [[f64; 2]; 2],
    translation: [f64; 2],
}

impl InverseTransform {
    fn apply(&self, point: [f64; 2]) -> [f64; 2] {
        let point = [
            point[0] - self.translation[0],
            point[1] - self.translation[1],
        ];
        [
            self.matrix[0][0] * point[0] + self.matrix[0][1] * point[1],
            self.matrix[1][0] * point[0] + self.matrix[1][1] * point[1],
        ]
    }
}

fn float_bits(value: f32) -> crate::software_scene::FloatBits {
    crate::software_scene::FloatBits::from_bits(value.to_bits())
}

fn checker_index_parity(coordinate: f64, origin: f64, size: f64) -> usize {
    let index = ((coordinate - origin) / size).floor();
    if index.is_finite() {
        index.rem_euclid(2.0) as usize
    } else {
        0
    }
}

fn left_aligned_pattern_origin(clip_start: f64, bounds_start: f64, size: f64) -> f64 {
    clip_start - (clip_start - bounds_start).rem_euclid(size)
}

fn tile_transform(tile_origin: [usize; 2]) -> Result<Affine> {
    let x = u32::try_from(tile_origin[0]).context("software tile x exceeds u32")?;
    let y = u32::try_from(tile_origin[1]).context("software tile y exceeds u32")?;
    Ok(Affine::translate((-f64::from(x), -f64::from(y))))
}

fn rect(bounds: RectData) -> Rect {
    let edges = bounds.edges();
    Rect::new(edges[0], edges[1], edges[2], edges[3])
}

fn rounded_path(bounds: RectData, corners: CornersData) -> BezPath {
    rounded_path_from_radii(bounds, corners.get())
}

fn rounded_path_from_radii(bounds: RectData, radii: [f64; 4]) -> BezPath {
    let edges = bounds.edges();
    RoundedRect::new(
        edges[0],
        edges[1],
        edges[2],
        edges[3],
        (radii[0], radii[1], radii[2], radii[3]),
    )
    .to_path(0.1)
}

fn average_radius(corners: CornersData) -> f64 {
    corners.get().into_iter().sum::<f64>() / 4.0
}

fn canonical_path(elements: &[PathElementData]) -> BezPath {
    let mut path = BezPath::new();
    for element in elements {
        match element {
            PathElementData::MoveTo(point) => path.move_to(kurbo_point(*point)),
            PathElementData::LineTo(point) => path.line_to(kurbo_point(*point)),
            PathElementData::QuadraticTo { control, to } => {
                path.quad_to(kurbo_point(*control), kurbo_point(*to))
            }
            PathElementData::CubicTo {
                control_1,
                control_2,
                to,
            } => path.curve_to(
                kurbo_point(*control_1),
                kurbo_point(*control_2),
                kurbo_point(*to),
            ),
            PathElementData::Close => path.close_path(),
        }
    }
    path
}

fn triangle_path(vertices: &[crate::software_scene::PathVertexData]) -> BezPath {
    let mut path = BezPath::new();
    for triangle in vertices.chunks_exact(3) {
        path.move_to(kurbo_point(triangle[0].point));
        path.line_to(kurbo_point(triangle[1].point));
        path.line_to(kurbo_point(triangle[2].point));
        path.close_path();
    }
    path
}

fn path_bounds(path: &BezPath) -> Result<RectData> {
    let bounds = path.bounding_box();
    RectData::new(
        bounds.x0 as f32,
        bounds.y0 as f32,
        bounds.width() as f32,
        bounds.height() as f32,
        "software path paint bounds",
    )
}

fn kurbo_point(point: PointData) -> (f64, f64) {
    let point = point.get();
    (point[0], point[1])
}

fn vello_stroke(stroke: StrokeData) -> Stroke {
    let cap = match stroke.line_cap {
        PathLineCap::Butt => Cap::Butt,
        PathLineCap::Round => Cap::Round,
        PathLineCap::Square => Cap::Square,
    };
    let join = match stroke.line_join {
        PathLineJoin::Miter => Join::Miter,
        PathLineJoin::Round => Join::Round,
        PathLineJoin::Bevel => Join::Bevel,
    };
    Stroke::new(f64::from(stroke.width.get()).max(0.0))
        .with_caps(cap)
        .with_join(join)
        .with_miter_limit(f64::from(stroke.miter_limit.get()).max(0.0))
}

fn inside_rounded_rect(point: [f64; 2], bounds: RectData, corners: CornersData) -> bool {
    let edges = bounds.edges();
    let radii = corners.get();
    let radius = if point[0] < (edges[0] + edges[2]) * 0.5 {
        if point[1] < (edges[1] + edges[3]) * 0.5 {
            radii[0]
        } else {
            radii[3]
        }
    } else if point[1] < (edges[1] + edges[3]) * 0.5 {
        radii[1]
    } else {
        radii[2]
    }
    .max(0.0)
    .min((edges[2] - edges[0]).abs() * 0.5)
    .min((edges[3] - edges[1]).abs() * 0.5);
    if radius <= 0.0 {
        return true;
    }
    let on_left = point[0] < (edges[0] + edges[2]) * 0.5;
    let on_top = point[1] < (edges[1] + edges[3]) * 0.5;
    let center_x = if on_left {
        if point[0] >= edges[0] + radius {
            return true;
        }
        edges[0] + radius
    } else {
        if point[0] <= edges[2] - radius {
            return true;
        }
        edges[2] - radius
    };
    let center_y = if on_top {
        if point[1] >= edges[1] + radius {
            return true;
        }
        edges[1] + radius
    } else {
        if point[1] <= edges[3] - radius {
            return true;
        }
        edges[3] - radius
    };
    let distance_x = (point[0] - center_x).abs();
    let distance_y = (point[1] - center_y).abs();
    distance_x.hypot(distance_y) <= radius
}

fn sample_monochrome(image: &AtlasImageData, coordinates: [f64; 2]) -> Result<f32> {
    sample_image(image, coordinates, |pixel| {
        pixel.first().copied().map(|value| f32::from(value) / 255.0)
    })
}

fn sample_bgra(image: &AtlasImageData, coordinates: [f64; 2]) -> Result<[f32; 4]> {
    sample_image(image, coordinates, |pixel| {
        let blue = *pixel.first()?;
        let green = *pixel.get(1)?;
        let red = *pixel.get(2)?;
        let alpha = *pixel.get(3)?;
        Some([
            f32::from(red) / 255.0,
            f32::from(green) / 255.0,
            f32::from(blue) / 255.0,
            f32::from(alpha) / 255.0,
        ])
    })
}

fn sample_surface(image: &SurfaceImageData, coordinates: [f64; 2]) -> Result<[f32; 4]> {
    ensure!(
        image.width > 0 && image.height > 0,
        "software surface image is empty"
    );
    let x = ((coordinates[0].clamp(0.0, 1.0 - f64::EPSILON) * f64::from(image.width)).floor()
        as u32)
        .min(image.width - 1);
    let y = ((coordinates[1].clamp(0.0, 1.0 - f64::EPSILON) * f64::from(image.height)).floor()
        as u32)
        .min(image.height - 1);
    let offset = usize::try_from(y)
        .ok()
        .and_then(|row| row.checked_mul(usize::try_from(image.width).ok()?))
        .and_then(|row| row.checked_add(usize::try_from(x).ok()?))
        .and_then(|pixel| pixel.checked_mul(4))
        .context("software surface sample offset overflowed")?;
    let pixel = image
        .pixels
        .get(offset..offset.saturating_add(4))
        .context("software surface sample is out of bounds")?;
    Ok([
        f32::from(pixel[0]) / 255.0,
        f32::from(pixel[1]) / 255.0,
        f32::from(pixel[2]) / 255.0,
        f32::from(pixel[3]) / 255.0,
    ])
}

fn sample_image<T: Copy + Default>(
    image: &AtlasImageData,
    coordinates: [f64; 2],
    decode: impl Fn(&[u8]) -> Option<T>,
) -> Result<T> {
    ensure!(
        image.width > 0 && image.height > 0,
        "software atlas image is empty"
    );
    let x = ((coordinates[0].clamp(0.0, 1.0 - f64::EPSILON) * f64::from(image.width)).floor()
        as u32)
        .min(image.width - 1);
    let y = ((coordinates[1].clamp(0.0, 1.0 - f64::EPSILON) * f64::from(image.height)).floor()
        as u32)
        .min(image.height - 1);
    let bytes_per_pixel = usize::from(image.bytes_per_pixel);
    let index = usize::try_from(y)
        .ok()
        .and_then(|row| row.checked_mul(usize::try_from(image.width).ok()?))
        .and_then(|row| row.checked_add(usize::try_from(x).ok()?))
        .and_then(|pixel| pixel.checked_mul(bytes_per_pixel))
        .context("software atlas sample offset overflowed")?;
    let end = index
        .checked_add(bytes_per_pixel)
        .context("software atlas sample range overflowed")?;
    let pixel = image
        .pixels
        .get(index..end)
        .context("software atlas sample is out of bounds")?;
    decode(pixel).context("software atlas pixel has an invalid format")
}

fn blend_straight(
    target: &mut RasterTarget,
    index: usize,
    color: [f32; 3],
    alpha: f32,
) -> Result<()> {
    let alpha = alpha.clamp(0.0, 1.0);
    let source = [
        color[0].clamp(0.0, 1.0) * alpha,
        color[1].clamp(0.0, 1.0) * alpha,
        color[2].clamp(0.0, 1.0) * alpha,
    ];
    blend_premultiplied(target, index, source, alpha)
}

fn blend_premultiplied(
    target: &mut RasterTarget,
    index: usize,
    source: [f32; 3],
    alpha: f32,
) -> Result<()> {
    let destination = target
        .pixmap
        .data_mut()
        .get_mut(index)
        .context("software raster destination is out of bounds")?;
    let inverse_alpha = 1.0 - alpha;
    destination.r = to_u8(source[0] + f32::from(destination.r) / 255.0 * inverse_alpha);
    destination.g = to_u8(source[1] + f32::from(destination.g) / 255.0 * inverse_alpha);
    destination.b = to_u8(source[2] + f32::from(destination.b) / 255.0 * inverse_alpha);
    destination.a = to_u8(alpha + f32::from(destination.a) / 255.0 * inverse_alpha);
    Ok(())
}

fn blend_subpixel(
    target: &mut RasterTarget,
    index: usize,
    color: [f32; 3],
    alpha: [f32; 3],
) -> Result<()> {
    let destination = target
        .pixmap
        .data_mut()
        .get_mut(index)
        .context("software subpixel destination is out of bounds")?;
    let channels = [&mut destination.r, &mut destination.g, &mut destination.b];
    for ((destination, foreground), alpha) in channels
        .into_iter()
        .zip(color)
        .zip(alpha.map(|alpha| alpha.clamp(0.0, 1.0)))
    {
        let value =
            foreground.clamp(0.0, 1.0) * alpha + f32::from(*destination) / 255.0 * (1.0 - alpha);
        *destination = to_u8(value);
    }
    destination.a = 255;
    Ok(())
}

fn apply_contrast_and_gamma(
    sample: f32,
    color: [f32; 3],
    enhanced_contrast: f32,
    gamma: [f32; 4],
) -> f32 {
    let brightness = color_brightness(color);
    let contrast = light_on_dark_contrast(enhanced_contrast, brightness);
    let contrasted = enhance_contrast(sample, contrast);
    apply_alpha_correction(contrasted, brightness, gamma).clamp(0.0, 1.0)
}

fn apply_subpixel_contrast_and_gamma(
    sample: [f32; 3],
    color: [f32; 3],
    enhanced_contrast: f32,
    gamma: [f32; 4],
) -> [f32; 3] {
    let brightness = color_brightness(color);
    let contrast = light_on_dark_contrast(enhanced_contrast, brightness);
    std::array::from_fn(|index| {
        let contrasted = enhance_contrast(sample[index], contrast);
        apply_alpha_correction(contrasted, color[index], gamma).clamp(0.0, 1.0)
    })
}

fn color_brightness(color: [f32; 3]) -> f32 {
    color[0] * 0.30 + color[1] * 0.59 + color[2] * 0.11
}

fn light_on_dark_contrast(enhanced_contrast: f32, brightness: f32) -> f32 {
    enhanced_contrast * (4.0 * (0.75 - brightness)).clamp(0.0, 1.0)
}

fn enhance_contrast(alpha: f32, contrast: f32) -> f32 {
    alpha * (contrast + 1.0) / (alpha * contrast + 1.0)
}

fn apply_alpha_correction(alpha: f32, brightness: f32, gamma: [f32; 4]) -> f32 {
    let brightness_adjustment = gamma[0] * brightness + gamma[1];
    let correction = brightness_adjustment * alpha + (gamma[2] * brightness + gamma[3]);
    alpha + alpha * (1.0 - alpha) * correction
}

fn premultiplied_color(color: Rgba) -> PremulRgba8 {
    vello_color(color).premultiply().to_rgba8()
}

fn vello_color(color: Rgba) -> AlphaColor<Srgb> {
    AlphaColor::new([
        color.r.clamp(0.0, 1.0),
        color.g.clamp(0.0, 1.0),
        color.b.clamp(0.0, 1.0),
        color.a.clamp(0.0, 1.0),
    ])
}

fn to_u8(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn validate_color(color: Rgba) -> Result<()> {
    ensure!(
        [color.r, color.g, color.b, color.a]
            .into_iter()
            .all(f32::is_finite),
        "software raster color contains a non-finite component"
    );
    Ok(())
}

fn validate_text_rendering(params: SoftwareTextRenderingParams) -> Result<()> {
    ensure!(
        params
            .gamma_ratios
            .into_iter()
            .chain([
                params.grayscale_enhanced_contrast,
                params.subpixel_enhanced_contrast,
            ])
            .all(f32::is_finite),
        "software text rendering parameters contain a non-finite value"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn color(red: f32, green: f32, blue: f32) -> Rgba {
        Rgba {
            r: red,
            g: green,
            b: blue,
            a: 1.0,
        }
    }

    #[test]
    fn converts_rgba_to_xrgb_and_recreates_equal_area_targets() -> Result<()> {
        let mut rasterizer = SoftwareRasterizer::new();
        rasterizer.begin([2, 1], color(0.0, 0.0, 0.0))?;
        rasterizer.fill_rectangle([0.0, 0.0, 2.0, 1.0], color(1.0, 0.0, 0.0))?;
        assert_eq!(rasterizer.finish()?, &[0x00ff_0000, 0x00ff_0000]);

        rasterizer.begin([1, 2], color(0.0, 0.0, 0.0))?;
        rasterizer.fill_rectangle([0.0, 0.0, 1.0, 2.0], color(0.0, 0.0, 1.0))?;
        assert_eq!(rasterizer.finish()?, &[0x0000_00ff, 0x0000_00ff]);
        assert!(rasterizer.begin([1, 1], color(f32::NAN, 0.0, 0.0)).is_err());
        Ok(())
    }

    #[test]
    fn gamma_and_subpixel_blending_remain_bounded() -> Result<()> {
        let corrected =
            apply_contrast_and_gamma(0.5, [0.2, 0.4, 0.8], 1.0, get_gamma_correction_ratios(1.8));
        assert!((0.0..=1.0).contains(&corrected));

        let mut target = RasterTarget {
            context: RenderContext::new(1, 1),
            resources: Resources::new(),
            pixmap: Pixmap::new(1, 1),
            has_pending_commands: false,
        };
        target.pixmap.data_mut()[0] = PremulRgba8 {
            r: 255,
            g: 255,
            b: 255,
            a: 255,
        };
        blend_subpixel(&mut target, 0, [1.0, 0.0, 0.0], [1.0, 0.0, 0.5])?;
        assert_eq!(target.pixmap.data()[0].r, 255);
        assert_eq!(target.pixmap.data()[0].g, 255);
        assert!(target.pixmap.data()[0].b < 255);
        Ok(())
    }

    #[test]
    fn checkerboard_clips_extreme_bounds_and_tiny_cells() -> Result<()> {
        let mut rasterizer = SoftwareRasterizer::new();
        rasterizer.begin_tile(
            [4, 4],
            [100, 100],
            color(0.0, 0.0, 0.0),
            SoftwareTextRenderingParams::default(),
        )?;
        rasterizer.fill_checkerboard(
            RectData::new(
                -1.0e20,
                -1.0e20,
                2.0e20,
                2.0e20,
                "extreme checkerboard bounds",
            )?,
            ColorData::from_rgba(color(1.0, 1.0, 1.0))?,
            f32::MIN_POSITIVE,
        )?;
        assert_eq!(rasterizer.finish()?.len(), 16);
        Ok(())
    }

    #[test]
    fn subpixel_checkerboard_keeps_phase_across_tiles() -> Result<()> {
        let bounds = RectData::new(0.0, 0.0, 8.0, 1.0, "subpixel checkerboard bounds")?;
        let foreground = ColorData::from_rgba(color(1.0, 1.0, 1.0))?;

        let mut full = SoftwareRasterizer::new();
        full.begin([8, 1], color(0.0, 0.0, 0.0))?;
        full.fill_checkerboard(bounds, foreground, 0.75)?;
        let full = full.finish()?.to_vec();

        let mut left = SoftwareRasterizer::new();
        left.begin_tile(
            [4, 1],
            [0, 0],
            color(0.0, 0.0, 0.0),
            SoftwareTextRenderingParams::default(),
        )?;
        left.fill_checkerboard(bounds, foreground, 0.75)?;
        let mut tiled = left.finish()?.to_vec();

        let mut right = SoftwareRasterizer::new();
        right.begin_tile(
            [4, 1],
            [4, 0],
            color(0.0, 0.0, 0.0),
            SoftwareTextRenderingParams::default(),
        )?;
        right.fill_checkerboard(bounds, foreground, 0.75)?;
        tiled.extend_from_slice(right.finish()?);

        assert_eq!(tiled, full);
        Ok(())
    }

    #[test]
    fn surface_pixels_scale_clip_and_use_an_explicit_placeholder() -> Result<()> {
        let mut rasterizer = SoftwareRasterizer::new();
        rasterizer.begin([4, 2], color(1.0, 1.0, 1.0))?;
        let image = SurfaceImageData::rgba(2, 1, vec![255, 0, 0, 255, 0, 0, 255, 255])?;
        rasterizer.render_surface(
            RectData::new(1.0, 0.0, 2.0, 2.0, "test surface clip")?,
            RectData::new(0.0, 0.0, 4.0, 2.0, "test surface bounds")?,
            1,
            &image,
        )?;
        assert_eq!(
            rasterizer.finish()?,
            &[
                0x00ff_ffff,
                0x00ff_0000,
                0x0000_00ff,
                0x00ff_ffff,
                0x00ff_ffff,
                0x00ff_0000,
                0x0000_00ff,
                0x00ff_ffff,
            ]
        );

        rasterizer.begin([1, 1], color(1.0, 1.0, 1.0))?;
        let unsupported = SurfaceImageData {
            width: 0,
            height: 0,
            kind: SurfaceImageKind::Unsupported(0x1234),
            pixels: Arc::from([]),
        };
        rasterizer.render_surface(
            RectData::new(0.0, 0.0, 1.0, 1.0, "test placeholder clip")?,
            RectData::new(0.0, 0.0, 1.0, 1.0, "test placeholder bounds")?,
            1,
            &unsupported,
        )?;
        assert_ne!(rasterizer.finish()?, &[0x00ff_ffff]);
        Ok(())
    }
}
