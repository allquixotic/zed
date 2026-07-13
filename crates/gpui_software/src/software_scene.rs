use crate::software_surface::SurfaceImageData;
use crate::{SOFTWARE_TILE_SIZE, SoftwareAtlas, SoftwareAtlasTile};
use anyhow::{Context as _, Result, ensure};
use gpui::{
    AtlasTextureKind, Background, BackgroundKind, BorderStyle, Bounds, ColorSpace, Corners,
    DevicePixels, Edges, Hsla, LinearColorStop, PaintSurface, Path, PathCommand, PathFillRule,
    PathLineCap, PathLineJoin, PathPaint, PathStroke, PathVertex, PolychromeSprite, PrimitiveBatch,
    Quad, Rgba, ScaledPixels, Scene, Shadow, Size, SubpixelSprite, TransformationMatrix, Underline,
};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FloatBits(u32);

impl FloatBits {
    pub(crate) fn new(value: f32, description: &'static str) -> Result<Self> {
        ensure!(value.is_finite(), "{description} is not finite");
        Ok(Self(value.to_bits()))
    }

    pub(crate) fn get(self) -> f32 {
        f32::from_bits(self.0)
    }

    pub(crate) const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PointData {
    pub(crate) x: FloatBits,
    pub(crate) y: FloatBits,
}

impl PointData {
    fn new(x: f32, y: f32, description: &'static str) -> Result<Self> {
        Ok(Self {
            x: FloatBits::new(x, description)?,
            y: FloatBits::new(y, description)?,
        })
    }

    pub(crate) fn get(self) -> [f64; 2] {
        [f64::from(self.x.get()), f64::from(self.y.get())]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RectData {
    pub(crate) origin: PointData,
    pub(crate) size: PointData,
}

impl RectData {
    fn from_bounds(bounds: Bounds<ScaledPixels>, description: &'static str) -> Result<Self> {
        Self::new(
            bounds.origin.x.0,
            bounds.origin.y.0,
            bounds.size.width.0,
            bounds.size.height.0,
            description,
        )
    }

    pub(crate) fn new(
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        description: &'static str,
    ) -> Result<Self> {
        Ok(Self {
            origin: PointData::new(x, y, description)?,
            size: PointData::new(width, height, description)?,
        })
    }

    pub(crate) fn edges(self) -> [f64; 4] {
        let [left, top] = self.origin.get();
        let [width, height] = self.size.get();
        [left, top, left + width, top + height]
    }

    pub(crate) fn intersect(self, other: Self) -> Option<[f64; 4]> {
        let first = self.edges();
        let second = other.edges();
        let edges = [
            first[0].max(second[0]),
            first[1].max(second[1]),
            first[2].min(second[2]),
            first[3].min(second[3]),
        ];
        (edges[2] > edges[0] && edges[3] > edges[1]).then_some(edges)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ColorData {
    pub(crate) red: FloatBits,
    pub(crate) green: FloatBits,
    pub(crate) blue: FloatBits,
    pub(crate) alpha: FloatBits,
}

impl ColorData {
    fn from_hsla(color: Hsla) -> Result<Self> {
        Self::from_rgba(color.into())
    }

    pub(crate) fn from_rgba(color: Rgba) -> Result<Self> {
        Ok(Self {
            red: FloatBits::new(color.r, "software scene color")?,
            green: FloatBits::new(color.g, "software scene color")?,
            blue: FloatBits::new(color.b, "software scene color")?,
            alpha: FloatBits::new(color.a, "software scene color")?,
        })
    }

    pub(crate) fn get(self) -> Rgba {
        Rgba {
            r: self.red.get(),
            g: self.green.get(),
            b: self.blue.get(),
            a: self.alpha.get(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GradientStopData {
    pub(crate) color: ColorData,
    pub(crate) percentage: FloatBits,
}

impl GradientStopData {
    fn new(stop: LinearColorStop) -> Result<Self> {
        Ok(Self {
            color: ColorData::from_hsla(stop.color)?,
            percentage: FloatBits::new(stop.percentage, "software gradient percentage")?,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BackgroundData {
    Solid(ColorData),
    LinearGradient {
        angle: FloatBits,
        color_space: ColorSpace,
        stops: [GradientStopData; 2],
    },
    PatternSlash {
        color: ColorData,
        width: FloatBits,
        interval: FloatBits,
    },
    Checkerboard {
        color: ColorData,
        size: FloatBits,
    },
}

impl BackgroundData {
    fn new(background: Background) -> Result<Self> {
        Ok(match background.kind() {
            BackgroundKind::Solid(color) => Self::Solid(ColorData::from_hsla(color)?),
            BackgroundKind::LinearGradient {
                angle,
                color_space,
                stops,
            } => Self::LinearGradient {
                angle: FloatBits::new(angle, "software gradient angle")?,
                color_space,
                stops: [
                    GradientStopData::new(stops[0])?,
                    GradientStopData::new(stops[1])?,
                ],
            },
            BackgroundKind::PatternSlash {
                color,
                width,
                interval,
            } => Self::PatternSlash {
                color: ColorData::from_hsla(color)?,
                width: FloatBits::new(width, "software slash width")?,
                interval: FloatBits::new(interval, "software slash interval")?,
            },
            BackgroundKind::Checkerboard { color, size } => Self::Checkerboard {
                color: ColorData::from_hsla(color)?,
                size: FloatBits::new(size, "software checker size")?,
            },
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CornersData {
    pub(crate) top_left: FloatBits,
    pub(crate) top_right: FloatBits,
    pub(crate) bottom_right: FloatBits,
    pub(crate) bottom_left: FloatBits,
}

impl CornersData {
    fn new(corners: Corners<ScaledPixels>) -> Result<Self> {
        Ok(Self {
            top_left: FloatBits::new(corners.top_left.0, "software corner radius")?,
            top_right: FloatBits::new(corners.top_right.0, "software corner radius")?,
            bottom_right: FloatBits::new(corners.bottom_right.0, "software corner radius")?,
            bottom_left: FloatBits::new(corners.bottom_left.0, "software corner radius")?,
        })
    }

    pub(crate) fn get(self) -> [f64; 4] {
        [
            f64::from(self.top_left.get()),
            f64::from(self.top_right.get()),
            f64::from(self.bottom_right.get()),
            f64::from(self.bottom_left.get()),
        ]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct EdgesData {
    pub(crate) top: FloatBits,
    pub(crate) right: FloatBits,
    pub(crate) bottom: FloatBits,
    pub(crate) left: FloatBits,
}

impl EdgesData {
    fn new(edges: Edges<ScaledPixels>) -> Result<Self> {
        Ok(Self {
            top: FloatBits::new(edges.top.0, "software edge width")?,
            right: FloatBits::new(edges.right.0, "software edge width")?,
            bottom: FloatBits::new(edges.bottom.0, "software edge width")?,
            left: FloatBits::new(edges.left.0, "software edge width")?,
        })
    }

    pub(crate) fn get(self) -> [f64; 4] {
        [
            f64::from(self.top.get()),
            f64::from(self.right.get()),
            f64::from(self.bottom.get()),
            f64::from(self.left.get()),
        ]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TransformData {
    pub(crate) matrix: [[FloatBits; 2]; 2],
    pub(crate) translation: [FloatBits; 2],
}

impl TransformData {
    fn new(transform: TransformationMatrix) -> Result<Self> {
        Ok(Self {
            matrix: [
                [
                    FloatBits::new(transform.rotation_scale[0][0], "software transform")?,
                    FloatBits::new(transform.rotation_scale[0][1], "software transform")?,
                ],
                [
                    FloatBits::new(transform.rotation_scale[1][0], "software transform")?,
                    FloatBits::new(transform.rotation_scale[1][1], "software transform")?,
                ],
            ],
            translation: [
                FloatBits::new(transform.translation[0], "software transform")?,
                FloatBits::new(transform.translation[1], "software transform")?,
            ],
        })
    }

    pub(crate) fn apply(self, point: [f64; 2]) -> [f64; 2] {
        [
            f64::from(self.matrix[0][0].get()) * point[0]
                + f64::from(self.matrix[0][1].get()) * point[1]
                + f64::from(self.translation[0].get()),
            f64::from(self.matrix[1][0].get()) * point[0]
                + f64::from(self.matrix[1][1].get()) * point[1]
                + f64::from(self.translation[1].get()),
        ]
    }
}

#[derive(Clone, Debug)]
pub(crate) struct AtlasImageData {
    pub(crate) texture_index: u32,
    pub(crate) texture_kind: AtlasTextureKind,
    pub(crate) tile_id: u32,
    pub(crate) padding: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) revision: u64,
    pub(crate) bytes_per_pixel: u8,
    pub(crate) pixels: Arc<[u8]>,
}

impl PartialEq for AtlasImageData {
    fn eq(&self, other: &Self) -> bool {
        self.texture_index == other.texture_index
            && self.texture_kind == other.texture_kind
            && self.tile_id == other.tile_id
            && self.padding == other.padding
            && self.width == other.width
            && self.height == other.height
            && self.revision == other.revision
            && self.bytes_per_pixel == other.bytes_per_pixel
    }
}

impl Eq for AtlasImageData {}

impl AtlasImageData {
    fn new(tile: SoftwareAtlasTile) -> Result<Self> {
        let width = u32::try_from(tile.tile.bounds.size.width.0)
            .context("software atlas image width is negative")?;
        let height = u32::try_from(tile.tile.bounds.size.height.0)
            .context("software atlas image height is negative")?;
        Ok(Self {
            texture_index: tile.tile.texture_id.index,
            texture_kind: tile.tile.texture_id.kind,
            tile_id: tile.tile.tile_id.0,
            padding: tile.tile.padding,
            width,
            height,
            revision: tile.revision,
            bytes_per_pixel: tile.bytes_per_pixel,
            pixels: tile.pixels,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PathElementData {
    MoveTo(PointData),
    LineTo(PointData),
    QuadraticTo {
        control: PointData,
        to: PointData,
    },
    CubicTo {
        control_1: PointData,
        control_2: PointData,
        to: PointData,
    },
    Close,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PathVertexData {
    pub(crate) point: PointData,
    pub(crate) curve_coordinate: PointData,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct StrokeData {
    pub(crate) width: FloatBits,
    pub(crate) miter_limit: FloatBits,
    pub(crate) line_cap: PathLineCap,
    pub(crate) line_join: PathLineJoin,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PathPaintData {
    Fill(PathFillRule),
    Stroke(StrokeData),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum PathGeometryData {
    Canonical(Arc<[PathElementData]>),
    Triangles(Arc<[PathVertexData]>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum CommandKind {
    Quad {
        border_style: BorderStyle,
        bounds: RectData,
        background: BackgroundData,
        border_color: ColorData,
        corner_radii: CornersData,
        border_widths: EdgesData,
    },
    Shadow {
        blur_radius: FloatBits,
        bounds: RectData,
        corner_radii: CornersData,
        color: ColorData,
        element_bounds: RectData,
        element_corner_radii: CornersData,
        inset: bool,
    },
    Path {
        bounds: RectData,
        color: BackgroundData,
        paint: PathPaintData,
        geometry: PathGeometryData,
    },
    Underline {
        bounds: RectData,
        color: ColorData,
        thickness: FloatBits,
        wavy: bool,
    },
    MonochromeSprite {
        bounds: RectData,
        color: ColorData,
        image: AtlasImageData,
        transform: TransformData,
    },
    SubpixelSprite {
        bounds: RectData,
        color: ColorData,
        image: AtlasImageData,
        transform: TransformData,
    },
    PolychromeSprite {
        bounds: RectData,
        grayscale: bool,
        opacity: FloatBits,
        corner_radii: CornersData,
        image: AtlasImageData,
    },
    Surface {
        bounds: RectData,
        frame_revision: u64,
        image: SurfaceImageData,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SoftwareCommand {
    pub(crate) order: u32,
    pub(crate) clip: RectData,
    pub(crate) kind: CommandKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SoftwareScene {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) columns: usize,
    pub(crate) rows: usize,
    pub(crate) scale_factor: FloatBits,
    pub(crate) background: ColorData,
    pub(crate) tiles: Vec<Vec<SoftwareCommand>>,
}

impl SoftwareScene {
    pub(crate) fn new(
        scene: &Scene,
        size: Size<DevicePixels>,
        scale_factor: f32,
        background: Rgba,
        atlas: &SoftwareAtlas,
    ) -> Result<Self> {
        let width = usize::try_from(size.width.0).context("software viewport width is negative")?;
        let height =
            usize::try_from(size.height.0).context("software viewport height is negative")?;
        let columns = width.div_ceil(SOFTWARE_TILE_SIZE);
        let rows = height.div_ceil(SOFTWARE_TILE_SIZE);
        let tile_count = columns
            .checked_mul(rows)
            .context("software tile grid dimensions overflowed")?;
        let mut tiles = Vec::new();
        tiles
            .try_reserve_exact(tile_count)
            .context("allocating software tile command lists")?;
        tiles.resize_with(tile_count, Vec::new);
        let mut software_scene = Self {
            width,
            height,
            columns,
            rows,
            scale_factor: FloatBits::new(scale_factor, "software display scale")?,
            background: ColorData::from_rgba(Rgba {
                a: 1.0,
                ..background
            })?,
            tiles,
        };

        for batch in scene.batches() {
            match batch {
                PrimitiveBatch::Quads(range) => {
                    for quad in scene
                        .quads
                        .get(range)
                        .context("scene quad batch is out of bounds")?
                    {
                        software_scene.push_command(quad_command(quad)?)?;
                    }
                }
                PrimitiveBatch::Shadows(range) => {
                    for shadow in scene
                        .shadows
                        .get(range)
                        .context("scene shadow batch is out of bounds")?
                    {
                        software_scene.push_command(shadow_command(shadow)?)?;
                    }
                }
                PrimitiveBatch::Paths(range) => {
                    for path in scene
                        .paths
                        .get(range)
                        .context("scene path batch is out of bounds")?
                    {
                        software_scene.push_command(path_command(path)?)?;
                    }
                }
                PrimitiveBatch::Underlines(range) => {
                    for underline in scene
                        .underlines
                        .get(range)
                        .context("scene underline batch is out of bounds")?
                    {
                        software_scene.push_command(underline_command(underline)?)?;
                    }
                }
                PrimitiveBatch::MonochromeSprites { range, .. } => {
                    for sprite in scene
                        .monochrome_sprites
                        .get(range)
                        .context("scene monochrome sprite batch is out of bounds")?
                    {
                        software_scene.push_command(monochrome_command(sprite, atlas)?)?;
                    }
                }
                PrimitiveBatch::SubpixelSprites { range, .. } => {
                    for sprite in scene
                        .subpixel_sprites
                        .get(range)
                        .context("scene subpixel sprite batch is out of bounds")?
                    {
                        software_scene.push_command(subpixel_command(sprite, atlas)?)?;
                    }
                }
                PrimitiveBatch::PolychromeSprites { range, .. } => {
                    for sprite in scene
                        .polychrome_sprites
                        .get(range)
                        .context("scene polychrome sprite batch is out of bounds")?
                    {
                        software_scene.push_command(polychrome_command(sprite, atlas)?)?;
                    }
                }
                PrimitiveBatch::Surfaces(range) => {
                    for surface in scene
                        .surfaces
                        .get(range)
                        .context("scene surface batch is out of bounds")?
                    {
                        software_scene.push_command(surface_command(surface)?)?;
                    }
                }
            }
        }
        Ok(software_scene)
    }

    pub(crate) fn tile_origin(&self, index: usize) -> Result<[usize; 2]> {
        ensure!(self.columns > 0, "zero-width software scene has no tiles");
        let row = index / self.columns;
        let column = index % self.columns;
        Ok([
            column
                .checked_mul(SOFTWARE_TILE_SIZE)
                .context("software tile x coordinate overflowed")?,
            row.checked_mul(SOFTWARE_TILE_SIZE)
                .context("software tile y coordinate overflowed")?,
        ])
    }

    pub(crate) fn tile_size(&self, index: usize) -> Result<[usize; 2]> {
        let origin = self.tile_origin(index)?;
        Ok([
            self.width.saturating_sub(origin[0]).min(SOFTWARE_TILE_SIZE),
            self.height
                .saturating_sub(origin[1])
                .min(SOFTWARE_TILE_SIZE),
        ])
    }

    fn push_command(&mut self, command: SoftwareCommand) -> Result<()> {
        let Some(bounds) = command_bounds(&command) else {
            return Ok(());
        };
        let viewport = [0.0, 0.0, self.width as f64, self.height as f64];
        let bounds = [
            bounds[0].max(viewport[0]),
            bounds[1].max(viewport[1]),
            bounds[2].min(viewport[2]),
            bounds[3].min(viewport[3]),
        ];
        if bounds[2] <= bounds[0] || bounds[3] <= bounds[1] {
            return Ok(());
        }

        let first_column = (bounds[0].floor().max(0.0) as usize) / SOFTWARE_TILE_SIZE;
        let first_row = (bounds[1].floor().max(0.0) as usize) / SOFTWARE_TILE_SIZE;
        let last_column = ((bounds[2].ceil().max(1.0) as usize).saturating_sub(1)
            / SOFTWARE_TILE_SIZE)
            .min(self.columns.saturating_sub(1));
        let last_row = ((bounds[3].ceil().max(1.0) as usize).saturating_sub(1)
            / SOFTWARE_TILE_SIZE)
            .min(self.rows.saturating_sub(1));
        for row in first_row..=last_row {
            for column in first_column..=last_column {
                let index = row
                    .checked_mul(self.columns)
                    .and_then(|row_start| row_start.checked_add(column))
                    .context("software tile index overflowed")?;
                self.tiles
                    .get_mut(index)
                    .context("software tile index is out of bounds")?
                    .push(command.clone());
            }
        }
        Ok(())
    }
}

fn quad_command(quad: &Quad) -> Result<SoftwareCommand> {
    Ok(SoftwareCommand {
        order: quad.order,
        clip: RectData::from_bounds(quad.content_mask.bounds, "software quad clip")?,
        kind: CommandKind::Quad {
            border_style: quad.border_style,
            bounds: RectData::from_bounds(quad.bounds, "software quad bounds")?,
            background: BackgroundData::new(quad.background)?,
            border_color: ColorData::from_hsla(quad.border_color)?,
            corner_radii: CornersData::new(quad.corner_radii)?,
            border_widths: EdgesData::new(quad.border_widths)?,
        },
    })
}

fn shadow_command(shadow: &Shadow) -> Result<SoftwareCommand> {
    Ok(SoftwareCommand {
        order: shadow.order,
        clip: RectData::from_bounds(shadow.content_mask.bounds, "software shadow clip")?,
        kind: CommandKind::Shadow {
            blur_radius: FloatBits::new(shadow.blur_radius.0, "software shadow blur")?,
            bounds: RectData::from_bounds(shadow.bounds, "software shadow bounds")?,
            corner_radii: CornersData::new(shadow.corner_radii)?,
            color: ColorData::from_hsla(shadow.color)?,
            element_bounds: RectData::from_bounds(
                shadow.element_bounds,
                "software shadow element bounds",
            )?,
            element_corner_radii: CornersData::new(shadow.element_corner_radii)?,
            inset: shadow.is_inset(),
        },
    })
}

fn path_command(path: &Path<ScaledPixels>) -> Result<SoftwareCommand> {
    let geometry = if path.uses_triangle_fallback() {
        ensure!(
            path.vertices.len().is_multiple_of(3),
            "software path triangle list is incomplete"
        );
        let vertices = path
            .vertices
            .iter()
            .map(path_vertex_data)
            .collect::<Result<Vec<_>>>()?;
        PathGeometryData::Triangles(Arc::from(vertices))
    } else {
        let commands = path
            .commands()
            .iter()
            .map(path_element_data)
            .collect::<Result<Vec<_>>>()?;
        PathGeometryData::Canonical(Arc::from(commands))
    };
    let paint = match path.paint() {
        PathPaint::Fill(fill_rule) => PathPaintData::Fill(fill_rule),
        PathPaint::Stroke(stroke) => PathPaintData::Stroke(stroke_data(stroke)?),
    };
    Ok(SoftwareCommand {
        order: path.order,
        clip: RectData::from_bounds(path.content_mask.bounds, "software path clip")?,
        kind: CommandKind::Path {
            bounds: RectData::from_bounds(path.bounds, "software path bounds")?,
            color: BackgroundData::new(path.color)?,
            paint,
            geometry,
        },
    })
}

fn path_element_data(command: &PathCommand<ScaledPixels>) -> Result<PathElementData> {
    let point = |point: gpui::Point<ScaledPixels>| {
        PointData::new(point.x.0, point.y.0, "software path point")
    };
    Ok(match command {
        PathCommand::MoveTo(to) => PathElementData::MoveTo(point(*to)?),
        PathCommand::LineTo(to) => PathElementData::LineTo(point(*to)?),
        PathCommand::QuadraticTo { control, to } => PathElementData::QuadraticTo {
            control: point(*control)?,
            to: point(*to)?,
        },
        PathCommand::CubicTo {
            control_1,
            control_2,
            to,
        } => PathElementData::CubicTo {
            control_1: point(*control_1)?,
            control_2: point(*control_2)?,
            to: point(*to)?,
        },
        PathCommand::Close => PathElementData::Close,
    })
}

fn path_vertex_data(vertex: &PathVertex<ScaledPixels>) -> Result<PathVertexData> {
    Ok(PathVertexData {
        point: PointData::new(
            vertex.xy_position.x.0,
            vertex.xy_position.y.0,
            "software path vertex",
        )?,
        curve_coordinate: PointData::new(
            vertex.st_position.x,
            vertex.st_position.y,
            "software path curve coordinate",
        )?,
    })
}

fn stroke_data(stroke: PathStroke) -> Result<StrokeData> {
    Ok(StrokeData {
        width: FloatBits::new(stroke.width, "software path stroke width")?,
        miter_limit: FloatBits::new(stroke.miter_limit, "software path miter limit")?,
        line_cap: stroke.line_cap,
        line_join: stroke.line_join,
    })
}

fn underline_command(underline: &Underline) -> Result<SoftwareCommand> {
    Ok(SoftwareCommand {
        order: underline.order,
        clip: RectData::from_bounds(underline.content_mask.bounds, "software underline clip")?,
        kind: CommandKind::Underline {
            bounds: RectData::from_bounds(underline.bounds, "software underline bounds")?,
            color: ColorData::from_hsla(underline.color)?,
            thickness: FloatBits::new(underline.thickness.0, "software underline thickness")?,
            wavy: underline.is_wavy(),
        },
    })
}

fn monochrome_command(
    sprite: &gpui::MonochromeSprite,
    atlas: &SoftwareAtlas,
) -> Result<SoftwareCommand> {
    Ok(SoftwareCommand {
        order: sprite.order,
        clip: RectData::from_bounds(sprite.content_mask.bounds, "software sprite clip")?,
        kind: CommandKind::MonochromeSprite {
            bounds: RectData::from_bounds(sprite.bounds, "software sprite bounds")?,
            color: ColorData::from_hsla(sprite.color)?,
            image: AtlasImageData::new(atlas.tile(sprite.tile)?)?,
            transform: TransformData::new(sprite.transformation)?,
        },
    })
}

fn subpixel_command(sprite: &SubpixelSprite, atlas: &SoftwareAtlas) -> Result<SoftwareCommand> {
    Ok(SoftwareCommand {
        order: sprite.order,
        clip: RectData::from_bounds(sprite.content_mask.bounds, "software sprite clip")?,
        kind: CommandKind::SubpixelSprite {
            bounds: RectData::from_bounds(sprite.bounds, "software sprite bounds")?,
            color: ColorData::from_hsla(sprite.color)?,
            image: AtlasImageData::new(atlas.tile(sprite.tile)?)?,
            transform: TransformData::new(sprite.transformation)?,
        },
    })
}

fn polychrome_command(sprite: &PolychromeSprite, atlas: &SoftwareAtlas) -> Result<SoftwareCommand> {
    Ok(SoftwareCommand {
        order: sprite.order,
        clip: RectData::from_bounds(sprite.content_mask.bounds, "software sprite clip")?,
        kind: CommandKind::PolychromeSprite {
            bounds: RectData::from_bounds(sprite.bounds, "software sprite bounds")?,
            grayscale: sprite.is_grayscale(),
            opacity: FloatBits::new(sprite.opacity, "software sprite opacity")?,
            corner_radii: CornersData::new(sprite.corner_radii)?,
            image: AtlasImageData::new(atlas.tile(sprite.tile)?)?,
        },
    })
}

fn surface_command(surface: &PaintSurface) -> Result<SoftwareCommand> {
    Ok(SoftwareCommand {
        order: surface.order,
        clip: RectData::from_bounds(surface.content_mask.bounds, "software surface clip")?,
        kind: CommandKind::Surface {
            bounds: RectData::from_bounds(surface.bounds, "software surface bounds")?,
            frame_revision: surface.frame_revision,
            #[cfg(target_os = "macos")]
            image: crate::software_surface::snapshot_surface(&surface.image_buffer)?,
            #[cfg(not(target_os = "macos"))]
            image: SurfaceImageData::unavailable(),
        },
    })
}

fn command_bounds(command: &SoftwareCommand) -> Option<[f64; 4]> {
    let (bounds, expansion) = match &command.kind {
        CommandKind::Quad { bounds, .. }
        | CommandKind::Underline { bounds, .. }
        | CommandKind::PolychromeSprite { bounds, .. }
        | CommandKind::Surface { bounds, .. } => (*bounds, 1.0),
        CommandKind::Path { bounds, paint, .. } => {
            let expansion = match paint {
                PathPaintData::Fill(_) => 1.0,
                PathPaintData::Stroke(stroke) => f64::from(stroke.width.get()).max(0.0) * 0.5 + 1.0,
            };
            (*bounds, expansion)
        }
        CommandKind::Shadow {
            bounds,
            blur_radius,
            inset,
            element_bounds,
            ..
        } => {
            if *inset {
                (*element_bounds, 1.0)
            } else {
                (*bounds, f64::from(blur_radius.get()).max(0.0) * 3.0 + 1.0)
            }
        }
        CommandKind::MonochromeSprite {
            bounds, transform, ..
        }
        | CommandKind::SubpixelSprite {
            bounds, transform, ..
        } => {
            let edges = bounds.edges();
            let corners = [
                transform.apply([edges[0], edges[1]]),
                transform.apply([edges[2], edges[1]]),
                transform.apply([edges[2], edges[3]]),
                transform.apply([edges[0], edges[3]]),
            ];
            let left = corners
                .iter()
                .map(|point| point[0])
                .fold(f64::INFINITY, f64::min);
            let top = corners
                .iter()
                .map(|point| point[1])
                .fold(f64::INFINITY, f64::min);
            let right = corners
                .iter()
                .map(|point| point[0])
                .fold(f64::NEG_INFINITY, f64::max);
            let bottom = corners
                .iter()
                .map(|point| point[1])
                .fold(f64::NEG_INFINITY, f64::max);
            let transformed = RectData::new(
                left as f32,
                top as f32,
                (right - left) as f32,
                (bottom - top) as f32,
                "software transformed sprite bounds",
            )
            .ok()?;
            (transformed, 1.0)
        }
    };
    let edges = bounds.edges();
    let clip = command.clip.edges();
    let clipped = [
        (edges[0] - expansion).max(clip[0]),
        (edges[1] - expansion).max(clip[1]),
        (edges[2] + expansion).min(clip[2]),
        (edges[3] + expansion).min(clip[3]),
    ];
    (clipped[2] > clipped[0] && clipped[3] > clipped[1]).then_some(clipped)
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::{ContentMask, Path, PathStroke, Point, point, px, solid_background};

    fn bounds(left: f32, top: f32, width: f32, height: f32) -> Bounds<ScaledPixels> {
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

    fn quad(order: u32, rect: Bounds<ScaledPixels>) -> Quad {
        Quad {
            order,
            border_style: BorderStyle::Solid,
            bounds: rect,
            content_mask: ContentMask {
                bounds: bounds(0.0, 0.0, 256.0, 256.0),
            },
            background: solid_background(Hsla::red()),
            border_color: Hsla::default(),
            corner_radii: Corners::default(),
            border_widths: Edges::default(),
        }
    }

    #[test]
    fn tile_lists_compare_exactly_and_preserve_order() -> Result<()> {
        let atlas = SoftwareAtlas::new();
        let mut first = Scene::default();
        first.quads.push(quad(0, bounds(2.0, 2.0, 70.0, 20.0)));
        first.quads.push(quad(1, bounds(3.0, 3.0, 4.0, 4.0)));
        first.finish();
        let first = SoftwareScene::new(
            &first,
            Size {
                width: DevicePixels(128),
                height: DevicePixels(64),
            },
            1.0,
            Rgba::default(),
            &atlas,
        )?;
        let second = first.clone();
        assert_eq!(first, second);
        assert_eq!(first.tiles.len(), 2);
        assert_eq!(first.tiles[0].len(), 2);
        assert_eq!(first.tiles[1].len(), 1);
        assert_eq!(first.tiles[0][0].order, 0);
        assert_eq!(first.tiles[0][1].order, 1);
        Ok(())
    }

    #[test]
    fn rejects_non_finite_scene_fields() {
        let atlas = SoftwareAtlas::new();
        let mut scene = Scene::default();
        scene.quads.push(quad(0, bounds(f32::NAN, 0.0, 1.0, 1.0)));
        scene.finish();
        assert!(
            SoftwareScene::new(
                &scene,
                Size {
                    width: DevicePixels(1),
                    height: DevicePixels(1),
                },
                1.0,
                Rgba::default(),
                &atlas,
            )
            .is_err()
        );
    }

    #[test]
    fn thick_path_is_binned_by_its_painted_bounds() -> Result<()> {
        let atlas = SoftwareAtlas::new();
        let mut path = Path::new(point(px(63.0), px(8.0)));
        path.line_to(point(px(63.0), px(24.0)));
        path.set_stroke(PathStroke {
            width: 8.0,
            ..PathStroke::default()
        });
        let mut path = path.scale(1.0);
        path.content_mask = ContentMask {
            bounds: bounds(0.0, 0.0, 128.0, 64.0),
        };
        let mut scene = Scene::default();
        scene.paths.push(path);
        scene.finish();

        let software_scene = SoftwareScene::new(
            &scene,
            Size {
                width: DevicePixels(128),
                height: DevicePixels(64),
            },
            1.0,
            Rgba::default(),
            &atlas,
        )?;
        assert_eq!(software_scene.tiles.len(), 2);
        assert_eq!(software_scene.tiles[0].len(), 1);
        assert_eq!(software_scene.tiles[1].len(), 1);
        Ok(())
    }
}
