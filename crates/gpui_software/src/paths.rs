use gpui::{BackgroundTag, Path, ScaledPixels};
use vello_cpu::{
    Pixmap, RenderContext, Resources,
    color::{AlphaColor, Srgb},
    kurbo::{Affine, BezPath},
};

use crate::{
    kernels,
    lower::{IRect, gradient_affine, gradient_lut, pack_hsla},
};

pub(crate) fn rasterize_path(
    destination: &mut [u32],
    stride: usize,
    band_y: usize,
    rect: IRect,
    path: &Path<ScaledPixels>,
) {
    let Ok(width) = u16::try_from(rect.width()) else {
        return;
    };
    let Ok(height) = u16::try_from(rect.height()) else {
        return;
    };
    if width == 0 || height == 0 {
        return;
    }
    let bezier = to_bezier_path(path);
    if bezier.is_empty() {
        return;
    }
    let gradient = (path.color.tag() == BackgroundTag::LinearGradient).then(|| {
        (
            gradient_lut(path.color),
            gradient_affine(path.color, path.bounds),
        )
    });
    let color = if gradient.is_some() {
        0xffff_ffff
    } else {
        pack_hsla(path.color.solid())
    };
    let mut context = RenderContext::new(width, height);
    context.set_transform(Affine::translate((
        f64::from(-rect.x0),
        f64::from(-rect.y0),
    )));
    context.set_paint(AlphaColor::<Srgb>::from_rgba8(
        ((color >> 16) & 0xff) as u8,
        ((color >> 8) & 0xff) as u8,
        (color & 0xff) as u8,
        (color >> 24) as u8,
    ));
    context.fill_path(&bezier);
    context.flush();
    let mut resources = Resources::new();
    let mut pixmap = Pixmap::new(width, height);
    context.render(&mut pixmap, &mut resources);

    for (source_row, y) in pixmap.data().chunks(width as usize).zip(rect.y0..rect.y1) {
        let destination_start = (y as usize - band_y) * stride + rect.x0 as usize;
        let destination_row =
            &mut destination[destination_start..destination_start + width as usize];
        for ((destination, source), x) in destination_row
            .iter_mut()
            .zip(source_row)
            .zip(rect.x0..rect.x1)
        {
            if source.a == 0 {
                continue;
            }
            if let Some((lut, (t0, dt_dx, dt_dy))) = &gradient {
                let t = t0 + (x as f32 + 0.5) * dt_dx + (y as f32 + 0.5) * dt_dy;
                let color = lut[(t.clamp(0.0, 1.0) * 255.0).round() as usize];
                let alpha = ((color >> 24) * u32::from(source.a) + 127) / 255;
                *destination = kernels::blend_pixel(*destination, color, alpha as u8);
                continue;
            }
            let straight = if source.a == 255 {
                (u32::from(source.r) << 16) | (u32::from(source.g) << 8) | u32::from(source.b)
            } else {
                let unpremultiply = |channel: u8| {
                    ((u32::from(channel) * 255 + u32::from(source.a) / 2) / u32::from(source.a))
                        .min(255)
                };
                (unpremultiply(source.r) << 16)
                    | (unpremultiply(source.g) << 8)
                    | unpremultiply(source.b)
            };
            *destination = kernels::blend_pixel(*destination, straight, source.a);
        }
    }
}

fn to_bezier_path(path: &Path<ScaledPixels>) -> BezPath {
    let mut bezier = BezPath::new();
    for vertices in path.vertices.chunks_exact(3) {
        let point = |index: usize| {
            (
                f64::from(vertices[index].xy_position.x.0),
                f64::from(vertices[index].xy_position.y.0),
            )
        };
        let points = [point(0), point(1), point(2)];
        let curved = vertices[0].st_position.x == 0.0
            && vertices[0].st_position.y == 0.0
            && vertices[1].st_position.x == 0.5
            && vertices[1].st_position.y == 0.0
            && vertices[2].st_position.x == 1.0
            && vertices[2].st_position.y == 1.0;
        let signed_area = (points[1].0 - points[0].0) * (points[2].1 - points[0].1)
            - (points[1].1 - points[0].1) * (points[2].0 - points[0].0);
        if curved && signed_area < 0.0 {
            bezier.move_to(points[2]);
            bezier.quad_to(points[1], points[0]);
        } else {
            bezier.move_to(points[0]);
            if curved {
                bezier.quad_to(points[1], points[2]);
            } else if signed_area < 0.0 {
                bezier.line_to(points[2]);
                bezier.line_to(points[1]);
            } else {
                bezier.line_to(points[1]);
                bezier.line_to(points[2]);
            }
        }
        bezier.close_path();
    }
    bezier
}
