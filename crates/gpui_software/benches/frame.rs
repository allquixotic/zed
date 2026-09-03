use std::{borrow::Cow, hint::black_box};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use gpui::{
    AtlasKey, AtlasTile, ContentMask, DevicePixels, MonochromeSprite, Path, Quad, RenderSvgParams,
    ScaledPixels, Scene, SharedString, Size, TransformationMatrix, bounds, hsla, point, px, size,
};
use gpui_software::{FontCorrection, SoftwareRenderer};

fn device_size(width: i32, height: i32) -> Size<DevicePixels> {
    Size {
        width: DevicePixels(width),
        height: DevicePixels(height),
    }
}

fn content_mask(width: i32, height: i32) -> ContentMask<ScaledPixels> {
    ContentMask {
        bounds: bounds(
            point(ScaledPixels(0.0), ScaledPixels(0.0)),
            size(ScaledPixels(width as f32), ScaledPixels(height as f32)),
        ),
    }
}

fn glyph_tile(renderer: &SoftwareRenderer) -> AtlasTile {
    let glyph_size = device_size(6, 10);
    let key = AtlasKey::Svg(RenderSvgParams {
        path: SharedString::from("software-benchmark-glyph"),
        size: glyph_size,
    });
    let coverage = (0..60)
        .map(|index| {
            if index % 6 == 0 || index % 6 == 5 {
                96
            } else {
                220
            }
        })
        .collect::<Vec<_>>();
    renderer
        .atlas()
        .get_or_insert_with(&key, &mut || {
            Ok(Some((glyph_size, Cow::Borrowed(&coverage))))
        })
        .expect("benchmark glyph atlas insertion failed")
        .expect("benchmark glyph builder returned no tile")
}

fn editor_scene(
    window_size: Size<DevicePixels>,
    tile: AtlasTile,
    changed_line: bool,
    scroll_offset: i32,
) -> Scene {
    let mut scene = Scene::default();
    let mask = content_mask(window_size.width.0, window_size.height.0);
    scene.insert_primitive(Quad {
        bounds: mask.bounds,
        content_mask: mask,
        background: hsla(0.62, 0.20, 0.10, 1.0).into(),
        ..Default::default()
    });
    for index in 0..200 {
        let x = ((index * 97) % window_size.width.0.max(1) as usize) as f32;
        let y = ((index * 53) % window_size.height.0.max(1) as usize) as f32;
        scene.insert_primitive(Quad {
            bounds: bounds(
                point(ScaledPixels(x), ScaledPixels(y)),
                size(ScaledPixels(120.0), ScaledPixels(22.0)),
            ),
            content_mask: mask,
            background: hsla(0.62, 0.12, 0.16, 1.0).into(),
            ..Default::default()
        });
    }
    for index in 0..20 {
        let x = 900.0 + (index % 5) as f32 * 48.0;
        let y = 80.0 + (index / 5) as f32 * 48.0;
        let mut path = Path::new(point(px(x), px(y)));
        path.line_to(point(px(x + 28.0), px(y + 12.0)));
        path.line_to(point(px(x + 8.0), px(y + 30.0)));
        path.color = hsla(0.55, 0.5, 0.55, 1.0).into();
        path.content_mask = ContentMask {
            bounds: bounds(
                point(px(0.0), px(0.0)),
                size(
                    px(window_size.width.0 as f32),
                    px(window_size.height.0 as f32),
                ),
            ),
        };
        scene.insert_primitive(path.scale(1.0));
    }
    for index in 0..8_000 {
        let column = index % 100;
        let row = index / 100;
        let y = 40 + row * 14 - scroll_offset;
        if y < -10 || y >= window_size.height.0 {
            continue;
        }
        let color = if changed_line && row == 20 {
            hsla(0.12, 0.8, 0.65, 1.0)
        } else {
            hsla(0.0, 0.0, 0.82, 1.0)
        };
        scene.insert_primitive(MonochromeSprite {
            order: 0,
            pad: 0,
            bounds: bounds(
                point(
                    ScaledPixels(24.0 + column as f32 * 8.0),
                    ScaledPixels(y as f32),
                ),
                size(ScaledPixels(6.0), ScaledPixels(10.0)),
            ),
            content_mask: mask,
            color,
            tile,
            transformation: TransformationMatrix::unit(),
        });
    }
    scene.finish();
    scene
}

fn frame_benchmarks(criterion: &mut Criterion) {
    let thread_count = rayon::current_num_threads();
    for (name, window_size) in [
        ("1080p", device_size(1920, 1080)),
        ("4k", device_size(3840, 2160)),
    ] {
        let mut renderer = SoftwareRenderer::new(window_size, FontCorrection::default());
        let tile = glyph_tile(&renderer);
        let scene = editor_scene(window_size, tile, false, 0);
        criterion.bench_with_input(
            BenchmarkId::new(format!("full-{name}"), thread_count),
            &thread_count,
            |bencher, _| {
                bencher.iter(|| black_box(renderer.draw(black_box(&scene), true)));
            },
        );

        let base = editor_scene(window_size, tile, false, 0);
        let changed = editor_scene(window_size, tile, true, 0);
        renderer.draw(&base, true);
        let mut alternate = false;
        criterion.bench_with_input(
            BenchmarkId::new(format!("single-line-{name}"), thread_count),
            &thread_count,
            |bencher, _| {
                bencher.iter(|| {
                    alternate = !alternate;
                    let scene = if alternate { &changed } else { &base };
                    black_box(renderer.draw(black_box(scene), false))
                });
            },
        );

        let scrolled = editor_scene(window_size, tile, false, 14);
        renderer.draw(&base, true);
        criterion.bench_with_input(
            BenchmarkId::new(format!("scroll-{name}"), thread_count),
            &thread_count,
            |bencher, _| {
                bencher.iter(|| {
                    alternate = !alternate;
                    let scene = if alternate { &scrolled } else { &base };
                    black_box(renderer.draw(black_box(scene), false))
                });
            },
        );
    }
}

criterion_group!(benches, frame_benchmarks);
criterion_main!(benches);
