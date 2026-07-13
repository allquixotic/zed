use criterion::{Criterion, black_box, criterion_group, criterion_main};
use gpui::{
    BorderStyle, Bounds, ContentMask, Corners, DevicePixels, Edges, Hsla, Point, Quad, Rgba,
    ScaledPixels, Scene, Size, solid_background,
};
use gpui_software::{SoftwareAtlas, SoftwareRenderer};
use std::sync::Arc;

const WIDTH: i32 = 1920;
const HEIGHT: i32 = 1080;

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

fn quad(order: u32, quad_bounds: Bounds<ScaledPixels>, color: Hsla) -> Quad {
    Quad {
        order,
        border_style: BorderStyle::Solid,
        bounds: quad_bounds,
        content_mask: ContentMask {
            bounds: bounds(0.0, 0.0, WIDTH as f32, HEIGHT as f32),
        },
        background: solid_background(color),
        border_color: Hsla::transparent_black(),
        corner_radii: Corners::default(),
        border_widths: Edges::default(),
    }
}

fn editor_scene(caret_line: usize) -> Scene {
    let mut scene = Scene::default();
    scene.quads.push(quad(
        0,
        bounds(0.0, 0.0, WIDTH as f32, HEIGHT as f32),
        Hsla {
            h: 0.62,
            s: 0.08,
            l: 0.12,
            a: 1.0,
        },
    ));
    scene.quads.push(quad(
        1,
        bounds(0.0, 0.0, 280.0, HEIGHT as f32),
        Hsla {
            h: 0.62,
            s: 0.08,
            l: 0.16,
            a: 1.0,
        },
    ));
    for line in 0..56 {
        let top = 24.0 + line as f32 * 18.0;
        let width = 280.0 + ((line * 73) % 920) as f32;
        scene.quads.push(quad(
            u32::try_from(line).unwrap_or(u32::MAX).saturating_add(2),
            bounds(320.0, top, width, 10.0),
            Hsla {
                h: (line % 7) as f32 / 7.0,
                s: 0.35,
                l: 0.64,
                a: 0.78,
            },
        ));
    }
    scene.quads.push(quad(
        100,
        bounds(316.0, 22.0 + caret_line as f32 * 18.0, 2.0, 16.0),
        Hsla::white(),
    ));
    scene.finish();
    scene
}

fn benchmark_software_renderer(criterion: &mut Criterion) {
    let size = Size {
        width: DevicePixels(WIDTH),
        height: DevicePixels(HEIGHT),
    };
    let background = Rgba {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 1.0,
    };
    let first_scene = editor_scene(10);
    let second_scene = editor_scene(11);
    let mut group = criterion.benchmark_group("software_renderer_1920x1080");

    let mut unchanged_renderer = SoftwareRenderer::new(Arc::new(SoftwareAtlas::new()));
    group.bench_function("unchanged_scene", |bencher| {
        bencher.iter(|| {
            unchanged_renderer
                .render_frame(&first_scene, size, 1.0, background)
                .map(|frame| black_box(frame.rasterized_tiles))
        })
    });

    let mut localized_renderer = SoftwareRenderer::new(Arc::new(SoftwareAtlas::new()));
    let mut use_first_scene = false;
    group.bench_function("localized_caret_change", |bencher| {
        bencher.iter(|| {
            use_first_scene = !use_first_scene;
            let scene = if use_first_scene {
                &first_scene
            } else {
                &second_scene
            };
            localized_renderer
                .render_frame(scene, size, 1.0, background)
                .map(|frame| black_box(frame.rasterized_tiles))
        })
    });

    let mut full_renderer = SoftwareRenderer::new(Arc::new(SoftwareAtlas::new()));
    group.bench_function("full_scene", |bencher| {
        bencher.iter(|| {
            full_renderer.invalidate();
            full_renderer
                .render_frame(&first_scene, size, 1.0, background)
                .map(|frame| black_box(frame.rasterized_tiles))
        })
    });
    group.finish();
}

criterion_group!(benches, benchmark_software_renderer);
criterion_main!(benches);
