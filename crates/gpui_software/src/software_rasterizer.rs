use anyhow::{Context as _, Result, ensure};
use gpui::Rgba;
use vello_cpu::{
    Level, Pixmap, RenderContext, RenderMode, RenderSettings, Resources,
    color::{AlphaColor, Srgb},
    kurbo::Rect,
};

pub const SOFTWARE_TILE_SIZE: usize = 64;

#[derive(Default)]
pub struct SoftwareRasterizer {
    target: Option<RasterTarget>,
    size: [u16; 2],
    xrgb: Vec<u32>,
}

struct RasterTarget {
    context: RenderContext,
    resources: Resources,
    pixmap: Pixmap,
}

impl SoftwareRasterizer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn begin(&mut self, size: [u16; 2], background: Rgba) -> Result<()> {
        ensure!(
            usize::from(size[0]) <= SOFTWARE_TILE_SIZE
                && usize::from(size[1]) <= SOFTWARE_TILE_SIZE,
            "software rasterizer target exceeds {SOFTWARE_TILE_SIZE}x{SOFTWARE_TILE_SIZE}"
        );
        validate_color(background)?;
        let target_size_is_unchanged = self.size == size;
        self.size = size;
        if size[0] == 0 || size[1] == 0 {
            self.target = None;
            self.xrgb.clear();
            return Ok(());
        }

        if let Some(target) = self.target.as_mut().filter(|_| target_size_is_unchanged) {
            target.context.reset();
        } else {
            let settings = RenderSettings {
                level: Level::new(),
                num_threads: 0,
                render_mode: RenderMode::OptimizeSpeed,
            };
            self.target = Some(RasterTarget {
                context: RenderContext::new_with(size[0], size[1], settings),
                resources: Resources::new(),
                pixmap: Pixmap::new(size[0], size[1]),
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

        let target = self
            .target
            .as_mut()
            .context("software raster target was not initialized")?;
        let background = Rgba {
            a: 1.0,
            ..background
        };
        target.context.set_paint(vello_color(background));
        target
            .context
            .fill_rect(&Rect::new(0.0, 0.0, f64::from(size[0]), f64::from(size[1])));
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
        let target = self
            .target
            .as_mut()
            .context("software rasterizer begin was not called")?;
        target.context.set_paint(vello_color(color));
        target
            .context
            .fill_rect(&Rect::new(rect[0], rect[1], rect[2], rect[3]));
        Ok(())
    }

    pub fn finish(&mut self) -> Result<&[u32]> {
        let Some(target) = self.target.as_mut() else {
            ensure!(
                self.size[0] == 0 || self.size[1] == 0,
                "software raster target disappeared"
            );
            return Ok(&self.xrgb);
        };
        target.context.flush();
        target
            .context
            .render_to_pixmap(&mut target.resources, &mut target.pixmap);
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
}

fn vello_color(color: Rgba) -> AlphaColor<Srgb> {
    AlphaColor::new([
        color.r.clamp(0.0, 1.0),
        color.g.clamp(0.0, 1.0),
        color.b.clamp(0.0, 1.0),
        color.a.clamp(0.0, 1.0),
    ])
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
