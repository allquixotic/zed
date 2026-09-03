use gpui::{Bounds, DevicePixels, Point, Size};

use crate::{
    bin_pass::{BinGrid, CELL_WIDTH},
    framebuffer::BAND_HEIGHT,
};

pub struct Damage {
    pub rects: Vec<Bounds<DevicePixels>>,
    pub(crate) dirty_cells: Vec<bool>,
    pub(crate) columns: usize,
}

impl Damage {
    pub fn is_empty(&self) -> bool {
        self.rects.is_empty()
    }
}

pub(crate) fn compute_damage(
    size: Size<DevicePixels>,
    bins: &BinGrid,
    previous_hashes: &[u64],
    force_full: bool,
) -> Damage {
    let mut dirty_cells = vec![false; bins.cells().len()];
    let mut rects = Vec::new();
    for row in 0..bins.rows() {
        let mut first_dirty = None;
        let mut last_dirty = None;
        for column in 0..bins.columns() {
            let index = row * bins.columns() + column;
            let dirty = force_full
                || previous_hashes
                    .get(index)
                    .is_none_or(|previous| *previous != bins.cells()[index].hash);
            dirty_cells[index] = dirty;
            if dirty {
                first_dirty.get_or_insert(column);
                last_dirty = Some(column);
            }
        }
        if let (Some(first), Some(last)) = (first_dirty, last_dirty) {
            let x0 = first * CELL_WIDTH;
            let x1 = ((last + 1) * CELL_WIDTH).min(size.width.0.max(0) as usize);
            let y0 = row * BAND_HEIGHT;
            let y1 = ((row + 1) * BAND_HEIGHT).min(size.height.0.max(0) as usize);
            rects.push(Bounds {
                origin: Point {
                    x: DevicePixels(x0 as i32),
                    y: DevicePixels(y0 as i32),
                },
                size: Size {
                    width: DevicePixels((x1 - x0) as i32),
                    height: DevicePixels((y1 - y0) as i32),
                },
            });
        }
    }
    Damage {
        rects,
        dirty_cells,
        columns: bins.columns(),
    }
}
