use std::hash::{Hash, Hasher};

use gpui::{DevicePixels, Size};

use crate::{
    framebuffer::BAND_HEIGHT,
    lower::{IRect, Op, framebuffer_rect},
};

pub(crate) const CELL_WIDTH: usize = 64;

#[derive(Default)]
pub(crate) struct Cell {
    pub ops: Vec<u32>,
    pub hash: u64,
    pub opaque_cutoff: Option<usize>,
}

pub(crate) struct BinGrid {
    columns: usize,
    rows: usize,
    size: Size<DevicePixels>,
    cells: Vec<Cell>,
}

impl BinGrid {
    pub fn new(
        size: Size<DevicePixels>,
        ops: &[Op],
        atlas: &crate::atlas::SoftwareAtlasState,
    ) -> Self {
        let width = size.width.0.max(0) as usize;
        let height = size.height.0.max(0) as usize;
        let columns = width.div_ceil(CELL_WIDTH);
        let rows = height.div_ceil(BAND_HEIGHT);
        let mut grid = Self {
            columns,
            rows,
            size,
            cells: (0..columns.saturating_mul(rows))
                .map(|_| Cell::default())
                .collect(),
        };
        let frame = framebuffer_rect(size);
        for (op_index, op) in ops.iter().enumerate() {
            let rect = op.rect().intersect(frame);
            if rect.is_empty() {
                continue;
            }
            let first_column = rect.x0.max(0) as usize / CELL_WIDTH;
            let last_column = (rect.x1 - 1).max(0) as usize / CELL_WIDTH;
            let first_row = rect.y0.max(0) as usize / BAND_HEIGHT;
            let last_row = (rect.y1 - 1).max(0) as usize / BAND_HEIGHT;
            let op_hash = op.hash(atlas);
            for row in first_row..=last_row.min(rows.saturating_sub(1)) {
                for column in first_column..=last_column.min(columns.saturating_sub(1)) {
                    let cell_rect = grid.cell_rect(row, column);
                    let cell = &mut grid.cells[row * columns + column];
                    if op.is_opaque_rectangle() && rect.contains(cell_rect) {
                        cell.opaque_cutoff = Some(cell.ops.len());
                    }
                    cell.ops.push(op_index as u32);
                    let mut hasher = collections::FxHasher::default();
                    cell.hash.hash(&mut hasher);
                    op_hash.hash(&mut hasher);
                    cell.hash = hasher.finish();
                }
            }
        }
        grid
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cells(&self) -> &[Cell] {
        &self.cells
    }

    pub fn cell(&self, row: usize, column: usize) -> &Cell {
        &self.cells[row * self.columns + column]
    }

    pub fn cell_rect(&self, row: usize, column: usize) -> IRect {
        IRect {
            x0: (column * CELL_WIDTH) as i32,
            y0: (row * BAND_HEIGHT) as i32,
            x1: ((column + 1) * CELL_WIDTH).min(self.size.width.0.max(0) as usize) as i32,
            y1: ((row + 1) * BAND_HEIGHT).min(self.size.height.0.max(0) as usize) as i32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opaque_cover_skips_earlier_operations() {
        let rect = IRect {
            x0: 0,
            y0: 0,
            x1: 64,
            y1: 32,
        };
        let ops = vec![
            Op::FillBlend {
                rect,
                color: 0x80ff_0000,
            },
            Op::FillOpaque {
                rect,
                color: 0xff00_ff00,
            },
            Op::FillBlend {
                rect,
                color: 0x8000_00ff,
            },
        ];
        let grid = BinGrid::new(
            Size {
                width: DevicePixels(64),
                height: DevicePixels(32),
            },
            &ops,
            &crate::SoftwareAtlas::new().lock(),
        );
        assert_eq!(grid.cell(0, 0).opaque_cutoff, Some(1));
    }
}
