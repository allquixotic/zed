use crate::{SoftwareDamageRect, SoftwareFrame};
use anyhow::{Context as _, Result, ensure};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use softbuffer::{Context, Rect, Surface};
use std::{collections::VecDeque, num::NonZeroU32};

const DAMAGE_HISTORY_LENGTH: usize = 4;

pub struct SoftwarePresenter<D, W> {
    surface: Surface<D, W>,
    size: [u32; 2],
    damage_history: VecDeque<Vec<SoftwareDamageRect>>,
    retry_pending: bool,
}

impl<D, W> SoftwarePresenter<D, W>
where
    D: HasDisplayHandle,
    W: HasWindowHandle,
{
    pub fn new(display: D, window: W) -> Result<Self> {
        let context = Context::new(display).map_err(|error| {
            anyhow::anyhow!("creating software presentation context: {error:?}")
        })?;
        let surface = Surface::new(&context, window).map_err(|error| {
            anyhow::anyhow!("creating software presentation surface: {error:?}")
        })?;
        Ok(Self {
            surface,
            size: [0, 0],
            damage_history: VecDeque::with_capacity(DAMAGE_HISTORY_LENGTH),
            retry_pending: false,
        })
    }

    pub fn present(&mut self, frame: SoftwareFrame<'_>) -> Result<bool> {
        let size = frame.size;
        self.present_sized(frame, size)
    }

    pub fn present_sized(&mut self, frame: SoftwareFrame<'_>, size: [u32; 2]) -> Result<bool> {
        if !frame.changed() && !self.retry_pending {
            return Ok(false);
        }
        ensure!(
            frame.size == size,
            "software presentation size does not match the rendered frame"
        );
        let retry_pending = self.retry_pending;
        match self.present_sized_inner(frame, size, retry_pending) {
            Ok(presented) => {
                self.retry_pending = false;
                Ok(presented)
            }
            Err(error) => {
                self.retry_pending = true;
                Err(error)
            }
        }
    }

    fn present_sized_inner(
        &mut self,
        frame: SoftwareFrame<'_>,
        size: [u32; 2],
        retry_pending: bool,
    ) -> Result<bool> {
        let width = NonZeroU32::new(size[0]).context("software surface width is zero")?;
        let height = NonZeroU32::new(size[1]).context("software surface height is zero")?;
        let expected_length = usize::try_from(size[0])
            .ok()
            .and_then(|width| width.checked_mul(usize::try_from(size[1]).ok()?))
            .context("software presentation dimensions overflowed")?;
        ensure!(
            frame.framebuffer.len() == expected_length,
            "software presentation framebuffer length does not match its size"
        );
        let resized = self.size != size;
        if resized {
            self.surface.resize(width, height).map_err(|error| {
                anyhow::anyhow!("resizing software presentation surface: {error:?}")
            })?;
            self.size = size;
            self.damage_history.clear();
        }

        let mut buffer = self.surface.buffer_mut().map_err(|error| {
            anyhow::anyhow!("acquiring software presentation buffer: {error:?}")
        })?;
        let age = buffer.age();
        let damage = accumulated_damage(
            age,
            resized || retry_pending,
            frame.damage,
            &self.damage_history,
            size,
        );
        copy_damage(&mut buffer, frame.framebuffer, size[0], &damage)?;
        let softbuffer_damage = damage
            .iter()
            .filter_map(|damage| {
                Some(Rect {
                    x: damage.x,
                    y: damage.y,
                    width: NonZeroU32::new(damage.width)?,
                    height: NonZeroU32::new(damage.height)?,
                })
            })
            .collect::<Vec<_>>();
        buffer
            .present_with_damage(&softbuffer_damage)
            .map_err(|error| anyhow::anyhow!("presenting software framebuffer: {error:?}"))?;
        self.damage_history.push_front(if retry_pending {
            full_damage(size).into_iter().collect()
        } else {
            frame.damage.to_vec()
        });
        self.damage_history.truncate(DAMAGE_HISTORY_LENGTH);
        Ok(true)
    }
}

fn accumulated_damage(
    age: u8,
    resized: bool,
    current: &[SoftwareDamageRect],
    history: &VecDeque<Vec<SoftwareDamageRect>>,
    size: [u32; 2],
) -> Vec<SoftwareDamageRect> {
    let history_needed = usize::from(age.saturating_sub(1));
    if resized
        || age == 0
        || history_needed > history.len()
        || history_needed >= DAMAGE_HISTORY_LENGTH
    {
        return full_damage(size).into_iter().collect();
    }
    let mut damage = current.to_vec();
    for previous in history.iter().take(history_needed) {
        damage.extend_from_slice(previous);
    }
    coalesce_present_damage(damage)
}

fn full_damage(size: [u32; 2]) -> Option<SoftwareDamageRect> {
    (size[0] > 0 && size[1] > 0).then_some(SoftwareDamageRect {
        x: 0,
        y: 0,
        width: size[0],
        height: size[1],
    })
}

fn coalesce_present_damage(mut damage: Vec<SoftwareDamageRect>) -> Vec<SoftwareDamageRect> {
    damage.sort_by_key(|rect| (rect.y, rect.x));
    let mut horizontal: Vec<SoftwareDamageRect> = Vec::new();
    for rect in damage {
        if let Some(previous) = horizontal.last_mut().filter(|previous| {
            previous.y == rect.y
                && previous.height == rect.height
                && previous.x.saturating_add(previous.width) >= rect.x
        }) {
            let right = previous
                .x
                .saturating_add(previous.width)
                .max(rect.x.saturating_add(rect.width));
            previous.width = right.saturating_sub(previous.x);
        } else {
            horizontal.push(rect);
        }
    }
    let mut coalesced: Vec<SoftwareDamageRect> = Vec::new();
    for rect in horizontal {
        if let Some(previous) = coalesced.iter_mut().rev().find(|previous| {
            previous.x == rect.x
                && previous.width == rect.width
                && previous.y.saturating_add(previous.height) >= rect.y
        }) {
            let bottom = previous
                .y
                .saturating_add(previous.height)
                .max(rect.y.saturating_add(rect.height));
            previous.height = bottom.saturating_sub(previous.y);
        } else {
            coalesced.push(rect);
        }
    }
    coalesced
}

fn copy_damage(
    destination: &mut [u32],
    source: &[u32],
    width: u32,
    damage: &[SoftwareDamageRect],
) -> Result<()> {
    let width = usize::try_from(width).context("software copy width exceeds usize")?;
    ensure!(
        destination.len() == source.len(),
        "software presentation buffers differ in length"
    );
    ensure!(width > 0, "software copy width is zero");
    ensure!(
        source.len().is_multiple_of(width),
        "software presentation buffer length is not divisible by its width"
    );
    let height = source.len() / width;
    for rect in damage {
        let x = usize::try_from(rect.x).context("software damage x exceeds usize")?;
        let y = usize::try_from(rect.y).context("software damage y exceeds usize")?;
        let rect_width =
            usize::try_from(rect.width).context("software damage width exceeds usize")?;
        let rect_height =
            usize::try_from(rect.height).context("software damage height exceeds usize")?;
        ensure!(
            x.checked_add(rect_width)
                .is_some_and(|right| right <= width)
                && y.checked_add(rect_height)
                    .is_some_and(|bottom| bottom <= height),
            "software damage rectangle is outside the framebuffer"
        );
        for row in y..y.saturating_add(rect_height) {
            let start = row
                .checked_mul(width)
                .and_then(|row_start| row_start.checked_add(x))
                .context("software damage copy offset overflowed")?;
            let end = start
                .checked_add(rect_width)
                .context("software damage copy range overflowed")?;
            let source_row = source
                .get(start..end)
                .context("software damage source range is out of bounds")?;
            let destination_row = destination
                .get_mut(start..end)
                .context("software damage destination range is out of bounds")?;
            destination_row.copy_from_slice(source_row);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn damage(x: u32) -> SoftwareDamageRect {
        SoftwareDamageRect {
            x,
            y: 0,
            width: 1,
            height: 1,
        }
    }

    #[test]
    fn buffer_age_accumulates_only_required_history() {
        let history = VecDeque::from([
            vec![damage(1)],
            vec![damage(2)],
            vec![damage(3)],
            vec![damage(4)],
        ]);
        assert_eq!(
            accumulated_damage(0, false, &[damage(0)], &history, [8, 8]),
            vec![SoftwareDamageRect {
                x: 0,
                y: 0,
                width: 8,
                height: 8
            }]
        );
        assert_eq!(
            accumulated_damage(1, false, &[damage(0)], &history, [8, 8]),
            vec![damage(0)]
        );
        assert_eq!(
            accumulated_damage(2, false, &[damage(0)], &history, [8, 8]),
            vec![SoftwareDamageRect {
                x: 0,
                y: 0,
                width: 2,
                height: 1
            }]
        );
        assert_eq!(
            accumulated_damage(3, false, &[damage(0)], &history, [8, 8]),
            vec![SoftwareDamageRect {
                x: 0,
                y: 0,
                width: 3,
                height: 1
            }]
        );
        assert_eq!(
            accumulated_damage(4, false, &[damage(0)], &history, [8, 8]),
            vec![SoftwareDamageRect {
                x: 0,
                y: 0,
                width: 4,
                height: 1
            }]
        );
        assert_eq!(
            accumulated_damage(5, false, &[damage(0)], &history, [8, 8]),
            vec![SoftwareDamageRect {
                x: 0,
                y: 0,
                width: 8,
                height: 8
            }]
        );
    }

    #[test]
    fn damaged_copy_checks_every_range() -> Result<()> {
        let source = [1, 2, 3, 4];
        let mut destination = [0; 4];
        copy_damage(
            &mut destination,
            &source,
            2,
            &[SoftwareDamageRect {
                x: 1,
                y: 0,
                width: 1,
                height: 2,
            }],
        )?;
        assert_eq!(destination, [0, 2, 0, 4]);
        assert!(
            copy_damage(
                &mut destination,
                &source,
                2,
                &[SoftwareDamageRect {
                    x: 2,
                    y: 0,
                    width: 1,
                    height: 1,
                }],
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn present_damage_coalesces_vertical_neighbors() {
        assert_eq!(
            coalesce_present_damage(vec![
                SoftwareDamageRect {
                    x: 2,
                    y: 0,
                    width: 3,
                    height: 2,
                },
                SoftwareDamageRect {
                    x: 2,
                    y: 2,
                    width: 3,
                    height: 4,
                },
            ]),
            vec![SoftwareDamageRect {
                x: 2,
                y: 0,
                width: 3,
                height: 6,
            }]
        );
    }
}
