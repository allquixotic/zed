#[cfg(any(target_os = "macos", test))]
use anyhow::{Context as _, Result, ensure};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SurfaceImageKind {
    Rgba,
    #[cfg(any(target_os = "macos", test))]
    Unsupported(u32),
    #[cfg(not(target_os = "macos"))]
    Unavailable,
}

#[derive(Clone, Debug)]
pub(crate) struct SurfaceImageData {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) kind: SurfaceImageKind,
    pub(crate) pixels: Arc<[u8]>,
}

impl PartialEq for SurfaceImageData {
    fn eq(&self, other: &Self) -> bool {
        self.width == other.width && self.height == other.height && self.kind == other.kind
    }
}

impl Eq for SurfaceImageData {}

impl SurfaceImageData {
    #[cfg(not(target_os = "macos"))]
    pub(crate) fn unavailable() -> Self {
        Self {
            width: 0,
            height: 0,
            kind: SurfaceImageKind::Unavailable,
            pixels: Arc::from([]),
        }
    }

    #[cfg(target_os = "macos")]
    fn unsupported(format: u32) -> Self {
        Self {
            width: 0,
            height: 0,
            kind: SurfaceImageKind::Unsupported(format),
            pixels: Arc::from([]),
        }
    }

    #[cfg(any(target_os = "macos", test))]
    pub(crate) fn rgba(width: usize, height: usize, pixels: Vec<u8>) -> Result<Self> {
        Ok(Self {
            width: u32::try_from(width).context("software surface width exceeds u32")?,
            height: u32::try_from(height).context("software surface height exceeds u32")?,
            kind: SurfaceImageKind::Rgba,
            pixels: Arc::from(pixels),
        })
    }
}

#[cfg(any(target_os = "macos", test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum YCbCrRange {
    Full,
    Video,
}

#[cfg(any(target_os = "macos", test))]
pub(crate) fn convert_bgra(
    source: &[u8],
    stride: usize,
    width: usize,
    height: usize,
) -> Result<Vec<u8>> {
    ensure!(
        stride >= width.checked_mul(4).context("BGRA row width overflowed")?,
        "BGRA surface stride is shorter than its width"
    );
    ensure!(
        source.len()
            >= stride
                .checked_mul(height)
                .context("BGRA surface byte length overflowed")?,
        "BGRA surface plane is shorter than its declared dimensions"
    );
    let pixel_count = width
        .checked_mul(height)
        .context("BGRA surface dimensions overflowed")?;
    let byte_count = pixel_count
        .checked_mul(4)
        .context("BGRA surface output length overflowed")?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(byte_count)
        .context("allocating BGRA surface conversion")?;
    for row in 0..height {
        let row_start = row
            .checked_mul(stride)
            .context("BGRA surface row offset overflowed")?;
        for column in 0..width {
            let offset = row_start
                .checked_add(
                    column
                        .checked_mul(4)
                        .context("BGRA surface pixel offset overflowed")?,
                )
                .context("BGRA surface pixel offset overflowed")?;
            let pixel = source
                .get(offset..offset.saturating_add(4))
                .context("BGRA surface pixel is out of bounds")?;
            output.extend_from_slice(&[pixel[2], pixel[1], pixel[0], pixel[3]]);
        }
    }
    Ok(output)
}

#[cfg(any(target_os = "macos", test))]
pub(crate) fn convert_nv12(
    luma: &[u8],
    luma_stride: usize,
    chroma: &[u8],
    chroma_stride: usize,
    width: usize,
    height: usize,
    range: YCbCrRange,
) -> Result<Vec<u8>> {
    ensure!(
        luma_stride >= width,
        "NV12 luma stride is shorter than its width"
    );
    let chroma_width = width.div_ceil(2);
    ensure!(
        chroma_stride
            >= chroma_width
                .checked_mul(2)
                .context("NV12 chroma row width overflowed")?,
        "NV12 chroma stride is shorter than its width"
    );
    let chroma_height = height.div_ceil(2);
    ensure!(
        luma.len()
            >= luma_stride
                .checked_mul(height)
                .context("NV12 luma byte length overflowed")?,
        "NV12 luma plane is shorter than its declared dimensions"
    );
    ensure!(
        chroma.len()
            >= chroma_stride
                .checked_mul(chroma_height)
                .context("NV12 chroma byte length overflowed")?,
        "NV12 chroma plane is shorter than its declared dimensions"
    );
    let byte_count = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(4))
        .context("NV12 surface output length overflowed")?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(byte_count)
        .context("allocating NV12 surface conversion")?;
    for row in 0..height {
        let luma_row = row
            .checked_mul(luma_stride)
            .context("NV12 luma row offset overflowed")?;
        let chroma_row = (row / 2)
            .checked_mul(chroma_stride)
            .context("NV12 chroma row offset overflowed")?;
        for column in 0..width {
            let y = f32::from(
                *luma
                    .get(luma_row.saturating_add(column))
                    .context("NV12 luma sample is out of bounds")?,
            );
            let chroma_offset = chroma_row
                .checked_add(
                    (column / 2)
                        .checked_mul(2)
                        .context("NV12 chroma sample offset overflowed")?,
                )
                .context("NV12 chroma sample offset overflowed")?;
            let cb = f32::from(
                *chroma
                    .get(chroma_offset)
                    .context("NV12 Cb sample is out of bounds")?,
            );
            let cr = f32::from(
                *chroma
                    .get(chroma_offset.saturating_add(1))
                    .context("NV12 Cr sample is out of bounds")?,
            );
            let [red, green, blue] = ycbcr_to_rgb(y, cb, cr, range);
            output.extend_from_slice(&[red, green, blue, 255]);
        }
    }
    Ok(output)
}

#[cfg(any(target_os = "macos", test))]
fn ycbcr_to_rgb(y: f32, cb: f32, cr: f32, range: YCbCrRange) -> [u8; 3] {
    let (red, green, blue) = match range {
        YCbCrRange::Full => {
            let cb = cb - 128.0;
            let cr = cr - 128.0;
            (
                y + 1.402 * cr,
                y - 0.344_136 * cb - 0.714_136 * cr,
                y + 1.772 * cb,
            )
        }
        YCbCrRange::Video => {
            let y = 1.164_384 * (y - 16.0);
            let cb = cb - 128.0;
            let cr = cr - 128.0;
            (
                y + 1.596_027 * cr,
                y - 0.391_762 * cb - 0.812_968 * cr,
                y + 2.017_232 * cb,
            )
        }
    };
    [to_u8(red), to_u8(green), to_u8(blue)]
}

#[cfg(any(target_os = "macos", test))]
fn to_u8(value: f32) -> u8 {
    value.clamp(0.0, 255.0).round() as u8
}

#[cfg(target_os = "macos")]
pub(crate) fn snapshot_surface(
    buffer: &core_video::pixel_buffer::CVPixelBuffer,
) -> Result<SurfaceImageData> {
    use core_video::{
        pixel_buffer::{
            kCVPixelBufferLock_ReadOnly, kCVPixelFormatType_32BGRA,
            kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
            kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
        },
        r#return::kCVReturnSuccess,
    };
    use std::{collections::HashSet, slice, sync::OnceLock};

    static UNSUPPORTED_FORMATS: OnceLock<parking_lot::Mutex<HashSet<u32>>> = OnceLock::new();

    let format = buffer.get_pixel_format();
    if format != kCVPixelFormatType_32BGRA
        && format != kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
        && format != kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
    {
        let mut formats = UNSUPPORTED_FORMATS
            .get_or_init(|| parking_lot::Mutex::new(HashSet::new()))
            .lock();
        if formats.insert(format) {
            log::warn!("software renderer does not support CVPixelBuffer format {format:#010x}");
        }
        return Ok(SurfaceImageData::unsupported(format));
    }

    let lock_status = buffer.lock_base_address(kCVPixelBufferLock_ReadOnly);
    ensure!(
        lock_status == kCVReturnSuccess,
        "locking CVPixelBuffer failed with status {lock_status}"
    );
    let conversion = (|| {
        let width = buffer.get_width();
        let height = buffer.get_height();
        ensure!(
            width > 0 && height > 0,
            "CVPixelBuffer dimensions are empty"
        );
        if format == kCVPixelFormatType_32BGRA {
            let stride = buffer.get_bytes_per_row();
            let length = stride
                .checked_mul(height)
                .context("CVPixelBuffer BGRA byte length overflowed")?;
            let address = unsafe { buffer.get_base_address() }.cast::<u8>();
            ensure!(
                !address.is_null(),
                "CVPixelBuffer BGRA base address is null"
            );
            // The buffer remains locked for this slice's lifetime, and CoreVideo reports its exact stride and height.
            let source = unsafe { slice::from_raw_parts(address, length) };
            SurfaceImageData::rgba(width, height, convert_bgra(source, stride, width, height)?)
        } else {
            ensure!(
                buffer.get_plane_count() == 2,
                "NV12 CVPixelBuffer does not have two planes"
            );
            let luma_stride = buffer.get_bytes_per_row_of_plane(0);
            let chroma_stride = buffer.get_bytes_per_row_of_plane(1);
            let luma_height = buffer.get_height_of_plane(0);
            let chroma_height = buffer.get_height_of_plane(1);
            let luma_length = luma_stride
                .checked_mul(luma_height)
                .context("CVPixelBuffer luma byte length overflowed")?;
            let chroma_length = chroma_stride
                .checked_mul(chroma_height)
                .context("CVPixelBuffer chroma byte length overflowed")?;
            let luma_address = unsafe { buffer.get_base_address_of_plane(0) }.cast::<u8>();
            let chroma_address = unsafe { buffer.get_base_address_of_plane(1) }.cast::<u8>();
            ensure!(
                !luma_address.is_null() && !chroma_address.is_null(),
                "CVPixelBuffer NV12 plane address is null"
            );
            // Both plane slices are bounded by CoreVideo's locked per-plane stride and height.
            let luma = unsafe { slice::from_raw_parts(luma_address, luma_length) };
            let chroma = unsafe { slice::from_raw_parts(chroma_address, chroma_length) };
            let range = if format == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange {
                YCbCrRange::Full
            } else {
                YCbCrRange::Video
            };
            SurfaceImageData::rgba(
                width,
                height,
                convert_nv12(
                    luma,
                    luma_stride,
                    chroma,
                    chroma_stride,
                    width,
                    height,
                    range,
                )?,
            )
        }
    })();
    let unlock_status = buffer.unlock_base_address(kCVPixelBufferLock_ReadOnly);
    match (conversion, unlock_status == kCVReturnSuccess) {
        (Ok(image), true) => Ok(image),
        (Err(error), true) => Err(error),
        (Ok(_), false) => {
            anyhow::bail!("unlocking CVPixelBuffer failed with status {unlock_status}")
        }
        (Err(error), false) => Err(error.context(format!(
            "unlocking CVPixelBuffer also failed with status {unlock_status}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_bgra_with_odd_stride() -> Result<()> {
        let source = [
            0, 0, 255, 255, 9, 8, 7, 6, 99, 255, 0, 0, 255, 1, 2, 3, 4, 88,
        ];
        assert_eq!(
            convert_bgra(&source, 9, 2, 2)?,
            [255, 0, 0, 255, 7, 8, 9, 6, 0, 0, 255, 255, 3, 2, 1, 4]
        );
        Ok(())
    }

    #[test]
    fn converts_full_and_video_range_nv12() -> Result<()> {
        let neutral = [128, 128];
        assert_eq!(
            convert_nv12(&[0], 1, &neutral, 2, 1, 1, YCbCrRange::Full)?,
            [0, 0, 0, 255]
        );
        assert_eq!(
            convert_nv12(&[255], 1, &neutral, 2, 1, 1, YCbCrRange::Full)?,
            [255, 255, 255, 255]
        );
        assert_eq!(
            convert_nv12(&[16], 1, &neutral, 2, 1, 1, YCbCrRange::Video)?,
            [0, 0, 0, 255]
        );
        assert_eq!(
            convert_nv12(&[235], 1, &neutral, 2, 1, 1, YCbCrRange::Video)?,
            [255, 255, 255, 255]
        );
        Ok(())
    }

    #[test]
    fn converts_nv12_with_odd_plane_strides() -> Result<()> {
        let pixels = convert_nv12(
            &[16, 235, 16, 99, 99, 235, 16, 235, 99, 99],
            5,
            &[128, 128, 128, 128, 99, 99],
            6,
            3,
            2,
            YCbCrRange::Video,
        )?;
        assert_eq!(pixels.len(), 24);
        assert_eq!(&pixels[0..4], &[0, 0, 0, 255]);
        assert_eq!(&pixels[4..8], &[255, 255, 255, 255]);
        assert_eq!(&pixels[20..24], &[255, 255, 255, 255]);
        Ok(())
    }

    #[test]
    fn malformed_surface_planes_are_rejected() {
        assert!(convert_bgra(&[0; 7], 8, 2, 1).is_err());
        assert!(convert_nv12(&[0], 1, &[0], 2, 1, 1, YCbCrRange::Full).is_err());
    }
}
