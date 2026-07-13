use anyhow::{Context as _, Result, ensure};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ColorGlyphRenderingParams {
    pub(crate) gamma_ratios: [f32; 4],
    pub(crate) enhanced_contrast: f32,
}

#[derive(Debug, PartialEq)]
pub(crate) struct ColorGlyphLayer {
    pub(crate) origin: [i32; 2],
    pub(crate) size: [u32; 2],
    pub(crate) color: [f32; 4],
    pub(crate) alpha_mask: Vec<u8>,
}

pub(crate) fn rasterize_color_glyph(
    size: [u32; 2],
    layers: impl IntoIterator<Item = ColorGlyphLayer>,
    rendering_params: ColorGlyphRenderingParams,
) -> Result<Vec<u8>> {
    let target_width = i32::try_from(size[0]).context("color glyph width exceeds i32")?;
    let target_height = i32::try_from(size[1]).context("color glyph height exceeds i32")?;
    let pixel_count = usize::try_from(size[0])
        .ok()
        .and_then(|width| {
            usize::try_from(size[1])
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .context("color glyph dimensions overflow")?;
    let byte_count = pixel_count
        .checked_mul(4)
        .context("color glyph byte length overflows")?;
    let mut rasterized = Vec::new();
    rasterized
        .try_reserve_exact(byte_count)
        .context("allocating color glyph buffer")?;
    rasterized.resize(byte_count, 0);

    for layer in layers {
        composite_layer(
            &mut rasterized,
            [target_width, target_height],
            layer,
            rendering_params,
        )?;
    }

    for pixel in rasterized.chunks_exact_mut(4) {
        if pixel[3] > 0 {
            let inverse_alpha = 255.0 / pixel[3] as f32;
            pixel[0] = (pixel[0] as f32 * inverse_alpha).clamp(0.0, 255.0) as u8;
            pixel[1] = (pixel[1] as f32 * inverse_alpha).clamp(0.0, 255.0) as u8;
            pixel[2] = (pixel[2] as f32 * inverse_alpha).clamp(0.0, 255.0) as u8;
        }
    }

    Ok(rasterized)
}

fn composite_layer(
    rasterized: &mut [u8],
    target_size: [i32; 2],
    layer: ColorGlyphLayer,
    rendering_params: ColorGlyphRenderingParams,
) -> Result<()> {
    let layer_width = usize::try_from(layer.size[0]).context("invalid color layer width")?;
    let layer_height = usize::try_from(layer.size[1]).context("invalid color layer height")?;
    let expected_mask_length = layer_width
        .checked_mul(layer_height)
        .context("color layer dimensions overflow")?;
    ensure!(
        layer.alpha_mask.len() == expected_mask_length,
        "color layer mask length is {}, expected {expected_mask_length}",
        layer.alpha_mask.len()
    );
    if layer_width == 0 || layer_height == 0 {
        return Ok(());
    }

    let layer_width_i32 = i32::try_from(layer.size[0]).context("color layer width exceeds i32")?;
    let layer_height_i32 =
        i32::try_from(layer.size[1]).context("color layer height exceeds i32")?;
    layer.origin[0]
        .checked_add(layer_width_i32)
        .context("color layer horizontal bounds overflow")?;
    layer.origin[1]
        .checked_add(layer_height_i32)
        .context("color layer vertical bounds overflow")?;

    for (source_index, sample) in layer.alpha_mask.into_iter().enumerate() {
        let source_x = source_index % layer_width;
        let source_y = source_index / layer_width;
        let destination_x = layer.origin[0]
            + i32::try_from(source_x).context("color layer x coordinate exceeds i32")?;
        let destination_y = layer.origin[1]
            + i32::try_from(source_y).context("color layer y coordinate exceeds i32")?;
        if destination_x < 0
            || destination_y < 0
            || destination_x >= target_size[0]
            || destination_y >= target_size[1]
        {
            continue;
        }

        let destination_index = usize::try_from(destination_y)
            .ok()
            .and_then(|y| usize::try_from(target_size[0]).ok().map(|width| (y, width)))
            .and_then(|(y, width)| y.checked_mul(width))
            .and_then(|row_start| {
                usize::try_from(destination_x)
                    .ok()
                    .and_then(|x| row_start.checked_add(x))
            })
            .and_then(|pixel_index| pixel_index.checked_mul(4))
            .context("color glyph destination index overflows")?;
        let destination_end = destination_index
            .checked_add(4)
            .context("color glyph destination range overflows")?;
        let destination = rasterized
            .get_mut(destination_index..destination_end)
            .context("color glyph destination lies outside the buffer")?;

        let corrected_alpha = apply_contrast_and_gamma_correction(
            sample as f32 / 255.0,
            [layer.color[0], layer.color[1], layer.color[2]],
            rendering_params,
        );
        let source_alpha = saturate(corrected_alpha * layer.color[3]);
        let inverse_source_alpha = 1.0 - source_alpha;
        let destination_blue = destination[0] as f32 / 255.0;
        let destination_green = destination[1] as f32 / 255.0;
        let destination_red = destination[2] as f32 / 255.0;
        let destination_alpha = destination[3] as f32 / 255.0;

        destination[0] = to_unorm(
            saturate(layer.color[2]) * source_alpha + destination_blue * inverse_source_alpha,
        );
        destination[1] = to_unorm(
            saturate(layer.color[1]) * source_alpha + destination_green * inverse_source_alpha,
        );
        destination[2] = to_unorm(
            saturate(layer.color[0]) * source_alpha + destination_red * inverse_source_alpha,
        );
        destination[3] = to_unorm(source_alpha + destination_alpha * inverse_source_alpha);
    }

    Ok(())
}

fn apply_contrast_and_gamma_correction(
    sample: f32,
    color: [f32; 3],
    rendering_params: ColorGlyphRenderingParams,
) -> f32 {
    let brightness = color[0] * 0.30 + color[1] * 0.59 + color[2] * 0.11;
    let contrast_multiplier = saturate(4.0 * (0.75 - brightness));
    let enhanced_contrast = rendering_params.enhanced_contrast * contrast_multiplier;
    let contrasted = sample * (enhanced_contrast + 1.0) / (sample * enhanced_contrast + 1.0);
    let gamma = rendering_params.gamma_ratios;
    let brightness_adjustment = gamma[0] * brightness + gamma[1];
    let correction = brightness_adjustment * contrasted + (gamma[2] * brightness + gamma[3]);
    contrasted + contrasted * (1.0 - contrasted) * correction
}

fn saturate(value: f32) -> f32 {
    value.clamp(0.0, 1.0)
}

fn to_unorm(value: f32) -> u8 {
    (saturate(value) * 255.0).round() as u8
}
