use anyhow::{Context as _, Result, ensure};
use collections::FxHashMap;
use etagere::{BucketedAtlasAllocator, size2};
use gpui::{
    AtlasKey, AtlasTextureId, AtlasTextureKind, AtlasTile, Bounds, DevicePixels, PlatformAtlas,
    Point, Size,
};
use parking_lot::Mutex;
use std::{borrow::Cow, collections::BTreeMap, sync::Arc};

const DEFAULT_ATLAS_SIZE: i32 = 1024;
const MAX_ATLAS_SIZE: i32 = 16_384;

pub struct SoftwareAtlas(Mutex<SoftwareAtlasState>);

#[derive(Clone, Debug)]
pub struct SoftwareAtlasTile {
    pub tile: AtlasTile,
    pub revision: u64,
    pub bytes_per_pixel: u8,
    pub pixels: Arc<[u8]>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SoftwareAtlasStats {
    pub cached_entries: usize,
    pub texture_pages: usize,
    pub stored_bytes: usize,
}

#[derive(Default)]
struct SoftwareAtlasState {
    storage: SoftwareAtlasStorage,
    tiles_by_key: FxHashMap<AtlasKey, AtlasTile>,
    next_revision: u64,
}

#[derive(Default)]
struct SoftwareAtlasStorage {
    monochrome: SoftwareTextureList,
    subpixel: SoftwareTextureList,
    polychrome: SoftwareTextureList,
}

#[derive(Default)]
struct SoftwareTextureList {
    textures: Vec<Option<SoftwareTexture>>,
    free_list: Vec<usize>,
}

struct SoftwareTexture {
    id: AtlasTextureId,
    allocator: BucketedAtlasAllocator,
    tiles: BTreeMap<u32, StoredTile>,
}

struct StoredTile {
    bounds: Bounds<DevicePixels>,
    revision: u64,
    pixels: Arc<[u8]>,
}

impl Default for SoftwareAtlas {
    fn default() -> Self {
        Self(Mutex::new(SoftwareAtlasState::default()))
    }
}

impl SoftwareAtlas {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn tile(&self, tile: AtlasTile) -> Result<SoftwareAtlasTile> {
        let state = self.0.lock();
        let texture = state
            .storage
            .texture(tile.texture_id)
            .context("software atlas texture does not exist")?;
        let stored = texture
            .tiles
            .get(&tile.tile_id.0)
            .context("software atlas tile does not exist")?;
        ensure!(
            stored.bounds == tile.bounds,
            "software atlas tile bounds do not match the allocation"
        );
        Ok(SoftwareAtlasTile {
            tile,
            revision: stored.revision,
            bytes_per_pixel: bytes_per_pixel(tile.texture_id.kind),
            pixels: stored.pixels.clone(),
        })
    }

    pub fn clear(&self) {
        let mut state = self.0.lock();
        state.storage = SoftwareAtlasStorage::default();
        state.tiles_by_key.clear();
    }

    pub fn stats(&self) -> SoftwareAtlasStats {
        let state = self.0.lock();
        let mut stats = SoftwareAtlasStats {
            cached_entries: state.tiles_by_key.len(),
            ..SoftwareAtlasStats::default()
        };
        for texture in state.storage.textures() {
            stats.texture_pages = stats.texture_pages.saturating_add(1);
            for tile in texture.tiles.values() {
                stats.stored_bytes = stats.stored_bytes.saturating_add(tile.pixels.len());
            }
        }
        stats
    }
}

impl PlatformAtlas for SoftwareAtlas {
    fn get_or_insert_with<'a>(
        &self,
        key: &AtlasKey,
        build: &mut dyn FnMut() -> Result<Option<(Size<DevicePixels>, Cow<'a, [u8]>)>>,
    ) -> Result<Option<AtlasTile>> {
        if let Some(tile) = self.0.lock().tiles_by_key.get(key).copied() {
            return Ok(Some(tile));
        }

        let Some((size, pixels)) = build()? else {
            return Ok(None);
        };
        validate_pixels(size, key.texture_kind(), &pixels)?;
        let mut owned_pixels = Vec::new();
        owned_pixels
            .try_reserve_exact(pixels.len())
            .context("allocating software atlas tile")?;
        owned_pixels.extend_from_slice(pixels.as_ref());
        let pixels: Arc<[u8]> = Arc::from(owned_pixels);

        let mut state = self.0.lock();
        if let Some(tile) = state.tiles_by_key.get(key).copied() {
            return Ok(Some(tile));
        }
        let tile = state.insert(size, key.texture_kind(), pixels)?;
        state.tiles_by_key.insert(key.clone(), tile);
        Ok(Some(tile))
    }

    fn remove(&self, key: &AtlasKey) {
        let mut state = self.0.lock();
        let Some(tile) = state.tiles_by_key.remove(key) else {
            return;
        };
        state.storage.remove(tile);
    }
}

impl SoftwareAtlasState {
    fn insert(
        &mut self,
        size: Size<DevicePixels>,
        kind: AtlasTextureKind,
        pixels: Arc<[u8]>,
    ) -> Result<AtlasTile> {
        self.next_revision = self
            .next_revision
            .checked_add(1)
            .context("software atlas revision overflowed")?;
        let revision = self.next_revision;
        let tile = self.storage.allocate(size, kind)?;
        let texture = self
            .storage
            .texture_mut(tile.texture_id)
            .context("newly allocated software atlas texture is missing")?;
        let previous = texture.tiles.insert(
            tile.tile_id.0,
            StoredTile {
                bounds: tile.bounds,
                revision,
                pixels,
            },
        );
        ensure!(
            previous.is_none(),
            "software atlas allocation ID was reused"
        );
        Ok(tile)
    }
}

impl SoftwareAtlasStorage {
    fn list(&self, kind: AtlasTextureKind) -> &SoftwareTextureList {
        match kind {
            AtlasTextureKind::Monochrome => &self.monochrome,
            AtlasTextureKind::Subpixel => &self.subpixel,
            AtlasTextureKind::Polychrome => &self.polychrome,
        }
    }

    fn list_mut(&mut self, kind: AtlasTextureKind) -> &mut SoftwareTextureList {
        match kind {
            AtlasTextureKind::Monochrome => &mut self.monochrome,
            AtlasTextureKind::Subpixel => &mut self.subpixel,
            AtlasTextureKind::Polychrome => &mut self.polychrome,
        }
    }

    fn texture(&self, id: AtlasTextureId) -> Option<&SoftwareTexture> {
        self.list(id.kind)
            .textures
            .get(usize::try_from(id.index).ok()?)?
            .as_ref()
    }

    fn texture_mut(&mut self, id: AtlasTextureId) -> Option<&mut SoftwareTexture> {
        self.list_mut(id.kind)
            .textures
            .get_mut(usize::try_from(id.index).ok()?)?
            .as_mut()
    }

    fn textures(&self) -> impl Iterator<Item = &SoftwareTexture> {
        self.monochrome
            .textures
            .iter()
            .chain(self.subpixel.textures.iter())
            .chain(self.polychrome.textures.iter())
            .flatten()
    }

    fn allocate(&mut self, size: Size<DevicePixels>, kind: AtlasTextureKind) -> Result<AtlasTile> {
        validate_size(size)?;
        if let Some(tile) = self
            .list_mut(kind)
            .textures
            .iter_mut()
            .rev()
            .flatten()
            .find_map(|texture| texture.allocate(size))
        {
            return Ok(tile);
        }

        self.push_texture(size, kind)?
            .allocate(size)
            .context("software atlas page could not fit the requested tile")
    }

    fn push_texture(
        &mut self,
        minimum_size: Size<DevicePixels>,
        kind: AtlasTextureKind,
    ) -> Result<&mut SoftwareTexture> {
        let width = minimum_size.width.0.max(DEFAULT_ATLAS_SIZE);
        let height = minimum_size.height.0.max(DEFAULT_ATLAS_SIZE);
        let list = self.list_mut(kind);
        let index = list.free_list.pop().unwrap_or(list.textures.len());
        let texture_id = AtlasTextureId {
            index: u32::try_from(index).context("software atlas has too many texture pages")?,
            kind,
        };
        let texture = SoftwareTexture {
            id: texture_id,
            allocator: BucketedAtlasAllocator::new(size2(width, height)),
            tiles: BTreeMap::new(),
        };
        if index == list.textures.len() {
            list.textures.push(Some(texture));
        } else {
            let slot = list
                .textures
                .get_mut(index)
                .context("software atlas free-list index is invalid")?;
            ensure!(slot.is_none(), "software atlas free-list slot is occupied");
            *slot = Some(texture);
        }
        list.textures
            .get_mut(index)
            .and_then(Option::as_mut)
            .context("software atlas texture insertion failed")
    }

    fn remove(&mut self, tile: AtlasTile) {
        let list = self.list_mut(tile.texture_id.kind);
        let Ok(index) = usize::try_from(tile.texture_id.index) else {
            return;
        };
        let Some(slot) = list.textures.get_mut(index) else {
            return;
        };
        let Some(texture) = slot.as_mut() else {
            return;
        };
        if texture.tiles.remove(&tile.tile_id.0).is_none() {
            return;
        }
        texture.allocator.deallocate(tile.tile_id.into());
        if texture.tiles.is_empty() {
            *slot = None;
            list.free_list.push(index);
        }
    }
}

impl SoftwareTexture {
    fn allocate(&mut self, size: Size<DevicePixels>) -> Option<AtlasTile> {
        let allocation = self
            .allocator
            .allocate(size2(size.width.0, size.height.0))?;
        Some(AtlasTile {
            texture_id: self.id,
            tile_id: allocation.id.into(),
            padding: 0,
            bounds: Bounds {
                origin: Point {
                    x: DevicePixels(allocation.rectangle.min.x),
                    y: DevicePixels(allocation.rectangle.min.y),
                },
                size,
            },
        })
    }
}

fn validate_size(size: Size<DevicePixels>) -> Result<()> {
    ensure!(
        size.width.0 > 0 && size.height.0 > 0,
        "software atlas tiles must have positive dimensions"
    );
    ensure!(
        size.width.0 <= MAX_ATLAS_SIZE && size.height.0 <= MAX_ATLAS_SIZE,
        "software atlas tile exceeds {MAX_ATLAS_SIZE} pixels"
    );
    Ok(())
}

fn validate_pixels(size: Size<DevicePixels>, kind: AtlasTextureKind, pixels: &[u8]) -> Result<()> {
    validate_size(size)?;
    let width = usize::try_from(size.width.0).context("software atlas width is negative")?;
    let height = usize::try_from(size.height.0).context("software atlas height is negative")?;
    let expected_length = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(usize::from(bytes_per_pixel(kind))))
        .context("software atlas byte length overflowed")?;
    ensure!(
        pixels.len() == expected_length,
        "software atlas received {} bytes for a {width}x{height} {kind:?} tile; expected {expected_length}",
        pixels.len()
    );
    Ok(())
}

fn bytes_per_pixel(kind: AtlasTextureKind) -> u8 {
    match kind {
        AtlasTextureKind::Monochrome => 1,
        AtlasTextureKind::Subpixel | AtlasTextureKind::Polychrome => 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::{ImageId, RenderImageParams, size};
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn image_key(id: usize) -> AtlasKey {
        AtlasKey::Image(RenderImageParams {
            image_id: ImageId(id),
            frame_index: 0,
        })
    }

    #[test]
    fn caches_and_returns_ram_backed_tiles() -> Result<()> {
        let atlas = SoftwareAtlas::new();
        let builds = AtomicUsize::new(0);
        let mut build = || -> Result<Option<(Size<DevicePixels>, Cow<'static, [u8]>)>> {
            builds.fetch_add(1, Ordering::SeqCst);
            Ok(Some((
                size(DevicePixels(2), DevicePixels(1)),
                Cow::Borrowed(&[1, 2, 3, 4, 5, 6, 7, 8]),
            )))
        };
        let key = image_key(1);

        let first = atlas
            .get_or_insert_with(&key, &mut build)?
            .context("tile was not inserted")?;
        let second = atlas
            .get_or_insert_with(&key, &mut build)?
            .context("cached tile was not returned")?;

        assert_eq!(first, second);
        assert_eq!(first.padding, 0);
        assert_eq!(builds.load(Ordering::SeqCst), 1);
        let snapshot = atlas.tile(first)?;
        assert_eq!(snapshot.revision, 1);
        assert_eq!(snapshot.bytes_per_pixel, 4);
        assert_eq!(&*snapshot.pixels, &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(
            atlas.stats(),
            SoftwareAtlasStats {
                cached_entries: 1,
                texture_pages: 1,
                stored_bytes: 8,
            }
        );
        Ok(())
    }

    #[test]
    fn removal_and_reuse_change_the_revision() -> Result<()> {
        let atlas = SoftwareAtlas::new();
        let first_key = image_key(1);
        let second_key = image_key(2);
        let mut build = || -> Result<Option<(Size<DevicePixels>, Cow<'static, [u8]>)>> {
            Ok(Some((
                size(DevicePixels(1), DevicePixels(1)),
                Cow::Borrowed(&[0, 0, 0, 255]),
            )))
        };
        let first = atlas
            .get_or_insert_with(&first_key, &mut build)?
            .context("first tile was not inserted")?;
        let first_revision = atlas.tile(first)?.revision;
        atlas.remove(&first_key);
        assert!(atlas.tile(first).is_err());

        let second = atlas
            .get_or_insert_with(&second_key, &mut build)?
            .context("second tile was not inserted")?;
        assert_eq!(second.texture_id.index, first.texture_id.index);
        assert!(atlas.tile(second)?.revision > first_revision);
        Ok(())
    }

    #[test]
    fn rejects_malformed_or_extreme_tile_data() {
        let atlas = SoftwareAtlas::new();
        let key = image_key(1);
        let mut malformed = || -> Result<Option<(Size<DevicePixels>, Cow<'static, [u8]>)>> {
            Ok(Some((
                size(DevicePixels(2), DevicePixels(2)),
                Cow::Borrowed(&[0; 15]),
            )))
        };
        assert!(atlas.get_or_insert_with(&key, &mut malformed).is_err());

        let mut extreme = || -> Result<Option<(Size<DevicePixels>, Cow<'static, [u8]>)>> {
            Ok(Some((
                size(DevicePixels(MAX_ATLAS_SIZE + 1), DevicePixels(1)),
                Cow::Borrowed(&[]),
            )))
        };
        assert!(atlas.get_or_insert_with(&key, &mut extreme).is_err());
        assert_eq!(atlas.stats(), SoftwareAtlasStats::default());
    }
}
