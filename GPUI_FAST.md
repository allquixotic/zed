# Portable CPU optimizations from gpui-fast

This build uses the shared CPU-side parts of [longbridge/gpui-fast](https://github.com/longbridge/gpui-fast) at `86a8228229c8646b552d747434ffd690216a872c` (Apache-2.0):

- `fast/bounds_tree.rs`: a spatial grid and safe replay of prior draw orders.
- `fast/scene.rs`: reusable sort buffers, counting sorts and packed sort keys.
- `fast/path_cache.rs`: a bounded cache of tessellated vector paths, including translated copies.

The scene sorter preserves Zed stable's ordering of sprites within each draw order and texture. The GPU-specific tile cache, native composition and retained-view invalidation changes are not included. The selected code runs on every desktop platform before either GPU or software rendering. Its original randomized ordering, cache bounds and path equivalence tests accompany the port.

The software renderer is ported from `allquixotic/zed`'s `feature/gpui-software` at `5f91feb67fc71d7e08098ba1bd2bf4e72b58d76d`. The release branch starts at upstream stable `v1.27.0` (`c84af5b99cbc366d675064e52b23664a015b6e35`). No fork-specific Bedrock changes are included.

Build the Windows x64 Dev-channel installer with `script/bundle-windows.ps1 -Architecture x86_64`. The Dev channel keeps it separate from the official stable installation. Set `GPUI_RENDERER=software` before launch to force software rendering.
