# GPUI renderer benchmark

This package drives a real GPUI window through deterministic editor-shaped workloads and exits without input. It records raw per-frame timings, process CPU time, renderer identity, and a Markdown comparison.

The primary comparison is valid only on a Windows system where the legacy DirectX path reports a software-emulated DXGI adapter. The runner rejects a baseline that accidentally uses hardware acceleration.

Build the optimized runner before collecting measurements:

```powershell
cargo build -p gpui_renderer_bench --profile release-fast
```

On a GPU-less Windows system, compare the two renderers in the same executable:

```powershell
& .\target\release-fast\gpui-renderer-bench.exe compare `
  --output-dir .\target\renderer-benchmark-results `
  --rounds 3 `
  --width 1600 `
  --height 900 `
  --warmup-frames 30 `
  --measure-frames 180
```

Rounds alternate baseline-first and candidate-first to reduce thermal and ordering bias. The baseline is forced with `GPUI_RENDERER=directx` and `GPUI_D3D_ADAPTER=warp`; the candidate is forced with `GPUI_RENDERER=software`. Selecting WARP explicitly makes the legacy no-GPU path reproducible even when the benchmark host also has a physical GPU.

For the strictest upstream comparison, apply the benchmark-only commit to an unmodified upstream checkout and build one runner from each checkout. Then invoke the candidate runner with both paths:

```powershell
& .\candidate\gpui-renderer-bench.exe compare `
  --output-dir .\target\renderer-benchmark-results `
  --baseline-exe .\upstream\gpui-renderer-bench.exe `
  --candidate-exe .\candidate\gpui-renderer-bench.exe `
  --rounds 3
```

When `--baseline-exe` is supplied, the controller removes `GPUI_RENDERER` from that process so upstream chooses its normal no-GPU path. The candidate remains forced to `gpui_software`.

Each child process has a four-minute watchdog. The raw reports preserve every timing sample. `summary.json` and `summary.md` aggregate the rounds and calculate candidate changes without discarding regressions.

The scenarios are:

- `caret`: toggles a narrow caret in one editor row;
- `single_line`: changes one visible line while keeping the rest stable;
- `scroll`: advances the complete visible text viewport by one line;
- `full_frame`: changes the window and panel backgrounds.

`draw_ns` measures GPUI render-tree, layout, text, and scene construction. `renderer_present_ns` measures the platform renderer and submission/presentation call. `dirty_to_present_ns` measures from invalidation through presentation. Process CPU includes worker threads used by WARP or `gpui_software`.

On macOS, `run` can verify that the workload and reporting are portable, but the result uses Metal and is not a valid no-GPU baseline. macOS has no upstream software Metal fallback corresponding to the Windows software DXGI adapter. A second cross-platform A/B should use Linux with llvmpipe once the `gpui_software` Linux presenter is available.

Verify that benchmark builds do not enable GPUI test support:

```sh
feature_tree="$(cargo tree -p gpui_renderer_bench -e normal,build,dev,features)"
if grep -F 'feature "test-support"' <<<"${feature_tree}"; then
  exit 1
fi
```
