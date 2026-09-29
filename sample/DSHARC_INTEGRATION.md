# DSHARC sample integration and results

Run `python sample/entry_point.py --renderer dsharc`. The UI offers REFERENCE,
SHARC and DSHARC buttons; mode changes invalidate both image and cache history.
SHARC remains the default so existing launch commands retain their behavior.
This is our local `include/DSharc*.h` implementation, not NVIDIA ReGIR. The core
headers are unchanged; this sample supplies tracing, materials and scheduling.

## Frame flow

`Begin → Request → Compact → indirect args → Material/Lighting → Resolve → Gather`

Explicit barriers separate all phases. Previous/current FP32 radiance buffers
are separate allocations and swap only after Gather. Update and Resolve use
GPU-written indirect compute arguments; no active-count CPU readback is needed.
The adapter limits capacity to 2^21 so a one-dimensional 64-thread indirect
dispatch stays below 65535 groups. All resource strides are checked by reflection.

- Request traces each camera sample's prefix. An eligible diffuse continuation
  longer than the cell diagonal calls `DSharcRequest`. It saves radiance,
  throughput and entry index, then suspends before shading that hit. Camera hits
  are never replaced. Failed requests continue ordinary PT, including full-table
  failures; the unit tests exercise a deliberately tiny 1024-slot table.
- The first requester supplies the representative position and geometric normal.
  A short ray along its negative normal reconstructs material and smooth normal.
  Distance and normal checks reject a probe that lands on another surface.
- Material/Lighting is fused per active cell. It samples solar NEE and a cosine
  continuation, reading only frozen previous radiance. A missing dependency
  continues PT instead of returning zero. The sun's BSDF-side MIS weight is
  preserved on the first escape segment. Update has a 32-event cap.
- `DSharcEvaluateDiffuseRadiance` applies emission and albedo exactly once.
  The cache stores total outgoing radiance; unlike the SHARC adapter, this
  baseline does not add material demodulation to the DSHARC library contract.
- Resolve temporally averages the cell's estimate. Gather adds cached radiance
  times saved throughput. If material/estimate/current radiance is invalid, it
  retraces the original camera sample with cache disabled and replaces the whole
  result, avoiding a dropped tail or a duplicated prefix.

Defaults: capacity 2^20, base cell .025, level distance 2, maximum level 12,
history 256, stale age 16. The distance bands approximately match the SHARC
adapter: .025 / .05 / .1 world-unit cells at distances below 4 / 8 / 16.
DSHARC uses six dominant-axis normal bins; SHARC's key uses normal sign bits.

Cache storage is 140 MiB at default capacity plus a 4-byte active counter.
Suspended paths require **32 bytes × width × height × spp per frame**; e.g.
480×320 costs 4.69 MiB at 1 spp or 150 MiB at 32 spp per frame. Cache state and
image both reset on scene/camera/light/grid/mode changes. Image-only reset keeps
cache history and advances the RNG sequence. Dynamic geometry isn't supported.

## Measured image quality

Windows / RTX 5080 / SlangPy 0.43.1, default diffuse room and lights, 480×320.
All displayed images have +1 EV and no denoiser. Both caches get 256 warmup frames
at 1 spp/frame, then their image accumulation is reset while cache history is
preserved. The following error is linear-luminance RMSE divided by the independent
32768-spp reference's mean luminance (not mean per-pixel relative error).

| Fresh image spp | Reference PT | Warm SHARC | Warm DSHARC |
| --- | ---: | ---: | ---: |
| 1 | 161.66% | 129.58% | 123.93% |
| 32 | 28.62% | 22.55% | 21.65% |

At 1 spp, DSHARC reduces RMSE 23.3% versus PT and 4.4% versus SHARC. Luminance
L1 / reference mean is 82.35% / 55.06% / 47.91%, respectively. At 32 spp, DSHARC
reduces RMSE 24.4% versus PT and 4.0% versus SHARC. First-bounce directional
variance remains visible in both caches. This is one reproducible seed, not a
multi-seed confidence interval.

Long-convergence comparison: 1024 frames × 32 spp, cold start, 32-bounce reference.

| DSHARC vs reference | Result |
| --- | ---: |
| Mean luminance signed difference | +0.0111% |
| Dark-half mean difference | +0.0854% |
| 8×8 block L1 / reference mean | 0.1030% |
| Local block relative error p95 | 0.6118% |
| Local block relative error maximum | 1.2880% |

At the last frame, 94.81% of paths requested a cell, 252922 cells were active,
and there were no failed allocations. Two retained representatives failed
material validation and one rendering sample used Gather's PT fallback. Missing
dependencies also continued tracing. No black-tail approximation is used.
At 8192 / 16384 / 32768 spp, mean luminance was
0.02825237 / 0.02825249 / 0.02825172, versus final reference 0.02824858.

These measurements do not prove unbiasedness. Finite cells, representative
selection, changing materials within cells, normal bins, finite depth and temporal
feedback retain approximation error. Thin geometry and high-albedo scenes need
their own validation; all present materials are diffuse.

## Work and timing: not an equal-budget win

At warmed 1 spp, DSHARC updates about **206000 active cells per frame**, including
retained cells. SHARC launches only **6144 sparse update paths**. DSHARC averages
`samples_per_frame` lighting samples per active cell so increasing image spp
also increases update quality/work. Equal warmup frames or image spp is not equal
work; no equal-time image-quality advantage is claimed.

`benchmark_caches.py` measures three 128-frame batches after 256 warmup frames,
480×320, 1 spp/frame. Includes Python submission and GPU work, excludes compilation,
UI and image readback, waits every 32 frames. This is end-to-end batch throughput,
not isolated GPU pass timing:

| Mode | Mean ms/frame |
| --- | ---: |
| Reference PT | 0.220 |
| SHARC | 0.345 |
| DSHARC | 0.810 |

The current DSHARC integration is about 2.35× the SHARC frame cost here, for a
modest lower image error. It is a correctness/quality baseline. Update budgets,
retained-cell refresh rate, larger grids, material side buffers and compaction
cost are possible optimization targets, each requiring another bias/cost check.

## Files and reproduction

- `dsharc.py`: reflected resources, settings, indirect dispatch, buffer swap, stats.
- `shaders/dsharc_bridge.slangh`: resource bindings and 32-byte path records.
- `shaders/dsharc_request.slang`: prefix tracing and demand requests.
- `shaders/dsharc_manage.slang`: Begin, Compact, indirect arguments and Resolve.
- `shaders/dsharc_update.slang`: representative probe and frozen-feedback lighting.
- `shaders/dsharc_gather.slang`: completion and tracing fallback.
- `shaders/path_tracer.slang`: shared transport and solar MIS, with compile-time
  `DSHARC_REQUEST`, `DSHARC_FEEDBACK`, `PATH_LIBRARY` permutations.
- `renderer.py`, `app.py`, `entry_point.py`: mode selection, reset and CLI/UI.
- `test_dsharc.py`, `compare_sharc.py`, `compare_low_spp.py`, `benchmark_caches.py`:
  transport checks, convergence exports, low-spp comparison and timing.

```powershell
./sample/.venv/Scripts/python.exe sample/entry_point.py --renderer dsharc
./sample/.venv/Scripts/python.exe sample/test_dsharc.py --backend d3d12
./sample/.venv/Scripts/python.exe sample/test_dsharc.py --backend vulkan
./sample/.venv/Scripts/python.exe sample/compare_sharc.py --cache dsharc `
  --width 480 --height 320 --frames 1024 --spp 32 --output sample/output/dsharc_convergence
./sample/.venv/Scripts/python.exe sample/compare_low_spp.py --dsharc `
  --reference sample/output/dsharc_convergence/reference.npy --output sample/output/dsharc_low_spp
./sample/.venv/Scripts/python.exe sample/benchmark_caches.py
```

Four new GPU tests pass on D3D12 and Vulkan: RGB furnace energy/feedback, room
bias/full-hash fallback, empty indirect dispatch, and invalidation/batch/resize/
mode switching. Original PT and SHARC D3D12 tests also pass after transport
refactoring. A DSHARC three-frame window startup/presentation test passes.
PNG and HDR `.npy` outputs plus JSON metrics are in the selected output folders.
