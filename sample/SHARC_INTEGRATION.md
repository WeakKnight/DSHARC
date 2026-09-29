# NVIDIA SHARC integration

The sample uses the local SHARC 1.8.3.0 SDK without changing its headers. The
application defaults to SHARC; `--renderer reference` compiles/runs the uncached
path and does not allocate cache resources. The Python `PathTracer` constructor
keeps `mode="reference"` for existing library callers/tests. The separate
`--renderer dsharc` mode is documented in [DSharc integration](DSHARC_INTEGRATION.md).

## Passes and storage

`sharc.py` owns the SDK resources and three compute pipelines:

1. **Update:** `SHARC_UPDATE=1`, `SHARC_QUERY=0`; one jittered camera path per
   5×5 pixel stratum per requested sample, including partial edge tiles. Its RNG
   stream is independent of the final image's. All strata cover the full view.
2. Explicit global UAV barrier.
3. **Resolve:** both flags zero; one thread per entry calls `SharcResolveEntry`.
   History defaults to 256 frames; stale entries expire after 1024 unused frames.
4. Explicit global UAV barrier.
5. **Query:** `SHARC_UPDATE=0`, `SHARC_QUERY=1`; full-resolution paths terminate
   on a valid cached indirect hit. Then normal FP32 image accumulation/tone mapping.

Buffers use reflected SDK element layouts: 8-byte keys, 16-byte accumulation,
16-byte resolved values. All have the same capacity and are zero-cleared before
use. The default 2^22 entries require 160 MiB total, excluding image/scene buffers.
Native Slang `float16_t4` preserves the SDK's half storage on DXIL and SPIR-V;
allocation checks catch an unexpected stride. Hardware must also support 64-bit
buffer atomics. Initialization errors are printed and select reference mode.

Camera, lighting, scene-object, mode, bounce-cap, seed, grid/settings changes,
resize and explicit reset clear both cache and image. This correctness-first
version deliberately rebuilds on camera movement; it does not yet preserve world
cache history during navigation. Exposure preserves both histories. Changing
samples per frame preserves accumulated sample weights and continues the RNG
sample sequence.

## Energy contract

- Every material is diffuse. `SHARC_MATERIAL_DEMODULATION=1` caches radiance
  divided by `max(albedo, .001)`, then reconstructs with the queried material.
  `SHARC_SEPARATE_EMISSIVE=1` keeps local emission out of that division.
- Update submits **local** solar NEE to `SharcUpdateHit`, not camera-weighted
  accumulated radiance. `SharcSetThroughput` receives each segment's albedo and
  roulette compensation, not the accumulated camera throughput.
- Misses submit sky plus the MIS-weighted solar disk. Sky has no NEE competitor;
  it is not downweighted by solar MIS. Query occurs before local direct/emissive
  addition because the returned cache value already includes both.
- The cache only takes over after a diffuse continuation, with segment length
  greater than the cell diagonal. Camera hits always shade normally. Entries need
  more than eight effective samples; misses/immature entries continue tracing.
- `SHARC_ENABLE_CACHE_RESAMPLING=1` enables SDK update-path early termination.
  Each hit supplies a random value selecting a resampling depth in [1,4]. A mature
  cache at an eligible depth replaces the tail and the SDK propagates it to the
  earlier roots. On success we stop immediately, without adding direct light again.
  `SHARC_PROPAGATION_DEPTH=4` records the first four roots. When resampling is not
  available (especially cold start), update traces the full tail up to 32 events.
  Later contributions are propagated to those roots with `SharcUpdateMiss` without
  inserting new roots or increasing their sample count. This SDK helper accepts
  any already-evaluated tail radiance, including a later vertex's direct light.
  A four-root sliding window without resampling would silently discard long
  tails and darken the cache; this implementation does not slide the window.
- Failed insertion still propagates that surface's radiance to earlier roots and
  continues the tail. It does not turn hash-table pressure into lost path energy.
  The SDK returns false for both insertion failure and successful resampling;
  we distinguish them by whether pathLength advanced. This is valid here because
  no new root is inserted after the four-root state fills.

The grid uses geometry-normal sign bins, logarithm base 2, level bias 0 and scale
80. For example, levels 1/2/3 have 0.025/0.05/0.1 world-unit cells. Larger scale
means smaller cells, more memory pressure and longer warm-up. Integer accumulation
scale 65536 reduces dark-room rounding loss. Very bright/high-throughput scenes
may need a lower `--sharc-radiance-scale`: SDK sums are uint32 and are not
overflow-proof for arbitrary radiance or samples per cell. There is no firefly
clamp, denoiser, exposure correction or reference-image blending hiding bias.

## Measured convergence: full-tail baseline

The following table records the original integration **before enabling cache
resampling**. Current enabled results and low-spp comparisons follow below.

Tested on Windows, NVIDIA GeForce RTX 5080, SlangPy 0.43.1, both D3D12 and Vulkan. Built-in diffuse room,
480×320, 1024 frames × 32 spp = **32,768 spp**, 32-bounce reference, default lights
and cache settings. Starts from an empty cache, without discarding warm-up frames.
Seeds: reference 1, independent reference 913, SHARC 271. Metrics use unexposed
linear HDR with luminance weights (0.2126, 0.7152, 0.0722).

| Measurement | D3D12 SHARC vs reference | Independent reference vs reference |
| --- | ---: | ---: |
| Mean luminance signed difference | −0.0416% | +0.0019% |
| Pixel luminance L1 / reference mean | 0.823% | 0.917% |
| 8×8 block luminance L1 / reference mean | 0.178% | 0.114% |
| 8×8 block luminance RMSE / reference mean | 0.339% | 0.155% |

SHARC's dark half of the image has +0.183% mean difference. The 95th percentile
of local 8×8 block relative error is 0.844%, with a 2.523% maximum (denominator
floored at .001). This checks dark/local regions as well as the global mean.
Vulkan agrees: mean difference −0.0417%, block L1 0.178%.

At the last frame, **94.79% of image paths terminate through SHARC**, 99.97% of
eligible lookups hit, and query paths average 1.986 intersection segments.
286,146 of 4,194,304 entries are occupied in the D3D12 run. These are actual
cache-use statistics, not a reference fallback result. They exclude update paths
and shadow rays and are not a speedup measurement.

SHARC mean luminance at 8192 / 16384 / 32768 spp was
0.02822397 / 0.02822868 / 0.02823682; the final reference was 0.02824858.
The independent reference estimates Monte Carlo noise, but these measurements
are not a proof of asymptotic unbiasedness. Spatial cells, coarse normal bins,
FP16 storage, fixed-point accumulation and finite temporal history retain bias.
The measured residual is small in this room; thin geometry, rapidly changing
lighting, very high albedo or different scene scales need their own validation.

The cache's 32-event tail can extend beyond the image path's remaining depth.
Use the default 32-bounce reference for comparisons. An intentionally truncated
one/eight-bounce PT can be darker for physical reasons unrelated to cache error.
Moving-camera reuse remains disabled; cache resampling is now enabled as below.

## Cache resampling enabled

With the same 480×320 / 32768-spp / 32-bounce D3D12 comparison, enabling SDK
resampling gives −0.1117% mean luminance difference, 0.1941% block L1 and 0.3974%
block RMSE (each normalized by reference mean). Dark-half mean difference is
+0.0713%, and local block error p95 is 0.8694%. The added feedback slightly
increases the measured bias, but it remains small in this diffuse room.

At the final frame, 85.95% of update paths terminate by resampling. Updates average
3.168 intersection segments; query paths average 1.986, with 94.77% terminating
through cache. Insertion failures are zero. These counters verify actual SDK
resampling, not just a compile-time define; they are not GPU timing results.
RGB furnace and room-bias tests pass on D3D12 and Vulkan with assertions that
resampling is actually used. Image-only reset is tested to retain cache state and
continue an unused RNG sequence.

## 32-spp cold versus warm cache

Run `compare_low_spp.py`: it uses 1 spp per frame and saves 1/4/8/16/32-spp HDR,
PNG and numerical checkpoints, plus a four-panel `comparison_32spp.png`.
Reference target is independent-seed uncached PT at 32768 spp. All panels use the
same camera and +1 EV; no denoiser is used. Default settings, 480×320, RTX 5080:

| 32-spp image | Luminance RMSE / reference mean | L1 / reference mean | Mean difference |
| --- | ---: | ---: | ---: |
| SHARC off | 28.62% | 20.44% | +0.053% |
| SHARC on, empty cache at frame 1 | 27.29% | 19.38% | +0.063% |
| SHARC on, 256 warmup frames then fresh image | 22.55% | 15.57% | −0.345% |

Cold-start RMSE improves by 4.7%; warm-cache RMSE improves by 21.2%. The warm
case has **256 extra 1-spp frames of cache history**: its image accumulation is
reset, but cache and RNG sequence are retained. It is not an equal-total-work
comparison. Cold and warm SHARC both also trace 6144 update roots per frame
(about 4% of image pixels); equal image spp is not equal runtime.

The normalized RMSE checkpoints show how the difference develops:

| Image spp | Off | On, cold | On, warm |
| --- | ---: | ---: | ---: |
| 1 | 161.66% | 161.66% | 129.58% |
| 4 | 81.44% | 81.44% | 63.93% |
| 8 | 57.56% | 57.46% | 45.35% |
| 16 | 40.58% | 40.04% | 32.05% |
| 32 | 28.62% | 27.29% | 22.55% |

Cold cache only reaches 55.45% query-path termination at frame 32; warm cache
reaches 93.22%. Earlier noisy frames still contribute to the cold accumulated
image. The first diffuse bounce remains sampled normally in both modes, so noise
from choosing bright window directions does not disappear merely by caching the
later path. The script uses one seed; this is a reproducible example rather than
a multi-seed confidence interval.

UI **Reset image (keep cache)** performs the warm-cache image reset. The existing
**Reset image + cache** button / R still resets everything; moving the camera also
invalidates both. Run the low-spp comparison from scratch:

```powershell
./sample/.venv/Scripts/python.exe sample/compare_low_spp.py
# Or reuse the matching high-spp default-room target to save work:
./sample/.venv/Scripts/python.exe sample/compare_low_spp.py `
  --reference sample/output/sharc_resampling_convergence/reference.npy
```

## Reproduce and inspect

```powershell
./sample/.venv/Scripts/python.exe sample/entry_point.py
# Click REFERENCE / SHARC in the panel to compare; R clears image + cache.
./sample/.venv/Scripts/python.exe sample/entry_point.py --renderer reference

./sample/.venv/Scripts/python.exe sample/test_sample.py --backend d3d12
./sample/.venv/Scripts/python.exe sample/test_sharc.py --backend d3d12
# Repeat both tests with --backend vulkan.

./sample/.venv/Scripts/python.exe sample/compare_sharc.py --width 480 --height 320 `
  --frames 1024 --spp 32 --output sample/output/sharc_comparison
# Repeat with --backend vulkan and a different output directory.
```

The comparison exports reference/independent-reference/SHARC `.npy` HDR arrays,
PNGs, `comparison.png` and `metrics.json`. Display images use identical +1 EV
exposure. Ten original CPU/GPU checks plus four SHARC GPU tests pass on each
backend, including an RGB furnace with analytic `L = E / (1-rho) = 1`, positive
cache coverage, room-error thresholds, zero-albedo transport and invalidation.
The local Slang compiler reports an SDK warning about an unset `out radiance`
on failed lookup; callers only read it when lookup returns true. SDK sources are
kept intact. DSharc has its own `test_dsharc.py` checks.
