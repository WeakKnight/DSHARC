# DSHARC Integration Guide

## Overview

DSHARC moves expensive diffuse-tail shading from individual rendering paths to
a shared set of world-space cache entries. Its high-level organization is:

1. Trace rendering paths until their footprints make caching appropriate.
2. Request an entry at each termination surface and save its index per path.
3. Compact the live hash entries and shade each representative surface once.
4. Resolve the estimates against frozen previous-frame radiance.
5. Finish each rendering path with its throughput multiplied by current radiance.

This is a renderer-independent HLSL library. It does not trace the scene, evaluate
materials, sample light sources, manage GPU memory or submit dispatches.

Unlike SHARC's usual sparse update paths and signal backpropagation, DSHARC
collects requests from the rendering paths themselves and updates unique entries.
Like LumenPT World Cache, it uses the first requester as a representative and can
feed cached indirect light back into later updates. Unlike the inspected LumenPT
shader's in-place lighting access, DSHARC explicitly separates previous/current
radiance to make this feedback a frame-frozen iteration.

## Compilation and device requirements

Add `include/` to your shader include path and include `DSharcCommon.h`. Compile
with DXC using `-T cs_6_6` for compute shaders, or the appropriate SM 6.6 ray tracing
library target. `-HV 2021 -Ges -WX` is used by the compile fixture. No FP16 flags
are required. There is no dependency on the reference snapshots.

The Request pass uses `InterlockedCompareExchange` on a
`RWStructuredBuffer<uint64_t>`. Check SM 6.6 and support for buffer integer-64
atomics on D3D12. For HLSL-to-SPIR-V, use a SPIR-V-enabled DXC with
`-spirv -fspv-target-env=vulkan1.2`, and enable `shaderInt64` and
`shaderBufferInt64Atomics` on the device. Match descriptor bindings and structured
strides to shader reflection. There is no 32-bit-atomic fallback or GLSL API.

`DSHARC_PROBE_COUNT` defaults to 16 and must be identical across all shaders
accessing a cache. It is a bounded linear-search window, not a maximum load factor.
Insertion can fail when this local window is full even when other slots are free.

## Resource allocation

For capacity `N`, allocate the following structured buffers with UAV/storage
usage. Every per-entry buffer, including the active list, has `N` elements.

| `DSharcParameters` member | Element type | Stride | Purpose |
| --- | --- | ---: | --- |
| `keys` | `uint64_t` | 8 bytes | Packed cell identity; zero means empty |
| `states` | `DSharcEntryState` | 16 bytes | Last demand frame, history count, flags |
| `surfaces` | `DSharcSurface` | 32 bytes | Representative position and normal |
| `materials` | `DSharcMaterial` | 32 bytes | Diffuse approximation, emissive, validity |
| `previousRadiance` | `float4` | 16 bytes | Frozen RGB history and validity in W |
| `currentRadiance` | `float4` | 16 bytes | Resolved RGB result and validity in W |
| `estimates` | `float4` | 16 bytes | This frame's RGB sample and validity in W |
| `activeEntries` | `uint` | 4 bytes | Compacted live slot indices |
| `activeCount` | `uint` | 4 bytes | One element, not `N` |

The baseline uses **140 bytes per capacity slot**, plus the 4-byte counter,
indirect arguments and renderer-owned path records. `N = 2^18` costs 35 MiB;
`2^20` costs 140 MiB. The explicit FP32 storage favors integration clarity over
packing efficiency. These sizes are not a performance recommendation.

Allocate at least 12 additional bytes for each indirect dispatch argument block.
Use an aligned byte offset, UAV/storage writes, and indirect-argument read usage.
Store path entry indices separately from path throughput and accumulated radiance.

`DSharcParameters` contains shader resource objects. Construct it in shader code;
do not upload it as a binary C++ constant-buffer structure. The three data structs
in `DSharcTypes.h` have explicit 16/32-byte layouts. The compile fixture shows
concrete bindings and a separately packed constant buffer.

## Parameter setup

Initialize every field. For a scene measured in meters, a starting configuration
for experimentation is:

```hlsl
DSharcParameters cache;
cache.grid.cameraPosition = cameraPosition;
cache.grid.origin = fixedGridOrigin;
cache.grid.baseCellSize = 0.25f;
cache.grid.levelDistance = 16.0f;
cache.grid.maxLevel = 12u;
cache.capacity = 1u << 18;
cache.frameIndex = cacheFrameIndex;
cache.staleFrameCount = 16u;
cache.maxAccumulatedFrames = 32u;
// Assign all nine resource members from your shader bindings.
```

Use the same parameter values for all passes in a frame. Keep capacity within
device buffer limits; the supplied dispatch convention supports capacities up to
`2^24`. A capacity of zero disables allocation/query, but the bound counter and
other shader resources must still satisfy your API's binding requirements.

The level is `clamp(floor(log2(max(distance / levelDistance, 1))), 0, maxLevel)`;
the cell size is `baseCellSize * 2^level`. Level 1 therefore begins at distance
`2 * levelDistance`. `maxLevel` is capped at 31. Choose finite positive sizes.

Each key exactly packs signed 18-bit XYZ cell coordinates, a 5-bit level and a
six-way normal bin, plus an occupied bit. Coordinates are measured from
`grid.origin` and must be in `[-131072, 131071]` at the selected level. Out-of-range
requests fail rather than wrap into another location. Hash collisions are handled
by comparing full keys. Different surfaces inside the same cell/normal bin still
intentionally share radiance.

Keep positions and camera in the same stable world coordinate system. Do not send
camera-relative positions that change every frame. For large worlds, use a fixed
local origin with adequate FP32 precision. Changing that coordinate system or the
grid origin/scale/level rules requires a complete cache reset. Camera motion alone
is allowed, but changing levels can cause cold entries; adjacent-level history
transfer is not implemented.

## Frame ordering and synchronization

```text
Initialization/reset: Clear all entries + ResetActiveCount → barrier

Frame:
  BeginFrame over capacity + ResetActiveCount
    → barrier
  Request from rendering paths
    → barrier
  Compact over capacity
    → barrier
  WriteDispatchArguments
    → argument transition/barrier
  SampleMaterial over active entries
    → barrier
  Lighting over active entries (reads previousRadiance)
    → barrier
  Resolve over active entries (writes currentRadiance)
    → barrier
  Gather into rendering paths (reads currentRadiance)
    → finish all consumers; swap previous/current for the next frame
```

Use separate dispatches for these phases. D3D12 requires appropriate UAV barriers
and state transitions; Vulkan requires shader-write to shader-read/write storage
dependencies and shader-write to indirect-command-read dependencies. Cross-queue
execution additionally requires queue synchronization. Keep the render graph
informed about every read/write, including the counter and path-index buffers.

`previousRadiance` and `currentRadiance` must never alias. Serialize cache frames
or provide independent cache instances: swapping descriptors on the CPU does not
make overlapping GPU frames safe. Do not reuse an entry index across cache frames.

`ClearEntry`, `BeginFrameEntry` and `CompactEntry` are called once for each capacity
slot. `ResetActiveCount` runs on exactly one thread, before Compact, not inside
the Compact dispatch. ResetCount and Begin may execute independently because they
write different resources, but both must finish before their consumers.

### Initialization and BeginFrame

Call `DSharcClearEntry(cache, slot)` over the entire capacity on creation and on
scene reset. Call `DSharcResetActiveCount(cache)` as well. Active-list contents do
not need clearing because its count defines the valid range.

On subsequent frames, `DSharcBeginFrameEntry` clears request marks, this frame's
materials/estimates/current results, and removes entries whose
`frameIndex - lastRequestedFrame > staleFrameCount`. Removed slots lose all history
before Request can reuse them. Keep `staleFrameCount < 2^31`; advance `frameIndex`
once per complete cache frame. Do not skip more than `2^31` cache frames without a
reset. Normal unsigned frame-counter wrap is supported by subtraction.

## Rendering paths: Request

At an eligible indirect hit, create a `DSharcSurface` with its position and valid
geometric normal. Optionally use `DSharcShouldRequest(hitDistance, pathRoughness,
cellSize, lastBounce)`. It accepts the last bounce, or a ray whose footprint
exceeds the cell size and whose length exceeds the cell diagonal.

```hlsl
uint entry = DSharcRequest(cache, hitSurface);
if (entry != DSHARC_INVALID_INDEX)
{
    // Save entry, current path throughput, and accumulated path radiance.
    // The throughput includes the segment that arrived at hitSurface.
    // End this path prefix; finish it after Resolve.
}
else
{
    // Continue tracing or queue an explicit fallback, even at the bounce limit.
}
```

The first atomic requester in the frame supplies the representative position and
normal. Other requesters only retain the entry index. This is neither the cell
center nor a uniform/reservoir sample of the surface. The representative can change
next frame while its radiance history persists.

An index returned by Request is an allocation result, not proof of valid lighting.
Do not read its payload or use it to terminate with radiance inside Request. Key
publication and payload writes are separated by the Request-to-update barrier.
Stop adding requests before Compact; cache-lighting rays must not allocate slots
concurrently with update/resolve.

Do not also add direct lighting or emission at the termination vertex: the cache
contains both. Contributions at earlier path vertices remain in path radiance.
Preserve enough continuation state for a later Gather failure if your renderer
requires a tracing fallback instead of accepting a missing tail.

## Compact and dispatch

`DSharcCompactEntry` appends every occupied entry, including retained entries not
requested this frame. Thus expensive work scales with the live working set, while
Begin and Compact still scan capacity. Cache-to-cache lookups do not refresh age.

After compaction, call `DSharcWriteDispatchArguments(cache, args, offset, 64)` once
for a compute update with 64 threads per group. It emits up to 65535 groups in X
and uses Y for larger counts. For an empty list it dispatches one guarded group.
Use the same linearization in Material, Lighting and Resolve:

```hlsl
uint3 groups = DSharcGetDispatchGroups(cache.activeCount[0], 64u);
uint activeIndex = DSharcLinearDispatchIndex(groupID, groupThreadIndex, groups.x, 64u);
uint entry;
if (!DSharcGetActiveEntry(cache, activeIndex, entry))
    return;
```

These are compute dispatch arguments, not D3D12 `DispatchRays` arguments. If your
material/lighting uses ray-generation shaders, construct your API's ray dispatch
description and use its ray index as `activeIndex` instead.

## Material sampling

Read `cache.surfaces[entry]` and obtain its material. As in LumenPT, a short ray from
`position + normal * bias` along `-normal` can recover material attributes without
storing a geometry-specific hit record. Choose the bias in scene units, use the
renderer's robust ray offset, and validate distance, orientation and surface
identity as far as your geometry system permits. A short ray can otherwise hit a
neighboring sheet or the wrong side of a thin surface.

Alternatively retain renderer-specific primitive/barycentric data in side buffers
using an appropriately extended request protocol. DSHARC itself stores no such ID.

Derive an effective diffuse/fully rough albedo and emissive value. LumenPT folds
its fully rough specular response and some subsurface colors into that effective
albedo; this renderer-specific conversion is not performed by DSHARC.

Call `DSharcStoreMaterial(cache, entry, material)` once for every active entry,
including on failure with `material.valid = 0`. The function clamps albedo to
`[0,1]`, emission to nonnegative values and rejects nonfinite inputs. Missing
geometry invalidates the result instead of silently reusing old current lighting.

## Lighting update

Run once per active entry after Material has completed. Generate random samples
using at least entry index, frame index and sample dimension; the library does not
provide an RNG. Use the actual representative point, not the cell center.

1. Estimate direct irradiance using the renderer's light sampler. Include surface
   cosine, visibility, light-selection probability and sample-PDF correction.
   Do not include this surface's albedo or its Lambertian `1/PI` factor.
2. Generate a cosine-weighted direction with `DSharcSampleCosineHemisphere` and
   trace from a robustly offset representative position.
3. On a sky miss, use sky radiance. On a valid surface hit, call
   `DSharcLookupPrevious(cache, hitSurface, incomingRadiance)`.
4. If that lookup fails, choose an explicit missing-dependency policy below.
5. Evaluate and store the radiance estimate:

```hlsl
float3 estimate = DSharcEvaluateDiffuseRadiance(
    cache.materials[entry], directIrradiance, indirectIncidentRadiance);
DSharcStoreEstimate(cache, entry, estimate);
```

The helper computes

```text
estimate = emissive + diffuseAlbedo * (directIrradiance / PI + indirectIncidentRadiance)
```

The indirect input is the mean incident radiance over cosine-weighted samples.
Its cosine/PDF cancels the diffuse `1/PI`; do not divide this term by PI again.
For other direction samplers, apply the corresponding cosine/PDF weights yourself.
Average multiple samples locally before StoreEstimate: concurrent writes to one
entry are not supported. Material and Lighting may be fused into one invocation
per entry with suitable bindings; Resolve must remain a separate, ordered pass.

The cache contains total diffuse-approximate outgoing radiance, including emission
and direct light, not irradiance. Keep all inputs/history in the same linear,
preferably unexposed, units. Apply camera exposure only at final composition. If
using changing pre-exposure, rescale history correctly or reset the cache.

Avoid double-counting sky or emissive lighting between direct sampling and the
cosine estimator. The simplest reference setup uses direct sampling for analytic
lights and uses the cosine ray for sky/emissive geometry. If those sources are
also sampled by NEE, the renderer must provide a compatible MIS/signal split.

### Missing indirect dependencies

`DSharcLookupPrevious` neither inserts nor marks the hit entry. It returns false
and zero when no valid history exists. Select and document a renderer policy:

- **LumenPT-like baseline:** accept zero. It is cheap but can darken uncovered
  transport, including dependencies that never receive rendering-path demand.
- **Trace fallback:** estimate the missing tail with the path tracer. This costs
  more, especially during warm-up, and avoids simply dropping that contribution.
- **Dependency expansion:** collect secondary requests into a separate queue and
  process them in an additional, correctly ordered request/update cycle. This is
  an extension, not an operation performed by these headers.

Never interpret a successful black sample as a cache miss: the boolean return
value distinguishes validity from radiance. An indirect read also does not keep
an otherwise unused entry alive.

## Resolve and Gather

Call `DSharcResolveEntry` once per active entry after all estimates are written.
It uses alpha `1 / accumulatedFrames`, caps the frame count at
`maxAccumulatedFrames`, then continues as an exponential moving average at that
cap. Invalid estimates invalidate current data and reset its accumulation count.
A valid first estimate bypasses history. The previous buffer is never modified
during Lighting or Resolve.

After a barrier, finish each suspended rendering path:

```hlsl
float3 tailRadiance;
if (DSharcGetCurrentRadiance(cache, savedEntry, tailRadiance))
    pathRadiance += savedThroughput * tailRadiance;
else
    /* run the renderer's saved-continuation fallback, or explicitly accept zero */;
```

Preserve any earlier path contributions on failure. Do not multiply by the cache
surface's albedo a second time. Apply pixel classification, exposure and denoising
in the renderer. Consume all saved entry indices before the next BeginFrame, then
swap previous/current radiance bindings. No copy is needed for the swap.

## Tuning and limitations

- Smaller cells preserve more detail but reduce sharing, increase occupancy and
  can increase local probe failures. Track failed requests, not just occupancy.
- Larger `maxAccumulatedFrames` reduces noise but increases response lag. Feedback
  already spreads multi-bounce changes over iterations, even without accumulation.
- Larger `staleFrameCount` stabilizes the working set but increases per-frame
  material and lighting work. Aging is measured from requests, not updates.
- Representative changes inside a cell do not reset history. Thin geometry,
  changing albedo, disocclusion and moving surfaces can therefore cause smearing
  or stale lighting. Invalidate the cache on major scene/lighting changes as needed.
- Six normal bins separate some opposing surfaces but do not encode outgoing view
  direction. Glossy/specular tails, fine geometry and material details remain
  approximations. Keep sharp path segments outside the cache when possible.
- The fixed coordinate range, FP32 positions and missing adjacent-level blending
  are explicit limits. Do not silently change grid parameters to solve overflow.

## Validation checklist

The executable compile fixture is `tests/Compile.hlsl`; run `tests/Compile.ps1`
with DXC. Optional `-Spirv` also compiles all entry points to SPIR-V. The fixture
uses supplied material/lighting buffers so every public operation can be compiled
without a renderer. It is not a scene-integrated ray tracing example.

`tests/Runtime.ps1` additionally builds `tests/Runtime.cpp` with MSVC and executes
the compiled DXIL on the default D3D12 SM 6.6 device. Run it from an x64 Visual
Studio developer PowerShell with the Windows SDK available. It checks concurrent
same-cell requests, coordinate bounds, normal separation, frozen feedback,
temporal averaging, invalid material/estimate handling, retained-entry updates,
deletion holes, full-table failures and history reset on slot reuse. Artifacts go
to a unique temporary directory. Vulkan execution, large dispatches and rendering
quality still require validation in the integrating renderer.

For runtime integration, verify these cases under your graphics API validation:

1. Many same-cell requests yield one active entry and identical gathered radiance.
2. Opposite normals and different levels produce different keys/entries.
3. A full probe window returns `DSHARC_INVALID_INDEX` without overwriting entries.
4. Deleting a colliding entry does not hide or duplicate a surviving key past the hole.
5. A new/reused slot cannot read the evicted entry's radiance history.
6. Empty active lists safely dispatch guarded work; large lists use correct 2D indexing.
7. Missing materials and nonfinite estimates return invalid current results.
8. Indirect queries read only previous radiance; pass order cannot affect feedback.
9. Constant lighting converges correctly; light steps and camera cuts expose expected latency.
10. High-gloss, thin-geometry and emissive scenes are compared against an uncached reference.

Measure Request/Begin/Compact costs as well as the saved path shading. Compare
performance at similar image quality; sharing alone does not establish a speedup.
