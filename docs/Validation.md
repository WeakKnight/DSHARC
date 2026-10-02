# Validation

## Standalone shader headers

Compile every public operation with warnings treated as errors:

```powershell
./tests/Compile.ps1 -Dxc 'path/to/dxc.exe'
./tests/Compile.ps1 -Dxc 'path/to/spirv-enabled/dxc.exe' -Spirv
```

Run the D3D12 compute tests from an x64 Visual Studio developer PowerShell:

```powershell
./tests/Runtime.ps1 -Dxc 'path/to/dxc.exe'
```

These execute shaders on the default D3D12 SM 6.6 device, checking concurrent
deduplication, frozen feedback, temporal averaging, eviction, collision holes,
full-table failures and invalid inputs. Build artifacts go to a temporary directory.
`tests/Compile.hlsl` is an API fixture, not a scene-rendering benchmark.

## Path tracer sample

After [installing the sample dependencies](../sample/README.md):

```powershell
./sample/.venv/Scripts/python.exe sample/test_sample.py --backend d3d12
./sample/.venv/Scripts/python.exe sample/test_sharc.py --backend d3d12
./sample/.venv/Scripts/python.exe sample/test_dsharc.py --backend d3d12
# Repeat with --backend vulkan.
```

These render actual GPU scenes: analytic diffuse/emissive energy, solar MIS,
shadows, cache coverage, full-table tracing fallback, empty indirect dispatch,
invalidation, mode switching, resizing and HDR export.

For the [local macOS build](../sample/README.md#local-source-builds-on-macos):

```bash
PYTHONPATH=../slangpy sample/.venv/bin/python sample/test_dsharc_hash.py --backend metal
PYTHONPATH=../slangpy sample/.venv/bin/python sample/test_dsharc.py --backend metal
```

The hash tests execute the actual shared headers on the GPU and check concurrent
deduplication, preservation of high key bits, lookup/insertion past deletion holes,
busy-slot fallback, and failure without overwrite when the probe window is full.

For image comparisons and timing:

```powershell
./sample/.venv/Scripts/python.exe sample/compare_sharc.py --cache dsharc `
  --width 480 --height 320 --frames 1024 --spp 32 --output sample/output/dsharc_convergence
./sample/.venv/Scripts/python.exe sample/compare_low_spp.py --dsharc `
  --reference sample/output/dsharc_convergence/reference.npy --output sample/output/dsharc_low_spp
./sample/.venv/Scripts/python.exe sample/benchmark_caches.py
```

See [measured results and limitations](../sample/DSHARC_INTEGRATION.md).
Generated HDR arrays, captures and metrics stay in ignored `sample/output/`.
