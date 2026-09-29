# Design references and third-party code

DSHARC is an independent implementation of a demand-driven diffuse radiance
cache. It is not a drop-in replacement for SHARC's API.

- [LumenPT World Cache](https://github.com/EpicGames/UnrealEngine/tree/ue6-main/Engine/Shaders/Private/LumenPT),
  examined at `165dbead74a9f3c23fc3e4bc5904e8168cb5d8b5`: demand marking,
  first-request surface representatives and per-cell lighting updates.
  Unreal Engine source is **not included** in this repository.
- [NVIDIA SHARC](https://github.com/NVIDIA-RTX/SHARC), examined at
  `4e21b585c33c83d723ca9a1e11bbb1090d145793`: sparse-path radiance caching and
  the shader-header distribution style. The sample's SHARC comparison uses the
  local `sample/SHARC/` snapshot, whose [license](../sample/SHARC/License.md) and copyright
  notices remain in place. DSHARC's standalone headers do not depend on it.
- [WeakKnight/Nan](https://github.com/WeakKnight/Nan), examined at
  `97909ff6b9d43fb5da7b85f27a22b7793a78bddd`: the sample follows its Python/SlangPy
  organization and explicit compute passes with hardware ray queries.

## README image

`images/dsharc-comparison.png` contains unfiltered renders from this repository's
built-in diffuse room, using the comparison commands in [Validation](Validation.md).
Left: uncached PT, 1 spp. Center: DSHARC, 256 warmup frames followed by a fresh
1-spp image. Right: uncached PT, 32768 spp. All use +1 EV exposure. Warmup image
samples are discarded; only the cache is retained. No external scene assets or
AI-generated render results are used.
