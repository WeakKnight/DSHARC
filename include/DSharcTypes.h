#ifndef DSHARC_TYPES_H
#define DSHARC_TYPES_H

// DSHARC: demand-driven, diffuse world-space radiance caching.
// HLSL / DXC, SM 6.6, or Slang / Metal 4. Positions use stable world space.
// Metal stores each exact key as two words and publishes it via a slot state.
#ifndef DSHARC_SPLIT_KEY_ATOMICS
#if defined(__METAL__) || defined(__TARGET_METAL__)
#define DSHARC_SPLIT_KEY_ATOMICS 1
#else
#define DSHARC_SPLIT_KEY_ATOMICS 0
#endif
#endif

static const uint DSHARC_INVALID_INDEX = 0xffffffffu;
static const uint DSHARC_FLAG_REQUESTED = 1u;
static const uint DSHARC_FLAG_VALID = 2u;
static const float DSHARC_PI = 3.14159265358979323846f;

#ifndef DSHARC_PROBE_COUNT
#define DSHARC_PROBE_COUNT 16
#endif
#if DSHARC_PROBE_COUNT < 1
#error DSHARC_PROBE_COUNT must be positive
#endif

struct DSharcGridParameters
{
    float3 cameraPosition;
    float baseCellSize;       // > 0, in world units
    float3 origin;            // fixed grid origin; changing it requires a reset
    float levelDistance;      // > 0; level 1 starts at 2 * this distance
    uint maxLevel;            // 0..31, cell size = baseCellSize * 2^level
};

// StructuredBuffer stride: 32 bytes. Normal must be a geometric unit normal.
struct DSharcSurface
{
    float3 position;
    float padding0;
    float3 normal;
    float padding1;
};

// StructuredBuffer stride: 16 bytes.
struct DSharcEntryState
{
    uint lastRequestedFrame;
    uint accumulatedFrames;
    uint flags;
    uint reserved;
};

// StructuredBuffer stride: 32 bytes. A fully rough / diffuse approximation.
// The renderer is responsible for deriving this from its material system.
struct DSharcMaterial
{
    float3 diffuseAlbedo;
    uint valid;
    float3 emissive;
    uint reserved;
};

// Resource-bearing shader object, NOT a host-uploadable constant-buffer layout.
// Bind all resources in every entry point that uses them. Radiance buffers must
// be distinct allocations/views of nonoverlapping ranges; swap roles per frame.
struct DSharcParameters
{
    DSharcGridParameters grid;
    uint capacity;            // same number of entries in every per-entry buffer
    uint frameIndex;          // advances once for each complete cache frame
    uint staleFrameCount;     // suggested: 16; < 2^31
    uint maxAccumulatedFrames;// suggested: 32; clamped to at least 1

#if DSHARC_SPLIT_KEY_ATOMICS
    RWStructuredBuffer<uint2> keys;
    RWStructuredBuffer<uint> keyStates; // 0 empty, 1 writing, 3 ready
#else
    RWStructuredBuffer<uint64_t> keys;
#endif
    RWStructuredBuffer<DSharcEntryState> states;
    RWStructuredBuffer<DSharcSurface> surfaces;
    RWStructuredBuffer<DSharcMaterial> materials;
    RWStructuredBuffer<float4> previousRadiance; // RGB radiance, W validity
    RWStructuredBuffer<float4> currentRadiance;  // RGB radiance, W validity
    RWStructuredBuffer<float4> estimates;        // RGB estimate, W validity
    RWStructuredBuffer<uint> activeEntries;
    RWStructuredBuffer<uint> activeCount;        // exactly one uint
};

#endif
