// Compile coverage for every public operation. This is a renderer adapter
// fixture, not a scene tracer. See docs/Integration.md for lighting contracts.
#include "DSharcCommon.h"

cbuffer Constants : register(b0)
{
    float3 CameraPosition;
    float BaseCellSize;
    float3 GridOrigin;
    float LevelDistance;
    uint MaxLevel;
    uint Capacity;
    uint FrameIndex;
    uint StaleFrames;
    uint AccumulatedFrames;
    uint DispatchGroupsX;
    uint RequestCount;
    uint Padding;
};

RWStructuredBuffer<uint64_t> Keys : register(u0);
RWStructuredBuffer<DSharcEntryState> States : register(u1);
RWStructuredBuffer<DSharcSurface> Surfaces : register(u2);
RWStructuredBuffer<DSharcMaterial> Materials : register(u3);
RWStructuredBuffer<float4> Previous : register(u4);
RWStructuredBuffer<float4> Current : register(u5);
RWStructuredBuffer<float4> Estimates : register(u6);
RWStructuredBuffer<uint> Active : register(u7);
RWStructuredBuffer<uint> Count : register(u8);
RWByteAddressBuffer Arguments : register(u9);
RWStructuredBuffer<uint> RequestIndices : register(u10);
RWStructuredBuffer<float4> Output : register(u11);
StructuredBuffer<DSharcSurface> InputSurfaces : register(t0);
StructuredBuffer<DSharcMaterial> InputMaterials : register(t1);
StructuredBuffer<float4> DirectIrradiance : register(t2);
StructuredBuffer<float4> IndirectIncidentRadiance : register(t3);

DSharcParameters GetCache()
{
    DSharcParameters p;
    p.grid.cameraPosition = CameraPosition;
    p.grid.baseCellSize = BaseCellSize;
    p.grid.origin = GridOrigin;
    p.grid.levelDistance = LevelDistance;
    p.grid.maxLevel = MaxLevel;
    p.capacity = Capacity;
    p.frameIndex = FrameIndex;
    p.staleFrameCount = StaleFrames;
    p.maxAccumulatedFrames = AccumulatedFrames;
    p.keys = Keys;
    p.states = States;
    p.surfaces = Surfaces;
    p.materials = Materials;
    p.previousRadiance = Previous;
    p.currentRadiance = Current;
    p.estimates = Estimates;
    p.activeEntries = Active;
    p.activeCount = Count;
    return p;
}

[numthreads(64, 1, 1)]
void ClearCS(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    DSharcClearEntry(GetCache(), DSharcLinearDispatchIndex(group, lane, DispatchGroupsX, 64u));
}

[numthreads(1, 1, 1)]
void ResetCountCS()
{
    DSharcResetActiveCount(GetCache());
}

[numthreads(64, 1, 1)]
void BeginCS(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    DSharcBeginFrameEntry(GetCache(), DSharcLinearDispatchIndex(group, lane, DispatchGroupsX, 64u));
}

[numthreads(64, 1, 1)]
void RequestCS(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    uint index = DSharcLinearDispatchIndex(group, lane, DispatchGroupsX, 64u);
    if (index < RequestCount)
        RequestIndices[index] = DSharcRequest(GetCache(), InputSurfaces[index]);
}

[numthreads(64, 1, 1)]
void CompactCS(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    DSharcCompactEntry(GetCache(), DSharcLinearDispatchIndex(group, lane, DispatchGroupsX, 64u));
}

[numthreads(1, 1, 1)]
void ArgumentsCS()
{
    DSharcWriteDispatchArguments(GetCache(), Arguments, 0u, 64u);
}

bool GetUpdateIndex(uint3 group, uint lane, out uint index)
{
    uint3 groups = DSharcGetDispatchGroups(Count[0], 64u);
    uint activeIndex = DSharcLinearDispatchIndex(group, lane, groups.x, 64u);
    return DSharcGetActiveEntry(GetCache(), activeIndex, index);
}

[numthreads(64, 1, 1)]
void MaterialCS(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    uint index;
    if (GetUpdateIndex(group, lane, index))
        DSharcStoreMaterial(GetCache(), index, InputMaterials[index]);
}

[numthreads(64, 1, 1)]
void LightingCS(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    uint index;
    if (!GetUpdateIndex(group, lane, index))
        return;
    float3 estimate = DSharcEvaluateDiffuseRadiance(Materials[index],
        DirectIrradiance[index].xyz, IndirectIncidentRadiance[index].xyz);
    DSharcStoreEstimate(GetCache(), index, estimate);
}

[numthreads(64, 1, 1)]
void ResolveCS(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    uint index;
    if (GetUpdateIndex(group, lane, index))
        DSharcResolveEntry(GetCache(), index);
}

[numthreads(64, 1, 1)]
void GatherCS(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    uint index = DSharcLinearDispatchIndex(group, lane, DispatchGroupsX, 64u);
    if (index >= RequestCount)
        return;
    float3 radiance;
    bool found = DSharcGetCurrentRadiance(GetCache(), RequestIndices[index], radiance);
    Output[index] = float4(radiance, found ? 1.0f : 0.0f);
}

[numthreads(64, 1, 1)]
void LookupCS(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    uint index = DSharcLinearDispatchIndex(group, lane, DispatchGroupsX, 64u);
    if (index >= RequestCount)
        return;
    float3 radiance;
    bool found = DSharcLookupPrevious(GetCache(), InputSurfaces[index], radiance);
    Output[index] = float4(radiance, found ? 1.0f : 0.0f);
}

[numthreads(64, 1, 1)]
void SamplingCS(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    uint index = DSharcLinearDispatchIndex(group, lane, DispatchGroupsX, 64u);
    if (index >= RequestCount)
        return;
    DSharcSurface surface = InputSurfaces[index];
    uint level = DSharcGetLevel(GetCache().grid, surface.position);
    float size = DSharcGetCellSize(GetCache().grid, level);
    bool shouldRequest = DSharcShouldRequest(DirectIrradiance[index].w,
        IndirectIncidentRadiance[index].w, size, false);
    float3 direction = DSharcSampleCosineHemisphere(surface.normal,
        frac(DirectIrradiance[index].xy));
    Output[index] = float4(direction, shouldRequest ? 1.0f : 0.0f);
}

// Runtime test diagnostics: preserve exact key/hash bits in the output buffer.
[numthreads(64, 1, 1)]
void KeyCS(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    uint index = DSharcLinearDispatchIndex(group, lane, DispatchGroupsX, 64u);
    if (index >= RequestCount)
        return;
    uint64_t key;
    bool valid = DSharcMakeKey(GetCache().grid, InputSurfaces[index], key);
    uint bucket = Capacity > 0u ? DSharcHashKey(key) % Capacity : 0u;
    Output[index] = asfloat(uint4(uint(key), uint(key >> 32), bucket, valid ? 1u : 0u));
}
