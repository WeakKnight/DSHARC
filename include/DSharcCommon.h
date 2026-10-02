#ifndef DSHARC_COMMON_H
#define DSHARC_COMMON_H

#include "DSharcHashGrid.h"
#include "DSharcSampling.h"

// All entry points below require separate host-ordered dispatches and resource
// barriers as documented in docs/Integration.md. Atomics are NOT pass barriers.

void DSharcClearEntry(DSharcParameters cache, uint index)
{
    if (index >= cache.capacity)
        return;
    DSharcStoreKey(cache, index, uint64_t(0));
#if DSHARC_SPLIT_KEY_ATOMICS
    cache.keyStates[index] = 0u;
#endif
    cache.states[index] = (DSharcEntryState)0;
    cache.surfaces[index] = (DSharcSurface)0;
    cache.materials[index] = (DSharcMaterial)0;
    cache.previousRadiance[index] = float4(0, 0, 0, 0);
    cache.currentRadiance[index] = float4(0, 0, 0, 0);
    cache.estimates[index] = float4(0, 0, 0, 0);
}

// Dispatch once, separately from Compact. Also call on initialization.
void DSharcResetActiveCount(DSharcParameters cache)
{
    cache.activeCount[0] = 0u;
}

// Call once per capacity slot BEFORE Request. Aging measures render demand,
// not cache-to-cache reads. Old slots are fully reset before any reuse.
void DSharcBeginFrameEntry(DSharcParameters cache, uint index)
{
    if (index >= cache.capacity || DSharcLoadKey(cache, index) == uint64_t(0))
        return;
    DSharcEntryState state = cache.states[index];
    if ((cache.frameIndex - state.lastRequestedFrame) > cache.staleFrameCount)
    {
        DSharcClearEntry(cache, index);
        return;
    }
    state.flags &= DSHARC_FLAG_VALID;
    cache.states[index] = state;
    cache.materials[index] = (DSharcMaterial)0;
    cache.estimates[index] = float4(0, 0, 0, 0);
    cache.currentRadiance[index] = float4(0, 0, 0, 0);
}

// The first thread to mark this entry supplies its representative hit point.
// Other threads may get its index before the payload write completes. They
// MUST NOT read the surface/material/radiance until the Request pass barrier.
uint DSharcRequest(DSharcParameters cache, DSharcSurface surface)
{
    uint64_t key;
    if (!DSharcMakeKey(cache.grid, surface, key))
        return DSHARC_INVALID_INDEX;
    uint index = DSharcFindOrInsertKey(cache, key);
    if (index == DSHARC_INVALID_INDEX)
        return index;
    uint previousFlags;
    InterlockedOr(cache.states[index].flags, DSHARC_FLAG_REQUESTED, previousFlags);
    if ((previousFlags & DSHARC_FLAG_REQUESTED) == 0u)
    {
        surface.normal = normalize(surface.normal);
        surface.padding0 = 0.0f;
        surface.padding1 = 0.0f;
        cache.surfaces[index] = surface;
        cache.states[index].lastRequestedFrame = cache.frameIndex;
    }
    return index;
}

// One invocation per capacity slot, after all Request dispatches. Retained
// unrequested entries are also updated, matching the LumenPT working set.
void DSharcCompactEntry(DSharcParameters cache, uint index)
{
    if (index >= cache.capacity || DSharcLoadKey(cache, index) == uint64_t(0))
        return;
    uint offset;
    InterlockedAdd(cache.activeCount[0], 1u, offset);
    cache.activeEntries[offset] = index;
}

// Optional indirect dispatch helper. Writes a 3-uint Dispatch argument block.
// Uses a 2D dispatch for >65535 groups; linearize as shown in the guide.
uint3 DSharcGetDispatchGroups(uint entries, uint groupSize)
{
    uint size = max(groupSize, 1u);
    uint groups = entries / size + (entries % size != 0u ? 1u : 0u);
    uint groupsX = min(max(groups, 1u), 65535u);
    uint groupsY = max(groups / groupsX + (groups % groupsX != 0u ? 1u : 0u), 1u);
    return uint3(groupsX, groupsY, 1u);
}

void DSharcWriteDispatchArguments(DSharcParameters cache,
    RWByteAddressBuffer arguments, uint byteOffset, uint groupSize)
{
    arguments.Store3(byteOffset, DSharcGetDispatchGroups(cache.activeCount[0], groupSize));
}

uint DSharcLinearDispatchIndex(uint3 groupID, uint groupThreadIndex,
    uint groupsX, uint groupSize)
{
    return (groupID.y * groupsX + groupID.x) * groupSize + groupThreadIndex;
}

bool DSharcGetActiveEntry(DSharcParameters cache, uint activeIndex, out uint index)
{
    index = DSHARC_INVALID_INDEX;
    if (activeIndex >= cache.activeCount[0])
        return false;
    index = cache.activeEntries[activeIndex];
    return index < cache.capacity;
}

// Called exactly once per active entry by the renderer's material pass.
// An invalid/missing representative surface explicitly invalidates this frame.
void DSharcStoreMaterial(DSharcParameters cache, uint index, DSharcMaterial material)
{
    if (index >= cache.capacity)
        return;
    if (material.valid == 0u || !all(isfinite(material.diffuseAlbedo)) ||
        !all(isfinite(material.emissive)))
        material = (DSharcMaterial)0;
    else
    {
        material.diffuseAlbedo = saturate(material.diffuseAlbedo);
        material.emissive = max(material.emissive, 0.0f);
        material.valid = 1u;
        material.reserved = 0u;
    }
    cache.materials[index] = material;
}

// Lighting-pass only: a read of the frozen previous frame. This function does
// not allocate, refresh age or request missing indirect dependencies.
bool DSharcLookupPrevious(DSharcParameters cache, DSharcSurface hit,
    out float3 radiance)
{
    radiance = float3(0, 0, 0);
    uint64_t key;
    if (!DSharcMakeKey(cache.grid, hit, key))
        return false;
    uint index = DSharcFindKey(cache, key);
    if (index == DSHARC_INVALID_INDEX)
        return false;
    float4 value = cache.previousRadiance[index];
    if (value.w <= 0.0f)
        return false;
    radiance = value.xyz;
    return true;
}

// One estimate per active entry, AFTER material writes are visible. Average
// multiple lighting samples locally before submitting; this is not atomic.
void DSharcStoreEstimate(DSharcParameters cache, uint index, float3 radiance)
{
    if (index >= cache.capacity)
        return;
    bool valid = cache.materials[index].valid != 0u && all(isfinite(radiance));
    cache.estimates[index] = valid ? float4(max(radiance, 0.0f), 1.0f) : float4(0, 0, 0, 0);
}

// One invocation per active entry, after ALL lighting invocations finish.
// Separate previous/current buffers implement a true frame-frozen feedback step.
void DSharcResolveEntry(DSharcParameters cache, uint index)
{
    if (index >= cache.capacity || DSharcLoadKey(cache, index) == uint64_t(0))
        return;
    DSharcEntryState state = cache.states[index];
    float4 estimate = cache.estimates[index];
    if (estimate.w <= 0.0f)
    {
        state.accumulatedFrames = 0u;
        state.flags &= ~DSHARC_FLAG_VALID;
        cache.currentRadiance[index] = float4(0, 0, 0, 0);
    }
    else
    {
        float4 history = cache.previousRadiance[index];
        uint limit = max(cache.maxAccumulatedFrames, 1u);
        uint frames = history.w > 0.0f ? min(state.accumulatedFrames, limit - 1u) + 1u : 1u;
        float3 value = history.w > 0.0f ? lerp(history.xyz, estimate.xyz, rcp(float(frames))) : estimate.xyz;
        cache.currentRadiance[index] = float4(value, 1.0f);
        state.accumulatedFrames = frames;
        state.flags |= DSHARC_FLAG_VALID;
    }
    cache.states[index] = state;
}

// Gather-pass only, after Resolve. Request indices have a one-frame lifetime:
// consume them before the next BeginFrame can delete/reuse their slots.
bool DSharcGetCurrentRadiance(DSharcParameters cache, uint index, out float3 radiance)
{
    radiance = float3(0, 0, 0);
    if (index >= cache.capacity)
        return false;
    float4 value = cache.currentRadiance[index];
    if (value.w <= 0.0f)
        return false;
    radiance = value.xyz;
    return true;
}

#endif
