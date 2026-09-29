#ifndef DSHARC_HASH_GRID_H
#define DSHARC_HASH_GRID_H

#include "DSharcTypes.h"

uint DSharcNormalBin(float3 normal)
{
    float3 a = abs(normal);
    if (a.x >= a.y && a.x >= a.z)
        return normal.x >= 0.0f ? 0u : 1u;
    if (a.y >= a.z)
        return normal.y >= 0.0f ? 2u : 3u;
    return normal.z >= 0.0f ? 4u : 5u;
}

uint DSharcGetLevel(DSharcGridParameters grid, float3 position)
{
    float distanceToCamera = length(position - grid.cameraPosition);
    float level = floor(log2(max(distanceToCamera / grid.levelDistance, 1.0f)));
    return uint(clamp(level, 0.0f, float(min(grid.maxLevel, 31u))));
}

float DSharcGetCellSize(DSharcGridParameters grid, uint level)
{
    return grid.baseCellSize * exp2(float(min(level, 31u)));
}

// Exact packed key, not a fingerprint: XYZ each use signed 18-bit coordinates,
// then 5 level bits, 3 normal bits and one occupied bit. Bit 63 is unused.
// Out-of-domain coordinates fail instead of silently wrapping/aliasing.
bool DSharcMakeKey(DSharcGridParameters grid, DSharcSurface surface,
    out uint64_t key)
{
    key = uint64_t(0);
    if (!all(isfinite(surface.position)) || !all(isfinite(surface.normal)) ||
        !all(isfinite(grid.cameraPosition)) || !all(isfinite(grid.origin)) ||
        !isfinite(grid.baseCellSize) || !isfinite(grid.levelDistance) ||
        grid.baseCellSize <= 0.0f || grid.levelDistance <= 0.0f ||
        dot(surface.normal, surface.normal) < 1e-12f)
        return false;

    uint level = DSharcGetLevel(grid, surface.position);
    float cellSize = DSharcGetCellSize(grid, level);
    if (!isfinite(cellSize) || cellSize <= 0.0f)
        return false;
    float3 coordinates = floor((surface.position - grid.origin) / cellSize);
    if (!all(isfinite(coordinates)) || any(coordinates < -131072.0f) ||
        any(coordinates > 131071.0f))
        return false;

    uint3 packed = uint3(int3(coordinates)) & 0x3ffffu;
    key = uint64_t(packed.x) | (uint64_t(packed.y) << 18) |
        (uint64_t(packed.z) << 36) | (uint64_t(level) << 54) |
        (uint64_t(DSharcNormalBin(surface.normal)) << 59) |
        (uint64_t(1) << 62);
    return true;
}

uint DSharcHashKey(uint64_t key)
{
    uint h = uint(key) ^ (uint(key >> 32) * 0x9e3779b9u);
    h ^= h >> 16;
    h *= 0x7feb352du;
    h ^= h >> 15;
    h *= 0x846ca68bu;
    return h ^ (h >> 16);
}

uint DSharcNextSlot(uint slot, uint capacity)
{
    return slot + 1u == capacity ? 0u : slot + 1u;
}

// Read-only lookup. Legal only after Request has completed and its UAV writes
// are visible. Deletion leaves holes, so an empty slot does NOT end the search.
uint DSharcFindKey(DSharcParameters cache, uint64_t key)
{
    if (cache.capacity == 0u || key == uint64_t(0))
        return DSHARC_INVALID_INDEX;
    uint slot = DSharcHashKey(key) % cache.capacity;
    uint count = min(uint(DSHARC_PROBE_COUNT), cache.capacity);
    [loop]
    for (uint i = 0u; i < count; ++i)
    {
        if (cache.keys[slot] == key)
            return slot;
        slot = DSharcNextSlot(slot, cache.capacity);
    }
    return DSHARC_INVALID_INDEX;
}

// Request-pass only. Atomic reads avoid mixing plain reads and atomic writes
// during insertion. Search the WHOLE window before claiming a hole: otherwise
// deletion could cause a duplicate of a surviving key farther down the window.
uint DSharcFindOrInsertKey(DSharcParameters cache, uint64_t key)
{
    if (cache.capacity == 0u || key == uint64_t(0))
        return DSHARC_INVALID_INDEX;
    uint start = DSharcHashKey(key) % cache.capacity;
    uint count = min(uint(DSHARC_PROBE_COUNT), cache.capacity);
    uint slot = start;
    [loop]
    for (uint i = 0u; i < count; ++i)
    {
        uint64_t observed;
        InterlockedCompareExchange(cache.keys[slot], uint64_t(0), uint64_t(0), observed);
        if (observed == key)
            return slot;
        slot = DSharcNextSlot(slot, cache.capacity);
    }

    slot = start;
    [loop]
    for (uint j = 0u; j < count; ++j)
    {
        uint64_t observed;
        InterlockedCompareExchange(cache.keys[slot], uint64_t(0), key, observed);
        if (observed == uint64_t(0) || observed == key)
            return slot;
        slot = DSharcNextSlot(slot, cache.capacity);
    }
    return DSHARC_INVALID_INDEX;
}

#endif
