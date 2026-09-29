#define PATH_LIBRARY 1
#include "path_tracer.slang"

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint3 tid : SV_DispatchThreadID)
{
    uint2 size;
    g_output.GetDimensions(size.x, size.y);
    if (any(tid.xy >= size)) return;
    float3 total = 0;
    uint pixel = tid.y * size.x + tid.x;
    for (uint i = 0; i < g_samples_per_frame; ++i)
    {
        DSharcPathRecord record = g_paths[pixel * g_samples_per_frame + i];
        float3 value = record.radiance;
        if (record.entry != DSHARC_INVALID_INDEX)
        {
            float3 tail;
            if (DSharcGetCurrentRadiance(g_dsharc, record.entry, tail))
                value += record.throughput * tail;
            else
            {
                // Reconstruct the SAME original camera path with cache disabled.
                // Replace the whole sample; never drop the tail or double the prefix.
                RNG rng = RNG(pixel, g_sample_base+i, g_seed);
                float3 direction = g_camera.direction(float2(tid.xy)+rng.next2(), size);
                float3 stats = 0;
                value = trace_path(g_camera.position, direction, rng, stats);
                InterlockedAdd(g_dstats[2], 1);
            }
        }
        total += value;
    }
    g_output[tid.xy] = float4(total / g_samples_per_frame, 1);
}
