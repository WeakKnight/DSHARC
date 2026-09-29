import common;
import scene;
import lighting;

#ifndef SHARC_UPDATE
#define SHARC_UPDATE 0
#endif
#ifndef SHARC_QUERY
#define SHARC_QUERY 0
#endif

#if SHARC_UPDATE || SHARC_QUERY
#include "sharc_bridge.slangh"
#endif

#ifndef DSHARC_REQUEST
#define DSHARC_REQUEST 0
#endif
#ifndef DSHARC_FEEDBACK
#define DSHARC_FEEDBACK 0
#endif
#ifndef PATH_LIBRARY
#define PATH_LIBRARY 0
#endif
#if DSHARC_REQUEST || DSHARC_FEEDBACK || PATH_LIBRARY
#include "dsharc_bridge.slangh"
#endif

ParameterBlock<Scene> g_scene;
uniform Camera g_camera;
uniform Lighting g_lighting;
uniform uint g_sample_base;
uniform uint g_samples_per_frame;
uniform uint g_max_bounces;
uniform uint g_seed;
RWTexture2D<float4> g_output;

float3 sample_sun_irradiance(Hit hit, inout RNG rng)
{
    if (g_lighting.sun_intensity <= 0.0) return 0.0;
    float3 to_sun = g_lighting.sample_sun(rng.next2());
    float cosine = max(dot(hit.normal, to_sun), 0.0);
    if (cosine <= 0 || dot(hit.geometric_normal, to_sun) <= 0 ||
        !g_scene.visible(g_scene.offset_origin(hit), to_sun)) return 0.0;
    float light_pdf = g_lighting.sun_pdf();
    return cosine * g_lighting.sun_radiance() *
        (power_heuristic(light_pdf, cosine * INV_PI) / light_pdf);
}

float3 trace_path(float3 origin, float3 direction, inout RNG rng, inout float3 stats
#if DSHARC_REQUEST
    , uint start_depth, float initial_pdf, out uint requested_entry, out float3 requested_throughput
#else
    , uint start_depth = 0, float initial_pdf = 0
#endif
    )
{
    float3 radiance = 0.0;
    float3 throughput = 1.0;
    float previous_bsdf_pdf = initial_pdf;
#if DSHARC_REQUEST
    requested_entry = DSHARC_INVALID_INDEX;
    requested_throughput = 0;
#endif
#if SHARC_UPDATE
    SharcState state;
    SharcInit(state);
    const uint max_depth = 32;
#else
    uint max_depth = g_max_bounces;
#endif

    // max_bounces counts scattering events. The final segment still sees
    // environment or emission, so the last sun NEE sample has its MIS partner.
    for (uint depth = start_depth; depth <= max_depth; ++depth)
    {
        stats.z += 1;
        Hit hit = g_scene.intersect(origin, direction);
        if (!hit.valid)
        {
            // Sky is BSDF-sampled only. MIS applies only to the solar disk,
            // which is also sampled by next-event estimation at surfaces.
            float3 environment = g_lighting.sky(direction);
            if (g_lighting.in_sun(direction))
            {
                float weight = depth == 0 ? 1.0 : power_heuristic(previous_bsdf_pdf, g_lighting.sun_pdf());
                environment += g_lighting.sun_radiance() * weight;
            }
#if SHARC_UPDATE
            SharcUpdateMiss(g_sharc, state, environment);
#endif
            radiance += throughput * environment;
            break;
        }
#if SHARC_QUERY
        // Only diffuse continuation segments, never camera visibility hits.
        if (depth > 0 && depth < max_depth)
        {
            uint level = HashGridGetLevel(hit.position, g_sharc.hashGridParameters);
            float voxel = HashGridGetVoxelSize(level, g_sharc.hashGridParameters);
            if (length(hit.position - origin) > 1.732051 * voxel)
            {
                stats.y += 1;
                float3 cached;
                if (SharcGetCachedRadiance(g_sharc, cache_hit(hit), cached, false))
                {
                    stats.x += 1;
                    radiance += throughput * cached;
                    break; // Cached value already contains emission and direct lighting.
                }
            }
        }
#endif
#if DSHARC_REQUEST
        if (depth > 0 && depth < max_depth && demand_eligible(hit, origin))
        {
            uint entry = DSharcRequest(g_dsharc, demand_surface(hit));
            if (entry != DSHARC_INVALID_INDEX)
            {
                requested_entry = entry;
                requested_throughput = throughput;
                InterlockedAdd(g_dstats[0], 1);
                break;
            }
            InterlockedAdd(g_dstats[1], 1); // Full probe window: continue PT.
        }
#endif
#if DSHARC_FEEDBACK
        if (depth < max_depth && demand_eligible(hit, origin))
        {
            float3 cached;
            if (DSharcLookupPrevious(g_dsharc, demand_surface(hit), cached))
            {
                radiance += throughput * cached;
                InterlockedAdd(g_dstats[3], 1);
                break;
            }
            InterlockedAdd(g_dstats[4], 1); // Missing dependency: trace onward.
        }
#endif
        radiance += throughput * hit.emission;
        if (depth == max_depth)
        {
#if SHARC_UPDATE
            SharcUpdateMiss(g_sharc, state, hit.emission);
#endif
            break;
        }
        origin = g_scene.offset_origin(hit);

        float3 direct = hit.albedo * INV_PI * sample_sun_irradiance(hit, rng);
        radiance += throughput * direct;
#if SHARC_UPDATE
        // Keep a full-tail fallback during cold start: never slide roots out
        // before their tail is accounted for. Mature roots can use SDK resampling.
        if (depth < SHARC_PROPAGATION_DEPTH)
        {
            uint previous_length = state.pathLength;
            bool continue_tracing = SharcUpdateHit(g_sharc, state, cache_hit(hit), direct, rng.next());
            if (!continue_tracing)
            {
                // SDK false means either insertion failure (state unchanged),
                // or a resampled tail already propagated to all earlier roots.
                // Since we insert at most four roots, pathLength cannot saturate
                // before this call, so the two outcomes are distinguishable.
                if (state.pathLength > previous_length)
                {
                    stats.x += 1;
                    break; // Do NOT also add direct or trace the replaced tail.
                }
                stats.y += 1;
                SharcUpdateMiss(g_sharc, state, direct + hit.emission);
            }
        }
        else
            SharcUpdateMiss(g_sharc, state, direct + hit.emission);
#endif

        direction = cosine_sample(rng.next2(), hit.normal);
        if (dot(direction, hit.geometric_normal) <= 0.0) break;
        previous_bsdf_pdf = max(dot(hit.normal, direction), 0.0) * INV_PI;
        throughput *= hit.albedo; // Lambert f*cos/pdf cancels to albedo.
        float3 segment_weight = hit.albedo;
        if (max(throughput.x, max(throughput.y, throughput.z)) <= 0.0) break;
        if (depth >= 3)
        {
            float survival = clamp(max(throughput.x, max(throughput.y, throughput.z)), 0.05, 0.95);
            if (rng.next() >= survival) break;
            throughput /= survival;
            segment_weight /= survival;
        }
#if SHARC_UPDATE
        // Incremental f*cos/pdf including roulette compensation, NOT the
        // accumulated camera throughput (which would multiply albedo twice).
        SharcSetThroughput(state, segment_weight);
#endif
    }
    return radiance;
}

#if !PATH_LIBRARY
[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint3 tid : SV_DispatchThreadID)
{
    uint2 size;
    g_output.GetDimensions(size.x, size.y);
#if SHARC_UPDATE
    uint2 launch_size = (size + g_update_downscale - 1) / g_update_downscale;
    if (any(tid.xy >= launch_size)) return;
#else
    if (any(tid.xy >= size)) return;
#endif
    float3 total = 0.0;
    float3 stats = 0.0;
    for (uint i = 0; i < g_samples_per_frame; ++i)
    {
        RNG rng = RNG(tid.y * size.x + tid.x, g_sample_base + i, g_seed);
#if SHARC_UPDATE
        // Independent stream and a uniform sample over each stratum, including
        // partial edge tiles. Covers the entire image even at odd resolutions.
        rng = RNG(tid.y * launch_size.x + tid.x, g_sample_base + i, g_seed ^ 0xa511e9b3);
        uint2 base = tid.xy * g_update_downscale;
        float2 pixel = float2(base) + rng.next2() * float2(min(uint2(g_update_downscale), size - base));
        float3 direction = g_camera.direction(pixel, size);
#else
        float3 direction = g_camera.direction(float2(tid.xy) + rng.next2(), size);
#endif
#if DSHARC_REQUEST
        DSharcPathRecord record;
        record.reserved = 0;
        record.radiance = trace_path(g_camera.position, direction, rng, stats, 0, 0,
            record.entry, record.throughput);
        g_paths[(tid.y * size.x + tid.x) * g_samples_per_frame + i] = record;
#else
        total += trace_path(g_camera.position, direction, rng, stats);
#endif
    }
#if !SHARC_UPDATE && !DSHARC_REQUEST
    g_output[tid.xy] = float4(total / float(g_samples_per_frame), 1.0);
#endif
#if SHARC_QUERY
    g_cache_stats[tid.xy] = float4(stats / float(g_samples_per_frame), 0.0);
#endif
#if SHARC_UPDATE
    g_update_stats[tid.xy] = float4(stats / float(g_samples_per_frame), 0.0);
#endif
}
#endif
