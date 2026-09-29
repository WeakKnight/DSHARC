#define PATH_LIBRARY 1
#define DSHARC_FEEDBACK 1
#include "path_tracer.slang"

[shader("compute")]
[numthreads(64, 1, 1)]
void compute_main(uint3 tid : SV_DispatchThreadID)
{
    uint entry;
    if (!DSharcGetActiveEntry(g_dsharc, tid.x, entry)) return;
    DSharcSurface surface = g_dsharc.surfaces[entry];
    // Reconstruct the representative's material, validating the short probe.
    float bias = g_scene.ray_epsilon * 4 + 1e-5;
    Hit hit = g_scene.intersect(surface.position + surface.normal * bias, -surface.normal);
    DSharcMaterial material = {};
    if (hit.valid && length(hit.position - surface.position) < bias * 2 &&
        dot(hit.geometric_normal, surface.normal) > .99)
    {
        material.diffuseAlbedo = hit.albedo;
        material.emissive = hit.emission;
        material.valid = 1;
    }
    DSharcStoreMaterial(g_dsharc, entry, material);
    if (material.valid == 0)
    {
        InterlockedAdd(g_dstats[5], 1);
        DSharcStoreEstimate(g_dsharc, entry, 0);
        return;
    }
    float3 total = 0;
    for (uint i = 0; i < g_samples_per_frame; ++i)
    {
        RNG rng = RNG(entry, g_sample_base + i, g_seed ^ 0x7541c319);
        float3 direct = sample_sun_irradiance(hit, rng);
        float3 direction = cosine_sample(rng.next2(), hit.normal);
        float3 incident = 0;
        if (dot(direction, hit.geometric_normal) > 0)
        {
            float3 stats = 0;
            incident = trace_path(g_scene.offset_origin(hit), direction, rng, stats,
                1, max(dot(hit.normal, direction), 0) * INV_PI);
        }
        total += DSharcEvaluateDiffuseRadiance(material, direct, incident);
    }
    DSharcStoreEstimate(g_dsharc, entry, total / g_samples_per_frame);
}
