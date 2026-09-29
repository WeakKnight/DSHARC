#ifndef DSHARC_SAMPLING_H
#define DSHARC_SAMPLING_H

#include "DSharcTypes.h"

// Supply two independent uniform samples in [0,1). Returned direction is unit
// length and has PDF max(dot(normal, direction), 0) / PI.
float3 DSharcSampleCosineHemisphere(float3 normal, float2 random)
{
    float3 n = normalize(normal);
    float3 helper = abs(n.z) < 0.999f ? float3(0, 0, 1) : float3(0, 1, 0);
    float3 tangent = normalize(cross(helper, n));
    float3 bitangent = cross(n, tangent);
    float radius = sqrt(saturate(random.x));
    float angle = 2.0f * DSHARC_PI * random.y;
    return radius * cos(angle) * tangent + radius * sin(angle) * bitangent +
        sqrt(max(1.0f - radius * radius, 0.0f)) * n;
}

// Optional LumenPT-style heuristic, not a requirement of the storage API.
// Evaluate at a valid indirect surface hit. On a failed Request, continue the
// path or explicitly enqueue a fallback; this predicate alone is not success.
bool DSharcShouldRequest(float hitDistance, float pathRoughness,
    float cellSize, bool lastBounce)
{
    float roughnessSquared = saturate(pathRoughness) * saturate(pathRoughness);
    float alphaSquared = min(roughnessSquared * roughnessSquared, 0.9999f);
    float footprint = hitDistance * sqrt(0.5f * alphaSquared / (1.0f - alphaSquared));
    return lastBounce || (hitDistance > 1.73205080757f * cellSize && footprint > cellSize);
}

// directIrradiance must include cosine, visibility and sampling/PDF weights.
// indirectIncidentRadiance is the mean radiance from cosine-weighted rays.
// Consequently only the direct term is divided by PI here. Do not multiply
// either input by this surface's albedo before calling this helper.
float3 DSharcEvaluateDiffuseRadiance(DSharcMaterial material,
    float3 directIrradiance, float3 indirectIncidentRadiance)
{
    return material.emissive + material.diffuseAlbedo *
        (directIrradiance / DSHARC_PI + indirectIncidentRadiance);
}

#endif
