struct Hit
{
    bool valid;
    float3 position;
    float3 geometric_normal;
    float3 normal;
    float3 albedo;
    float3 emission;
};

struct Scene
{
    RaytracingAccelerationStructure tlas;
    StructuredBuffer<float3> positions;
    StructuredBuffer<float3> normals;
    StructuredBuffer<uint3> triangles;
    StructuredBuffer<float4> albedos;
    StructuredBuffer<float4> emissions;
    float ray_epsilon;

    Hit intersect(float3 origin, float3 direction)
    {
        RayDesc ray = {origin, 0.0, direction, 1e30};
        RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> query;
        query.TraceRayInline(tlas, RAY_FLAG_NONE, 0xff, ray);
        while (query.Proceed()) {}
        Hit hit = {};
        hit.valid = query.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
        if (!hit.valid) return hit;

        uint primitive = query.CommittedPrimitiveIndex();
        uint3 ids = triangles[primitive];
        float3 a = positions[ids.x], b = positions[ids.y], c = positions[ids.z];
        float2 uv = query.CommittedTriangleBarycentrics();
        float3 bary = float3(1.0 - uv.x - uv.y, uv.x, uv.y);
        hit.position = bary.x * a + bary.y * b + bary.z * c;
        float3 ng = normalize(cross(b - a, c - a));
        if (dot(ng, direction) > 0.0) ng = -ng;
        float3 ns = normals[ids.x] * bary.x + normals[ids.y] * bary.y + normals[ids.z] * bary.z;
        ns = dot(ns, ns) > 1e-12 ? normalize(ns) : ng;
        if (dot(ns, ng) < 0.0) ns = -ns;
        if (dot(ns, -direction) <= 0.0) ns = ng;
        hit.geometric_normal = ng;
        hit.normal = ns;
        hit.albedo = albedos[primitive].xyz;
        hit.emission = emissions[primitive].xyz;
        return hit;
    }

    float3 offset_origin(Hit hit)
    {
        float scale = max(1.0, max(abs(hit.position.x), max(abs(hit.position.y), abs(hit.position.z))));
        return hit.position + hit.geometric_normal * (ray_epsilon + 2e-7 * scale);
    }

    bool visible(float3 origin, float3 direction)
    {
        RayDesc ray = {origin, 0.0, direction, 1e30};
        RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH |
            RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> query;
        query.TraceRayInline(tlas, RAY_FLAG_NONE, 0xff, ray);
        while (query.Proceed()) {}
        return query.CommittedStatus() == COMMITTED_NOTHING;
    }
};
