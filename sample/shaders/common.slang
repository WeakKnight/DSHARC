static const float PI = 3.14159265358979323846;
static const float INV_PI = 1.0 / PI;

uint hash32(uint value)
{
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    return value ^ (value >> 16);
}

struct RNG
{
    uint state;
    __init(uint pixel, uint sampleIndex, uint seed)
    {
        state = hash32(pixel ^ hash32(sampleIndex + 0x9e3779b9u) ^ hash32(seed));
    }
    [mutating] float next()
    {
        state = state * 747796405u + 2891336453u;
        uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
        word = (word >> 22u) ^ word;
        return float(word >> 8u) * (1.0 / 16777216.0);
    }
    [mutating] float2 next2() { return float2(next(), next()); }
};

float3 local_to_world(float3 local, float3 normal)
{
    float3 helper = abs(normal.y) < 0.999 ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 tangent = normalize(cross(helper, normal));
    return local.x * tangent + local.y * cross(normal, tangent) + local.z * normal;
}

float3 cosine_sample(float2 u, float3 normal)
{
    float radius = sqrt(u.x);
    float phi = 2.0 * PI * u.y;
    return local_to_world(float3(radius * cos(phi), radius * sin(phi), sqrt(1.0 - u.x)), normal);
}

float power_heuristic(float pdfA, float pdfB)
{
    float a = pdfA * pdfA;
    float b = pdfB * pdfB;
    return a / max(a + b, 1e-30);
}

struct Camera
{
    float3 position;
    float3 forward;
    float3 right;
    float3 up;
    float tan_half_fov;

    float3 direction(float2 pixel, uint2 size)
    {
        float2 ndc = pixel / float2(size) * 2.0 - 1.0;
        ndc.y = -ndc.y;
        return normalize(forward + tan_half_fov *
            (ndc.x * (float(size.x) / float(size.y)) * right + ndc.y * up));
    }
};
