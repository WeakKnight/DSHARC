Texture2D<float4> g_input;
RWTexture2D<float4> g_output;
uniform float g_exposure_ev;

float3 linear_to_srgb(float3 value)
{
    return select(value <= 0.0031308, value * 12.92, 1.055 * pow(value, 1.0 / 2.4) - 0.055);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint3 tid : SV_DispatchThreadID)
{
    uint2 size;
    g_output.GetDimensions(size.x, size.y);
    if (any(tid.xy >= size)) return;
    float3 color = max(g_input[tid.xy].xyz * exp2(g_exposure_ev), 0.0);
    // Fitted filmic curve; this is a display transform, not an ACES pipeline.
    float3 mapped = saturate((color * (2.51 * color + 0.03)) /
        (color * (2.43 * color + 0.59) + 0.14));
    g_output[tid.xy] = float4(linear_to_srgb(mapped), 1.0);
}
