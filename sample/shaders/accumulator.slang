Texture2D<float4> g_sample;
RWTexture2D<float4> g_history;
uniform uint g_previous_samples;
uniform uint g_new_samples;

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint3 tid : SV_DispatchThreadID)
{
    uint2 size;
    g_history.GetDimensions(size.x, size.y);
    if (any(tid.xy >= size)) return;
    float3 value = g_sample[tid.xy].xyz;
    // The first frame must never read uninitialized texture contents.
    if (g_previous_samples > 0)
        value = lerp(g_history[tid.xy].xyz, value,
            float(g_new_samples) / float(g_previous_samples + g_new_samples));
    g_history[tid.xy] = float4(value, 1.0);
}
