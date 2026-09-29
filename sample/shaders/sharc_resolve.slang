import scene;
#define SHARC_UPDATE 0
#define SHARC_QUERY 0
#include "sharc_bridge.slangh"

[shader("compute")]
[numthreads(256, 1, 1)]
void compute_main(uint3 tid : SV_DispatchThreadID)
{
    SharcResolveParameters params;
    // Camera/config changes clear the cache; no cross-level reprojection.
    params.cameraPositionPrev = g_sharc.hashGridParameters.cameraPosition;
    params.accumulationFrameNum = g_cache_history;
    params.responsiveFrameNum = 1;
    params.staleFrameNumMax = 1024;
    params.frameIndex = g_cache_frame;
    SharcResolveEntry(tid.x, g_sharc, params);
}
