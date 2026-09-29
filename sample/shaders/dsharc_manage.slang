import scene;
#include "dsharc_bridge.slangh"

[shader("compute")]
[numthreads(256, 1, 1)]
void begin_main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x == 0) DSharcResetActiveCount(g_dsharc);
    DSharcBeginFrameEntry(g_dsharc, tid.x);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void compact_main(uint3 tid : SV_DispatchThreadID)
{
    DSharcCompactEntry(g_dsharc, tid.x);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void args_main(uint3 tid : SV_DispatchThreadID)
{
    DSharcWriteDispatchArguments(g_dsharc, g_dispatch, 0, 64);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void resolve_main(uint3 tid : SV_DispatchThreadID)
{
    uint entry;
    if (DSharcGetActiveEntry(g_dsharc, tid.x, entry))
        DSharcResolveEntry(g_dsharc, entry);
}
