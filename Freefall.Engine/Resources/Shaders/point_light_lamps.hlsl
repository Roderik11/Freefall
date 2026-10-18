// Lamps (PointLight.OnlyWhenDark) lit by the daylight where each one hangs (PointLightBuffer.UpdateLamps): once per
// frame after the sky captures are drawn, every lamp in the scene whatever the camera looks at (a lamp coming into view is
// already as it should be). The light cull reads the result the frame after.
//
// The daylight at a lamp, in lux on level ground: the sun's (or moon's) SunLight × SunHeight and the sky's Ambient, as far
// as they reach the lamp past the terrain (sky_visibility.hlsli: under a hill, the horizons of a valley, light through a
// cave mouth) and past the meshes (sky_capture.hlsli: walls, roofs, the cave kit). In the open that is the ground
// illuminance Daylight.Lamps follows; in a closed room or deep in a cave it is 0 at noon, and the lamp is lit. Hysteresis
// per lamp (on below LAMPS_ON_BELOW_LUX, out above LAMPS_OFF_ABOVE_LUX) and a fade, as Daylight does for the open ground.
//
// The sun's cascades are not used: they cover only what the camera looks at, so a lamp in a room at the edge of the view
// or behind the camera was sunlit to them (out), and came on as the camera turned to it. Outdoors a lamp switches with
// the sun under the horizon, where only the sky counts; indoors the roof that hides the sky hides the sun too, so the
// meshes' share of the sky stands in for the sun's. No SkyCaptureBounce: it stands in for daylight that reaches a room's
// surfaces through its doors and windows, and with it no lamp would light at noon (4 % of 15000 lux).
//
// LAMPS_PER_GROUP lamps per group of 64 threads, one thread per sky direction: each looks its direction up at the lamp's
// five taps, the first thread of the lamp adds them up and judges it. One thread doing all 80 lookups of a lamp took
// 0.045 ms for Eldermoor's 225 lamps (latency: four waves on the whole GPU).
// Docs: .agent/knowledge/rendering_engine/features/lighting/point_lights.md.

#pragma kernel CSLamps

cbuffer PushConstants : register(b3)
{
    uint RecordsIdx;            // StructuredBuffer<PointLightRecord>: the registry
    uint RecordCountIdx;        // registry slots to walk
    uint TransformsIdx;         // StructuredBuffer<row_major float4x4>: the global transform buffer
    uint FrameParamsIdx;        // StructuredBuffer<PointLightFrame>: [0] (sun, sky, terrain, fade step)
    uint LampsUAVIdx;           // RWStructuredBuffer<LampState> per slot
    uint TickIdx;               // Engine.TickCount: LampState.Seen (a lamp not judged the tick before starts as it should be)
    uint SkyCaptureViewsIdx;    // StructuredBuffer<SkyCaptureView>[SKY_CAPTURE_VIEWS]; 0 = no captures
    uint SkyCaptureDepthIdx;    // their depth maps (Texture2DArray)
};

#include "point_lights.hlsli"
#include "sky_visibility.hlsli"
#include "sky_capture.hlsli"

#define LAMPS_PER_GROUP (64 / SKY_CAPTURE_VIEWS)    // PointLightBuffer.LampsPerGroup
#define LAMP_TAPS 5

// The global root signature's linear clamp sampler (s2), for the heightmap and the depth maps
SamplerState LinearClamp : register(s2);

// The meshes' share of the sky is looked up at the light and half a metre to four sides; the middle of the five counts. A
// street lamp's own cap over the flame then counts for one tap, as does a tap that passes through a thin wall next to a
// lamp in a room (two in its corner). 0.1 m over the light: a lamp is a point in the air, with no surface of its own to
// clear (the surfaces' 0.4 m along the normal would lift a lamp under the rafters through the roof boards).
static const float3 LampTaps[LAMP_TAPS] =
{
    float3(0, 0.1, 0), float3(0.5, 0.1, 0), float3(-0.5, 0.1, 0), float3(0, 0.1, 0.5), float3(0, 0.1, -0.5)
};

groupshared SkyCaptureView gViews[SKY_CAPTURE_VIEWS];
groupshared float gOpen[LAMPS_PER_GROUP][LAMP_TAPS][SKY_CAPTURE_VIEWS];

// The daylight where a lamp hangs, lux on level ground, given the meshes' share of the sky there
float LampDaylight(float3 worldPos, float meshes, PointLightFrame frame)
{
    TerrainCover terrain;
    terrain.Heightmap = frame.TerrainHeightmap;
    terrain.MaxHeight = frame.TerrainMaxHeight;
    terrain.Origin = frame.TerrainOrigin;
    terrain.Size = frame.TerrainSize;

    // The terrain's tolerance as near the camera: a lamp is placed on the heightmap, not on the coarser terrain drawn far
    // off, and the judgement must not change with the camera's distance
    float open, portal;
    float sky = SkyVisibilityParts(terrain, LinearClamp, worldPos, float3(0, 1, 0), TerrainTolerance(0.0), open, portal);

    // Joined as sky_common.fx SkyVisibilityWith does, without the bounce: in the open the meshes take their share of what
    // the terrain's horizons leave; under the terrain the smaller of the two (both judge the same rock)
    float skyVisible = lerp(min(portal, meshes), sky * meshes, open);
    float sunVisible = open * meshes;
    return frame.SunLight * frame.SunHeight * sunVisible + frame.Ambient * skyVisible;
}

[numthreads(64, 1, 1)]
void CSLamps(uint3 gid : SV_GroupID, uint gi : SV_GroupIndex)
{
    bool meshes = SkyCaptureViewsIdx != 0 && SkyCaptureDepthIdx != 0;
    if (meshes && gi < SKY_CAPTURE_VIEWS)
    {
        StructuredBuffer<SkyCaptureView> views = ResourceDescriptorHeap[SkyCaptureViewsIdx];
        gViews[gi] = views[gi];
    }
    GroupMemoryBarrierWithGroupSync();

    uint local = gi / SKY_CAPTURE_VIEWS;    // the lamp in this group
    uint dir = gi % SKY_CAPTURE_VIEWS;      // the direction this thread looks up
    uint slot = gid.x * LAMPS_PER_GROUP + local;

    PointLightRecord r = (PointLightRecord)0;
    if (slot < RecordCountIdx)
    {
        StructuredBuffer<PointLightRecord> records = ResourceDescriptorHeap[RecordsIdx];
        r = records[slot];
    }
    bool isLamp = (r.Flags & POINT_LIGHT_ENABLED) != 0 && (r.Flags & POINT_LIGHT_WHEN_DARK) != 0;
    float3 world = 0;
    if (isLamp)
    {
        StructuredBuffer<row_major float4x4> transforms = ResourceDescriptorHeap[TransformsIdx];
        world = mul(float4(0, 0, 0, 1), transforms[r.TransformSlot]).xyz;
    }

    // This direction at the five taps; one not captured yet is not read (its map may never have been drawn)
    if (isLamp && meshes && gViews[dir].Valid > 0.0)
    {
        Texture2DArray<float> depths = ResourceDescriptorHeap[SkyCaptureDepthIdx];
        float w, h, slices;
        depths.GetDimensions(w, h, slices);
        [unroll] for (uint t = 0; t < LAMP_TAPS; t++)
            gOpen[local][t][dir] = SkyCaptureOpenAt(gViews[dir], dir, depths, float2(w, h), LinearClamp, world + LampTaps[t]);
    }
    GroupMemoryBarrierWithGroupSync();

    if (dir != 0 || slot >= RecordCountIdx)
        return;

    RWStructuredBuffer<LampState> lamps = ResourceDescriptorHeap[LampsUAVIdx];
    LampState state = lamps[slot];
    if (!isLamp)
    {
        // Not a lamp (or off): the next lamp in this slot, or this one enabled again, starts as it should be
        if (state.Seen != 0)
            lamps[slot] = (LampState)0;
        return;
    }

    // The meshes' share of the sky at each tap: the directions are spread cosine-weighted about the zenith, so for level
    // ground each counts alike. 1 without captures. Then the middle of the five taps.
    float v[LAMP_TAPS];
    [unroll] for (uint t = 0; t < LAMP_TAPS; t++)
    {
        float lit = 0, total = 0;
        [unroll] for (uint i = 0; i < SKY_CAPTURE_VIEWS; i++)
        {
            if (meshes && gViews[i].Valid > 0.0)
            {
                lit += gOpen[local][t][i];
                total += 1.0;
            }
        }
        v[t] = total > 0.0 ? lit / total : 1.0;
    }
    [unroll] for (uint a = 0; a < 3; a++)    // sorted far enough that v[2] is the middle
    {
        [unroll] for (uint b = a + 1; b < LAMP_TAPS; b++)
        {
            float lo = min(v[a], v[b]);
            v[b] = max(v[a], v[b]);
            v[a] = lo;
        }
    }

    StructuredBuffer<PointLightFrame> frames = ResourceDescriptorHeap[FrameParamsIdx];
    PointLightFrame frame = frames[0];
    float lux = LampDaylight(world, v[2], frame);

    bool fresh = state.Seen == 0 || state.Seen + 1 != TickIdx;
    bool lit = fresh || state.Lit != 0;     // as Daylight: lit unless it is bright enough to go out
    if (lit ? lux > LAMPS_OFF_ABOVE_LUX : lux < LAMPS_ON_BELOW_LUX)
        lit = !lit;

    float target = lit ? 1.0 : 0.0;
    state.Lamp = fresh ? target
        : (target > state.Lamp ? min(target, state.Lamp + frame.LampFadeStep) : max(target, state.Lamp - frame.LampFadeStep));
    state.Lit = lit ? 1u : 0u;
    state.Seen = TickIdx;
    state.Lux = lux;
    lamps[slot] = state;
}
