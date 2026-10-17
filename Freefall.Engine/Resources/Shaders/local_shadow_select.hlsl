// Point light shadow selection on the GPU (LocalShadows.Select). Picks this frame's shadowed lights out of the
// visible list (point_light_cull.hlsl), gives their cube faces tiles in the local shadow atlas and builds the
// faces' views. The rules are those of the CPU picker it replaces (2026-10-16 phase 2), plus the hidden-centre test.
//
// CSScore      per visible light: rank score = luminance × intensity × range² / max(distance, range/4)²,
//              ×1.3 while it has a shadow (no swapping between near-equal lights), ×0.25 when its centre is hidden
//              behind the depth buffer (lights walled in), 0 under 32 px of on-screen radius; × its sun factor:
//              how much the light outshines the sunlight reaching it (cascade shadow map around the light, clouds
//              in the frame's sun irradiance). A street lamp at clear noon gets none, a room light that day keeps its
//              shadow, an overcast or night street gets them. Score histogram.
// CSGather     per visible light: the lights in the histogram's top bins (at least the budget) and those still
//              fading out become candidates, with their visible faces and preferred tier.
// CSSelect     one group: sorts the candidates (bitonic), then one thread fits them into the atlas in rank order
//              (smaller tier when short), keeps the dropped ones' tiles while they fade out, packs the tiles and
//              writes the shadow lights, the face list and the counts.
// CSBuildViews per face: matrices, planes, tile.
//
// State per registry slot (LightShadowState) carries fade and tier across frames; it starts over for a light
// that was not in view the frame before.

#pragma kernel CSScore
#pragma kernel CSGather
#pragma kernel CSSelect
#pragma kernel CSBuildViews

cbuffer PushConstants : register(b3)
{
    uint FrameParamsIdx;        // StructuredBuffer<PointLightFrame>: [0]
    uint VisibleUAVIdx;         // RWStructuredBuffer<PointLightData>
    uint VisibleCountUAVIdx;    // RWByteAddressBuffer: [0] = visible lights
    uint StateUAVIdx;           // RWStructuredBuffer<LightShadowState>, per registry slot
    uint StateCountIdx;         // slots the state buffer holds
    uint HistogramUAVIdx;       // RWStructuredBuffer<uint>[SCORE_BINS]
    uint CandidatesUAVIdx;      // RWStructuredBuffer<uint4>[MAX_CANDIDATES]: score bits, visible index, packed, slot
    uint CandidateCountUAVIdx;  // RWByteAddressBuffer: [0]
    uint PicksUAVIdx;           // RWStructuredBuffer<ShadowPick>[MAX_SHADOW_LIGHTS]
    uint FacesUAVIdx;           // RWStructuredBuffer<uint4>[LOCAL_SHADOW_MAX_VIEWS]: pick, face, tier, index in tier
    uint CountsUAVIdx;          // RWStructuredBuffer<uint>[16]: COUNT_*
    uint ShadowLightsUAVIdx;    // RWStructuredBuffer<LocalShadowLight>
    uint ViewsUAVIdx;           // RWStructuredBuffer<LocalShadowView>
    uint DepthTexIdx;           // linear depth (view-space Z, 0 = sky)
    uint SunShadowMapIdx;       // the sun's cascades this frame (Texture2DArray, standard Z), 0 = none rendered
    uint SunCascadesIdx;        // StructuredBuffer<CascadeData> (absolute-world VP)
    uint SunCascadeCountIdx;
};

#include "local_shadows.hlsli"
#include "point_lights.hlsli"
#include "sky_visibility.hlsli"

// The global root signature's linear clamp sampler (s2), for the terrain heightmap
SamplerState TerrainLinearClamp : register(s2);

// The sun's cascades (common.fx's CascadeData; this file does not include common.fx)
struct SunCascade
{
    float4 Planes[6];
    row_major float4x4 VP;      // absolute world -> clip
    row_major float4x4 PrevVP;
    float4 SplitDistances;
};

#define SCORE_BINS          64u
#define MAX_CANDIDATES      1024u   // LocalShadows.MaxCandidates
#define MAX_SHADOW_LIGHTS   32u     // LocalShadows.MaxLights
#define SLOT_COUNT          64u     // 512² atlas slots: 4 pages × 16
#define MIN_RADIUS_PIXELS   32.0    // a sphere smaller on screen gets no shadow
#define NEAR_PLANE          0.1     // casters nearer to the light are not drawn
#define GUARD_TEXELS        2.0     // a face's 90° sits this far inside its tile
#define STICKY_BONUS        1.3
#define HIDDEN_FACTOR       0.25
#define TIER_BAND           0.1

// Counts buffer (LocalShadows reads them back for the stats; the culler reads COUNT_LIGHTS)
#define COUNT_LIGHTS        0u
#define COUNT_VIEWS         1u
#define COUNT_SLOTS         2u
#define COUNT_FACES_TIER0   3u      // 3..5: faces per tier
#define COUNT_SLOT_TIER0    6u      // 6..8: first slot per tier
#define COUNT_CANDIDATES    9u

// Candidate .z: faces (bits 0-5) | preferred tier << 8 | ranked << 16 | picked << 17
#define CAND_RANKED         (1u << 16)
#define CAND_PICKED         (1u << 17)

// Per registry slot and renderer. Must match C# LocalShadows.LightState (16 bytes).
struct LightShadowState
{
    float Fade;     // fade 0..1
    uint Tier;      // 0 = none, else tier + 1
    uint Seen;      // last frame it was in view
    float SunFactor;// 0..1: how much the light outshines the sunlight reaching it (CSScore)
};

// Must match C# LocalShadows.Pick (16 bytes).
struct ShadowPick
{
    uint VisibleIndex;
    uint Slot;
    uint Packed;    // faces (bits 0-5) | tier << 8 | wanted << 16
    float Fade;
};

static const float3 FACE_FORWARD[6] =
{
    float3(1, 0, 0), float3(-1, 0, 0), float3(0, 1, 0), float3(0, -1, 0), float3(0, 0, 1), float3(0, 0, -1)
};
static const float3 FACE_UP[6] =
{
    float3(0, 1, 0), float3(0, 1, 0), float3(0, 0, -1), float3(0, 0, 1), float3(0, 1, 0), float3(0, 1, 0)
};

uint VisibleCount()
{
    RWByteAddressBuffer countBuf = ResourceDescriptorHeap[VisibleCountUAVIdx];
    return min(countBuf.Load(0), MAX_VISIBLE_LIGHTS);
}

PointLightFrame LoadFrame()
{
    StructuredBuffer<PointLightFrame> frames = ResourceDescriptorHeap[FrameParamsIdx];
    return frames[0];
}

uint ScoreBin(float score)
{
    // half-octave bins from 2^-16 to 2^16
    return (uint)clamp((int)floor(log2(score) * 2.0) + 32, 0, (int)SCORE_BINS - 1);
}

float RadiusPixels(PointLightData light, PointLightFrame frame)
{
    return light.Range * frame.PixelsPerMetre / max(length(light.Position), 0.5 * light.Range);
}

// The light's centre is behind what the camera sees there, by more than a margin (a cage bar or a thin post in
// front of the flame does not count): every one of five taps around it is nearer.
bool CentreHidden(float3 p, float range, PointLightFrame frame)
{
    float4 clip = mul(float4(p, 1.0), frame.CameraRelativeVP);
    if (clip.w <= NEAR_PLANE)
        return false;
    float2 uv = clip.xy / clip.w * float2(0.5, -0.5) + 0.5;
    if (any(uv < 0.0) || any(uv > 1.0))
        return false;

    Texture2D<float> depth = ResourceDescriptorHeap[DepthTexIdx];
    uint w, h;
    depth.GetDimensions(w, h);
    int2 centre = int2(uv * float2(w, h));
    int2 maxCoord = int2(w, h) - 1;
    float behind = clip.w - max(0.5, 0.1 * range);

    static const int2 taps[5] = { int2(0, 0), int2(-4, 0), int2(4, 0), int2(0, -4), int2(0, 4) };
    [unroll]
    for (uint t = 0; t < 5; t++)
    {
        float d = depth.Load(int3(clamp(centre + taps[t], int2(0, 0), maxCoord), 0));
        if (d <= 0.0 || d >= behind) // sky, or nothing nearer than the light
            return false;
    }
    return true;
}

// Bit per cube face whose pyramid (light to its range) is not wholly outside one of the camera's planes
uint VisibleFaces(float3 p, float range, PointLightFrame frame)
{
    uint faces = 0;
    [unroll]
    for (uint f = 0; f < 6; f++)
    {
        float3 forward = FACE_FORWARD[f];
        float3 a = FACE_UP[f];
        float3 b = cross(forward, a);
        float3 corners[5] =
        {
            p,
            p + range * (forward + a + b),
            p + range * (forward + a - b),
            p + range * (forward - a + b),
            p + range * (forward - a - b),
        };

        bool culled = false;
        for (uint k = 0; k < 5 && !culled; k++)
        {
            float4 plane = frame.Planes[k];
            bool allOut = true;
            for (uint j = 0; j < 5; j++)
            {
                if (dot(plane.xyz, corners[j]) + plane.w >= 0.0)
                {
                    allOut = false;
                    break;
                }
            }
            culled = allOut;
        }
        if (!culled)
            faces |= 1u << f;
    }
    return faces;
}

// 512² over 512 px of on-screen radius, 256² over 256 px, else 128²: a face texel then covers about a pixel
// a third of the range from the light. The thresholds move TIER_BAND toward the tier the light has.
uint PreferredTier(float radiusPixels, uint stateTier)
{
    int current = (int)stateTier - 1;
    float to512 = 512.0 * (current == 0 ? 1.0 - TIER_BAND : 1.0 + TIER_BAND);
    float to256 = 256.0 * ((current == 0 || current == 1) ? 1.0 - TIER_BAND : 1.0 + TIER_BAND);
    return radiusPixels > to512 ? 0u : radiusPixels > to256 ? 1u : 2u;
}

// Share of the sun the area around the light gets (0..1): the light's centre and four points 0.75 m around it in
// the sun's cascades. Five points, so a lamp's own cage over the flame does not count as shade. 1 without cascades.
float SunVisibility(float3 worldPos)
{
    if (SunShadowMapIdx == 0 || SunCascadeCountIdx == 0)
        return 1.0;

    StructuredBuffer<SunCascade> cascades = ResourceDescriptorHeap[SunCascadesIdx];
    Texture2DArray<float> shadowMap = ResourceDescriptorHeap[SunShadowMapIdx];
    float w, h, slices;
    shadowMap.GetDimensions(w, h, slices);

    static const float3 offsets[5] =
    {
        float3(0, 0, 0), float3(0.75, 0, 0), float3(-0.75, 0, 0), float3(0, 0, 0.75), float3(0, 0, -0.75)
    };
    float lit = 0.0;
    [unroll]
    for (uint t = 0; t < 5; t++)
    {
        float3 p = worldPos + offsets[t];
        float sample = 1.0; // past the last cascade: sunlit
        for (uint c = 0; c < SunCascadeCountIdx; c++)
        {
            float4 lsp = mul(float4(p, 1.0), cascades[c].VP);
            float3 ndc = lsp.xyz / lsp.w;
            if (any(abs(ndc.xy) > 1.0) || ndc.z < 0.0 || ndc.z > 1.0)
                continue;
            float2 uv = ndc.xy * float2(0.5, -0.5) + 0.5;
            float stored = shadowMap.Load(int4(int2(uv * float2(w, h)), c, 0));
            // 25 cm toward the sun, in this cascade's depth units: a cascade reaches far back toward the sun (to
            // catch distant casters), so a fixed depth bias would be metres and let a roof over the lamp through
            float4x4 vp = cascades[c].VP;
            float bias = 0.25 * length(float3(vp._13, vp._23, vp._33));
            sample = ndc.z - bias <= stored ? 1.0 : 0.0;
            break;
        }
        lit += sample;
    }
    return lit / 5.0;
}

// How much the light outshines the daylight around it: 0 when the sun and sky drown it, 1 when the light rules its
// surroundings. All in lux: the light's a third of its range out (candela / d², its window leaves 0.975 there), the
// sun's on a surface facing it where the cascades see it, the sky's on level ground. Both only as far as they reach
// the light past the terrain (sky_visibility.hlsli): in a cave at noon the open sky's 15000 lux drowned every lamp,
// and none got a shadow.
float SunFactor(PointLightData light, PointLightFrame frame)
{
    TerrainCover terrain;
    terrain.Heightmap = frame.TerrainHeightmap;
    terrain.MaxHeight = frame.TerrainMaxHeight;
    terrain.Origin = frame.TerrainOrigin;
    terrain.Size = frame.TerrainSize;
    float3 worldPos = light.Position + frame.CameraPosition;
    float tolerance = TerrainTolerance(length(light.Position));
    float own = light.Score * 0.975 * 9.0 / max(light.Range * light.Range, 1e-4);   // the cull's luminance × base candela
    float sun = frame.SunLight > 0.0
        ? frame.SunLight * SunVisibility(worldPos) * TerrainOpenAt(terrain, TerrainLinearClamp, worldPos, float3(0, 1, 0), tolerance) : 0.0;
    float sky = frame.Ambient * SkyVisibilityAt(terrain, TerrainLinearClamp, worldPos, float3(0, 1, 0), tolerance);
    float dominance = own / max(own + sun + sky, 1e-6);
    return smoothstep(0.25, 0.5, dominance);
}

// ── CSScore ──────────────────────────────────────────────────────────────────
[numthreads(64, 1, 1)]
void CSScore(uint3 dtid : SV_DispatchThreadID)
{
    uint i = dtid.x;
    if (i >= VisibleCount())
        return;

    PointLightFrame frame = LoadFrame();
    RWStructuredBuffer<PointLightData> visible = ResourceDescriptorHeap[VisibleUAVIdx];
    PointLightData light = visible[i];

    float score = 0.0;
    if (light.Slot < StateCountIdx)
    {
        RWStructuredBuffer<LightShadowState> states = ResourceDescriptorHeap[StateUAVIdx];
        LightShadowState state = states[light.Slot];
        if (state.Seen + 1 != frame.Frame)
        {
            // Not in view last frame: whatever it had is gone
            state.Fade = 0.0;
            state.Tier = 0;
            state.SunFactor = 0.0;
        }
        state.Seen = frame.Frame;

        // Below the size limit the light keeps its last sun factor, for its shadow to fade out with
        if (RadiusPixels(light, frame) >= MIN_RADIUS_PIXELS && light.Range > 2.0 * NEAR_PLANE)
        {
            state.SunFactor = SunFactor(light, frame);
            float near = max(length(light.Position), 0.25 * light.Range);
            score = light.Score * light.Range * light.Range / (near * near) * state.SunFactor;
            if (state.Fade > 0.0)
                score *= STICKY_BONUS;
            if (score > 0.0 && CentreHidden(light.Position, light.Range, frame))
                score *= HIDDEN_FACTOR;
        }
        states[light.Slot] = state;
    }

    visible[i].Score = score;
    if (score > 0.0)
    {
        RWStructuredBuffer<uint> histogram = ResourceDescriptorHeap[HistogramUAVIdx];
        InterlockedAdd(histogram[ScoreBin(score)], 1);
    }
}

// ── CSGather ─────────────────────────────────────────────────────────────────
groupshared uint g_Threshold;

[numthreads(64, 1, 1)]
void CSGather(uint3 dtid : SV_DispatchThreadID, uint gi : SV_GroupIndex)
{
    PointLightFrame frame = LoadFrame();

    // Lowest bin that, with every bin above it, holds the budget (0 when fewer lights scored)
    if (gi == 0)
    {
        RWStructuredBuffer<uint> histogram = ResourceDescriptorHeap[HistogramUAVIdx];
        uint budget = max(frame.ShadowBudget, 1u);
        uint sum = 0, threshold = 0;
        for (int b = (int)SCORE_BINS - 1; b >= 0; b--)
        {
            sum += histogram[b];
            if (sum >= budget)
            {
                threshold = (uint)b;
                break;
            }
        }
        g_Threshold = threshold;
    }
    GroupMemoryBarrierWithGroupSync();

    uint i = dtid.x;
    if (i >= VisibleCount())
        return;

    RWStructuredBuffer<PointLightData> visible = ResourceDescriptorHeap[VisibleUAVIdx];
    PointLightData light = visible[i];
    if (light.Slot >= StateCountIdx)
        return;

    RWStructuredBuffer<LightShadowState> states = ResourceDescriptorHeap[StateUAVIdx];
    LightShadowState state = states[light.Slot];

    bool ranked = light.Score > 0.0 && ScoreBin(light.Score) >= g_Threshold;
    if (!ranked && state.Fade <= 0.0)
        return;

    uint faces = VisibleFaces(light.Position, light.Range, frame);
    if (faces == 0)
    {
        if (state.Fade > 0.0)
        {
            state.Fade = 0.0;   // nothing of it on screen: nothing to fade
            state.Tier = 0;
            states[light.Slot] = state;
        }
        return;
    }

    uint tier = PreferredTier(RadiusPixels(light, frame), state.Tier);

    RWByteAddressBuffer countBuf = ResourceDescriptorHeap[CandidateCountUAVIdx];
    uint index;
    countBuf.InterlockedAdd(0, 1, index);
    if (index < MAX_CANDIDATES)
    {
        RWStructuredBuffer<uint4> candidates = ResourceDescriptorHeap[CandidatesUAVIdx];
        candidates[index] = uint4(asuint(light.Score), i, faces | (tier << 8) | (ranked ? CAND_RANKED : 0u), light.Slot);
    }
}

// ── CSSelect ─────────────────────────────────────────────────────────────────
groupshared uint4 g_Candidates[MAX_CANDIDATES];
groupshared ShadowPick g_Picks[MAX_SHADOW_LIGHTS];

// Counts faceCount more faces in at the preferred tier or a smaller one, if the atlas holds them
bool TryFit(uint faceCount, uint preferred, inout uint tierFaces[3], out uint tier)
{
    for (tier = preferred; tier < 3; tier++)
    {
        uint t0 = tierFaces[0] + (tier == 0 ? faceCount : 0);
        uint t1 = tierFaces[1] + (tier == 1 ? faceCount : 0);
        uint t2 = tierFaces[2] + (tier == 2 ? faceCount : 0);
        uint slots = t0 + (t1 + 3) / 4 + (t2 + 15) / 16;
        if (slots <= SLOT_COUNT && t0 + t1 + t2 <= LOCAL_SHADOW_MAX_VIEWS)
        {
            tierFaces[tier] += faceCount;
            return true;
        }
    }
    tier = 0;
    return false;
}

[numthreads(512, 1, 1)]
void CSSelect(uint gi : SV_GroupIndex)
{
    RWByteAddressBuffer candidateCount = ResourceDescriptorHeap[CandidateCountUAVIdx];
    uint n = min(candidateCount.Load(0), MAX_CANDIDATES);
    RWStructuredBuffer<uint4> candidates = ResourceDescriptorHeap[CandidatesUAVIdx];

    uint size = 2;
    while (size < n) size <<= 1;

    for (uint j = gi; j < size; j += 512)
        g_Candidates[j] = j < n ? candidates[j] : uint4(0, 0xFFFFFFFFu, 0, 0xFFFFFFFFu);
    GroupMemoryBarrierWithGroupSync();

    // Bitonic sort: highest score first, ties by registry slot (no dependence on the cull's append order)
    for (uint k = 2; k <= size; k <<= 1)
    {
        for (uint s = k >> 1; s > 0; s >>= 1)
        {
            for (uint t = gi; t < size; t += 512)
            {
                uint partner = t ^ s;
                if (partner > t)
                {
                    uint4 a = g_Candidates[t];
                    uint4 b = g_Candidates[partner];
                    bool aFirst = a.x > b.x || (a.x == b.x && a.w < b.w);
                    bool up = (t & k) == 0;
                    if (aFirst != up)
                    {
                        g_Candidates[t] = b;
                        g_Candidates[partner] = a;
                    }
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }

    if (gi != 0)
        return;

    // ── One thread from here: fit, fade, pack ──
    PointLightFrame frame = LoadFrame();
    RWStructuredBuffer<LightShadowState> states = ResourceDescriptorHeap[StateUAVIdx];
    RWStructuredBuffer<PointLightData> visible = ResourceDescriptorHeap[VisibleUAVIdx];

    uint tierFaces[3] = { 0, 0, 0 };
    uint pickCount = 0, wanted = 0;
    uint budget = min(frame.ShadowBudget, MAX_SHADOW_LIGHTS);

    // A. In rank order, the budget's lights, a tier lower when the atlas is short
    for (uint c = 0; c < n && wanted < budget; c++)
    {
        uint4 cand = g_Candidates[c];
        if ((cand.z & CAND_RANKED) == 0)
            continue;
        uint faces = cand.z & 0x3Fu, tier;
        if (!TryFit(countbits(faces), (cand.z >> 8) & 3u, tierFaces, tier))
            continue;
        ShadowPick pick;
        pick.VisibleIndex = cand.y;
        pick.Slot = cand.w;
        pick.Packed = faces | (tier << 8) | (1u << 16);
        pick.Fade = 0.0;
        g_Picks[pickCount++] = pick;
        g_Candidates[c].z = cand.z | CAND_PICKED;
        wanted++;
    }

    // B. Lights that had a shadow and lost their place keep tiles (their tier or smaller) while it fades out
    for (uint d = 0; d < n && pickCount < MAX_SHADOW_LIGHTS; d++)
    {
        uint4 cand = g_Candidates[d];
        if ((cand.z & CAND_PICKED) != 0)
            continue;
        LightShadowState state = states[cand.w];
        if (state.Fade <= 0.0)
            continue;
        uint faces = cand.z & 0x3Fu, tier;
        if (state.Fade - frame.FadeStep <= 0.0
            || !TryFit(countbits(faces), state.Tier > 0 ? state.Tier - 1 : 0, tierFaces, tier))
        {
            state.Fade = 0.0;
            state.Tier = 0;
            states[cand.w] = state;
            continue;
        }
        ShadowPick pick;
        pick.VisibleIndex = cand.y;
        pick.Slot = cand.w;
        pick.Packed = faces | (tier << 8);
        pick.Fade = 0.0;
        g_Picks[pickCount++] = pick;
    }

    // Fades: picked lights rise, dropped ones fall. The shadow's strength is the fade × the sun factor, so shadows
    // also come and go smoothly with the weather and the time of day.
    float strength[MAX_SHADOW_LIGHTS];
    for (uint p = 0; p < pickCount; p++)
    {
        ShadowPick pick = g_Picks[p];
        LightShadowState state = states[pick.Slot];
        bool isWanted = (pick.Packed & (1u << 16)) != 0;
        state.Fade = isWanted ? min(1.0, state.Fade + frame.FadeStep) : max(0.0, state.Fade - frame.FadeStep);
        state.Tier = ((pick.Packed >> 8) & 3u) + 1;
        states[pick.Slot] = state;
        g_Picks[p].Fade = state.Fade;
        strength[p] = state.Fade * state.SunFactor;
    }

    // Tiles, largest faces first: 512s take whole slots, then 256s four to a slot, 128s sixteen
    uint firstSlot[3];
    firstSlot[0] = 0;
    firstSlot[1] = tierFaces[0];
    firstSlot[2] = firstSlot[1] + (tierFaces[1] + 3) / 4;
    uint slotsUsed = firstSlot[2] + (tierFaces[2] + 15) / 16;

    RWStructuredBuffer<ShadowPick> picks = ResourceDescriptorHeap[PicksUAVIdx];
    RWStructuredBuffer<LocalShadowLight> shadowLights = ResourceDescriptorHeap[ShadowLightsUAVIdx];
    for (uint q = 0; q < pickCount; q++)
    {
        ShadowPick pick = g_Picks[q];
        PointLightData light = visible[pick.VisibleIndex];
        LocalShadowLight shadowLight;
        shadowLight.Position = light.Position + frame.CameraPosition;
        shadowLight.Range = light.Range;
        [unroll]
        for (uint f = 0; f < 6; f++)
            shadowLight.FaceView[f] = LOCAL_SHADOW_NO_VIEW;
        shadowLight.Strength = strength[q];
        shadowLight._pad = 0;
        shadowLights[q] = shadowLight;
        picks[q] = pick;
        visible[pick.VisibleIndex].ShadowLight = q;
    }

    RWStructuredBuffer<uint4> faceList = ResourceDescriptorHeap[FacesUAVIdx];
    uint views = 0;
    uint nextInTier[3] = { 0, 0, 0 };
    for (uint tier = 0; tier < 3; tier++)
    {
        for (uint r = 0; r < pickCount; r++)
        {
            uint packed = g_Picks[r].Packed;
            if (((packed >> 8) & 3u) != tier)
                continue;
            for (uint f = 0; f < 6; f++)
            {
                if ((packed & (1u << f)) == 0)
                    continue;
                faceList[views] = uint4(r, f, tier, nextInTier[tier]++);
                shadowLights[r].FaceView[f] = views;
                views++;
            }
        }
    }

    RWStructuredBuffer<uint> counts = ResourceDescriptorHeap[CountsUAVIdx];
    counts[COUNT_LIGHTS] = pickCount;
    counts[COUNT_VIEWS] = views;
    counts[COUNT_SLOTS] = slotsUsed;
    counts[COUNT_FACES_TIER0 + 0] = tierFaces[0];
    counts[COUNT_FACES_TIER0 + 1] = tierFaces[1];
    counts[COUNT_FACES_TIER0 + 2] = tierFaces[2];
    counts[COUNT_SLOT_TIER0 + 0] = firstSlot[0];
    counts[COUNT_SLOT_TIER0 + 1] = firstSlot[1];
    counts[COUNT_SLOT_TIER0 + 2] = firstSlot[2];
    counts[COUNT_CANDIDATES] = n;
}

// ── CSBuildViews ─────────────────────────────────────────────────────────────
// Row-vector matrices, as System.Numerics builds them (LocalShadows.cs before 2026-10-16 phase 3)
float4x4 LookAtLH(float3 eye, float3 forward, float3 up)
{
    float3 z = forward;
    float3 x = normalize(cross(up, z));
    float3 y = cross(z, x);
    return float4x4(
        x.x, y.x, z.x, 0,
        x.y, y.y, z.y, 0,
        x.z, y.z, z.z, 0,
        -dot(x, eye), -dot(y, eye), -dot(z, eye), 1);
}

float4x4 PerspectiveLH(float tanHalf, float n, float f)
{
    float s = 1.0 / tanHalf;
    float q = f / (f - n);
    return float4x4(
        s, 0, 0, 0,
        0, s, 0, 0,
        0, 0, q, 1,
        0, 0, -n * q, 0);
}

float4 NormalisePlane(float4 p)
{
    return p / length(p.xyz);
}

[numthreads(64, 1, 1)]
void CSBuildViews(uint3 dtid : SV_DispatchThreadID)
{
    RWStructuredBuffer<uint> counts = ResourceDescriptorHeap[CountsUAVIdx];
    uint v = dtid.x;
    if (v >= counts[COUNT_VIEWS])
        return;

    RWStructuredBuffer<uint4> faceList = ResourceDescriptorHeap[FacesUAVIdx];
    RWStructuredBuffer<ShadowPick> picks = ResourceDescriptorHeap[PicksUAVIdx];
    RWStructuredBuffer<PointLightData> visible = ResourceDescriptorHeap[VisibleUAVIdx];
    PointLightFrame frame = LoadFrame();

    uint4 entry = faceList[v];
    uint pickIndex = entry.x, face = entry.y, tier = entry.z, k = entry.w;
    PointLightData light = visible[picks[pickIndex].VisibleIndex];

    uint size = 512u >> tier;
    uint perSlot = 1u << (2u * tier);
    uint perRow = 512u / size;
    uint slot = counts[COUNT_SLOT_TIER0 + tier] + k / perSlot;
    uint sub = k % perSlot;
    uint x = (slot % 4u) * 512u + (sub % perRow) * size;
    uint y = ((slot / 4u) % 4u) * 512u + (sub / perRow) * size;

    float tanHalf = size / (size - 2.0 * GUARD_TEXELS);
    float4x4 faceView = LookAtLH(light.Position, FACE_FORWARD[face], FACE_UP[face]);
    float4x4 sampleVP = mul(faceView, PerspectiveLH(tanHalf, NEAR_PLANE, light.Range));
    float3 c = frame.CameraPosition;
    float4x4 toCameraRelative = float4x4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -c.x, -c.y, -c.z, 1);
    float4x4 vp = mul(toCameraRelative, sampleVP);

    LocalShadowView view;
    view.VP = vp;
    view.SampleVP = sampleVP;
    view.Tile = float4(size, size, x, y) / LOCAL_SHADOW_PAGE_SIZE;
    view.Page = slot / 16u;
    view.Light = pickIndex;
    view.TexelPerMetre = 2.0 * tanHalf / size;
    view._pad = 0;

    // Frustum planes, normals out (Frustum.cs's convention), from the absolute VP's columns
    float4 c0 = float4(vp[0][0], vp[1][0], vp[2][0], vp[3][0]);
    float4 c1 = float4(vp[0][1], vp[1][1], vp[2][1], vp[3][1]);
    float4 c2 = float4(vp[0][2], vp[1][2], vp[2][2], vp[3][2]);
    float4 c3 = float4(vp[0][3], vp[1][3], vp[2][3], vp[3][3]);
    view.Planes[0] = NormalisePlane(-c2);       // near (z >= 0)
    view.Planes[1] = NormalisePlane(c2 - c3);   // far  (z <= w)
    view.Planes[2] = NormalisePlane(-c3 - c0);  // left
    view.Planes[3] = NormalisePlane(c0 - c3);   // right
    view.Planes[4] = NormalisePlane(c1 - c3);   // top
    view.Planes[5] = NormalisePlane(-c3 - c1);  // bottom

    RWStructuredBuffer<LocalShadowView> views = ResourceDescriptorHeap[ViewsUAVIdx];
    views[v] = view;
}
