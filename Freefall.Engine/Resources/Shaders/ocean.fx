cbuffer PushConstants : register(b3)
{
    // Slots 0-1: set by the forward pass for every forward batch
    uint ShadowMapIdx;              // 0: Shadow cascade array SRV
    uint _reserved1;                // 1: CompositeSnapshot SRV (taken from OceanData instead)
    // Slots 2-3: PER-DRAW (command signature writes these)
    uint MeshPartId;                // 2: Index into MeshRegistry
    uint InstanceBaseOffset;        // 3: Base offset for instance ID (per-command)
    // Slots 4-8: PER-BATCH (set before ExecuteIndirect)
    uint DescriptorBufIdx;          // 4: StructuredBuffer<InstanceDescriptor>
    uint SortedIndicesIdx;          // 5: StructuredBuffer<uint> - sorted draw order indices
    uint MeshRegistryIdx;           // 6: StructuredBuffer<MeshPartEntry>
    uint MaterialsIdx;              // 7: Index to materials buffer
    uint GlobalTransformBufferIdx;  // 8: Index to global TransformBuffer
    // Slots 9-15: Custom / Reserved
    uint OceanDataIdx;              // 9: Per-instance OceanData buffer
    uint _reserved10;
    uint _reserved11;
    uint _reserved12;
    uint _reserved13;
    uint _reserved14;
    uint _reserved15;
    // Slot 16: Debug
    uint DebugMode;                 // 16: Debug visualization mode
    uint _reserved17;
    uint _reserved18;
    uint _reserved19;
    // Slots 20-21: Shadow pass
    uint ExpansionBufferIdx;        // 20: SRV: expansion buffer
    uint CascadeBufferSRVIdx;       // 21: SRV: StructuredBuffer<CascadeData>
};

#include "common.fx"
#include "sky_common.fx"
#include "water_common.fx"
// @RenderState(RenderTargets=1)



struct OceanData
{
    float WaveTime;
    float Choppiness;
    float FoamDecayRate;
    float FoamThreshold;
    float3 OceanColor;
    uint SeaMaskSRV;        // where the open sea is (C# SeaMask), 0 = everywhere. Was padding: an older build writes 0
    float3 DeepColor;
    uint WindBands;         // bit per band carrying the wind sea (C# FollowWind). Was padding: an older build writes 0
    float3 SunDirection;
    float SunIntensity;
    float3 SunColor;
    uint WhitecapBand;      // the band whose foam marks the whitecaps + 1 (C# WhitecapBand); 0 = the coarsest wind band
    uint DisplacementSRV;
    uint SlopeSRV;
    uint NumBands;
    uint TileScalesSRV;      // bindless SRV to float[] buffer
    float DisplacementAtten;
    float NormalAtten;
    uint HeightmapSRV;      // bindless SRV to terrain heightmap
    float ShoreDepth;       // water depth (meters) for full wave strength
    float MaxTerrainHeight; // terrain MaxHeight scale
    float OceanPlaneY;      // ocean entity world Y
    float TerrainBaseY;     // terrain entity world Y
    float TerrainOriginX;   // terrain world origin X
    float TerrainOriginZ;   // terrain world origin Z
    float TerrainSizeX;     // terrain size X
    float TerrainSizeZ;     // terrain size Z
    float ShoreMinWave;     // minimum displacement at shore (0-1)
    uint DepthGBufferSRV;   // bindless SRV to linear depth gbuffer (PS shore effects)
    uint CompositeSRV;      // bindless SRV to composite buffer (terrain show-through)
    float ShoreFadeDepth;   // linear depth range for PS soft intersection (meters)
    float3 ShallowColor;    // shallow water tint color
    float RefractionStrength; // how much normal distorts terrain show-through
    uint NoiseSRV;              // bindless SRV for noise texture (Perlin+Worley)
    float2 InvViewportSize;     // 1.0 / viewport dimensions (replaces GetDimensions)
    float _shorePad;            // was ShoreFoamDepth (the clock-driven swash bands, gone with the surf 2026-10-19)
    float ShoreFoam;            // shore foam amount (0 = none)
    float ShoreSurge;           // the most the swash runs up the beach (m over the still sea, SwashLevel)
    uint HiZSRV;                // this view's Hi-Z pyramid of the opaque depth (0 = no occlusion test)
    uint ShoreWarpSRV;          // the surf (C# ShoreWarp): Texture2DArray RGBA32F per warped band: Δu.xz, amplitude, c/c0; 0 = none
    uint ShoreWarpJacobianSRV;  // its ∂u/∂x (RGBA16F: ∂ux/∂x, ∂ux/∂z, ∂uz/∂x, ∂uz/∂z)
    uint WarpSlots;             // band + 1 of warp slice 0 (bits 0-7) and slice 1 (bits 8-15); 0 = none
    float _warpPad;
    float4 WarpDirs;            // the deep heading (world xz) of slice 0 (xy) and slice 1 (zw)
};

// Tessellation params
static const float TessMinFactor = 1.0;
static const float TessMaxFactor = 48.0;
static const float TargetPixelsPerEdge = 10.0; // tessellate until edges are ~10 pixels

// Screen-space edge length tessellation heuristic
float ScreenSpaceEdgeFactor(float3 p0, float3 p1)
{
    // Project both endpoints to clip space
    float4 clip0 = mul(mul(float4(p0, 1.0), View), Projection);
    float4 clip1 = mul(mul(float4(p1, 1.0), View), Projection);

    // NDC [-1,1] → screen pixels
    // Use Projection[0][0] for horizontal FOV scale
    float2 screen0 = clip0.xy / clip0.w;
    float2 screen1 = clip1.xy / clip1.w;

    // Get viewport size from projection matrix
    // ViewportWidth ≈ 2 / Projection[0][0], but we just need relative pixel count
    float2 pixelScale = float2(abs(Projection[0][0]), abs(Projection[1][1])) * 512.0;
    float2 diff = (screen1 - screen0) * pixelScale;
    float edgePixels = length(diff);

    return clamp(edgePixels / TargetPixelsPerEdge, TessMinFactor, TessMaxFactor);
}

// How far the waves can move a point of the grid, in every direction (metres): patch bounds are widened by it
static const float PatchDisplacementBound = 20.0;

// Whether a patch can be left out (tessellation factors 0): its box, widened by the waves, lies outside the
// view, or behind the opaque scene in the Hi-Z pyramid. This used to test only "entirely behind the camera",
// so every patch in front of the camera plane was tessellated and displaced, on screen or not, hidden behind
// terrain or not (0.74 ms with no water in view on Eldermoor, 2026-10-15).
bool CullTriangle(float3 p0, float3 p1, float3 p2, uint hiZSrv)
{
    float3 minP = min(min(p0, p1), p2) - PatchDisplacementBound;
    float3 maxP = max(max(p0, p1), p2) + PatchDisplacementBound;

    // Clip-space corners; frustum planes as half-spaces (valid for corners behind the camera too)
    uint outside = 0x1F;
    float2 ndcMin = 1e30, ndcMax = -1e30;
    float nearestW = 1e30;
    bool reachesBehind = false;
    [unroll]
    for (uint i = 0; i < 8; i++)
    {
        float3 c = float3((i & 1) ? maxP.x : minP.x, (i & 2) ? maxP.y : minP.y, (i & 4) ? maxP.z : minP.z);
        float4 clip = mul(mul(float4(c, 1.0), View), Projection);

        uint beyond = 0;
        if (clip.x < -clip.w) beyond |= 1;
        if (clip.x >  clip.w) beyond |= 2;
        if (clip.y < -clip.w) beyond |= 4;
        if (clip.y >  clip.w) beyond |= 8;
        if (clip.w <= 0.0)    beyond |= 16;
        outside &= beyond;

        if (clip.w <= 0.01) reachesBehind = true;
        else
        {
            float2 ndc = clip.xy / clip.w;
            ndcMin = min(ndcMin, ndc);
            ndcMax = max(ndcMax, ndc);
            nearestW = min(nearestW, clip.w);
        }
    }

    if (outside != 0) return true;              // all corners beyond one plane of the view
    if (hiZSrv == 0 || reachesBehind) return false;

    // Hi-Z: linear view depth (clip w), max over each texel's footprint, sky = FLT_MAX (never occludes).
    // At the mip where the box's screen rectangle spans at most one texel, it touches at most 2x2 texels;
    // it is hidden if even its nearest point is behind the farthest opaque depth there.
    ndcMin = max(ndcMin, -1.0);
    ndcMax = min(ndcMax, 1.0);
    float2 uvMin = float2(ndcMin.x, -ndcMax.y) * 0.5 + 0.5;
    float2 uvMax = float2(ndcMax.x, -ndcMin.y) * 0.5 + 0.5;

    Texture2D<float> hiZ = ResourceDescriptorHeap[hiZSrv];
    float w, h, levels;
    hiZ.GetDimensions(0, w, h, levels);
    float2 extent = (uvMax - uvMin) * float2(w, h);
    uint mip = (uint)min(ceil(log2(max(max(extent.x, extent.y), 1.0))), levels - 1.0);

    float mw, mh, unused;
    hiZ.GetDimensions(mip, mw, mh, unused);
    int2 maxCoord = int2(mw, mh) - 1;
    int2 t0 = clamp(int2(uvMin * float2(mw, mh)), int2(0, 0), maxCoord);
    int2 t1 = clamp(int2(uvMax * float2(mw, mh)), int2(0, 0), maxCoord);

    float occluder = max(max(hiZ.Load(int3(t0.x, t0.y, mip)), hiZ.Load(int3(t1.x, t0.y, mip))),
                         max(hiZ.Load(int3(t0.x, t1.y, mip)), hiZ.Load(int3(t1.x, t1.y, mip))));
    return nearestW > occluder;
}

SamplerState OceanSampler : register(s0);

// The swash's cycle (SwashLevel): the share of a wave's period its bore runs up the beach, the rest it drains back
// (0.5 would be symmetric; 0.25 felt too abrupt up and too sluggish back)
static const float SurgeRise = 0.38;

// The surf's foam (PS): broken water breaks up along the sea's own finest crests (the bands the surf does not carry,
// their squeeze over their slopes, so it shows at any wind): its opacity where they are not squeezed and how much
// their crests raise it; and the bubbles all through the surf zone once its waves have broken
#define SURF_VEIL 0.15
#define SURF_CRESTS 1.2
#define SURF_BUBBLES 0.06
// The roller's opacity where the crests are not squeezed: nearly solid
#define SURF_ROLLER 0.6

// Light through the crests (PS): full at this height (m) of the wind sea above the mean, and the share of the light on a
// crest that scatters through it to the eye
// Whitecaps (PS): the opacity of the coarsest wind band's foam where the finer bands' crests are not squeezed, and how
// much their squeeze (1 - J, summed) raises it
#define WHITECAP_VEIL 0.08
#define WHITECAP_CRESTS 3.0
#define WHITECAP_THIN 0.03
#define WHITECAP_RAMP 0.55

#define CREST_GLOW_HEIGHT 3.0
#define CREST_GLOW 0.25
// Bubbles under a whitecap (PS): the share of the light on the water that they send back up, per unit of whitecap foam
#define AERATION 0.1

// Per-band UV rotation to decorrelate cascade tiling patterns
static const float BandAngles[8] = { 0.0, 0.297, 0.645, 0.925, 1.17, 1.42, 1.73, 2.05 };

float2 RotateUV(float2 uv, float angle)
{
    float s, c;
    sincos(angle, s, c);
    return float2(uv.x * c - uv.y * s, uv.x * s + uv.y * c);
}

// How much of a band's waves the shore lets stand, for the bands the surf does not carry (below): waves feel the bottom
// where the water is shallower than about half their length, so the long waves flatten toward the beach while the
// ripples run on over a shallow shelf. They stand in full from an eighth of the band's tile deep (17 m band: 2 m, 64 m:
// 8 m, 250 m: 31 m, the 1000 m swell: 125 m), and everywhere from ShoreDepth. (A wind band holds from a quarter of the
// next finer wind band's tile up to its own: 250 m band 16-250 m, 64 m band 4.25-16 m.) Until 2026-10-18 the shore damped
// every band alike, ripples too, and only the heights: the PS shaded a flattened shelf as the full open-sea chop.
//   shoreAtten: the shore's damping of the longest waves (ShoreMinWave..1), depth: still-water depth (m)
float BandShoreAtten(float shoreAtten, float depth, float tileScale)
{
    return lerp(shoreAtten, 1.0, saturate(depth * tileScale * 8.0));
}

// ── The surf (C# ShoreWarp): the open sea's own waves carried to the beach ──
// Near the coast a warped band is sampled not where a point is but where its wave was out at sea, u = x + Δu, baked
// from the seabed by the wave's travel time: its crests turn toward the beach, crowd together and grow as they shoal,
// and a lee shore stays calm. What comes in is the sea's own field, its sets as sets.
struct BandWarp
{
    float2 u;          // where to sample the band (world xz, in the open sea's frame)
    float amp;         // the wave's height here over the open sea's: shoaling, the rays' convergence, the island's shadow
    float speedRatio;  // the wave's speed here over its deep-water speed
    float2x2 J;        // ∂u/∂x: rows ux, uz; columns ∂/∂x, ∂/∂z
    float2 turn;       // cos, sin of the turn from the deep heading to the local one (sideways displacement, breakers)
    float2 heading;    // the local heading (world xz unit vector)
};

// A band's slice in the warp, 0xFF if the surf does not carry it
uint WarpSlice(OceanData ocean, uint band)
{
    if (ocean.ShoreWarpSRV == 0) return 0xFF;
    if ((ocean.WarpSlots & 0xFFu) == band + 1) return 0;
    if (((ocean.WarpSlots >> 8) & 0xFFu) == band + 1) return 1;
    return 0xFF;
}

//   terrainUV: the point's terrain UV (outside the terrain the border's, where the warp is none)
BandWarp SampleBandWarp(OceanData ocean, uint slice, float2 worldXZ, float2 terrainUV)
{
    Texture2DArray<float4> warpTex = ResourceDescriptorHeap[ocean.ShoreWarpSRV];
    Texture2DArray<float4> jacobianTex = ResourceDescriptorHeap[ocean.ShoreWarpJacobianSRV];
    float w, h, slices;
    warpTex.GetDimensions(w, h, slices);
    // The bake's cell i lies at UV i / (n - 1), its texel's centre at (i + 0.5) / n
    float2 uv = (saturate(terrainUV) * (float2(w, h) - 1.0) + 0.5) / float2(w, h);
    float4 a = warpTex.SampleLevel(OceanSampler, float3(uv, slice), 0);
    float4 j = jacobianTex.SampleLevel(OceanSampler, float3(uv, slice), 0);

    BandWarp b;
    b.u = worldXZ + a.xy;
    b.amp = a.z;
    b.speedRatio = a.w;
    b.J = float2x2(j.x, j.y, j.z, j.w);
    // The local heading: the gradient of the deep phase d·u, Jᵀd
    float2 d = slice == 0 ? ocean.WarpDirs.xy : ocean.WarpDirs.zw;
    float2 r = mul(d, b.J);
    r *= rsqrt(max(dot(r, r), 1e-12));
    b.turn = float2(dot(d, r), d.x * r.y - d.y * r.x);
    b.heading = r;
    return b;
}

// The surf's wave at a point. A carried band's height and its sideways displacement along the local heading are in
// quadrature (η = A·cos φ, D = −λ·A·sin φ, λ the choppiness), which gives the wave's local amplitude and phase with no
// history: so each wave knows on its own whether it has broken, and where its crest and its trail are.
// Per carried band (a warp slice), so the two blend by their energy: following whichever was stronger (until
// 2026-10-19) jumped the phase where they were about equal, a zig-zag in the swash's edge on the sand
struct SurfWave
{
    float eta;             // the carried bands' height (m)
    float2 envelope;       // each carried band's local amplitude A (m)
    float2 phase;          // its phase: 0 on the crest, > 0 ahead of it (still to come), < 0 behind it (passed)
    float4 heading;        // its local heading (world xz): slice 0 in xy, slice 1 in zw
};

//   d: a carried band's displacement in world xz (turned to the local heading) and height, both × its amplitude
void AddSurfWave(inout SurfWave s, OceanData ocean, float3 d, float2 heading, uint slice)
{
    float quadrature = -dot(d.xz, heading) / max(ocean.Choppiness, 0.05);
    float envelope = sqrt(d.y * d.y + quadrature * quadrature);
    float phase = atan2(quadrature, d.y);
    s.eta += d.y;
    if (slice == 0) { s.envelope.x = envelope; s.phase.x = phase; s.heading.xy = heading; }
    else            { s.envelope.y = envelope; s.phase.y = phase; s.heading.zw = heading; }
}

// The carried bands' amplitude together, and each band's share of their energy
float SurfEnvelope(SurfWave s) { return length(s.envelope); }
float2 SurfShare(SurfWave s) { return s.envelope * s.envelope / max(dot(s.envelope, s.envelope), 1e-8); }

// The local heading, weighted by the bands' energy
float2 SurfHeading(SurfWave s)
{
    float2 share = SurfShare(s);
    float2 h = s.heading.xy * share.x + s.heading.zw * share.y;
    return dot(h, h) > 1e-8 ? normalize(h) : float2(1.0, 0.0);
}

// Shoaling waves break when they stand about 0.8 × the depth from trough to crest: 0 before, 1 once broken (inside the
// surf zone every wave stays broken: as the water shallows its height keeps up with the depth)
float SurfBreaking(SurfWave s, float depth)
{
    return smoothstep(0.55, 0.85, 2.0 * SurfEnvelope(s) / max(depth, 0.05));
}

// Where in its period the arriving wave is, 0 as its crest arrives (rising with time), and how much the bore is running
// up the beach then (0..1, quick up for SurgeRise of the period, slow down)
float SwashRise(float phase)
{
    float cycle = frac(1.0 - phase / 6.2831853);
    return cycle < SurgeRise ? smoothstep(0.0, SurgeRise, cycle) : 1.0 - smoothstep(SurgeRise, 1.0, cycle);
}
float SwashAdvancing(float phase)
{
    float cycle = frac(1.0 - phase / 6.2831853);
    return max(1.0 - smoothstep(SurgeRise, SurgeRise + 0.2, cycle), smoothstep(0.92, 1.0, cycle));
}

// How far a breaking face leans forward over its trough, per metre of crest height (m/m)
#define SURF_LEAN 1.6
// The swash: within this depth of the waterline (and up the beach) the sea's level is the run-up's
#define SWASH_DEPTH 0.6
// How high the bores run up the beach per metre of the arriving wave's height, and how far the backwash draws the
// water down below the still level, per metre of run-up
#define SWASH_RUNUP 0.35
#define SWASH_DRAWDOWN 0.3

// The water level of the swash (m over the still sea): each arriving crest's bore runs up quickly (SurgeRise of the
// wave's period) as far as its height carries it (at most ShoreSurge) and drains back slowly, in step with the wave
float SwashLevel(SurfWave s, float maxRunUp)
{
    float2 share = SurfShare(s);
    float rise = SwashRise(s.phase.x) * share.x + SwashRise(s.phase.y) * share.y;
    float runUp = min(SWASH_RUNUP * 2.0 * SurfEnvelope(s), maxRunUp);
    return lerp(-SWASH_DRAWDOWN * runUp, runUp, rise);
}

// The broken water a carried band's wave carries at its phase: the roller on its crest and face, the trail behind it
float2 SurfRollerTrail(float phase)
{
    float roller = exp(-(phase - 0.1) * (phase - 0.1) / 0.12);
    float trail = phase < 0.0 ? 0.7 * exp(phase / 0.8) : 0.0;
    return float2(roller, trail);
}

// The water stands no higher over the seabed than a breaking wave allows (a wave breaks at about 0.78 × the depth from
// trough to crest, its crest some 0.6 × the depth over the still level) and sinks no lower than 0.8 × the depth. Soft,
// with what it leaves of the slopes (d limited / d height). The surf's shoaling waves would otherwise rise into walls.
float DepthLimit(float y, float depth, out float slopeScale)
{
    float lim = (y > 0.0 ? 0.6 : 0.8) * max(depth, 0.02);
    float t = tanh(y / lim);
    slopeScale = 1.0 - t * t;
    return lim * t;
}

//   shoreAtten, depth: as BandShoreAtten (1 and a large depth away from the shore)
//   terrainUV, surf: the point's terrain UV for the surf's warp (SampleBandWarp), and whether to warp at all
//   wave: the surf's wave here (its carried bands)
float3 SampleDisplacement(OceanData ocean, float2 worldXZ, float depthAtten, float camDist, float shoreAtten, float depth,
                          float2 terrainUV, bool surf, out SurfWave wave)
{
    wave = (SurfWave)0;
    Texture2DArray<float4> dispTex = ResourceDescriptorHeap[ocean.DisplacementSRV];
    StructuredBuffer<float> tileScales = ResourceDescriptorHeap[ocean.TileScalesSRV];
    float3 totalDisp = 0;

    [loop]
    for (uint i = 0; i < ocean.NumBands; ++i)
    {
        float2 p = worldXZ;
        float amp;
        float stretch = 1.0;
        float2 turn = float2(1.0, 0.0);
        float2 heading = float2(1.0, 0.0);
        uint slice = surf ? WarpSlice(ocean, i) : 0xFF;
        if (slice != 0xFF)
        {
            // The surf: where this wave was out at sea, as high as it has grown; the band's texture is squeezed by the
            // warp, so its mip goes by the largest stretch
            BandWarp w = SampleBandWarp(ocean, slice, worldXZ, terrainUV);
            p = w.u;
            amp = w.amp;
            turn = w.turn;
            heading = w.heading;
            stretch = sqrt(max(w.J[0][0] * w.J[0][0] + w.J[1][0] * w.J[1][0], w.J[0][1] * w.J[0][1] + w.J[1][1] * w.J[1][1]));
        }
        else
            amp = BandShoreAtten(shoreAtten, depth, tileScales[i]);

        float2 uv = RotateUV(p, BandAngles[i]) * tileScales[i];
        float mipLevel = log2(max(1.0, camDist * tileScales[i] * 0.15 * stretch));
        float3 d = dispTex.SampleLevel(OceanSampler, float3(uv, i), mipLevel).xyz;
        // The sideways displacement is in the band's turned frame: turn it back to the world, as the PS does the slopes.
        // Added as it came (until 2026-10-18), it pushed the surface off to one side of the band's waves instead of
        // toward their crests: choppiness sheared the pattern sideways and never sharpened a crest.
        d.xz = RotateUV(d.xz, -BandAngles[i]);
        // A carried wave runs along its local heading
        d.xz = float2(d.x * turn.x - d.z * turn.y, d.x * turn.y + d.z * turn.x);
        totalDisp += d * amp;
        if (slice != 0xFF)
            AddSurfWave(wave, ocean, d * amp, heading, slice);
    }

    return totalDisp * depthAtten;
}

// SampleNormal + SampleFoam merged into PS loop below

// ═══════════════════════════════════════════════════════════════════════
// CONTROL POINT SHADER — pass world position through
// ═══════════════════════════════════════════════════════════════════════

struct ControlPoint
{
    float3 WorldPos : WORLDPOS;
    nointerpolation uint InstanceIdx : TEXCOORD0;
};

ControlPoint VS_Control(uint primitiveVertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
    ControlPoint output;

    StructuredBuffer<InstanceDescriptor> descriptors = ResourceDescriptorHeap[DescriptorBufIdx];
    StructuredBuffer<row_major matrix> globalTransforms = ResourceDescriptorHeap[GlobalTransformBufferIdx];
    StructuredBuffer<uint> sortedIndices = ResourceDescriptorHeap[SortedIndicesIdx];

    // Look up mesh buffer indices from MeshRegistry
    StructuredBuffer<MeshPartEntry> meshRegistry = ResourceDescriptorHeap[MeshRegistryIdx];
    MeshPartEntry part = meshRegistry[MeshPartId];

    StructuredBuffer<uint> indices = ResourceDescriptorHeap[part.IndexBufferIdx];
    uint vertexID = indices[primitiveVertexID + part.BaseIndex];
    StructuredBuffer<float3> positions = ResourceDescriptorHeap[part.PosBufferIdx];

    uint dataPos = InstanceBaseOffset + instanceID;
    uint packedIdx = sortedIndices[dataPos];
    uint idx = packedIdx & 0x7FFFFFFFu;

    InstanceDescriptor desc = descriptors[idx];
    row_major matrix World = globalTransforms[desc.TransformSlot];

    output.WorldPos = mul(float4(positions[vertexID], 1.0f), World).xyz;
    output.InstanceIdx = idx;

    return output;
}

// ═══════════════════════════════════════════════════════════════════════
// HULL SHADER — screen-space adaptive tessellation
// ═══════════════════════════════════════════════════════════════════════

struct HullConstantOutput
{
    float EdgeTess[3] : SV_TessFactor;
    float InsideTess : SV_InsideTessFactor;
};

HullConstantOutput PatchConstantFunc(InputPatch<ControlPoint, 3> patch, uint patchID : SV_PrimitiveID)
{
    HullConstantOutput output;

    float3 p0 = patch[0].WorldPos;
    float3 p1 = patch[1].WorldPos;
    float3 p2 = patch[2].WorldPos;

    // Off screen or hidden behind the opaque scene — skip
    StructuredBuffer<OceanData> oceanData = ResourceDescriptorHeap[OceanDataIdx];
    if (CullTriangle(p0, p1, p2, oceanData[patch[0].InstanceIdx].HiZSRV))
    {
        output.EdgeTess[0] = output.EdgeTess[1] = output.EdgeTess[2] = 0;
        output.InsideTess = 0;
        return output;
    }

    // Screen-space edge length — tessellate proportional to pixel coverage
    // Edge indices: edge[0] = opposite vertex 0 = edge(1,2)
    //               edge[1] = opposite vertex 1 = edge(0,2)
    //               edge[2] = opposite vertex 2 = edge(0,1)
    output.EdgeTess[0] = ScreenSpaceEdgeFactor(p1, p2);
    output.EdgeTess[1] = ScreenSpaceEdgeFactor(p2, p0);
    output.EdgeTess[2] = ScreenSpaceEdgeFactor(p0, p1);

    output.InsideTess = (output.EdgeTess[0] + output.EdgeTess[1] + output.EdgeTess[2]) / 3.0;

    return output;
}

[domain("tri")]
[partitioning("fractional_odd")]
[outputtopology("triangle_cw")]
[outputcontrolpoints(3)]
[patchconstantfunc("PatchConstantFunc")]
ControlPoint HS(InputPatch<ControlPoint, 3> patch, uint cpID : SV_OutputControlPointID)
{
    return patch[cpID];
}

// ═══════════════════════════════════════════════════════════════════════
// DOMAIN SHADER — FFT texture displacement
// ═══════════════════════════════════════════════════════════════════════

struct DSOutput
{
    float4 Position : SV_POSITION;
    float3 WorldPos : TEXCOORD0;       // Displaced world position
    float4 UndisplacedXZ : TEXCOORD1;  // xy: pre-displacement XZ for PS texture sampling, z, w: the shore's damping of the
                                       // longest waves and the depth that sets each band's share of it (BandShoreAtten)
    float Depth : TEXCOORD2;
    nointerpolation uint InstanceIdx : TEXCOORD3;
    float4 SkyAmbient : TEXCOORD4;     // rgb: the sky's light on a level surface (SkyAmbient(up), the sky probe), constant per frame
};

[domain("tri")]
DSOutput DS(HullConstantOutput patchConstants,
            float3 bary : SV_DomainLocation,
            const OutputPatch<ControlPoint, 3> patch)
{
    DSOutput output;

    // Barycentric interpolation of world position
    float3 worldPos = patch[0].WorldPos * bary.x
                    + patch[1].WorldPos * bary.y
                    + patch[2].WorldPos * bary.z;

    uint idx = patch[0].InstanceIdx;

    StructuredBuffer<OceanData> oceanData = ResourceDescriptorHeap[OceanDataIdx];
    OceanData ocean = oceanData[idx];

    // Camera distance for mip selection
    float3 camPos = ViewInverse[3].xyz;
    float camDist = length(worldPos - camPos);

    // How far the shore flattens the longest waves here and the depth that sets how much each band feels of it
    // (BandShoreAtten): the PS weighs the bands' slopes and foam alike
    float waveAtten = 1.0;
    float shoreDepth = 1e4;
    float waterDepth = 1e4;
    float outside = 1e4;
    float2 surfUV = 0.5;

    // ── Shore attenuation: sample terrain heightmap for world-space water depth ──
    if (ocean.HeightmapSRV != 0)
    {
        // Convert world XZ to terrain UV [0,1]
        float2 terrainUV = float2(
            (worldPos.x - ocean.TerrainOriginX) / ocean.TerrainSizeX,
            (worldPos.z - ocean.TerrainOriginZ) / ocean.TerrainSizeZ);

        // The seabed outside the terrain is unknown: extend the border's depth outward (clamped UV) and
        // fade to open-sea waves over the next 150 m. Stopping at the border instead makes the wave
        // height jump along it wherever the seabed there is shallower than ShoreDepth.
        float2 clampedUV = clamp(terrainUV, 0.002, 0.998);
        surfUV = clampedUV;
        outside = length((terrainUV - clampedUV) * float2(ocean.TerrainSizeX, ocean.TerrainSizeZ));
        if (outside < 150.0)
        {
            Texture2D<float> heightmap = ResourceDescriptorHeap[ocean.HeightmapSRV];
            float h = heightmap.SampleLevel(OceanSampler, clampedUV, 0);
            float terrainWorldY = ocean.TerrainBaseY + h * ocean.MaxTerrainHeight;

            // Water depth = ocean surface - terrain surface (positive = water exists)
            waterDepth = ocean.OceanPlaneY - terrainWorldY;
            float shoreBlend = smoothstep(0, ocean.ShoreDepth, waterDepth);
            float shoreAtten = lerp(ocean.ShoreMinWave, 1.0, shoreBlend);

            // The terrain is the seabed: a sea built on a flat floor needs seabed stamps to deepen (Eldermoor since
            // 2026-10-19). For a day before that, terrain lying on its floor counted as open sea here, a stand-in
            float open = smoothstep(0.0, 150.0, outside);
            waveAtten = lerp(shoreAtten, 1.0, open);
            shoreDepth = max(waterDepth, 0.0) + open * 1e4;
        }
    }

    // Sample FFT displacement from all bands (with distance-based mips), each as far as the shore lets it stand, the
    // surf's bands where their waves were out at sea
    SurfWave surf;
    float3 displacement = SampleDisplacement(ocean, worldPos.xz, ocean.DisplacementAtten, camDist, waveAtten, shoreDepth,
                                             surfUV, ocean.HeightmapSRV != 0, surf);

    if (ocean.ShoreWarpSRV != 0 && shoreDepth < 1e3)
    {
        // The surf's waves grow as they shoal: over shallow water the sea stands no higher than a breaking wave and sinks
        // no lower than the bed, and nothing is pushed sideways further than the water is deep
        float slopeScale;
        displacement.y = DepthLimit(displacement.y, shoreDepth, slopeScale);
        float sideways = length(displacement.xz);
        displacement.xz *= min(1.0, max(shoreDepth, 0.05) / max(sideways, 1e-4));

        // A breaking wave's face leans forward over its trough: the higher a point of the crest, the further ahead
        float breaking = SurfBreaking(surf, shoreDepth);
        displacement.xz += SurfHeading(surf) * (SURF_LEAN * breaking * max(displacement.y, 0.0));

        // The swash: by the waterline and up the beach the level is the bores' run-up, each arriving wave's own, in step
        // with it. Until 2026-10-19 a clock (ShoreClock, noise-staggered) and a hash per wave (WaveStrength) ran it
        float swashZone = 1.0 - smoothstep(0.0, SWASH_DEPTH, waterDepth);
        displacement.y = lerp(displacement.y, SwashLevel(surf, ocean.ShoreSurge), swashZone);
    }

    float3 displaced = worldPos + displacement;

    output.UndisplacedXZ = float4(worldPos.xz, waveAtten, shoreDepth);
    output.WorldPos = displaced;
    output.Position = mul(mul(float4(displaced, 1.0), View), Projection);
    output.Depth = output.Position.w;
    output.InstanceIdx = idx;
    output.SkyAmbient = float4(SkyAmbient(float3(0, 1, 0)), 1.0);

    return output;
}

// ═══════════════════════════════════════════════════════════════════════
// PIXEL SHADER — per-pixel normals from FFT slope maps
// ═══════════════════════════════════════════════════════════════════════

struct PSOutput
{
    float4 Color : SV_Target0;
};

// The sea plane is endless, but the sea is only where it can flow in from the open sea across ground below sea level
// (C# SeaMask: flooded from the terrain's border, reaching a little up the beaches). Enclosed low ground, a lake bed
// deeper than the sea or a hall under a hill, gets no sea. Past the terrain it is all sea.
bool OutsideSea(OceanData ocean, float2 xz)
{
    if (ocean.SeaMaskSRV == 0) return false;
    float2 uv = (xz - float2(ocean.TerrainOriginX, ocean.TerrainOriginZ)) / float2(ocean.TerrainSizeX, ocean.TerrainSizeZ);
    if (any(uv < 0.0) || any(uv > 1.0)) return false;

    Texture2D<float> mask = ResourceDescriptorHeap[ocean.SeaMaskSRV];
    uint w, h;
    mask.GetDimensions(w, h);
    uv = clamp(uv, 0.5 / float2(w, h), 1.0 - 0.5 / float2(w, h));   // no blending in the far border's texels
    return mask.SampleLevel(OceanSampler, uv, 0) < 0.5;
}

PSOutput PS(DSOutput input)
{
    PSOutput output;

    float3 camPos = ViewInverse[3].xyz;

    StructuredBuffer<OceanData> oceanData = ResourceDescriptorHeap[OceanDataIdx];
    OceanData ocean = oceanData[input.InstanceIdx];

    if (OutsideSea(ocean, input.UndisplacedXZ.xy))
        discard;

    float3 worldPos = input.WorldPos;
    // How much sky this pixel of sea sees past the terrain (per pixel, as water.fx: per vertex, corners on land darkened the coast).
    // The terrain only: the sky captures' 16 lookups per pixel cost 0.29 ms where the sea fills the view (the coast),
    // and over open water little but a pier stands in the sky's way
    float skyVisible = SkyVisibilityWith(worldPos, float3(0, 1, 0), 1.0);
    float3 V = normalize(camPos - worldPos);
    float dist = distance(worldPos, camPos);

    // ── Merged per-band sampling: slope + displacement in one loop ──
    Texture2DArray<float4> dispTex = ResourceDescriptorHeap[ocean.DisplacementSRV];
    Texture2DArray<float4> slopeTex = ResourceDescriptorHeap[ocean.SlopeSRV];
    StructuredBuffer<float> tileScales = ResourceDescriptorHeap[ocean.TileScalesSRV];

    // Slope moments of every band (water_common.fx WaterSlopes): the mean slope makes the normal, the variance of
    // the slopes under this pixel its roughness. NormalAtten scales the slopes (and so the variance by its square).
    WaterSlopes slopes = (WaterSlopes)0;
    float resolvedTotal = 0;
    float totalFoam = 0;
    float waveHeight = 0;
    float windHeight = 0;      // the wind sea's part of it (not the swell): what stands up in crests
    float2 worldXZ = input.UndisplacedXZ.xy;
    // The shore flattens each band's waves by its length (DS, BandShoreAtten): their slopes, crests and foam shrink with
    // them. Until 2026-10-18 only the heights did, and over the shelf a nearly flat sea was shaded as the full open-sea chop
    float shoreAtten = input.UndisplacedXZ.z;
    float shoreDepth = input.UndisplacedXZ.w;
    // The surf's bands are sampled where their waves were out at sea (SampleBandWarp)
    float2 surfUV = saturate((worldXZ - float2(ocean.TerrainOriginX, ocean.TerrainOriginZ))
                             / float2(ocean.TerrainSizeX, ocean.TerrainSizeZ));
    SurfWave surfWave = (SurfWave)0;

    // Whitecaps (C# WhitecapBand, NoWhitecapBands): only the wind band holding the wind sea's peak, the waves one sees
    // break (the finer the calmer the sea), makes foam; the other wind bands store how squeezed their crests are right
    // now (ocean_fft.hlsl), and the finer ones' light up inside its foam. An older build sends no band: the coarsest
    uint windBands = ocean.WindBands;
    uint capBand = 0xFFFFFFFF;
    float capScale = 1e30;
    [loop]
    for (uint wb = 0; wb < ocean.NumBands; ++wb)
    {
        if (((windBands >> wb) & 1u) && tileScales[wb] < capScale)
        {
            capScale = tileScales[wb];
            capBand = wb;
        }
    }
    if (ocean.WhitecapBand != 0 && ocean.WhitecapBand <= ocean.NumBands)
    {
        capBand = ocean.WhitecapBand - 1;
        capScale = tileScales[capBand];
    }
    float capFoam = 0;         // the whitecap band's foam: where the whitecaps are
    float smallCrests = 0;     // the finer wind bands' crest squeeze now: the structure inside them
    float fineSqueeze = 0;     // the surf's structure: the uncarried wind bands' crest squeeze...
    float fineNorm = 0;        // ...and what squeeze their slopes make at a typical crest

    [loop]
    for (uint i = 0; i < ocean.NumBands; ++i)
    {
        float angle = BandAngles[i];
        float scale = tileScales[i];

        // The surf's bands where their waves were out at sea, as high as they have grown (DS SampleDisplacement)
        uint slice = WarpSlice(ocean, i);
        float2 p = worldXZ;
        float waveAtten;
        float2x2 J = float2x2(1.0, 0.0, 0.0, 1.0);
        float2 turn = float2(1.0, 0.0), heading = float2(1.0, 0.0);
        if (slice != 0xFF)
        {
            BandWarp w = SampleBandWarp(ocean, slice, worldXZ, surfUV);
            p = w.u;
            waveAtten = w.amp;
            J = w.J;
            turn = w.turn;
            heading = w.heading;
        }
        else
            waveAtten = BandShoreAtten(shoreAtten, shoreDepth, scale);
        float2 uv = RotateUV(p, angle) * scale;

        // Filtered: the foam alpha is only a coverage mask, the visible detail comes from the foam texture
        float4 disp = dispTex.Sample(OceanSampler, float3(uv, i));
        float4 moments = slopeTex.Sample(OceanSampler, float3(uv, i));
        disp *= waveAtten;
        if (slice != 0xFF)
        {
            // The surf's wave: its height and its sideways displacement along the local heading (as the DS has them)
            float2 side = RotateUV(disp.xz, -angle);
            side = float2(side.x * turn.x - side.y * turn.y, side.x * turn.y + side.y * turn.x);
            AddSurfWave(surfWave, ocean, float3(side.x, disp.y, side.y), heading, slice);
        }

        // Counter-rotate the mean slope back to world space (the variance does not depend on the rotation)
        float s, c;
        sincos(-angle, s, c);
        float2 meanWorld = float2(moments.x * c - moments.y * s, moments.x * s + moments.y * c);
        if (slice != 0xFF)
        {
            // Through the warp the slopes are Jᵀ·(the open sea's): the crowded crests are steeper. The variance grows
            // with the mean square stretch
            float slopeScale = ocean.NormalAtten * waveAtten;
            float stretch2 = 0.5 * (J[0][0] * J[0][0] + J[0][1] * J[0][1] + J[1][0] * J[1][0] + J[1][1] * J[1][1]);
            slopes.mean += mul(meanWorld, J) * slopeScale;
            slopes.variance += max(moments.z + moments.w - dot(moments.xy, moments.xy), 0.0) * slopeScale * slopeScale * stretch2;
            resolvedTotal += WaterBandTotalVariance(slopeTex, OceanSampler, i, slopeScale) * stretch2;
        }
        else
        {
            AddWaterBand(slopes, moments, meanWorld, ocean.NormalAtten * waveAtten);
            float bandVariance = WaterBandTotalVariance(slopeTex, OceanSampler, i, ocean.NormalAtten * waveAtten);
            resolvedTotal += bandVariance;
            if (((windBands >> i) & 1u) && i != capBand)
            {
                fineSqueeze += disp.a;
                fineNorm += ocean.Choppiness * sqrt(bandVariance);
            }
        }
        totalFoam += disp.a;
        waveHeight += disp.y;
        if ((windBands >> i) & 1u)
        {
            windHeight += disp.y;
            // Weighted by the band's size against the whitecap band's: the 64 m band's crest lines give the structure,
            // the ripples (whose squeeze is the strongest, in round cells) only its finest detail. Unweighted they
            // spotted the whitecaps like a dalmatian
            if (i == capBand)
                capFoam = disp.a;
            else if (scale > capScale)
                smallCrests += disp.a * sqrt(capScale / scale);
        }
    }

    // The DS's depth limit flattens the shoaled crests (DepthLimit): their slopes with them
    if (ocean.ShoreWarpSRV != 0 && shoreDepth < 1e3)
    {
        float capSlope;
        waveHeight = DepthLimit(waveHeight, shoreDepth, capSlope);
        slopes.mean *= capSlope;
        slopes.variance *= capSlope * capSlope;
    }

    float3 N = normalize(float3(-slopes.mean.x, 1.0, -slopes.mean.y));
    float alpha = WaterAlpha(slopes, WaterUnresolvedVariance(resolvedTotal));

    float NdotV = saturate(dot(N, V));

    // ── Sun vectors ──
    float3 sunDir = normalize(ocean.SunDirection);
    float3 L = normalize(-sunDir);
    float NdotL = saturate(dot(N, L));
    // The directional light (sun or moon) in the older units, pre-exposed: what a white Lambert surface facing it
    // sends back, as in the deferred light (E / π). The specular BRDF takes the illuminance itself (× π): until
    // 2026-10-17 it got this radiance, and the water's sun was π times weaker than the deferred one.
    float3 sunRadiance = ocean.SunColor * ocean.SunIntensity * SunLightScale;
    sunRadiance *= GetCloudShadow(OceanSampler, worldPos - camPos, L);
    // The cascades, and the terrain's shadow past them (a headland the sun stands behind), as the deferred light has them
    sunRadiance *= min(WaterSunShadow(ShadowMapIdx, worldPos - camPos, input.Position.xy), TerrainLightShadow(worldPos, L));
    sunRadiance *= TerrainOpen(worldPos, float3(0, 1, 0));   // a sea cave under a hill: no sun, whatever the shadow map holds

    float H = max(0.0, waveHeight);

    // ── Base color: dark deep ocean ──
    float depthBlend = saturate(dist / 4000.0);
    float3 baseColor = lerp(ocean.OceanColor, ocean.DeepColor, depthBlend);

    // ── Subsurface scattering ──
    float3 bubbleColor = float3(0.0, 0.008, 0.006);

    float k1 = 0.3 * H * pow(saturate(dot(L, -V)), 4.0)
             * pow(0.5 - 0.5 * dot(L, N), 3.0);
    float k2 = 0.1 * NdotV * NdotV;
    float k3 = 0.08 * NdotL;

    float3 scatter = ((k1 + k2) * baseColor + k3 * baseColor + 0.02 * bubbleColor) * sunRadiance
                   + baseColor * WaterSkyIrradiance(input.SkyAmbient.rgb * skyVisible) * 0.1;   // light from the sky above, too

    // ── Reflectance, sky reflection and sun glint: shared with lakes and rivers (water_common.fx) ──
    // All from the mean normal and the slope variance under the pixel: no distance ramps.
    float F = WaterSpecularReflectance(NdotV, alpha);
    float3 reflectColor = WaterSkyReflection(V, N, alpha) * skyVisible;   // no sky to mirror under a cliff
    float3 specular = WaterSunSpecular(N, V, L, sunRadiance * 3.14159, alpha);

    // ── Water column: what is seen through the surface ──
    // pathLen  = distance the view ray travels under water before it hits the scene (depth buffer)
    // bedDepth = vertical water depth right under this pixel (terrain heightmap, view independent)

    // Terrain seabed. Its geometry and heightmap stop at the terrain border while the sea goes on, so
    // everything derived from it is faded out across the border (seabedFade) to avoid a straight seam:
    // show-through over the last 60 m inside, the heightmap depth over the first 150 m outside.
    float terrainDepth = 1000.0;
    float stillDepth = 1000.0;      // depth below the undisturbed sea level (no waves, no surge)
    float seabedFade = 1.0;
    float shallowFade = 0.0;
    float2 terrainSize = float2(ocean.TerrainSizeX, ocean.TerrainSizeZ);
    float2 clampedUV = 0.5;
    if (ocean.HeightmapSRV != 0)
    {
        float2 terrainUV = (worldPos.xz - float2(ocean.TerrainOriginX, ocean.TerrainOriginZ)) / terrainSize;
        clampedUV = clamp(terrainUV, 0.002, 0.998);
        float outside = length((terrainUV - clampedUV) * terrainSize);
        float2 toBorder = min(clampedUV, 1.0 - clampedUV) * terrainSize;

        seabedFade = smoothstep(0.0, 60.0, min(toBorder.x, toBorder.y));
        if (outside < 150.0)
        {
            Texture2D<float> heightmap = ResourceDescriptorHeap[ocean.HeightmapSRV];
            float h = heightmap.SampleLevel(OceanSampler, clampedUV, 0);
            float terrainY = ocean.TerrainBaseY + h * ocean.MaxTerrainHeight;
            terrainDepth = worldPos.y - terrainY;
            stillDepth = ocean.OceanPlaneY - terrainY;
            // The turquoise shelf is wherever the seabed is shallow (no floor rule since 2026-10-19, as in the DS)
            shallowFade = 1.0 - smoothstep(0.0, 150.0, outside);
            // Under the land: where the terrain stands more than 3 m above the still sea (far past any wave's run-up)
            // the sea shows only where the terrain does not cover what is under it, a sea cave under a hill or a
            // terrain hole. That is no beach: read as one, it was a sheet of "shore" foam. Open water there
            if (stillDepth < -3.0)
            {
                terrainDepth = 1000.0;
                stillDepth = 1000.0;
                shallowFade = 0.0;
            }
        }
    }

    WaterColumn column = GetWaterColumn(OceanSampler, ocean.DepthGBufferSRV, ocean.CompositeSRV,
        input.Position.xy * ocean.InvViewportSize, input.Depth, dist, abs(camPos.y - worldPos.y), N,
        ocean.RefractionStrength, scatter, ocean.ShallowColor, ocean.ShoreFadeDepth, seabedFade);
    float3 body = column.body;
    float edgeFade = column.edgeFade;
    float pathLen = column.pathLen;
    float bufDepth = column.bufDepth;

    // Vertical water depth under the pixel: heightmap where there is one, depth buffer otherwise
    float bedDepth = (terrainDepth < 999.0) ? terrainDepth : bufDepth;

    // Light scattered back from a bright shallow bed: the turquoise shelf along the coast
    float3 skyAmbient = input.SkyAmbient.rgb * skyVisible;
    float shallow = exp(-max(bedDepth, 0.0) / max(0.01, ocean.ShoreFadeDepth)) * saturate(pathLen);
    shallow *= (ocean.HeightmapSRV != 0) ? shallowFade : 1.0;
    body += ocean.ShallowColor * (sunRadiance * saturate(L.y) + WaterSkyIrradiance(skyAmbient)) * 0.08 * shallow;

    // Light through the crests: a wind wave's crest is a thin wall of water. Sky light (and the sun's, strongest when
    // looking toward it) entering its back and top scatters through it toward the eye, tinted by what ~1.5 m of water
    // lets through (ShallowColor is its per-metre transmittance): the teal glow of a storm sea's crests. It grows with the
    // crest's height above the mean (wind sea only: the long swell is thick) and on faces seen at a slant. Under a storm
    // sky the sun-only term above (k1) left the whole sea a dark sky mirror.
    float crestRise = smoothstep(CREST_GLOW_HEIGHT * 0.25, CREST_GLOW_HEIGHT, windHeight);
    crestRise *= crestRise;
    float slant = 1.0 - NdotV;
    float3 throughCrest = pow(max(ocean.ShallowColor, 0.02), 2.5);
    throughCrest = lerp(dot(throughCrest, float3(0.2126, 0.7152, 0.0722)), throughCrest, 0.7);   // storm water is murky
    float3 crestLight = skyAmbient + sunRadiance * (0.2 + 3.0 * pow(saturate(dot(L, -V)), 4.0));
    body += crestLight * throughCrest * (CREST_GLOW * crestRise * lerp(0.3, 1.0, slant * slant)) * skyVisible;

    // Under a whitecap the water is full of the air the breaking crest carried down: the bubbles scatter the light
    // back up, pale turquoise water under the foam and behind it that lasts as long as the whitecap band's foam does
    // (its foam is the measure, as for the whitecaps below)
    body += (skyAmbient + sunRadiance * 0.3) * throughCrest * (AERATION * saturate(capFoam)) * skyVisible;

    // ── Combine lighting ──
    float3 color = (1.0 - F) * body + (specular + F * reflectColor) * edgeFade;
    color = max(0.0, color);

    // ── Foam ──
    // Open water: foam's opacity comes straight from the bands' Jacobian foam, as in the reference implementation (the
    // GarrettGunnell port, 2026-02-28). From 2026-10-04 it went through the shore's coverage path below (eroded by the
    // cells, a soft fill past 150 m), which turned saturated foam into solid white sheets with holes and far foam into
    // blobs (user, 2026-10-18). With a wind sea the coarsest wind band's foam is the mask of where the whitecaps are,
    // and inside it the finer bands' crests, squeezed right now, raise the opacity: the small waves' crests light up
    // in the whitecap and move with it, and as the whitecap fades only they remain, a lace of the sea's own pattern
    // (the user's idea, 2026-10-18). Outside a whitecap they show nothing: on their own the finer bands' foam came up
    // as random blobs where no visible crest is. Without wind bands every band's foam is summed as in the reference.
    // The foam lives in mipmapped textures, so far away it averages out instead of aliasing.
    float t = ocean.WaveTime;
    // Foam is bubbles: there or not. Thin foam drawn at a fraction of its opacity greyed the water like smoke, so the
    // wind sea's foam drops out below WHITECAP_THIN and is solid from WHITECAP_THIN + WHITECAP_RAMP: as it thins it
    // breaks up along the small crests' pattern instead of fading as a whole
    float whitecap = capBand != 0xFFFFFFFF
        ? saturate((saturate(capFoam) * saturate(WHITECAP_VEIL + WHITECAP_CRESTS * smallCrests) - WHITECAP_THIN) / WHITECAP_RAMP)
        : saturate(totalFoam);
    whitecap *= saturate(1.0 - dist / 3000.0);

    // ── The surf: foam from the waves themselves (C# ShoreWarp, SurfWave) ──
    // Every carried wave knows its own height and phase: once it stands taller than ~0.8 × the depth it has broken, and
    // stays broken toward the beach (big waves of a set break further out). Its white water rides its crest and face (the
    // roller) and is left behind it, thinning over about a third of a wave, with the bubbles of the broken water all
    // through the surf zone. Drawn like the whitecaps, through the finer bands' crests. Until 2026-10-19 the shore's foam
    // ran on a clock: bands crossing the swash zone on a noise-staggered schedule (ShoreClock), a hash per wave
    // (WaveStrength), a drifting Voronoi web (FoamWeb) and streaks, none of it a wave.
    float surfFoam = 0.0;
    if (ocean.ShoreWarpSRV != 0 && shoreDepth < 1e3)
    {
        float breaking = SurfBreaking(surfWave, shoreDepth);
        // Each carried band's roller and trail at its own phase, the weaker band's in proportion to its height
        float2 weight = surfWave.envelope / max(max(surfWave.envelope.x, surfWave.envelope.y), 1e-6);
        float2 rt0 = SurfRollerTrail(surfWave.phase.x) * weight.x;
        float2 rt1 = SurfRollerTrail(surfWave.phase.y) * weight.y;
        float roller = breaking * max(rt0.x, rt1.x);
        float trail = breaking * max(rt0.y, rt1.y) + SURF_BUBBLES * breaking * breaking;
        // The roller is a churning white mass, broken only at its edges; what it leaves behind breaks up along the crests
        float structure = saturate(fineSqueeze / max(fineNorm, 1e-4));
        surfFoam = max(roller * saturate(SURF_ROLLER + SURF_CRESTS * structure),
                       trail * saturate(SURF_VEIL + SURF_CRESTS * structure));
        // Up the beach the warp holds the waterline's phase (C# ShoreWarp): the bore's white water ends at the waterline,
        // and the swash's edge is the waterline's lace (below)
        surfFoam *= smoothstep(0.0, 0.4, shoreDepth);
        surfFoam = saturate((saturate(surfFoam) - WHITECAP_THIN) / WHITECAP_RAMP);
        surfFoam *= saturate(1.0 - dist / 3000.0);
    }

    // The waterline: foam gathers where the water meets the sand, a rock or a pier, a ragged band that swells and pinches
    // along the coast; on a beach the swash's edge, crisp as a bore runs up and thinner as it drains. Everything below
    // builds a COVERAGE value, eroded by a cellular foam texture so the edge breaks up into bubbles.
    float coverage = 0.0;
    float foamTex = 0.5;
    if (ocean.NoiseSRV != 0)
    {
        Texture2D<float4> noiseTex = ResourceDescriptorHeap[ocean.NoiseSRV];
        float frontBroad = noiseTex.SampleLevel(OceanSampler, worldPos.xz * 0.045 + t * 0.004, 0).r;
        float frontFine = noiseTex.SampleLevel(OceanSampler, worldPos.xz * 0.21 - t * 0.006, 0).a;
        float edgeWidth = lerp(0.4, 2.4, saturate((frontBroad - 0.3) / 0.4)) * lerp(0.7, 1.3, frontFine);

        // The beach's steepness from the heightmap, over a 3 m baseline so it stays smooth: it converts the edge's width
        // along the beach into water depth, so the edge is as wide on a steep shore as on a flat one
        float beachSlope = 0.1;
        float laceDepth = min(bedDepth, bufDepth);
        bool hasBeach = ocean.HeightmapSRV != 0 && terrainDepth < 999.0;
        if (hasBeach && laceDepth < 1.0)
        {
            Texture2D<float> heightmap = ResourceDescriptorHeap[ocean.HeightmapSRV];
            float2 slopeStep = 3.0 / terrainSize;
            float2 uphill = float2(
                heightmap.SampleLevel(OceanSampler, clampedUV + float2(slopeStep.x, 0), 0)
              - heightmap.SampleLevel(OceanSampler, clampedUV - float2(slopeStep.x, 0), 0),
                heightmap.SampleLevel(OceanSampler, clampedUV + float2(0, slopeStep.y), 0)
              - heightmap.SampleLevel(OceanSampler, clampedUV - float2(0, slopeStep.y), 0));
            beachSlope = clamp(length(uphill) * ocean.MaxTerrainHeight / 6.0, 0.03, 1.0);
        }
        static const float LaceWidth = 0.7;      // meters along the beach, before the variation
        float lace = 1.0 - smoothstep(0.0, min(LaceWidth * edgeWidth * beachSlope, 0.7), laceDepth);

        // On a beach the edge goes by the swash's cycle (the arriving wave's phase, as SwashLevel): crisp on the run-up,
        // thinner while the water drains. Objects standing in deeper water keep their lace.
        if (ocean.ShoreWarpSRV != 0 && hasBeach)
        {
            float2 share = SurfShare(surfWave);
            float advancing = SwashAdvancing(surfWave.phase.x) * share.x + SwashAdvancing(surfWave.phase.y) * share.y;
            float onBeach = 1.0 - smoothstep(0.0, 0.5, bedDepth);
            lace *= lerp(1.0, lerp(0.45, 1.0, advancing), onBeach);
        }

        // Stays below full coverage so the foam texture always breaks it up. The surf's white water is drawn the same way:
        // a mass of bubbles, ragged at its edges and grey where it thins
        coverage = saturate(max(lace * 0.8 * ocean.ShoreFoam, surfFoam * 0.92));

        // Foam texture: two drifting scales of cellular noise (noise texture has no mips: flatten it far away)
        float cells1 = noiseTex.SampleLevel(OceanSampler, worldPos.xz * 0.23 + t * float2(0.010, 0.006), 0).g;
        float cells2 = noiseTex.SampleLevel(OceanSampler, worldPos.xz * 0.83 - t * float2(0.013, 0.004), 0).g;
        foamTex = saturate((cells1 * 0.6 + cells2 * 0.4 - 0.15) / 0.6);
    }

    // Thin foam is translucent, thick foam keeps the cell structure as shading.
    // The noise texture has no mips: far away the cells would only alias, so fade to a soft
    // version of the coverage itself.
    float foamFar = saturate(dist / 150.0);
    float foam = saturate((coverage * 1.25 - (1.0 - foamTex)) * 3.0);
    foam = lerp(foam, saturate((coverage - 0.35) * 1.5) * 0.7, foamFar);
    foam = max(foam, surfFoam * foamFar);           // far away the surf's white water as it is (the lace's fade is for thin lines)

    // Always feather the last few centimeters of water: whatever the foam pattern, it must not end
    // in the hard pixel edge where the water surface cuts into the beach.
    float feather = smoothstep(0.0, 0.025, pathLen);
    foam *= feather;
    whitecap *= feather;
    foamTex = lerp(foamTex, 0.5, foamFar);
    if (foam > 0.001 || whitecap > 0.001)
    {
        float3 skyOnFoam = SkyAmbient(N) * skyVisible;
        color = lerp(color, WaterFoamLit(float3(0.80, 0.80, 0.78), NdotL, sunRadiance, skyOnFoam), whitecap);
        float3 foamAlbedo = float3(0.80, 0.80, 0.78) * lerp(0.7, 1.0, foamTex);
        color = lerp(color, WaterFoamLit(foamAlbedo, NdotL, sunRadiance, skyOnFoam), foam * 0.9);
    }

    // ── The air between the camera and the water (the atmosphere's aerial perspective) ──
    color = ApplyAerialPerspective(color, input.Position.xy, input.Depth, skyVisible);

    // HDR output — no tonemapping/gamma here.
    // The finalize pass applies ACES + gamma to the entire Composite.

    output.Color = float4(color, 1.0);

    return output;
}

// ═══════════════════════════════════════════════════════════════════════
// SHADOW PASS — tessellated, FFT displacement only
// ═══════════════════════════════════════════════════════════════════════

struct ShadowDSOutput { float4 Position : SV_POSITION; };

[domain("tri")]
[partitioning("fractional_odd")]
[outputtopology("triangle_cw")]
[outputcontrolpoints(3)]
[patchconstantfunc("PatchConstantFunc")]
ControlPoint HS_Shadow(InputPatch<ControlPoint, 3> patch, uint cpID : SV_OutputControlPointID)
{
    return patch[cpID];
}

[domain("tri")]
ShadowDSOutput DS_Shadow(HullConstantOutput patchConstants,
                         float3 bary : SV_DomainLocation,
                         const OutputPatch<ControlPoint, 3> patch)
{
    ShadowDSOutput output;

    float3 worldPos = patch[0].WorldPos * bary.x
                    + patch[1].WorldPos * bary.y
                    + patch[2].WorldPos * bary.z;

    StructuredBuffer<OceanData> oceanData = ResourceDescriptorHeap[OceanDataIdx];
    OceanData ocean = oceanData[patch[0].InstanceIdx];

    // Without the surf's warp: the sun's shadow of the water near the coast is the open sea's
    SurfWave surf;
    float3 displacement = SampleDisplacement(ocean, worldPos.xz, 1.0, 0.0, 1.0, 1e4, 0.5, false, surf);
    float3 displaced = worldPos + displacement;

    output.Position = mul(mul(float4(displaced, 1.0), View), Projection);
    return output;
}

// ═══════════════════════════════════════════════════════════════════════
technique11 GBuffer
{
    pass Forward
    {
        SetVertexShader(CompileShader(vs_6_6, VS_Control()));
        SetHullShader(CompileShader(hs_6_6, HS()));
        SetDomainShader(CompileShader(ds_6_6, DS()));
        SetPixelShader(CompileShader(ps_6_6, PS()));
    }
}
