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



#include "ocean_surf.hlsli"

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
    OceanShore shore = ShoreAt(ocean, worldPos.xz);
    float waveAtten = shore.waveAtten;
    float shoreDepth = shore.shoreDepth;
    float waterDepth = shore.waterDepth;
    float2 surfUV = shore.surfUV;

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
        // (not by the waterline: the beach's phase runs across the shore, SURF_FOAM_END)
        breaking *= smoothstep(SURF_FOAM_END.x, SURF_FOAM_END.y, shoreDepth);
        displacement.xz += SurfHeading(surf) * (SURF_LEAN * breaking * max(displacement.y, 0.0));

        // The swash: by the waterline and up the beach the level is the bores' run-up, each arriving wave's own, in step
        // with it. Until 2026-10-19 a clock (ShoreClock, noise-staggered) and a hash per wave (WaveStrength) ran it
        float swashZone = 1.0 - smoothstep(0.0, SWASH_DEPTH, waterDepth);
        float level = SwashLevel(surf, ocean.ShoreSurge);
        // Up the beach the run-up wets a spot only once it has topped everything between the waterline and there: a
        // hollow behind a ridge stays dry until the water pours over. One level over the whole beach filled every hollow
        // below it within the sea mask's reach, puddles far up the sand outlined by the lace (until 2026-10-19)
        // The ridge is the sand's own height (1 m heightmap) plus how far below it the spot lies (C# ShoreWarp, 0 on a
        // plain slope): baked as the ridge's height, its 4 m cells frayed the waterline into a sawtooth
        if (waterDepth < 0.0)
        {
            float sand = -waterDepth;                                          // the sand's height over the sea
            float ridge = sand + surf.hollow;
            float inHollow = smoothstep(0.02, 0.1, surf.hollow);
            float topped = smoothstep(ridge - 0.05, ridge + 0.05, level);
            level = lerp(level, sand - 0.3, inHollow * (1.0 - topped));
        }
        displacement.y = lerp(displacement.y, level, swashZone);
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
    float surfTrailWeb = 0.0;     // how much thinning trail there is, for the web (below)
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
        // Up the beach the warp holds the waterline's phase (C# ShoreWarp), lines across the shore that the swash runs on:
        // drawn as white water they made hooks along the waterline (user, 2026-10-19). The bore's white water ends where
        // the water gets that shallow (SURF_FOAM_END), and the swash's edge is the waterline's lace (below)
        surfFoam *= smoothstep(SURF_FOAM_END.x, SURF_FOAM_END.y, shoreDepth);
        surfFoam = saturate((saturate(surfFoam) - WHITECAP_THIN) / WHITECAP_RAMP);
        surfFoam *= saturate(1.0 - dist / 3000.0);
        // The trail as it thins, away from the roller: the web takes it over
        surfTrailWeb = saturate(trail * 1.4) * (1.0 - saturate(roller * 2.0))
                     * smoothstep(SURF_FOAM_END.x, SURF_FOAM_END.y, shoreDepth);
    }

    // The waterline: foam gathers where the water meets the sand, a rock or a pier, a ragged band that swells and pinches
    // along the coast; on a beach the swash's edge, crisp as a bore runs up and thinner as it drains. Everything below
    // builds a COVERAGE value, eroded by a cellular foam texture so the edge breaks up into bubbles.
    float coverage = 0.0;
    float foamTex = 0.5;
    float laceTex = 0.5;
    float streakFoam = 0.0;
    if (ocean.NoiseSRV != 0)
    {
        Texture2D<float4> noiseTex = ResourceDescriptorHeap[ocean.NoiseSRV];
        float frontBroad = noiseTex.SampleLevel(OceanSampler, worldPos.xz * 0.045 + t * 0.004, 0).r;
        float frontFine = noiseTex.SampleLevel(OceanSampler, worldPos.xz * 0.21 - t * 0.006, 0).a;
        float edgeWidth = lerp(0.4, 2.4, saturate((frontBroad - 0.3) / 0.4)) * lerp(0.7, 1.3, frontFine);

        // The beach's steepness from the heightmap, over a 3 m baseline so it stays smooth: it converts the edge's width
        // along the beach into water depth, so the edge is as wide on a steep shore as on a flat one
        float beachSlope = 0.1;
        float2 downhill = float2(0.0, 1.0);
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
            float steepness = length(uphill);
            downhill = -uphill / max(steepness, 1e-6);
            beachSlope = clamp(steepness * ocean.MaxTerrainHeight / 6.0, 0.03, 1.0);
        }
        static const float LaceWidth = 0.45;     // meters along the beach, before the variation (0.7 until 2026-10-19)
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

        // Stays below full coverage so the foam texture always breaks it up
        coverage = saturate(lace * 0.8 * ocean.ShoreFoam);

        // Foam texture: two drifting scales of cellular noise (noise texture has no mips: flatten it far away). Sampled
        // where the surface's water rests (the undisplaced position): the bubbles ride the water they float on, out with
        // a breaking lip and back with the swash. At the displaced position the white water slid over a fixed pattern
        float2 foamPos = input.UndisplacedXZ.xy;
        float cells1 = noiseTex.SampleLevel(OceanSampler, foamPos * 0.23 + t * float2(0.010, 0.006), 0).g;
        float cells2 = noiseTex.SampleLevel(OceanSampler, foamPos * 0.83 - t * float2(0.013, 0.004), 0).g;
        foamTex = saturate((cells1 * 0.6 + cells2 * 0.4 - 0.15) / 0.6);

        // The waterline's foam is fine: a ragged line of small bubbles (0.3 to 0.6 m), not the coarse cells above. Cut by
        // those it broke into round clusters along the edge like paw prints (user, 2026-10-19)
        float fine1 = noiseTex.SampleLevel(OceanSampler, foamPos * 1.75 - t * float2(0.013, 0.004), 0).g;
        float fine2 = noiseTex.SampleLevel(OceanSampler, foamPos * 3.7 + t * float2(0.009, 0.011), 0).g;
        laceTex = lerp(saturate((fine1 * 0.55 + fine2 * 0.45 - 0.15) / 0.6), 0.5, saturate(dist / 45.0));

        // The web (FoamWeb): the net of bubble films foam breaks into as it thins, where and when the waves leave thin
        // foam: the film of each wave's backwash draining off the sand, strongest at the top of its run-up and gone by
        // the time the next bore arrives, and the thinning trail behind the surf's broken crests. Its look only is
        // made up; until 2026-10-19 a clock decided when it showed too, and it went with the clock that day. Only near
        // the camera: far away the cells would only alias
        float film = 0.0;
        float rise = 1.0;
        if (ocean.ShoreWarpSRV != 0 && hasBeach)
        {
            float2 share = SurfShare(surfWave);
            float age0 = SwashBackwash(surfWave.phase.x), age1 = SwashBackwash(surfWave.phase.y);
            float fresh = (age0 < 0.0 ? 0.0 : 1.0 - smoothstep(0.1, 0.9, age0)) * share.x
                        + (age1 < 0.0 ? 0.0 : 1.0 - smoothstep(0.1, 0.9, age1)) * share.y;
            film = fresh * (1.0 - smoothstep(0.0, max(0.15, ocean.ShoreSurge * 1.5), bedDepth));   // in thin water only
            rise = SwashRise(surfWave.phase.x) * share.x + SwashRise(surfWave.phase.y) * share.y;
        }
        float webAmount = max(film, surfTrailWeb) * (1.0 - saturate(dist / 150.0));
        if (webAmount > 0.01)
        {
            // It floats on the water: the backwash's film is carried down the beach as the water drains and back up with
            // the next bore (a bounded shift that follows the swash, so a flow that bends along the coast cannot shear
            // it), the trail's is stretched along the waves' heading
            bool backwash = film >= surfTrailWeb;
            float2 flowDir = backwash ? downhill : SurfHeading(surfWave);
            float swashZone = 1.0 - smoothstep(0.0, 2.0 * SWASH_DEPTH, stillDepth);
            float waterTravel = min(2.0 * ocean.ShoreSurge / beachSlope, 8.0) * 0.6 * swashZone;
            float2 webBase = foamPos - flowDir * (backwash ? waterTravel * (1.0 - rise) : 0.0);

            const float webCellSize = 0.95;    // meters across a cell (before stretching and merging)
            const float webStretch = 2.0;      // cells are this much longer along the flow
            const float webRate = 0.9;         // how fast the cell points drift (radians per second of wave time)
            const float webDrop = 0.33;        // a fixed third of the points retired, for uneven cell sizes

            // Bend the space first: a large slow warp squeezes some regions and opens others (uneven cell sizes), a
            // medium one curves the strands, a little fine wobble roughens them
            float2 warpUV = webBase * 0.19;
            float2 bigWarp = float2(noiseTex.SampleLevel(OceanSampler, warpUV * 0.31 + 0.27, 0).r,
                                    noiseTex.SampleLevel(OceanSampler, warpUV * 0.31 + 0.83, 0).r) - 0.5;
            float2 warp = float2(noiseTex.SampleLevel(OceanSampler, warpUV + t * 0.006, 0).r,
                                 noiseTex.SampleLevel(OceanSampler, warpUV + 0.43 - t * 0.005, 0).r) - 0.5;
            float2 wobble = float2(noiseTex.SampleLevel(OceanSampler, warpUV * 2.9 + 0.11, 0).a,
                                   noiseTex.SampleLevel(OceanSampler, warpUV * 2.9 + 0.71, 0).a) - 0.5;
            float2 webPos = webBase + bigWarp * 4.5 + warp * 1.8 + wobble * 0.12;

            // A soft mask, 1 on a strand's centre line, swelling into knots where cells meet (thin hard lines of even
            // brightness read as drawn cracks), with a finer, fainter web inside the cells
            float2 web = FoamWeb(webPos / webCellSize, flowDir, webStretch, t * webRate, webDrop);
            float webLace = saturate(1.0 - web.x / (0.09 + 0.17 * smoothstep(0.4, 0.9, web.y)));
            float2 fine = FoamWeb(webPos / (webCellSize * 0.4) + 17.3, flowDir, webStretch, t * webRate * 1.3, webDrop);
            webLace = max(webLace, saturate(1.0 - fine.x / 0.11) * 0.6 * smoothstep(0.2, 0.55, web.x));

            // Patchy: it thins out and tears open in places instead of covering the water evenly
            float patches = noiseTex.SampleLevel(OceanSampler, webBase * 0.06 - t * 0.004, 0).b;
            webLace *= lerp(0.25, 1.0, smoothstep(0.38, 0.62, patches));

            // Strands are made of bubbles: the fine cells erode the soft mask, so the cores stay solid and the edges
            // break into ragged clusters (cells ~7 cm; flattened before they start to shimmer)
            float bubbles = noiseTex.SampleLevel(OceanSampler, webBase * 1.75 - t * float2(0.013, 0.004), 0).g;
            bubbles = lerp(saturate((bubbles - 0.15) / 0.55), 0.5, saturate(dist / 45.0));
            float webFoam = saturate((webLace * 1.35 - (1.0 - bubbles) * 0.75) * 2.2);

            // Translucent filaments, lightly broken up by the fine cells
            streakFoam = webFoam * webAmount * 0.85 * lerp(0.75, 1.0, cells2) * ocean.ShoreFoam;
        }
    }

    // Thin foam is translucent, thick foam keeps the cell structure as shading.
    // The noise texture has no mips: far away the cells would only alias, so fade to a soft
    // version of the coverage itself.
    float foamFar = saturate(dist / 150.0);
    float foam = saturate((coverage * 1.25 - (1.0 - laceTex)) * 3.0);
    foam = lerp(foam, saturate((coverage - 0.35) * 1.5) * 0.7, foamFar);
    // The surf's white water: a mass of bubbles, eroded more gently than the waterline's thin lace (cut as hard, its
    // edges looked like paper), solid where it is thick; far away as it is (the lace's fade is for thin lines)
    float surfNear = saturate((surfFoam * 1.3 - (1.0 - foamTex)) * 2.0);
    foam = max(foam, lerp(surfNear, surfFoam, foamFar));
    foam = max(foam, streakFoam * (1.0 - foamFar));   // the web of thinning foam
    float thickness = saturate(max(coverage, surfFoam));

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
        // Thick foam is bright, thinning foam greys as the water shows through its bubbles
        float3 foamAlbedo = float3(0.80, 0.80, 0.78) * lerp(0.65, 1.0, foamTex) * lerp(0.78, 1.05, thickness);
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
