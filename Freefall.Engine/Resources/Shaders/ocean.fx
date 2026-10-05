cbuffer PushConstants : register(b3)
{
    // Slots 0-1: Reserved for light/composition passes
    uint _reserved0;
    uint _reserved1;
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
    float _pad0;
    float3 DeepColor;
    float _pad1;
    float3 SunDirection;
    float SunIntensity;
    float3 SunColor;
    float _pad2;
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
    float3 CloudColor;          // SkyboxRenderer.CurrentCloudColor (clouds in the sky reflection)
    float ShoreFoamDepth;       // water depth (meters) below which swash foam bands roll in
    float ShoreFoam;            // shore foam amount (0 = none)
    float ShoreSurge;           // water level swing (meters) at the beach, in step with the swash bands
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

// Simple frustum cull for patches — zero out tessellation for off-screen triangles
bool CullTriangle(float3 p0, float3 p1, float3 p2)
{
    // Expand bounds slightly to account for displacement
    float bias = -20.0;
    float3 minP = min(min(p0, p1), p2) + bias;
    float3 maxP = max(max(p0, p1), p2) - bias;

    float4 clipMin = mul(mul(float4(minP, 1.0), View), Projection);
    float4 clipMax = mul(mul(float4(maxP, 1.0), View), Projection);

    // Behind camera check
    if (clipMin.w < 0 && clipMax.w < 0) return true;

    return false;
}

SamplerState OceanSampler : register(s0);

// ── Shore waves: one event per wave, shared by the DS (water level) and the PS (foam) ──
//
// Each wave is a foam front that crosses the swash zone (still-water depth ShoreFoamDepth → 0) in
// exactly one cycle, so there is one front in the zone at a time. The water level bottoms out at
// -ShoreSurge, which puts the retreated waterline at still depth = ShoreSurge. The surge cycle is
// offset so the run-up starts at the moment the front gets there: the front becomes the run-up.
//
//   ShoreClock  = cycles elapsed; the front of wave m is at still depth D * (m - ShoreClock)
//   SurgeClock  = ShoreClock shifted so that frac() == 0 is "front meets the retreated waterline"
//   frac(SurgeClock): 0..SurgeRise = run-up, SurgeRise..1 = backwash (0.5 would be symmetric)
static const float SurgeRise = 0.38;

float ShoreClock(float waveTime, float bandNoise)
{
    return waveTime * 0.22 + bandNoise * 1.5;   // noise staggers the waves along the coast
}

float SurgeClock(float shoreClock, float shoreSurge, float shoreFoamDepth)
{
    return shoreClock + shoreSurge / max(0.1, shoreFoamDepth);
}

float2 Hash22(float2 p)
{
    p = float2(dot(p, float2(127.1, 311.7)), dot(p, float2(269.5, 183.3)));
    return frac(sin(p) * 43758.5453);
}

// Animated cellular noise for foam webs. Returns (F2 - F1, F1): x is 0 on the border between two
// cells and grows toward the cell centers; y is the distance to the nearest center, largest where
// several cells meet (used to thicken the knots). The feature points orbit with time, so the web
// shifts and reconnects in place. The cells are stretched along flowDir through the distance metric
// only — the grid itself stays world-aligned, so a flow direction that changes across the beach
// cannot shear the pattern. On its own this is straight-edged Voronoi: warp the input to bend it.
// drop (0..~0.65) is the fraction of points retired: raise it over the foam's life to make the cells grow.
float2 FoamWeb(float2 x, float2 flowDir, float stretch, float time, float drop)
{
    float2 cell = floor(x);
    float2 f = frac(x);
    float d1 = 64.0, d2 = 64.0;

    [unroll]
    for (int j = -2; j <= 2; ++j)
    {
        [unroll]
        for (int i = -2; i <= 2; ++i)
        {
            float2 g = float2(i, j);
            float2 h = Hash22(cell + g);

            float2 o = 0.5 + 0.48 * sin(time + 6.2831853 * h);
            float2 r = g + o - f;

            float along = dot(r, flowDir);
            float2 rr = (r - flowDir * along) + flowDir * (along / stretch);
            float d = dot(rr, rr);

            // Each point retires once 'drop' passes its own random threshold. Its distance is pushed
            // out gradually, so its cell shrinks away and the neighbours grow into the space — cells
            // of very different sizes, and a web that coarsens smoothly as drop rises (no popping,
            // and no zooming of the pattern, which would make it slide across the beach).
            float retire = frac(h.x * 13.7 + h.y * 91.3);
            float gone = smoothstep(retire - 0.1, retire + 0.1, drop);
            d += gone * gone * 6.0;

            if (d < d1) { d2 = d1; d1 = d; }
            else if (d < d2) { d2 = d; }
        }
    }
    d1 = sqrt(d1);
    return float2(sqrt(d2) - d1, d1);
}

// How far a given wave runs up the beach (0.5..1), so no two waves look alike
float WaveStrength(float waveId)
{
    return lerp(0.5, 1.0, frac(sin(waveId * 127.1) * 43758.5453));
}

// Per-band UV rotation to decorrelate cascade tiling patterns
static const float BandAngles[8] = { 0.0, 0.297, 0.645, 0.925, 1.17, 1.42, 1.73, 2.05 };

float2 RotateUV(float2 uv, float angle)
{
    float s, c;
    sincos(angle, s, c);
    return float2(uv.x * c - uv.y * s, uv.x * s + uv.y * c);
}

float3 SampleDisplacement(OceanData ocean, float2 worldXZ, float depthAtten, float camDist)
{
    Texture2DArray<float4> dispTex = ResourceDescriptorHeap[ocean.DisplacementSRV];
    StructuredBuffer<float> tileScales = ResourceDescriptorHeap[ocean.TileScalesSRV];
    float3 totalDisp = 0;

    [loop]
    for (uint i = 0; i < ocean.NumBands; ++i)
    {
        float2 uv = RotateUV(worldXZ, BandAngles[i]) * tileScales[i];
        float mipLevel = log2(max(1.0, camDist * tileScales[i] * 0.15));
        totalDisp += dispTex.SampleLevel(OceanSampler, float3(uv, i), mipLevel).xyz;
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

    // Frustum cull — skip off-screen patches
    if (CullTriangle(p0, p1, p2))
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
    float3 UndisplacedXZ : TEXCOORD1;  // Pre-displacement XZ for PS texture sampling
    float Depth : TEXCOORD2;
    nointerpolation uint InstanceIdx : TEXCOORD3;
    float3 SkyAmbient : TEXCOORD4;     // hemisphere sky ambient (same as composition), constant per frame
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

    // Sample FFT displacement from all bands (with distance-based mips)
    float3 displacement = SampleDisplacement(ocean, worldPos.xz, ocean.DisplacementAtten, camDist);

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
        float outside = length((terrainUV - clampedUV) * float2(ocean.TerrainSizeX, ocean.TerrainSizeZ));
        if (outside < 150.0)
        {
            Texture2D<float> heightmap = ResourceDescriptorHeap[ocean.HeightmapSRV];
            float h = heightmap.SampleLevel(OceanSampler, clampedUV, 0);
            float terrainWorldY = ocean.TerrainBaseY + h * ocean.MaxTerrainHeight;

            // Water depth = ocean surface - terrain surface (positive = water exists)
            float waterDepth = ocean.OceanPlaneY - terrainWorldY;
            float shoreBlend = smoothstep(0, ocean.ShoreDepth, waterDepth);
            float shoreAtten = lerp(ocean.ShoreMinWave, 1.0, shoreBlend);

            displacement *= lerp(shoreAtten, 1.0, smoothstep(0.0, 150.0, outside));

            // Swash surge: the water level near the beach rises and falls in step with the foam bands
            // (same phase as the PS), so the waterline travels up and down the sand instead of standing still.
            if (ocean.NoiseSRV != 0)
            {
                Texture2D<float4> noiseTex = ResourceDescriptorHeap[ocean.NoiseSRV];
                float bandNoise = noiseTex.SampleLevel(OceanSampler, worldPos.xz * 0.011, 0).r;
                float swashZone = 1.0 - smoothstep(0.0, max(0.1, ocean.ShoreFoamDepth), waterDepth);

                // Asymmetric like real swash: a quicker run-up (SurgeRise of the cycle) followed by a
                // slower backwash. 0.5 = symmetric; 0.25 felt too abrupt up and too sluggish back.
                float surgeClock = SurgeClock(ShoreClock(ocean.WaveTime, bandNoise), ocean.ShoreSurge, ocean.ShoreFoamDepth);
                float surgePhase = frac(surgeClock);
                float surge = surgePhase < SurgeRise
                    ? smoothstep(0.0, SurgeRise, surgePhase)
                    : 1.0 - smoothstep(SurgeRise, 1.0, surgePhase);

                // Every wave starts from the same low water (-ShoreSurge) but runs up a different
                // distance. Keeping the low point fixed keeps the level continuous from wave to wave.
                float strength = WaveStrength(floor(surgeClock));
                displacement.y += ocean.ShoreSurge * swashZone * (2.0 * strength * surge - 1.0);
            }
        }
    }

    float3 displaced = worldPos + displacement;

    output.UndisplacedXZ = worldPos.xzx; // Pass XZ for PS sampling (z component unused)
    output.WorldPos = displaced;
    output.Position = mul(mul(float4(displaced, 1.0), View), Projection);
    output.Depth = output.Position.w;
    output.InstanceIdx = idx;
    output.SkyAmbient = GetSkyColor(float3(0, 1, 0), FogSunDirection) * 0.45 * AmbientScale;

    return output;
}

// ═══════════════════════════════════════════════════════════════════════
// PIXEL SHADER — per-pixel normals from FFT slope maps
// ═══════════════════════════════════════════════════════════════════════

struct PSOutput
{
    float4 Color : SV_Target0;
};

PSOutput PS(DSOutput input)
{
    PSOutput output;

    float3 camPos = ViewInverse[3].xyz;

    StructuredBuffer<OceanData> oceanData = ResourceDescriptorHeap[OceanDataIdx];
    OceanData ocean = oceanData[input.InstanceIdx];

    float3 worldPos = input.WorldPos;
    float3 V = normalize(camPos - worldPos);
    float dist = distance(worldPos, camPos);

    // ── Merged per-band sampling: slope + displacement in one loop ──
    Texture2DArray<float4> dispTex = ResourceDescriptorHeap[ocean.DisplacementSRV];
    Texture2DArray<float2> slopeTex = ResourceDescriptorHeap[ocean.SlopeSRV];
    StructuredBuffer<float> tileScales = ResourceDescriptorHeap[ocean.TileScalesSRV];

    float2 totalSlope = 0;
    float totalFoam = 0;
    float waveHeight = 0;
    float2 worldXZ = input.UndisplacedXZ.xy;

    [loop]
    for (uint i = 0; i < ocean.NumBands; ++i)
    {
        float angle = BandAngles[i];
        float scale = tileScales[i];
        float2 uv = RotateUV(worldXZ, angle) * scale;

        // Filtered: the foam alpha is only a coverage mask, the visible detail comes from the foam texture
        float4 disp = dispTex.Sample(OceanSampler, float3(uv, i));
        float2 slope = slopeTex.Sample(OceanSampler, float3(uv, i)).xy;

        // Counter-rotate slopes back to world space
        float s, c;
        sincos(-angle, s, c);
        totalSlope += float2(slope.x * c - slope.y * s, slope.x * s + slope.y * c);
        totalFoam += disp.a;
        waveHeight += disp.y;
    }

    // Build normal from accumulated slopes
    float3 N = normalize(float3(-totalSlope.x, 1.0, -totalSlope.y));
    N = normalize(lerp(float3(0, 1, 0), N, ocean.NormalAtten));

    float NdotV = saturate(dot(N, V));

    // ── Sun vectors ──
    float3 sunDir = normalize(ocean.SunDirection);
    float3 L = normalize(-sunDir);
    float NdotL = saturate(dot(N, L));
    float3 sunRadiance = ocean.SunColor * ocean.SunIntensity;
    sunRadiance *= GetCloudShadow(OceanSampler, worldPos - camPos, L);

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

    float3 scatter = ((k1 + k2) * baseColor + k3 * baseColor + 0.02 * bubbleColor) * sunRadiance;

    // ── Fresnel, sky reflection and sun glitter: shared with lakes and rivers (water_common.fx) ──
    float F = WaterFresnel(NdotV);
    float3 reflectColor = WaterSkyReflection(OceanSampler, V, N, worldPos - camPos, dist, ocean.CloudColor);
    float3 specular = WaterSunSpecular(N, V, L, sunRadiance, F, dist);

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
            shallowFade = 1.0 - smoothstep(0.0, 150.0, outside);
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
    float3 skyAmbient = input.SkyAmbient;
    float shallow = exp(-max(bedDepth, 0.0) / max(0.01, ocean.ShoreFadeDepth)) * saturate(pathLen);
    shallow *= (ocean.HeightmapSRV != 0) ? shallowFade : 1.0;
    body += ocean.ShallowColor * (sunRadiance * saturate(L.y) + skyAmbient) * 0.06 * shallow;

    // ── Combine lighting ──
    float3 color = (1.0 - F) * body + (specular + F * reflectColor) * edgeFade;
    color = max(0.0, color);

    // ── Foam ──
    // Everything below only builds a COVERAGE value; the visible foam is that coverage eroded by a
    // cellular foam texture, so edges break up into bubbles/streaks instead of showing FFT texels.
    float t = ocean.WaveTime;

    // Open water: Jacobian foam from the FFT, faded at distance
    float coverage = saturate(totalFoam) * saturate(1.0 - dist / 3000.0);

    float foamTex = 0.5;
    float streakFoam = 0.0;
    if (ocean.NoiseSRV != 0)
    {
        Texture2D<float4> noiseTex = ResourceDescriptorHeap[ocean.NoiseSRV];

        // Shore: one foam front per wave (see ShoreClock). It crosses the swash zone on the STILL-water
        // depth at a steady speed — on the actual depth, which includes the surge, a front stalls
        // during the backwash and lurches forward with the next run-up. Sharp front on the shallow
        // side, foam trailing off seaward. It only shows where there is water, so once it has caught
        // up with the waterline the run-up edge is this front; its trail is what the backwash drains.
        float bandNoise = noiseTex.SampleLevel(OceanSampler, worldPos.xz * 0.011, 0).r;
        float shoreClock = ShoreClock(t, bandNoise);
        float foamDepth = max(0.1, ocean.ShoreFoamDepth);
        float bandDepth = (stillDepth < 999.0) ? stillDepth : bedDepth;
        float swashZone = 1.0 - smoothstep(0.0, foamDepth, bandDepth);
        float phase = bandDepth / foamDepth + shoreClock;
        float bandPos = frac(phase);                        // 0 at the front, growing seaward
        float waveStrength = lerp(0.55, 1.0, WaveStrength(floor(phase)));   // same wave id as its run-up

        // Solid bubbly foam right behind the front, which thins out into the lacy web further back
        // (its depth varies along the front — frontWidth — so it is a ragged band, not an even ribbon)
        float frontBroad = noiseTex.SampleLevel(OceanSampler, worldPos.xz * 0.045 + t * 0.004, 0).r;
        float frontFine = noiseTex.SampleLevel(OceanSampler, worldPos.xz * 0.21 - t * 0.006, 0).a;
        float frontWidth = lerp(0.5, 2.6, saturate((frontBroad - 0.3) / 0.4)) * lerp(0.7, 1.3, frontFine);
        float band = exp(-bandPos * 12.0 / frontWidth) * smoothstep(0.0, 0.025, bandPos) * waveStrength;
        float trail = 1.0 - bandPos;
        trail = trail * trail * trail * smoothstep(0.02, 0.14, bandPos) * waveStrength * swashZone;

        // Downhill direction and steepness of the beach from the heightmap, over a 3 m baseline so
        // they stay smooth. (Screen-space derivatives of the depth are too noisy: every pixel then
        // gets a slightly different direction and stretched patterns turn into moire.)
        // The slope converts widths given in meters along the beach into water depth, so the foam
        // edge is equally wide on a steep shore and a flat one.
        bool hasBeach = ocean.HeightmapSRV != 0 && terrainDepth < 999.0;
        float2 flowDir = float2(0.0, 1.0);
        float beachSlope = 0.1;
        if (hasBeach && bandDepth < foamDepth)
        {
            Texture2D<float> heightmap = ResourceDescriptorHeap[ocean.HeightmapSRV];
            float2 slopeStep = 3.0 / terrainSize;
            float2 uphill = float2(
                heightmap.SampleLevel(OceanSampler, clampedUV + float2(slopeStep.x, 0), 0)
              - heightmap.SampleLevel(OceanSampler, clampedUV - float2(slopeStep.x, 0), 0),
                heightmap.SampleLevel(OceanSampler, clampedUV + float2(0, slopeStep.y), 0)
              - heightmap.SampleLevel(OceanSampler, clampedUV - float2(0, slopeStep.y), 0));
            float steepness = length(uphill);
            flowDir = -uphill / max(steepness, 1e-6);
            beachSlope = clamp(steepness * ocean.MaxTerrainHeight / 6.0, 0.03, 1.0);
        }

        // Lace along the waterline and around anything standing in the water (piers, rocks, hulls)
        // Its width varies along the coast (a broad noise plus a finer one), so the edge is a ragged
        // band that swells and pinches rather than an even ribbon.
        float edgeWidth = lerp(0.4, 2.4, saturate((frontBroad - 0.3) / 0.4)) * lerp(0.7, 1.3, frontFine);

        float laceDepth = min(bedDepth, bufDepth);
        static const float RunUpLaceWidth = 0.7;      // meters along the beach, before the variation
        float lace = 1.0 - smoothstep(0.0, min(RunUpLaceWidth * edgeWidth * beachSlope, 0.7), laceDepth);

        // The beach waterline follows the surge: a crisp foam edge while the water runs up, which
        // breaks apart during the backwash. Objects standing in deeper water keep their lace.
        float surgePhase = frac(SurgeClock(shoreClock, ocean.ShoreSurge, ocean.ShoreFoamDepth));
        float advancing = max(1.0 - smoothstep(SurgeRise, SurgeRise + 0.2, surgePhase),
                              smoothstep(0.92, 1.0, surgePhase));
        float onBeach = 1.0 - smoothstep(0.0, 0.5, bedDepth);
        static const float BackwashLace = 0.9;  // 0 = no foam on the retreating edge, 1 = same as the run-up
        static const float BackwashLaceWidth = 1.1;   // meters along the beach on the retreat
        float retreatLace = 1.0 - smoothstep(0.0, min(BackwashLaceWidth * edgeWidth * beachSlope, 0.7), laceDepth);
        // On the retreat the edge foam is NOT this world-fixed bubble band: revealing the same pattern
        // in reverse looks like the run-up being rewound. It becomes a dense part of the foam web
        // below, which rides out with the water.
        float retreatEdge = retreatLace * BackwashLace * (1.0 - advancing) * onBeach;
        lace = hasBeach
            ? lerp(lace, lace * advancing, onBeach)
            : lerp(lace, lerp(retreatLace * BackwashLace, lace, advancing), onBeach);   // no heightmap: no web to hand over to

        // Foam web: the lacy network foam stretches into as it thins. Used twice — on the back of the
        // incoming wave (trail) and in the thin film draining off the beach (film).
        float film = (1.0 - smoothstep(0.0, max(0.15, ocean.ShoreSurge * 1.5), bedDepth)) * (1.0 - advancing);

        // ...and it fades away as the water retreats: strongest just after the top of the run-up,
        // gone by the time the next wave arrives.
        float backwashAge = saturate((surgePhase - SurgeRise) / (1.0 - SurgeRise));
        film *= 1.0 - smoothstep(0.1, 0.9, backwashAge);
        float webAmount = max(max(film, retreatEdge), trail);

        if (webAmount > 0.01 && hasBeach)
        {
            // The web floats on the water, so it moves with it: carried seaward as the backwash drains
            // and back in on the next run-up (straight up and down the slope; a sideways slide was
            // tried and removed). The shift follows the
            // surge's own profile, so it is continuous and returns to zero every cycle — a bounded
            // offset, which is why a flow direction that bends along the coast cannot shear it apart.
            float surgeShape = surgePhase < SurgeRise
                ? smoothstep(0.0, SurgeRise, surgePhase)
                : 1.0 - smoothstep(SurgeRise, 1.0, surgePhase);
            float waterTravel = min(2.0 * ocean.ShoreSurge / beachSlope, 8.0) * 0.6 * swashZone;
            float2 waterShift = flowDir * waterTravel * (1.0 - surgeShape);
            float2 foamPos = worldPos.xz - waterShift;
            // The borders between cells of animated Voronoi noise (FoamWeb). Iso-contours of blob noise
            // can only close into rings; the negative space between cells is connected.
            const float webCellSize = 0.95;    // meters across a cell (before stretching and merging)
            const float webStretch = 2.0;      // cells are this much longer along the downhill flow
            const float webRate = 0.9;         // how fast the cell points drift (radians per second of wave time)

            // Bend the space first: a large slow warp squeezes some regions and opens others (uneven
            // cell sizes), a medium one curves the strands, a little fine wobble roughens them.
            float2 warpUV = foamPos * 0.19;
            float2 bigWarp = float2(noiseTex.SampleLevel(OceanSampler, warpUV * 0.31 + 0.27, 0).r,
                                    noiseTex.SampleLevel(OceanSampler, warpUV * 0.31 + 0.83, 0).r) - 0.5;
            float2 warp = float2(noiseTex.SampleLevel(OceanSampler, warpUV + t * 0.006, 0).r,
                                 noiseTex.SampleLevel(OceanSampler, warpUV + 0.43 - t * 0.005, 0).r) - 0.5;
            float2 wobble = float2(noiseTex.SampleLevel(OceanSampler, warpUV * 2.9 + 0.11, 0).a,
                                   noiseTex.SampleLevel(OceanSampler, warpUV * 2.9 + 0.71, 0).a) - 0.5;
            float2 webPos = foamPos + bigWarp * 4.5 + warp * 1.8 + wobble * 0.12;

            // Main web as a SOFT mask: 1 on the strand's center line, falling off to 0 at its edge.
            // Strands swell into knots where cells meet. Thin hard lines of even brightness read as
            // drawn cracks, however much they bend.
            // A fixed third of the points is retired, for uneven cell sizes. Raising this over the
            // foam's life (cells merging as it ages) was tried and rejected: the collapsing cells look
            // worse than a web that simply fades.
            const float webDrop = 0.33;

            float2 web = FoamWeb(webPos / webCellSize, flowDir, webStretch, t * webRate, webDrop);
            float webLace = saturate(1.0 - web.x / (0.09 + 0.17 * smoothstep(0.4, 0.9, web.y)));

            // A finer, fainter web inside the cells for mixed sizes
            float2 fine = FoamWeb(webPos / (webCellSize * 0.4) + 17.3, flowDir, webStretch, t * webRate * 1.3, webDrop);
            webLace = max(webLace, saturate(1.0 - fine.x / 0.11) * 0.6 * smoothstep(0.2, 0.55, web.x));

            // Patchy: the web thins out and tears open in places instead of covering the water evenly
            float patches = noiseTex.SampleLevel(OceanSampler, foamPos * 0.06 - t * 0.004, 0).b;
            webLace *= lerp(0.25, 1.0, smoothstep(0.38, 0.62, patches));

            // Strands are made of bubbles: erode the soft mask with the fine cell texture, so the
            // strand cores stay solid and the edges break into ragged clusters.
            // (cells ~7 cm; the noise texture has no mips, so flatten it before it starts to shimmer)
            float bubbles = noiseTex.SampleLevel(OceanSampler, foamPos * 1.75 - t * float2(0.013, 0.004), 0).g;
            bubbles = saturate((bubbles - 0.15) / 0.55);
            bubbles = lerp(bubbles, 0.5, saturate(dist / 45.0));
            float webFoam = saturate((webLace * 1.35 - (1.0 - bubbles) * 0.75) * 2.2);

            // The retreating edge: the web closes up into a torn sheet toward the waterline
            float webFoamEdge = saturate(((webLace * 2.2 + 0.45) * 1.35 - (1.0 - bubbles) * 0.75) * 2.2);
            webFoam = lerp(webFoam, webFoamEdge, retreatEdge);

            streakFoam = webFoam * webAmount;
        }

        // Stays below full coverage so the foam texture always breaks it up
        coverage = saturate(coverage + saturate(band * saturate(swashZone * 1.8) * 1.8 + lace) * 0.8 * ocean.ShoreFoam);

        // Foam texture: two drifting scales of cellular noise (noise texture has no mips: flatten it far away)
        float cells1 = noiseTex.SampleLevel(OceanSampler, worldPos.xz * 0.23 + t * float2(0.010, 0.006), 0).g;
        float cells2 = noiseTex.SampleLevel(OceanSampler, worldPos.xz * 0.83 - t * float2(0.013, 0.004), 0).g;
        foamTex = saturate((cells1 * 0.6 + cells2 * 0.4 - 0.15) / 0.6);

        // Backwash filaments: translucent, lightly broken up by the fine cells
        streakFoam *= 0.85 * lerp(0.75, 1.0, cells2) * ocean.ShoreFoam;
    }

    // Thin foam is translucent, thick foam keeps the cell structure as shading.
    // The noise texture has no mips: far away the cells would only alias, so fade to a soft
    // version of the coverage itself (whitecaps become faint streaks instead of solid blobs).
    float foamFar = saturate(dist / 150.0);
    float foam = saturate((coverage * 1.25 - (1.0 - foamTex)) * 3.0);
    foam = lerp(foam, saturate((coverage - 0.35) * 1.5) * 0.7, foamFar);
    foam = max(foam, streakFoam * (1.0 - foamFar));

    // Always feather the last few centimeters of water: whatever the foam pattern, it must not end
    // in the hard pixel edge where the water surface cuts into the beach.
    foam *= smoothstep(0.0, 0.025, pathLen);
    foamTex = lerp(foamTex, 0.5, foamFar);
    float3 foamAlbedo = float3(0.80, 0.80, 0.78) * lerp(0.7, 1.0, foamTex);
    float3 foamLit = foamAlbedo * ((0.27 + 0.5 * NdotL) * sunRadiance + skyAmbient * 1.15);
    color = lerp(color, foamLit, foam * 0.9);

    // ── Atmospheric extinction — shared aerial perspective ──
    if (FogEnabled > 0)
    {
        color = ApplyAerialPerspective(color, worldPos, camPos, FogSunDirection);
    }

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

    float3 displacement = SampleDisplacement(ocean, worldPos.xz, 1.0, 0.0);
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
