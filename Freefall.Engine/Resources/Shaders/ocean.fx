#include "common.fx"
// @RenderState(RenderTargets=1)

// Standard push constant layout — MUST match gbuffer.fx exactly
#define OceanDataIdx GET_INDEX(1)
#define DescriptorBufIdx GET_INDEX(2)
#define SortedIndicesIdx GET_INDEX(4)
#define IndexBufferIdx GET_INDEX(7)
#define BaseIndex GET_INDEX(8)
#define PosBufferIdx GET_INDEX(9)
#define NormBufferIdx GET_INDEX(10)
#define UVBufferIdx GET_INDEX(11)
#define InstanceBaseOffset GET_INDEX(13)

struct InstanceDescriptor
{
    uint TransformSlot;
    uint MaterialId;
    uint Padding;
};

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

// Per-band UV rotation to decorrelate cascade tiling patterns
static const float BandAngles[8] = { 0.0, 0.297, 0.645, 0.925, 1.17, 1.42, 1.73, 2.05 };

float2 RotateUV(float2 uv, float angle)
{
    float s, c;
    sincos(angle, s, c);
    return float2(uv.x * c - uv.y * s, uv.x * s + uv.y * c);
}

float3 SampleDisplacement(OceanData ocean, float2 worldXZ, float depthAtten)
{
    Texture2DArray<float4> dispTex = ResourceDescriptorHeap[ocean.DisplacementSRV];
    StructuredBuffer<float> tileScales = ResourceDescriptorHeap[ocean.TileScalesSRV];
    float3 totalDisp = 0;

    [loop]
    for (uint i = 0; i < ocean.NumBands; ++i)
    {
        float2 uv = RotateUV(worldXZ, BandAngles[i]) * tileScales[i];
        totalDisp += dispTex.SampleLevel(OceanSampler, float3(uv, i), 0).xyz;
    }

    return totalDisp * depthAtten;
}

float3 SampleNormal(OceanData ocean, float2 worldXZ, float normalStrength, float depthAtten, float camDist)
{
    Texture2DArray<float2> slopeTex = ResourceDescriptorHeap[ocean.SlopeSRV];
    StructuredBuffer<float> tileScales = ResourceDescriptorHeap[ocean.TileScalesSRV];
    float2 totalSlope = 0;

    [loop]
    for (uint i = 0; i < ocean.NumBands; ++i)
    {
        float angle = BandAngles[i];
        float2 uv = RotateUV(worldXZ, angle) * tileScales[i];
        float2 slope = slopeTex.SampleLevel(OceanSampler, float3(uv, i), 0).xy;

        // Distance-based band fadeout: high-frequency bands alias at distance
        // tileScales[i] is proportional to frequency — higher = finer detail = fades sooner
        float bandFade = saturate(1.0 - camDist * tileScales[i] * 0.01);
        slope *= bandFade;

        // Counter-rotate slopes back to world space
        float s, c;
        sincos(-angle, s, c);
        slope = float2(slope.x * c - slope.y * s, slope.x * s + slope.y * c);

        totalSlope += slope;
    }

    totalSlope *= normalStrength;
    float3 normal = normalize(float3(-totalSlope.x, 1.0, -totalSlope.y));
    return normalize(lerp(float3(0, 1, 0), normal, depthAtten));
}

float SampleFoam(OceanData ocean, float2 worldXZ)
{
    Texture2DArray<float4> dispTex = ResourceDescriptorHeap[ocean.DisplacementSRV];
    StructuredBuffer<float> tileScales = ResourceDescriptorHeap[ocean.TileScalesSRV];
    float totalFoam = 0;

    [loop]
    for (uint i = 0; i < ocean.NumBands; ++i)
    {
        float2 uv = RotateUV(worldXZ, BandAngles[i]) * tileScales[i];
        totalFoam += dispTex.SampleLevel(OceanSampler, float3(uv, i), 0).a;
    }

    return saturate(totalFoam);
}

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
    StructuredBuffer<uint> indices = ResourceDescriptorHeap[IndexBufferIdx];
    uint vertexID = indices[primitiveVertexID + BaseIndex];
    StructuredBuffer<float3> positions = ResourceDescriptorHeap[PosBufferIdx];

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

    // Sample FFT displacement from all bands
    float3 displacement = SampleDisplacement(ocean, worldPos.xz, ocean.DisplacementAtten);

    // ── Shore attenuation: sample terrain heightmap for world-space water depth ──
    if (ocean.HeightmapSRV != 0)
    {
        // Convert world XZ to terrain UV [0,1]
        float2 terrainUV = float2(
            (worldPos.x - ocean.TerrainOriginX) / ocean.TerrainSizeX,
            (worldPos.z - ocean.TerrainOriginZ) / ocean.TerrainSizeZ);

        // Only attenuate inside terrain bounds
        if (all(terrainUV >= 0) && all(terrainUV <= 1))
        {
            Texture2D<float> heightmap = ResourceDescriptorHeap[ocean.HeightmapSRV];
            float h = heightmap.SampleLevel(OceanSampler, terrainUV, 0);
            float terrainWorldY = ocean.TerrainBaseY + h * ocean.MaxTerrainHeight;

            // Water depth = ocean surface - terrain surface (positive = water exists)
            float waterDepth = ocean.OceanPlaneY - terrainWorldY;
            float shoreBlend = smoothstep(0, ocean.ShoreDepth, waterDepth);
            float shoreAtten = lerp(ocean.ShoreMinWave, 1.0, shoreBlend);
            displacement *= shoreAtten;
        }
    }

    float3 displaced = worldPos + displacement;

    output.UndisplacedXZ = worldPos.xzx; // Pass XZ for PS sampling (z component unused)
    output.WorldPos = displaced;
    output.Position = mul(mul(float4(displaced, 1.0), View), Projection);
    output.Depth = output.Position.w;
    output.InstanceIdx = idx;

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

    // ── Per-pixel normal from FFT slope maps ──
    float3 N = SampleNormal(ocean, input.UndisplacedXZ.xy, 1.0, ocean.NormalAtten, dist);

    float NdotV = saturate(dot(N, V));

    // ── Sun vectors ──
    float3 sunDir = normalize(ocean.SunDirection);
    float3 L = normalize(-sunDir);
    float NdotL = saturate(dot(N, L));

    // ── Wave height for scatter/foam ──
    Texture2DArray<float4> dispTex = ResourceDescriptorHeap[ocean.DisplacementSRV];
    StructuredBuffer<float> tileScalesPS = ResourceDescriptorHeap[ocean.TileScalesSRV];
    float waveHeight = 0;
    [loop] for (uint b = 0; b < ocean.NumBands; ++b)
        waveHeight += dispTex.SampleLevel(OceanSampler, float3(input.UndisplacedXZ.xy * tileScalesPS[b], b), 0).y;
    float H = max(0.0, waveHeight);

    // ── Base color: dark deep ocean ──
    float depthBlend = saturate(dist / 4000.0);
    float3 baseColor = lerp(ocean.OceanColor, ocean.DeepColor, depthBlend);

    // ── Subsurface scattering (from GarrettGunnell reference) ──
    float3 scatterColor = float3(0.016, 0.074, 0.16);
    float3 bubbleColor = float3(0.0, 0.02, 0.016);

    // k1: wave-peak forward scatter — light transmitting through thin wave crests
    float k1 = 0.3 * H * pow(saturate(dot(L, -V)), 4.0)
             * pow(0.5 - 0.5 * dot(L, N), 3.0);
    // k2: view-dependent scatter — ocean glows when looking into the surface
    float k2 = 0.1 * pow(NdotV, 2.0);
    // k3: shadow-side scatter
    float k3 = 0.08 * NdotL;

    float3 scatter = (k1 + k2) * scatterColor * ocean.SunColor * ocean.SunIntensity
                   + k3 * scatterColor * ocean.SunColor * ocean.SunIntensity
                   + 0.02 * bubbleColor * ocean.SunColor * ocean.SunIntensity;

    // ── Fresnel (Schlick, F0 = 0.02 for water) ──
    // Roughness-adjusted Fresnel (Schlick-Roughness approximation)
    float waterRough = 0.075;
    float base_f = 1.0 - NdotV;
    float numerator = pow(base_f, 5.0 * exp(-2.69 * waterRough));
    float F = 0.02 + (1.0 - 0.02) * numerator / (1.0 + 22.7 * pow(waterRough, 1.5));
    F = saturate(F);

    // ── Environment reflection ──
    // These dominate the ocean appearance via Fresnel — must be saturated blue
    float3 skyColor = float3(0.25, 0.45, 0.75);
    float3 horizonSkyColor = float3(0.5, 0.65, 0.85);
    float3 reflectColor = lerp(skyColor, horizonSkyColor, pow(1.0 - saturate(N.y), 3.0));

    // ── GGX sun specular (Beckmann-like for broader highlight) ──
    float3 halfDir = normalize(L + V);
    float NdotH = max(0.0001, dot(N, halfDir));
    float a = waterRough * waterRough;
    float a2 = a * a;
    float dGGX = (NdotH * NdotH) * (a2 - 1.0) + 1.0;
    float D = a2 / max(3.14159 * dGGX * dGGX, 1e-4);
    float3 specular = ocean.SunColor * ocean.SunIntensity * F * D * NdotL;
    specular /= max(0.001, 4.0 * max(0.001, NdotL));
    specular *= NdotL;

    // ── Combine lighting ──
    float3 color = (1.0 - F) * scatter + specular + F * reflectColor;
    color = max(0.0, color);

    // ── Foam from FFT Jacobian (stored in displacement alpha) ──
    float foam = SampleFoam(ocean, input.UndisplacedXZ.xy);
    float3 foamColor = float3(0.6, 0.557, 0.492); // warm off-white foam (GarrettGunnell)
    // Foam roughens the specular — increase roughness, add diffuse
    float foamRough = lerp(waterRough, 0.8, foam);
    float3 foamLit = foamColor * (0.3 + 0.7 * NdotL) * ocean.SunColor * ocean.SunIntensity;
    color = lerp(color, foamLit, saturate(foam));

    // ── Atmospheric extinction / horizon haze ──
    float horizonFade = saturate(dist / 6000.0);
    float3 hazeColor = float3(0.45, 0.55, 0.72); // steel blue haze
    float hazeAmount = horizonFade * horizonFade * 0.85;
    color = lerp(color, hazeColor, hazeAmount);

    // ── Shore pixel effects: shallow color, shore foam, terrain show-through ──
    if (ocean.DepthGBufferSRV != 0)
    {
        Texture2D<float> depthGB = ResourceDescriptorHeap[ocean.DepthGBufferSRV];
        float2 screenUV = input.Position.xy;
        uint dw, dh;
        depthGB.GetDimensions(dw, dh);
        screenUV /= float2(dw, dh);

        float sceneLinearZ = depthGB.SampleLevel(OceanSampler, screenUV, 0);
        float oceanLinearZ = input.Depth;

        // Shore proximity: 0 = at shore edge, 1 = deep water
        float depthDiff = (sceneLinearZ > 0) ? (sceneLinearZ - oceanLinearZ) : 1000.0;
        float shoreProximity = saturate(depthDiff / max(0.01, ocean.ShoreFadeDepth));
        float shallowBlend = 1.0 - shoreProximity;

        // Terrain show-through: transparent shallow water with refraction
        if (ocean.CompositeSRV != 0)
        {
            Texture2D<float4> compositeBuffer = ResourceDescriptorHeap[ocean.CompositeSRV];
            float2 refrUV = screenUV + N.xz * ocean.RefractionStrength * shallowBlend;
            refrUV = saturate(refrUV);
            float3 terrainColor = compositeBuffer.SampleLevel(OceanSampler, refrUV, 0).rgb;
            // Tint seabed with shallow water color (depth-dependent absorption)
            float3 tintedTerrain = lerp(terrainColor, terrainColor * ocean.ShallowColor, shallowBlend);
            // Strong linear blend: shallow = terrain, deep = ocean
            color = lerp(color, tintedTerrain, shallowBlend);
        }

        // Shore foam: animated foam line that masks the hard waterline edge
        // Use world-space noise for organic, irregular foam shape
        float foamEdge = 1.0 - smoothstep(0, 0.4, shoreProximity); // peaks at waterline, fades by 40% depth
        float foamNoise = frac(sin(dot(worldPos.xz * 0.5, float2(12.9898, 78.233))) * 43758.5453);
        float foamWave = sin(worldPos.x * 0.3 + worldPos.z * 0.2 + ocean.WaveTime * 1.5) * 0.5 + 0.5;
        float shoreFoamAmount = foamEdge * saturate(foamNoise * 0.6 + foamWave * 0.5);
        color = lerp(color, foamLit, saturate(shoreFoamAmount));
    }

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

    float3 displacement = SampleDisplacement(ocean, worldPos.xz, 1.0);
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
