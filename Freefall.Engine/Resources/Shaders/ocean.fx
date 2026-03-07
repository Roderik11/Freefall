#include "common.fx"
#include "sky_common.fx"
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
    uint NoiseSRV;              // bindless SRV for noise texture (Perlin+Worley)
    float2 InvViewportSize;     // 1.0 / viewport dimensions (replaces GetDimensions)
    float3 HorizonSkyColor;     // precomputed GetSkyColor at horizon direction
    float _pad3;
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

        // Sample displacement (wave height + foam) and slope in one pass
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

    float H = max(0.0, waveHeight);

    // ── Base color: dark deep ocean ──
    float depthBlend = saturate(dist / 4000.0);
    float3 baseColor = lerp(ocean.OceanColor, ocean.DeepColor, depthBlend);

    // ── Subsurface scattering ──
    float3 scatterColor = float3(0.016, 0.074, 0.16);
    float3 bubbleColor = float3(0.0, 0.02, 0.016);

    float k1 = 0.3 * H * pow(saturate(dot(L, -V)), 4.0)
             * pow(0.5 - 0.5 * dot(L, N), 3.0);
    float k2 = 0.1 * NdotV * NdotV;
    float k3 = 0.08 * NdotL;

    float3 scatter = ((k1 + k2) * scatterColor + k3 * scatterColor + 0.02 * bubbleColor) * sunRadiance;

    // ── Fresnel (Schlick, precomputed for F0=0.02, roughness=0.075) ──
    float base_f = 1.0 - NdotV;
    float F = saturate(0.02 + 0.668 * pow(base_f, 4.08));

    // ── Environment reflection ──
    float reflSmooth = saturate(dist * 0.003);
    float3 reflectDir = reflect(-V, N);
    reflectDir.y = abs(reflectDir.y);
    float3 skyRefl = GetSkyColor(reflectDir, FogSunDirection);
    float3 reflectColor = lerp(skyRefl, ocean.HorizonSkyColor, reflSmooth);

    // ── GGX sun specular (widen at distance to reduce tessellation sparkle) ──
    float waterRough = 0.075;
    float distRough = lerp(waterRough, 0.6, reflSmooth);
    float3 halfDir = normalize(L + V);
    float NdotH = max(0.0001, dot(N, halfDir));
    float a = distRough * distRough;
    float a2 = a * a;
    float dGGX = (NdotH * NdotH) * (a2 - 1.0) + 1.0;
    float D = a2 / max(3.14159 * dGGX * dGGX, 1e-4);
    float3 specular = sunRadiance * F * D * NdotL;
    specular /= max(0.001, 4.0 * max(0.001, NdotL));
    specular *= NdotL;

    // ── Combine lighting ──
    float3 color = (1.0 - F) * scatter + specular + F * reflectColor;
    color = max(0.0, color);

    // ── Foam from FFT Jacobian ──
    float foam = saturate(totalFoam);
    float3 foamColor = float3(0.6, 0.557, 0.492);
    float3 foamLit = foamColor * (0.3 + 0.7 * NdotL) * sunRadiance;
    color = lerp(color, foamLit, foam);

    // ── Atmospheric extinction / horizon haze ──
    float horizonFade = saturate(dist / 6000.0);
    float3 hazeColor = GetSkyColor(float3(0, 0.01, 1), FogSunDirection);
    float hazeAmount = horizonFade * horizonFade * 0.85;
    color = lerp(color, hazeColor, hazeAmount);

    // ── Shore pixel effects ──
    if (ocean.DepthGBufferSRV != 0)
    {
        Texture2D<float> depthGB = ResourceDescriptorHeap[ocean.DepthGBufferSRV];
        float2 screenUV = input.Position.xy * ocean.InvViewportSize;

        float sceneLinearZ = depthGB.SampleLevel(OceanSampler, screenUV, 0);
        float oceanLinearZ = input.Depth;

        float depthDiff = (sceneLinearZ > 0) ? (sceneLinearZ - oceanLinearZ) : 1000.0;
        float shoreProximity = saturate(depthDiff / max(0.01, ocean.ShoreFadeDepth));

        // ── Shore foam ──
        float shoreFoamAmount = 0;
        if (ocean.NoiseSRV != 0)
        {
            Texture2D<float4> noiseTex = ResourceDescriptorHeap[ocean.NoiseSRV];
            float t = ocean.WaveTime;

            float2 shapeUV = worldPos.xz * 0.02 + float2(t * 0.008, t * 0.005);
            float foamShape = noiseTex.SampleLevel(OceanSampler, shapeUV, 0).r;
            float foamDetail = noiseTex.SampleLevel(OceanSampler, worldPos.xz * 0.15, 0).g;

            float foamThresh = lerp(0.3, 0.9, shoreProximity);
            float shoreFoam = smoothstep(foamThresh, foamThresh + 0.1, foamShape);
            float foamMask = 1.0 - smoothstep(0, 0.5, shoreProximity);
            float edgeFoam = (1.0 - smoothstep(0, 0.05, shoreProximity)) * (foamDetail * 0.3 + 0.7);
            shoreFoamAmount = saturate(shoreFoam * foamMask * (0.4 + 0.6 * foamDetail) + edgeFoam);
        }

        // ── Terrain show-through ──
        if (ocean.CompositeSRV != 0)
        {
            Texture2D<float4> compositeBuffer = ResourceDescriptorHeap[ocean.CompositeSRV];
            float2 refrUV = screenUV + N.xz * ocean.RefractionStrength * (1.0 - shoreProximity);
            refrUV = saturate(refrUV);
            float3 terrainColor = compositeBuffer.SampleLevel(OceanSampler, refrUV, 0).rgb;

            float absorption = shoreProximity * shoreProximity;
            float3 tintedTerrain = lerp(terrainColor, terrainColor * ocean.ShallowColor, absorption);

            float terrainVisibility = (1.0 - shoreProximity);
            terrainVisibility *= terrainVisibility;
            terrainVisibility *= (1.0 - shoreFoamAmount);
            color = lerp(color, tintedTerrain, terrainVisibility);
        }

        // ── Apply shore foam on top ──
        color = lerp(color, foamLit, shoreFoamAmount);
    }

    // ── Distance fog — matches deferred composition fog ──
    if (FogEnabled > 0)
    {
        float3 fogColor = GetSkyColor(float3(0, 0.01, 1), FogSunDirection);
        color = FOG(color, dist, fogColor);
    }

    // ── Tone mapping + gamma — match deferred composition pipeline ──
    // ACES Filmic (Narkowicz 2015)
    float3 tm = color;
    color = (tm * (2.51 * tm + 0.03)) / (tm * (2.43 * tm + 0.59) + 0.14);
    color = saturate(color);
    color = pow(abs(color), 1.0 / 2.2);

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
