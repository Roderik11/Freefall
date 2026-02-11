#include "common.fx"
// @RenderState(RenderTargets=4, DepthWrite=false, DepthFunc=LessEqual, CullMode=None)

// Unified Push Constant Layout (Slots 2-15) - matches gbuffer.fx
// Used by all batched geometry shaders (gbuffer, gbuffer_skinned, mesh_skybox)
// Slots 0-1: Reserved for light/composition passes (texture indices)
#define DescriptorBufIdx GET_INDEX(2)    // StructuredBuffer<InstanceDescriptor> - per-instance descriptor
#define Reserved0Idx GET_INDEX(3)         // Reserved (was MaterialIDsIdx)
#define SortedIndicesIdx GET_INDEX(4)    // StructuredBuffer<uint> - sorted draw order indices
#define BoneWeightsIdx GET_INDEX(5)      // Unused for skybox (0)
#define BonesIdx GET_INDEX(6)            // Unused for skybox (0)
#define IndexBufferIdx GET_INDEX(7)      // StructuredBuffer<uint> - mesh index buffer
#define BaseIndex GET_INDEX(8)           // Base index offset into index buffer
#define PosBufferIdx GET_INDEX(9)        // StructuredBuffer<float3> - vertex positions
#define NormBufferIdx GET_INDEX(10)      // StructuredBuffer<float3> - vertex normals
#define UVBufferIdx GET_INDEX(11)        // StructuredBuffer<float2> - vertex UVs
#define NumBones GET_INDEX(12)           // Unused for skybox
#define InstanceBaseOffset GET_INDEX(13) // Base offset for instance ID (per-command)
#define MaterialsIdx GET_INDEX(14)       // Index to materials buffer
#define GlobalTransformBufferIdx GET_INDEX(15) // Index to global TransformBuffer

// SceneConstants (b0) from common.fx: Time, View, Projection, ViewProjection, ViewInverse

cbuffer ObjectConstants : register(b1)
{
    float4x4 World;
    float3 SunDirection; // direction to sun
    float TimeOfDay; // 0-24
    float CloudCoverage; // 0-1
    float CloudTime;
    float CloudSpeed; // speed multiplier
    float SunIntensity;
    float StarDensity;
    float StarBrightness;
}



struct VertexOutput
{
    float4 Position : SV_POSITION;
    float3 UV : TEXCOORD0;
    float3 ViewDir : TEXCOORD2;
};

struct FragmentOutput
{
    float4 Albedo : SV_TARGET0;
    float4 Normal : SV_TARGET1;
    float4 Data : SV_TARGET2;
    float4 Depth : SV_TARGET3;
    float fDepth : SV_DEPTH;
};

// ────────────────────────────────────────────────
// Noise functions
float hash(float n)
{
    return frac(sin(n) * 43758.5453123);
}

float hash13(float3 p3)
{
    p3 = frac(p3 * 0.1031);
    p3 += dot(p3, p3.zyx + 31.32);
    return frac((p3.x + p3.y) * p3.z);
}

float noise(float3 x)
{
    float3 p = floor(x);
    float3 f = frac(x);
    f = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

    float n = p.x + p.y * 57.0 + 113.0 * p.z;
    return lerp(
        lerp(lerp(hash(n + 0.0), hash(n + 1.0), f.x),
             lerp(hash(n + 57.0), hash(n + 58.0), f.x), f.y),
        lerp(lerp(hash(n + 113.0), hash(n + 114.0), f.x),
             lerp(hash(n + 170.0), hash(n + 171.0), f.x), f.y), f.z);
}

float fbm(float3 p, int octaves)
{
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;

    for (int i = 0; i < octaves; i++)
    {
        value += amplitude * noise(p * frequency);
        frequency *= 2.0;
        amplitude *= 0.5;
    }
    return value;
}

float worley(float3 p)
{
    float3 id = floor(p);
    float3 fd = frac(p);

    float minDist = 1.0;

    for (int z = -1; z <= 1; z++)
        for (int y = -1; y <= 1; y++)
            for (int x = -1; x <= 1; x++)
            {
                float3 offset = float3(x, y, z);
                float3 cellId = id + offset;
                float3 cellPoint = float3(
                    hash13(cellId),
                    hash13(cellId + 100.0),
                    hash13(cellId + 200.0)
                );

                float3 diff = offset + cellPoint - fd;
                float dist = length(diff);
                minDist = min(minDist, dist);
            }

    return minDist;
}

float2 CloudDomainWarp(float2 p, float t)
{
    float wx = fbm(float3(p * 0.25, t * 0.02), 3) - 0.5;
    float wz = fbm(float3(p * 0.25 + 19.1, t * 0.02 + 7.3), 3) - 0.5;
    return float2(wx, wz);
}

// ────────────────────────────────────────────────
// Main cloud function
float GetClouds(float3 viewDir)
{
    if (viewDir.y <= 0.001)
        return 0.0;

    const float cloudHeight = 1800.0;
    float t = cloudHeight / viewDir.y;
    float2 cloudPos = viewDir.xz * t;
    float2 uv = cloudPos * 0.0006;

    float2 wind = float2(CloudTime * CloudSpeed * 0.015, CloudTime * CloudSpeed * 0.008);
    uv += wind;

    float2 warp = CloudDomainWarp(uv, CloudTime) * 0.35;
    float2 p = uv + warp;

    float coverageEvolve = (fbm(float3(p * 0.12, CloudTime * 0.005) + 200.0, 3) - 0.5) * 0.25;
    float coverage = saturate(CloudCoverage + coverageEvolve);

    float c = fbm(float3(p * 2.0, CloudTime * 0.02), 8);
    c = smoothstep(1.0 - coverage, 1.0, c);

    float fineDetail = fbm(float3(p * 8.0, CloudTime * 0.05), 4) * 0.5;
    c = c * (0.6 + fineDetail * 0.4);

    float worleyNoise = worley(float3(p * 4.0, CloudTime * 0.03));
    c = c * (0.7 + worleyNoise * 0.3);

    float turbulence = fbm(float3(p * 12.0, CloudTime * 0.06), 3);
    float edgeDetail = pow(c, 0.5) * turbulence * 0.4;
    c = saturate(c + edgeDetail);

    float microDetail = noise(float3(p * 32.0, CloudTime * 0.12)) * 0.15;
    c = saturate(c + microDetail * c);

    c = smoothstep(0.08, 0.92, c);
    float thickness = fbm(float3(p * 1.5, CloudTime * 0.015) + 100.0, 3);
    c *= (0.7 + thickness * 0.3);

    c *= smoothstep(0.0, 0.28, viewDir.y);

    return c;
}

float3 GetStars(float3 viewDir, float nightFactor)
{
    float upMask = smoothstep(0.02, 0.25, viewDir.y);
    float visibility = saturate(nightFactor) * upMask;

    float3 d0 = normalize(viewDir);
    float starScale = lerp(120.0, 420.0, saturate(StarDensity));
    float3 p = d0 * starScale;

    float3 cell = floor(p);
    float3 fracP = frac(p);

    float3 jitter = float3(
        hash13(cell + float3(1.0, 0.0, 0.0)),
        hash13(cell + float3(0.0, 1.0, 0.0)),
        hash13(cell + float3(0.0, 0.0, 1.0))
    );

    float3 delta = fracP - jitter;
    float dist = length(delta);

    float r = lerp(0.12, 0.30, hash13(cell + 7.0)) / starScale;
    float aa = max(fwidth(dist), 1e-4);

    float gate = step(lerp(0.995, 0.90, saturate(StarDensity)), hash13(cell + 13.0));
    float disc = 1.0 - smoothstep(r - aa, r + aa, dist);
    float star = gate * disc;

    float bVar = lerp(0.4, 1.0, hash13(cell + 21.0));
    float twinkle = 0.75 + 0.25 * sin(CloudTime * 2.0 + hash13(cell + 31.0) * 50.0);

    float intensity = StarBrightness * visibility * bVar * twinkle;
    return star * intensity;
}

float3 GetSkyColor(float3 viewDir, float3 sunDir)
{
    float horizon = abs(viewDir.y);

    float3 dayZenith = float3(0.2, 0.5, 1.0);
    float3 dayHorizon = float3(0.6, 0.8, 1.0);
    float3 sunsetZenith = float3(0.4, 0.3, 0.6);
    float3 sunsetHorizon = float3(1.0, 0.5, 0.3);
    float3 nightZenith = float3(0.01, 0.01, 0.05);
    float3 nightHorizon = float3(0.05, 0.05, 0.1);

    float sunElevation = sunDir.y;

    float dayFactor = saturate((sunElevation - 0.0) / 0.8);
    dayFactor = pow(dayFactor, 0.7);

    float sunsetFactor = 0.0;
    if (sunElevation < 0.15 && sunElevation > -0.2)
    {
        sunsetFactor = 1.0 - abs((sunElevation - (-0.025)) / 0.175);
        sunsetFactor = max(0.0, sunsetFactor);
    }

    float nightFactor = saturate((-sunElevation - 0.15) / 0.3);

    float total = dayFactor + sunsetFactor + nightFactor;
    if (total > 0.0)
    {
        dayFactor /= total;
        sunsetFactor /= total;
        nightFactor /= total;
    }

    float3 zenithColor = dayZenith * dayFactor + sunsetZenith * sunsetFactor + nightZenith * nightFactor;
    float3 horizonColor = dayHorizon * dayFactor + sunsetHorizon * sunsetFactor + nightHorizon * nightFactor;

    float3 skyColor = lerp(horizonColor, zenithColor, pow(horizon, 0.5));

    float sunAngle = dot(viewDir, sunDir);
    float horizonScatter = pow(1.0 - horizon, 3.0);
    float sunScatter = pow(saturate(sunAngle), 3.0);
    float scatter = (horizonScatter + sunScatter * 0.5) * saturate(dayFactor + sunsetFactor * 0.5);
    skyColor += float3(1.0, 0.8, 0.6) * scatter * 0.3;

    skyColor += GetStars(viewDir, nightFactor);

    return skyColor;
}

float GetSun(float3 viewDir, float3 sunDir)
{
    float sun = saturate(dot(viewDir, sunDir));
    float sunDisc = smoothstep(0.9995, 0.9998, sun);
    float sunGlow = pow(sun, 32.0) * 0.3;
    return sunDisc + sunGlow;
}

VertexOutput VS(uint primitiveVertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
    VertexOutput output;

    // Per-instance descriptor (matches C# InstanceDescriptor: 12 bytes)
    struct InstanceDescriptor
    {
        uint TransformSlot;
        uint MaterialId;
        uint CustomDataIdx;
    };

    // Bindless index buffer - primitiveVertexID is 0 to N-1, add BaseIndex to offset into correct mesh part
    StructuredBuffer<uint> indices = ResourceDescriptorHeap[IndexBufferIdx];
    uint vertexID = indices[primitiveVertexID + BaseIndex];

    StructuredBuffer<float3> positions = ResourceDescriptorHeap[PosBufferIdx];
    StructuredBuffer<row_major matrix> globalTransforms = ResourceDescriptorHeap[GlobalTransformBufferIdx];
    StructuredBuffer<uint> sortedIndices = ResourceDescriptorHeap[SortedIndicesIdx];
    StructuredBuffer<InstanceDescriptor> descriptors = ResourceDescriptorHeap[DescriptorBufIdx];

    // Double-indirection: command signature → sorted index → original instance → descriptor
    uint dataPos = InstanceBaseOffset + instanceID;
    uint packedIdx = sortedIndices[dataPos];
    uint idx = packedIdx & 0x7FFFFFFFu;

    // Look up transform slot from descriptor buffer
    uint slot = descriptors[idx].TransformSlot;

    float3 rawPos = positions[vertexID];
    
    // Extract camera position from inverse view
    float3 cameraPos = ViewInverse[3].xyz;
    
    // Get world matrix from global transform buffer using the slot
    row_major float4x4 mat = globalTransforms[slot];
    mat._41 = cameraPos.x;
    mat._42 = cameraPos.y;
    mat._43 = cameraPos.z;
    mat._44 = 1;

    float4 worldPosition = mul(float4(rawPos, 1), mat);
    float4 viewPosition = mul(worldPosition, View);

    output.Position = mul(viewPosition, Projection).xyww;
    output.UV = rawPos.xyz * 2;
    output.ViewDir = rawPos.xyz;

    return output;
}

FragmentOutput PS_Procedural(VertexOutput input)
{
    FragmentOutput output;

    float3 viewDir = normalize(input.ViewDir);
    float3 sunDir = normalize(SunDirection);

    float3 skyColor = GetSkyColor(viewDir, sunDir);

    float sun = GetSun(viewDir, sunDir) * SunIntensity;
    skyColor += float3(1.0, 0.9, 0.7) * sun;

    float clouds = GetClouds(viewDir);

    // Bright white cloud base, tinted by sun/sky light
    float sunDot = saturate(dot(viewDir, sunDir));
    float3 cloudLit = float3(1.0, 0.98, 0.95);               // sunlit side
    float3 cloudShaded = float3(0.7, 0.75, 0.85);             // shaded/ambient side
    float3 cloudColor = lerp(cloudShaded, cloudLit, sunDot * 0.5 + 0.5);

    // Sunset tint on clouds
    float sunElevation = sunDir.y;
    float sunsetAmount = saturate(1.0 - abs((sunElevation - (-0.025)) / 0.175));
    cloudColor = lerp(cloudColor, float3(1.0, 0.7, 0.4), sunsetAmount * 0.5);

    skyColor = lerp(skyColor, cloudColor, clouds * 0.95);

    output.Albedo = float4(skyColor, 1);
    output.Normal = float4(0, 1, 0, 1);
    output.Data = float4(0, 0, 0, 0);
    output.Depth = float4(0, 0, 0, 0);
    output.fDepth = 1;

    return output;
}

RasterizerState DisableCull
{
    CullMode = None;
};

technique11 GBuffer
{
    pass Sky
    {
        SetRasterizerState(DisableCull);
        SetVertexShader(CompileShader(vs_6_6, VS()));
        SetPixelShader(CompileShader(ps_6_6, PS_Procedural()));
    }
}
