cbuffer PushConstants : register(b3)
{
    uint _reserved0;
    uint _reserved1;
    uint DescriptorBufIdx;      // 2
    uint _reserved3;            // 3
    uint SortedIndicesIdx;      // 4
    uint BoneWeightsIdx;        // 5
    uint BonesIdx;              // 6
    uint IndexBufferIdx;        // 7
    uint BaseIndex;             // 8
    uint PosBufferIdx;          // 9
    uint NormBufferIdx;         // 10
    uint UVBufferIdx;           // 11
    uint NumBones;              // 12
    uint InstanceBaseOffset;    // 13
    uint MaterialsIdx;          // 14
    uint GlobalTransformBufferIdx; // 15
};

#include "common.fx"
// @RenderState(RenderTargets=4, DepthWrite=false, DepthFunc=GreaterEqual, CullMode=None)

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
    float CloudBrightness;       // overall cloud brightness (0-3)
    float _cloudPad0;
    float3 CloudShadowColor;     // color of cloud shade side
    float CloudAltitude;         // cloud layer height in world units
    float3 CloudSunlitColor;     // color of sun-facing cloud tops
    uint CloudNoiseLUTIdx;       // bindless index for 3D noise texture
}

SamplerState linearWrap : register(s0); // Linear filter, wrap addressing (3D noise LUT)

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
// Noise utility (used by GetStars)
float hash13(float3 p3)
{
    p3 = frac(p3 * 0.1031);
    p3 += dot(p3, p3.zyx + 31.32);
    return frac((p3.x + p3.y) * p3.z);
}

// ────────────────────────────────────────────────
// Main cloud function — LUT-based Nubis-style clouds
//
// LUT channels (from cloud_noise_gen.hlsl):
//   R: Perlin FBM     (smooth connected shapes)
//   G: Worley FBM     (low-freq billowy erosion)
//   B: Worley FBM     (high-freq detail erosion)
//   A: Perlin-Worley  (pre-combined cloud shape)
//
float GetClouds(float3 viewDir)
{
    if (viewDir.y <= 0.001)
        return 0.0;

    float t = CloudAltitude / viewDir.y;
    float2 cloudPos = viewDir.xz * t;
    float2 uv = cloudPos * 0.00035;

    float2 wind = float2(CloudTime * CloudSpeed * 0.01, CloudTime * CloudSpeed * 0.005);
    uv += wind;

    Texture3D<float4> noiseLUT = ResourceDescriptorHeap[CloudNoiseLUTIdx];
    float timeZ = CloudTime * 0.005;

    // Mip level based on viewing angle — blur out detail near the horizon
    // to hide tiling and create natural atmospheric softening
    float mip = saturate(1.0 - viewDir.y * 5.0) * 3.0;  // 0 at zenith, up to 3 at horizon

    // ── Base shape: multi-scale Perlin-Worley (A channel) ──
    float baseShape = 0;
    baseShape += noiseLUT.SampleLevel(linearWrap, float3(uv * 0.25,        timeZ        ), mip    ).a * 0.625;
    baseShape += noiseLUT.SampleLevel(linearWrap, float3(uv * 0.5 + 0.37,  timeZ * 0.7  ), mip    ).a * 0.25;
    baseShape += noiseLUT.SampleLevel(linearWrap, float3(uv * 1.0 + 0.71,  timeZ * 1.3  ), mip * 0.5).a * 0.125;

    // ── Coverage threshold ──
    float coverageNoise = noiseLUT.SampleLevel(linearWrap, float3(uv * 0.06, timeZ * 0.15), 0).r;
    float coverage = saturate(CloudCoverage + (coverageNoise - 0.5) * 0.3);

    // Remap: only the brightest noise survives as clouds
    float cloudDensity = remap(baseShape, 1.0 - coverage, 1.0, 0.0, 1.0);

    // ── Detail erosion: Worley carves billowy edges ──
    // Use higher mip near horizon to blur out repetition
    float4 detailSample = noiseLUT.SampleLevel(linearWrap, float3(uv * 2.0 + 1.13, timeZ * 1.5), mip);
    float detailFBM = detailSample.g * 0.625 + detailSample.b * 0.375;

    // Key fix: erosion strength scales with (1 - density)
    // Thick cloud cores resist erosion; only thin edges get carved
    float erodeStrength = detailFBM * 0.2 * (1.0 - cloudDensity * 0.7);
    float eroded = remap(cloudDensity, erodeStrength, 1.0, 0.0, 1.0);

    // Softer smoothstep for fuller cloud bodies
    eroded = smoothstep(0.0, 0.45, eroded);

    // Horizon fade
    eroded *= smoothstep(0.0, 0.22, viewDir.y);

    return saturate(eroded);
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

#include "sky_common.fx"

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

    // Reverse-Z: far plane = 0, so set z=0 to place skybox at max distance
    float4 projPos = mul(viewPosition, Projection);
    output.Position = float4(projPos.xy, 0.0, projPos.w);
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

    // Stars (not in shared GetSkyColor — ocean reflections don't need them)
    float nightSunElev = sunDir.y;
    float nightFactor = saturate((-nightSunElev - 0.15) / 0.3);
    skyColor += GetStars(viewDir, nightFactor);

    float sun = GetSun(viewDir, sunDir) * SunIntensity;
    skyColor += float3(1.0, 0.9, 0.7) * sun;

    float density = GetClouds(viewDir);

    if (density > 0.001)
    {
        // ── Opacity via Beer-Lambert ──
        float absorption = 1.0 - exp(-density * 4.5);

        // ── Core shading: density drives the shadow-sunlit gradient ──
        // This is the key: the cloud shape detail IS the shading detail.
        // Thin edges transmit light (bright), thick cores self-shadow (dark).
        float transmittance = exp(-density * 3.0);  // 1.0 at edges, ~0.05 at dense cores

        // ── Directional sun shading ──
        float sunDot = saturate(dot(viewDir, sunDir));
        float sunGradient = sunDot * 0.5 + 0.5;  // 0.5 (shade side) to 1.0 (sun facing)

        // Combine: transmittance provides detail, sunGradient provides directionality
        float shadeFactor = lerp(transmittance * 0.7, transmittance, sunGradient);

        // Map from shadow color to sunlit color
        float3 cloudColor = lerp(CloudShadowColor, CloudSunlitColor, shadeFactor);

        // ── Silver lining: bright rim where thin cloud faces sun ──
        float edgeMask = smoothstep(0.0, 0.2, density) * smoothstep(0.5, 0.15, density);
        float silverLining = edgeMask * pow(sunDot, 2.0) * 0.4;
        cloudColor += float3(1.0, 1.0, 0.95) * silverLining;

        // ── Powder/backlit glow ──
        float powder = (1.0 - transmittance) * transmittance * 2.0;  // peaks at medium density
        cloudColor += float3(1.0, 0.9, 0.7) * powder * pow(sunDot, 3.0) * 0.3;

        // ── Sunset tint ──
        float cloudSunElev = sunDir.y;
        float sunsetAmount = saturate(1.0 - abs((cloudSunElev - (-0.025)) / 0.175));
        cloudColor = lerp(cloudColor, float3(1.0, 0.6, 0.3), sunsetAmount * 0.5);

        // ── Apply brightness ──
        cloudColor *= CloudBrightness * 1.3;

        // Blend into sky
        skyColor = lerp(skyColor, cloudColor, absorption);
    }

    output.Albedo = float4(skyColor, 1);
    output.Normal = float4(0, 1, 0, 1);
    output.Data = float4(0, 0, 0, 0);
    output.Depth = float4(0, 0, 0, 0);
    output.fDepth = 0; // Reverse-Z: far plane = 0

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
