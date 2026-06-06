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
    // Slots 9-15: Reserved
    uint _reserved9;
    uint _reserved10;
    uint _reserved11;
    uint _reserved12;
    uint _reserved13;
    uint _reserved14;
    uint _reserved15;
    // Slot 16: Debug
    uint DebugMode;             // 16: Debug visualization mode
    uint _reserved17;
    uint _reserved18;
    uint _reserved19;
    // Slots 20-21: Shadow pass
    uint ExpansionBufferIdx;    // 20: SRV: expansion buffer (cascadeIdx<<30 | instanceIdx)
    uint CascadeBufferSRVIdx;   // 21: SRV: StructuredBuffer<CascadeData>
};

#include "common.fx"
// @RenderState(RenderTargets=6)

// Trunk/Branch GBuffer shader — based on gbuffer.fx with:
// - Wind sway animation (low-frequency, large-scale)
// - Standard PBR output

inline MaterialData GetMaterial(uint id)
{
    StructuredBuffer<MaterialData> materials = ResourceDescriptorHeap[MaterialsIdx];
    return materials[id];
}
#define GET_MATERIAL(id) GetMaterial(id)



struct VSOutput
{
    float4 Position : SV_POSITION;
    float3 Normal : NORMAL;
    float2 TexCoord : TEXCOORD0;
    float4 WorldPos : TEXCOORD1;
    nointerpolation uint MaterialID : TEXCOORD2;
    float Depth : TEXCOORD3;
    nointerpolation uint TransformSlot : TEXCOORD4;
    nointerpolation uint MeshPartIdx : TEXCOORD5;
};

// Low-frequency trunk sway
float3 TrunkSway(float3 worldPos, float weight)
{
    float phase = Time * 0.8 + worldPos.x * 0.05 + worldPos.z * 0.07;
    float swayX = sin(phase) * 0.25 + sin(phase * 1.7) * 0.1;
    float swayZ = sin(phase * 0.9 + 1.5) * 0.2;
    return float3(swayX, 0, swayZ) * weight;
}

VSOutput VS(uint primitiveVertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
    VSOutput output;

    StructuredBuffer<InstanceDescriptor> descriptors = ResourceDescriptorHeap[DescriptorBufIdx];
    StructuredBuffer<row_major matrix> globalTransforms = ResourceDescriptorHeap[GlobalTransformBufferIdx];
    StructuredBuffer<uint> sortedIndices = ResourceDescriptorHeap[SortedIndicesIdx];

    uint dataPos = InstanceBaseOffset + instanceID;
    uint packedIdx = sortedIndices[dataPos];
    uint idx = packedIdx & 0x7FFFFFFFu;

    InstanceDescriptor desc = descriptors[idx];
    row_major matrix World = globalTransforms[desc.TransformSlot];

    // Negative-scale winding fix (see gbuffer.fx)
    float3x3 W3 = (float3x3)World;
    float det = determinant(W3);
    
    uint fetchID = primitiveVertexID;
    if (det < 0.0f)
    {
        uint triLocal = primitiveVertexID % 3;
        uint triBase = primitiveVertexID - triLocal;
        fetchID = triBase + (triLocal == 1u ? 2u : (triLocal == 2u ? 1u : 0u));
    }

    // Look up mesh buffer indices from MeshRegistry
    StructuredBuffer<MeshPartEntry> meshRegistry = ResourceDescriptorHeap[MeshRegistryIdx];
    MeshPartEntry part = meshRegistry[MeshPartId];

    StructuredBuffer<uint> indices = ResourceDescriptorHeap[part.IndexBufferIdx];
    uint vertexID = indices[fetchID + part.BaseIndex];
    StructuredBuffer<float3> positions = ResourceDescriptorHeap[part.PosBufferIdx];
    StructuredBuffer<float3> normals = ResourceDescriptorHeap[part.NormBufferIdx];
    StructuredBuffer<float2> uvs = ResourceDescriptorHeap[part.UVBufferIdx];

    float3 pos = positions[vertexID];
    float3 norm = normals[vertexID];
    float2 uv = uvs[vertexID];

    float4 worldPos = mul(float4(pos, 1.0f), World);
    
    // Trunk sway — stronger at height, zero at base
    float swayWeight = saturate(pos.y * 0.15);
    worldPos.xyz += TrunkSway(worldPos.xyz, swayWeight * 0.12);
    
    output.WorldPos = worldPos;
    output.Position = mul(mul(worldPos, View), Projection);
    output.Normal = mul(norm, W3);
    output.TexCoord = uv;
    output.TexCoord.y = 1 - output.TexCoord.y;
    output.MaterialID = desc.MaterialId;
    output.Depth = output.Position.w;
    output.TransformSlot = desc.TransformSlot;
    output.MeshPartIdx = desc.MeshPartIdx;
    return output;
}

struct PSOutput
{
    float4 Albedo : SV_Target0;
    float4 Normal : SV_Target1;
    float4 Data : SV_Target2;
    float  Depth : SV_Target3;
    uint   EntityId : SV_Target4;
    float2 Displacement : SV_Target5;    
};

SamplerState Sampler : register(s0);

PSOutput PS(VSOutput input)
{
    PSOutput output;

    MaterialData mat = GET_MATERIAL(input.MaterialID);
    Texture2D albedoTex = ResourceDescriptorHeap[mat.AlbedoIdx];

    float4 color = albedoTex.Sample(Sampler, input.TexCoord);
    clip(color.a - 0.25f);

    float roughness = 0.75; // bark is rough
    float metal = 0.0;
    float ao = 1.0;

    if (mat.RoughnessIdx != 0) { Texture2D rTex = ResourceDescriptorHeap[mat.RoughnessIdx]; roughness = 1.0 - rTex.Sample(Sampler, input.TexCoord).r; }
    if (mat.MetallicIdx  != 0) { Texture2D mTex = ResourceDescriptorHeap[mat.MetallicIdx];  metal     = mTex.Sample(Sampler, input.TexCoord).r; }
    if (mat.AOIdx        != 0) { Texture2D aTex = ResourceDescriptorHeap[mat.AOIdx];        ao        = aTex.Sample(Sampler, input.TexCoord).r; }

    // Emissive: sample texture, apply tint/intensity, blend into albedo
    float emissiveMask = 0;
    if (mat.EmissiveIdx != 0)
    {
        Texture2D eTex = ResourceDescriptorHeap[mat.EmissiveIdx];
        float3 emissive = eTex.Sample(Sampler, input.TexCoord).rgb * mat.EmissiveColor * mat.EmissiveIntensity;
        emissiveMask = max(emissive.r, max(emissive.g, emissive.b));
        color.rgb = lerp(color.rgb, emissive, saturate(emissiveMask));
    }

    float3 N = normalize(input.Normal);
    
    float3 dp1 = ddx(input.WorldPos.xyz);
    float3 dp2 = ddy(input.WorldPos.xyz);
    float2 duv1 = ddx(input.TexCoord);
    float2 duv2 = ddy(input.TexCoord);
    
    float3 dp2perp = cross(dp2, N);
    float3 dp1perp = cross(N, dp1);
    float3 T = dp2perp * duv1.x + dp1perp * duv2.x;
    float3 B = dp2perp * duv1.y + dp1perp * duv2.y;
    
    // Fix TBN handedness for mirrored geometry (negative-scale transforms)
    // Without this, mirrored instances get inverted normal map lighting
    float handedness = dot(cross(T, B), N) < 0.0 ? -1.0 : 1.0;
    T *= handedness;
    
    float invmax = rsqrt(max(dot(T, T), dot(B, B)));
    float3x3 TBN = float3x3(T * invmax, B * invmax, N);
    
    if (mat.NormalIdx != 0)
    {
        Texture2D normalTex = ResourceDescriptorHeap[mat.NormalIdx];
        float2 nXY = normalTex.Sample(Sampler, input.TexCoord).rg * 2.0 - 1.0;
        nXY.y = -nXY.y;
        float3 texNormal = float3(nXY, sqrt(max(0.001, 1.0 - dot(nXY, nXY))));

        if (mat.DetailNormalIdx != 0)
        {
            float dist = length(input.WorldPos.xyz);
            float detailFade = 1.0 - saturate((dist - 20.0) / 40.0);
            if (detailFade > 0.01)
            {
                float detailTiling = mat.DetailTiling.x;
                if (detailTiling <= 0.0) detailTiling = 5.0;
                float2 detailUV = input.TexCoord * detailTiling;
                Texture2D detailTex = ResourceDescriptorHeap[mat.DetailNormalIdx];
                float2 dXY = detailTex.Sample(Sampler, detailUV).rg * 2.0 - 1.0;
                dXY.y = -dXY.y;
                float detailMask = 1.0;
                if (mat.DetailMaskIdx != 0)
                {
                    Texture2D maskTex = ResourceDescriptorHeap[mat.DetailMaskIdx];
                    detailMask = maskTex.Sample(Sampler, input.TexCoord).a;
                }
                texNormal.xy += dXY * detailMask * detailFade;
                texNormal.z = sqrt(max(0.001, 1.0 - dot(texNormal.xy, texNormal.xy)));
            }
        }

        N = normalize(mul(texNormal, TBN));
    }

    output.Albedo = float4(color.rgb, emissiveMask);
    output.Normal = float4(N, 1.0f);
    // Geometric specular anti-aliasing (Kaplanyan 2016 / Tokuyoshi 2019)
    float3 dNdx = ddx(N), dNdy = ddy(N);
    float normalVariance = max(dot(dNdx, dNdx), dot(dNdy, dNdy));
    roughness = sqrt(saturate(roughness * roughness + min(2.0 * normalVariance, 0.18)));

    output.Data = float4(saturate(roughness), saturate(metal), saturate(ao), 1.0);
    output.Depth = input.Depth;
    output.Displacement = float2(0,0);
    output.EntityId = (input.TransformSlot << 8u) | (input.MeshPartIdx & 0xFFu);
    return output;
}


// Shadow pass
struct ShadowVSOutput
{
    float4 Position : SV_POSITION;
    float2 TexCoord : TEXCOORD0;
    nointerpolation uint MaterialID : TEXCOORD1;
    nointerpolation uint RTIndex : SV_RenderTargetArrayIndex;
};

ShadowVSOutput VS_Shadow(uint primitiveVertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
    ShadowVSOutput output;
    output.Position = float4(0, 0, 0, 1);
    output.TexCoord = float2(0, 0);
    output.MaterialID = 0;
    output.RTIndex = 0;

    // Read expansion entry: bits 30-31 = cascadeIdx, bits 0-29 = instance index
    StructuredBuffer<uint> expansion = ResourceDescriptorHeap[ExpansionBufferIdx];
    uint entry = expansion[InstanceBaseOffset + instanceID];
    uint cascadeIdx = entry >> 30;
    uint idx = entry & 0x3FFFFFFFu;
    output.RTIndex = cascadeIdx;

    StructuredBuffer<row_major matrix> globalTransforms = ResourceDescriptorHeap[GlobalTransformBufferIdx];

    // Look up mesh buffer indices from MeshRegistry
    StructuredBuffer<MeshPartEntry> meshRegistry = ResourceDescriptorHeap[MeshRegistryIdx];
    MeshPartEntry part = meshRegistry[MeshPartId];

    StructuredBuffer<uint> indices = ResourceDescriptorHeap[part.IndexBufferIdx];
    uint vertexID = indices[primitiveVertexID + part.BaseIndex];
    StructuredBuffer<float3> positions = ResourceDescriptorHeap[part.PosBufferIdx];
    StructuredBuffer<float2> uvs = ResourceDescriptorHeap[part.UVBufferIdx];

    StructuredBuffer<InstanceDescriptor> descriptors = ResourceDescriptorHeap[DescriptorBufIdx];
    InstanceDescriptor desc = descriptors[idx];
    row_major matrix World = globalTransforms[desc.TransformSlot];

    float3 pos = positions[vertexID];
    float4 worldPos = mul(float4(pos, 1.0f), World);
    float swayWeight = saturate(pos.y * 0.15);
    worldPos.xyz += TrunkSway(worldPos.xyz, swayWeight * 0.12);

    StructuredBuffer<CascadeData> cascadeData = ResourceDescriptorHeap[CascadeBufferSRVIdx];
    output.Position = mul(worldPos, cascadeData[cascadeIdx].VP);
    output.TexCoord = uvs[vertexID];
    output.TexCoord.y = 1 - output.TexCoord.y;
    output.MaterialID = desc.MaterialId;
    return output;
}

void PS_Shadow(ShadowVSOutput input)
{
    MaterialData mat = GET_MATERIAL(input.MaterialID);
    Texture2D albedoTex = ResourceDescriptorHeap[mat.AlbedoIdx];
    float alpha = albedoTex.Sample(Sampler, input.TexCoord).a;
    clip(alpha - 0.25f);
}

technique11 GBuffer
{
    pass Opaque
    {
        SetVertexShader(CompileShader(vs_6_6, VS()));
        SetPixelShader(CompileShader(ps_6_6, PS()));
    }
    
    pass Shadow
    {
        SetVertexShader(CompileShader(vs_6_6, VS_Shadow()));
        SetPixelShader(CompileShader(ps_6_6, PS_Shadow()));
    }
}
