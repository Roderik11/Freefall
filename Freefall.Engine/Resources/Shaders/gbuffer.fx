cbuffer PushConstants : register(b3)
{
    // Slots 0-1: Reserved for light/composition passes
    uint _reserved0;
    uint _reserved1;
    // Slots 2-13: Mesh draw data
    uint DescriptorBufIdx;      // 2: StructuredBuffer<InstanceDescriptor> - per-instance descriptor
    uint _reserved3;            // 3: Reserved (was MaterialIDsIdx)
    uint SortedIndicesIdx;      // 4: StructuredBuffer<uint> - sorted draw order indices
    uint BoneWeightsIdx;        // 5: Unused for static meshes (0)
    uint BonesIdx;              // 6: Unused for static meshes (0)
    uint IndexBufferIdx;        // 7: StructuredBuffer<uint> - mesh index buffer
    uint BaseIndex;             // 8: Base index offset into index buffer
    uint PosBufferIdx;          // 9: StructuredBuffer<float3> - vertex positions
    uint NormBufferIdx;         // 10: StructuredBuffer<float3> - vertex normals
    uint UVBufferIdx;           // 11: StructuredBuffer<float2> - vertex UVs
    uint NumBones;              // 12: Number of bones (for skinned shaders)
    uint InstanceBaseOffset;    // 13: Base offset for instance ID (per-command)
    // Slots 14-15: Global bindless buffers
    uint MaterialsIdx;          // 14: Index to materials buffer
    uint GlobalTransformBufferIdx; // 15: Index to global TransformBuffer
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


// Helper to get material from instance's MaterialID (override common.fx version)
inline MaterialData GetMaterial(uint id)
{
    StructuredBuffer<MaterialData> materials = ResourceDescriptorHeap[MaterialsIdx];
    return materials[id];
}
#define GET_MATERIAL(id) GetMaterial(id)

SamplerState Sampler : register(s0);

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

struct PSOutput
{
    float4 Albedo : SV_Target0;
    float4 Normal : SV_Target1;
    float4 Data : SV_Target2;
    float  Depth : SV_Target3;
    uint   EntityId : SV_Target4;
    float2 Displacement : SV_Target5;
};

VSOutput VS(uint primitiveVertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
    VSOutput output;

    // Instance data buffers - descriptor contains TransformSlot + MaterialId
    StructuredBuffer<InstanceDescriptor> descriptors = ResourceDescriptorHeap[DescriptorBufIdx];
    StructuredBuffer<row_major matrix> globalTransforms = ResourceDescriptorHeap[GlobalTransformBufferIdx];
    StructuredBuffer<uint> sortedIndices = ResourceDescriptorHeap[SortedIndicesIdx];
    
    // Get instance data position using InstanceBaseOffset + local instance ID
    uint dataPos = InstanceBaseOffset + instanceID;
    
    // sortedIndices contains compacted original instanceIdx
    uint packedIdx = sortedIndices[dataPos];
    uint idx = packedIdx & 0x7FFFFFFFu;
    
    // Double-indirect: use original instance index to look up per-instance data from descriptor
    InstanceDescriptor desc = descriptors[idx];
    uint slot = desc.TransformSlot;
    uint materialID = desc.MaterialId;
    
    row_major matrix World = globalTransforms[slot];
    
    // Negative-scale fix: detect mirrored transforms via 3x3 determinant.
    // Negative determinant means odd number of axis flips — triangle winding is reversed.
    // Swap vertices 1↔2 within each triangle to restore correct winding for the rasterizer.
    float3x3 W3 = (float3x3)World;
    float det = determinant(W3);
    
    uint fetchID = primitiveVertexID;
    if (det < 0.0f)
    {
        uint triLocal = primitiveVertexID % 3;
        uint triBase = primitiveVertexID - triLocal;
        // Remap {0,1,2} → {0,2,1} to flip winding
        fetchID = triBase + (triLocal == 1u ? 2u : (triLocal == 2u ? 1u : 0u));
    }
    
    // Bindless index buffer - fetchID is 0 to N-1, add BaseIndex to offset into correct mesh part
    StructuredBuffer<uint> indices = ResourceDescriptorHeap[IndexBufferIdx];
    uint vertexID = indices[fetchID + BaseIndex];
    
    // Mesh data buffers - use resolved vertexID
    StructuredBuffer<float3> positions = ResourceDescriptorHeap[PosBufferIdx];
    StructuredBuffer<float3> normals = ResourceDescriptorHeap[NormBufferIdx];
    StructuredBuffer<float2> uvs = ResourceDescriptorHeap[UVBufferIdx];
    
    float3 pos = positions[vertexID];
    float3 norm = normals[vertexID];
    float2 uv = uvs[vertexID];
    
    // Apply world transform (no skinning for static meshes)
    float4 worldPos = mul(float4(pos, 1.0f), World);
    output.WorldPos = worldPos;
    output.Position = mul(mul(worldPos, View), Projection);
    
    output.Normal = mul(norm, W3);
    output.TexCoord = uv;
    output.TexCoord.y = 1 - output.TexCoord.y;
    output.MaterialID = materialID;
    output.Depth = output.Position.w; // View-space Z (linear)
    output.TransformSlot = slot;
    output.MeshPartIdx = desc.MeshPartIdx;
    return output;
}

// Shadow pass - minimal output structure
struct ShadowVSOutput
{
    float4 Position : SV_POSITION;
    float2 TexCoord : TEXCOORD0;
    nointerpolation uint MaterialID : TEXCOORD1;
    nointerpolation uint RTIndex : SV_RenderTargetArrayIndex;
};

// Shadow vertex shader - single-pass multi-cascade via expansion buffer
// Each instanceID maps to exactly one (instance, cascade) pair from the expansion buffer.
// No cascade mask check needed — expansion only contains visible pairs.
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
    StructuredBuffer<uint> indices = ResourceDescriptorHeap[IndexBufferIdx];
    uint vertexID = indices[primitiveVertexID + BaseIndex];

    StructuredBuffer<float3> positions = ResourceDescriptorHeap[PosBufferIdx];
    StructuredBuffer<float2> uvs = ResourceDescriptorHeap[UVBufferIdx];

    StructuredBuffer<InstanceDescriptor> descriptors = ResourceDescriptorHeap[DescriptorBufIdx];
    InstanceDescriptor desc = descriptors[idx];
    row_major matrix World = globalTransforms[desc.TransformSlot];

    float3 pos = positions[vertexID];
    float4 worldPos = mul(float4(pos, 1.0f), World);

    // Project using cascade-specific VP from shared CascadeData buffer
    StructuredBuffer<CascadeData> cascades = ResourceDescriptorHeap[CascadeBufferSRVIdx];
    output.Position = mul(worldPos, cascades[cascadeIdx].VP);
    output.TexCoord = uvs[vertexID];
    output.TexCoord.y = 1 - output.TexCoord.y;
    output.MaterialID = desc.MaterialId;
    return output;
}

// Shadow pixel shader - alpha test only, depth written by hardware
void PS_Shadow(ShadowVSOutput input)
{
    //MaterialData mat = GET_MATERIAL(input.MaterialID);
    //Texture2D albedoTex = ResourceDescriptorHeap[mat.AlbedoIdx];
    //float alpha = albedoTex.Sample(Sampler, input.TexCoord).a;
    //clip(alpha - 0.25f);
}

PSOutput PS(VSOutput input)
{
    PSOutput output;
    
    uint materialID = input.MaterialID;
    
    // Material lookup via MaterialID indirection
    MaterialData mat = GET_MATERIAL(materialID);
    Texture2D albedoTex = ResourceDescriptorHeap[mat.AlbedoIdx];
    
    float4 color = albedoTex.Sample(Sampler, input.TexCoord);
    //clip(color.a - 0.25f);
    
    // PBR material properties — defaults for meshes without PBR textures
    float roughness = 0.65;
    float metal = 0.0;
    float ao = 1.0;
    
    // Sample PBR textures if bound (index 0 = not bound)
    // RoughnessIdx holds a specular map — invert to get roughness
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
    
    // Normal mapping via cotangent frame (no tangent buffer needed)
    float3 N = normalize(input.Normal);
    float3 faceNormal = N;
    
    // Cotangent frame from screen-space derivatives (shared by base + detail normals)
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
        // Decode normal map: handle BC5 (2-channel, Z reconstructed) and OpenGL Y convention
        float2 nXY = normalTex.Sample(Sampler, input.TexCoord).rg * 2.0 - 1.0;
        nXY.y = -nXY.y; // Flip Y: OpenGL (Unity) → DirectX convention
        float3 texNormal = float3(nXY, sqrt(max(0.001, 1.0 - dot(nXY, nXY))));
        // Blend detail normal if present (UDN blending in tangent space)
        //if (mat.DetailNormalIdx != 0)
        //{
        //    // Distance fade: full detail within 20m, gone by 60m
        //    float dist = length(input.WorldPos.xyz);
        //    float detailFade = 1.0 - saturate((dist - 20.0) / 40.0);
            
        //    if (detailFade > 0.01)
        //    {
        //        float detailTiling = asfloat(mat.DetailTilingPacked);
        //        if (detailTiling <= 0.0) detailTiling = 5.0;
                
        //        float2 detailUV = input.TexCoord * detailTiling;
        //        Texture2D detailTex = ResourceDescriptorHeap[mat.DetailNormalIdx];
        //        float2 dXY = detailTex.Sample(Sampler, detailUV).rg * 2.0 - 1.0;
        //        dXY.y = -dXY.y;
                
        //        // UDN blend: add XY with strength, recompute Z
        //        texNormal.xy += dXY * detailFade * 2.0;
        //        texNormal.z = sqrt(max(0.001, 1.0 - dot(texNormal.xy, texNormal.xy)));
        //    }
        //}
        
        N = normalize(mul(texNormal, TBN));
    }
    
    float depth = input.Depth;
    float2 displacement = float2(0, 0);
    
    // Blend explicit height using same visual weight
    if (mat.DetailMaskIdx != 0)
    {
        Texture2D heightMap = ResourceDescriptorHeap[mat.DetailMaskIdx];
        float height = heightMap.Sample(Sampler, input.TexCoord).r;
        
        float uvRate = length(ddy(input.TexCoord));
        float worldRate = length(ddy(input.WorldPos));
        float uvPerMeter = uvRate / max(worldRate, 0.001);
        float dispScale = 0.01 / uvPerMeter;
        
        depth = input.Depth - height * dispScale;

        float4 clipPos = mul(float4(input.WorldPos), ViewProjection);
        float4 clipDisp = mul(float4(input.WorldPos + faceNormal * height * dispScale, 1), ViewProjection);
        float2 ndcPos = clipPos.xy / clipPos.w;
        float2 ndcDisp = clipDisp.xy / clipDisp.w;
        displacement = (ndcDisp - ndcPos) * float2(0.5, -0.5);
        
        // Fade displacement at screen edges so the inversion never needs off-screen sources.
        // Margin is proportional to displacement magnitude — large displacements fade earlier.
        float2 screenUV = ndcPos * float2(0.5, -0.5) + 0.5;
        float dispMag = length(displacement);
        float margin = dispMag * 1.0;
        float edgeFade = saturate(min(min(screenUV.x, 1.0 - screenUV.x),
                                  min(screenUV.y, 1.0 - screenUV.y)) / margin);
        displacement *= edgeFade;
        //depth *= edgeFade;
    }
    
    output.Albedo = float4(color.rgb, emissiveMask);
    output.Normal = float4(N, 1.0f);
    output.Data = float4(saturate(roughness), saturate(metal), saturate(ao), 1.0);
    output.Depth = depth;
    output.Displacement = displacement;
    output.EntityId = (input.TransformSlot << 8u) | (input.MeshPartIdx & 0xFFu);

    return output;
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
