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
    uint IndexedDraw;           // 17: 1 = an indexed draw, the vertex index is SV_VertexID (common.fx MeshVertexIndex)
    uint _reserved18;
    uint _reserved19;
    // Slots 20-21: Shadow pass
    uint ExpansionBufferIdx;    // 20: SRV: expansion buffer (cascadeIdx<<30 | instanceIdx)
    uint CascadeBufferSRVIdx;   // 21: SRV: StructuredBuffer<CascadeData>
};

#include "common.fx"
// @RenderState(RenderTargets=6, CullMode=None, LodFade=true)

// Foliage GBuffer shader — based on gbuffer.fx with:
// - Two-sided rendering (CullMode=None)
// - Backface normal flip via SV_IsFrontFace
// - Subsurface scattering approximation
// - Vegetation flag (data.a = 0.5)
// - Wind flutter animation
// - Distance-based alpha threshold

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
    float CanopyAO : TEXCOORD6;     // 0 = deep inside / under the canopy, 1 = outer sunlit shell
    float3 DomeNormal : TEXCOORD7;  // original look (mostly up) — used for materials that have a normal map
    nointerpolation float3 TreeTint : TEXCOORD8;   // this tree's colour (common.fx TreeTint)
    nointerpolation uint LodFade : TEXCOORD9;      // DrawEntryFade (0 = no LOD crossfade)
};

// Wind (common.fx). The crown rides its trunk's bend (TreeSway, as gbuffer_trunk.fx), and the leaves flutter about
// where they hang: clumps of ~0.75 m (from the rest position 'localPos', so a clump keeps its rhythm while the crown
// moves) each on their own rate and direction, harder in the gusts at the tree. Without per-leaf pivots in the mesh
// this is the best a vertex shader can do; a smooth function of the world position here (the first version) made
// waves run through the crown — "wiggles like it's underwater".
float3 WindDisplacement(float3 worldPos, float3 localPos, float3 origin)
{
    float3 sway = TreeSway(worldPos, origin);

    float3 cell = floor(localPos / 0.75);
    float seed = MacroHash(cell.xz + cell.y * 17.0 + origin.xz);
    float rate = 0.8 + 0.4 * seed;
    float s = saturate(WindStrength / 12.0);
    float wave = sin(WindPhase * 7.0 * rate + seed * 40.0) * 0.6 + sin(WindPhase * 11.3 * rate + seed * 23.0) * 0.4;
    float amount = (0.01 + 0.05 * s) * (0.4 + 0.6 * saturate(WindPush(origin.xz)));
    float angle = seed * 6.2831853;
    return sway + float3(cos(angle), 0.4 * (seed - 0.5), sin(angle)) * wave * amount;
}

VSOutput VS(uint primitiveVertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
    VSOutput output;

    StructuredBuffer<InstanceDescriptor> descriptors = ResourceDescriptorHeap[DescriptorBufIdx];
    StructuredBuffer<row_major matrix> globalTransforms = ResourceDescriptorHeap[GlobalTransformBufferIdx];
    StructuredBuffer<uint> sortedIndices = ResourceDescriptorHeap[SortedIndicesIdx];
    
    // Look up mesh buffer indices from MeshRegistry
    StructuredBuffer<MeshPartEntry> meshRegistry = ResourceDescriptorHeap[MeshRegistryIdx];
    MeshPartEntry part = meshRegistry[MeshPartId];
    
    uint vertexID = MeshVertexIndex(part, primitiveVertexID, primitiveVertexID, IndexedDraw);
    StructuredBuffer<float3> positions = ResourceDescriptorHeap[part.PosBufferIdx];
    StructuredBuffer<float3> normals = ResourceDescriptorHeap[part.NormBufferIdx];
    StructuredBuffer<float2> uvs = ResourceDescriptorHeap[part.UVBufferIdx];

    uint dataPos = InstanceBaseOffset + instanceID;
    uint packedIdx = sortedIndices[dataPos];
    uint idx = DrawEntryIndex(packedIdx);   // and the LOD crossfade's tags (lod_fade.hlsli)

    InstanceDescriptor desc = descriptors[idx];
    row_major matrix World = globalTransforms[desc.TransformSlot];

    float3 pos = positions[vertexID];
    float3 norm = normals[vertexID];
    float2 uv = uvs[vertexID];

    float4 worldPos = mul(float4(pos, 1.0f), World);
    
    // Wind: the tree's bend plus the leaves' flutter (identical in VS_Shadow)
    worldPos.xyz += WindDisplacement(worldPos.xyz, pos.xyz, float3(World._41, World._42, World._43));
    
    output.WorldPos = worldPos;
    output.Position = mul(mul(worldPos, View), Projection);
    
    // Original canopy normal: mostly upward with a subtle outward dome from the tree origin. Kept for foliage whose
    // material has a normal map (Sweetgum, beech): those trees were authored against this look.
    float3 treeOrigin = float3(World._41, World._42, World._43);
    output.DomeNormal = normalize(lerp(float3(0, 1, 0), normalize(worldPos.xyz - treeOrigin), 0.25));
    output.TreeTint = TreeTint(treeOrigin);

    // Fake-volume canopy for foliage WITHOUT a normal map (e.g. Aquarius shrubs): normals point out of the leaf
    // part's bounding sphere, biased up — the dome lit them like flat ground, so they read as flat, bright blobs.
    float3 canopyCenterLocal = part.LocalBounds.xyz;
    float canopyRadius = max(part.LocalBounds.w, 0.01);
    float3 canopyCenter = mul(float4(canopyCenterLocal, 1.0f), World).xyz;
    float3 outward = normalize(worldPos.xyz - canopyCenter + float3(0, 1e-4, 0));
    output.Normal = normalize(lerp(float3(0, 1, 0), outward, 0.6));

    // Fake canopy self-occlusion: leaves deep inside and on the underside get less light
    float3 rel = (pos - canopyCenterLocal) / canopyRadius;              // -1..1 inside the bounding sphere
    float shell = smoothstep(0.15, 0.9, length(rel));                   // 0 core → 1 outer shell
    float top = saturate(rel.y * 0.5 + 0.5);                            // 0 bottom → 1 top
    output.CanopyAO = shell * lerp(0.55, 1.0, top);
    output.TexCoord = uv;
    output.TexCoord.y = 1 - output.TexCoord.y;
    output.MaterialID = desc.MaterialId;
    output.Depth = output.Position.w;
    output.TransformSlot = desc.TransformSlot;
    output.MeshPartIdx = desc.MeshPartIdx;
    output.LodFade = DrawEntryFade(packedIdx);
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

PSOutput PS(VSOutput input, bool isFrontFace : SV_IsFrontFace)
{
    PSOutput output;
    LodFadeClip(input.Position.xy, input.LodFade);

    MaterialData mat = GET_MATERIAL(input.MaterialID);
    Texture2D albedoTex = ResourceDescriptorHeap[NonUniformResourceIndex(mat.AlbedoIdx)];

    float4 color = albedoTex.Sample(Sampler, input.TexCoord);
    color.rgb *= mat.Tint * input.TreeTint; // Material Tint (default white), this tree's own colour

    // Distance-based alpha threshold — preserve coverage at distance
    float dist = length(input.WorldPos.xyz);
   // float alphaThreshold = lerp(0.5, 0.15, saturate((dist - 30.0) / 50.0));
   // clip(color.a - alphaThreshold);
    clip(color.a - 0.2f);

    // Use interpolated dome normal from VS — provides directional response
    // while still being smoothed enough to avoid harsh self-shadowing
    // No backface flip: both sides of each leaf card should receive equal lighting
    // Fake canopy volume only when the material has no normal map; normal-mapped foliage keeps the original shading
    bool fakeVolume = mat.NormalIdx == 0;
    float3 N = normalize(fakeVolume ? input.Normal : input.DomeNormal);

    // Subsurface scattering hint: store translucency in albedo alpha
    // The directional light shader can use this for wrap lighting
    float translucency = 0.0;
    if (mat.AOIdx != 0)
    {
        // If a translucency map (_t.dds) is bound, use it
        Texture2D tTex = ResourceDescriptorHeap[NonUniformResourceIndex(mat.AOIdx)];
        translucency = tTex.Sample(Sampler, input.TexCoord).r;
    }

    // Canopy occlusion darkens interior leaves (direct light too, via albedo) and feeds the ambient AO channel
    float canopyAO = fakeVolume ? lerp(0.35, 1.0, saturate(input.CanopyAO)) : 1.0;
    output.Albedo = float4(color.rgb * lerp(0.6, 1.0, canopyAO), 0);
    output.Normal = float4(N, translucency); // store translucency in normal.w
    output.Data = float4(0.7, 0.0, canopyAO, 0.5); // roughness=0.7, metal=0, ao=canopy occlusion, flag=vegetation
    output.Depth = input.Depth;
    output.Displacement = float2(0,0);
    output.EntityId = (input.TransformSlot << 8u) | (input.MeshPartIdx & 0xFFu) | (2u << 30);   // class 2: vegetation (decals_cs.hlsl receivers)
    return output;
}


// Shadow pass
struct ShadowVSOutput
{
    float4 Position : SV_POSITION;
    float2 TexCoord : TEXCOORD0;
    nointerpolation uint MaterialID : TEXCOORD1;
    nointerpolation uint RTIndex : SV_RenderTargetArrayIndex;
    float4 ClipDistance : SV_ClipDistance0;
};

// Sun cascades, or point light cube faces when local
ShadowVSOutput ShadowVertex(uint primitiveVertexID, uint instanceID, bool local)
{
    ShadowVSOutput output;
    output.Position = float4(0, 0, 0, 1);
    output.TexCoord = float2(0, 0);
    output.MaterialID = 0;
    output.RTIndex = 0;

    StructuredBuffer<uint> expansion = ResourceDescriptorHeap[ExpansionBufferIdx];
    uint view, idx;
    DecodeShadowEntry(expansion[InstanceBaseOffset + instanceID], local, view, idx);

    StructuredBuffer<row_major matrix> globalTransforms = ResourceDescriptorHeap[GlobalTransformBufferIdx];
    
    // Look up mesh buffer indices from MeshRegistry
    StructuredBuffer<MeshPartEntry> meshRegistry = ResourceDescriptorHeap[MeshRegistryIdx];
    MeshPartEntry part = meshRegistry[MeshPartId];
    
    uint vertexID = MeshVertexIndex(part, primitiveVertexID, primitiveVertexID, IndexedDraw);
    StructuredBuffer<float3> positions = ResourceDescriptorHeap[part.PosBufferIdx];
    StructuredBuffer<float2> uvs = ResourceDescriptorHeap[part.UVBufferIdx];

    StructuredBuffer<InstanceDescriptor> descriptors = ResourceDescriptorHeap[DescriptorBufIdx];
    InstanceDescriptor desc = descriptors[idx];
    row_major matrix World = globalTransforms[desc.TransformSlot];

    float3 pos = positions[vertexID];
    float4 worldPos = mul(float4(pos, 1.0f), World);

    // Match wind displacement from VS (it used a different weight here once: shadows moved unlike the leaves)
    worldPos.xyz += WindDisplacement(worldPos.xyz, pos.xyz, float3(World._41, World._42, World._43));

    output.Position = ShadowClipPosition(worldPos, view, local, CascadeBufferSRVIdx, output.RTIndex, output.ClipDistance);
    output.TexCoord = uvs[vertexID];
    output.TexCoord.y = 1 - output.TexCoord.y;
    output.MaterialID = desc.MaterialId;
    return output;
}

ShadowVSOutput VS_Shadow(uint primitiveVertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
    return ShadowVertex(primitiveVertexID, instanceID, false);
}

ShadowVSOutput VS_LocalShadow(uint primitiveVertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
    return ShadowVertex(primitiveVertexID, instanceID, true);
}

void PS_Shadow(ShadowVSOutput input)
{
    MaterialData mat = GET_MATERIAL(input.MaterialID);
    Texture2D albedoTex = ResourceDescriptorHeap[NonUniformResourceIndex(mat.AlbedoIdx)];
    float alpha = albedoTex.Sample(Sampler, input.TexCoord).a;
    clip(alpha - 0.2f);
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

    pass LocalShadow
    {
        SetVertexShader(CompileShader(vs_6_6, VS_LocalShadow()));
        SetPixelShader(CompileShader(ps_6_6, PS_Shadow()));
    }
}
