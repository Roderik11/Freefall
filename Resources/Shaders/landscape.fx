#include "common.fx"
// @RenderState(RenderTargets=4)

// LandscapePatchData: per-instance data for each clipmap ring segment.
// Transform comes from the global TransformBuffer (standard path).
struct LandscapePatchData
{
    float4 rect;       // UV rect into heightmap (xy=min, zw=max)
    float2 level;      // x=LOD level, y=morph factor
    float  ringScale;  // world scale of this ring
    float  padding;
};

// Standard InstanceBatch push constant layout (slots 2-15 set by command signature)
#define DescriptorBufIdx GET_INDEX(2)
#define Reserved0Idx GET_INDEX(3)
#define SortedIndicesIdx GET_INDEX(4)
#define IndexBufferIdx GET_INDEX(7)
#define BaseIndex GET_INDEX(8)
#define PosBufferIdx GET_INDEX(9)
#define InstanceBaseOffset GET_INDEX(13)
#define DebugMode GET_INDEX(16)

// Per-instance buffer: LandscapePatchData (slot 1, via generic per-instance buffer system)
#define LandscapeDataIdx GET_INDEX(1)

// Landscape-specific texture indices (slots 17-20, set by Material.Apply, above command signature range)
#define HeightTexIdx GET_INDEX(17)
#define ControlMapsIdx GET_INDEX(18)
#define DiffuseMapsIdx GET_INDEX(19)
#define NormalMapsIdx GET_INDEX(20)

cbuffer landscape : register(b1)
{
    float3 CameraPos;
    float HeightTexel;
    float MaxHeight;
    float _pad1;
    float2 TerrainSize;
    float2 TerrainOrigin;
    float2 SnapOffset;   // fractional remainder for smooth sub-cell scrolling
}

cbuffer tiling : register(b2)
{
    float4 LayerTiling[32];
}

SamplerState sampData : register(s0);         // WrappedAnisotropic
SamplerState sampHeight : register(s1);       // ClampedPoint2D
SamplerState sampHeightFilter : register(s2); // ClampedBilinear2D

// Per-instance descriptor (matches C# InstanceDescriptor: 12 bytes)
struct InstanceDescriptor
{
    uint TransformSlot;
    uint MaterialId;
    uint CustomDataIdx;
};

// ============================================================
// Control point data passed from VS → HS → DS
// ============================================================
struct VSOutput
{
    float3 WorldPos : WORLDPOS;
    float2 UV       : TEXCOORD0;
    float2 HeightUV : TEXCOORD1;
    float  Level    : TEXCOORD2;
    uint   Flags    : TEXCOORD3;
    uint   Idx      : TEXCOORD4;  // original instance index for DS lookups
};

struct DSOutput
{
    float4 Position : SV_POSITION;
    float2 UV       : TEXCOORD0;
    float2 UV2      : TEXCOORD1;
    float  Depth    : TEXCOORD2;
    float  Level    : TEXCOORD3;
    nointerpolation uint Flags : TEXCOORD4;
};

struct FragmentOutput
{
    float4 Albedo  : SV_TARGET0;
    float4 Normals : SV_TARGET1;
    float4 Data    : SV_TARGET2;
    float  Depth   : SV_TARGET3;
};

// ============================================================
// Vertex Shader — pass-through, no projection
// ============================================================
VSOutput VS(uint primitiveVertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
    VSOutput output;

    // Double-indirection: command signature → sorted index → original instance
    StructuredBuffer<uint> sortedIndices = ResourceDescriptorHeap[SortedIndicesIdx];
    uint dataPos = InstanceBaseOffset + instanceID;
    uint packedIdx = sortedIndices[dataPos];
    bool isOccluded = (packedIdx & 0x80000000u) != 0;
    uint idx = packedIdx & 0x7FFFFFFFu;

    // Transform from global buffer
    StructuredBuffer<InstanceDescriptor> descriptors = ResourceDescriptorHeap[DescriptorBufIdx];
    StructuredBuffer<row_major matrix> globalTransforms = ResourceDescriptorHeap[GlobalTransformBufferIdx];
    uint slot = descriptors[idx].TransformSlot;
    row_major matrix world = globalTransforms[slot];

    // Per-instance landscape data
    StructuredBuffer<LandscapePatchData> landscapeData = ResourceDescriptorHeap[LandscapeDataIdx];
    LandscapePatchData patch = landscapeData[idx];

    // Vertex position from bindless buffers
    StructuredBuffer<uint> indices = ResourceDescriptorHeap[IndexBufferIdx];
    uint vertexID = indices[primitiveVertexID + BaseIndex];
    StructuredBuffer<float3> positions = ResourceDescriptorHeap[PosBufferIdx];
    float3 pos = positions[vertexID];

    // Map mesh vertices from [-16,+16] to [0,1] UV
    float2 uv = (pos.xz + 16.0) / 32.0;

    // Derive world XZ from the per-patch rect
    // rect = (boundsMin.x/totalX, boundsMax.z/totalZ, boundsMax.x/totalX, boundsMin.z/totalZ)
    float4 rect = patch.rect;
    float worldX = TerrainOrigin.x + TerrainSize.x * (rect.x + uv.x * (rect.z - rect.x));
    float worldZ = TerrainOrigin.y + TerrainSize.y * (rect.w + uv.y * (rect.y - rect.w));

    float2 heightUV = float2(0, 0); // DS recomputes from world position

    // World position (no height yet — DS will displace)
    float3 worldPosition = float3(worldX, 0, worldZ);

    output.WorldPos = worldPosition;
    output.UV = uv;
    output.HeightUV = heightUV;
    output.Level = patch.level.x;
    output.Flags = isOccluded ? 1u : 0u;
    output.Idx = idx;

    return output;
}

// ============================================================
// Hull Shader — adaptive tessellation
// ============================================================
struct PatchConstants
{
    float EdgeTessFactor[3] : SV_TessFactor;
    float InsideTessFactor  : SV_InsideTessFactor;
};

// Screen-space edge length tessellation factor
float ComputeEdgeTessFactor(float3 p0, float3 p1)
{
    float3 mid = (p0 + p1) * 0.5;
    float dist = distance(mid, CameraPos);
    
    // Adaptive: more tess near camera, less far away
    // With CellSize-based rings, near vertices are already dense — light tessellation suffices
    float factor = 8.0 * saturate(1.0 - dist / 200.0);
    return clamp(factor, 1.0, 16.0);
}

PatchConstants PatchConstantFunc(InputPatch<VSOutput, 3> patch, uint patchID : SV_PrimitiveID)
{
    PatchConstants pc;

    // Edge tessellation factors based on screen-space distance
    pc.EdgeTessFactor[0] = ComputeEdgeTessFactor(patch[1].WorldPos, patch[2].WorldPos);
    pc.EdgeTessFactor[1] = ComputeEdgeTessFactor(patch[2].WorldPos, patch[0].WorldPos);
    pc.EdgeTessFactor[2] = ComputeEdgeTessFactor(patch[0].WorldPos, patch[1].WorldPos);
    pc.InsideTessFactor = (pc.EdgeTessFactor[0] + pc.EdgeTessFactor[1] + pc.EdgeTessFactor[2]) / 3.0;

    return pc;
}

[domain("tri")]
[partitioning("fractional_odd")]
[outputtopology("triangle_cw")]
[outputcontrolpoints(3)]
[patchconstantfunc("PatchConstantFunc")]
[maxtessfactor(64.0)]
VSOutput HS(InputPatch<VSOutput, 3> patch, uint cpID : SV_OutputControlPointID, uint patchId : SV_PrimitiveID)
{
    return patch[cpID]; // pass through control points
}

// ============================================================
// Domain Shader — heightmap displacement + projection
// ============================================================
[domain("tri")]
DSOutput DS(PatchConstants pc, float3 bary : SV_DomainLocation, OutputPatch<VSOutput, 3> patch)
{
    DSOutput output;

    // Interpolate control point data
    float3 worldPos = patch[0].WorldPos * bary.x + patch[1].WorldPos * bary.y + patch[2].WorldPos * bary.z;
    float2 uv = patch[0].UV * bary.x + patch[1].UV * bary.y + patch[2].UV * bary.z;
    float level = patch[0].Level;
    uint flags = patch[0].Flags;

    // Apply snap remainder offset FIRST: vertices slide smoothly with the camera.
    // The grid itself snaps to CellSize boundaries (ring stability),
    // but all texture reads use the actual smooth world position.
    worldPos.xz += SnapOffset;

    // Compute heightUV from the smooth (post-offset) world position.
    // This ensures height and control-map reads are continuous across snap boundaries,
    // eliminating the 2m popping artefact.
    float2 heightUV = (worldPos.xz - TerrainOrigin) / TerrainSize;

    // Sample heightmap and displace
    Texture2D HeightTex = ResourceDescriptorHeap[HeightTexIdx];
    float height = HeightTex.SampleLevel(sampHeightFilter, heightUV, 0).r;
    worldPos.y += height * MaxHeight;

    // Project to clip space
    float4 clipPos = mul(float4(worldPos, 1), ViewProjection);

    // X-ray mode: push occluded instances to near depth
    if (flags == 1u)
        clipPos.z = 0.0;

    output.Position = clipPos;
    output.UV = uv;
    output.UV2 = heightUV;
    output.Depth = clipPos.w;
    output.Level = level;
    output.Flags = flags;

    return output;
}

// ============================================================
// Pixel Shader — texture splatting to GBuffer
// ============================================================
float3 GetNormal(float2 uv)
{
    float4 h;
    Texture2D HeightTex = ResourceDescriptorHeap[HeightTexIdx];

    h[0] = HeightTex.SampleLevel(sampHeightFilter, uv + float2(0, -HeightTexel), 0).r;
    h[1] = HeightTex.SampleLevel(sampHeightFilter, uv + float2(-HeightTexel, 0), 0).r;
    h[2] = HeightTex.SampleLevel(sampHeightFilter, uv + float2(HeightTexel, 0), 0).r;
    h[3] = HeightTex.SampleLevel(sampHeightFilter, uv + float2(0, HeightTexel), 0).r;

    float texelWorldSize = TerrainSize.x * HeightTexel;
    float heightScale = MaxHeight / texelWorldSize;

    float3 n;
    n.z = (h[0] - h[3]) * heightScale;
    n.x = (h[1] - h[2]) * heightScale;
    n.y = 1.0f;

    return normalize(n);
}

FragmentOutput PS(DSOutput input)
{
    FragmentOutput output;

    float3 terrainNormal = GetNormal(input.UV2);

    float4 color = float4(0, 0, 0, 0);
    float4 normal = float4(0, 0, 0, 0);

    float2 uv = input.UV2;
    uv.y = 1 - uv.y;

    Texture2DArray controlMaps = ResourceDescriptorHeap[ControlMapsIdx];
    Texture2DArray diffuseMaps = ResourceDescriptorHeap[DiffuseMapsIdx];
    Texture2DArray normalMaps = ResourceDescriptorHeap[NormalMapsIdx];

    for (int i = 0; i < 4; ++i)
    {
        int startIndex = i * 4;
        float4 weights = controlMaps.Sample(sampData, float3(uv, i));

        for (int j = 0; j < 4; ++j)
        {
            float weight = weights[j];

            if (weight <= 0)
                continue;

            int layer = startIndex + j;
            float2 texuv = uv * LayerTiling[layer].xy;
            float3 layerUV = float3(texuv, layer);

            float4 c = diffuseMaps.Sample(sampData, layerUV);
            float4 n = normalMaps.Sample(sampData, layerUV);

            color += c * weight;
            normal += n * weight;
        }
    }

    normal = float4(normal.xzy, 1);
    normal.xz = normal.xz * 2 - 1;
    normal = normalize(normal);

    terrainNormal = blend_linear(terrainNormal, normal.xyz);
    terrainNormal = mul(terrainNormal, (float3x3) View);

    output.Albedo = float4(color.rgb, 1);
    output.Normals = float4(EncodeNormal(terrainNormal), 1);
    output.Data = float4(0, 1, 0, 0);
    output.Depth = input.Depth;

    return output;
}

// ============================================================
// Shadow pass — depth only with tessellation
// ============================================================
struct ShadowDSOutput
{
    float4 Position : SV_POSITION;
};

// Shadow VS is the same as main VS — pass-through for tessellation
// HS is reused (same tessellation logic)

[domain("tri")]
ShadowDSOutput DS_Shadow(PatchConstants pc, float3 bary : SV_DomainLocation, OutputPatch<VSOutput, 3> patch)
{
    ShadowDSOutput output;

    float3 worldPos = patch[0].WorldPos * bary.x + patch[1].WorldPos * bary.y + patch[2].WorldPos * bary.z;

    // Apply snap remainder offset FIRST (same as main DS)
    worldPos.xz += SnapOffset;

    // Compute heightUV from smooth world position (not interpolated from VS)
    float2 heightUV = (worldPos.xz - TerrainOrigin) / TerrainSize;

    Texture2D HeightTex = ResourceDescriptorHeap[HeightTexIdx];
    float height = HeightTex.SampleLevel(sampHeightFilter, heightUV, 0).r;
    worldPos.y += height * MaxHeight;

    output.Position = mul(float4(worldPos, 1), ViewProjection);
    return output;
}

// ============================================================
// Technique
// ============================================================
technique11 Landscape
{
    pass Opaque
    {
        SetVertexShader(CompileShader(vs_6_6, VS()));
        SetHullShader(CompileShader(hs_6_6, HS()));
        SetDomainShader(CompileShader(ds_6_6, DS()));
        SetPixelShader(CompileShader(ps_6_6, PS()));
    }
    pass Shadow
    {
        SetVertexShader(CompileShader(vs_6_6, VS()));
        SetHullShader(CompileShader(hs_6_6, HS()));
        SetDomainShader(CompileShader(ds_6_6, DS_Shadow()));
    }
}
