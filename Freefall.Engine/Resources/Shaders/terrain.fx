#include "common.fx"
// @RenderState(RenderTargets=4)

// TerrainPatchData: per-instance data passed via the generic per-instance buffer system.
// Transform comes from the global TransformBuffer (standard path).
struct TerrainPatchData
{
	float4 rect;
	float2 level;
	float2 padding;
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

// Per-instance buffer: TerrainPatchData (slot 1, via generic per-instance buffer system)
#define TerrainDataIdx GET_INDEX(1)

// Terrain-specific texture indices (slots 17-20, set by Material.Apply, above command signature range)
#define HeightTexIdx GET_INDEX(17)
#define ControlMapsIdx GET_INDEX(18)
#define DiffuseMapsIdx GET_INDEX(19)
#define NormalMapsIdx GET_INDEX(20)

cbuffer terrain : register(b1)
{
    float3 CameraPos;
    float HeightTexel;
    float MaxHeight;
    float _pad1;
    float2 TerrainSize;
}

cbuffer tiling : register(b2)
{
    float4 LayerTiling[32];
}

SamplerState sampData : register(s0); // WrappedAnisotropic
SamplerState sampHeight : register(s1); // ClampedPoint2D
SamplerState sampHeightFilter : register(s2); // ClampedBilinear2D

struct VertexOutput
{
	float4 Position		: SV_POSITION;
	float2 UV			: TEXCOORD0;
	float2 UV2			: TEXCOORD1;
    float Depth			: TEXCOORD2;
	float Level			: TEXCOORD3;
};

// Per-instance descriptor (matches C# InstanceDescriptor: 12 bytes)
struct InstanceDescriptor
{
    uint TransformSlot;
    uint MaterialId;
    uint CustomDataIdx;
};

VertexOutput VS(uint primitiveVertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
	VertexOutput output;

	// Double-indirection: command signature → sorted index → original instance
	StructuredBuffer<uint> sortedIndices = ResourceDescriptorHeap[SortedIndicesIdx];
	uint dataPos = InstanceBaseOffset + instanceID;
	uint packedIdx = sortedIndices[dataPos];
	uint idx = packedIdx & 0x7FFFFFFFu;

	// Transform from global buffer (standard path)
	StructuredBuffer<InstanceDescriptor> descriptors = ResourceDescriptorHeap[DescriptorBufIdx];
	StructuredBuffer<row_major matrix> globalTransforms = ResourceDescriptorHeap[GlobalTransformBufferIdx];
	uint slot = descriptors[idx].TransformSlot;
	row_major matrix world = globalTransforms[slot];

	// Per-instance terrain patch data
	StructuredBuffer<TerrainPatchData> terrainData = ResourceDescriptorHeap[TerrainDataIdx];
	TerrainPatchData patch = terrainData[idx]; // indexed by instance (dense, matches staging order)

	// Vertex position from bindless index + position buffers
	StructuredBuffer<uint> indices = ResourceDescriptorHeap[IndexBufferIdx];
	uint vertexID = indices[primitiveVertexID + BaseIndex];
    StructuredBuffer<float3> positions = ResourceDescriptorHeap[PosBufferIdx];
    float3 pos = positions[vertexID];

    float2 uv = (pos.xz + float2(16, 16)) * 0.03125f; // 1/32;
    uv.y = 1 - uv.y;
	
	float4 rect = patch.rect;
	float2 heightUV = rect.xy + uv * float2(rect.z - rect.x, rect.w - rect.y);
    
    Texture2D HeightTex = ResourceDescriptorHeap[HeightTexIdx];
    float height = HeightTex.SampleLevel(sampHeightFilter, heightUV, 0).r;

    float4 worldPosition = mul(float4(pos, 1), world);
    worldPosition.y += height * MaxHeight;
    worldPosition = mul(worldPosition, ViewProjection);
    
    output.Position = worldPosition;
    
    output.Depth = worldPosition.w;
    output.UV = uv;
    output.UV2 = heightUV;
	output.Level = patch.level.x;
	return output;
}

struct FragmentOutput
{
	float4 Albedo		: SV_TARGET0;
	float4 Normals		: SV_TARGET1;
	float4 Data			: SV_TARGET2;
	float  Depth		: SV_TARGET3;
};

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

FragmentOutput PS(VertexOutput input)
{
    FragmentOutput output;

    float3 terrainNormal = GetNormal(input.UV2);

    float4 color = float4(0, 0, 0, 0);
    float4 normal = float4(0, 0, 0, 0);
	
    float2 uv = input.UV2;
    uv.y = 1 - uv.y;

    Texture2DArray ControlMaps = ResourceDescriptorHeap[ControlMapsIdx];
    Texture2DArray DiffuseMaps = ResourceDescriptorHeap[DiffuseMapsIdx];
    Texture2DArray NormalMaps = ResourceDescriptorHeap[NormalMapsIdx];

    for (int i = 0; i < 4; ++i)
    {
        int startIndex = i * 4;
        
        float4 weights = ControlMaps.Sample(sampData, float3(uv, i));

        for (int j = 0; j < 4; ++j)
        {
            float weight = weights[j];
            
            if (weight <= 0)
                continue;

            int layer = startIndex + j;
            float2 texuv = uv * LayerTiling[layer].xy;
            float3 layerUV = float3(texuv, layer);
            
            float4 c = DiffuseMaps.Sample(sampData, layerUV);
            float4 n = NormalMaps.Sample(sampData, layerUV);
            
            color += c * weight;
            normal += n * weight;
        }
    }
	
    normal = float4(normal.xzy, 1);
    normal.xz = normal.xz * 2 - 1;
    normal = normalize(normal);

    terrainNormal = blend_linear(terrainNormal, normal.xyz);

    output.Albedo = color;
    output.Normals = float4(terrainNormal.xyz, 1); 
    output.Data = float4(0, 1, 0, 0);
    output.Depth = input.Depth;

    return output;
}

// Shadow pass: depth-only, apply height displacement
struct ShadowVSOutput
{
    float4 Position : SV_POSITION;
};

ShadowVSOutput VS_Shadow(uint primitiveVertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
    ShadowVSOutput output;

    StructuredBuffer<uint> sortedIndices = ResourceDescriptorHeap[SortedIndicesIdx];
    uint dataPos = InstanceBaseOffset + instanceID;
    uint packedIdx = sortedIndices[dataPos];
    uint idx = packedIdx & 0x7FFFFFFFu;

    StructuredBuffer<InstanceDescriptor> descriptors = ResourceDescriptorHeap[DescriptorBufIdx];
    StructuredBuffer<row_major matrix> globalTransforms = ResourceDescriptorHeap[GlobalTransformBufferIdx];
    uint slot = descriptors[idx].TransformSlot;
    row_major matrix world = globalTransforms[slot];

    StructuredBuffer<TerrainPatchData> terrainData = ResourceDescriptorHeap[TerrainDataIdx];
    TerrainPatchData patch = terrainData[idx];

    StructuredBuffer<uint> indices = ResourceDescriptorHeap[IndexBufferIdx];
    uint vertexID = indices[primitiveVertexID + BaseIndex];
    StructuredBuffer<float3> positions = ResourceDescriptorHeap[PosBufferIdx];
    float3 pos = positions[vertexID];

    float2 uv = (pos.xz + float2(16, 16)) * 0.03125f;
    uv.y = 1 - uv.y;

    float4 rect = patch.rect;
    float2 heightUV = rect.xy + uv * float2(rect.z - rect.x, rect.w - rect.y);
    
    Texture2D HeightTex = ResourceDescriptorHeap[HeightTexIdx];
    float height = HeightTex.SampleLevel(sampHeightFilter, heightUV, 0).r;

    float4 worldPosition = mul(float4(pos, 1), world);
    worldPosition.y += height * MaxHeight;
    worldPosition = mul(worldPosition, ViewProjection);

    output.Position = worldPosition;
    return output;
}

RasterizerState Wireframe
{
    FillMode = WireFrame;
    CullMode = None;
};

technique11 Terrain
{
    pass Opaque
    {
        SetRasterizerState(Wireframe);
        SetVertexShader(CompileShader(vs_6_6, VS()));
        SetPixelShader(CompileShader(ps_6_6, PS()));
    }
    pass Shadow
    {
        SetVertexShader(CompileShader(vs_6_6, VS_Shadow()));
    }
}
