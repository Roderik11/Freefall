#include "common.fx"
// @RenderState(RenderTargets=4)

// TerrainPatchData: per-instance data written by terrain_quadtree.hlsl compute shader.
// World transform is computed from patch rect — no per-patch TransformSlot needed.
struct TerrainPatchData
{
	float4 rect;    // (minX, maxZ, maxX, minZ) — Z components are swapped (matches compute shader)
	float2 level;   // (lod, geomorphBlend)
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
    float LODRange0;
    float MaxLodDepth;
}

cbuffer tiling : register(b2)
{
    float4 LayerTiling[32];
}

SamplerState sampData : register(s0); // WrappedAnisotropic
SamplerState sampHeight : register(s1); // ClampedPoint2D
SamplerState sampHeightFilter : register(s2); // ClampedBilinear2D

// Standard CDLOD: LODRange[level] = LODRange0 * 2^level.
// level 0 = finest (smallest range), level maxDepth = coarsest (largest).
float ComputeLodRange(int level)
{
    if (level < 0) return 0.0;
    return LODRange0 * (float)(1u << (uint)level);
}

// Standard CDLOD morph: ramp over the outer 30% of each LOD band.
// Morphing starts at 70% of the way from low to high and completes at high.
float GetMorphValue(float dist, float lodLevel)
{
    int lod = (int)lodLevel;
    float low  = ComputeLodRange(lod);
    float high = ComputeLodRange(lod + 1);

    int nextLevel = lod + 1;
    if (nextLevel > (int)MaxLodDepth) return 0.0;

    float morphStart = lerp(low, high, 0.7);
    return saturate((dist - morphStart) / (high - morphStart));
}

// CDLOD per-vertex morph: dual-sample fine/coarse heights and lerp.
// Morphs XZ by subtracting the grid fraction scaled by morph factor.
float3 MorphVertex(float3 pos, float2 rect_xy, float2 rect_size, float lodLevel, row_major matrix entityWorld, out float2 outHeightUV)
{
    const float gridRes = 32.0;
    const float invGridRes = 1.0 / gridRes;

    // Grid pos in [0, gridRes]
    float2 gridPos = pos.xz + float2(16, 16);

    // Fine UV → fine height → world pos → distance
    float2 uvFine = gridPos * invGridRes;
    uvFine.y = 1.0 - uvFine.y;
    float2 heightUVFine = rect_xy + uvFine * rect_size;
    Texture2D HeightTex = ResourceDescriptorHeap[HeightTexIdx];
    float fineH = HeightTex.SampleLevel(sampHeightFilter, heightUVFine, 0).r * MaxHeight;

    // Compute distance in world space
    float2 terrainXZ = heightUVFine * TerrainSize;
    float3 localPos = float3(terrainXZ.x, fineH, terrainXZ.y);
    float3 worldPos = mul(float4(localPos, 1), entityWorld).xyz;
    
    float3 camForDist = CameraPos;
    if (camForDist.y < MaxHeight) { camForDist.y = 0; worldPos.y = 0; }
    float dist = distance(worldPos, camForDist);

    // Per-vertex morph factor
    float morph = GetMorphValue(dist, lodLevel);

    // Compute grid fraction scaled by LOD level.
    // At level 0 (finest), we snap every-other vertex (step=1 grid unit).
    // At level N, each patch covers 2^N times the area with the same 32-vertex grid,
    // so the morph step in grid-space is always 1 grid unit — but we need to identify
    // which vertices are "odd" relative to the NEXT coarser level's grid.
    // The grid positions that survive at the next coarser level are those divisible by 2.
    // frac(gridPos * 0.5) gives 0.5 for odd-indexed verts, 0.0 for even.
    float2 fracOffset = frac(gridPos * 0.5) * 2.0;

    // Morph XZ: subtract fraction scaled by morph so odd verts slide to even neighbors
    float2 morphedGrid = gridPos - fracOffset * morph;

    // Coarse UV → coarse height
    float2 uvCoarse = morphedGrid * invGridRes;
    uvCoarse.y = 1.0 - uvCoarse.y;
    float2 heightUVCoarse = rect_xy + uvCoarse * rect_size;
    float coarseH = HeightTex.SampleLevel(sampHeightFilter, heightUVCoarse, 0).r * MaxHeight;

    // Lerp height: smooth transition from fine to coarse
    float finalH = lerp(fineH, coarseH, morph);

    // Output morphed UV for texturing (lerp UV too for consistent normal/texture reads)
    outHeightUV = lerp(heightUVFine, heightUVCoarse, morph);

    float3 result;
    result.xz = morphedGrid - float2(16, 16);
    result.y = finalH;
    return result;
}

struct VertexOutput
{
	float4 Position		: SV_POSITION;
	float2 UV			: TEXCOORD0;
	float2 UV2			: TEXCOORD1;
    float Depth			: TEXCOORD2;
	float Level			: TEXCOORD3;
	nointerpolation uint Flags : TEXCOORD4;
};

// Per-instance descriptor (matches C# InstanceDescriptor: 12 bytes)
struct InstanceDescriptor
{
    uint TransformSlot;
    uint MaterialId;
    uint CustomDataIdx;
};

// Compute world matrix from patch rect (no per-patch TransformSlot overhead).
// The single shared TransformSlot holds the terrain entity's world matrix (typically identity).
// Note: rect format is (minX, maxZ, maxX, minZ) — Z components are swapped.
// Use abs() to ensure positive scale since rect.w - rect.y is negative.
row_major matrix ComputePatchWorld(TerrainPatchData patch, row_major matrix entityWorld)
{
    float2 patchSize = abs(patch.rect.zw - patch.rect.xy) * TerrainSize;
    float gridRes = 32.0; // must match Mesh.CreatePatch grid resolution
    float2 patchScale = patchSize / gridRes;

    // Mesh vertices span [-16,+16] centered at origin, so translate to patch CENTER
    // (rect.xy + rect.zw) * 0.5 gives correct center since addition is commutative
    float2 patchCenter = (patch.rect.xy + patch.rect.zw) * 0.5 * TerrainSize;

    // Patch-local world transform
    row_major matrix patchWorld = {
        patchScale.x, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, patchScale.y, 0,
        patchCenter.x, 0, patchCenter.y, 1
    };

    return mul(patchWorld, entityWorld);
}

VertexOutput VS(uint primitiveVertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
	VertexOutput output;

	// Double-indirection: command signature → sorted index → original instance
	StructuredBuffer<uint> sortedIndices = ResourceDescriptorHeap[SortedIndicesIdx];
	uint dataPos = InstanceBaseOffset + instanceID;
	uint packedIdx = sortedIndices[dataPos];
	bool isOccluded = (packedIdx & 0x80000000u) != 0;
	uint idx = packedIdx & 0x7FFFFFFFu;

	// Get entity transform from global buffer (single shared slot — typically identity)
	StructuredBuffer<InstanceDescriptor> descriptors = ResourceDescriptorHeap[DescriptorBufIdx];
	StructuredBuffer<row_major matrix> globalTransforms = ResourceDescriptorHeap[GlobalTransformBufferIdx];
	uint slot = descriptors[idx].TransformSlot;
	row_major matrix entityWorld = globalTransforms[slot];

	// Per-instance terrain patch data
	StructuredBuffer<TerrainPatchData> terrainData = ResourceDescriptorHeap[TerrainDataIdx];
	TerrainPatchData patch = terrainData[idx];

	// Compute world matrix from patch rect — the key difference from terrain.fx
	row_major matrix world = ComputePatchWorld(patch, entityWorld);

	// Vertex position from bindless index + position buffers
	StructuredBuffer<uint> indices = ResourceDescriptorHeap[IndexBufferIdx];
	uint vertexID = indices[primitiveVertexID + BaseIndex];
    StructuredBuffer<float3> positions = ResourceDescriptorHeap[PosBufferIdx];
    float3 pos = positions[vertexID];

    // CDLOD per-vertex morphing
    float4 rect = patch.rect;
    float2 rectSize = float2(rect.z - rect.x, rect.w - rect.y);
    float2 heightUV;
    float3 morphedPos = MorphVertex(pos, rect.xy, rectSize, patch.level.x, entityWorld, heightUV);

    float4 worldPosition = mul(float4(morphedPos, 1), world);
    worldPosition = mul(worldPosition, ViewProjection);
    
    output.Position = worldPosition;
    
    // X-ray mode: push occluded instances to near depth
    if (isOccluded)
        output.Position.z = 0.0;
    
    float2 uv = (morphedPos.xz + float2(16, 16)) * 0.03125;
    uv.y = 1 - uv.y;
    output.Depth = worldPosition.w;
    output.UV = uv;
    output.UV2 = heightUV;
	output.Level = patch.level.x;
	output.Flags = isOccluded ? 1u : 0u;
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
    normal.xyz = normalize(normal.xyz);

    terrainNormal = blend_linear(terrainNormal, normal.xyz);
    
    // X-Ray occlusion debug mode (mode 4)
    bool isOccluded = input.Flags != 0;
    if (DebugMode == 4)
    {
        if (isOccluded)
        {
            float hue = frac(float(input.Level * 2654435761u) * (1.0 / 4294967296.0));
            float3 xrayColor = float3(
                abs(hue * 6.0 - 3.0) - 1.0,
                2.0 - abs(hue * 6.0 - 2.0),
                2.0 - abs(hue * 6.0 - 4.0)
            );
            xrayColor = saturate(xrayColor);
            
            output.Albedo = float4(xrayColor * 0.7, 0.5);
            output.Normals = float4(terrainNormal, 0.0);
            output.Data = float4(1, 0, 0, 1);
            output.Depth = 99999.0;
        }
        else
        {
            float luma = dot(color.rgb, float3(0.299, 0.587, 0.114));
            output.Albedo = float4(lerp(float3(luma, luma, luma), color.rgb, 0.3), color.a);
            output.Normals = float4(terrainNormal, 1.0);
            output.Data = float4(0, 0, 0, 1);
            output.Depth = input.Depth;
        }
        return output;
    }

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
    row_major matrix entityWorld = globalTransforms[slot];

    StructuredBuffer<TerrainPatchData> terrainData = ResourceDescriptorHeap[TerrainDataIdx];
    TerrainPatchData patch = terrainData[idx];

    // Compute world from rect
    row_major matrix world = ComputePatchWorld(patch, entityWorld);

    StructuredBuffer<uint> indices = ResourceDescriptorHeap[IndexBufferIdx];
    uint vertexID = indices[primitiveVertexID + BaseIndex];
    StructuredBuffer<float3> positions = ResourceDescriptorHeap[PosBufferIdx];
    float3 pos = positions[vertexID];

    // CDLOD per-vertex morphing (must match VS() for consistent shadow depth)
    float4 rect = patch.rect;
    float2 rectSize = float2(rect.z - rect.x, rect.w - rect.y);
    float2 heightUV;
    float3 morphedPos = MorphVertex(pos, rect.xy, rectSize, patch.level.x, entityWorld, heightUV);

    float4 worldPosition = mul(float4(morphedPos, 1), world);
    worldPosition = mul(worldPosition, ViewProjection);

    output.Position = worldPosition;
    return output;
}

RasterizerState Wireframe
{
    FillMode = WireFrame;
    CullMode = None;
};

technique11 GPUTerrain
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
