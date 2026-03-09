cbuffer PushConstants : register(b3)
{
    uint _reserved0;
    uint TerrainDataIdx;        // 1: Per-instance TerrainPatchData buffer
    uint DescriptorBufIdx;      // 2
    uint _reserved3;
    uint _reserved4;
    uint _reserved5;
    uint _reserved6;
    uint IndexBufferIdx;        // 7
    uint BaseIndex;             // 8
    uint PosBufferIdx;          // 9
    uint _reserved10;
    uint _reserved11;
    uint _reserved12;
    uint _reserved13;
    uint _reserved14;
    uint GlobalTransformBufferIdx; // 15
    uint DebugMode;             // 16
    uint HeightTexIdx;          // 17
    uint ControlMapsIdx;        // 18
    uint DiffuseMapsIdx;        // 19
    uint NormalMapsIdx;         // 20
    uint DecoControlMapIdx;     // 21: Decoration control texture
    uint _reserved22;
    uint CascadeBufferSRVIdx;   // 23: SRV: StructuredBuffer<CascadeData>
    uint ShadowCascadeCount;    // 24: uint: number of cascades
    uint CascadeIdxBufIdx;      // 25: SRV: per-entry cascade index (uint)
};

#include "common.fx"
// @RenderState(RenderTargets=5)

// TerrainPatchData: per-instance data written by terrain_quadtree.hlsl compute shader.
// World transform is computed from patch rect — no per-patch TransformSlot needed.
struct TerrainPatchData
{
	float4 rect;    // (minX, maxZ, maxX, minZ) — Z components are swapped (matches compute shader)
	float2 level;   // (lod, geomorphBlend)
	float2 padding;
};

cbuffer terrain : register(b1)
{
    float3 CameraPos;
    float HeightTexel;
    float MaxHeight;
    float _pad1;
    float2 TerrainSize;
    float2 TerrainOrigin;
}

cbuffer tiling : register(b2)
{
    float4 LayerTiling[32];
}

SamplerState sampData : register(s0); // WrappedAnisotropic
SamplerState sampHeight : register(s1); // ClampedPoint2D
SamplerState sampHeightFilter : register(s2); // ClampedBilinear2D

// Edge-stitched vertex: snaps odd edge vertices to even positions on edges
// adjacent to coarser neighbors. No morphing — hard LOD transition masked
// by sub-pixel screen-space error threshold.
// stitchMask bits: 0=S(-Z), 1=E(+X), 2=N(+Z), 3=W(-X)
float3 StitchVertex(float3 pos, float2 rect_xy, float2 rect_size, uint stitchMask, row_major matrix entityWorld, out float2 outHeightUV)
{
    const float gridRes = 32.0;
    const float invGridRes = 1.0 / gridRes;

    // Grid pos in [0, gridRes]
    float2 gridPos = pos.xz + float2(16, 16);

    // Snap odd edge vertices to even positions on stitched edges
    // South edge: gridPos.y == 0
    if ((stitchMask & 1u) && gridPos.y < 0.5)
        gridPos.x = round(gridPos.x * 0.5) * 2.0;
    // East edge: gridPos.x == 32
    if ((stitchMask & 2u) && gridPos.x > 31.5)
        gridPos.y = round(gridPos.y * 0.5) * 2.0;
    // North edge: gridPos.y == 32
    if ((stitchMask & 4u) && gridPos.y > 31.5)
        gridPos.x = round(gridPos.x * 0.5) * 2.0;
    // West edge: gridPos.x == 0
    if ((stitchMask & 8u) && gridPos.x < 0.5)
        gridPos.y = round(gridPos.y * 0.5) * 2.0;

    // Compute height UV
    float2 uv = gridPos * invGridRes;
    uv.y = 1.0 - uv.y;
    outHeightUV = rect_xy + uv * rect_size;

    // Sample height
    Texture2D HeightTex = ResourceDescriptorHeap[HeightTexIdx];
    float h = HeightTex.SampleLevel(sampHeightFilter, outHeightUV, 0).r * MaxHeight;

    float3 result;
    result.xz = gridPos - float2(16, 16);
    result.y = h;
    return result;
}

struct VertexOutput
{
	float4 Position		: SV_POSITION;
	float2 UV			: TEXCOORD0;
	float2 UV2			: TEXCOORD1;
    float Depth			: TEXCOORD2;
	float Level			: TEXCOORD3;
	nointerpolation uint TransformSlot : TEXCOORD4;
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

	// Direct access — terrain self-draws, no culler indirection needed.
	// instanceID maps directly into the compute shader's append buffers.
	uint idx = instanceID;

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

    // Edge-stitched vertex displacement (no morphing)
    float4 rect = patch.rect;
    float2 rectSize = float2(rect.z - rect.x, rect.w - rect.y);
    float2 heightUV;
    uint stitchMask = (uint)patch.level.y;
    float3 stitchedPos = StitchVertex(pos, rect.xy, rectSize, stitchMask, entityWorld, heightUV);

    float4 worldPosition = mul(float4(stitchedPos, 1), world);
    worldPosition = mul(worldPosition, ViewProjection);
    
    output.Position = worldPosition;

    float2 uv = (stitchedPos.xz + float2(16, 16)) * 0.03125;
    uv.y = 1 - uv.y;
    output.Depth = worldPosition.w;
    output.UV = uv;
    output.UV2 = heightUV;
	output.Level = patch.level.x;
	output.TransformSlot = slot;
	return output;
}

struct FragmentOutput
{
	float4 Albedo		: SV_TARGET0;
	float4 Normals		: SV_TARGET1;
	float4 Data			: SV_TARGET2;
	float  Depth		: SV_TARGET3;
	uint   EntityId		: SV_TARGET4;
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
	
    // Normalize splatmap weights: if they don't sum to 1.0, both color and normal
    // accumulations will be biased. Compute totalWeight once for both.
    float totalWeight = 0;
    for (int wi = 0; wi < 4; ++wi)
    {
        float4 w = ControlMaps.Sample(sampData, float3(uv, wi));
        totalWeight += w.x + w.y + w.z + w.w;
    }
    if (totalWeight > 0.001)
    {
        color.rgb /= totalWeight;
        normal /= totalWeight;

        // BC5_UNORM: R,G in [0,1], 0.5 = flat. Decode to signed tangent-space XY.
        float2 nXY = normal.rg * 2.0 - 1.0;

        // UDN (Partial Derivative Addition):
        // For flat terrain: tangent X → world X, tangent Y → world Z
        const float normalScale = 1.0;
        terrainNormal.x += nXY.x * normalScale;
        terrainNormal.z += nXY.y * normalScale;
        terrainNormal = normalize(terrainNormal);
    }

    output.Albedo = float4(color.rgb, 1.0);
    output.Normals = float4(terrainNormal.xyz, 1); 
    output.Data = float4(0.95, 0.0, 1.0, 1.0); // roughness=very matte, metal=0, ao=1, lit=PBR
    output.Depth = input.Depth;

    // Debug mode 5: Decoration control map overlay
    if (DebugMode == 5 && DecoControlMapIdx != 0)
    {
        Texture2DArray<uint4> decoCtrl = ResourceDescriptorHeap[DecoControlMapIdx];
        uint dW, dH, dS;
        decoCtrl.GetDimensions(dW, dH, dS);
        float2 ctrlUV = float2(input.UV2.x, 1.0 - input.UV2.y);
        int2 ctrlTexel = clamp(int2(ctrlUV * float2(dW, dH)), int2(0,0), int2(dW-1, dH-1));

        uint4 s0 = decoCtrl.Load(int4(ctrlTexel, 0, 0));
        uint4 s1 = decoCtrl.Load(int4(ctrlTexel, 1, 0));
        uint maxWt = 0;
        uint slotCount = 0;
        uint4 slices[2] = { s0, s1 };
        [unroll] for (uint k = 0; k < 8; k++)
        {
            uint packed = slices[k / 4][k % 4];
            uint sIdx = packed >> 8;
            uint wt = packed & 0xFF;
            if (sIdx == 255) break;
            maxWt = max(maxWt, wt);
            slotCount++;
        }

        // Heatmap: black=0, red=low(1-30), yellow=mid(31-128), green=high(129-255)
        float t = maxWt / 255.0;
        float3 decoColor;
        if (maxWt == 0)
            decoColor = color.rgb * 0.2; // dark = no coverage
        else if (t < 0.12)
            decoColor = float3(0.8, 0.1, 0.1); // red = very low weight
        else if (t < 0.5)
            decoColor = lerp(float3(0.8, 0.1, 0.1), float3(0.9, 0.9, 0.1), (t - 0.12) / 0.38); // red→yellow
        else
            decoColor = lerp(float3(0.9, 0.9, 0.1), float3(0.1, 0.9, 0.1), (t - 0.5) / 0.5); // yellow→green

        output.Albedo = float4(decoColor, 1.0);
    }

    output.EntityId = (input.TransformSlot << 8u);
    return output;
}

// Shadow pass — single-pass multi-cascade via per-cascade compute culling.
// CSEmitLeavesShadow emits compact (patch, cascadeIdx) pairs.
// VS_Shadow reads cascadeIdx from a per-entry buffer. instanceID = patchIdx directly.


struct ShadowVSOutput
{
    float4 Position : SV_POSITION;
    nointerpolation uint RTIndex : SV_RenderTargetArrayIndex;
};

ShadowVSOutput VS_Shadow(uint primitiveVertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
    ShadowVSOutput output;
    output.Position = float4(0, 0, 0, 1);
    output.RTIndex = 0;

    uint patchIdx = instanceID; // direct 1:1 mapping

    // Read cascade index from per-entry buffer
    StructuredBuffer<uint> cascadeIndices = ResourceDescriptorHeap[CascadeIdxBufIdx];
    uint cascadeIdx = cascadeIndices[patchIdx];

    // Patch data lookup (same as opaque VS)
    StructuredBuffer<InstanceDescriptor> descriptors = ResourceDescriptorHeap[DescriptorBufIdx];
    StructuredBuffer<row_major matrix> globalTransforms = ResourceDescriptorHeap[GlobalTransformBufferIdx];
    uint slot = descriptors[patchIdx].TransformSlot;
    row_major matrix entityWorld = globalTransforms[slot];

    StructuredBuffer<TerrainPatchData> terrainData = ResourceDescriptorHeap[TerrainDataIdx];
    TerrainPatchData patch = terrainData[patchIdx];

    row_major matrix world = ComputePatchWorld(patch, entityWorld);

    StructuredBuffer<uint> indices = ResourceDescriptorHeap[IndexBufferIdx];
    uint vertexID = indices[primitiveVertexID + BaseIndex];
    StructuredBuffer<float3> positions = ResourceDescriptorHeap[PosBufferIdx];
    float3 pos = positions[vertexID];

    // Edge-stitched vertex displacement (must match VS() for consistent shadow depth)
    float4 rect = patch.rect;
    float2 rectSize = float2(rect.z - rect.x, rect.w - rect.y);
    float2 heightUV;
    uint stitchMask = (uint)patch.level.y;
    float3 stitchedPos = StitchVertex(pos, rect.xy, rectSize, stitchMask, entityWorld, heightUV);

    float4 worldPosition = mul(float4(stitchedPos, 1), world);

    // Project using cascade-specific VP from cascade buffer
    StructuredBuffer<CascadeData> cascades = ResourceDescriptorHeap[CascadeBufferSRVIdx];
    output.Position = mul(worldPosition, cascades[cascadeIdx].VP);
    output.RTIndex = cascadeIdx;
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
        SetPixelShader(CompileShader(ps_6_6, PS_Shadow()));
    }
}

// Minimal depth-only PS — required for SV_RenderTargetArrayIndex propagation
void PS_Shadow(ShadowVSOutput input) { }
