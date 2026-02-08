#include "common.fx"
// @RenderState(RenderTargets=4)

struct TerrainPatch
{
	row_major float4x4 transform;
	float4 rect;
	float2 level;
	float2 padding;
};

// Terrain Push Constant Layout (Slots 0-5)
// Uses unified GET_INDEX approach like all other shaders
#define PosBufferIdx GET_INDEX(0)       // StructuredBuffer<float3> - patch vertex positions
#define TerrainDataIdx GET_INDEX(1)     // StructuredBuffer<TerrainPatch> - per-patch instance data
#define HeightTexIdx GET_INDEX(2)       // Height map texture
#define ControlMapsIdx GET_INDEX(3)     // Control/splat maps (Texture2DArray)
#define DiffuseMapsIdx GET_INDEX(4)     // Diffuse texture array
#define NormalMapsIdx GET_INDEX(5)      // Normal texture array

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

VertexOutput VS(uint vertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
	VertexOutput output;

    StructuredBuffer<float3> positions = ResourceDescriptorHeap[PosBufferIdx]; // PosBuffer (PushConstants)
    float3 pos = positions[vertexID];

    float2 uv = (pos.xz + float2(16, 16)) * 0.03125f; // 1/32;
    uv.y = 1 - uv.y;
	
	StructuredBuffer<TerrainPatch> terrainData = ResourceDescriptorHeap[TerrainDataIdx]; // TerrainData (PushConstants)
	TerrainPatch patch = terrainData[instanceID];
	float4 rect = patch.rect;
	float2 heightUV = rect.xy + uv * float2(rect.z - rect.x, rect.w - rect.y);
    
    Texture2D HeightTex = ResourceDescriptorHeap[HeightTexIdx];
    float height = HeightTex.SampleLevel(sampHeightFilter, heightUV, 0).r;

    float4 worldPosition = mul(float4(pos, 1), patch.transform);
    worldPosition.y += height * MaxHeight;
    worldPosition = mul(worldPosition, ViewProjection);
    
    output.Position = worldPosition;
    output.Depth = worldPosition.w; // View-space Z (linear)
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
	
    // Scale by MaxHeight to get world-space height differences
    // Then divide by texel world size for proper gradient
    float texelWorldSize = TerrainSize.x * HeightTexel; // World units per texel
    float heightScale = MaxHeight / texelWorldSize;
    
	float3 n;
	n.z = (h[0] - h[3]) * heightScale;
	n.x = (h[1] - h[2]) * heightScale;
	n.y = 1.0f; // Up direction

	return normalize(n);
}

FragmentOutput PS(VertexOutput input)
{
    FragmentOutput output;

    float3 terrainNormal = GetNormal(input.UV2);

    float4 color = float4(0, 0, 0, 0);
    float4 normal = float4(0, 0, 0, 0);
	
    float2 uv = input.UV2;
    uv.y = 1 - uv.y; // Match Apex reference for splatting/texturing

    // Get texture arrays from bindless heap
    // Slot 1: ControlMaps (Texture2DArray)
    // Slot 2: DiffuseMaps (Texture2DArray)  
    // Slot 3: NormalMaps (Texture2DArray)
    Texture2DArray ControlMaps = ResourceDescriptorHeap[ControlMapsIdx];
    Texture2DArray DiffuseMaps = ResourceDescriptorHeap[DiffuseMapsIdx];
    Texture2DArray NormalMaps = ResourceDescriptorHeap[NormalMapsIdx];

    // Sample all 4 control map slices and blend 16 layers
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
	
    // Normalize the accumulated normal
    normal = float4(normal.xzy, 1);
    normal.xz = normal.xz * 2 - 1;
    normal = normalize(normal);

    terrainNormal = blend_linear(terrainNormal, normal.xyz);
    // NOTE: Freefall uses world-space normals (unlike Apex which uses view-space)
    
    output.Albedo = color;

    output.Normals = float4(terrainNormal.xyz, 1); 
    output.Data = float4(0, 1, 0, 0); // Material info (G=Roughness?)
    output.Depth = input.Depth;

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
}
