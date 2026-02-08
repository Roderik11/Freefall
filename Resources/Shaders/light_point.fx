// Point Light Shader - Instanced rendering via bindless buffers
// Uses LightInstanceData from structured buffer for per-instance transforms and properties

#include "common.fx"

// Per-instance light data (matches C# struct)
struct LightInstanceData
{
    row_major float4x4 World;   // 64 bytes - Transform (position + range scale)
    float3 LightColor;          // 12 bytes
    float LightIntensity;       // 4 bytes
    float LightRange;           // 4 bytes
    float3 _Padding;            // 12 bytes
};                              // Total: 96 bytes

struct VSOutput
{
    float4 Position   : SV_POSITION;
    float4 Screen     : TEXCOORD0;
    float3 LightWorld : TEXCOORD1;
    float3 Color      : TEXCOORD2;
    float Intensity   : TEXCOORD3;
    float Range       : TEXCOORD4;
};

// Light Pass Push Constant Layout
// Slots 0-2: Light-specific textures and data
// Slot 7: Vertex positions (shared with unified geometry layout)
#define NormalTexIdx GET_INDEX(0)           // G-Buffer normal texture
#define DepthTexIdx GET_INDEX(1)            // G-Buffer depth texture 
#define LightInstanceBufferIdx GET_INDEX(2) // StructuredBuffer<LightInstanceData>
#define PosBufferIdx GET_INDEX(7)           // StructuredBuffer<float3> - sphere vertex positions

VSOutput VS(uint vertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
    VSOutput output = (VSOutput)0;

    // Read sphere vertex position from bindless buffer
    StructuredBuffer<float3> positions = ResourceDescriptorHeap[PosBufferIdx];
    float3 pos = positions[vertexID];

    // Read light instance data from bindless buffer
    StructuredBuffer<LightInstanceData> instances = ResourceDescriptorHeap[LightInstanceBufferIdx];
    LightInstanceData light = instances[instanceID];

    // Transform vertex position through World, View, Projection
    float4 worldPos = mul(float4(pos, 1.0f), light.World);
    float4 viewPos = mul(worldPos, View);
    float4 clipPos = mul(viewPos, Projection);
    
    output.Position = clipPos;
    output.Screen = clipPos;
    output.LightWorld = light.World[3].xyz;  // Light position in world space
    output.Color = light.LightColor;
    output.Intensity = light.LightIntensity;
    output.Range = light.LightRange;
    
    return output;
}

SamplerState Sampler : register(s0);
SamplerState SamplerPoint : register(s1);

// posFromDepth is defined in common.fx, returns float4

float4 PS(VSOutput IN) : SV_TARGET
{
    // DEBUG: Just output the light color to confirm geometry is being drawn
    return float4(IN.Color * IN.Intensity * 0.1, 1.0);
    
    /*
    // Sample G-buffer
    Texture2D<float4> normalTex = ResourceDescriptorHeap[NormalTexIdx];
    Texture2D<float> depthTex = ResourceDescriptorHeap[DepthTexIdx];
    
    float2 UV = IN.Screen.xy / IN.Screen.w;
    UV = 0.5f * (float2(UV.x, -UV.y) + 1);
    
    float4 vNormals = normalTex.Sample(Sampler, UV);
    float depth = depthTex.Sample(SamplerPoint, UV);
    
    // Reconstruct world position from depth
    float3 worldPosition = posFromDepth(UV, depth, CameraInverse).xyz;
    
    // Calculate light contribution
    float3 lightVec = IN.LightWorld - worldPosition;
    float dist = length(lightVec);
    float atten = 1 - saturate(dist / IN.Range);
    atten *= atten; // Quadratic falloff
    
    float3 lightDir = normalize(lightVec);
    float3 normal = normalize(mul(vNormals.xyz, (float3x3)ViewInverse));
    
    float NdotL = saturate(dot(normal, lightDir));
    
    return float4(IN.Color * IN.Intensity * atten * NdotL, 0);
    */
}

technique11 Standard
{
    pass Light
    {
        SetVertexShader(CompileShader(vs_6_6, VS()));
        SetPixelShader(CompileShader(ps_6_6, PS()));
    }
}
