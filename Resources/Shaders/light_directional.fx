#include "common.fx"
// @RenderState(DepthTest=false, DepthWrite=false, Blend=Additive)

// Light Pass Push Constant Layout
// Slots 0-3: Light-specific textures (procedural fullscreen quad, no vertex buffers)
#define NormalTexIdx GET_INDEX(0)   // G-Buffer normal texture
#define DepthTexIdx GET_INDEX(1)    // G-Buffer depth texture (hardware)
#define ShadowMapIdx GET_INDEX(2)   // Shadow map texture array
#define DepthGBufIdx GET_INDEX(3)   // G-Buffer linear depth (R32_Float, for debug viz)

// Light params from ObjectConstants (Slot 2, b1)
cbuffer ObjectConstants : register(b1)
{
    float3 LightColor;
    float LightIntensity;
    float3 LightDirection;
    float _pad0;
    
    // Shadow cascade data
    row_major float4x4 LightSpaces[4];  // Light view-projection matrices per cascade
    float4 Cascades[4];                  // X=near, Y=far for each cascade
    
    int DebugVisualizationMode;          // 0=normal, 1=cascade colors, 2=shadow factor, 3=depth
    float3 _pad1;
};

SamplerState Sampler : register(s0);
SamplerState ShadowClampSampler : register(s2); // Linear+Clamp for shadow map sampling (debug)
SamplerComparisonState ShadowSampler : register(s3); // Comparison+Bilinear for PCF

struct VSOutput
{
    float4 Position : SV_POSITION;
    float2 TexCoord : TEXCOORD0;
};

VSOutput VS(uint vertexID : SV_VertexID)
{
    VSOutput output;
    
    // Procedural fullscreen quad from SV_VertexID (TriangleStrip, 4 verts)
    float2 pos[4] = {
        float2(-1, 1),
        float2(1, 1),
        float2(-1, -1),
        float2(1, -1)
    };
    float2 uv[4] = {
        float2(0, 0),
        float2(1, 0),
        float2(0, 1),
        float2(1, 1)
    };

    output.Position = float4(pos[vertexID], 0.0f, 1.0f);
    output.TexCoord = uv[vertexID];
    return output;
}

float4 PS(VSOutput input) : SV_Target
{
    Texture2D NormalTex = ResourceDescriptorHeap[NormalTexIdx];
    Texture2D DepthTex = ResourceDescriptorHeap[DepthTexIdx];
    Texture2DArray ShadowMap = ResourceDescriptorHeap[ShadowMapIdx];
    
    float3 normal = NormalTex.Sample(Sampler, input.TexCoord).xyz;
    float depth = DepthTex.Sample(Sampler, input.TexCoord).r;
    
    // Mode 3: Visualize GBuffer linear depth
    if (DebugVisualizationMode == 3)
    {
        Texture2D DepthGBuf = ResourceDescriptorHeap[DepthGBufIdx];
        float gbufDepth = DepthGBuf.Sample(Sampler, input.TexCoord).r;
        return float4(gbufDepth, gbufDepth, gbufDepth, 1);
    }
    
    // Skip pixels at max depth (sky)
    if (depth >= 1.0f)
        return float4(0, 0, 0, 0);
    
    // Reconstruct world position from depth
    float4 worldPos = posFromDepth(input.TexCoord, depth, CameraInverse);
    
    // Compute NdotL early — surfaces facing away from the light receive no
    // direct illumination, so skip shadow sampling entirely.  This prevents
    // shadows from projecting *through* geometry (e.g. tree shadows on a
    // cliff face that is turned away from the light).
    float3 L = normalize(-LightDirection);
    float NdotL = max(dot(normal, L), 0.0);
    
    if (NdotL <= 0.0 && DebugVisualizationMode == 0)
        return float4(0, 0, 0, 0);
    
    // Calculate view-space depth for cascade selection
    // worldPos is camera-relative (from zero-translation CameraInverse),
    // so use dot product with camera forward (View column 2) to get linear depth
    // without the translation component that's in the full View matrix
    float viewDepth = dot(worldPos.xyz, float3(View._13, View._23, View._33));
    
    // Select cascade based on view depth
    int cascadeIndex = 3;
    [unroll]
    for (int i = 0; i < 4; i++)
    {
        if (viewDepth < Cascades[i].y)
        {
            cascadeIndex = i;
            break;
        }
    }
    
    // Calculate shadow with cascade blending
    float shadowFactor = 1.0f;
    float4 lightSpacePos = float4(0,0,0,1);
    float2 shadowUV = float2(0,0);
    float sampledDepth = 0;
    
    if (ShadowMapIdx > 0) // Only sample if shadow map is bound
    {
        // Transform world position to light space for current cascade
        lightSpacePos = mul(worldPos, LightSpaces[cascadeIndex]);
        lightSpacePos /= lightSpacePos.w;
        
        // Convert from [-1,1] to [0,1] UV space
        shadowUV = lightSpacePos.xy * 0.5f + 0.5f;
        shadowUV.y = 1.0f - shadowUV.y; // Flip Y for D3D
        
        // Bounds check: skip shadow for pixels outside the shadow map
        if (shadowUV.x >= 0.0f && shadowUV.x <= 1.0f && shadowUV.y >= 0.0f && shadowUV.y <= 1.0f)
        {
            sampledDepth = ShadowMap.SampleLevel(ShadowClampSampler, float3(shadowUV, cascadeIndex), 0).r;
            float zScale = abs(LightSpaces[cascadeIndex]._33);
            shadowFactor = GetShadowFactor(ShadowMap, ShadowSampler, shadowUV, lightSpacePos.z, cascadeIndex, normal, LightDirection, zScale, input.Position.xy);
            
            // Cascade blending: cross-fade near cascade boundaries to eliminate seams
            // Outgoing blend: fade toward next cascade at far edge
            if (cascadeIndex < 3)
            {
                float cascadeRange = Cascades[cascadeIndex].y - Cascades[cascadeIndex].x;
                float blendZone = cascadeRange * 0.1f;
                float distToEdge = Cascades[cascadeIndex].y - viewDepth;
                
                if (distToEdge < blendZone)
                {
                    float blendFactor = distToEdge / blendZone; // 1 at start of zone, 0 at boundary
                    
                    int nextCascade = cascadeIndex + 1;
                    float4 nextLSP = mul(worldPos, LightSpaces[nextCascade]);
                    nextLSP /= nextLSP.w;
                    float2 nextUV = nextLSP.xy * 0.5f + 0.5f;
                    nextUV.y = 1.0f - nextUV.y;
                    
                    if (nextUV.x >= 0.0f && nextUV.x <= 1.0f && nextUV.y >= 0.0f && nextUV.y <= 1.0f)
                    {
                        float nextZScale = abs(LightSpaces[nextCascade]._33);
                        float nextShadow = GetShadowFactor(ShadowMap, ShadowSampler, nextUV, nextLSP.z, nextCascade, normal, LightDirection, nextZScale, input.Position.xy);
                        shadowFactor = lerp(nextShadow, shadowFactor, blendFactor);
                    }
                }
            }

        }
    }
    
    // Debug visualization modes
    if (DebugVisualizationMode == 1)
    {
        // Cascade index as color: R=0, G=1, B=2, Y=3
        float4 cascadeColors[4] = { float4(1,0,0,1), float4(0,1,0,1), float4(0,0,1,1), float4(1,1,0,1) };
        return cascadeColors[cascadeIndex];
    }
    if (DebugVisualizationMode == 2)
    {
        // Raw shadow factor as grayscale
        return float4(shadowFactor.xxx, 1);
    }
    
    return float4(LightColor * LightIntensity * NdotL * shadowFactor, 1.0f);
}

technique11 Standard
{
    pass Light
    {
        SetVertexShader(CompileShader(vs_6_6, VS()));
        SetPixelShader(CompileShader(ps_6_6, PS()));
    }
}
