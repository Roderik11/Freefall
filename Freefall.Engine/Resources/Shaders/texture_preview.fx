// Fullscreen quad shader for inspector texture preview.
// Renders outside the deferred pipeline — samples a single texture with channel masking.
// @RenderState(CullMode=None)

cbuffer PushConstants : register(b3)
{
    uint TextureIdx;          // 0
    uint _pad1;               // 1
    uint _pad2;               // 2
    uint _pad3;               // 3
};

cbuffer PreviewConstants : register(b1)
{
    float4 ChannelMask;       // e.g. (1,0,0,0) for red only, (1,1,1,0) for RGB
    float  ShowAlpha;         // 1 = show alpha as grayscale, 0 = use mask
    float  MipLevel;          // mip level to sample (0 = auto)
    float2 _padTex;
};

SamplerState Sampler : register(s0);

struct VSOutput
{
    float4 Position : SV_POSITION;
    float2 TexCoord : TEXCOORD0;
};

VSOutput VS(uint vertexID : SV_VertexID)
{
    VSOutput output;

    // Fullscreen triangle from vertex ID (3 vertices cover entire screen)
    float2 uv = float2((vertexID << 1) & 2, vertexID & 2);
    output.Position = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    output.TexCoord = float2(uv.x, 1.0 - uv.y);

    return output;
}

float4 PS(VSOutput input) : SV_Target0
{
    Texture2D tex = ResourceDescriptorHeap[TextureIdx];
    float4 texColor = tex.SampleLevel(Sampler, input.TexCoord, MipLevel);

    if (ShowAlpha > 0.5)
    {
        // Show alpha channel as grayscale
        return float4(texColor.aaa, 1.0);
    }

    // Apply channel mask
    float3 rgb = texColor.rgb * ChannelMask.xyz;

    // If only one channel is active, show as grayscale for visibility
    float activeCount = ChannelMask.x + ChannelMask.y + ChannelMask.z;
    if (activeCount == 1.0)
        rgb = float3(dot(rgb, float3(1, 1, 1)).xxx);

    return float4(rgb, 1.0);
}

technique11 GBuffer
{
    pass Forward
    {
        SetRenderState(RenderTargets=1, DepthTest=false, DepthWrite=false);
        SetVertexShader(CompileShader(vs_6_6, VS()));
        SetPixelShader(CompileShader(ps_6_6, PS()));
    }
}
