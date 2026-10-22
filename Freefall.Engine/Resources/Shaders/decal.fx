// Decal materials (DECALS_PLAN.md): the effect a decal .mat names, so the material inspector, the .mat's Parameters and
// Textures, asset_set_properties and material_set_textures handle its settings (decal_settings.hlsli). Nothing draws
// with it: decals are applied by a compute pass over the G-buffer (decals_cs.hlsl, Graphics/DecalPass.cs), which reads
// each decal's settings block by its MaterialId. The one pass below is there because a material needs an effect that
// compiles; its name is no render pass, so no batch ever draws it.
//
// @RenderState(RenderTargets=1, DepthWrite=false, Blend=AlphaBlend, CullMode=None)

cbuffer PushConstants : register(b3)
{
    uint _reserved0;
};

#include "common.fx"
#include "decal_settings.hlsli"

float4 VS(uint vertexID : SV_VertexID) : SV_Position
{
    return float4(0, 0, 0, 0);
}

float4 PS() : SV_Target0
{
    return float4(0, 0, 0, 0);
}

technique11 Decal
{
    pass Decal
    {
        SetVertexShader(CompileShader(vs_6_6, VS()));
        SetPixelShader(CompileShader(ps_6_6, PS()));
    }
}
