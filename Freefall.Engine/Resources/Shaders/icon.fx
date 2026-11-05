// An item's icon: its own meshes and materials (albedo, normal, smoothness, metallic, AO from the bindless materials
// buffer, as the G-buffer reads them) lit by a fixed studio rig in view space, on a transparent background.
// Drawn outside the deferred pipeline by IconRenderer: one draw per mesh part, everything per draw in push constants
// (a material's constant buffer is one per frame, so a World set per draw would leave every draw with the last one).
// Orthographic: the projection is a scale and an offset per axis. Output: the tonemapped colour in linear light,
// alpha 1 where something is drawn; IconRenderer averages the supersamples and writes sRGB.

cbuffer PushConstants : register(b3)
{
    uint PosBufferIdx;        // 0
    uint NormBufferIdx;       // 1
    uint UVBufferIdx;         // 2
    uint IndexBufferIdx;      // 3
    uint BaseIndex;           // 4
    uint MaterialsIdx;        // 5: StructuredBuffer<MaterialData> (Material.MaterialsBufferIndex)
    uint MaterialID;          // 6
    uint Flags;               // 7: bits 0-1 the half kept (0 both, 1 x >= SideCut, 2 x < SideCut), bit 2 alpha test
    // Mesh space to view space, row_major 4x3: the rows of the rotation-scale, then the translation (xyz). In float4s,
    // as a float3 never straddles a 16-byte register (one after another they would shift); w carries the projection.
    float4 RowX;              // 8-11:  w = view x to clip
    float4 RowY;              // 12-15: w = view y to clip
    float4 RowZ;              // 16-19: w = view z to depth (0 near, 1 far), the scale
    float4 RowT;              // 20-23: w = the same, the offset
    float IconExposure;       // 24
    float SideCut;            // 25: the mesh-space X a half is cut at
    float _pad26;             // 26
    float _pad27;             // 27
    float4 PartCenter;        // 28-31: the middle of the part (its kept half) in view space; w = how dark its inside goes (0 = not)
};

#define WorldViewX RowX.xyz
#define WorldViewY RowY.xyz
#define WorldViewZ RowZ.xyz
#define WorldViewT RowT.xyz

#include "common.fx"
// @RenderState(RenderTargets=1, DepthWrite=true, DepthFunc=LessEqual, CullMode=None, RenderTargetFormat=R16G16B16A16_Float)

SamplerState Sampler : register(s0);

struct VSOutput
{
    float4 Position : SV_POSITION;
    float3 Normal   : NORMAL;
    float3 ViewPos  : TEXCOORD1;
    float2 TexCoord : TEXCOORD0;
    float  MeshX    : TEXCOORD2;
};

float3 ToView(float3 p)
{
    return p.x * WorldViewX + p.y * WorldViewY + p.z * WorldViewZ + WorldViewT;
}

VSOutput VS(uint vertexID : SV_VertexID)
{
    VSOutput output;

    StructuredBuffer<uint>   indices   = ResourceDescriptorHeap[IndexBufferIdx];
    StructuredBuffer<float3> positions = ResourceDescriptorHeap[PosBufferIdx];
    StructuredBuffer<float3> normals   = ResourceDescriptorHeap[NormBufferIdx];
    StructuredBuffer<float2> uvs       = ResourceDescriptorHeap[UVBufferIdx];

    uint idx = indices[vertexID + BaseIndex];
    float3 pos  = positions[idx];
    float3 norm = normals[idx];

    float3 viewPos = ToView(pos);
    output.Position = float4(viewPos.x * RowX.w, viewPos.y * RowY.w, viewPos.z * RowZ.w + RowT.w, 1.0);
    // Normals by the rotation-scale itself: the parts are scaled evenly, and a mirrored one is flipped in the PS
    output.Normal   = norm.x * WorldViewX + norm.y * WorldViewY + norm.z * WorldViewZ;
    output.ViewPos  = viewPos;
    output.TexCoord = uvs[idx];
    output.TexCoord.y = 1 - output.TexCoord.y;
    output.MeshX    = pos.x;
    return output;
}

// --- The studio: directions towards the light in view space (x right, y up, z away from the camera) ---
static const float3 KeyDir   = normalize(float3(-0.55, 0.80, -0.55));   // upper left, in front
static const float3 KeyColor = float3(1.00, 0.95, 0.88) * 1.7;
static const float3 FillDir  = normalize(float3(0.85, 0.05, -0.45));    // right, low
static const float3 FillColor = float3(0.72, 0.80, 0.92) * 0.45;
static const float3 RimDir   = normalize(float3(0.45, 0.55, 0.75));     // behind, upper right: the outline
static const float3 RimColor = float3(1.00, 0.97, 0.92) * 1.2;
// The lights' diffuse share against their highlights. Measured against the pack's drawings the icons replace (linear
// mean of the covered pixels, 2026-11-05): with one intensity for both, polished plate came out 1.5-4x brighter than
// the pack while wood and leather matched it; the key and rim were lowered for the metal and the diffuse given back here
static const float DiffuseGain = 1.4;

// A soft studio around the item for what metal reflects: dark below, a lit ceiling, and a softbox where the key is
float3 Environment(float3 R, float roughness)
{
    float up = R.y;
    // Dim: polished plate reflects it nearly whole, and at 0.55 for the ceiling the plate icons came out 2-4x brighter
    // than the pack's drawings they replace (linear mean of the covered pixels), their highlights near white
    float3 ground = float3(0.03, 0.028, 0.025);
    float3 horizon = float3(0.15, 0.145, 0.14);
    float3 ceiling = float3(0.30, 0.30, 0.31);
    float3 sky = up < 0 ? lerp(horizon, ground, saturate(-up * 2.5)) : lerp(horizon, ceiling, saturate(up * 1.4));
    float sharp = lerp(48.0, 2.0, saturate(roughness));
    float box = pow(saturate(dot(R, KeyDir)), sharp) * lerp(3.0, 0.35, saturate(roughness));
    float boxRim = pow(saturate(dot(R, RimDir)), sharp) * lerp(1.5, 0.2, saturate(roughness));
    // A rough surface sees the room averaged
    float3 average = float3(0.14, 0.137, 0.134);
    return lerp(sky, average, saturate(roughness * roughness)) + box * KeyColor * 0.22 + boxRim * RimColor * 0.25;
}

float D_GGX(float NdotH, float a)
{
    float a2 = a * a;
    float d = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (3.14159265 * d * d);
}

float V_Smith(float NdotV, float NdotL, float a)
{
    float k = a * 0.5;
    float gv = NdotV / (NdotV * (1.0 - k) + k);
    float gl = NdotL / (NdotL * (1.0 - k) + k);
    return gv * gl / max(4.0 * NdotV * NdotL, 1e-4);
}

float3 Light(float3 N, float3 V, float3 L, float3 color, float3 diffuse, float3 F0, float a)
{
    float NdotL = saturate(dot(N, L));
    if (NdotL <= 0) return 0;
    float3 H = normalize(L + V);
    float NdotV = max(dot(N, V), 1e-3);
    float NdotH = saturate(dot(N, H));
    float VdotH = saturate(dot(V, H));
    float3 F = F0 + (1.0 - F0) * pow(1.0 - VdotH, 5.0);
    float3 spec = D_GGX(NdotH, a) * V_Smith(NdotV, NdotL, a) * F;
    return (diffuse / 3.14159265 * (1.0 - F) * DiffuseGain + spec) * color * NdotL * 3.14159265;
}

float3 Tonemap(float3 x)
{
    // ACES fitted (Narkowicz)
    return saturate((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14));
}

float4 PS(VSOutput input) : SV_Target0
{
    uint side = Flags & 3u;
    if (side == 1u && input.MeshX < SideCut) discard;
    if (side == 2u && input.MeshX >= SideCut) discard;

    StructuredBuffer<MaterialData> materials = ResourceDescriptorHeap[MaterialsIdx];
    MaterialData mat = materials[MaterialID];

    float4 color = float4(0.5, 0.5, 0.5, 1);
    if (mat.AlbedoIdx != 0)
    {
        Texture2D albedoTex = ResourceDescriptorHeap[NonUniformResourceIndex(mat.AlbedoIdx)];
        color = albedoTex.Sample(Sampler, input.TexCoord);
    }
    color.rgb *= mat.Tint;
    if ((Flags & 4u) != 0u) clip(color.a - 0.25f);

    float roughness = 0.65;
    float metal = 0.0;
    float ao = 1.0;
    if (mat.RoughnessIdx != 0) { Texture2D t = ResourceDescriptorHeap[NonUniformResourceIndex(mat.RoughnessIdx)]; roughness = 1.0 - t.Sample(Sampler, input.TexCoord).r; }
    if (mat.MetallicIdx  != 0) { Texture2D t = ResourceDescriptorHeap[NonUniformResourceIndex(mat.MetallicIdx)];  metal = t.Sample(Sampler, input.TexCoord).r; }
    if (mat.AOIdx        != 0) { Texture2D t = ResourceDescriptorHeap[NonUniformResourceIndex(mat.AOIdx)];        ao = t.Sample(Sampler, input.TexCoord).r; }

    // Two-sided: the side the camera sees is the front (open shells, a helmet's inside)
    float3 dp1 = ddx(input.ViewPos);
    float3 dp2 = ddy(input.ViewPos);
    float3 face = normalize(cross(dp1, dp2));
    if (face.z > 0) face = -face;                  // towards the camera (-z)
    float3 N = normalize(input.Normal);
    if (dot(N, face) < 0) N = -N;

    if (mat.NormalIdx != 0)
    {
        float2 duv1 = ddx(input.TexCoord);
        float2 duv2 = ddy(input.TexCoord);
        float3 dp2perp = cross(dp2, N);
        float3 dp1perp = cross(N, dp1);
        float3 T = dp2perp * duv1.x + dp1perp * duv2.x;
        float3 B = dp2perp * duv1.y + dp1perp * duv2.y;
        float handedness = dot(cross(T, B), N) < 0.0 ? -1.0 : 1.0;
        T *= handedness;
        float invmax = rsqrt(max(max(dot(T, T), dot(B, B)), 1e-20));
        float3x3 TBN = float3x3(T * invmax, B * invmax, N);

        Texture2D normalTex = ResourceDescriptorHeap[NonUniformResourceIndex(mat.NormalIdx)];
        float2 nXY = normalTex.Sample(Sampler, input.TexCoord).rg * 2.0 - 1.0;
        nXY.y = -nXY.y;
        float3 texNormal = float3(nXY, sqrt(max(0.001, 1.0 - dot(nXY, nXY))));
        N = normalize(mul(texNormal, TBN));
    }

    roughness = clamp(roughness, 0.08, 1.0);
    float a = roughness * roughness;
    float3 V = float3(0, 0, -1);
    float3 albedo = color.rgb;
    float3 diffuse = albedo * (1.0 - metal);
    float3 F0 = lerp(0.04, albedo, metal);

    float3 lit = Light(N, V, KeyDir, KeyColor, diffuse, F0, a)
               + Light(N, V, FillDir, FillColor, diffuse, F0, a)
               + Light(N, V, RimDir, RimColor, diffuse, F0, a);

    // Ambient: a hemisphere for the diffuse, the studio for what is reflected
    float NdotV = saturate(dot(N, V));
    float3 hemi = lerp(float3(0.12, 0.115, 0.11), float3(0.46, 0.46, 0.48), N.y * 0.5 + 0.5);
    float3 Fr = F0 + (max(1.0 - roughness, F0) - F0) * pow(1.0 - NdotV, 5.0);
    float3 R = reflect(-V, N);
    lit += (diffuse * hemi * (1.0 - Fr) + Environment(R, roughness) * Fr) * ao;

    // The inside: a surface facing in towards the part's middle (a helmet's lining, into a boot or a gauntlet's cuff)
    // is in the dark there; the light here comes from all round, so without this an inside shows as lit as the outside,
    // with whatever the generator baked onto it (a face inside the helmet)
    float3 outward = input.ViewPos - PartCenter.xyz;
    float inward = -dot(N, outward * rsqrt(max(dot(outward, outward), 1e-8)));
    // Only what faces clearly in: a part of several pieces (a couter over a vambrace) has its middle between them,
    // and a piece's front at a slant to it is outside
    lit *= lerp(1.0, 0.12, smoothstep(0.3, 0.7, inward) * PartCenter.w);

    return float4(Tonemap(lit * IconExposure), 1.0);
}

technique11 Icon
{
    pass Forward
    {
        SetVertexShader(CompileShader(vs_6_6, VS()));
        SetPixelShader(CompileShader(ps_6_6, PS()));
    }
}
