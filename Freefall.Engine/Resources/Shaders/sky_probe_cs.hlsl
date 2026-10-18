// The sky probe (ATMOSPHERE_PLAN stage 4): the sky as a cubemap round the camera, every frame, for the ambient light
// and reflections (Graphics/Atmosphere.cs dispatches it after CSAmbient).
//   CSProbeFaces      mip 0, 64² per face: the sky's light and the cloud layer above the horizon (as the dome draws
//                     them, sky_common.fx), below it the air's light and the lit ground behind it. No sun or moon discs
//                     and no stars: the sun and the moons are the directional light, the stars too faint to matter.
//   CSProbeDownsample the mip chain, 2×2 box per face (glass looks its roughness up in it)
//   CSProbeSH         SH9 of the 16² mip, convolved with the cosine lobe and divided by π: sky_common.fx SkyAmbient(N)
//                     reads the light on a surface facing N as a multiple of its albedo (E / π × Exposure), as
//                     composition's ambient always has. One group of 64.
// Light is pre-exposed (physical × Exposure), like everything the sky gives. Docs: physical_atmosphere.md §12.

#pragma kernel CSProbeFaces
#pragma kernel CSProbeDownsample
#pragma kernel CSProbeSH

cbuffer PushConstants : register(b3)
{
    uint OutputUAVIdx;  // RWTexture2DArray<float4>: the mip written (CSProbeFaces: mip 0)
    uint SourceUAVIdx;  // RWTexture2DArray<float4>: CSProbeDownsample the mip above; CSProbeSH the mip projected (the
                        // whole cube stays in UAV state while it is made: no per-mip transitions)
    uint SizeIdx;       // texels per face edge: of the mip written, or (CSProbeSH) of the mip projected
    uint SHUAVIdx;      // RWStructuredBuffer<float4>, 9 entries (CSProbeSH)
};

#include "common.fx"
#include "sky_common.fx"

SamplerState linearWrap : register(s0);    // the cloud noise

// Direction of a texel of cube face `face` (D3D order +X −X +Y −Y +Z −Z), uv in −1..1 (u right, v down)
float3 CubeDirection(uint face, float2 uv)
{
    float3 d;
    switch (face)
    {
        case 0:  d = float3( 1.0, -uv.y, -uv.x); break;
        case 1:  d = float3(-1.0, -uv.y,  uv.x); break;
        case 2:  d = float3( uv.x,  1.0,  uv.y); break;
        case 3:  d = float3( uv.x, -1.0, -uv.y); break;
        case 4:  d = float3( uv.x, -uv.y,  1.0); break;
        default: d = float3(-uv.x, -uv.y, -1.0); break;
    }
    return normalize(d);
}

// The probe's light from direction dir
float3 SkyProbeRadiance(float3 dir)
{
    if (AtmosphereIdx == 0)
        return 0;
    AtmosphereParams A = LoadAtmosphere();

    // The air's own light: sun, moon and airglow scattered toward the camera (in front of the ground below the horizon)
    float3 sky = AtmosphereSkyLuminance(A, dir) * Exposure;

    float viewHeight = AtmosphereViewHeight(A, CamPos.y);
    float3 P0 = float3(0, viewHeight, 0);
    float tGround = RaySphereIntersectNearest(P0, dir, float3(0, 0, 0), A.BottomRadius);
    if (tGround >= 0.0)
    {
        // The ground: its albedo lit by the sun, the moon and the sky (CSAmbient's ground light, E / π × albedo ×
        // Exposure), seen through the air between (transmittance camera → ground from two LUT reads, along −dir)
        float3 P1 = P0 + dir * tGround;
        float r1 = length(P1);
        float3 T = saturate(AtmTransmittance(A, r1, dot(P1 / r1, -dir))
                          / max(AtmTransmittance(A, viewHeight, -dir.y), 1e-6));
        StructuredBuffer<float4> ambient = ResourceDescriptorHeap[A.Ambient];
        return sky + ambient[1].rgb * T;
    }

    // The cloud layer, blurred two mips more than the dome's (a probe texel spans ~1.4°)
    float density;
    return SkyThroughClouds(linearWrap, dir, sky, 2.0, density);
}

[numthreads(8, 8, 1)]
void CSProbeFaces(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= SizeIdx || id.y >= SizeIdx)
        return;
    float2 uv = (float2(id.xy) + 0.5) / SizeIdx * 2.0 - 1.0;
    RWTexture2DArray<float4> output = ResourceDescriptorHeap[OutputUAVIdx];
    output[id] = float4(SkyProbeRadiance(CubeDirection(id.z, uv)), 1.0);
}

[numthreads(8, 8, 1)]
void CSProbeDownsample(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= SizeIdx || id.y >= SizeIdx)
        return;
    RWTexture2DArray<float4> source = ResourceDescriptorHeap[SourceUAVIdx];
    RWTexture2DArray<float4> output = ResourceDescriptorHeap[OutputUAVIdx];
    uint2 s = id.xy * 2;
    output[id] = (source[uint3(s, id.z)] + source[uint3(s + uint2(1, 0), id.z)]
                + source[uint3(s + uint2(0, 1), id.z)] + source[uint3(s + uint2(1, 1), id.z)]) * 0.25;
}

// Real SH basis to band 2
void SHBasis9(float3 n, out float b[9])
{
    b[0] = 0.282095;
    b[1] = 0.488603 * n.y;
    b[2] = 0.488603 * n.z;
    b[3] = 0.488603 * n.x;
    b[4] = 1.092548 * n.x * n.y;
    b[5] = 1.092548 * n.y * n.z;
    b[6] = 0.315392 * (3.0 * n.z * n.z - 1.0);
    b[7] = 1.092548 * n.x * n.z;
    b[8] = 0.546274 * (n.x * n.x - n.y * n.y);
}

#define SH_THREADS 64
groupshared float3 gSH[SH_THREADS][9];

[numthreads(SH_THREADS, 1, 1)]
void CSProbeSH(uint3 id : SV_DispatchThreadID)
{
    RWTexture2DArray<float4> source = ResourceDescriptorHeap[SourceUAVIdx];
    uint size = SizeIdx;
    uint count = size * size * 6;

    float3 acc[9];
    [unroll] for (uint k = 0; k < 9; k++)
        acc[k] = 0;

    for (uint i = id.x; i < count; i += SH_THREADS)
    {
        uint face = i / (size * size);
        uint2 px = uint2(i % size, (i / size) % size);
        float2 uv = (float2(px) + 0.5) / size * 2.0 - 1.0;
        // Solid angle of the texel: (2 / size)² / (1 + u² + v²)^1.5
        float w = 4.0 / (size * size) * pow(1.0 + dot(uv, uv), -1.5);
        float3 L = source[uint3(px, face)].rgb * w;
        float b[9];
        SHBasis9(CubeDirection(face, uv), b);
        [unroll] for (uint k = 0; k < 9; k++)
            acc[k] += L * b[k];
    }
    [unroll] for (uint k = 0; k < 9; k++)
        gSH[id.x][k] = acc[k];
    GroupMemoryBarrierWithGroupSync();

    [unroll] for (uint stride = SH_THREADS / 2; stride > 0; stride >>= 1)
    {
        if (id.x < stride)
        {
            [unroll] for (uint k = 0; k < 9; k++)
                gSH[id.x][k] += gSH[id.x + stride][k];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (id.x == 0)
    {
        // Irradiance from radiance: band l × Â_l (π, 2π/3, π/4); / π for the ambient as a multiple of albedo
        RWStructuredBuffer<float4> sh = ResourceDescriptorHeap[SHUAVIdx];
        static const float bandScale[9] = { 1.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 0.25, 0.25, 0.25, 0.25, 0.25 };
        [unroll] for (uint k = 0; k < 9; k++)
            sh[k] = float4(gSH[0][k] * bandScale[k], 0);
    }
}
