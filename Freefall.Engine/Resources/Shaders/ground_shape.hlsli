// ground_shape.hlsli — the shape of the land in the colour of what lies and grows on it. Shared by the terrain
// surface (gputerrain.fx) and the decorators standing on it (grass_compute.hlsl, per instance), so the grass in a
// hollow darkens like the bare ground of the hollow next to it and the pattern runs on past the decoration radius.

#ifndef GROUND_SHAPE_HLSLI
#define GROUND_SHAPE_HLSLI

// How far the ground at 'uv' lies below (+) or above (-) its surroundings, in metres: the mean height of a ring
// of four points 'radius' texels away minus the height here. Hollows and gullies come out positive, ridges negative.
float GroundCavity(Texture2D heightTex, SamplerState samp, float2 uv, float texel, float maxHeight, float radius)
{
    float d = radius * texel;
    float ring = heightTex.SampleLevel(samp, uv + float2(d, 0), 0).r
               + heightTex.SampleLevel(samp, uv - float2(d, 0), 0).r
               + heightTex.SampleLevel(samp, uv + float2(0, d), 0).r
               + heightTex.SampleLevel(samp, uv - float2(0, d), 0).r;
    return (ring * 0.25 - heightTex.SampleLevel(samp, uv, 0).r) * maxHeight;
}

// Rings at ~6 m (gullies, crevices in rock) and ~24 m (hollows between hills), with a 1 m texel; about +-1 for a
// metre or two of relief
float GroundShape(Texture2D heightTex, SamplerState samp, float2 uv, float texel, float maxHeight)
{
    return GroundCavity(heightTex, samp, uv, texel, maxHeight, 6.0) * 0.8
         + GroundCavity(heightTex, samp, uv, texel, maxHeight, 24.0) * 0.25;
}

// Colour multiplier for a GroundShape value: hollows collect shade, damp and soil and read darker and lusher, ridges
// and crests dry out and catch the light. 'growing' 0 = rock (only darker and lighter), 1 = ground that grows.
float3 GroundShapeTint(float shape, float growing)
{
    float hollow = saturate(shape), crest = saturate(-shape);
    float3 tint = lerp(1.0.xxx, lerp(0.66.xxx, float3(0.62, 0.7, 0.62), growing), hollow);
    tint *= lerp(1.0.xxx, lerp(1.17.xxx, float3(1.16, 1.12, 1.06), growing), crest);
    return tint;
}

// Per-instance ground tint as DecoInstance carries it: three 10-bit channels over 0..2
uint PackGroundTint(float3 tint)
{
    uint3 q = (uint3)round(saturate(tint * 0.5) * 1023.0);
    return q.x | (q.y << 10) | (q.z << 20);
}

float3 UnpackGroundTint(uint packed)
{
    return float3(packed & 1023, (packed >> 10) & 1023, (packed >> 20) & 1023) * (2.0 / 1023.0);
}

#endif // GROUND_SHAPE_HLSLI
