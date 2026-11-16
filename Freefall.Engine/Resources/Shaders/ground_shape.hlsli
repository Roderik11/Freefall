// ground_shape.hlsli — the shape of the land in the colour of what lies and grows on it. Shared by the terrain
// surface (gputerrain.fx) and the decorators standing on it (grass_compute.hlsl, per instance), so the grass in a
// hollow darkens like the bare ground of the hollow next to it and the pattern runs on past the decoration radius.
// Both read it as the paint's page bake left it (terrain_paint.hlsli PaintShape, baked by terrain_splat_bake.hlsl with
// GroundShape below) and compute it only where there is none: computed per pixel its nine heights 6 and 24 m apart were
// 0.5 ms of a TestWorld view's terrain in scattered reads (2026-11-16).

#ifndef GROUND_SHAPE_HLSLI
#define GROUND_SHAPE_HLSLI

#include "terrain_pages_near.hlsli"   // includes terrain_pages.hlsli; TerrainHeight01Near

// How far the ground at a place lies below (+) or above (-) its surroundings, in metres: the mean height of a ring
// of four points 'radius' samples away minus the height here. Hollows and gullies come out positive, ridges negative.
float GroundCavity(TerrainPagesParams t, float2 worldXZ, float radius)
{
    float d = radius * t.Spacing;
    float ring = TerrainHeight01(t, worldXZ + float2(d, 0))
               + TerrainHeight01(t, worldXZ - float2(d, 0))
               + TerrainHeight01(t, worldXZ + float2(0, d))
               + TerrainHeight01(t, worldXZ - float2(0, d));
    return (ring * 0.25 - TerrainHeight01(t, worldXZ)) * t.MaxHeight;
}

// The ring's four heights for GroundShape (its sum)
float GroundRing(TerrainPagesParams t, TerrainPageCursor c, float2 worldXZ, float radius, SamplerState anySampler)
{
    float d = radius * t.Spacing;
    return TerrainHeight01Near(t, c, (worldXZ + float2(d, 0) - t.Origin.xz) / t.Spacing, anySampler)
         + TerrainHeight01Near(t, c, (worldXZ - float2(d, 0) - t.Origin.xz) / t.Spacing, anySampler)
         + TerrainHeight01Near(t, c, (worldXZ + float2(0, d) - t.Origin.xz) / t.Spacing, anySampler)
         + TerrainHeight01Near(t, c, (worldXZ - float2(0, d) - t.Origin.xz) / t.Spacing, anySampler);
}

// Rings at ~6 m (gullies, crevices in rock) and ~24 m (hollows between hills), with a 1 m sample spacing; about +-1 for
// a metre or two of relief. GroundCavity's sums, the same to the bit, for a fifth of the reads
// (terrain_pages_near.hlsli TerrainHeight01Near): the nine heights share the centre's page-table read where they lie in
// its page, and each is one gather. Any sampler.
float GroundShape(TerrainPagesParams t, float2 worldXZ, SamplerState anySampler)
{
    float2 s = (worldXZ - t.Origin.xz) / t.Spacing;
    TerrainPageCursor c = TerrainCursorAt(t, s);
    float here = TerrainHeight01Near(t, c, s, anySampler);
    return (GroundRing(t, c, worldXZ, 6.0, anySampler) * 0.25 - here) * t.MaxHeight * 0.8
         + (GroundRing(t, c, worldXZ, 24.0, anySampler) * 0.25 - here) * t.MaxHeight * 0.25;
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
