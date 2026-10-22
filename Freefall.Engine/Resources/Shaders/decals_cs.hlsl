// Decals and the weather on the opaque G-buffer (DECALS_PLAN.md Parts A and D, WEATHER_PLAN.md stage 4), by
// Graphics/DecalPass.cs right after the opaque pass, after the cull and sort of decal_cull.hlsl:
//   CSTileDecals  a group per 16 × 16 tile: the tile's depth range and side planes; a bit per visible decal whose box
//                 reaches into them, in the tile's mask (the bit index is the sorted index: walking the bits is walking
//                 the order)
//   CSDecals      a group per tile, a thread per pixel: the surface's G-buffer values into registers, every decal of the
//                 tile's mask applied in order, then the weather (wet surfaces, snow), and one write per target
// The surface's place comes from the hardware depth (the undisplaced surface), its class from EntityId's top two bits.
// Docs: .agent/knowledge/rendering_engine/features/decals.md, atmospheric_effects/weather.md.

#pragma kernel CSTileDecals
#pragma kernel CSDecals

cbuffer PushConstants : register(b3)
{
    uint RecordsIdx;            // StructuredBuffer<DecalRecord>: the registry
    uint TransformsIdx;         // StructuredBuffer<row_major float4x4>: the global transform buffer
    uint FrameIdx;              // StructuredBuffer<DecalFrame>: [0]
    uint SortedIdx;             // StructuredBuffer<uint2>: (registry slot, asuint(fade)) of the visible decals in order
    uint CountUAVIdx;           // RWByteAddressBuffer: [0] = decals in the sorted list
    uint MasksUAVIdx;           // RWByteAddressBuffer: DECAL_MASK_WORDS per tile
    uint MaterialsIdx;          // StructuredBuffer<MaterialData>: the decal materials' settings blocks
    uint DepthTexIdx;           // the hardware depth (reverse Z, 0 = sky)
    uint EntityIdTexIdx;        // the surfaces' class in the top two bits
    uint AlbedoUAVIdx;          // the G-buffer, read and written: RGBA16F
    uint NormalUAVIdx;          // RGBA16 SNorm
    uint DataUAVIdx;            // RGBA8
    uint LinearDepthUAVIdx;     // R32F: the linear depth minus the height bias
    uint HeightUAVIdx;          // RG16F: the screen-space displacement vector
};

#include "common.fx"
#include "decals.hlsli"
#include "decal_settings.hlsli"
#include "weather_cover.hlsli"

SamplerState Sampler : register(s0);           // anisotropic wrap
SamplerState LinearClamp : register(s2);

uint VisibleCount()
{
    RWByteAddressBuffer countBuf = ResourceDescriptorHeap[CountUAVIdx];
    return min(countBuf.Load(0), MAX_VISIBLE_DECALS);
}

// ── Tiles ──

groupshared uint g_MinZ, g_MaxZ;
groupshared float3 g_Planes[4];
groupshared uint g_Mask[DECAL_MASK_WORDS];

[numthreads(DECAL_TILE, DECAL_TILE, 1)]
void CSTileDecals(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID, uint gi : SV_GroupIndex)
{
    StructuredBuffer<DecalFrame> frames = ResourceDescriptorHeap[FrameIdx];
    DecalFrame f = frames[0];
    uint count = VisibleCount();
    uint words = (count + 31u) / 32u;

    if (gi == 0)
    {
        g_MinZ = 0x7F7FFFFFu;
        g_MaxZ = 0u;
    }
    for (uint w = gi; w < DECAL_MASK_WORDS; w += DECAL_TILE * DECAL_TILE)
        g_Mask[w] = 0u;
    // The tile's side planes through the camera: from the rays to its corners on the near plane (reverse Z: depth 1)
    if (gi < 4)
    {
        float2 lo = float2(gid.xy * DECAL_TILE) / ViewportSize;
        float2 hi = min(float2((gid.xy + 1u) * DECAL_TILE), ViewportSize) / ViewportSize;
        float2 corners[4] = { float2(lo.x, lo.y), float2(hi.x, lo.y), float2(hi.x, hi.y), float2(lo.x, hi.y) };
        float3 a = posFromDepth(corners[gi], 1.0, CameraInverse).xyz;
        float3 b = posFromDepth(corners[(gi + 1u) & 3u], 1.0, CameraInverse).xyz;
        float3 centre = posFromDepth(0.5 * (lo + hi), 1.0, CameraInverse).xyz;
        float3 n = normalize(cross(a, b));
        g_Planes[gi] = dot(n, centre) < 0.0 ? -n : n;
    }
    GroupMemoryBarrierWithGroupSync();

    // Its depth range (sky left out)
    uint2 px = gid.xy * DECAL_TILE + gtid.xy;
    if (all(float2(px) < ViewportSize))
    {
        Texture2D<float> depthTex = ResourceDescriptorHeap[DepthTexIdx];
        float depth = depthTex.Load(int3(px, 0));
        if (depth > 0.0)
        {
            float3 rel = posFromDepth((float2(px) + 0.5) / ViewportSize, depth, CameraInverse).xyz;
            uint z = asuint(max(dot(rel, f.CameraForward), 0.0));
            InterlockedMin(g_MinZ, z);
            InterlockedMax(g_MaxZ, z);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // The visible decals whose box reaches into the tile: its four planes and its depth range
    if (g_MaxZ > 0u)
    {
        float minZ = asfloat(g_MinZ), maxZ = asfloat(g_MaxZ);
        float margin = 0.05 + 0.002 * maxZ;
        StructuredBuffer<uint2> sorted = ResourceDescriptorHeap[SortedIdx];
        StructuredBuffer<DecalRecord> records = ResourceDescriptorHeap[RecordsIdx];
        for (uint i = gi; i < count; i += DECAL_TILE * DECAL_TILE)
        {
            float4x4 m = DecalMatrix(TransformsIdx, records[sorted[i].x].TransformSlot);
            float3 c = m[3].xyz - CamPos;
            float3 ax = 0.5 * m[0].xyz, ay = 0.5 * m[1].xyz, az = 0.5 * m[2].xyz;
            bool inside = true;
            [unroll]
            for (uint k = 0; k < 4; k++)
            {
                float3 n = g_Planes[k];
                float reach = abs(dot(n, ax)) + abs(dot(n, ay)) + abs(dot(n, az));
                inside = inside && dot(n, c) >= -reach;
            }
            float zc = dot(c, f.CameraForward);
            float zr = abs(dot(f.CameraForward, ax)) + abs(dot(f.CameraForward, ay)) + abs(dot(f.CameraForward, az));
            inside = inside && zc + zr >= minZ - margin && zc - zr <= maxZ + margin;
            if (inside)
                InterlockedOr(g_Mask[i >> 5], 1u << (i & 31u));
        }
    }
    GroupMemoryBarrierWithGroupSync();

    RWByteAddressBuffer masks = ResourceDescriptorHeap[MasksUAVIdx];
    uint tile = gid.y * f.TilesX + gid.x;
    for (uint w2 = gi; w2 < words; w2 += DECAL_TILE * DECAL_TILE)
        masks.Store((tile * DECAL_MASK_WORDS + w2) * 4u, g_Mask[w2]);
}

// ── Applying them ──

// The surface as the G-buffer holds it, in registers while the decals and the weather change it
struct Surface
{
    float4 Albedo;          // rgb colour, a the emissive mask (kept)
    float4 Normal;          // xyz world normal, w translucency (kept)
    float4 Data;            // roughness, metal, AO, a the lighting model (kept)
    float Depth;            // linear depth minus the height bias
    float2 Displacement;    // the screen-space displacement vector
};

// Camera-relative place of the surface at a pixel (the hardware depth; 0 = sky)
float3 SurfaceAt(Texture2D<float> depthTex, int2 px, out float depth)
{
    depth = depthTex.Load(int3(px, 0));
    return posFromDepth((float2(px) + 0.5) / ViewportSize, depth, CameraInverse).xyz;
}

// A lattice point's random value (0..1) by an integer hash (PCG's output step), with a seed for separate streams. No
// trig: the usual frac(sin(dot(p, k)) × big) loses its float precision at Eldermoor's coordinates (thousands of
// metres × the noise's scale), and the snow's patches came out as streaks along one axis (user, 2026-10-22).
float HashCell(int2 cell, uint seed)
{
    uint h = ((uint)cell.x * 0x8DA6B343u) ^ ((uint)cell.y * 0xD8163841u) ^ (seed * 0x9E3779B9u);
    h = h * 747796405u + 2891336453u;
    h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
    h = (h >> 22u) ^ h;
    return (h & 0xFFFFFFu) / 16777216.0;
}

// Rain striking a puddle: expanding rings in cells of 0.2 m, a drop a cell at a time, three layers out of step. The
// tilt of the water's surface in the decal's plane.
float2 RippleTilt(float2 p, float rate)
{
    float2 tilt = 0;
    [unroll]
    for (uint layer = 0; layer < 3; layer++)
    {
        float2 q = p * 5.0 + float2(layer * 0.37, layer * 0.71);
        int2 cell = (int2)floor(q);
        float h = HashCell(cell, 1u + layer);
        // Each cell strikes about once a second in heavy rain, less often in lighter
        float t = Time * (0.5 + 0.7 * rate) + h * 7.0;
        float dropTime = floor(t);
        float age = t - dropTime;
        uint drop = (uint)(int)dropTime * 4u + layer * 0x10000u;
        if (HashCell(cell, drop + 11u) > saturate(rate * 1.2))
            continue;
        float2 centre = 0.25 + 0.5 * float2(HashCell(cell, drop + 12u), HashCell(cell, drop + 13u));
        float2 d = frac(q) - centre;
        float r = length(d);
        float x = (r - age * 0.45) / 0.05;
        // A ring travelling out and fading: the slope of a damped wave across it
        float slope = cos(x * 3.14159) * exp(-x * x) * (1.0 - age) * (1.0 - age);
        tilt += d / max(r, 1e-3) * slope;
    }
    return tilt * 0.35;
}

// One decal at this pixel: into its box, its receivers, its fades, its material, then its modes on the surface
void ApplyDecal(uint i, float3 rel, float3 geo, float3 dx, float3 dy, uint surfaceClass, float rain, float puddleFill,
    inout Surface s)
{
    StructuredBuffer<uint2> sorted = ResourceDescriptorHeap[SortedIdx];
    uint2 entry = sorted[i];
    StructuredBuffer<DecalRecord> records = ResourceDescriptorHeap[RecordsIdx];
    DecalRecord r = records[entry.x];
    float4x4 m = DecalMatrix(TransformsIdx, r.TransformSlot);

    float3 d = rel - (m[3].xyz - CamPos);
    float3 lx = m[0].xyz, ly = m[1].xyz, lz = m[2].xyz;
    float3 invLen2 = 1.0 / float3(dot(lx, lx), dot(ly, ly), dot(lz, lz));
    float3 local = float3(dot(d, lx), dot(d, ly), dot(d, lz)) * invLen2;
    if (any(abs(local) > 0.5))
        return;

    Settings S = LoadDecalSettings(MaterialsIdx, r.MaterialId);
    if (((S.Receivers >> surfaceClass) & 1u) == 0u)
        return;

    float3 axis = normalize(ly);
    float3 tangent = normalize(lx), bitangent = -normalize(lz);  // the texture's u and v across the surface

    // Fades: by the angle between the surface and the axis, toward the box's top and bottom, with distance (the cull)
    float cosStart = cos(radians(S.AngleFade.x)), cosEnd = cos(radians(S.AngleFade.y));
    float fade = saturate((dot(geo, axis) - cosEnd) / max(cosStart - cosEnd, 1e-3));
    if (S.EdgeFade > 0.0)
        fade *= saturate((0.5 - abs(local.y)) / S.EdgeFade);
    fade *= asfloat(entry.y) * r.Opacity;
    if (fade <= 0.0)
        return;

    // The texture coordinates and their steps to the neighbouring pixels, from the neighbours' places
    uint cols = max(S.AtlasColumns, 1u), rows = max(S.AtlasRows, 1u);
    uint tile = r.Variant % (cols * rows);
    float2 tileSize = 1.0 / float2(cols, rows);
    float2 uv = (float2(tile % cols, tile / cols) + float2(local.x + 0.5, 0.5 - local.z)) * tileSize;
    float2 uvDx = float2(dot(dx, lx) * invLen2.x, -dot(dx, lz) * invLen2.z) * tileSize;
    float2 uvDy = float2(dot(dy, lx) * invLen2.x, -dot(dy, lz) * invLen2.z) * tileSize;

    Texture2D albedoTex = ResourceDescriptorHeap[NonUniformResourceIndex(S.Albedo)];
    float4 albedo = albedoTex.SampleGrad(Sampler, uv, uvDx, uvDy);
    float3 colour = albedo.rgb * S.Color * r.Tint;

    // The opacity of colour and roughness, and that of normal and relief: one for plain decals; for a puddle the water
    // (all of them) and a damp edge round it (colour and roughness only)
    float paint = albedo.a, water = albedo.a;
    if (S.RainFilled)
    {
        float level = 1.0 - saturate(puddleFill);
        water = smoothstep(level, level + 0.08, albedo.a) * step(0.001, puddleFill);
        paint = max(water, 0.6 * smoothstep(level - 0.25, level, albedo.a) * step(0.001, puddleFill));
    }
    float aColour = paint * fade, aShape = water * fade;
    if (aColour <= 0.0 && aShape <= 0.0)
        return;

    // Colour
    float aA = aColour * S.AlbedoOpacity;
    if (S.AlbedoMode == 1u)
        s.Albedo.rgb *= lerp(1.0, colour, aA);
    else
        s.Albedo.rgb = lerp(s.Albedo.rgb, colour, aA);

    // Normal (the map's y flipped as gbuffer.fx does: OpenGL-convention maps), the rain's rings on a puddle's water
    float aN = aShape * S.NormalOpacity;
    if (aN > 0.0)
    {
        Texture2D normalTex = ResourceDescriptorHeap[NonUniformResourceIndex(S.Normal)];
        float2 nXY = (normalTex.SampleGrad(Sampler, uv, uvDx, uvDy).rg * 2.0 - 1.0) * S.NormalStrength;
        nXY.y = -nXY.y;
        if (S.Ripples > 0.0 && rain > 0.0)
        {
            float3 world = rel + CamPos;
            nXY += RippleTilt(float2(dot(world, tangent), dot(world, bitangent)), rain) * S.Ripples * water;
        }
        float3 n = normalize(tangent * nXY.x + bitangent * nXY.y + axis * sqrt(max(0.001, 1.0 - saturate(dot(nXY, nXY)))));
        s.Normal.xyz = S.NormalMode == 0u ? normalize(lerp(s.Normal.xyz, n, aN)) : normalize(s.Normal.xyz + (n - axis) * aN);
    }

    // Roughness, metal, AO
    float aM = aColour * S.MaskOpacity;
    if (aM > 0.0)
    {
        Texture2D maskTex = ResourceDescriptorHeap[NonUniformResourceIndex(S.Mask)];
        float3 mask = maskTex.SampleGrad(Sampler, uv, uvDx, uvDy).rgb;
        s.Data.rgb = lerp(s.Data.rgb, float3(mask.r * S.Roughness, mask.g * S.Metal, mask.b), aM);
    }

    // Relief for the screen-space displacement: as gbuffer.fx writes it, along the axis out of the surface
    float aH = aShape * S.HeightOpacity;
    if (aH > 0.0)
    {
        Texture2D heightTex = ResourceDescriptorHeap[NonUniformResourceIndex(S.Height)];
        float h = lerp(S.HeightMin, S.HeightMax, heightTex.SampleGrad(Sampler, uv, uvDx, uvDy).r);
        float4 clipPos = mul(float4(rel, 1.0), CameraRelativeVP);
        float4 clipDisp = mul(float4(rel + axis * h, 1.0), CameraRelativeVP);
        float2 ndcPos = clipPos.xy / clipPos.w;
        float2 displacement = (clipDisp.xy / clipDisp.w - ndcPos) * float2(0.5, -0.5);
        // Faded at the screen's edges, as gbuffer.fx: the inversion never needs off-screen sources
        float2 screenUV = ndcPos * float2(0.5, -0.5) + 0.5;
        displacement *= saturate(min(min(screenUV.x, 1.0 - screenUV.x), min(screenUV.y, 1.0 - screenUV.y))
                                 / max(length(displacement), 1e-6));
        if (S.HeightMode == 1u)
        {
            s.Displacement = lerp(s.Displacement, displacement, aH);
            s.Depth = lerp(s.Depth, clipPos.w - h, aH);
        }
        else
        {
            s.Displacement += displacement * aH;
            s.Depth -= h * aH;
        }
    }
}

// Value noise for the snow's patches while it is thin (0..1)
float SnowNoise1(float2 p, uint seed)
{
    float2 i = floor(p), f = p - i;
    int2 c = (int2)i;
    float2 u = f * f * (3.0 - 2.0 * f);
    return lerp(lerp(HashCell(c, seed), HashCell(c + int2(1, 0), seed), u.x),
                lerp(HashCell(c + int2(0, 1), seed), HashCell(c + int2(1, 1), seed), u.x), u.y);
}
float SnowNoise(float2 xz) { return 0.65 * SnowNoise1(xz * 0.8, 101u) + 0.35 * SnowNoise1(xz * 3.1, 202u); }

#define SNOW_ALBEDO float3(0.70, 0.72, 0.76)   // packed old snow, a little blue: the dark palette's near white
#define SNOW_ROUGHNESS 0.8
#define SNOW_RELIEF_TERRAIN 0.035                // m of relief that counts as a high spot: the terrain's layers (up to 0.05)
#define SNOW_RELIEF_MESH 0.008                   // ... a mesh's height map (gbuffer.fx: ~0.01 m at a tile a metre)

// The weather on an open surface: wet (porous surfaces darken, smooth ones and flat open ground take a water film) and
// snow (on what faces up, patchy while it is thin, smoothing the relief)
void ApplyWeather(DecalFrame f, float3 rel, float3 geo, uint surfaceClass, inout Surface s)
{
    // Emissive pixels (lamp glass, windows) hold their light in the albedo: left as they are
    if (s.Albedo.a > 0.01)
        return;
    WeatherCover c;
    c.ShelterVP = f.ShelterVP;
    c.ShelterDepth = f.ShelterDepth;
    c.ShelterSlice = f.ShelterSlice;
    c.TerrainHeightmap = f.TerrainHeightmap;
    c.TerrainMaxHeight = f.TerrainMaxHeight;
    c.TerrainOrigin = f.TerrainOrigin;
    c.TerrainSize = f.TerrainSize;
    // How much of the fall reaches it (straight down, soft under eaves), looked for a little out from the surface: a wall
    // in the open is not covered by its own top, a floor under a roof is
    float3 world = rel + CamPos;
    float open = WeatherOpenness(c, LinearClamp, world + geo * 0.35 + float3(0, 0.05, 0), 0.1);
    if (open <= 0.0)
        return;

    float3 N = normalize(s.Normal.xyz);
    float vegetation = s.Data.a < 0.75 ? 1.0 : 0.0;

    // Snow: thin, it lies on the surface's high spots first (the stones' tops, pebbles, clods: their relief, which the
    // linear depth holds as its height bias, gbuffer.fx and gputerrain.fx), varied by soft drifts metres across; as it
    // deepens it fills down into the joints and hollows and buries them. Only what faces up; not on characters (they
    // move). A noise alone made random patches (user, 2026-10-22: "shouldnt it accumulate on the height spots first")
    float snow = 0.0;
    float cover = saturate(f.SnowCover) * open;      // less falls where it is partly sheltered
    if (cover > 0.0 && surfaceClass != 3u)
    {
        float up = smoothstep(0.35, 0.85, lerp(N.y, geo.y, cover)) * smoothstep(0.2, 0.6, geo.y);
        // How far this pixel stands up from its surface (m): the terrain's layers up to 5 cm, a mesh's ~1 cm
        float relief = saturate(max(dot(rel, f.CameraForward) - s.Depth, 0.0) / (surfaceClass == 1u ? SNOW_RELIEF_TERRAIN : SNOW_RELIEF_MESH));
        float drift = SnowNoise(world.xz * 0.35);
        float t = 1.0 - cover;
        snow = smoothstep(t - 0.06, t + 0.06, up * (0.1 + 0.5 * relief + 0.4 * drift)) * saturate(up * 3.0);
    }

    float facing = geo.y >= 0.0 ? lerp(0.35, 1.0, geo.y) : saturate(0.35 + geo.y);
    float wet = saturate(f.Wetness) * open * facing * (surfaceClass == 3u ? 0.5 : 1.0) * (1.0 - snow);
    if (wet > 0.0)
    {
        float rough = s.Data.r, metal = s.Data.g;
        float porosity = saturate((rough - 0.25) / 0.6) * (1.0 - metal) * (vegetation > 0.0 ? 0.5 : 1.0);
        s.Albedo.rgb *= lerp(1.0, 0.5, porosity * wet);
        // What the wet surface's roughness tends to: a water film on smooth ones, less on porous ones (the water sinks
        // in); flat open ground gathers a film first once it is soaked
        float film = lerp(0.07, rough * 0.55 + 0.05, porosity);
        film = lerp(film, 0.06, saturate((geo.y - 0.9) * 10.0) * saturate((wet - 0.6) / 0.4));
        s.Data.r = lerp(rough, min(rough, film), wet);
    }

    if (snow > 0.0)
    {
        // A thin dusting lets the ground show through; deep snow is its own surface, smooth over the stones' relief
        // (their normals, AO and displacement buried under it). Thin, it lies on the relief and keeps it.
        float dust = snow * lerp(0.45, 1.0, smoothstep(0.0, 0.6, cover));
        float buried = snow * smoothstep(0.3, 0.9, cover);
        s.Albedo.rgb = lerp(s.Albedo.rgb, SNOW_ALBEDO, dust);
        s.Data.rgb = lerp(s.Data.rgb, float3(SNOW_ROUGHNESS, 0.0, lerp(s.Data.b, 1.0, buried)), dust);
        s.Normal.xyz = normalize(lerp(N, normalize(geo + float3(0, 1, 0)), buried));
        s.Displacement *= 1.0 - buried;
    }
}

groupshared uint g_TileMask[DECAL_MASK_WORDS];
groupshared uint g_TileAny;

[numthreads(DECAL_TILE, DECAL_TILE, 1)]
void CSDecals(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID, uint gi : SV_GroupIndex)
{
    StructuredBuffer<DecalFrame> frames = ResourceDescriptorHeap[FrameIdx];
    DecalFrame f = frames[0];
    uint count = VisibleCount();
    uint words = (count + 31u) / 32u;

    if (gi == 0)
        g_TileAny = 0u;
    GroupMemoryBarrierWithGroupSync();
    RWByteAddressBuffer masks = ResourceDescriptorHeap[MasksUAVIdx];
    uint tile = gid.y * f.TilesX + gid.x;
    for (uint w = gi; w < words; w += DECAL_TILE * DECAL_TILE)
    {
        uint bits = masks.Load((tile * DECAL_MASK_WORDS + w) * 4u);
        g_TileMask[w] = bits;
        if (bits != 0u)
            InterlockedOr(g_TileAny, 1u);
    }
    GroupMemoryBarrierWithGroupSync();

    if (g_TileAny == 0u && f.WeatherOn == 0u)
        return;
    int2 px = int2(gid.xy * DECAL_TILE + gtid.xy);
    if (any(float2(px) >= ViewportSize))
        return;

    Texture2D<float> depthTex = ResourceDescriptorHeap[DepthTexIdx];
    float depth;
    float3 rel = SurfaceAt(depthTex, px, depth);
    if (depth <= 0.0)
        return;                                         // sky

    // The geometric normal and the steps to the neighbours, from the depth: of the two neighbours on each axis the one
    // whose step is smaller (an edge of the surface lies toward the other)
    float dl, dr, du, dd;
    float3 pl = SurfaceAt(depthTex, px + int2(-1, 0), dl), pr = SurfaceAt(depthTex, px + int2(1, 0), dr);
    float3 pu = SurfaceAt(depthTex, px + int2(0, -1), du), pd = SurfaceAt(depthTex, px + int2(0, 1), dd);
    float3 dx = abs(dr - depth) < abs(depth - dl) ? pr - rel : rel - pl;
    float3 dy = abs(dd - depth) < abs(depth - du) ? pd - rel : rel - pu;
    float3 geo = normalize(cross(dy, dx));
    if (dot(geo, -rel) < 0.0)
        geo = -geo;

    Texture2D<uint> ids = ResourceDescriptorHeap[EntityIdTexIdx];
    uint surfaceClass = ids.Load(int3(px, 0)) >> 30;

    RWTexture2D<float4> albedoUav = ResourceDescriptorHeap[AlbedoUAVIdx];
    RWTexture2D<float4> normalUav = ResourceDescriptorHeap[NormalUAVIdx];
    RWTexture2D<float4> dataUav = ResourceDescriptorHeap[DataUAVIdx];
    RWTexture2D<float> depthUav = ResourceDescriptorHeap[LinearDepthUAVIdx];
    RWTexture2D<float2> heightUav = ResourceDescriptorHeap[HeightUAVIdx];
    Surface s;
    s.Albedo = albedoUav[px];
    s.Normal = normalUav[px];
    s.Data = dataUav[px];
    s.Depth = depthUav[px];
    s.Displacement = heightUav[px];
    Surface before = s;

    if (g_TileAny != 0u)
    {
        for (uint w2 = 0; w2 < words; w2++)
        {
            uint bits = g_TileMask[w2];
            while (bits != 0u)
            {
                uint b = firstbitlow(bits);
                bits &= bits - 1u;
                ApplyDecal(w2 * 32u + b, rel, geo, dx, dy, surfaceClass, f.Rain, f.PuddleFill, s);
            }
        }
    }
    if (f.WeatherOn != 0u)
        ApplyWeather(f, rel, geo, surfaceClass, s);

    if (any(s.Albedo != before.Albedo))
        albedoUav[px] = s.Albedo;
    if (any(s.Normal != before.Normal))
        normalUav[px] = s.Normal;
    if (any(s.Data != before.Data))
        dataUav[px] = s.Data;
    if (s.Depth != before.Depth)
        depthUav[px] = s.Depth;
    if (any(s.Displacement != before.Displacement))
        heightUav[px] = s.Displacement;
}
