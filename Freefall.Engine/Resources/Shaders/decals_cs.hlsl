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
#include "atmosphere_common.hlsli"
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
    float Dent;             // metres the decals sank it (puddles' hollows): the weather's relief is measured from there
    float Print;            // footprints here (Print decals), 0..1 after the snow that has filled them again
    float PrintDepth;       // metres the deepest of them sinks
    float3 PrintTilt;       // their slope, the tilt toward their floors (the weather applies it as the ground allows)
};

// Metres of relief that count as a surface's high spot: the terrain's layers (gputerrain.fx, up to 0.05), a mesh's
// height map (gbuffer.fx: ~0.01 m at a tile a metre)
float ReliefScale(uint surfaceClass) { return surfaceClass == 1u ? 0.035 : 0.008; }


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

// The screen-space displacement of the surface at `rel` raised h metres along `axis`, as gbuffer.fx writes it, faded at
// the screen's edges (the inversion never needs off-screen sources); viewZ its undisplaced view depth
float2 DisplacementFor(float3 rel, float3 axis, float h, out float viewZ)
{
    float4 clipPos = mul(float4(rel, 1.0), CameraRelativeVP);
    float4 clipDisp = mul(float4(rel + axis * h, 1.0), CameraRelativeVP);
    viewZ = clipPos.w;
    float2 ndcPos = clipPos.xy / clipPos.w;
    float2 displacement = (clipDisp.xy / clipDisp.w - ndcPos) * float2(0.5, -0.5);
    float2 screenUV = ndcPos * float2(0.5, -0.5) + 0.5;
    return displacement * saturate(min(min(screenUV.x, 1.0 - screenUV.x), min(screenUV.y, 1.0 - screenUV.y))
                                   / max(length(displacement), 1e-6));
}

// One decal at this pixel: into its box, its receivers, its fades, its material, then its modes on the surface
void ApplyDecal(uint i, float3 rel, float3 geo, float3 dx, float3 dy, uint surfaceClass, float rain, float puddleFill,
    float snowFallen, inout Surface s)
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

    // A footprint: its depth from the opacity (a mip as the screen needs, at least the second: the tread's grain is no
    // depth), its slope the tilt toward its floor; filled in again by the snow fallen since it was made. What it does
    // is the ground's to say, so it only leaves itself on the surface here (ApplyWeather: Prints)
    if (S.Print)
    {
        uint tw, th;
        albedoTex.GetDimensions(tw, th);
        float2 size = float2(tw, th);
        float mip = max(log2(max(max(length(uvDx * size), length(uvDy * size)), 1e-6)), 1.0);
        float depth = albedoTex.SampleLevel(Sampler, uv, mip).a;
        float2 e = 1.5 * exp2(mip) / size;
        float2 grad = float2(albedoTex.SampleLevel(Sampler, uv + float2(e.x, 0), mip).a - albedoTex.SampleLevel(Sampler, uv - float2(e.x, 0), mip).a,
                             albedoTex.SampleLevel(Sampler, uv + float2(0, e.y), mip).a - albedoTex.SampleLevel(Sampler, uv - float2(0, e.y), mip).a);
        float2 pslope = grad / (2.0 * e) * tileSize * sqrt(float2(invLen2.x, invLen2.z));
        float refill = (r.Flags & DECAL_SPAWNED) != 0 ? saturate((snowFallen - r.SpawnSnow) / max(S.PrintFill, 1e-3)) : 0.0;
        float strength = fade * (1.0 - refill);
        s.Print = max(s.Print, depth * strength);
        s.PrintDepth = max(s.PrintDepth, S.PrintDepth * depth * strength);
        s.PrintTilt += (tangent * pslope.x + bitangent * pslope.y) * (S.PrintDepth * S.NormalStrength * strength);
        return;
    }

    float4 albedo = albedoTex.SampleGrad(Sampler, uv, uvDx, uvDy);
    float3 colour = albedo.rgb * S.Color * r.Tint;

    // The opacity of colour and roughness, and that of normal and relief: one for plain decals; for a puddle the water
    // (all of them) and the damp (colour and roughness only), and its hollow and silt, there when dry too
    float paint = albedo.a, water = albedo.a;
    float body = 0.0, bowl = 0.0, level = 1.0, silt = 0.0, rim = 0.0;
    float2 slope = 0;
    if (S.RainFilled)
    {
        // The puddle's shape from the opacity blurred (mip 3 or coarser, as the screen needs): its specks are damp spots
        // round it, not water, and once the puddle was nearly full they speckled its edge with drops. The same depth
        // sinks the ground (DentDepth), and its slope per metre across the surface tilts the stones toward the middle
        uint tw, th;
        albedoTex.GetDimensions(tw, th);
        float2 size = float2(tw, th);
        float mip = max(log2(max(max(length(uvDx * size), length(uvDy * size)), 1e-6)) + 1.0, 3.0);
        // (faded out over the box's sides: blurred, the specks leave the depth above 0 out to the tile's edge, and the
        // box cut the silt off in a straight line)
        float sides = saturate((0.5 - max(abs(local.x), abs(local.z))) / 0.06);
        body = albedoTex.SampleLevel(Sampler, uv, mip).a * sides;
        // The hollow itself is a wide bowl, the depth two mips blurrier: from the outline's depth (flat middles, short
        // edges) its slope was a thin ring round the water and the hollow did not read
        float bowlMip = mip + 2.0;
        bowl = albedoTex.SampleLevel(Sampler, uv, bowlMip).a * sides;
        float2 e = 1.5 * exp2(bowlMip) / size;
        float2 grad = float2(albedoTex.SampleLevel(Sampler, uv + float2(e.x, 0), bowlMip).a - albedoTex.SampleLevel(Sampler, uv - float2(e.x, 0), bowlMip).a,
                             albedoTex.SampleLevel(Sampler, uv + float2(0, e.y), bowlMip).a - albedoTex.SampleLevel(Sampler, uv - float2(0, e.y), bowlMip).a);
        slope = grad / (2.0 * e) * tileSize * sqrt(float2(invLen2.x, invLen2.z)) * sides;
        // Rain makes it damp first, the damp spreading out from the middle; water once the rain's puddles are a quarter
        // full, its surface level with the shoreline. Drying runs back: the water goes, then the damp shrinks to the
        // middle, a stain that outlasts the water (the fill drains 4× slower than the ground dries: Weather.PuddleFill)
        float fill = saturate(puddleFill);
        float waterFill = saturate((fill - 0.25) / 0.75);
        level = 1.0 - waterFill;
        water = smoothstep(level, level + 0.08, body) * step(0.001, waterFill);
        float dampLevel = 1.0 - min(fill * 1.6, 1.0);
        paint = max(water, 0.6 * smoothstep(dampLevel, dampLevel + 0.15, albedo.a) * step(0.001, fill));
        // Dry, what the water left in the hollow: silt in the stones' joints, a dusting on their tops, and a darker line
        // where the water stood
        float relief = saturate((mul(float4(rel, 1.0), CameraRelativeVP).w - s.Depth) / ReliefScale(surfaceClass));
        silt = S.SiltOpacity * smoothstep(0.04, 0.35, bowl) * lerp(1.0, 0.5, relief);
        rim = smoothstep(0.1, 0.17, body) * (1.0 - smoothstep(0.17, 0.3, body)) * saturate(S.SiltOpacity * 2.0);
    }
    float aColour = paint * fade, aShape = water * fade;
    float aDent = S.RainFilled && S.DentDepth > 0.0 ? fade * S.HeightOpacity : 0.0;
    float aSilt = silt * fade;
    if (aColour <= 0.0 && aShape <= 0.0 && aDent <= 0.0 && aSilt <= 0.0 && rim <= 0.0)
        return;

    // Colour: a puddle's silt multiplies the ground by SiltColor (a set colour was a grey haze on bright sandstone and
    // would vanish on dark cobbles), and the water line darkens it; then the decal's colour (its water and damp darken
    // the silt too)
    if (aSilt > 0.0 || rim > 0.0)
        s.Albedo.rgb = lerp(s.Albedo.rgb, s.Albedo.rgb * S.SiltColor, aSilt) * (1.0 - 0.15 * rim * fade);
    float aA = aColour * S.AlbedoOpacity;
    if (S.AlbedoMode == 1u)
        s.Albedo.rgb *= lerp(1.0, colour, aA);
    else
        s.Albedo.rgb = lerp(s.Albedo.rgb, colour, aA);

    // A puddle's hollow: the silt evens out the stones in it, its slope tilts them toward the middle (a height field's
    // normal, axis + DentDepth × the depth's slope along the tangent and bitangent, steepened by NormalStrength)
    if (aDent > 0.0 || aSilt > 0.0)
    {
        float3 n0 = normalize(lerp(s.Normal.xyz, axis, aSilt * S.SiltFlatten));
        s.Normal.xyz = normalize(n0 + (tangent * slope.x + bitangent * slope.y) * (S.DentDepth * S.NormalStrength * aDent));
    }

    // Normal (the map's y flipped as gbuffer.fx does: OpenGL-convention maps), the rain's rings on a puddle's water
    float aN = aShape * S.NormalOpacity;
    if (aN > 0.0)
    {
        Texture2D normalTex = ResourceDescriptorHeap[NonUniformResourceIndex(S.Normal)];
        float2 nXY = (normalTex.SampleGrad(Sampler, uv, uvDx, uvDy).rg * 2.0 - 1.0) * S.NormalStrength;
        nXY.y = -nXY.y;
        // The rings fade out as a pixel grows past them (~2 cm): smaller, they lit single pixels, a sparkle far off
        float ripples = S.Ripples * (1.0 - smoothstep(0.01, 0.03, max(length(dx), length(dy))));
        if (ripples > 0.0 && rain > 0.0)
        {
            float3 world = rel + CamPos;
            nXY += RippleTilt(float2(dot(world, tangent), dot(world, bitangent)), rain) * ripples * water;
        }
        float3 n = normalize(tangent * nXY.x + bitangent * nXY.y + axis * sqrt(max(0.001, 1.0 - saturate(dot(nXY, nXY)))));
        s.Normal.xyz = S.NormalMode == 0u ? normalize(lerp(s.Normal.xyz, n, aN)) : normalize(s.Normal.xyz + (n - axis) * aN);
    }

    // Roughness, metal, AO (a puddle's dry silt first)
    if (aSilt > 0.0)
        s.Data.r = lerp(s.Data.r, S.SiltRoughness, aSilt);
    float aM = aColour * S.MaskOpacity;
    if (aM > 0.0)
    {
        Texture2D maskTex = ResourceDescriptorHeap[NonUniformResourceIndex(S.Mask)];
        float3 mask = maskTex.SampleGrad(Sampler, uv, uvDx, uvDy).rgb;
        s.Data.rgb = lerp(s.Data.rgb, float3(mask.r * S.Roughness, mask.g * S.Metal, mask.b), aM);
    }

    // Relief for the screen-space displacement: as gbuffer.fx writes it, along the axis out of the surface
    if (S.RainFilled)
    {
        // A puddle: the silt fills the stones' joints part of the way up (SiltFlatten × a high spot's relief), the
        // hollow sinks it all by the bowl × DentDepth; the water's surface flat, level with the shoreline (about the
        // depth at the water's edge)
        float viewZ;
        float hDent = -S.DentDepth * bowl;
        float2 dent = DisplacementFor(rel, axis, hDent, viewZ);
        if (aSilt > 0.0)
            s.Depth = lerp(s.Depth, min(s.Depth, viewZ - S.SiltFlatten * ReliefScale(surfaceClass)), saturate(aSilt * 2.0));
        if (aDent > 0.0)
        {
            s.Displacement += dent * aDent;
            s.Depth -= hDent * aDent;
            s.Dent -= hDent * aDent;
        }
        float aW = aShape * S.HeightOpacity;
        if (aW > 0.0)
        {
            float hWater = -S.DentDepth * level;
            float2 surface = DisplacementFor(rel, axis, hWater, viewZ);
            s.Displacement = lerp(s.Displacement, surface, aW);
            s.Depth = lerp(s.Depth, viewZ - hWater, aW);
        }
        return;
    }
    float aH = aShape * S.HeightOpacity;
    if (aH > 0.0)
    {
        Texture2D heightTex = ResourceDescriptorHeap[NonUniformResourceIndex(S.Height)];
        float h = lerp(S.HeightMin, S.HeightMax, heightTex.SampleGrad(Sampler, uv, uvDx, uvDy).r), viewZ;
        float2 displacement = DisplacementFor(rel, axis, h, viewZ);
        if (S.HeightMode == 1u)
        {
            s.Displacement = lerp(s.Displacement, displacement, aH);
            s.Depth = lerp(s.Depth, viewZ - h, aH);
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

// Value noise on a unit lattice (0..1) with its gradient, quintic fade (its slope is smooth too: lit normals show no
// lattice): x the value, yz d/dx, d/dy
float3 NoiseGrad(float2 p, uint seed)
{
    float2 i = floor(p), f = p - i;
    int2 c = (int2)i;
    float a = HashCell(c, seed), b = HashCell(c + int2(1, 0), seed);
    float d = HashCell(c + int2(0, 1), seed), e = HashCell(c + int2(1, 1), seed);
    float2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    float2 du = 30.0 * f * f * (f * (f - 2.0) + 1.0);
    float k = a - b - d + e;
    return float3(a + (b - a) * u.x + (d - a) * u.y + k * u.x * u.y,
                  du * float2(b - a + k * u.y, d - a + k * u.x));
}

// Deep snow's own surface over its large-scale normal `n` (world place, metres a pixel spans): a height field's slope
// tilting it, world-anchored so nothing swims. Drifts 20-100 cm high, 2-7 m across (value noise's typical slope is a
// third of its peak: lower ones did not read at all); wind ripples 1.5 cm high, 30 cm apart, in patches, their heading
// wandering slowly over the world (not today's wind, which would turn them); a fine grain,
// gone where a pixel outgrows it. Full snow cover had no shape at all, one flat white (user, 2026-10-23). `glint`: a
// crystal here (round, 1 in 25 cells of 6 mm, only while a pixel is smaller than a cell, so they cannot turn to noise
// far off) with its facet half-way between the light `toLight` and the eye `toEye`: a crystal flashes when it lines up
// so, and only ever brightens. Each with its own random tilt drew dark specks, most of them facing away from the sun.
// The caller makes it glossy
float3 SnowSurfaceNormal(float3 world, float3 n, float footprint, float3 toLight, float3 toEye, out float glint)
{
    float2 xz = world.xz;
    float2 slope = 0;
    float3 drift = NoiseGrad(xz / 7.0, 301u);
    slope += drift.yz * (1.0 / 7.0);
    float3 drift2 = NoiseGrad(xz / 2.3, 302u);
    slope += drift2.yz * (0.2 / 2.3);

    // Ripples: along their heading short (crests across it), across it longer, in patches. One prevailing heading that
    // wanders ±17° over 400 m, a 2.5 : 1 stretch and the lattice warped: a full turn every 90 m and 5.6 : 1 drew
    // straight parallel streaks (value noise's lattice rows) bending into swirls (user, 2026-10-23)
    float heading = 0.6 + (NoiseGrad(xz / 400.0, 303u).x - 0.5) * 0.6;
    float2 along = float2(cos(heading), sin(heading)), across = float2(-along.y, along.x);
    float2 q = float2(dot(xz, along) / 0.3, dot(xz, across) / 0.75);
    q += (float2(NoiseGrad(q * 0.37, 306u).x, NoiseGrad(q * 0.37 + 17.3, 307u).x) - 0.5) * 1.6;
    float3 ripple = NoiseGrad(q, 304u);
    float patches = smoothstep(0.35, 0.75, NoiseGrad(xz / 9.0, 308u).x);
    float rippleFade = (1.0 - smoothstep(0.02, 0.06, footprint)) * patches;
    slope += (along * (ripple.y / 0.3) + across * (ripple.z / 0.75)) * 0.015 * rippleFade;

    float3 grain = NoiseGrad(xz / 0.035, 305u);
    slope += grain.yz * (0.003 / 0.035) * (1.0 - smoothstep(0.003, 0.009, footprint));

    // The height field's normal on the large-scale surface (gentle slopes: the tilt added in the ground's plane)
    float3 sn = normalize(n - float3(slope.x, 0.0, slope.y));

    // Glints
    glint = 0.0;
    const float cell = 0.006;
    float glintFade = 1.0 - smoothstep(cell * 0.4, cell * 1.2, footprint);
    if (glintFade > 0.0)
    {
        int2 c = (int2)floor(xz / cell);
        if (HashCell(c, 401u) < 0.04 && toLight.y > 0.0)
        {
            float2 centre = 0.25 + 0.5 * float2(HashCell(c, 402u), HashCell(c, 403u));
            float2 d = frac(xz / cell) - centre;
            if (dot(d, d) < 0.1)
            {
                sn = normalize(toLight + toEye);
                glint = glintFade;
            }
        }
    }
    return sn;
}

// How much of the falling rain and snow reaches a surface (straight down, soft under eaves), looked for a little out
// from it: a wall in the open is not covered by its own top, a floor under a roof is. Vegetation takes the terrain's
// cover and how deep in its crown it lies (CrownExposure), not the crown's cover outright: that counted near the camera,
// where the cover map resolves a crown over its own lower branches, and not far off, where its coarse levels do not, so
// near trees stayed green and far ones white. A far tree's cross-mesh card brings its own exposure in the vegetation
// flag (common.fx FarTreeExposure): the map holds no top for vertical cards, so they came out white where it had none
// and in green stripes where a card's edge had left one (user, 2026-10-23: "the snow on crossmeshes looks bad")
float SurfaceOpenness(DecalFrame f, float3 rel, float3 geo, uint surfaceClass, float vegetationFlag)
{
    WeatherShelter s;
    s.Levels = f.Shelter;
    s.Eye = f.ShelterEye;
    s.MinY = f.ShelterMinY;
    s.RangeY = f.ShelterRangeY;
    s.TerrainHeightmap = f.TerrainHeightmap;
    s.TerrainMaxHeight = f.TerrainMaxHeight;
    s.TerrainOrigin = f.TerrainOrigin;
    s.TerrainSize = f.TerrainSize;
    float crown = 1.0;
    if (surfaceClass == 2u)
    {
        float farTree;
        crown = FarTreeExposure(vegetationFlag, farTree) ? farTree : CrownExposure(s, LinearClamp, rel + CamPos);
        [unroll]
        for (uint k = 0; k < SHELTER_LEVELS; k++)
            s.Levels[k].Size = 0.0;
    }
    return crown * WeatherOpenness(s, LinearClamp, rel + CamPos + geo * 0.35 + float3(0, 0.05, 0), 0.1);
}

// The terrain's smooth normal at a place (world xz) from its height map: its own normal before its layers' normal maps
// (gputerrain.fx GetNormal), central differences a texel each way
float3 TerrainNormal(DecalFrame f, float2 xz)
{
    Texture2D<float> heights = ResourceDescriptorHeap[f.TerrainHeightmap];
    uint w, h;
    heights.GetDimensions(w, h);
    float2 uv = (xz - f.TerrainOrigin.xz) / f.TerrainSize;
    float2 du = float2(1.0 / w, 0.0), dv = float2(0.0, 1.0 / h);
    float2 metres = 2.0 * f.TerrainSize / float2(w, h);
    float dx = (heights.SampleLevel(LinearClamp, uv + du, 0) - heights.SampleLevel(LinearClamp, uv - du, 0)) * f.TerrainMaxHeight / metres.x;
    float dz = (heights.SampleLevel(LinearClamp, uv + dv, 0) - heights.SampleLevel(LinearClamp, uv - dv, 0)) * f.TerrainMaxHeight / metres.y;
    return normalize(float3(-dx, 1.0, -dz));
}

// The weather on an open surface (`open`: SurfaceOpenness; `normalVariance`: NormalVariance): wet (porous surfaces
// darken, smooth ones and flat open ground take a water film) and snow (on what faces up, patchy while it is thin,
// smoothing the relief). `geo` is the surface's smooth large-scale normal: on the terrain its height map's, elsewhere
// the geometric normal from the depth. On the terrain that one is flat per triangle of the CDLOD mesh: deep snow was
// shaded in facets, steep triangles stayed bare edge to edge, and the normal bent halfway up lit the slopes that the
// shadow map darkens, so the terrain's shadow showed as long smears on the snow (user, 2026-10-23)
void ApplyWeather(DecalFrame f, float3 rel, float3 geo, uint surfaceClass, float open, float normalVariance, inout Surface s)
{
    // Emissive pixels (lamp glass, windows) hold their light in the albedo: left as they are
    if (s.Albedo.a > 0.01 || (open <= 0.0 && s.Print <= 0.0))
        return;
    float3 world = rel + CamPos;

    float3 N = normalize(s.Normal.xyz);
    float vegetation = s.Data.a < 0.75 ? 1.0 : 0.0;
    // How far this pixel stands up from its surface, 0..1: the linear depth's height bias (gbuffer.fx, gputerrain.fx)
    // over the terrain layers' 5 cm or a mesh's ~1 cm, from the floor of a puddle's hollow where one sank it. Snow lies
    // on the high spots first, the rain stands in the low ones; a hollow (0..1 over 5 cm of dent) gathers a little more
    // of it, its middle first, a little ahead of the ground round it. Saturated at 2 cm with a head start of 0.7 the
    // whole bowl went white at a cover of 0.2 while the street was still bare (user, 2026-10-23: "the puddles fill up
    // with snow too fast")
    float linearDepth = dot(rel, f.CameraForward);
    float relief = saturate(max(linearDepth - s.Depth + s.Dent, 0.0) / ReliefScale(surfaceClass));
    float hollow = saturate(s.Dent / 0.05);
    float footprint = linearDepth / f.PixelsPerMetre;   // metres a pixel spans

    // Snow: thin, it lies on the surface's high spots first (the stones' tops, pebbles, clods: their relief, which the
    // linear depth holds as its height bias, gbuffer.fx and gputerrain.fx), varied by soft drifts metres across; as it
    // deepens it fills down into the joints and hollows and buries them. Only what faces up; not on characters (they
    // move). A noise alone made random patches (user, 2026-10-22: "shouldnt it accumulate on the height spots first")
    float snow = 0.0;
    float cover = saturate(f.SnowCover) * open;      // less falls where it is partly sheltered
    if (cover > 0.0 && surfaceClass != 3u)
    {
        float up = smoothstep(0.35, 0.85, lerp(N.y, geo.y, cover)) * smoothstep(0.2, 0.6, geo.y);
        // Plants: their leaves are the high spots (they have no height bias of their own, so a crown in the open took
        // half the snow of the ground at full cover: user, 2026-10-23, "the trees dont have enough snow"), and their
        // drifts vary up the crown too, in clumps about a metre across: by the ground's position alone every column of
        // a crown had one value
        bool plant = surfaceClass == 2u;
        float lift = plant ? 1.0 : relief;
        float drift = plant ? SnowNoise((world.xz + world.y * float2(0.7, -0.5)) * 0.8) : SnowNoise(world.xz * 0.35);
        float t = 1.0 - cover;
        snow = smoothstep(t - 0.06, t + 0.06, up * (0.1 + 0.5 * lift + 0.4 * drift + 0.2 * hollow)) * saturate(up * 3.0);
    }

    // Footprints (Print decals, what they left in s.Print): thin snow is scraped off down to the ground, deep snow is
    // packed, the dent with the snow still in it, pressed smoother and a little darker. Soft ground (the terrain) takes
    // a shallow dent of its own, a dark damp one when wet; dry stone and paving none
    float snowHere = snow;      // before it was trodden off: where there is snow the print sinks in full
    float pressed = 0.0;
    float softGround = surfaceClass == 1u ? 1.0 : 0.0;
    if (s.Print > 0.0)
    {
        float deep = smoothstep(0.35, 0.85, cover);
        pressed = s.Print * deep;
        snow *= 1.0 - s.Print * (1.0 - deep);
    }

    float facing = geo.y >= 0.0 ? lerp(0.35, 1.0, geo.y) : saturate(0.35 + geo.y);
    float wet = saturate(f.Wetness) * open * facing * (surfaceClass == 3u ? 0.5 : 1.0) * (1.0 - snow);
    if (wet > 0.0)
    {
        float rough = s.Data.r, metal = s.Data.g;
        float porosity = saturate((rough - 0.25) / 0.6) * (1.0 - metal) * (vegetation > 0.0 ? 0.5 : 1.0);
        s.Albedo.rgb *= lerp(1.0, 0.6, porosity * wet);
        // What the wet surface's roughness tends to: a water film on smooth ones, less on porous ones (the water sinks
        // in), glossier once flat open ground is soaked; never glossier than the pixel can hold of its normal map
        // (Kaplanyan & Hoffman's filtered roughness: the lobe widened by how much the normal changes to the next
        // pixel). Soaked flat ground went to a mirror (0.06) over the stones' bumps, and from mid distance, where the
        // bumps change from pixel to pixel, single pixels caught the sky: the plazas sparkled into noise (user,
        // 2026-10-23). Near, where a bump spans many pixels, the film stays glossy
        float soaked = saturate((wet - 0.6) / 0.4) * saturate((geo.y - 0.9) * 10.0)
                       * (surfaceClass == 3u ? 0.0 : 1.0) * (1.0 - vegetation);
        float film = lerp(0.1, rough * 0.4 + 0.1, porosity);
        film = lerp(film, min(film, 0.22), soaked);
        float film2 = film * film;
        film = sqrt(sqrt(saturate(film2 * film2 + min(2.0 * normalVariance, 0.18))));
        s.Data.r = lerp(rough, min(rough, film), wet);

        // Water standing in the low spots of soaked flat ground, the joints first, rising as the rain's puddles fill:
        // flat (the surface's own normal, not the stones'), a mirror. Where a pixel spans more than the joints the
        // share blends instead of flickering between stone and water
        if (soaked > 0.0)
        {
            float level = 0.1 + 0.25 * saturate(f.PuddleFill);
            float soft = 0.05 + 0.3 * saturate(footprint / 0.02);
            float water = soaked * (1.0 - smoothstep(level - soft, level + soft, relief));
            s.Data.r = lerp(s.Data.r, 0.04, water);
            s.Normal.xyz = normalize(lerp(s.Normal.xyz, geo, water));
        }
    }
    // A print in wet soft ground: darker, water in it
    if (s.Print > 0.0 && wet > 0.0 && softGround > 0.0)
    {
        float muddy = s.Print * wet;
        s.Albedo.rgb *= lerp(1.0, 0.7, muddy);
        s.Data.r = lerp(s.Data.r, min(s.Data.r, 0.2), muddy);
    }

    if (snow > 0.0)
    {
        // A thin dusting lets the ground show through; deep snow is its own surface, smooth over the stones' relief
        // (their normals, AO and displacement buried under it). Thin, it lies on the relief and keeps it.
        // (deeper sooner in a hollow, the snow drifted into it)
        float dust = snow * lerp(0.45, 1.0, smoothstep(0.0, 0.6, cover + 0.1 * hollow));
        float buried = snow * smoothstep(0.3, 0.9, cover + 0.1 * hollow);
        s.Albedo.rgb = lerp(s.Albedo.rgb, SNOW_ALBEDO, dust);
        s.Data.rgb = lerp(s.Data.rgb, float3(SNOW_ROUGHNESS, 0.0, lerp(s.Data.b, 1.0, buried)), dust);
        // Deep snow's own surface (drifts, ripples, grain, a few glinting crystals)
        float glint;
        float3 toLight = float3(0, 1, 0);
        if (AtmosphereIdx != 0)
        {
            AtmosphereParams A = LoadAtmosphere();
            toLight = A.SunDirection.y > 0.0 ? A.SunDirection : A.MoonDirection;
        }
        float3 snowN = SnowSurfaceNormal(world, geo, footprint, toLight, -normalize(rel), glint);
        s.Normal.xyz = normalize(lerp(N, snowN, buried));
        s.Data.r = lerp(s.Data.r, 0.15, glint * buried * (1.0 - pressed));
        s.Displacement *= 1.0 - buried;
        // Packed by a foot: a little darker, smoother, no crystals
        s.Albedo.rgb *= lerp(1.0, 0.86, pressed * dust);
        s.Data.r = lerp(s.Data.r, 0.6, pressed * buried);
    }

    // The prints' dent, as deep as the ground lets it be: in full where there was snow (trodden off or packed), half on
    // soft ground without it, none on dry stone and paving
    if (s.PrintDepth > 0.0)
    {
        float give = max(softGround * 0.5, saturate(snowHere * 2.0));
        if (give > 0.0)
        {
            float h = -s.PrintDepth * give, viewZ;
            float2 disp = DisplacementFor(rel, geo, h, viewZ);
            s.Displacement += disp;
            s.Depth -= h;
            s.Normal.xyz = normalize(s.Normal.xyz + s.PrintTilt * give);
        }
    }
}

groupshared uint g_TileMask[DECAL_MASK_WORDS];
groupshared uint g_TileAny;
groupshared float3 g_Normals[DECAL_TILE * DECAL_TILE];

// How much the surface's normal changes from this pixel to the next (screen-space variance, Kaplanyan & Hoffman's
// filtered roughness: σ² 0.25 of a pixel's filter): the tile's normals as the opaque pass wrote them, the neighbour on
// the other side at the tile's edge, none toward the sky or off the screen
float NormalVariance(uint2 t, float3 n)
{
    uint ix = t.x + 1u < DECAL_TILE ? t.x + 1u : t.x - 1u;
    uint iy = t.y + 1u < DECAL_TILE ? t.y + 1u : t.y - 1u;
    float3 nx = g_Normals[t.y * DECAL_TILE + ix], ny = g_Normals[iy * DECAL_TILE + t.x];
    float3 dx = dot(nx, nx) > 0.25 ? nx - n : 0.0;
    float3 dy = dot(ny, ny) > 0.25 ? ny - n : 0.0;
    return 0.25 * (dot(dx, dx) + dot(dy, dy));
}

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
    bool onScreen = all(float2(px) < ViewportSize);

    Texture2D<float> depthTex = ResourceDescriptorHeap[DepthTexIdx];
    RWTexture2D<float4> normalUav = ResourceDescriptorHeap[NormalUAVIdx];
    float depth;
    float3 rel = SurfaceAt(depthTex, px, depth);
    // The tile's normals before anything here changes them (NormalVariance); 0 where there is no surface
    g_Normals[gi] = onScreen && depth > 0.0 ? normalUav[px].xyz : 0.0;
    GroupMemoryBarrierWithGroupSync();
    if (!onScreen || depth <= 0.0)
        return;                                         // off the screen, the sky

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
    RWTexture2D<float4> dataUav = ResourceDescriptorHeap[DataUAVIdx];
    RWTexture2D<float> depthUav = ResourceDescriptorHeap[LinearDepthUAVIdx];
    RWTexture2D<float2> heightUav = ResourceDescriptorHeap[HeightUAVIdx];
    Surface s;
    s.Albedo = albedoUav[px];
    s.Normal = normalUav[px];
    s.Data = dataUav[px];
    s.Depth = depthUav[px];
    s.Displacement = heightUav[px];
    s.Dent = 0.0;
    s.Print = 0.0;
    s.PrintDepth = 0.0;
    s.PrintTilt = 0.0;
    Surface before = s;

    // The weather reaches puddles and surfaces alike: a puddle under a stall's roof stays dry, no rings under an eave
    float open = f.WeatherOn != 0u ? SurfaceOpenness(f, rel, geo, surfaceClass, s.Data.a) : 1.0;
    if (g_TileAny != 0u)
    {
        for (uint w2 = 0; w2 < words; w2++)
        {
            uint bits = g_TileMask[w2];
            while (bits != 0u)
            {
                uint b = firstbitlow(bits);
                bits &= bits - 1u;
                ApplyDecal(w2 * 32u + b, rel, geo, dx, dy, surfaceClass, f.Rain * open, f.PuddleFill * open,
                           f.SnowFallen, s);
            }
        }
    }
    // (also where footprints lie in dry weather: what they do there is the ground's to say)
    if (f.WeatherOn != 0u || s.Print > 0.0)
    {
        // (vegetation: its own dome normal, gbuffer_foliage.fx / gbuffer_crossmesh.fx: the depth's normal of a far tree's
        // vertical cards pointed sideways, and the snow stopped a few hundred metres out, far trees stood green)
        float3 largeN = geo;
        if (surfaceClass == 1u && f.TerrainHeightmap != 0)
            largeN = TerrainNormal(f, rel.xz + CamPos.xz);
        else if (surfaceClass == 2u)
        {
            largeN = normalize(s.Normal.xyz);
            // A leaf seen from below shows its bare underside: the depth's normal (the leaf card, turned to the eye)
            // points down. The G-buffer's normal cannot say so: two-sided cards, and the dome normal faces up on both
            open *= smoothstep(-0.5, -0.15, geo.y);
        }
        ApplyWeather(f, rel, largeN, surfaceClass, open, NormalVariance(gtid.xy, before.Normal.xyz), s);
    }

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
