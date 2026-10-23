// Lakes and rivers (WaterBody component): a flat or gently sloping mesh generated from a spline.
// No FFT displacement — ripples are normals only, borrowed from the ocean simulation's finest bands.
// All lighting comes from water_common.fx, shared with ocean.fx.

cbuffer PushConstants : register(b3)
{
    // Slots 0-1: set by the forward pass for every forward batch
    uint ShadowMapIdx;              // 0: Shadow cascade array SRV
    uint _reserved1;                // 1: CompositeSnapshot SRV (taken from WaterData instead)
    // Slots 2-3: PER-DRAW (command signature writes these)
    uint MeshPartId;                // 2: Index into MeshRegistry
    uint InstanceBaseOffset;        // 3: Base offset for instance ID (per-command)
    // Slots 4-8: PER-BATCH (set before ExecuteIndirect)
    uint DescriptorBufIdx;          // 4: StructuredBuffer<InstanceDescriptor>
    uint SortedIndicesIdx;          // 5: StructuredBuffer<uint> - sorted draw order indices
    uint MeshRegistryIdx;           // 6: StructuredBuffer<MeshPartEntry>
    uint MaterialsIdx;              // 7: Index to materials buffer
    uint GlobalTransformBufferIdx;  // 8: Index to global TransformBuffer
    // Slots 9-15: Custom / Reserved
    uint WaterDataIdx;              // 9: Per-instance WaterData buffer
    uint _reserved10;
    uint _reserved11;
    uint _reserved12;
    uint _reserved13;
    uint _reserved14;
    uint _reserved15;
    // Slot 16: Debug
    uint DebugMode;                 // 16: Debug visualization mode
    uint _reserved17;
    uint _reserved18;
    uint _reserved19;
    // Slots 20-21: Shadow pass
    uint ExpansionBufferIdx;        // 20: SRV: expansion buffer
    uint CascadeBufferSRVIdx;       // 21: SRV: StructuredBuffer<CascadeData>
};

#include "common.fx"
#include "sky_common.fx"
#include "water_common.fx"
// @RenderState(RenderTargets=1, CullMode=None)

// Must match WaterBody.WaterData (C#)
struct WaterData
{
    float Time;
    float FlowSpeed;            // m/s along the spline (rivers)
    float RippleStrength;
    float RippleSize;
    float3 WaterColor;          // deep water
    float Visibility;           // meters of water at which the bed is mostly gone
    float3 ShallowColor;
    float RefractionStrength;
    float3 SunDirection;
    float SunIntensity;
    float3 SunColor;
    float EdgeFoam;
    float RapidsFoam;
    uint SlopeSRV;              // ocean FFT slope maps (Texture2DArray), 0 = flat water
    uint NoiseSRV;              // ocean noise texture (Perlin + Worley)
    uint DepthGBufferSRV;
    uint CompositeSRV;
    float2 InvViewportSize;
    float TileScaleA;           // 1 / tile size (m) of the two ripple bands
    float TileScaleB;
    uint BandA;
    uint BandB;
    float Flowing;              // 1 = river (open spline), 0 = lake
    float Ice;                  // Graphics.Weather.Ice: how frozen the water is, 0..1
    float SnowCover;            // Graphics.Weather.SnowCover: the snow lies on the ice too
    uint ShoreField;            // Texture2D<float>: the share of the way from the shore to the middle (0 = none yet)
    float2 ShoreOrigin;         // world xz of its corner
    float2 ShoreSize;           // metres it spans
};

SamplerState WaterSampler : register(s0);

// How thick the ice is (0..1; 0 = open water) at a place shore of the way from the shore to the middle of its water
// (WaterBody's shore field) whose current runs at speed m/s, at the ice amount: one sheet growing out from the shore, the
// middle of a lake frozen over at 0.75; a current holds it off, slowing to nothing toward the banks, so a river freezes
// from its banks in and a fast one keeps an open channel; the sheet's edge is thin over 0.1 of the ice.
// Graphics.Weather.IceThickness has the same numbers (the ice collider)
float WaterIceThickness(float ice, float shore, float speed)
{
    float needed = 0.15 + 0.6 * shore + 0.4 * saturate(speed * shore / 2.5);
    return saturate((ice - needed) / 0.1);
}

#define ICE_ALBEDO  float3(0.28, 0.32, 0.35)    // white ice, grey with the air frozen into it (the dark palette's ice;
                                                // 0.36..0.45 read as near white in the noon sun)
#define SNOW_ON_ICE float3(0.70, 0.72, 0.76)    // decals_cs.hlsl SNOW_ALBEDO: the same snow as on the ground

// The plates of a sheet of ice: cells round jittered points (Voronoi) of a grid of 1 (scale p to the plates' size),
// the distance to the nearest border between two of them (F2 − F1: about the distance to the crack) and the cell's id
float IceCells(float2 p, out float2 id)
{
    float2 i = floor(p), f = frac(p);
    float d1 = 8.0, d2 = 8.0;
    id = i;
    [unroll] for (int y = -1; y <= 1; y++)
    [unroll] for (int x = -1; x <= 1; x++)
    {
        float2 g = float2(x, y);
        float2 r = g + 0.1 + 0.8 * float2(MacroHash(i + g), MacroHash(i + g + 17.3)) - f;
        float d = dot(r, r);
        if (d < d1) { d2 = d1; d1 = d; id = i + g; }
        else if (d < d2) d2 = d;
    }
    return sqrt(d2) - sqrt(d1);
}

// Value noise interpolated linearly, -0.5..0.5: kinks at its lattice, not curves (the cracks' zigzag)
float IceJag(float2 p)
{
    float2 i = floor(p), f = frac(p);
    float a = MacroHash(i), b = MacroHash(i + float2(1, 0)), c = MacroHash(i + float2(0, 1)), d = MacroHash(i + 1.0);
    return lerp(lerp(a, b, f.x), lerp(c, d, f.x), f.y) - 0.5;
}

// A crack along the plates' borders, width metres wide in a sheet of plates cell metres across, as a share of the pixel
// it covers: no thinner than the pixel, fainter instead, so it never shimmers
float IceCrack(float edge, float width, float cell)
{
    float w = width / cell;
    float pixel = max(fwidth(edge), 1e-5);
    return (1.0 - smoothstep(0.0, max(w, pixel), edge)) * saturate(w / pixel);
}

struct VSOutput
{
    float4 Position : SV_POSITION;
    float3 WorldPos : TEXCOORD0;
    float2 UV : TEXCOORD1;             // rivers: x = meters along the spline, y = meters across
    float Depth : TEXCOORD2;
    nointerpolation uint InstanceIdx : TEXCOORD3;
    float4 SkyAmbient : TEXCOORD4;     // rgb: the sky's light on a level surface (SkyAmbient(up), the sky probe), constant per frame
    float3 Flow : TEXCOORD5;           // rivers: xy = world xz direction of the current, z = drop per meter
};

VSOutput VS(uint primitiveVertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
    VSOutput output;

    StructuredBuffer<InstanceDescriptor> descriptors = ResourceDescriptorHeap[DescriptorBufIdx];
    StructuredBuffer<row_major matrix> globalTransforms = ResourceDescriptorHeap[GlobalTransformBufferIdx];
    StructuredBuffer<uint> sortedIndices = ResourceDescriptorHeap[SortedIndicesIdx];

    StructuredBuffer<MeshPartEntry> meshRegistry = ResourceDescriptorHeap[MeshRegistryIdx];
    MeshPartEntry part = meshRegistry[MeshPartId];

    StructuredBuffer<uint> indices = ResourceDescriptorHeap[part.IndexBufferIdx];
    uint vertexID = indices[primitiveVertexID + part.BaseIndex];
    StructuredBuffer<float3> positions = ResourceDescriptorHeap[part.PosBufferIdx];
    StructuredBuffer<float2> uvs = ResourceDescriptorHeap[part.UVBufferIdx];

    uint dataPos = InstanceBaseOffset + instanceID;
    uint idx = sortedIndices[dataPos] & 0x7FFFFFFFu;

    InstanceDescriptor desc = descriptors[idx];
    row_major matrix World = globalTransforms[desc.TransformSlot];

    float3 worldPos = mul(float4(positions[vertexID], 1.0f), World).xyz;

    // WaterBody stores the current in the normal stream of a river's mesh: xz = direction, y = steepness
    StructuredBuffer<float3> normals = ResourceDescriptorHeap[part.NormBufferIdx];
    float3 flow = normals[vertexID];
    output.Flow = float3(mul(float4(flow.x, 0.0f, flow.z, 0.0f), World).xz, flow.y);

    output.WorldPos = worldPos;
    output.UV = uvs[vertexID];
    output.Position = mul(mul(float4(worldPos, 1.0), View), Projection);
    output.Depth = output.Position.w;
    output.InstanceIdx = idx;
    output.SkyAmbient = float4(SkyAmbient(float3(0, 1, 0)), 1.0);
    return output;
}

struct PSOutput
{
    float4 Color : SV_Target0;
};

PSOutput PS(VSOutput input)
{
    PSOutput output;

    StructuredBuffer<WaterData> waterData = ResourceDescriptorHeap[WaterDataIdx];
    WaterData water = waterData[input.InstanceIdx];

    float3 camPos = ViewInverse[3].xyz;
    float3 worldPos = input.WorldPos;
    // How much sky this pixel of water sees past the terrain (per pixel: a lake is a few big triangles whose rim lies
    // under the banks, and per vertex those dark corners painted wedges across it)
    float skyVisible = SkyVisibility(worldPos, float3(0, 1, 0));
    float3 V = normalize(camPos - worldPos);
    float dist = distance(worldPos, camPos);
    float t = water.Time;
    bool flowing = water.Flowing > 0.5;

    // ── Flow frame (rivers) ──
    // The current's direction and how steeply this stretch falls (rapids) come per vertex from the
    // spline. Screen-space derivatives of the surface would be constant per triangle, and in a bend
    // the two triangles of a quad differ: that showed as stripes of white water.
    float2 flowDir = flowing && dot(input.Flow.xy, input.Flow.xy) > 1e-8 ? normalize(input.Flow.xy) : float2(1, 0);
    float2 acrossDir = float2(-flowDir.y, flowDir.x);
    float steep = flowing ? saturate(input.Flow.z) : 0.0;

    // ── Ice: how thick here, by how far out from the shore (WaterBody's shore field) and the current; the sheet's edge
    // jagged by noises interpolated linearly (kinks, not blobs: two smooth octaves made islands of ice all over a lake
    // whose middle sat at the threshold). The collider over it (WaterBody) follows the same rule without the edge ──
    float ice = 0.0, shore = 1.0;
    if (water.Ice > 0.0 && water.ShoreField != 0)
    {
        Texture2D<float> shoreField = ResourceDescriptorHeap[water.ShoreField];
        shore = shoreField.SampleLevel(AtmosphereLinearClamp, (worldPos.xz - water.ShoreOrigin) / water.ShoreSize, 0);
        float speed = flowing ? water.FlowSpeed * (1.0 + 4.0 * steep) : 0.0;
        float jag = IceJag(worldPos.xz / 23.0) * 0.55 + IceJag(worldPos.xz / 7.7 + 7.3) * 0.3 + IceJag(worldPos.xz / 2.3 - 2.9) * 0.15;
        ice = WaterIceThickness(water.Ice + jag * 0.09, shore, speed);
    }

    // Pattern space: lakes sample in world space; rivers in strip space (u along the flow, v across),
    // scrolled downstream. Strip space bends with the river, so the scroll never shears.
    float2 patternPos = flowing ? float2(input.UV.x - water.FlowSpeed * t, input.UV.y) : worldPos.xz;

    // ── Ripple normal and roughness: two ocean bands' slope moments (water_common.fx WaterSlopes) ──
    WaterSlopes slopes = (WaterSlopes)0;
    float resolvedTotal = 0;
    if (water.SlopeSRV != 0)
    {
        Texture2DArray<float4> slopeTex = ResourceDescriptorHeap[water.SlopeSRV];
        float2 p = patternPos / max(0.05, water.RippleSize);
        float scale = water.RippleStrength * (1.0 + steep * 4.0);
        float4 a = slopeTex.Sample(WaterSampler, float3(p * water.TileScaleA, water.BandA));
        float4 b = slopeTex.Sample(WaterSampler, float3(p * water.TileScaleB + 0.37, water.BandB));

        // Strip space → world (a rotation: the variance stays)
        float2 meanA = flowing ? flowDir * a.x + acrossDir * a.y : a.xy;
        float2 meanB = flowing ? flowDir * b.x + acrossDir * b.y : b.xy;
        AddWaterBand(slopes, a, meanA, scale);
        AddWaterBand(slopes, b, meanB, scale);
        resolvedTotal = WaterBandTotalVariance(slopeTex, WaterSampler, water.BandA, scale)
                      + WaterBandTotalVariance(slopeTex, WaterSampler, water.BandB, scale);
    }
    float3 N = normalize(float3(-slopes.mean.x, 1.0, -slopes.mean.y));
    float alpha = WaterAlpha(slopes, WaterUnresolvedVariance(resolvedTotal));
    float NdotV = saturate(dot(N, V));

    // ── Sun ──
    float3 L = normalize(-water.SunDirection);
    float NdotL = saturate(dot(N, L));
    // Pre-exposed E / π of the directional light, as the deferred light has it; the specular takes E (× π)
    float3 sunRadiance = water.SunColor * water.SunIntensity * SunLightScale;
    sunRadiance *= CloudBeam(WaterSampler, worldPos - camPos, L);     // the clouds' beam: no glint under a deck
    // The cascades, and the terrain's shadow past them (a mountain the sun stands behind), as the deferred light has them
    sunRadiance *= min(WaterSunShadow(ShadowMapIdx, worldPos - camPos, input.Position.xy), TerrainLightShadow(worldPos, L));
    sunRadiance *= TerrainOpen(worldPos, float3(0, 1, 0));   // a sea cave under a hill: no sun, whatever the shadow map holds
    float3 skyAmbient = input.SkyAmbient.rgb * skyVisible;

    // ── Deep water color (same model as the ocean, without the wave-height term) ──
    float3 scatter = (0.1 * NdotV * NdotV + 0.08 * NdotL) * water.WaterColor * sunRadiance
                   + water.WaterColor * skyAmbient * 0.15;

    // ── Shared water shading ──
    float F = WaterSpecularReflectance(NdotV, alpha);
    float3 reflectColor = WaterSkyReflection(V, N, alpha) * skyVisible;   // no sky to mirror under a hill
    float3 specular = WaterSunSpecular(N, V, L, sunRadiance * 3.14159, alpha);

    WaterColumn column = GetWaterColumn(WaterSampler, water.DepthGBufferSRV, water.CompositeSRV,
        input.Position.xy * water.InvViewportSize, input.Depth, dist, abs(camPos.y - worldPos.y), N,
        water.RefractionStrength, scatter, water.ShallowColor, water.Visibility, 1.0);

    // Light scattered back from a bright shallow bed
    float shallow = exp(-max(column.bufDepth, 0.0) / max(0.01, water.Visibility)) * saturate(column.pathLen);
    float3 body = column.body
                + water.ShallowColor * (sunRadiance * saturate(L.y) + skyAmbient) * 0.06 * shallow;

    float3 color = (1.0 - F) * body + (specular + F * reflectColor) * column.edgeFade;
    color = max(0.0, color);

    // ── Foam: coverage eroded by a cellular texture (as on the ocean) ──
    float coverage = 0.0;

    // Along the banks and around anything standing in the water
    coverage += (1.0 - smoothstep(0.0, 0.12, column.bufDepth)) * water.EdgeFoam;

    // White water where the surface runs steeply downhill
    // (kept below full coverage so the cell texture always breaks it up into streaks of dark water)
    coverage += smoothstep(0.04, 0.3, steep) * 0.68 * water.RapidsFoam;
    coverage = saturate(coverage);

    float foamTex = 0.5;
    if (water.NoiseSRV != 0 && coverage > 0.001)
    {
        Texture2D<float4> noiseTex = ResourceDescriptorHeap[water.NoiseSRV];

        // Foam rides the current, faster where it is steep (bounded extra speed keeps it from shearing)
        float2 foamPos = flowing
            ? float2(input.UV.x - water.FlowSpeed * 1.6 * t, input.UV.y)
            : worldPos.xz + t * float2(0.01, 0.006);
        float cells1 = noiseTex.SampleLevel(WaterSampler, foamPos * 0.23, 0).g;
        float cells2 = noiseTex.SampleLevel(WaterSampler, foamPos * 0.83 + 0.5, 0).g;
        foamTex = saturate((cells1 * 0.6 + cells2 * 0.4 - 0.15) / 0.6);
        foamTex = lerp(foamTex, 0.5, saturate(dist / 150.0));   // no mips on the noise texture
    }

    float foam = saturate((coverage * 1.25 - (1.0 - foamTex)) * 3.0);
    foam *= smoothstep(0.0, 0.025, column.pathLen);     // never end on the hard edge at the bank
    if (foam > 0.001)
    {
        float3 foamLit = WaterFoamLit(float3(0.80, 0.80, 0.78) * lerp(0.7, 1.0, foamTex), NdotL, sunRadiance,
            SkyAmbient(N) * skyVisible);
        color = lerp(color, foamLit, foam * 0.9);
    }

    // ── The ice over the water, and the snow on it ──
    if (ice > 0.0)
    {
        // A sheet of plates (~7 m, cracked again at ~2 m) with straight borders between their corners, jagged by a
        // small zigzag (a smooth warp bent them into curves: "cracks mustnt be smooth curves", user); each plate tilted
        // a little, so they catch the light each its own way. Two kinds of ice in broad patches: clear (black) ice that
        // shows the dark water under it and mirrors the sky, and white ice, grey with the air frozen into it. A crack
        // shows light in clear ice, dark in white ice
        float2 p = worldPos.xz;
        float2 bent = p + float2(IceJag(p * 3.0), IceJag(p * 3.0 + 9.1)) * 0.12 + float2(IceJag(p * 11.0 + 4.2), IceJag(p * 11.0 - 2.7)) * 0.04;
        float2 plate, shard;
        float plateEdge = IceCells(bent / 7.0, plate);
        float shardEdge = IceCells(bent / 2.0 + 31.7, shard);
        float cracks = max(IceCrack(plateEdge, 0.035, 7.0), 0.5 * IceCrack(shardEdge, 0.015, 2.0)) * smoothstep(0.3, 0.8, ice);
        float2 tilt = (float2(MacroHash(plate), MacroHash(plate + 5.3)) - 0.5) * 0.015;
        float3 Ni = normalize(float3(tilt.x + (MacroNoise(p / 9.0) - 0.5) * 0.03, 1.0, tilt.y + (MacroNoise(p / 3.1 + 7.7) - 0.5) * 0.03));
        float NdotLi = saturate(dot(Ni, L));
        // The oldest ice, along the shore where the sheet began, has frozen whiter; out from there it is clear black ice
        // to the newest at the sheet's edge. Mottled by frost and frozen-in air a few metres across (jagged noise). Big
        // soft patches of white ice over the whole sheet read as smears painted on it (user, 2026-10-23: "looks roughed
        // in"); the white of a frozen lake is its snow, laid in streaks
        float frost = IceJag(p / 3.0) + 0.5 * IceJag(p / 1.1 + 6.1);
        float white = smoothstep(0.4, 0.05, shore) * saturate(0.7 + frost) * smoothstep(0.5, 1.0, ice);

        // Thin ice is a skin over the dark water; the water under it keeps what it showed but its mirror
        float opacity = lerp(0.35, 0.9, ice) * lerp(0.2, 1.0, white) * (1.0 + 0.3 * frost);
        float roughness = lerp(0.04, 0.15, white);
        float3 albedo = ICE_ALBEDO * (1.0 + 0.3 * frost) * lerp(1.0, 0.6, cracks * white);
        opacity = lerp(opacity, 0.8, cracks * (1.0 - white));                  // a white crack in clear ice
        float3 lit = albedo * (sunRadiance * NdotLi + SkyAmbient(Ni) * skyVisible);
        float3 iceBody = lerp((1.0 - F) * body, lit, opacity);
        float Fi = WaterSpecularReflectance(saturate(dot(Ni, V)), roughness);
        float3 iceColor = (1.0 - Fi) * iceBody + (WaterSunSpecular(Ni, V, L, sunRadiance * 3.14159, roughness)
                        + Fi * WaterSkyReflection(V, Ni, roughness) * skyVisible) * column.edgeFade;

        // Snow lies on ice that holds it, as the cover grows (on the ground: decals_cs.hlsl), in streaks the wind laid
        // down along itself over the open sheet (round blots of noise read as foam); none on the thin ice at the edge
        float2 windAlong = dot(WindDirection.xz, WindDirection.xz) > 1e-4 ? normalize(WindDirection.xz) : float2(1, 0);
        float2 windFrame = float2(dot(p, windAlong), dot(p, float2(-windAlong.y, windAlong.x)));
        float drift = MacroNoise(windFrame / float2(14.0, 3.5)) * 0.65 + MacroNoise(windFrame / float2(5.0, 1.4) + 21.0) * 0.35;
        float bare = 1.0 - saturate(water.SnowCover);
        float snow = smoothstep(bare - 0.08, bare + 0.08, (0.35 + 0.65 * drift) * smoothstep(0.5, 1.0, ice));
        float3 snowLit = SNOW_ON_ICE * (sunRadiance * saturate(L.y) + SkyAmbient(float3(0, 1, 0)) * skyVisible);
        iceColor = lerp(iceColor, snowLit, snow);

        // The sheet's edge a line, not a blur: the rippled open water stops at the glossy ice, and along it a thin rim of
        // grey slush (frazil the wind and the current pile against the edge), a metre or two wide
        float rim = smoothstep(0.0, 0.02, ice) * (1.0 - smoothstep(0.04, 0.12, ice)) * saturate(0.2 + 2.0 * frost);
        float3 slush = ICE_ALBEDO * (sunRadiance * saturate(L.y) + SkyAmbient(float3(0, 1, 0)) * skyVisible);
        iceColor = lerp(iceColor, slush, rim * 0.8);
        color = lerp(color, iceColor, smoothstep(0.0, 0.02, ice));
    }

    // ── Aerial perspective ──
    color = ApplyAerialPerspective(color, input.Position.xy, input.Depth, skyVisible);

    // HDR output — the finalize pass tonemaps the whole Composite
    output.Color = float4(color, 1.0);
    return output;
}

technique11 GBuffer
{
    pass Forward
    {
        SetVertexShader(CompileShader(vs_6_6, VS()));
        SetPixelShader(CompileShader(ps_6_6, PS()));
    }
}
