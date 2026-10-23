// ═══════════════════════════════════════════════════════════════════════════
// Grass blades — card grass as clumps of procedural blades (Amplification → Mesh → Pixel)
//
// Every instance in a Blades distance band becomes a clump of procedural blades (see grass_procedural.hlsli for
// the routing; flowers and spikes are grass_plants.fx). Since 2026-10-23 out to the variant's range, with no card
// behind it: the spawn thins the clumps with distance and the survivors grow to keep the coverage (grass_compute.hlsl
// "Thinning"). Cards near the camera were what grass cost: ~85% of a card's pixels are transparent and still run the
// alpha test, and with `clip` there is little early-Z. Blades are opaque geometry: no alpha test, hidden blades
// rejected early, each pixel shaded about once. Blade colours come from the card's own texture (CS_BladeColors).
// ═══════════════════════════════════════════════════════════════════════════
#define PROCEDURAL_KINDS(mode) ((mode) == MODE_BLADES)   // the kinds this effect builds (AS, AS_Shadow)
#include "grass_procedural.hlsli"

// Mesh groups are small: 64 threads and room for 64 vertices, a run of blades each (grass_procedural.hlsli says why)
#define BLADE_THREADS 64
#define BLADE_GROUP_VERTS 64
#define BLADE_GROUP_TRIS 64
#define MAX_GROUP_BLADES 21     // 64 vertices / 3 (one-segment blades)
#define MAX_GROUP_CLUMPS 8      // a group's run of blades spans at most this many clumps (PackClass)
#define MAX_BLADES 28
#define BLADE_MAX_GROWTH 16.0   // a thinned clump stands in for at most this many

// Segments per blade: the clump's distance picks how smooth the bend is; from 25 m a blade is one leaning triangle
uint SegmentsAt(float dist)
{
    return dist < 5.0 ? 4u : (dist < 10.0 ? 3u : (dist < 25.0 ? 2u : 1u));
}

// What a clump looks like at its distance. A clump the spawn's distance thinning kept (di.Kept < 1) stands in for
// 1 / Kept clumps: it spreads over that much more ground (radius x sqrt) and its blades widen by the whole, so the
// meadow keeps its coverage with fewer, bigger clumps of the same blade count. Each clump changes its segment count
// at its own distance (within ±15%), so no ring.
struct ClumpShape
{
    uint  Segments;
    float Spread;           // clump radius factor
    float WidthScale;
};

ClumpShape ClumpAt(DecoInstance di)
{
    ClumpShape c;
    c.Segments = SegmentsAt(distance(CamPos, di.Position) * (0.85 + 0.3 * frac(di.InstanceSeed * 5.37)));
    float grow = min(1.0 / max(di.Kept, 1e-4), BLADE_MAX_GROWTH);
    c.Spread = sqrt(grow);
    c.WidthScale = grow;
    return c;
}

uint BladesPerClumpNow() { return min(BladesPerClump, (uint)MAX_BLADES); }

// A clump is its blades. The blades of clumps with the same segment count run on into each other, as many to a mesh
// group as fit its 64 vertices: 21 one-segment blades, 12 of two, 9 of three, 7 of four
uint PackClass(DecoInstance di, out uint units, out uint perGroup)
{
    uint segs = ClumpAt(di).Segments;
    units = BladesPerClumpNow();
    perGroup = min(BLADE_GROUP_VERTS / (2 * segs + 1), (MAX_GROUP_CLUMPS - 1) * units);   // any run of that many spans at most MAX_GROUP_CLUMPS clumps
    return segs;
}

// What a group's blades share with the rest of their clump, worked out once per clump of the group (a thread each):
// the wind's push (two gust noises), the ground variation, the colour lookups. Worked out per vertex, and then per
// blade, these were most of the work of the blade pass.
struct ClumpSetup
{
    float3 Position;
    uint   Seed;
    float2 Scale;
    float  Spread;
    float  WidthScale;
    float  Dist;
    float  Push;
    float3 Up;              // between straight up and the ground's normal (SlopeBias)
    float3 TerrainNormal;
    float3 Bands[BLADE_COLOR_BANDS];    // the card's colour root → tip x HealthyColor x the variation there
};

// Everything about a blade but where along it a vertex sits, worked out once per blade (a thread each)
struct BladeSetup
{
    float3 Root;
    float  Height;
    float2 Facing;          // the way the blade leans (horizontal, unit)
    float  Lean;
    float  Width;
    float3 Sway;            // the wind at the tip per metre of height (GrassSway)
    float  Brightness;
    uint   Clump;           // in gs_Clumps
};

groupshared ClumpSetup gs_Clumps[MAX_GROUP_CLUMPS];
groupshared BladeSetup gs_Blades[MAX_GROUP_BLADES];

BladeSetup MakeBlade(ClumpSetup c, uint blade)
{
    uint h = pcg(c.Seed ^ (blade * 0x9E3779B9u));
    float2 disc = float2(Rand(h), Rand(h));
    float radius = 0.45 * c.Scale.x * c.Spread * sqrt(disc.x);
    float rootAngle = disc.y * 6.2831853;
    BladeSetup b;
    b.Root = c.Position + float3(cos(rootAngle), 0, sin(rootAngle)) * radius;
    b.Height = c.Scale.y * lerp(0.55, 1.0, Rand(h));
    float yaw = Rand(h) * 6.2831853;
    b.Facing = float2(cos(yaw), sin(yaw));
    b.Lean = lerp(0.1, 0.55, Rand(h));
    b.Brightness = lerp(0.65, 1.05, Rand(h));
    float phase = Rand(h) * 6.2831853;
    // 0.8 – 1.6 cm (times a thinned clump's growth), never thinner than about 1.5 pixels (3440 px wide view: ~1680 px
    // per radian)
    b.Width = max(lerp(0.008, 0.016, Rand(h)) * c.WidthScale, c.Dist * 0.0009);
    b.Sway = GrassSway(c.Push, phase);
    b.Clump = 0;
    return b;
}

// The group's blades: units firstUnit .. firstUnit + count - 1 of class 'segs' (PackClass). First its clumps, a thread
// each, then the blades, a thread each. Every thread calls it (it ends in a barrier). 'colours': the camera pass' band
// colours.
void SetupGroup(uint gtid, ProceduralPayload payload, uint segs, uint firstUnit, uint count, bool colours)
{
    uint perClump = BladesPerClumpNow();
    uint firstClump = firstUnit / perClump;
    uint clumps = (firstUnit + count - 1) / perClump - firstClump + 1;
    if (gtid < clumps)
    {
        StructuredBuffer<DecoInstance> instances = ResourceDescriptorHeap[BladeInstanceSRV];
        StructuredBuffer<DecoratorSlot> slots = ResourceDescriptorHeap[DecoratorSlotsIdx];
        DecoInstance di = instances[PayloadInstance(payload, segs, firstClump + gtid)];
        DecoratorSlot slot = slots[di.SlotIdx];
        ClumpShape shape = ClumpAt(di);
        ClumpSetup c = (ClumpSetup)0;
        c.Position = di.Position;
        c.Seed = asuint(di.InstanceSeed);
        c.Scale = di.Scale;
        c.Spread = shape.Spread;
        c.WidthScale = shape.WidthScale;
        c.Dist = distance(CamPos, di.Position);
        c.Push = WindPush(di.Position.xz);
        c.Up = normalize(lerp(float3(0, 1, 0), di.TerrainNormal, abs(slot.SlopeBias)));
        c.TerrainNormal = di.TerrainNormal;
        if (colours)
        {
            StructuredBuffer<float4> bands = ResourceDescriptorHeap[BladeColorsSRV];
            float3 tint = slot.HealthyColor * GroundVariation(di.Position.xz) * UnpackGroundTint(di.GroundTint);   // x large-scale variation (common.fx) x the ground at the root (spawn)
            float4 cRoot = bands[di.SlotIdx * COLORS_PER_VARIANT];
            [unroll] for (uint k = 0; k < BLADE_COLOR_BANDS; k++)
            {
                // A band the card barely covers (the very tips) has a noisy mean: fall back towards the root colour
                float4 band = bands[di.SlotIdx * COLORS_PER_VARIANT + k];
                c.Bands[k] = lerp(cRoot.rgb, band.rgb, saturate(band.a * 20.0)) * tint;
            }
        }
        gs_Clumps[gtid] = c;
    }
    GroupMemoryBarrierWithGroupSync();

    if (gtid < count)
    {
        uint unit = firstUnit + gtid;
        uint clump = unit / perClump - firstClump;
        BladeSetup b = MakeBlade(gs_Clumps[clump], unit % perClump);
        b.Clump = clump;
        gs_Blades[gtid] = b;
    }
    GroupMemoryBarrierWithGroupSync();
}

// Vertex 'v' of a blade: 0 .. 2*segs-1 are left/right pairs up the blade, 2*segs is the tip. The blade leans out along
// its facing, more towards the tip, losing a little height doing so; the wind on top.
float3 BladePoint(BladeSetup b, float3 up, uint v, uint segs, out float t, out float sideSign)
{
    bool tip = v == 2 * segs;
    t = tip ? 1.0 : float(v / 2) / float(segs);
    sideSign = tip ? 0.0 : ((v & 1) ? 1.0 : -1.0);
    float3 facing = float3(b.Facing.x, 0, b.Facing.y);
    float3 side = float3(-facing.z, 0, facing.x);
    float3 centre = b.Root + up * b.Height * t * (1.0 - 0.3 * b.Lean * t) + facing * b.Height * b.Lean * t * t
                  + b.Sway * (t * t * b.Height);
    return centre + side * sideSign * 0.5 * b.Width * (1.0 - t);
}

// ─── Mesh Shader ───────────────────────────────────────────────────────────

struct MSOutput
{
    float4 Position    : SV_POSITION;
    float3 Color       : COLOR0;
    float3 BladeNormal : NORMAL;
    nointerpolation float3 TerrainNormal : TEXCOORD0;
    float  Depth       : TEXCOORD1;
};

[outputtopology("triangle")]
[numthreads(BLADE_THREADS, 1, 1)]
void MS(
    uint gtid : SV_GroupThreadID,
    uint3 gid : SV_GroupID,
    in payload ProceduralPayload payload,
    out vertices MSOutput verts[BLADE_GROUP_VERTS],
    out indices uint3 tris[BLADE_GROUP_TRIS])
{
    uint segs, firstUnit, count;
    GroupRun(payload, gid.x, segs, firstUnit, count);
    uint vertsPerBlade = 2 * segs + 1;
    uint trisPerBlade = 2 * segs - 1;
    uint vertCount = count * vertsPerBlade;
    uint primCount = count * trisPerBlade;
    SetMeshOutputCounts(vertCount, primCount);
    SetupGroup(gtid, payload, segs, firstUnit, count, true);

    if (gtid < vertCount)
    {
        BladeSetup b = gs_Blades[gtid / vertsPerBlade];
        ClumpSetup c = gs_Clumps[b.Clump];
        float t, sideSign;
        float3 worldPos = BladePoint(b, c.Up, gtid % vertsPerBlade, segs, t, sideSign);

        // Face normal of the bent blade, tilted towards each edge so the blade reads as rounded
        float3 facing = float3(b.Facing.x, 0, b.Facing.y);
        float3 side = float3(-facing.z, 0, facing.x);
        float3 tangent = normalize(c.Up * (1.0 - 0.6 * b.Lean * t) + facing * 2.0 * b.Lean * t);
        float3 faceN = normalize(cross(side, tangent));

        // Colour of the card at this height (bands root → tip), a per-blade brightness, darker at the root
        float bandPos = t * (BLADE_COLOR_BANDS - 1);
        uint b0 = min((uint)bandPos, BLADE_COLOR_BANDS - 2);
        float3 color = lerp(c.Bands[b0], c.Bands[b0 + 1], saturate(bandPos - b0));

        MSOutput o;
        o.Position = mul(float4(worldPos, 1.0), ViewProjection);
        o.Color = color * b.Brightness * lerp(0.55, 1.0, saturate(t / 0.35));
        o.BladeNormal = normalize(faceN + side * sideSign * 0.4);
        o.TerrainNormal = c.TerrainNormal;
        o.Depth = o.Position.w;
        verts[gtid] = o;
    }

    if (gtid < primCount)
        tris[gtid] = StripTriangle((gtid / trisPerBlade) * vertsPerBlade, gtid % trisPerBlade, segs);
}

// ─── Pixel Shader ──────────────────────────────────────────────────────────

struct PSOutput
{
    float4 Albedo : SV_Target0;
    float4 Normal : SV_Target1;
    float4 Data   : SV_Target2;
    float  Depth  : SV_Target3;
    uint   EntityId : SV_Target4;
    float2 Displacement : SV_Target5;
};

[earlydepthstencil]
PSOutput PS(MSOutput input, bool isFrontFace : SV_IsFrontFace)
{
    // Lit mostly like the ground it grows on (as the cards are: they write the terrain normal), with some of the
    // blade's own facing for relief. Both sides of a blade are seen: the back face uses the flipped normal.
    float3 bladeN = normalize(input.BladeNormal) * (isFrontFace ? 1.0 : -1.0);
    float3 N = normalize(lerp(input.TerrainNormal, bladeN, 0.35));

    PSOutput output;
    output.Albedo = float4(input.Color, 0);
    output.Normal = float4(N, 1.0);
    output.Data = float4(0.95, 0.0, 1.0, 0.5);   // as the cards: matte, no metal, ao 1, vegetation lighting
    output.Depth = input.Depth;
    output.EntityId = 2u << 30;   // class 2: vegetation (decals_cs.hlsl receivers)
    output.Displacement = float2(0, 0);
    return output;
}

// ─── Shadow ────────────────────────────────────────────────────────────────
// Grass casts into the nearest cascade only (see grass.fx); AS_Shadow (grass_procedural.hlsli) launches only the
// clumps inside it.

struct ShadowMSOutput
{
    float4 Position : SV_POSITION;
};

struct ShadowPrimAttribs
{
    uint RTIndex : SV_RenderTargetArrayIndex;
};

[outputtopology("triangle")]
[numthreads(BLADE_THREADS, 1, 1)]
void MS_Shadow(
    uint gtid : SV_GroupThreadID,
    uint3 gid : SV_GroupID,
    in payload ProceduralPayload payload,
    out vertices ShadowMSOutput verts[BLADE_GROUP_VERTS],
    out indices uint3 tris[BLADE_GROUP_TRIS],
    out primitives ShadowPrimAttribs primAttribs[BLADE_GROUP_TRIS])
{
    uint segs, firstUnit, count;
    GroupRun(payload, gid.x, segs, firstUnit, count);
    uint vertsPerBlade = 2 * segs + 1;
    uint trisPerBlade = 2 * segs - 1;
    uint vertCount = count * vertsPerBlade;
    uint primCount = count * trisPerBlade;
    SetMeshOutputCounts(vertCount, primCount);
    SetupGroup(gtid, payload, segs, firstUnit, count, false);

    if (gtid < vertCount)
    {
        StructuredBuffer<CascadeData> cascades = ResourceDescriptorHeap[CascadeBufferSRVIdx];
        float t, sideSign;
        BladeSetup b = gs_Blades[gtid / vertsPerBlade];
        float3 worldPos = BladePoint(b, gs_Clumps[b.Clump].Up, gtid % vertsPerBlade, segs, t, sideSign);
        ShadowMSOutput o;
        o.Position = mul(float4(worldPos, 1.0), cascades[0].VP);
        verts[gtid] = o;
    }

    if (gtid < primCount)
    {
        tris[gtid] = StripTriangle((gtid / trisPerBlade) * vertsPerBlade, gtid % trisPerBlade, segs);
        primAttribs[gtid].RTIndex = 0;
    }
}

void PS_Shadow(ShadowMSOutput input)
{
}

// ─── Technique ─────────────────────────────────────────────────────────────

// @RenderState(RenderTargets=4, CullMode=None)
technique11 GrassBlades
{
    pass Opaque
    {
        SetAmplificationShader(CompileShader(as_6_5, AS()));
        SetMeshShader(CompileShader(ms_6_5, MS()));
        SetPixelShader(CompileShader(ps_6_6, PS()));
    }

    pass Shadow
    {
        SetAmplificationShader(CompileShader(as_6_6, AS_Shadow()));
        SetMeshShader(CompileShader(ms_6_6, MS_Shadow()));
        SetPixelShader(CompileShader(ps_6_6, PS_Shadow()));
    }
}
