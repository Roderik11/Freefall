// ═══════════════════════════════════════════════════════════════════════════
// Grass blades — what card grass becomes near the camera (Amplification → Mesh → Pixel)
//
// Every instance in a Blades distance band becomes a clump of procedural blades (see grass_procedural.hlsli for
// the routing; flowers and spikes are grass_plants.fx). Cards near the camera are what grass cost: ~85% of a
// card's pixels are transparent and still run the alpha test, and with `clip` there is little early-Z. Blades
// are opaque geometry: no alpha test, hidden blades rejected early, each pixel shaded about once. Blade colours
// come from the card's own texture (CS_BladeColors), so the hand-over matches.
// ═══════════════════════════════════════════════════════════════════════════
#include "grass_procedural.hlsli"

#define MAX_BLADES 28           // x (2 x 4 segments + 1) = 252 vertices, x (2 x 4 - 1) = 196 triangles

// Segments per blade: the clump's distance picks how smooth the bend is
uint SegmentsAt(float dist)
{
    return dist < 5.0 ? 4u : (dist < 10.0 ? 3u : 2u);
}

struct BladeVertexData
{
    float3 WorldPos;
    float3 BladeNormal;     // face normal, rounded across the width
    float3 Color;
};

// One vertex of blade 'blade' in a clump. Vertices 0 .. 2*segs-1 are left/right pairs up the blade, 2*segs is the tip.
BladeVertexData BladeVertex(DecoInstance di, DecoratorSlot slot, uint blade, uint v, uint segs, float dist)
{
    uint h = pcg(asuint(di.InstanceSeed) ^ (blade * 0x9E3779B9u));
    float2 disc = float2(Rand(h), Rand(h));
    float radius = 0.45 * di.Scale.x * sqrt(disc.x);
    float rootAngle = disc.y * 6.2831853;
    float height = di.Scale.y * lerp(0.55, 1.0, Rand(h));
    float yaw = Rand(h) * 6.2831853;
    float lean = lerp(0.1, 0.55, Rand(h));
    float brightness = lerp(0.65, 1.05, Rand(h));
    float phase = Rand(h) * 6.2831853;
    // 0.8 – 1.6 cm, and never thinner than about 1.5 pixels (3440 px wide view: ~1680 px per radian)
    float width = max(lerp(0.008, 0.016, Rand(h)), dist * 0.0009);

    float3 up = normalize(lerp(float3(0, 1, 0), di.TerrainNormal, abs(slot.SlopeBias)));
    float3 facing = float3(cos(yaw), 0, sin(yaw));
    float3 side = float3(-facing.z, 0, facing.x);
    float3 root = di.Position + float3(cos(rootAngle), 0, sin(rootAngle)) * radius;

    bool tip = v == 2 * segs;
    float t = tip ? 1.0 : float(v / 2) / float(segs);
    float sideSign = tip ? 0.0 : ((v & 1) ? 1.0 : -1.0);

    // Bend: the blade leans out along 'facing', more towards the tip; it loses a little height doing so
    float3 centre = root + up * height * t * (1.0 - 0.3 * lean * t) + facing * height * lean * t * t;
    centre += Wind(root, t, height, phase);

    BladeVertexData o;
    o.WorldPos = centre + side * sideSign * 0.5 * width * (1.0 - t);

    // Face normal of the bent blade, tilted towards each edge so the blade reads as rounded
    float3 tangent = normalize(up * (1.0 - 0.6 * lean * t) + facing * 2.0 * lean * t);
    float3 faceN = normalize(cross(side, tangent));
    o.BladeNormal = normalize(faceN + side * sideSign * 0.4);

    // Colour of the card at this height (bands root → tip), a per-blade brightness, darker at the root
    StructuredBuffer<float4> bands = ResourceDescriptorHeap[BladeColorsSRV];
    float bandPos = t * (BLADE_COLOR_BANDS - 1);
    uint b0 = min((uint)bandPos, BLADE_COLOR_BANDS - 2);
    float4 c0 = bands[di.SlotIdx * COLORS_PER_VARIANT + b0];
    float4 c1 = bands[di.SlotIdx * COLORS_PER_VARIANT + b0 + 1];
    float4 cRoot = bands[di.SlotIdx * COLORS_PER_VARIANT];
    // A band the card barely covers (the very tips) has a noisy mean: fall back towards the root colour
    c0.rgb = lerp(cRoot.rgb, c0.rgb, saturate(c0.a * 20.0));
    c1.rgb = lerp(cRoot.rgb, c1.rgb, saturate(c1.a * 20.0));
    float3 color = lerp(c0.rgb, c1.rgb, saturate(bandPos - b0));
    float rootShade = lerp(0.55, 1.0, saturate(t / 0.35));
    o.Color = color * brightness * rootShade * slot.HealthyColor;
    return o;
}

// What one MS group outputs for an instance: blade clumps only (plants and padding: nothing)
void ClumpCounts(DecoInstance di, float dist, out uint verts, out uint prims, out uint segs)
{
    segs = SegmentsAt(dist);
    uint blades = (di.Mode == MODE_BLADES && di.Scale.y > 0) ? min(BladesPerClump, (uint)MAX_BLADES) : 0;
    verts = blades * (2 * segs + 1);
    prims = blades * (2 * segs - 1);
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
[numthreads(MS_THREADS, 1, 1)]
void MS(
    uint gtid : SV_GroupThreadID,
    uint3 gid : SV_GroupID,
    in payload ProceduralPayload payload,
    out vertices MSOutput verts[MAX_VERTS],
    out indices uint3 tris[MAX_PRIMS])
{
    StructuredBuffer<DecoInstance> instances = ResourceDescriptorHeap[BladeInstanceSRV];
    DecoInstance di = instances[payload.BaseInstance + gid.x];

    float dist = distance(CamPos, di.Position);
    uint vertCount, primCount, segs;
    ClumpCounts(di, dist, vertCount, primCount, segs);
    SetMeshOutputCounts(vertCount, primCount);

    StructuredBuffer<DecoratorSlot> slots = ResourceDescriptorHeap[DecoratorSlotsIdx];
    DecoratorSlot slot = slots[di.SlotIdx];

    for (uint vi = gtid; vi < vertCount; vi += MS_THREADS)
    {
        BladeVertexData bv = BladeVertex(di, slot, vi / (2 * segs + 1), vi % (2 * segs + 1), segs, dist);
        MSOutput o;
        o.Position = mul(float4(bv.WorldPos, 1.0), ViewProjection);
        o.Color = bv.Color;
        o.BladeNormal = bv.BladeNormal;
        o.TerrainNormal = di.TerrainNormal;
        o.Depth = o.Position.w;
        verts[vi] = o;
    }

    for (uint ti = gtid; ti < primCount; ti += MS_THREADS)
        tris[ti] = StripTriangle((ti / (2 * segs - 1)) * (2 * segs + 1), ti % (2 * segs - 1), segs);
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
    output.EntityId = 0;
    output.Displacement = float2(0, 0);
    return output;
}

// ─── Shadow ────────────────────────────────────────────────────────────────
// Grass casts into the nearest cascade only (see grass.fx); blade clumps are near the camera, so they are in it.

struct ShadowMSOutput
{
    float4 Position : SV_POSITION;
};

struct ShadowPrimAttribs
{
    uint RTIndex : SV_RenderTargetArrayIndex;
};

[outputtopology("triangle")]
[numthreads(MS_THREADS, 1, 1)]
void MS_Shadow(
    uint gtid : SV_GroupThreadID,
    uint3 gid : SV_GroupID,
    in payload ProceduralPayload payload,
    out vertices ShadowMSOutput verts[MAX_VERTS],
    out indices uint3 tris[MAX_PRIMS],
    out primitives ShadowPrimAttribs primAttribs[MAX_PRIMS])
{
    StructuredBuffer<DecoInstance> instances = ResourceDescriptorHeap[BladeInstanceSRV];
    DecoInstance di = instances[payload.BaseInstance + gid.x];

    float dist = distance(CamPos, di.Position);
    uint vertCount, primCount, segs;
    ClumpCounts(di, dist, vertCount, primCount, segs);
    SetMeshOutputCounts(vertCount, primCount);

    StructuredBuffer<DecoratorSlot> slots = ResourceDescriptorHeap[DecoratorSlotsIdx];
    StructuredBuffer<CascadeData> cascades = ResourceDescriptorHeap[CascadeBufferSRVIdx];
    DecoratorSlot slot = slots[di.SlotIdx];
    row_major float4x4 cascadeVP = cascades[0].VP;

    for (uint vi = gtid; vi < vertCount; vi += MS_THREADS)
    {
        BladeVertexData bv = BladeVertex(di, slot, vi / (2 * segs + 1), vi % (2 * segs + 1), segs, dist);
        ShadowMSOutput o;
        o.Position = mul(float4(bv.WorldPos, 1.0), cascadeVP);
        verts[vi] = o;
    }

    for (uint ti = gtid; ti < primCount; ti += MS_THREADS)
    {
        tris[ti] = StripTriangle((ti / (2 * segs - 1)) * (2 * segs + 1), ti % (2 * segs - 1), segs);
        primAttribs[ti].RTIndex = 0;
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
        SetAmplificationShader(CompileShader(as_6_6, AS()));
        SetMeshShader(CompileShader(ms_6_6, MS_Shadow()));
        SetPixelShader(CompileShader(ps_6_6, PS_Shadow()));
    }
}
