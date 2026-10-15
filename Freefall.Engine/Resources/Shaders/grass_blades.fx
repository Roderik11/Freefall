// ═══════════════════════════════════════════════════════════════════════════
// Grass blades — what card grass becomes near the camera (Amplification → Mesh → Pixel)
//
// CS_SpawnInstances routes card instances closer than BladeDistance (each at its own distance within ±15%) into
// a buffer of their own; here every one of them becomes a clump of procedural blades. Cards near the camera are
// what grass cost: ~85% of a card's pixels are transparent and still run the alpha test, and with `clip` there is
// little early-Z. Blades are opaque geometry: no alpha test, hidden blades rejected early, each pixel shaded
// about once. Blade colours come from the card's own texture (CS_BladeColors), so the hand-over matches.
// ═══════════════════════════════════════════════════════════════════════════
cbuffer PushConstants : register(b3)
{
    uint DecoratorSlotsIdx;     // 0
    uint BladeInstanceSRV;      // 1: DecoInstance, padded to whole AS groups with zero-scale entries
    uint BladeColorsSRV;        // 2: float4 per variant x BLADE_COLOR_BANDS (rgb, coverage), root → tip
    uint BladesPerClump;        // 3
    uint CascadeBufferSRVIdx;   // 4: shadow pass only
};

#include "common.fx"

// ─── GPU structs (must match grass_compute.hlsl and TerrainRenderer.cs) ─────

struct DecoratorSlot
{
    float Density;
    float MinH, MaxH;
    float MinW, MaxW;
    uint LODCount;
    uint LODTableOffset;
    float Rot00, Rot01, Rot02;
    float Rot10, Rot11, Rot12;
    float Rot20, Rot21, Rot22;
    float SlopeBias;
    uint Seed;
    uint BladesNearCamera;
    uint Mode;
    uint TextureIdx;
    float3 HealthyColor;
    float3 DryColor;
    float NoiseSpread;
    uint _unused1;
    float _unused2;
    float ClusterScale;
    float ClusterAmount;
    uint AlphaClip;
    float HeightNoiseScale;
    float HeightNoiseAmount;
};

struct DecoInstance
{
    float3 Position;
    float  Rotation;
    float3 TerrainNormal;
    float  FadeFactor;
    float2 Scale;           // (width, height) of the card this clump stands in for
    float2 TerrainUV;
    uint   SlotIdx;
    uint   LOD;
    float  InstanceSeed;
    uint   _pad;
};

// ─── Constants ─────────────────────────────────────────────────────────────

#define CLUMPS_PER_AS 32        // = BLADE_CLUMPS_PER_AS in grass_compute.hlsl
#define BLADE_COLOR_BANDS 4     // = grass_compute.hlsl
#define MAX_BLADES 28
#define MAX_SEGMENTS 4
#define MS_THREADS 128          // each thread writes up to two vertices and two triangles
#define MAX_VERTS 256           // MAX_BLADES x (2 x MAX_SEGMENTS + 1) = 252
#define MAX_PRIMS 256           // MAX_BLADES x (2 x MAX_SEGMENTS - 1) = 196

// ─── Helpers ───────────────────────────────────────────────────────────────

uint pcg(uint v)
{
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float Rand(inout uint h)
{
    h = pcg(h);
    return float(h) / 4294967295.0;
}

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

    // Wind: the cards' sway, plus a per-blade phase so a clump does not move as one
    float windInfluence = t * t;
    float windWave1 = sin(Time * 2.1 + root.x * 0.8 + root.z * 0.6 + phase * 0.3) * 0.12;
    float windWave2 = sin(Time * 1.4 + root.x * 0.5 - root.z * 0.9 + phase * 0.2) * 0.08;
    centre.x += (windWave1 + windWave2) * windInfluence * height;
    centre.z += (windWave2 - windWave1 * 0.5) * windInfluence * height;

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
    float4 c0 = bands[di.SlotIdx * BLADE_COLOR_BANDS + b0];
    float4 c1 = bands[di.SlotIdx * BLADE_COLOR_BANDS + b0 + 1];
    float4 cRoot = bands[di.SlotIdx * BLADE_COLOR_BANDS];
    // A band the card barely covers (the very tips) has a noisy mean: fall back towards the root colour
    c0.rgb = lerp(cRoot.rgb, c0.rgb, saturate(c0.a * 20.0));
    c1.rgb = lerp(cRoot.rgb, c1.rgb, saturate(c1.a * 20.0));
    float3 color = lerp(c0.rgb, c1.rgb, saturate(bandPos - b0));
    float rootShade = lerp(0.55, 1.0, saturate(t / 0.35));
    o.Color = color * brightness * rootShade * slot.HealthyColor;
    return o;
}

// ─── Amplification Shader ──────────────────────────────────────────────────

struct BladePayload
{
    uint BaseInstance;
};

groupshared BladePayload s_Payload;

// One clump per MS group. The instance buffer is padded to whole AS groups, so every group launches all of them.
[numthreads(1, 1, 1)]
void AS(uint3 gid : SV_GroupID)
{
    s_Payload.BaseInstance = gid.x * CLUMPS_PER_AS;
    DispatchMesh(CLUMPS_PER_AS, 1, 1, s_Payload);
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

// Triangle 'p' of a blade: quads up the blade as two triangles each, then one triangle to the tip
uint3 BladeTriangle(uint baseVertex, uint p, uint segs)
{
    if (p >= 2 * (segs - 1))
        return baseVertex + uint3(2 * (segs - 1), 2 * (segs - 1) + 1, 2 * segs);
    uint k = p / 2;
    uint l0 = baseVertex + 2 * k;
    return (p & 1) ? uint3(l0 + 1, l0 + 3, l0 + 2) : uint3(l0, l0 + 1, l0 + 2);
}

[outputtopology("triangle")]
[numthreads(MS_THREADS, 1, 1)]
void MS(
    uint gtid : SV_GroupThreadID,
    uint3 gid : SV_GroupID,
    in payload BladePayload payload,
    out vertices MSOutput verts[MAX_VERTS],
    out indices uint3 tris[MAX_PRIMS])
{
    StructuredBuffer<DecoInstance> instances = ResourceDescriptorHeap[BladeInstanceSRV];
    DecoInstance di = instances[payload.BaseInstance + gid.x];

    float dist = distance(CamPos, di.Position);
    uint segs = SegmentsAt(dist);
    uint blades = di.Scale.y > 0 ? min(BladesPerClump, (uint)MAX_BLADES) : 0;
    uint vertsPerBlade = 2 * segs + 1;
    uint trisPerBlade = 2 * segs - 1;
    SetMeshOutputCounts(blades * vertsPerBlade, blades * trisPerBlade);

    StructuredBuffer<DecoratorSlot> slots = ResourceDescriptorHeap[DecoratorSlotsIdx];
    DecoratorSlot slot = slots[di.SlotIdx];

    for (uint vi = gtid; vi < blades * vertsPerBlade; vi += MS_THREADS)
    {
        BladeVertexData bv = BladeVertex(di, slot, vi / vertsPerBlade, vi % vertsPerBlade, segs, dist);
        MSOutput o;
        o.Position = mul(float4(bv.WorldPos, 1.0), ViewProjection);
        o.Color = bv.Color;
        o.BladeNormal = bv.BladeNormal;
        o.TerrainNormal = di.TerrainNormal;
        o.Depth = o.Position.w;
        verts[vi] = o;
    }

    for (uint ti = gtid; ti < blades * trisPerBlade; ti += MS_THREADS)
        tris[ti] = BladeTriangle((ti / trisPerBlade) * vertsPerBlade, ti % trisPerBlade, segs);
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
    in payload BladePayload payload,
    out vertices ShadowMSOutput verts[MAX_VERTS],
    out indices uint3 tris[MAX_PRIMS],
    out primitives ShadowPrimAttribs primAttribs[MAX_PRIMS])
{
    StructuredBuffer<DecoInstance> instances = ResourceDescriptorHeap[BladeInstanceSRV];
    DecoInstance di = instances[payload.BaseInstance + gid.x];

    float dist = distance(CamPos, di.Position);
    uint segs = SegmentsAt(dist);
    uint blades = di.Scale.y > 0 ? min(BladesPerClump, (uint)MAX_BLADES) : 0;
    uint vertsPerBlade = 2 * segs + 1;
    uint trisPerBlade = 2 * segs - 1;
    SetMeshOutputCounts(blades * vertsPerBlade, blades * trisPerBlade);

    StructuredBuffer<DecoratorSlot> slots = ResourceDescriptorHeap[DecoratorSlotsIdx];
    StructuredBuffer<CascadeData> cascades = ResourceDescriptorHeap[CascadeBufferSRVIdx];
    DecoratorSlot slot = slots[di.SlotIdx];
    row_major float4x4 cascadeVP = cascades[0].VP;

    for (uint vi = gtid; vi < blades * vertsPerBlade; vi += MS_THREADS)
    {
        BladeVertexData bv = BladeVertex(di, slot, vi / vertsPerBlade, vi % vertsPerBlade, segs, dist);
        ShadowMSOutput o;
        o.Position = mul(float4(bv.WorldPos, 1.0), cascadeVP);
        verts[vi] = o;
    }

    for (uint ti = gtid; ti < blades * trisPerBlade; ti += MS_THREADS)
    {
        tris[ti] = BladeTriangle((ti / trisPerBlade) * vertsPerBlade, ti % trisPerBlade, segs);
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
