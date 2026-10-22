// ═══════════════════════════════════════════════════════════════════════════
// Flowers, spikes and ferns — what plant cards become near the camera (Amplification → Mesh → Pixel)
//
// Every instance in a Flowers or Spikes distance band becomes a small plant of PLANT_STEMS stems, one in a Ferns
// band a crown of FERN_FRONDS fronds (routing in grass_procedural.hlsli; grass blades are grass_blades.fx).
// Geometry gives the structure — stems, where blooms and leaves sit, how fronds arch, which way they face, wind — and
// small parts of the variant's parts atlas (DecoratorVariant.PlantParts, rendered by
// Tools/BlenderKits/build_flowers.py) give the detail: a bloom cap seen from above, a crossed pair of spike strips,
// leaves, a frond. Each part fills most of its quad, so the alpha test wastes few pixels (a whole flower card is
// ~15% opaque). Without an atlas the parts are flat colours from the card.
// ═══════════════════════════════════════════════════════════════════════════
#define PROCEDURAL_SHADOW_MODE(mode) ((mode) >= MODE_FLOWERS && (mode) <= MODE_FERNS)   // this effect's kinds
#include "grass_procedural.hlsli"

#define PLANT_STEMS 3
#define STEM_SEGMENTS 4
#define STEM_VERTS 9            // 2 x STEM_SEGMENTS + 1
#define STEM_TRIS 7             // 2 x STEM_SEGMENTS - 1
#define LEAF_VERTS 6            // three rows of two: base, middle (where it bends down), tip
#define LEAF_TRIS 4
#define FLOWER_LEAVES 4         // two opposite pairs
#define CAP_SPOKES 8            // bloom cap: centre, a ring halfway out, the rim
#define CAP_VERTS (1 + 2 * CAP_SPOKES)
#define CAP_TRIS (3 * CAP_SPOKES)
#define SPIKE_LEAVES 2
#define SPIKE_SEGMENTS 4        // per spike strip; two strips crossed at right angles
#define SPIKE_STRIP_VERTS (2 * (SPIKE_SEGMENTS + 1))
#define SPIKE_STRIP_TRIS (2 * SPIKE_SEGMENTS)
#define FLOWER_VERTS (STEM_VERTS + FLOWER_LEAVES * LEAF_VERTS + CAP_VERTS)                 // 50
#define FLOWER_TRIS (STEM_TRIS + FLOWER_LEAVES * LEAF_TRIS + CAP_TRIS)                     // 47
#define SPIKE_VERTS (STEM_VERTS + SPIKE_LEAVES * LEAF_VERTS + 2 * SPIKE_STRIP_VERTS)       // 41
#define SPIKE_TRIS (STEM_TRIS + SPIKE_LEAVES * LEAF_TRIS + 2 * SPIKE_STRIP_TRIS)           // 31
#define FERN_FRONDS 10
#define FROND_SEGMENTS 6
#define FROND_VERTS (3 * (FROND_SEGMENTS + 1))  // rows of three: left edge, rachis, right edge (cupped)
#define FROND_TRIS (4 * FROND_SEGMENTS)
#define FERN_VERTS (FERN_FRONDS * FROND_VERTS)  // 210
#define FERN_TRIS (FERN_FRONDS * FROND_TRIS)    // 240

// Atlas layout (build_flowers.py; UV v down): bloom from above centred at (0.25, 0.25) with radius 0.24; the leaf
// in the top-right cell from its base at v 0.49 up to its tip at v 0.01 around u 0.75; the spike in the left half
// from its base at v 0.99 to its tip at v 0.01 around u 0.25, a quarter as wide as it is tall.
#define BLOOM_UV float2(0.25, 0.25)
#define BLOOM_UV_RADIUS 0.245
#define LEAF_U 0.75
#define LEAF_HALF_U 0.11
#define SPIKE_U 0.25
#define SPIKE_HALF_U 0.125
#define SPIKE_FROM 0.35         // the spike strips cover the stem from here to its tip
// Ferns: one frond filling the atlas, base at v 0.99, tip at v 0.01, |u - 0.5| < FROND_HALF_U
#define FROND_HALF_U 0.245
#define FROND_WIDTH 0.85        // the atlas frond is half as wide as long; a male fern's less: squeeze it a little
#define FROND_CUP 0.18          // the frond's edges stand this share of its half-width above the rachis

struct PlantVertexData
{
    float3 WorldPos;
    float3 Normal;
    float3 Color;           // flat colour, or the tint the atlas texel is multiplied by
    float2 UV;
    uint   Textured;        // 1 = sample the parts atlas
};

struct Stem
{
    float3 Root;
    float3 Up;
    float3 Dir;             // the way the stem leans
    float  Height;
    float  Lean;
    float  Yaw;
    float  Phase;
    float  Width;
    float  Brightness;
    uint   Hash;            // more randomness for the stem's parts
};

Stem MakeStem(DecoInstance di, DecoratorSlot slot, uint stem, float dist)
{
    uint h = pcg(asuint(di.InstanceSeed) ^ (stem * 0x9E3779B9u) ^ 0x51ED270Bu);
    Stem s;
    float2 disc = float2(Rand(h), Rand(h));
    float rootAngle = disc.y * 6.2831853;
    s.Root = di.Position + float3(cos(rootAngle), 0, sin(rootAngle)) * 0.4 * di.Scale.x * sqrt(disc.x);
    s.Up = normalize(lerp(float3(0, 1, 0), di.TerrainNormal, abs(slot.SlopeBias)));
    s.Height = di.Scale.y * lerp(0.6, 1.0, Rand(h));
    s.Yaw = Rand(h) * 6.2831853;
    s.Dir = float3(cos(s.Yaw), 0, sin(s.Yaw));
    s.Lean = lerp(0.04, 0.22, Rand(h));
    s.Phase = Rand(h) * 6.2831853;
    s.Width = max(0.004, dist * 0.0009);
    s.Brightness = lerp(0.8, 1.1, Rand(h));
    s.Hash = h;
    return s;
}

float3 StemPoint(Stem s, float t)
{
    return s.Root + s.Up * s.Height * t * (1.0 - 0.3 * s.Lean * t) + s.Dir * s.Height * s.Lean * t * t
         + Wind(s.Root, t, s.Height, s.Phase);
}

float3 StemTangent(Stem s, float t)
{
    return normalize(s.Up * (1.0 - 0.6 * s.Lean * t) + s.Dir * 2.0 * s.Lean * t);
}

// Vertex 'v' of a leaf: rows at its base, middle and tip, each as wide as the atlas' leaf strip (the texture's
// alpha draws the outline); it leaves the stem along 'outDir' and droops towards the tip
void LeafVertex(inout PlantVertexData o, uint v, float3 base, float3 outDir, float3 up, float len, float3 tint)
{
    float3 side = normalize(cross(up, outDir));
    float t = float(v / 2) * 0.5;
    float sideSign = (v & 1) ? 1.0 : -1.0;
    float halfWidth = len * LEAF_HALF_U / 0.48;
    o.WorldPos = base + outDir * len * t - up * len * 0.35 * t * t + side * sideSign * halfWidth;
    o.Normal = normalize(cross(side, outDir));
    o.Color = tint;
    o.UV = float2(LEAF_U + sideSign * LEAF_HALF_U, 0.49 - 0.48 * t);
    o.Textured = 1;
}

// Vertex 'vi' of a fern: FERN_FRONDS fronds from a small crown, spread unevenly round it. Each rises at 32..75
// degrees (most low and long, a few young ones standing up in the middle) and arches over (its elevation falls
// linearly along it, so its tip points down), is rolled a little about its rachis and cupped (edges higher than the
// rachis). The crown leans with the slope and a little its own way. The first version rose at 52..80 degrees with
// every axis near vertical: every plant the same upright shuttlecock. The frond's length is the instance height;
// its width the atlas' squeezed by FROND_WIDTH.
PlantVertexData FernVertex(DecoInstance di, DecoratorSlot slot, uint vi)
{
    uint frond = vi / FROND_VERTS;
    uint v = vi % FROND_VERTS;
    uint row = v / 3;
    uint column = v % 3;                    // 0 left edge, 1 rachis, 2 right edge
    uint h = pcg(asuint(di.InstanceSeed) ^ (frond * 0x9E3779B9u) ^ 0x2F6B3A15u);

    // The crown's axis: between straight up and the slope (SlopeBias), leaning its own way by up to ~17 degrees (a
    // hash of the instance only, so all fronds of a plant share it)
    uint hp = pcg(asuint(di.InstanceSeed) ^ 0x7A3C51E9u);
    float leanAngle = Rand(hp) * 6.2831853;
    float3 up = normalize(lerp(float3(0, 1, 0), di.TerrainNormal, abs(slot.SlopeBias))
                          + float3(cos(leanAngle), 0, sin(leanAngle)) * 0.3 * Rand(hp));
    float azimuth = di.Rotation + (frond + 0.9 * Rand(h)) * (6.2831853 / FERN_FRONDS);
    float3 out0 = float3(cos(azimuth), 0, sin(azimuth));
    float3 outDir = normalize(out0 - up * dot(out0, up));
    float young = Rand(h);                             // 1 = a young frond: shorter, standing up in the middle
    float len = di.Scale.y * lerp(1.1, 0.55, young) * lerp(0.9, 1.1, Rand(h));
    float rise = lerp(0.55, 1.3, young * young);       // radians above the ground at the base: most fronds spread
    float bend = lerp(0.9, 1.7, Rand(h));              // how far the elevation falls to the tip
    float roll = lerp(-0.35, 0.35, Rand(h));
    float phase = Rand(h) * 6.2831853;
    float brightness = lerp(0.85, 1.1, Rand(h));
    float3 root = di.Position + outDir * 0.05 * di.Scale.y;

    // Walk the rachis to this row: segment k points at elevation rise - bend * (k + 0.5) / segments
    float3 p = root;
    float segLen = len / FROND_SEGMENTS;
    for (uint k = 0; k < row; k++)
    {
        float e = rise - bend * (k + 0.5) / FROND_SEGMENTS;
        p += (outDir * cos(e) + up * sin(e)) * segLen;
    }
    float t = float(row) / FROND_SEGMENTS;
    float eRow = rise - bend * t;
    float3 tangent = outDir * cos(eRow) + up * sin(eRow);
    float3 side = normalize(cross(up, outDir));
    float3 normal = normalize(cross(tangent, side));
    float3 across = side * cos(roll) + normal * sin(roll);
    normal = normalize(cross(tangent, across));

    float halfWidth = len * FROND_HALF_U / 0.98 * FROND_WIDTH;
    float sideSign = float(column) - 1.0;              // -1, 0, 1
    p += across * sideSign * halfWidth + normal * abs(sideSign) * halfWidth * FROND_CUP;
    p += Wind(root, t, len * 0.6, phase);

    PlantVertexData o;
    o.WorldPos = p;
    o.Normal = normalize(normal - across * sideSign * FROND_CUP * 2.0);
    // Darker in the crown; old (long, low) fronds a little yellower and duller than the young ones in the middle
    float3 age = lerp(float3(1.12, 1.0, 0.78), float3(0.95, 1.0, 1.0), young);
    o.Color = slot.HealthyColor * age * brightness * lerp(0.55, 1.0, saturate(t / 0.4));
    o.UV = float2(0.5 + sideSign * FROND_HALF_U, 0.99 - 0.98 * t);
    o.Textured = 1;
    if (slot.PlantPartsIdx == 0)
    {
        StructuredBuffer<float4> colors = ResourceDescriptorHeap[BladeColorsSRV];
        float4 foliage = colors[di.SlotIdx * COLORS_PER_VARIANT + COLOR_FOLIAGE];
        if (foliage.a <= 0) foliage = colors[di.SlotIdx * COLORS_PER_VARIANT];
        o.Color *= foliage.rgb;
        o.Textured = 0;
    }
    return o;
}

// Triangle 'ti' of a fern: per frond, per segment two quads (left edge to rachis, rachis to right edge)
uint3 FernTriangle(uint ti)
{
    uint vb = (ti / FROND_TRIS) * FROND_VERTS;
    uint t = ti % FROND_TRIS;
    uint row = t / 4;
    uint quad = (t % 4) / 2;                          // 0 left half, 1 right half
    uint a = vb + row * 3 + quad;                     // this row's vertex on the quad's left
    uint b = a + 3;                                   // the next row's
    return (t & 1) ? uint3(a + 1, b + 1, b) : uint3(a, a + 1, b);
}

// Vertex 'vi' of a flower or spike plant: per stem the stalk, the leaves, then the bloom cap or the spike strips
PlantVertexData PlantVertex(DecoInstance di, DecoratorSlot slot, uint vi, float dist)
{
    if (di.Mode == MODE_FERNS)
        return FernVertex(di, slot, vi);

    bool spikes = di.Mode == MODE_SPIKES;
    uint vertsPerStem = spikes ? SPIKE_VERTS : FLOWER_VERTS;
    uint v = vi % vertsPerStem;
    Stem s = MakeStem(di, slot, vi / vertsPerStem, dist);

    StructuredBuffer<float4> colors = ResourceDescriptorHeap[BladeColorsSRV];
    float4 foliage = colors[di.SlotIdx * COLORS_PER_VARIANT + COLOR_FOLIAGE];
    float4 bloom = colors[di.SlotIdx * COLORS_PER_VARIANT + COLOR_BLOOM];
    if (foliage.a <= 0) foliage = colors[di.SlotIdx * COLORS_PER_VARIANT];     // no green texels: the root band
    float3 tint = slot.HealthyColor * s.Brightness;
    bool atlas = slot.PlantPartsIdx != 0;

    PlantVertexData o;
    o.UV = 0;
    o.Textured = 0;
    if (v < STEM_VERTS)
    {
        bool tip = v == 2 * STEM_SEGMENTS;
        float t = tip ? 1.0 : float(v / 2) / STEM_SEGMENTS;
        float sideSign = tip ? 0.0 : ((v & 1) ? 1.0 : -1.0);
        float3 side = float3(-s.Dir.z, 0, s.Dir.x);
        o.WorldPos = StemPoint(s, t) + side * sideSign * 0.5 * s.Width * (1.0 - 0.4 * t);
        o.Normal = normalize(cross(side, StemTangent(s, t)) + side * sideSign * 0.6);
        o.Color = foliage.rgb * tint * lerp(0.6, 1.0, saturate(t / 0.3));
        return o;
    }
    v -= STEM_VERTS;

    uint leaves = spikes ? SPIKE_LEAVES : FLOWER_LEAVES;
    if (v < leaves * LEAF_VERTS)
    {
        // Flowers: opposite pairs at 30% and 55% of the stem, the second pair turned 90°; spikes: two long leaves low
        uint leaf = v / LEAF_VERTS;
        float t = spikes ? (leaf == 0 ? 0.18 : 0.3) : (leaf < 2 ? 0.3 : 0.55);
        float azimuth = s.Yaw + (leaf & 1) * 3.14159265 + (leaf >> 1) * 1.5707963 + 0.6;
        float elevation = spikes ? 0.9 : 0.6;                                   // radians above horizontal
        float3 outDir = normalize(float3(cos(azimuth), 0, sin(azimuth)) * cos(elevation) + s.Up * sin(elevation));
        LeafVertex(o, v % LEAF_VERTS, StemPoint(s, t), outDir, s.Up, s.Height * (spikes ? 0.32 : 0.22), tint);
        if (!atlas)
        {
            o.Color = foliage.rgb * tint;
            o.Textured = 0;
        }
        return o;
    }
    v -= leaves * LEAF_VERTS;

    if (spikes)
    {
        // Two strips crossed at right angles along the top of the stem, a quarter as wide as long, following its
        // bend and sway
        uint strip = v / SPIKE_STRIP_VERTS;
        uint sv = v % SPIKE_STRIP_VERTS;
        float k = float(sv / 2) / SPIKE_SEGMENTS;
        float sideSign = (sv & 1) ? 1.0 : -1.0;
        float t = lerp(SPIKE_FROM, 1.0, k);
        float halfWidth = s.Height * (1.0 - SPIKE_FROM) * SPIKE_HALF_U / 0.98;
        float3 across = strip == 0 ? float3(-s.Dir.z, 0, s.Dir.x) : s.Dir;
        o.WorldPos = StemPoint(s, t) + across * sideSign * halfWidth;
        o.Normal = normalize(cross(across, StemTangent(s, t)));
        o.Color = atlas ? tint : bloom.rgb * tint;
        o.UV = float2(SPIKE_U + sideSign * SPIKE_HALF_U, 0.99 - 0.98 * k);
        o.Textured = atlas ? 1 : 0;
        return o;
    }

    // Flowers: the bloom cap at the tip, each turned its own way between straight up and well over to the side,
    // its petals cupped up towards the rim
    float3 top = StemPoint(s, 1.0);
    uint hb = s.Hash;
    float3 facing = normalize(s.Up + s.Dir * lerp(0.2, 1.1, Rand(hb)));
    float3 u = normalize(cross(facing, abs(facing.y) < 0.99 ? float3(0, 1, 0) : float3(1, 0, 0)));
    float3 w = cross(facing, u);
    float radius = clamp(s.Height * 0.1, 0.02, 0.045) * lerp(0.75, 1.15, Rand(hb));
    float spin = Rand(hb) * 6.2831853;
    float ring = v == 0 ? 0.0 : (v <= CAP_SPOKES ? 0.5 : 1.0);
    float a = spin + (v == 0 ? 0.0 : float((v - 1) % CAP_SPOKES)) * (6.2831853 / CAP_SPOKES);
    float2 dir2 = float2(cos(a), sin(a));
    o.WorldPos = top + (u * dir2.x + w * dir2.y) * radius * ring + facing * radius * (0.05 * ring + 0.12 * ring * ring);
    o.Normal = facing;
    o.Color = atlas ? tint : bloom.rgb * tint;
    o.UV = BLOOM_UV + float2(dir2.x, -dir2.y) * BLOOM_UV_RADIUS * ring;
    o.Textured = atlas ? 1 : 0;
    return o;
}

// Triangle 'ti' of a plant: per stem the stalk, the leaves (two quads each), then the cap or the spike strips
uint3 PlantTriangle(uint mode, uint ti)
{
    if (mode == MODE_FERNS)
        return FernTriangle(ti);
    bool spikes = mode == MODE_SPIKES;
    uint trisPerStem = spikes ? SPIKE_TRIS : FLOWER_TRIS;
    uint t = ti % trisPerStem;
    uint vb = (ti / trisPerStem) * (spikes ? SPIKE_VERTS : FLOWER_VERTS);
    if (t < STEM_TRIS)
        return StripTriangle(vb, t, STEM_SEGMENTS);
    t -= STEM_TRIS;
    vb += STEM_VERTS;

    uint leaves = spikes ? SPIKE_LEAVES : FLOWER_LEAVES;
    if (t < leaves * LEAF_TRIS)
    {
        uint l = vb + (t / LEAF_TRIS) * LEAF_VERTS + ((t % LEAF_TRIS) / 2) * 2;
        return (t & 1) ? uint3(l + 1, l + 3, l + 2) : uint3(l, l + 1, l + 2);
    }
    t -= leaves * LEAF_TRIS;
    vb += leaves * LEAF_VERTS;

    if (spikes)
    {
        uint l = vb + (t / SPIKE_STRIP_TRIS) * SPIKE_STRIP_VERTS + ((t % SPIKE_STRIP_TRIS) / 2) * 2;
        return (t & 1) ? uint3(l + 1, l + 3, l + 2) : uint3(l, l + 1, l + 2);
    }

    // Cap: a fan from the centre to the inner ring, then quads from the inner ring to the rim
    if (t < CAP_SPOKES)
        return uint3(vb, vb + 1 + t, vb + 1 + (t + 1) % CAP_SPOKES);
    t -= CAP_SPOKES;
    uint spoke = t / 2;
    uint next = (spoke + 1) % CAP_SPOKES;
    uint i0 = vb + 1 + spoke, i1 = vb + 1 + next;
    uint o0 = vb + 1 + CAP_SPOKES + spoke, o1 = vb + 1 + CAP_SPOKES + next;
    return (t & 1) ? uint3(i1, o1, o0) : uint3(i0, i1, o0);
}

// What one MS group outputs for an instance: flowers, spikes and ferns only (blade clumps and padding: nothing)
void PlantCounts(DecoInstance di, out uint verts, out uint prims)
{
    verts = 0;
    prims = 0;
    if (di.Scale.y <= 0) return;
    if (di.Mode == MODE_FLOWERS) { verts = PLANT_STEMS * FLOWER_VERTS; prims = PLANT_STEMS * FLOWER_TRIS; }
    else if (di.Mode == MODE_SPIKES) { verts = PLANT_STEMS * SPIKE_VERTS; prims = PLANT_STEMS * SPIKE_TRIS; }
    else if (di.Mode == MODE_FERNS) { verts = FERN_VERTS; prims = FERN_TRIS; }
}

// ─── Mesh Shader ───────────────────────────────────────────────────────────

struct MSOutput
{
    float4 Position    : SV_POSITION;
    float3 Color       : COLOR0;
    float3 PartNormal  : NORMAL;
    nointerpolation float3 TerrainNormal : TEXCOORD0;
    float  Depth       : TEXCOORD1;
    float2 UV          : TEXCOORD2;
    nointerpolation uint TextureIdx : TEXCOORD3;   // parts atlas, 0 = flat colour
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

    uint vertCount, primCount;
    PlantCounts(di, vertCount, primCount);
    SetMeshOutputCounts(vertCount, primCount);

    StructuredBuffer<DecoratorSlot> slots = ResourceDescriptorHeap[DecoratorSlotsIdx];
    DecoratorSlot slot = slots[di.SlotIdx];
    float dist = distance(CamPos, di.Position);
    float3 variation = GroundVariation(di.Position.xz) * UnpackGroundTint(di.GroundTint);   // large-scale variation (common.fx) x the ground at the root (spawn)

    for (uint vi = gtid; vi < vertCount; vi += MS_THREADS)
    {
        PlantVertexData pv = PlantVertex(di, slot, vi, dist);
        MSOutput o;
        o.Position = mul(float4(pv.WorldPos, 1.0), ViewProjection);
        o.Color = pv.Color * variation;
        o.PartNormal = pv.Normal;
        o.TerrainNormal = di.TerrainNormal;
        o.Depth = o.Position.w;
        o.UV = pv.UV;
        o.TextureIdx = pv.Textured ? slot.PlantPartsIdx : 0;
        verts[vi] = o;
    }

    for (uint ti = gtid; ti < primCount; ti += MS_THREADS)
        tris[ti] = PlantTriangle(di.Mode, ti);
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

PSOutput PS(MSOutput input, bool isFrontFace : SV_IsFrontFace)
{
    float3 color = input.Color;
    if (input.TextureIdx != 0)
    {
        Texture2D parts = ResourceDescriptorHeap[NonUniformResourceIndex(input.TextureIdx)];
        // Half a mip sharper and a fixed cutoff: a lower cutoff with distance (as the cards) filled the gaps between
        // a fern's pinnae, and the fronds read as broad smooth leaves
        float4 texel = parts.SampleBias(ClampSampler, input.UV, -0.5);
        clip(texel.a - 0.5);
        color *= texel.rgb;
    }

    // As the blades: mostly the ground's normal, some of the part's own facing; back faces flip it
    float3 partN = normalize(input.PartNormal) * (isFrontFace ? 1.0 : -1.0);
    float3 N = normalize(lerp(input.TerrainNormal, partN, 0.35));

    PSOutput output;
    output.Albedo = float4(color, 0);
    output.Normal = float4(N, 1.0);
    output.Data = float4(0.9, 0.0, 1.0, 0.5);    // matte, no metal, ao 1, vegetation lighting
    output.Depth = input.Depth;
    output.EntityId = 0;
    output.Displacement = float2(0, 0);
    return output;
}

// ─── Shadow ────────────────────────────────────────────────────────────────
// Cascade 0 only, like the cards and blades (AS_Shadow launches only the plants inside it); textured parts cut
// their outline into the shadow too.

struct ShadowMSOutput
{
    float4 Position : SV_POSITION;
    float2 UV       : TEXCOORD0;
    nointerpolation uint TextureIdx : TEXCOORD1;
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
    in payload ProceduralShadowPayload payload,
    out vertices ShadowMSOutput verts[MAX_VERTS],
    out indices uint3 tris[MAX_PRIMS],
    out primitives ShadowPrimAttribs primAttribs[MAX_PRIMS])
{
    StructuredBuffer<DecoInstance> instances = ResourceDescriptorHeap[BladeInstanceSRV];
    DecoInstance di = instances[payload.Instances[gid.x]];

    uint vertCount, primCount;
    PlantCounts(di, vertCount, primCount);
    SetMeshOutputCounts(vertCount, primCount);

    StructuredBuffer<DecoratorSlot> slots = ResourceDescriptorHeap[DecoratorSlotsIdx];
    StructuredBuffer<CascadeData> cascades = ResourceDescriptorHeap[CascadeBufferSRVIdx];
    DecoratorSlot slot = slots[di.SlotIdx];
    row_major float4x4 cascadeVP = cascades[0].VP;
    float dist = distance(CamPos, di.Position);

    for (uint vi = gtid; vi < vertCount; vi += MS_THREADS)
    {
        PlantVertexData pv = PlantVertex(di, slot, vi, dist);
        ShadowMSOutput o;
        o.Position = mul(float4(pv.WorldPos, 1.0), cascadeVP);
        o.UV = pv.UV;
        o.TextureIdx = pv.Textured ? slot.PlantPartsIdx : 0;
        verts[vi] = o;
    }

    for (uint ti = gtid; ti < primCount; ti += MS_THREADS)
    {
        tris[ti] = PlantTriangle(di.Mode, ti);
        primAttribs[ti].RTIndex = 0;
    }
}

void PS_Shadow(ShadowMSOutput input)
{
    if (input.TextureIdx != 0)
    {
        Texture2D parts = ResourceDescriptorHeap[NonUniformResourceIndex(input.TextureIdx)];
        clip(parts.SampleLevel(ClampSampler, input.UV, 0).a - 0.5);
    }
}

// ─── Technique ─────────────────────────────────────────────────────────────

// @RenderState(RenderTargets=4, CullMode=None)
technique11 GrassPlants
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
