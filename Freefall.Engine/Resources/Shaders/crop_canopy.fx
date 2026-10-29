// Crop canopy (CropCanopy): the far look of a field of crops. A low mesh of the field's top surface and its edge,
// built by the component from the DecoStamp that sows the field: every vertex at the canopy's height over the ground's
// crests, u = -1 - how far the crests stand over the ground, v = 1 + shell (+ 0.5 at the edge's foot, which stays on
// the ground); the edge slopes from under the top's border down to the ground past the outline, its normal leaning out.
//
// Near the camera every top vertex sinks below the ground (CANOPY_SINK). It rises as the crop's plants thin out with
// distance (grass_compute.hlsl CS_SpawnInstances: THIN_PLANTS_PER_SR plants of a variant per steradian, each kept plant
// grown to cover for the dropped ones, KeptCover, all of them gone over the last 30% of the range), so the canopy stands
// where the plants stop covering the field, porous where it comes up, the ones still standing poke out of it, and from
// afar the field keeps its height, its edge and the shadow that edge throws. The shadow passes lift the same way (CamPos
// is the main camera's there too), so the canopy's shadow is the canopy's.
//
// Surface: the plants' own colours (PlantColour) times what varies over metres, per vertex (CanopyLook: the ground and
// its relief, the plants' tall and short patches, lodging, patches, the land's variation, gusts); per pixel shells of
// tufts, streaks along the way the field was sown and rows of plants over dark soil, each faded where a pixel spans it.

cbuffer PushConstants : register(b3)
{
    uint _reserved0;
    uint _reserved1;
    uint MeshPartId;                // 2: Index into MeshRegistry
    uint InstanceBaseOffset;        // 3: Base offset for instance ID (per-command)
    uint DescriptorBufIdx;          // 4: StructuredBuffer<InstanceDescriptor>
    uint SortedIndicesIdx;          // 5: StructuredBuffer<uint> - sorted draw order indices
    uint MeshRegistryIdx;           // 6: StructuredBuffer<MeshPartEntry>
    uint MaterialsIdx;              // 7: Index to materials buffer
    uint GlobalTransformBufferIdx;  // 8: Index to global TransformBuffer
    uint _reserved9;
    uint _reserved10;
    uint _reserved11;
    uint _reserved12;
    uint _reserved13;
    uint _reserved14;
    uint _reserved15;
    uint DebugMode;                 // 16: Debug visualization mode
    uint IndexedDraw;               // 17: 1 = an indexed draw, the vertex index is SV_VertexID (common.fx MeshVertexIndex)
    uint _reserved18;
    uint _reserved19;
    uint ExpansionBufferIdx;        // 20: SRV: shadow expansion buffer
    uint CascadeBufferSRVIdx;       // 21: SRV: StructuredBuffer<CascadeData>
};

#include "common.fx"
#include "decoration_noise.hlsli"
#include "ground_shape.hlsli"
// Both sides: a field's far edge slopes away from a low eye, and its tufts make the far edge's ragged silhouette
// @RenderState(RenderTargets=6, CullMode=None)

SamplerState SamplerClamp : register(s2);   // linear, clamp

// The spawn's thinning (grass_compute.hlsl THIN_PLANTS_PER_SR, THIN_MIN_COS) and how far a kept plant grows to cover
// for the dropped ones (grass_plants.fx PLANT_MAX_GROWTH, grass_blades.fx BLADE_MAX_GROWTH)
#define THIN_PLANTS_PER_SR 20000.0
#define THIN_MIN_COS 0.02
#define THIN_MAX_GROWTH 16.0

#define CANOPY_SINK 0.3     // m below the ground a sunken top vertex lies (CropCanopy.Sink)
#define EDGE_TUCK 0.03      // m the edge's inner ring lies under the top's border (CropCanopy.EdgeTuck tucks it in)
#define EDGE_FOOT 0.05      // m under the ground the edge's foot lies

// Shells: the top is drawn CANOPY_SHELLS times (CropCanopy.Shells; uv.y 1 + shell), from SHELL_DEPTH of the plants'
// height below the surface (shell 0, solid) up to it, each above the lowest only where a tuft reaches that high
// (ShellTuft). Near enough to see tufts the crop has depth: tufts stand in front of each other and its far edge is
// ragged against what lies behind. Where a pixel spans more than about SHELL_TUFT the shells close up into one surface
// (they would only shimmer), and only the lowest throws a shadow.
#define CANOPY_SHELLS 4
#define SHELL_DEPTH 0.45
#define SHELL_TUFT 0.2      // m: the tufts' size

struct Instance
{
    float  Lift;              //= 1.3      -- Canopy height above the ground + CANOPY_SINK, m (set by CropCanopy)
    float  Density;           //= 50       -- The crop's plants per square metre at full cover (set by CropCanopy)
    float  Range;             //= 140      -- How far the crop's plants are drawn, m (set by CropCanopy)
    uint   TintBuffer;        //= 0        -- The terrain's decorator far colours, TerrainRenderer.DecoratorTintSrv (set by CropCanopy)
    uint   Group;             //= 0        -- The crop decorator's slot on the terrain (set by CropCanopy)
    float2 TerrainCorner;     //= 0, 0     -- World XZ of the terrain's corner (set by CropCanopy)
    uint   Heightmap;         //= 0        -- The terrain's heightmap (set by CropCanopy)
    float2 TerrainSize;       //= 4096, 4096 -- The terrain's size, m (set by CropCanopy)
    float  MaxHeight;         //= 600      -- The terrain's height range, m (set by CropCanopy)
    float2 RowAcross;         //= 0, 0     -- Unit vector across the way the field was sown, rows or not (set by CropCanopy)
    float  RowSpacing;        //= 0        -- m between rows, 0 = sown broadcast (set by CropCanopy)
    float  RowCover;          //= 1        -- Share of a row's spacing the plants cover (set by CropCanopy)
    float  HeightNoiseScale;  //= 0        -- The crop variant's HeightNoiseScale (set by CropCanopy)
    float  HeightNoiseAmount; //= 0        -- The crop variant's HeightNoiseAmount (set by CropCanopy)
    uint   Seed;              //= 0        -- The crop variant's Seed (set by CropCanopy)
    float  Lodging;           //= 0        -- Share of the field lying flat, beaten down by rain and wind: grain (set by CropCanopy)
    float  KeptCover;         //= 16       -- How many dropped plants a kept one covers for at most: 16 blades, 4 grain and flowers (set by CropCanopy)
    uint   BladeColors;       //= 0        -- The variants' colours, TerrainRenderer.BladeColorsSrv; 0 = the decorator's far colour (set by CropCanopy)
    uint   VariantSlot;       //= 0        -- The crop variant's slot in them (set by CropCanopy)
    float3 HealthyColor;      //= 1, 1, 1  -- The crop variant's HealthyColor (set by CropCanopy)
};

#define COLORS_PER_VARIANT 6    // CS_BladeColors (grass_compute.hlsl): 4 height bands root → tip, foliage, bloom

// The colour the crop's plants are drawn with at a height on them, 0 = the root .. 1 = the heads (vertex shader only:
// the colours rest in NonPixelShaderResource): the card's height bands up the stalk, and on top its heads, the
// foliage and the bloom by how much of the plant each is (a wheat atlas has no green, beans are mostly leaf). The
// decorator's far colour (the sward's) mixes stalks and heads: on the canopy it read greyer than the wheat beside it.
float3 PlantColour(Instance inst, float height)
{
    StructuredBuffer<float4> colors = ResourceDescriptorHeap[inst.BladeColors];
    uint at = inst.VariantSlot * COLORS_PER_VARIANT;
    float4 foliage = colors[at + 4], bloom = colors[at + 5];
    // (the bloom by the square root of its share: blooms sit on top and catch the eye, and by area alone a flowering
    // flax field went plain green)
    float3 heads = lerp(foliage.rgb, bloom.rgb, sqrt(bloom.a / max(foliage.a + bloom.a, 1e-4)));
    float3 stalk = height < 0.5 ? lerp(colors[at].rgb, colors[at + 1].rgb, height * 2.0)
                                : lerp(colors[at + 1].rgb, colors[at + 2].rgb, height * 2.0 - 1.0);
    return lerp(stalk, heads, smoothstep(0.45, 0.95, height)) * inst.HealthyColor;
}

Instance LoadInstance(uint slot)
{
    ByteAddressBuffer all = ResourceDescriptorHeap[InstanceDataIdx];
    return all.Load<Instance>(slot * 128u);
}

// The plants' tall and short patches at a terrain-local point, as the spawn sizes them (grass_compute.hlsl)
float PlantHeightFactor(float2 local, Instance inst)
{
    if (inst.HeightNoiseScale <= 0.0 || inst.HeightNoiseAmount <= 0.0) return 1.0;
    return DecoHeightFactor(local, inst.HeightNoiseScale, inst.HeightNoiseAmount, inst.Seed);
}

// How much of the crop lies flat at a terrain-local point, 0..1: patches metres across, about Lodging of the field
// (the fBm's spread is ~0.1 round 0.5, so its top tenth starts near 0.61)
float LodgedAt(float2 local, Instance inst)
{
    if (inst.Lodging <= 0.0) return 0.0;
    float edge = 0.5 + (0.5 - inst.Lodging) * 0.27;
    return smoothstep(edge - 0.03, edge + 0.04, DecoFbm(local, 9.0, float(inst.Seed) + 57.0));
}

// How far the canopy has risen at a point of the field's ground: 0 where the plants still cover the field, 1 where
// they show none of it. The share of plants the spawn keeps there on flat ground, times how many a kept plant covers
// for (KeptCover: the blades' 16, the grain's and flowers' 4), times the share left by the range's end.
float CanopyRise(float3 ground, Instance inst)
{
    float3 toCam = CamPos - ground;
    float distSq = max(dot(toCam, toCam), 1e-4);
    float dist = sqrt(distSq);
    float facing = max(abs(toCam.y) / dist, THIN_MIN_COS);
    float keep = THIN_PLANTS_PER_SR * facing / max(inst.Density * distSq, 1e-6);
    float endKeep = saturate((inst.Range - dist) / (0.3 * inst.Range));
    return 1.0 - saturate(keep * min(inst.KeptCover, THIN_MAX_GROWTH)) * endKeep;
}

// A canopy vertex where the camera has it: a top vertex from below the ground up to the plants' height there, by how
// far the canopy has risen. It comes out of the ground quickly, to under half the plants' height while they still
// cover most of the field (CANOPY_EMERGE), then rises with their thinning: a canopy rising slowly from the ground cut
// through every unevenness of it (the crests of ridge and furrow) in bands of soil.
#define CANOPY_EMERGE 0.12      // rise by which the canopy stands at CANOPY_EMERGED of the plants' height
#define CANOPY_EMERGED 0.45

// Metres a pixel spans on a surface facing the eye at a distance (the vertex shader's estimate; the pixel shader
// measures its footprint, PixelMetres)
float PixelMetresAt(float dist)
{
    return dist * 2.0 / (ViewportSize.y * Projection._22);
}

// How much the shells stand apart for a pixel's size in metres: fully while a tuft spans several pixels, not at all
// once it spans under ~1.5
float ShellDetail(float metresPerPixel)
{
    return saturate(1.5 - 2.0 * metresPerPixel / SHELL_TUFT);
}

// How high the crop's tufts reach at a terrain-local point, 0..1 (the shells keep what reaches them): tufts with ears
// on them, irregular
float ShellTuft(float2 local)
{
    float n = DecoGradientNoise(local / SHELL_TUFT + 23.3) * 0.6 + DecoGradientNoise(local / (0.35 * SHELL_TUFT) + 61.9) * 0.4;
    return saturate((n - 0.32) / 0.42);
}

// The canopy's surface over a point of the ground: below it where the plants still cover the field, out to
// CANOPY_EMERGED of their height by CANOPY_EMERGE, then up to their height ('tall') as they go. 'lift' = how far out.
float CanopySurface(float3 ground, float crests, Instance inst, out float rise, out float tall, out float lift)
{
    float height = inst.Lift - CANOPY_SINK;
    float2 local = ground.xz - inst.TerrainCorner;
    tall = height * PlantHeightFactor(local, inst) * (1.0 - 0.45 * LodgedAt(local, inst));
    rise = CanopyRise(ground, inst);
    lift = smoothstep(0.0, CANOPY_EMERGE, rise);
    return lerp(ground.y - CANOPY_SINK, crests + CANOPY_EMERGED * tall, lift)
         + (1.0 - CANOPY_EMERGED) * tall * saturate((rise - CANOPY_EMERGE) / (1.0 - CANOPY_EMERGE));
}

// Which shell a vertex or pixel is on and how far down the edge towards its foot (0 on the top and the edge's inner
// ring, 1 at the foot): the mesh's uv.y is 1 + shell, + 0.5 at the edge's foot
void ShellOf(float v, out float shell, out float toFoot)
{
    float at = v - 1.0;
    shell = floor(at + 0.01);
    toFoot = saturate((at - shell) * 2.0);
}

// The mesh puts every vertex at the canopy's height over the ground's crests (CropCanopy.GroundGrid: far off the
// terrain bridges the furrows of ridge and furrow, and a canopy dipping into them let the ground through) and says in
// u how far those crests stand above the ground there (-1 - that), in v its shell and whether it is the edge's foot
// (ShellOf). The top and the edge's inner ring (its normal leans out: the top's is straight up) go to the canopy's
// surface, the inner ring just under the top's border; the edge's foot stays on the ground; each shell lies its share
// of the depth below, the depth gone where tufts are under a pixel. Out: how far the canopy has risen here.
float3 CanopyPosition(float3 world, float2 uv, float3 normal, Instance inst, out float rise)
{
    float height = inst.Lift - CANOPY_SINK;
    float crests = world.y - height;
    float groundY = crests - (-1.0 - uv.x);
    float shell, toFoot;
    ShellOf(uv.y, shell, toFoot);
    bool edge = normal.y < 0.999;
    float tall, lift;
    float surface = CanopySurface(float3(world.x, groundY, world.z), crests, inst, rise, tall, lift);
    world.y = toFoot > 0.5 ? groundY - EDGE_FOOT : surface - (edge ? EDGE_TUCK : 0.0);
    float depth = SHELL_DEPTH * tall * lift * ShellDetail(PixelMetresAt(distance(world, CamPos)));
    world.y -= depth * (1.0 - shell / (CANOPY_SHELLS - 1.0));
    return world;
}

// How much of the canopy shows where it has risen this far: it closes while the plants still cover most of the field,
// so its gaps sit among them. (Closing over the whole thinning, its holes showed the shaded ground under it: black
// polka dots on a pale sheet.)
float CanopyOpacity(float rise)
{
    return smoothstep(0.0, 0.3, rise);
}

// Whether a point of the canopy shows (>= 0) or is a gap between the plants: world-space holes the size of tufts and
// ears, irregular over three scales, so where it comes up it is a scatter of specks among the plants and closes into a
// surface as they thin. (An opaque canopy rising out of the ground drew a hard line across the plants.)
float CanopyCoverage(float2 p, float opacity)
{
    float n = DecoGradientNoise(p / 0.11 + 5.1) * 0.5 + DecoGradientNoise(p / 0.045 + 2.7) * 0.3
            + DecoGradientNoise(p / 0.4 + 8.9) * 0.2;
    return opacity * 1.02 - 0.01 - smoothstep(0.3, 0.7, n);
}

struct Fetched
{
    float3 World;
    float3 Normal;
    float2 UV;
    float Rise;
    uint TransformSlot;
    uint InstanceSlot;
    uint MeshPartIdx;
};

Fetched FetchVertex(uint primitiveVertexID, InstanceDescriptor desc)
{
    StructuredBuffer<row_major matrix> globalTransforms = ResourceDescriptorHeap[GlobalTransformBufferIdx];
    StructuredBuffer<MeshPartEntry> meshRegistry = ResourceDescriptorHeap[MeshRegistryIdx];
    MeshPartEntry part = meshRegistry[MeshPartId];
    uint vertexID = MeshVertexIndex(part, primitiveVertexID, primitiveVertexID, IndexedDraw);
    StructuredBuffer<float3> positions = ResourceDescriptorHeap[part.PosBufferIdx];
    StructuredBuffer<float3> normals = ResourceDescriptorHeap[part.NormBufferIdx];
    StructuredBuffer<float2> uvs = ResourceDescriptorHeap[part.UVBufferIdx];

    row_major matrix World = globalTransforms[desc.TransformSlot];
    Fetched f;
    f.UV = uvs[vertexID];
    f.World = mul(float4(positions[vertexID], 1.0), World).xyz;
    f.Normal = normalize(mul(normals[vertexID], (float3x3)World));
    f.TransformSlot = desc.TransformSlot;
    f.InstanceSlot = desc.CustomDataIdx;
    f.MeshPartIdx = desc.MeshPartIdx;
    f.World = CanopyPosition(f.World, f.UV, f.Normal, LoadInstance(desc.CustomDataIdx), f.Rise);
    return f;
}

// The ground's slope at a terrain-local point over 'reach' heightmap texels either side (central differences, as
// gputerrain.fx GetNormal takes them over one): metres down per metre along x and z
float2 GroundSlope(Texture2D heights, float2 uv, float2 texel, float reach, Instance inst)
{
    float2 d = texel * reach;
    float hl = heights.SampleLevel(SamplerClamp, uv - float2(d.x, 0), 0).r;
    float hr = heights.SampleLevel(SamplerClamp, uv + float2(d.x, 0), 0).r;
    float hb = heights.SampleLevel(SamplerClamp, uv - float2(0, d.y), 0).r;
    float hf = heights.SampleLevel(SamplerClamp, uv + float2(0, d.y), 0).r;
    return float2(hl - hr, hb - hf) * inst.MaxHeight / (2.0 * inst.TerrainSize * d);
}

// The ground under the crop: its normal, with the relief of a few metres (ridge and furrow) exaggerated, since the
// vegetation lighting answers a third as strongly as the ground's to a turned normal; and how far the point lies
// below (+) or above (-) the ground 3 m round it (ground_shape.hlsli GroundCavity: furrows +, crests -)
float3 GroundUnderCrop(float2 local, Instance inst, out float cavity)
{
    Texture2D heights = ResourceDescriptorHeap[NonUniformResourceIndex(inst.Heightmap)];
    uint w, h;
    heights.GetDimensions(w, h);
    float2 texel = 1.0 / float2(w, h);
    float2 uv = local / inst.TerrainSize;
    float2 broad = GroundSlope(heights, uv, texel, 4.0, inst);
    float2 fine = GroundSlope(heights, uv, texel, 1.0, inst);
    cavity = GroundCavity(heights, SamplerClamp, uv, texel.x, inst.MaxHeight, 3.0);
    float2 slope = broad + (fine - broad) * 2.5;
    return normalize(float3(slope.x, 1.0, slope.y));
}

// What varies over metres, worked out per vertex (the top's grid is 1 m) instead of per pixel of every shell: the
// land's variation and the sward's grain, the ground under the crop and its relief, the plants' tall and short
// patches, lodging, riper and greener patches and the wind's gusts. Tint multiplies the plants' colour; Standing = 1 -
// how flat the crop lies. On the edge the normal keeps its lean out (the slope), the ground's relief only tints it.
struct Look
{
    float3 Tint;
    float3 Normal;
    float Standing;
};

Look CanopyLook(float3 world, float3 normal, Instance inst)
{
    Look look;
    float2 local = world.xz - inst.TerrainCorner;
    float dist = distance(world, CamPos);
    bool edge = normal.y < 0.999;
    // (half the sward's grain: in full its soft clouds made the canopy read as felt from above)
    look.Tint = lerp(1.0.xxx, SwardGrain(local), 0.5)
              * lerp(1.0.xxx, GroundVariation(world.xz), lerp(0.5, 1.0, saturate(dist / 150.0)));
    look.Normal = normal;
    look.Standing = 1.0;

    // The ground's relief: the crop greener and darker in the furrows, drier and paler on the crests
    if (inst.Heightmap != 0)
    {
        float cavity;
        float3 ground = GroundUnderCrop(local, inst, cavity);
        if (!edge) look.Normal = ground;
        look.Tint *= GroundShapeTint(cavity * 2.0, 1.0);
    }

    // The slope of the plants' tall and short patches
    float height = inst.Lift - CANOPY_SINK;
    const float e = 0.75;
    float f0 = PlantHeightFactor(local, inst);
    float2 tilt = float2(f0 - PlantHeightFactor(local + float2(e, 0), inst), f0 - PlantHeightFactor(local + float2(0, e), inst)) * height / e;

    // Lodging: patches of grain beaten flat, lower (CanopySurface), all lying one way within a patch, which turns
    // slowly across the field: paler where they lie away from the eye (the stalks show), darker towards it (the heads);
    // flat, they lose the tufts and streaks (Standing). An even +20 % read as camouflage blotches.
    float lodged = LodgedAt(local, inst);
    if (lodged > 0.0)
    {
        float lie = 6.2831853 * DecoGradientNoise(local / 30.0 + 7.7);
        float2 lying = float2(cos(lie), sin(lie));
        tilt += lying * 0.9 * lodged;
        look.Tint *= 1.0 - 0.12 * lodged * dot(lying, normalize(CamPos.xz - world.xz + 1e-3));
    }
    look.Standing = 1.0 - 0.7 * lodged;

    // Riper and greener patches, with edges, a few tens of metres across
    float patch = smoothstep(0.44, 0.56, DecoFbm(local, 22.0, float(inst.Seed) + 31.0));
    look.Tint *= lerp(float3(1.06, 1.03, 0.93), float3(0.93, 1.0, 0.96), patch);

    // The wind running over the crop: where a gust presses it the heads bend downwind and show their paler sides (the
    // same gusts sway the plants near the camera, common.fx WindPush)
    float gust = saturate(WindStrength / 8.0) * (WindPush(world.xz) - 0.4);
    tilt -= WindDirection.xz * gust * 0.35;
    look.Tint *= 1.0 + 0.45 * gust;

    look.Normal = normalize(look.Normal + float3(tilt.x, 0, tilt.y));
    return look;
}

struct VSOutput
{
    float4 Position : SV_POSITION;
    float3 WorldPos : TEXCOORD0;
    float3 Normal : NORMAL;
    float2 UV : TEXCOORD1;
    float Depth : TEXCOORD2;
    nointerpolation uint TransformSlot : TEXCOORD3;
    nointerpolation uint MeshPartIdx : TEXCOORD4;
    nointerpolation uint InstanceSlot : TEXCOORD5;
    float Rise : TEXCOORD6;         // how far the canopy has risen here (CanopyRise)
    float4 Look : TEXCOORD7;        // CanopyLook: tint, standing
    float3 Plant : TEXCOORD8;       // PlantColour here (BladeColors set), else 0
};

VSOutput VS(uint primitiveVertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
    StructuredBuffer<InstanceDescriptor> descriptors = ResourceDescriptorHeap[DescriptorBufIdx];
    StructuredBuffer<uint> sortedIndices = ResourceDescriptorHeap[SortedIndicesIdx];
    InstanceDescriptor desc = descriptors[DrawEntryIndex(sortedIndices[InstanceBaseOffset + instanceID])];
    Fetched f = FetchVertex(primitiveVertexID, desc);
    Instance inst = LoadInstance(f.InstanceSlot);
    Look look = CanopyLook(f.World, f.Normal, inst);

    // Where on the plants this vertex shows: the lowest shell halfway up them, the upper shells their heads; down the
    // edge to its foot, the stalks of the short plants there
    float shell, toFoot;
    ShellOf(f.UV.y, shell, toFoot);
    float onPlant = lerp(0.55 + 0.45 * shell / (CANOPY_SHELLS - 1.0), 0.2, toFoot);

    VSOutput output;
    output.Position = mul(mul(float4(f.World, 1.0), View), Projection);
    output.WorldPos = f.World;
    output.Normal = look.Normal;
    output.UV = f.UV;
    output.Depth = output.Position.w;
    output.TransformSlot = f.TransformSlot;
    output.MeshPartIdx = f.MeshPartIdx;
    output.InstanceSlot = f.InstanceSlot;
    output.Rise = f.Rise;
    output.Look = float4(look.Tint, look.Standing);
    output.Plant = inst.BladeColors != 0 ? PlantColour(inst, onPlant) : 0.0.xxx;
    return output;
}

struct PSOutput
{
    float4 Albedo : SV_Target0;
    float4 Normal : SV_Target1;
    float4 Data : SV_Target2;
    float  Depth : SV_Target3;
    uint   EntityId : SV_Target4;
    float2 Displacement : SV_Target5;
};

// A pulse train (1 where frac(x) < width) integrated from 0 to x: box-filtering the rows over a pixel is a difference
float RowIntegral(float x, float width)
{
    return floor(x) * width + min(frac(x), width);
}

// The metres a pixel spans on a surface: the mean of its footprint's two axes. The longest axis (a grazing view's,
// stretched along the line of sight) faded every fine detail away from eye level, and the crop read as cloth.
float PixelMetres(float2 dx, float2 dy)
{
    return max(sqrt(length(dx) * length(dy)), 1e-3);
}

PSOutput PS(VSOutput input, bool front : SV_IsFrontFace)
{
    Instance inst = LoadInstance(input.InstanceSlot);
    float2 local = input.WorldPos.xz - inst.TerrainCorner;
    float dist = distance(input.WorldPos, CamPos);
    float metresPerPixel = PixelMetres(ddx(local), ddy(local));

    // A far edge seen from behind (it slopes away from a low eye) takes the light on its side towards the eye
    float3 normal = normalize(input.Normal) * (front ? 1.0 : -1.0);

    // Porous where the plants still stand (CanopyCoverage)
    float opacity = CanopyOpacity(input.Rise);
    if (opacity < 1.0)
        clip(CanopyCoverage(local, opacity));

    // A shell above the lowest keeps the tufts that reach it, by the vertex shader's measure of how far apart the
    // shells stand (by its own footprint, a grazing view would drop the upper shells while the lowest still stood low).
    // Down the edge its tufts thin out towards the foot, so its outline is the tufts'.
    float shell, toFoot;
    ShellOf(input.UV.y, shell, toFoot);
    float layer = shell / (CANOPY_SHELLS - 1.0);
    float tuft = ShellTuft(local) * (1.0 - 0.6 * toFoot);
    if (shell > 0.5)
        clip(tuft * ShellDetail(PixelMetresAt(dist)) - layer);
    else if (toFoot > 0.0)
        clip(ShellTuft(local + 3.7) + 0.35 - toFoot * ShellDetail(PixelMetresAt(dist)));

    // The colour the plants are drawn with at this height on them (PlantColour), else the decorator's far colour (the
    // sward's), times what varies over metres (CanopyLook)
    float3 crop = float3(0.11, 0.1, 0.05);
    if (inst.BladeColors != 0)
        crop = input.Plant;
    else if (inst.TintBuffer != 0)
    {
        StructuredBuffer<float4> tints = ResourceDescriptorHeap[NonUniformResourceIndex(inst.TintBuffer)];
        crop = tints[inst.Group * 2].rgb;
    }
    crop *= input.Look.rgb;

    float ao = 1.0;
    {
        // Deeper in the crop, darker: the lowest shell in the shade of the tufts over it (where the shells stand apart),
        // and down the edge the stalks of the short plants there
        float sunk = (1.0 - layer) * ShellDetail(PixelMetresAt(dist));
        crop *= (1.0 - 0.25 * sunk) * (1.0 - 0.3 * toFoot);
        ao = (1.0 - 0.3 * sunk) * (1.0 - 0.35 * toFoot);

        // The ground, its relief, the plants' patches, lodging and the wind are in the vertex's normal and tint
        // (CanopyLook); here the detail finer than its 1 m grid
        float2 tilt = 0;
        float standing = input.Look.a;

        // Tufts and hollows, faded out where a pixel spans them (they would only shimmer); finer still, the heads in
        // the light and the dark between them, so near the crop is grainy instead of a cloth
        float shade = 1.0;
        {
            float earFade = saturate(1.5 - 2.0 * metresPerPixel / 0.09) * standing;
            if (earFade > 0.0)
                shade *= lerp(1.0, 0.5 + DecoGradientNoise(local / 0.09 + 41.0), earFade * 0.75);
            float clumpFade = saturate(1.5 - 2.0 * metresPerPixel / 0.25) * standing;
            if (clumpFade > 0.0)
                shade *= lerp(1.0, 0.7 + 0.6 * DecoGradientNoise(local / 0.25 + 17.0), clumpFade * 0.5);
        }
        {
            const float size = 0.7, relief = 0.1;          // m
            float fade = saturate(1.5 - 2.0 * metresPerPixel / size) * standing;
            if (fade > 0.0)
            {
                const float2 offset = float2(13.1, 71.7);
                const float d = size * 0.25;
                float n0 = DecoGradientNoise(local / size + offset);
                float nx = DecoGradientNoise((local + float2(d, 0)) / size + offset);
                float nz = DecoGradientNoise((local + float2(0, d)) / size + offset);
                tilt += float2(n0 - nx, n0 - nz) / d * relief * fade;
                shade *= lerp(1.0, 0.75 + 0.5 * n0, fade * 0.6);
            }
        }

        // Streaks along the way the field was sown: the drill's and the plough's lines, where the crop stands a little
        // denser or thinner, taller or shorter, eight times as long as they are wide. Seen from afar this grain, not
        // round blotches, is what reads as a field. Each scale fades where a pixel spans it across.
        if (dot(inst.RowAcross, inst.RowAcross) > 0.5)
        {
            float2 along = float2(inst.RowAcross.y, -inst.RowAcross.x);
            float2 q = float2(dot(local, along), dot(local, inst.RowAcross));
            float pixelAcross = max(abs(ddx(q.y)) + abs(ddy(q.y)), 1e-4);
            [unroll] for (uint k = 0; k < 2; k++)
            {
                float width = k == 0 ? 0.5 : 1.8;                  // m across
                float fade = saturate(1.5 - 2.0 * pixelAcross / width) * standing;
                if (fade <= 0.0) continue;
                float2 offset = k == 0 ? float2(3.3, 19.1) : float2(-27.7, 8.3);
                float n0 = DecoGradientNoise(float2(q.x / (8.0 * width), q.y / width) + offset);
                float n1 = DecoGradientNoise(float2(q.x / (8.0 * width), (q.y + 0.25 * width) / width) + offset);
                tilt += inst.RowAcross * (n0 - n1) / (0.25 * width) * (k == 0 ? 0.06 : 0.15) * fade;
                shade *= lerp(1.0, 0.78 + 0.44 * n0, fade * (k == 0 ? 0.6 : 0.45));
            }
        }
        crop *= shade;

        // Rows: stripes of plants over shaded soil, box-filtered across the pixel; each row rounded, its sides
        // facing the light or away while a pixel still holds a row
        if (inst.RowSpacing > 0.0)
        {
            float x = dot(input.WorldPos.xz, inst.RowAcross) / inst.RowSpacing + 0.5 * inst.RowCover;
            float fw = max(fwidth(x), 1e-4);
            float cover = (RowIntegral(x + 0.5 * fw, inst.RowCover) - RowIntegral(x - 0.5 * fw, inst.RowCover)) / fw;
            const float3 SHADED_SOIL = float3(0.05, 0.038, 0.027);
            crop = lerp(SHADED_SOIL, crop, cover);
            float rowFade = saturate(1.0 - fw * 2.0);
            float side = sin(6.2831853 * (x - 0.5 * inst.RowCover));
            tilt -= inst.RowAcross * side * 0.5 * rowFade;
            ao *= lerp(0.75, 1.0, cover);
        }

        normal = normalize(normal + float3(tilt.x, 0, tilt.y));
    }

    PSOutput output;
    output.Albedo = float4(crop, 0);
    output.Normal = float4(normal, 1.0);
    output.Data = float4(0.9, 0.0, ao, 0.5);       // matte, no metal, ao, vegetation lighting
    output.Depth = input.Depth;
    output.Displacement = float2(0, 0);
    output.EntityId = (input.TransformSlot << 8u) | (input.MeshPartIdx & 0xFFu) | (2u << 30);   // class 2: vegetation
    return output;
}

// Shadow passes: sun cascades, or point light cube faces when local
struct ShadowVSOutput
{
    float4 Position : SV_POSITION;
    nointerpolation uint RTIndex : SV_RenderTargetArrayIndex;
    float4 ClipDistance : SV_ClipDistance0;
    float3 Canopy : TEXCOORD0;      // terrain-local x, z and the rise: the same gaps as the canopy (by ground position)
};

ShadowVSOutput ShadowVertex(uint primitiveVertexID, uint instanceID, bool local)
{
    StructuredBuffer<uint> expansion = ResourceDescriptorHeap[ExpansionBufferIdx];
    uint view, idx;
    DecodeShadowEntry(expansion[InstanceBaseOffset + instanceID], local, view, idx);
    StructuredBuffer<InstanceDescriptor> descriptors = ResourceDescriptorHeap[DescriptorBufIdx];
    Fetched f = FetchVertex(primitiveVertexID, descriptors[idx]);

    ShadowVSOutput output;
    output.Position = ShadowClipPosition(float4(f.World, 1.0), view, local, CascadeBufferSRVIdx, output.RTIndex, output.ClipDistance);
    output.Canopy = float3(f.World.xz - LoadInstance(f.InstanceSlot).TerrainCorner, f.Rise);
    // Only the lowest shell throws a shadow: the others' triangles collapse to a point
    float shell, toFoot;
    ShellOf(f.UV.y, shell, toFoot);
    if (shell > 0.5)
        output.Position = float4(0.0, 0.0, 0.0, 1.0);
    return output;
}

ShadowVSOutput VS_Shadow(uint primitiveVertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
    return ShadowVertex(primitiveVertexID, instanceID, false);
}

ShadowVSOutput VS_LocalShadow(uint primitiveVertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
    return ShadowVertex(primitiveVertexID, instanceID, true);
}

// Where the canopy is porous its shadow is too, so the ground under it darkens as under the plants
void PS_Shadow(ShadowVSOutput input)
{
    float opacity = CanopyOpacity(input.Canopy.z);
    if (opacity < 1.0)
        clip(CanopyCoverage(input.Canopy.xy, opacity));
}

technique11 CropCanopy
{
    pass Opaque
    {
        SetVertexShader(CompileShader(vs_6_6, VS()));
        SetPixelShader(CompileShader(ps_6_6, PS()));
    }
    pass Shadow
    {
        SetVertexShader(CompileShader(vs_6_6, VS_Shadow()));
        SetPixelShader(CompileShader(ps_6_6, PS_Shadow()));
    }
    pass LocalShadow
    {
        SetVertexShader(CompileShader(vs_6_6, VS_LocalShadow()));
        SetPixelShader(CompileShader(ps_6_6, PS_Shadow()));
    }
}
