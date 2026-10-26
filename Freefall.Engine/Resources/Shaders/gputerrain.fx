cbuffer PushConstants : register(b3)
{
    uint _reserved0;
    uint TerrainDataIdx;        // 1: Per-instance TerrainPatchData buffer
    uint DescriptorBufIdx;      // 2
    uint _reserved3;
    uint _reserved4;
    uint _reserved5;
    uint _reserved6;
    uint _reserved7;            // 7: was the patch's index buffer, pulled by the VS (drawn indexed now)
    uint _reserved8;            // 8
    uint PosBufferIdx;          // 9
    uint _reserved10;
    uint _reserved11;
    uint _reserved12;
    uint _reserved13;
    uint _reserved14;
    uint GlobalTransformBufferIdx; // 15
    uint DebugMode;             // 16
    uint HeightTexIdx;          // 17
    uint ControlMapsIdx;        // 18
    uint _reserved19;           // 19: was the layers' diffuse array (LayerTextures now)
    uint _reserved20;           // 20: was the layers' normal array
    uint DecoControlMapIdx;     // 21: Decoration control texture
    uint DecoTintIdx;           // 22: per decorator, the colour the ground under it takes (rgb, strength)
    uint CascadeBufferSRVIdx;   // 23: SRV: StructuredBuffer<CascadeData>
    uint ShadowCascadeCount;    // 24: uint: number of cascades
    uint CascadeIdxBufIdx;      // 25: SRV: per-entry cascade index (uint)
    uint LayerHeights;          // 26: 1 = some layer has a height map (LayerTextures.z; height blend and SSDM)
    uint HoleTexIdx;            // 27: R8 hole mask (HoleStamp, terrain_holes.hlsli), 0 = no holes
};

#include "common.fx"
#include "ground_shape.hlsli"
#include "decoration_noise.hlsli"     // DecoEdgeWarp: the ground tint keeps the edge the spawn kernel draws
// @RenderState(RenderTargets=6)

// TerrainPatchData: per-instance data written by terrain_quadtree.hlsl compute shader.
// World transform is computed from patch rect — no per-patch TransformSlot needed.
struct TerrainPatchData
{
	float4 rect;    // (minX, maxZ, maxX, minZ) — Z components are swapped (matches compute shader)
	float2 level;   // (lod, geomorphBlend)
	float2 padding;
};

cbuffer terrain : register(b1)
{
    float3 CameraPos;
    float HeightTexel;
    float MaxHeight;
    float BlendDepth;   // layer transition softness in weight space (Terrain.LayerBlendDepth)
    float2 TerrainSize;
    float2 TerrainOrigin;
    float DecoRadius;       // decorators are drawn this far
    float DecoGroundTint;   // Terrain.DecorationGroundTint: 0 = off
}

cbuffer tiling : register(b2)
{
    float4 LayerTiling[32];
    float4 LayerTint[32];      // TerrainLayer.Tint (rgb); w = TerrainLayer.TileRotation in radians
    float4 LayerPlantTint[32]; // TerrainLayer.PlantTint (rgb): what the layer does to the sward on it
    // Each layer's own textures, bindless: diffuse, normals, height (TerrainRenderer.RebuildTextureArrays). Until
    // 2026-10-24 copies in three Texture2DArrays, uncompressed RGBA8 at the largest size whenever the layers' textures
    // differed in size or format (Eldermoor's do): ~800 MB, and 4..8x the bytes per tap this shader fetches.
    uint4 LayerTextures[32];
}

SamplerState sampData : register(s0); // WrappedAnisotropic
SamplerState sampHeight : register(s1); // ClampedPoint2D
SamplerState sampHeightFilter : register(s2); // ClampedBilinear2D
SamplerState sampTrilinear : register(s4); // WrappedTrilinear — displacement/height maps

#include "terrain_holes.hlsli"

// Holes (HoleStamp): nothing of the terrain is drawn there, in any pass
bool InTerrainHole(float2 worldXZ)
{
    return TerrainHoleAt(HoleTexIdx, sampHeightFilter, worldXZ - TerrainOrigin, TerrainSize);
}

// Edge-stitched vertex: snaps odd edge vertices to even positions on edges
// adjacent to coarser neighbors. No morphing — hard LOD transition masked
// by sub-pixel screen-space error threshold.
// stitchMask bits: 0=S(-Z), 1=E(+X), 2=N(+Z), 3=W(-X)
float3 StitchVertex(float3 pos, float2 rect_xy, float2 rect_size, uint stitchMask, row_major matrix entityWorld, out float2 outHeightUV)
{
    const float gridRes = 32.0;
    const float invGridRes = 1.0 / gridRes;

    // Grid pos in [0, gridRes]
    float2 gridPos = pos.xz + float2(16, 16);

    // Snap odd edge vertices to even positions on stitched edges
    // South edge: gridPos.y == 0
    if ((stitchMask & 1u) && gridPos.y < 0.5)
        gridPos.x = round(gridPos.x * 0.5) * 2.0;
    // East edge: gridPos.x == 32
    if ((stitchMask & 2u) && gridPos.x > 31.5)
        gridPos.y = round(gridPos.y * 0.5) * 2.0;
    // North edge: gridPos.y == 32
    if ((stitchMask & 4u) && gridPos.y > 31.5)
        gridPos.x = round(gridPos.x * 0.5) * 2.0;
    // West edge: gridPos.x == 0
    if ((stitchMask & 8u) && gridPos.x < 0.5)
        gridPos.y = round(gridPos.y * 0.5) * 2.0;

    // Compute height UV
    float2 uv = gridPos * invGridRes;
    uv.y = 1.0 - uv.y;
    outHeightUV = rect_xy + uv * rect_size;

    // Sample height
    Texture2D HeightTex = ResourceDescriptorHeap[HeightTexIdx];
    float h = HeightTex.SampleLevel(sampHeightFilter, outHeightUV, 0).r * MaxHeight;

    float3 result;
    result.xz = gridPos - float2(16, 16);
    result.y = h;
    return result;
}

struct VertexOutput
{
	float4 Position : SV_POSITION;
	float2 UV			: TEXCOORD0;
	float2 UV2			: TEXCOORD1;
    float Depth			: TEXCOORD2;
	float Level			: TEXCOORD3;
	nointerpolation uint TransformSlot : TEXCOORD4;
	float3 WorldPos		: TEXCOORD5;
};



// Compute world matrix from patch rect (no per-patch TransformSlot overhead).
// The single shared TransformSlot holds the terrain entity's world matrix (typically identity).
// Note: rect format is (minX, maxZ, maxX, minZ) — Z components are swapped.
// Use abs() to ensure positive scale since rect.w - rect.y is negative.
row_major matrix ComputePatchWorld(TerrainPatchData patch, row_major matrix entityWorld)
{
    float2 patchSize = abs(patch.rect.zw - patch.rect.xy) * TerrainSize;
    float gridRes = 32.0; // must match Mesh.CreatePatch grid resolution
    float2 patchScale = patchSize / gridRes;

    // Mesh vertices span [-16,+16] centered at origin, so translate to patch CENTER
    // (rect.xy + rect.zw) * 0.5 gives correct center since addition is commutative
    float2 patchCenter = (patch.rect.xy + patch.rect.zw) * 0.5 * TerrainSize;

    // Patch-local world transform
    row_major matrix patchWorld = {
        patchScale.x, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, patchScale.y, 0,
        patchCenter.x, 0, patchCenter.y, 1
    };

    return mul(patchWorld, entityWorld);
}

VertexOutput VS(uint primitiveVertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
	VertexOutput output;

	// Direct access — terrain self-draws, no culler indirection needed.
	// instanceID maps directly into the compute shader's append buffers.
	uint idx = instanceID;

	// Get entity transform from global buffer (single shared slot — typically identity)
	StructuredBuffer<InstanceDescriptor> descriptors = ResourceDescriptorHeap[DescriptorBufIdx];
	StructuredBuffer<row_major matrix> globalTransforms = ResourceDescriptorHeap[GlobalTransformBufferIdx];
	uint slot = descriptors[idx].TransformSlot;
	row_major matrix entityWorld = globalTransforms[slot];

	// Per-instance terrain patch data
	StructuredBuffer<TerrainPatchData> terrainData = ResourceDescriptorHeap[TerrainDataIdx];
	TerrainPatchData patch = terrainData[idx];

	// Compute world matrix from patch rect — the key difference from terrain.fx
	row_major matrix world = ComputePatchWorld(patch, entityWorld);

	// Vertex position from the bindless position buffer: drawn indexed (TerrainRenderer.DrawTerrain), SV_VertexID is
	// the grid vertex
    StructuredBuffer<float3> positions = ResourceDescriptorHeap[PosBufferIdx];
    float3 pos = positions[primitiveVertexID];

    // Edge-stitched vertex displacement (no morphing)
    float4 rect = patch.rect;
    float2 rectSize = float2(rect.z - rect.x, rect.w - rect.y);
    float2 heightUV;
    uint stitchMask = (uint)patch.level.y;
    float3 stitchedPos = StitchVertex(pos, rect.xy, rectSize, stitchMask, entityWorld, heightUV);

    // precise: the depth prepass and the shading pass (technique below) draw this VS twice and the shading pass
    // tests GreaterEqual against the prepass' depth, so both must compute bit-identical positions
    precise float4 worldPosition = mul(float4(stitchedPos, 1), world);
    output.WorldPos = worldPosition.xyz;
    worldPosition = mul(worldPosition, ViewProjection);

    output.Position = worldPosition;

    float2 uv = (stitchedPos.xz + float2(16, 16)) * 0.03125;
    uv.y = 1 - uv.y;
    output.Depth = worldPosition.w;
    output.UV = uv;
    output.UV2 = heightUV;
	output.Level = patch.level.x;
	output.TransformSlot = slot;
	return output;
}

struct FragmentOutput
{
	float4 Albedo		: SV_TARGET0;
	float4 Normals		: SV_TARGET1;
	float4 Data			: SV_TARGET2;
	float  Depth		: SV_TARGET3;
	uint   EntityId		: SV_TARGET4;
	float2 Displacement	: SV_TARGET5;
};

float3 GetNormal(float2 uv)
{
	float4 h;
	Texture2D HeightTex = ResourceDescriptorHeap[HeightTexIdx];

    h[0] = HeightTex.SampleLevel(sampHeightFilter, uv + float2(0, -HeightTexel), 0).r;
    h[1] = HeightTex.SampleLevel(sampHeightFilter, uv + float2(-HeightTexel, 0), 0).r;
    h[2] = HeightTex.SampleLevel(sampHeightFilter, uv + float2(HeightTexel, 0), 0).r;
    h[3] = HeightTex.SampleLevel(sampHeightFilter, uv + float2(0, HeightTexel), 0).r;
	
    // Central difference spans two texels (h[1]-h[2], h[0]-h[3])
    float texelWorldSize = TerrainSize.x * HeightTexel;
    float heightScale = MaxHeight / (2.0 * texelWorldSize);
    
	float3 n;
	n.z = (h[0] - h[3]) * heightScale;
	n.x = (h[1] - h[2]) * heightScale;
	n.y = 1.0f;

	return normalize(n);
}


// The colour of what grows here, as the decorators look from afar (CS_DecoratorTints), and how much of the ground it
// claims. As the spawn kernel weighs its instances: coverage bilinear between the decoration control texels, looked up
// at the same small offset (DecoEdgeWarp), then cubed; times the decorators' strength. Cubing after interpolating
// keeps the tint to where the cards are: cubing each texel first spread a lime fringe a metre past every patch edge.
// A decorator whose plants grow in strong clumps (ferns) also takes the spawn's clump field here, cubed as the spawn's
// density, so the far ground shows the patches; one noise per decorator per pixel, and only where the clumped part
// weighs (> 0.25).
// How much the tint shows with distance: even plants (grass) lerp(0.2, 1) over 10..70 m, the ground between blades
// staying for texture near the camera; clumped plants (ferns, between them bare floor) nothing while they are drawn,
// then exactly as the spawn thins them out over the last 30% of their range, so they hand over without a band.
// Returns rgb and that strength; 'growth' is the strength without the distance ramp (where something grows).
float4 DecoratorGroundTint(float2 ctrlUV, float2 terrainLocal, float dist, out float growth)
{
    Texture2DArray<uint4> decoCtrl = ResourceDescriptorHeap[DecoControlMapIdx];
    StructuredBuffer<float4> tints = ResourceDescriptorHeap[DecoTintIdx];      // 2 per decorator (CS_DecoratorTints)
    uint w, h, slices;
    decoCtrl.GetDimensions(w, h, slices);
    float2 warp = DecoEdgeWarp(terrainLocal);
    ctrlUV += float2(warp.x, -warp.y) / TerrainSize;     // control rows run against terrain Z
    float2 p = ctrlUV * float2(w, h) - 0.5;
    int2 i0 = int2(floor(p));
    float2 f = p - floor(p);
    float2 clumpPos = float2(terrainLocal.x, TerrainSize.y - terrainLocal.y);   // the spawn's frame for the clumps

    float evenRamp = lerp(0.2, 1.0, smoothstep(10.0, 70.0, dist));
    float3 colorSum = 0;
    float strengthSum = 0;      // coverage x strength (ramped with distance)
    float growthSum = 0;        // coverage x strength
    float coverageSum = 0;      // coverage, all decorators
    uint cachedGroup = 0xFFFFFFFFu;
    float4 cached = 0;          // the last decorator's colour and ramped strength here
    float cachedGrowth = 0;
    [unroll] for (uint k = 0; k < 4; k++)
    {
        int2 offset = int2(k & 1, k >> 1);
        float bilinear = (offset.x ? f.x : 1 - f.x) * (offset.y ? f.y : 1 - f.y);
        int2 texel = clamp(i0 + offset, int2(0, 0), int2(w - 1, h - 1));
        uint4 s0 = decoCtrl.Load(int4(texel, 0, 0));
        uint4 s1 = decoCtrl.Load(int4(texel, 1, 0));
        [unroll] for (uint c = 0; c < 8; c++)
        {
            uint packed = c < 4 ? s0[c] : s1[c - 4];
            uint group = packed >> 8;
            float coverage = (packed & 0xFF) / 255.0;
            if (group >= 254 || coverage <= 0) continue;     // 255 empty, 254 names the rows (decoration_prepass.hlsl)
            if (group != cachedGroup)
            {
                float4 even = tints[group * 2];
                float4 clumps = tints[group * 2 + 1];      // scale, seed x 2 + amount, range, weight
                float evenStrength = even.a;
                float clumpStrength = 0;
                float clumpRamp = 1.0;
                if (clumps.w > 0.25)
                {
                    float seed = floor(clumps.y * 0.5);
                    float amount = clumps.y - 2.0 * seed;
                    float m = lerp(1.0, DecoClusterMask(clumpPos, clumps.x, (uint)seed), amount);
                    // Plants dropped at random cover 1 - e^-(leaf area) of the ground: ~0.85 in a fern patch's core,
                    // softly less towards its edge. Saturating the area made flat, opaque, hard-edged patches. x 0.75:
                    // at full cover the far patches were "a bit too present" (user)
                    clumpStrength = 0.75 * (1.0 - exp(-clumps.w * m * m * m));
                    clumpRamp = saturate((dist - 0.7 * clumps.z) / (0.3 * clumps.z));     // = 1 - the spawn's keep
                }
                else
                    evenStrength += clumps.w;
                float growthHere = saturate(evenStrength + clumpStrength);
                float ramped = saturate(evenStrength * evenRamp + clumpStrength * clumpRamp);
                cached = float4(even.rgb, ramped);
                cachedGrowth = growthHere;
                cachedGroup = group;
            }
            float wgt = bilinear * coverage * cached.a;
            colorSum += cached.rgb * wgt;
            strengthSum += wgt;
            growthSum += bilinear * coverage * cachedGrowth;
            coverageSum += bilinear * coverage;
        }
    }
    growth = 0;
    if (coverageSum <= 0) return 0;
    float cover = saturate(coverageSum);
    float cover3 = cover * cover * cover;
    growth = growthSum / coverageSum * cover3;
    return float4(colorSum / max(strengthSum, 1e-5), strengthSum / coverageSum * cover3);
}

// ── Hex-tiling (Mikkelsen, "Practical Real-Time Hex-Tiling", JCGT 2022; after Heitz and Neyret 2018) ──
// Breaks up the tiling of every layer: a 2..3 m scan repeated in rows was a grid across every meadow and forest floor.
// The plane is cut into hexagons; each samples the texture with its own rotation and offset, and a point blends the
// three hexagons around it. The blend is sharpened (HEX_EXPONENT) and steered by the samples' content: by height where
// the layer has a height map (the higher sample wins, as between layers), else by brightness, so the seams run around
// stones and leaves instead of showing a blurred double image. Gradients are rotated with each sample (correct mips),
// normal maps are rotated back. A point deep inside one hexagon samples only that one (HEX_DOMINANT).
// An offset-only variant tried first (no rotation) kept every copy in the same orientation: still rows.
// How far a hexagon may turn is per layer (TerrainLayer.TileRotation, LayerTint.w): sand ripples turned every way
// became swirls, so a pattern that runs one way turns only a few degrees, and its far scale turns as little.
static const float FAR_TILING = 4.0;        // see PS: two scales per layer
static const float FAR_ANGLE = 0.6458;      // 37°: the far scale's turn against the near one, for a layer that may turn fully

static const float HEX_CONTRAST = 0.6;      // how much the samples' content steers the blend (Mikkelsen's 0.6)
static const float HEX_EXPONENT = 7.0;      // sharpness of the blend between hexagons (Mikkelsen's 7; HexSetup hard-codes w^7)
static const float HEX_DOMINANT = 0.94;     // near: a hexagon with more of the geometric weight than this is sampled alone
static const float HEX_DOMINANT_FAR = 0.8;  // far scale: its 8..12 m hexagons need less blending to hide
static const float3 HEX_LUMA = float3(0.299, 0.587, 0.114);

struct HexTiles
{
    float2 UV[3];           // the point in each of the three hexagons' rotated, offset texture space
    float2x2 Rot[3];
    float2 Ddx[3], Ddy[3];  // the texture gradients, rotated with each
    float3 Geometric;       // sharpened barycentric weights, normalised
    float3 Single;          // one-hot when one hexagon dominates, else 0
};

void HexTriangleGrid(float2 st, out float3 w, out int2 v1, out int2 v2, out int2 v3)
{
    st *= 2.0 * 1.7320508;
    float2 skewed = mul(float2x2(1.0, -0.57735027, 0.0, 1.15470054), st);
    int2 baseId = int2(floor(skewed));
    float3 t = float3(frac(skewed), 0);
    t.z = 1.0 - t.x - t.y;
    float s = step(0.0, -t.z);
    float s2 = 2.0 * s - 1.0;
    w = float3(-t.z * s2, s - t.y * s2, s - t.x * s2);
    v1 = baseId + int2(s, s);
    v2 = baseId + int2(s, 1 - s);
    v3 = baseId + int2(1 - s, s);
}

// Integer hash of a hexagon (no sin: the setup runs three times per layer per pixel)
uint HexHashU(int2 p)
{
    uint h = uint(p.x) * 1597334677u ^ uint(p.y) * 3812015801u;
    h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15; h *= 0x846ca68bu; h ^= h >> 16;
    return h;
}

float2 HexOffset(uint h)
{
    return float2(h & 0xFFFFu, h >> 16) * (1.0 / 65535.0);
}

float2 HexCenter(int2 v)
{
    return mul(float2x2(1.0, 0.5, 0.0, 1.0 / 1.15470054), float2(v)) / (2.0 * 1.7320508);
}

// A random rotation of up to 'maxAngle' (radians) either way, from a second hash
float2x2 HexRotation(uint h, float maxAngle)
{
    h = h * 747796405u + 2891336453u;
    h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
    float s, c;
    sincos((float(h >> 8) * (2.0 / 16777215.0) - 1.0) * maxAngle, s, c);
    return float2x2(c, -s, s, c);
}

// The far scale's turn against the near one: FAR_ANGLE, scaled down for a layer whose hexagons may turn less
float2x2 FarRotation(float maxAngle)
{
    float s, c;
    sincos(FAR_ANGLE * maxAngle * (1.0 / 3.14159265), s, c);
    return float2x2(c, -s, s, c);
}

// 'dominant': above this share of the geometric weight a hexagon is sampled alone; 'maxAngle': the layer's
// TileRotation (radians)
HexTiles HexSetup(float2 st, float2 dx, float2 dy, float dominant, float maxAngle)
{
    HexTiles h;
    float3 w;
    int2 v1, v2, v3;
    HexTriangleGrid(st, w, v1, v2, v3);
    int2 v[3] = { v1, v2, v3 };
    [unroll] for (int i = 0; i < 3; i++)
    {
        uint hash = HexHashU(v[i]);
        h.Rot[i] = HexRotation(hash, maxAngle);
        float2 cen = HexCenter(v[i]);
        h.UV[i] = mul(st - cen, h.Rot[i]) + cen + HexOffset(hash);
        h.Ddx[i] = mul(dx, h.Rot[i]);
        h.Ddy[i] = mul(dy, h.Rot[i]);
    }
    float3 w2 = w * w, w4 = w2 * w2;
    float3 g = w4 * w2 * max(w, 0.0);           // pow(w, 7) without pow
    h.Geometric = g / (g.x + g.y + g.z);
    h.Single = float3(h.Geometric.x > dominant, h.Geometric.y > dominant, h.Geometric.z > dominant);
    return h;
}

float3 HexWeights(HexTiles h, float3 content)
{
    float3 w = lerp(1.0.xxx, content, HEX_CONTRAST) * h.Geometric;
    return w / max(w.x + w.y + w.z, 1e-5);
}

// A layer's height through the hexagons; 'weights' returns the blend for the layer's colour and normals
float HexHeight(Texture2D heights, HexTiles h, out float3 weights)
{
    bool single = any(h.Single > 0);
    float3 hs = 0;
    [unroll] for (int i = 0; i < 3; i++)
        if (!single || h.Single[i] > 0)
            hs[i] = heights.SampleGrad(sampTrilinear, h.UV[i], h.Ddx[i], h.Ddy[i]).r;
    weights = single ? h.Single : HexWeights(h, hs);
    return dot(weights, hs);
}

// A layer's colour and normal tilt through the hexagons: with the given weights (from its height), or with weights from
// the colours' brightness when 'byColor'
void HexColorNormal(Texture2D diffuse, Texture2D normals, HexTiles h, float3 weights, bool byColor,
                    out float4 color, out float2 tilt)
{
    bool single = any(h.Single > 0);
    float3 need = single ? h.Single : (byColor ? 1.0.xxx : weights);
    float4 c[3];
    float2 t[3];
    [unroll] for (int i = 0; i < 3; i++)
    {
        c[i] = 0;
        t[i] = 0;
        if (need[i] > 0.001)
        {
            c[i] = diffuse.SampleGrad(sampData, h.UV[i], h.Ddx[i], h.Ddy[i]);
            float2 n = normals.SampleGrad(sampData, h.UV[i], h.Ddx[i], h.Ddy[i]).rg * 2.0 - 1.0;
            // Back from the hexagon's rotated space. The maps are OpenGL (green = up = -v), a reflection against the
            // UVs, which turns the back-rotation Rot into its transpose: mul(n, Rot)
            t[i] = mul(n, h.Rot[i]);
        }
    }
    float3 w = single ? h.Single : (byColor ? HexWeights(h, float3(dot(c[0].rgb, HEX_LUMA), dot(c[1].rgb, HEX_LUMA), dot(c[2].rgb, HEX_LUMA))) : weights);
    color = w.x * c[0] + w.y * c[1] + w.z * c[2];
    tilt = w.x * t[0] + w.y * t[1] + w.z * t[2];
}

// One projection of a layer at two scales (PS: near and far tiling). A scale whose weight is zero is not sampled.
// 'useHex': go through the hexagons ('hex' and 'hexWeights' for the near scale, set up by the caller: the top
// projection with the layer's height, the side projections by brightness; the far scale sets up its own, turned
// 'farRot' with hexagons turning up to 'maxAngle').
void SampleTwoScale(Texture2D diffuse, Texture2D normals, float2 uvNear, float2 uvFar,
                    float farBlend, out float4 color, out float2 tilt,
                    HexTiles hex, float3 hexWeights, bool hexByColor, float2 dx, float2 dy, bool useHex,
                    float2x2 farRot, float maxAngle)
{
    color = 0;
    tilt = 0;
    if (farBlend < 1)
    {
        float4 c;
        float2 t;
        if (useHex)
            HexColorNormal(diffuse, normals, hex, hexWeights, hexByColor, c, t);
        else
        {
            c = diffuse.Sample(sampData, uvNear);
            t = normals.Sample(sampData, uvNear).rg * 2.0 - 1.0;
        }
        color += c * (1 - farBlend);
        tilt += t * (1 - farBlend);
    }
    if (farBlend > 0)
    {
        float4 c;
        float2 t;
        if (useHex)
        {
            HexTiles hexFar = HexSetup(uvFar, mul(dx / FAR_TILING, farRot), mul(dy / FAR_TILING, farRot), HEX_DOMINANT_FAR, maxAngle);
            HexColorNormal(diffuse, normals, hexFar, 0, true, c, t);
            t = mul(t, farRot);             // back from the far sample's turn (same reflection as in HexColorNormal)
        }
        else
        {
            c = diffuse.Sample(sampData, uvFar);
            t = normals.Sample(sampData, uvFar).rg * 2.0 - 1.0;
        }
        color += c * farBlend;
        tilt += t * 0.5 * farBlend;         // the far sample's relief is FAR_TILING x too big: flatten it halfway
    }
}

FragmentOutput PS(VertexOutput input)
{
    FragmentOutput output;

    // Not in the depth prepass' variant without holes (DepthPrepass, no discard), so this pass must cut them itself:
    // with no depth there, the hole would otherwise be shaded over whatever lies below it
    if (InTerrainHole(input.WorldPos.xz)) discard;

    float3 terrainNormal = GetNormal(input.UV2);
    float3 faceNormal = terrainNormal;

    float4 color = float4(0, 0, 0, 0);
    float3 layerMean = 0;                   // the layers' average colour (smallest mip x tint), as blended here
    float3 normalTilt = 0;                  // world-space tilt from the layers' normal maps
	
    float2 uv = input.UV2;
    uv.y = 1 - uv.y;
    float2 uvDdx = ddx(uv), uvDdy = ddy(uv);   // for SampleGrad inside the branching layer loops (tile variation)

    Texture2DArray ControlMaps = ResourceDescriptorHeap[ControlMapsIdx];

    // ControlMaps: ceil(layerCount/4) RGBA slices, each channel = one layer weight
    uint cmW, cmH, sliceCount;
    ControlMaps.GetDimensions(cmW, cmH, sliceCount);

    // Height-based texture blending with back-to-front masking.
    // Later layers mask earlier layers (matching sequential lerp priority),
    // then height maps add micro-detail at transitions between similar-weight layers.
    // blendDepth is the weight-space window in which layers crossfade: small = crisp
    // height-map-shaped edge, 1 = fade spans the whole weight ramp (falloff / HeightBlend).
    // Must stay <= 1 or zero-weight layers (skipped below) would pop in.
    const float blendDepth = clamp(BlendDepth, 0.01, 1.0);

    bool hasAnyLayer = false;
    float blendedHeight = 0;

    // Read all slice weights
    uint clampedSlices = min(sliceCount, 8u);
    float4 sliceWeights[8];
    for (uint s = 0; s < clampedSlices; s++)
        sliceWeights[s] = ControlMaps.Sample(sampData, float3(uv, s));

    // Back-to-front: later layers mask earlier layers (same priority as sequential lerp)
    // effective[i] = raw[i] * product(1 - raw[k]) for all k > i
    float remaining = 1.0;
    for (int si = (int)clampedSlices - 1; si >= 0; si--)
    {
        for (int ci = 3; ci >= 0; ci--)
        {
            float w = sliceWeights[si][ci];
            sliceWeights[si][ci] = w * remaining;
            remaining *= saturate(1.0 - w);
        }
    }

    float viewDist = distance(input.WorldPos, CameraPos);
    float farBlend = smoothstep(15.0, 90.0, viewDist);
    // Beyond the near scale (farBlend 1) a layer's height comes from one plain sample: the hexagons only matter where
    // the near texture shows, and three rotated height taps per layer, twice, were most of hex-tiling's cost far out
    bool nearShows = farBlend < 0.999;

    // Find max(effectiveWeight * (1 + heightVal))
    float maxCombined = -1;
    for (uint i = 0; i < clampedSlices; ++i)
    {
        for (uint j = 0; j < 4; ++j)
        {
            float ew = sliceWeights[i][j];
            if (ew <= 0) continue;

            uint layer = i * 4 + j;
            float heightVal = 0;
            if (LayerHeights != 0 && LayerTiling[layer].z > 0.5)
            {
                Texture2D heights = ResourceDescriptorHeap[LayerTextures[layer].z];
                float2 texuv = uv * LayerTiling[layer].xy;
                float3 unusedWeights;
                heightVal = nearShows
                    ? HexHeight(heights, HexSetup(texuv, uvDdx * LayerTiling[layer].xy, uvDdy * LayerTiling[layer].xy, HEX_DOMINANT, LayerTint[layer].w), unusedWeights)
                    : heights.SampleGrad(sampTrilinear, texuv, uvDdx * LayerTiling[layer].xy, uvDdy * LayerTiling[layer].xy).r;
            }
            maxCombined = max(maxCombined, ew + ew * heightVal);
        }
    }

    // Two scales per layer. Its own tiling (2–8 m) up close; with distance a second sample of the same textures at
    // FAR_TILING x the size, turned FAR_ANGLE so the two grids don't line up, takes over. Past ~50 m a 2 m tile
    // is a few pixels wide and the mips average every layer into one flat colour: the far sample keeps structure.

    // Steep ground also takes the layers projected from the side (along world X and Z), weighted by the slope's
    // normal: a top-down projection stretches a 6 m rock tile into streaks down a cliff. Image up = world up.
    float3 planeWeights = pow(abs(faceNormal), 4.0);
    planeWeights /= planeWeights.x + planeWeights.y + planeWeights.z;
    bool sideProjection = planeWeights.y < 0.98;
    if (!sideProjection) planeWeights = float3(0, 1, 0);
    float2 sideX = float2(input.WorldPos.z, -input.WorldPos.y);     // seen along X: u = +Z, v = up
    float2 sideZ = float2(input.WorldPos.x, -input.WorldPos.y);     // seen along Z: u = +X, v = up
    float2 sideXDdx = ddx(sideX), sideXDdy = ddy(sideX), sideZDdx = ddx(sideZ), sideZDdy = ddy(sideZ);

    // Blend using height-weighted factors
    if (maxCombined > 0)
    {
        hasAnyLayer = true;
        float threshold = maxCombined - blendDepth;
        // SSDM height: the same height blend, over the widest window (1) whatever the colour's depth. Displacing exactly
        // the surface that shows (the colour's window, 0.2 on Eldermoor) flips the height from one layer to the next
        // within a pixel or two (turf against the gap between two cobbles), a step the screen-space inversion cannot
        // follow: dark specks along every patch edge; 0.5 still left a few. Until 2026-10-17 a lerp over the stack
        // weights in channel order instead: a stone moved by its weight only, and halfway to the grass the height sank
        // below both layers (0.5 hc + 0.25 hg)
        float dispThreshold = maxCombined - 1.0;
        float totalDisp = 0;

        float4 blendedColor = 0;
        float3 blendedMean = 0;
        float3 blendedTilt = 0;
        float totalBlend = 0;

        for (uint i2 = 0; i2 < clampedSlices; ++i2)
        {
            for (uint j2 = 0; j2 < 4; ++j2)
            {
                float ew = sliceWeights[i2][j2];
                if (ew <= 0) continue;

                uint layer = i2 * 4 + j2;
                float2 texuv = uv * LayerTiling[layer].xy;
                float2 texDdx = uvDdx * LayerTiling[layer].xy, texDdy = uvDdy * LayerTiling[layer].xy;
                float maxAngle = LayerTint[layer].w;
                HexTiles hex = HexSetup(texuv, texDdx, texDdy, HEX_DOMINANT, maxAngle);
                float2x2 farRot = FarRotation(maxAngle);

                float heightVal = 0;
                float3 hexWeights = 0;
                bool hexByColor = true;     // without a height map the hexagons blend by brightness
                if (LayerHeights != 0 && LayerTiling[layer].z > 0.5)
                {
                    Texture2D heights = ResourceDescriptorHeap[LayerTextures[layer].z];
                    if (nearShows)
                    {
                        heightVal = HexHeight(heights, hex, hexWeights);
                        hexByColor = false;
                    }
                    else
                        heightVal = heights.SampleGrad(sampTrilinear, texuv, texDdx, texDdy).r;
                }

                float bd = max(ew + ew * heightVal - dispThreshold, 0);    // a layer without a height map is flat
                blendedHeight += heightVal * LayerTiling[layer].w * bd;
                totalDisp += bd;

                float bf = max(ew + ew * heightVal - threshold, 0);
                if (bf <= 0) continue;

                // World-space tilt of the normal (UDN) per projection: the top's tangent is +X, bitangent +Z
                Texture2D diffuse = ResourceDescriptorHeap[LayerTextures[layer].x];
                Texture2D normals = ResourceDescriptorHeap[LayerTextures[layer].y];
                float4 c;
                float2 nTop;
                SampleTwoScale(diffuse, normals, texuv, mul(texuv / FAR_TILING, farRot), farBlend, c, nTop,
                               hex, hexWeights, hexByColor, texDdx, texDdy, true, farRot, maxAngle);
                float3 tilt = float3(nTop.x, 0, nTop.y) * planeWeights.y;
                if (sideProjection)
                {
                    // The side projections go through the hexagons too, blended by brightness (no extra height taps).
                    // Sampled plainly (until 2026-10-15), every cliff was an exact grid of 6 m tiles: "reads
                    // repetitive no matter what" (user). A layer's TileRotation keeps strata level here as well.
                    float2 sideScale = LayerTiling[layer].xy / TerrainSize;     // 1 / metres per tile
                    float4 cX, cZ;
                    float2 nX, nZ;
                    if (planeWeights.x > 0.01)
                    {
                        float2 st = sideX * sideScale, sdx = sideXDdx * sideScale, sdy = sideXDdy * sideScale;
                        HexTiles hexX = HexSetup(st, sdx, sdy, HEX_DOMINANT, maxAngle);
                        SampleTwoScale(diffuse, normals, st, mul(st / FAR_TILING, farRot), farBlend, cX, nX,
                                       hexX, 0, true, sdx, sdy, true, farRot, maxAngle);
                    }
                    else { cX = 0; nX = 0; }
                    if (planeWeights.z > 0.01)
                    {
                        float2 st = sideZ * sideScale, sdx = sideZDdx * sideScale, sdy = sideZDdy * sideScale;
                        HexTiles hexZ = HexSetup(st, sdx, sdy, HEX_DOMINANT, maxAngle);
                        SampleTwoScale(diffuse, normals, st, mul(st / FAR_TILING, farRot), farBlend, cZ, nZ,
                                       hexZ, 0, true, sdx, sdy, true, farRot, maxAngle);
                    }
                    else { cZ = 0; nZ = 0; }
                    c = c * planeWeights.y + cX * planeWeights.x + cZ * planeWeights.z;
                    tilt += float3(0, nX.y, nX.x) * planeWeights.x;      // tangent +Z, bitangent +Y
                    tilt += float3(nZ.x, nZ.y, 0) * planeWeights.z;      // tangent +X, bitangent +Y
                }

                c.rgb *= LayerTint[layer].rgb;
                blendedColor += c * bf;
                blendedMean += diffuse.SampleLevel(sampData, float2(0.5, 0.5), 16).rgb * LayerTint[layer].rgb * bf;
                blendedTilt += tilt * bf;
                totalBlend += bf;
            }
        }

        if (totalBlend > 0)
        {
            float invTotal = 1.0 / totalBlend;
            color = blendedColor * invTotal;
            layerMean = blendedMean * invTotal;
            normalTilt = blendedTilt * invTotal;
        }
        if (totalDisp > 0)
            blendedHeight /= totalDisp;
    }

    // Normal maps, UDN (partial derivative addition) in world space: each projection's tilt is already oriented
    if (hasAnyLayer)
        terrainNormal = normalize(terrainNormal + normalTilt);

    // Ground under decorators is the sward: the plants' colour (CS_DecoratorTints: the cards' mean colour at the
    // brightness of a dense, self-shadowing sward), tinted by the layers here as the spawn tints each plant
    // (TerrainLayer.PlantTint: a dry patch yellows grass and ground alike), lit as the cards are (Data.a towards
    // 0.5, light_directional_cs.hlsl: wrapped diffuse, no specular). The same at every distance and angle, so
    // where the cards thin out and end they only reveal more of the same colour. Up close a quarter of the scanned
    // ground stays for texture between the blades. (Before 2026-10-15 a stack of distance- and angle-dependent
    // shades made up for vegetation lit at 0.4x by a ported 1.14-for-pi; the edge of the grass was a hard line.)
    float swardLighting = 0;                // 1 = lit as vegetation (Data.a 0.5)
    float plantCover = 1;                   // how much grows here (below); without decorator data: as if everywhere
    if (DecoTintIdx != 0 && DecoControlMapIdx != 0)
        plantCover = 0;
    if (DecoGroundTint > 0 && DecoTintIdx != 0 && DecoControlMapIdx != 0)
    {
        float dist = distance(input.WorldPos, CameraPos);
        float growth;
        float4 tint = DecoratorGroundTint(uv, input.WorldPos.xz - TerrainOrigin, dist, growth);
        plantCover = saturate(growth * 3.0);
        float3 plantTint = 0;
        float covered = 0;
        for (uint ps = 0; ps < clampedSlices; ps++)
        {
            for (uint pc = 0; pc < 4; pc++)
            {
                plantTint += LayerPlantTint[ps * 4 + pc].rgb * sliceWeights[ps][pc];
                covered += sliceWeights[ps][pc];
            }
        }
        plantTint += saturate(1.0 - covered);
        float amount = tint.a * DecoGroundTint;      // the distance ramp is per decorator (DecoratorGroundTint)
        // ...with the plants' own grain (decoration_noise.hlsli SwardGrain, also on every plant in the spawn) and the
        // scanned ground's light and dark (its brightness against its own average colour): a flat sward colour read
        // as felt from above, the scan's tufts and gaps (its far scale, 8..12 m tiles) read as grass
        const float3 LUMA_SWARD = float3(0.2126, 0.7152, 0.0722);
        float scanDetail = clamp(dot(color.rgb, LUMA_SWARD) / max(dot(layerMean, LUMA_SWARD), 1e-4), 0.35, 2.0);
        float3 grain = SwardGrain(input.WorldPos.xz - TerrainOrigin) * lerp(1.0, scanDetail, 0.85);
        color.rgb = lerp(color.rgb, tint.rgb * plantTint * grain, amount);
        swardLighting = amount;
    }

    // Large-scale variation (common.fx), shared with the plants: half of it up close, where the layer textures carry
    // the detail, all of it from ~150 m, where their mips have flattened every layer to one colour
    // Its dry/lush tones are for what grows: elsewhere only its brightness. Steep ground (rock) took the dry tone as
    // sandstone, and sand, dirt and paving under no decorator took the lush and hollow greens (a green beach).
    const float3 LUMA = float3(0.2126, 0.7152, 0.0722);
    float growing = smoothstep(0.7, 0.9, faceNormal.y) * plantCover;
    float macroReach = lerp(0.5, 1.0, saturate(distance(input.WorldPos, CameraPos) / 150.0));
    float3 variation = GroundVariation(input.WorldPos.xz);
    variation = lerp(dot(variation, LUMA).xxx, variation, growing);
    color.rgb *= lerp(1.0.xxx, variation, macroReach);

    // The shape of the land in its colour (ground_shape.hlsli; the grass standing here takes the same tint)
    Texture2D HeightTex = ResourceDescriptorHeap[HeightTexIdx];
    color.rgb *= GroundShapeTint(GroundShape(HeightTex, sampHeightFilter, input.UV2, HeightTexel, MaxHeight), growing);
    output.Albedo = float4(color.rgb, 0);
    output.Normals = float4(terrainNormal.xyz, 1);
    output.Data = float4(0.95, 0.0, 1.0, lerp(1.0, 0.5, swardLighting)); // matte, metal 0, ao 1, lit: PBR .. vegetation
    // Write height-offset depth: view-independent bias for consistent SSS shadows.
    float depthHeight = blendedHeight * 0.05;
    output.Depth = input.Depth - depthHeight;

    // Debug mode 5: Decoration control map overlay
    if (DebugMode == 5 && DecoControlMapIdx != 0)
    {
        Texture2DArray<uint4> decoCtrl = ResourceDescriptorHeap[DecoControlMapIdx];
        uint dW, dH, dS;
        decoCtrl.GetDimensions(dW, dH, dS);
        float2 ctrlUV = float2(input.UV2.x, 1.0 - input.UV2.y);
        int2 ctrlTexel = clamp(int2(ctrlUV * float2(dW, dH)), int2(0,0), int2(dW-1, dH-1));

        uint4 s0 = decoCtrl.Load(int4(ctrlTexel, 0, 0));
        uint4 s1 = decoCtrl.Load(int4(ctrlTexel, 1, 0));
        uint maxWt = 0;
        uint slotCount = 0;
        uint4 slices[2] = { s0, s1 };
        [unroll] for (uint k = 0; k < 8; k++)
        {
            uint packed = slices[k / 4][k % 4];
            uint sIdx = packed >> 8;
            uint wt = packed & 0xFF;
            if (sIdx >= 254) break;                         // 255 empty, 254 names the rows
            maxWt = max(maxWt, wt);
            slotCount++;
        }

        // Heatmap: black=0, red=low(1-30), yellow=mid(31-128), green=high(129-255)
        float t = maxWt / 255.0;
        float3 decoColor;
        if (maxWt == 0)
            decoColor = color.rgb * 0.2; // dark = no coverage
        else if (t < 0.12)
            decoColor = float3(0.8, 0.1, 0.1); // red = very low weight
        else if (t < 0.5)
            decoColor = lerp(float3(0.8, 0.1, 0.1), float3(0.9, 0.9, 0.1), (t - 0.12) / 0.38); // red→yellow
        else
            decoColor = lerp(float3(0.9, 0.9, 0.1), float3(0.1, 0.9, 0.1), (t - 0.5) / 0.5); // yellow→green

        output.Albedo = float4(decoColor, 0);
    }

    // Debug mode 6: Height map visualization (SSDM)
    if (DebugMode == 6)
    {
        if (LayerHeights == 0)
            output.Albedo = float4(1, 0, 0, 0); // Red = no layer has a height map
        else
            output.Albedo = float4(blendedHeight, blendedHeight, blendedHeight, 0);
    }

    // SSDM: displacement = projected_normal * height
    // faceNormal gives clean displacement direction for terrain.
    // Smooth the height for displacement using ddx/ddy to prevent fold-over:
    // average with neighbors reduces sharp gradients at  edges.
    //float hx = ddx(blendedHeight);
    //float hy = ddy(blendedHeight);
    //float smoothHeight = blendedHeight - 0.5 * (abs(hx) + abs(hy));
    //smoothHeight = max(smoothHeight, 0);
    float worldHeight = blendedHeight * 0.05;

    float4 clipPos = mul(float4(input.WorldPos, 1), ViewProjection);
    float4 clipDisp = mul(float4(input.WorldPos + faceNormal * worldHeight, 1), ViewProjection);
    float2 ndcPos = clipPos.xy / clipPos.w;
    float2 ndcDisp = clipDisp.xy / clipDisp.w;
    float2 dispVector = (ndcDisp - ndcPos) * float2(0.5, -0.5);

    // Fade displacement at screen edges so the inversion never needs off-screen sources.
    // Margin is proportional to displacement magnitude — large displacements fade earlier.
    float2 screenUV = ndcPos * float2(0.5, -0.5) + 0.5;
    float dispMag = length(dispVector);
    float margin = max(dispMag * 2.0, 0.01); // fade zone = 2x displacement size
    float edgeFade = saturate(min(min(screenUV.x, 1.0 - screenUV.x),
                                  min(screenUV.y, 1.0 - screenUV.y)) / margin);
    dispVector *= edgeFade;

    output.EntityId = (input.TransformSlot << 8u) | (1u << 30);   // class 1: terrain (decals_cs.hlsl receivers)
    output.Displacement = dispVector;
    return output;
}

// Shadow pass — single-pass multi-cascade via per-cascade compute culling.
// CSEmitLeavesShadow emits compact (patch, cascadeIdx) pairs.
// VS_Shadow reads cascadeIdx from a per-entry buffer. instanceID = patchIdx directly.


struct ShadowVSOutput
{
    float4 Position : SV_POSITION;
    float2 WorldXZ : TEXCOORD0;     // for the hole test (Shadow_Holes)
    nointerpolation uint RTIndex : SV_RenderTargetArrayIndex;
};

ShadowVSOutput VS_Shadow(uint primitiveVertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
    ShadowVSOutput output;
    output.Position = float4(0, 0, 0, 1);
    output.WorldXZ = 0;
    output.RTIndex = 0;

    uint patchIdx = instanceID; // direct 1:1 mapping

    // Read cascade index from per-entry buffer
    StructuredBuffer<uint> cascadeIndices = ResourceDescriptorHeap[CascadeIdxBufIdx];
    uint cascadeIdx = cascadeIndices[patchIdx];

    // Patch data lookup (same as opaque VS)
    StructuredBuffer<InstanceDescriptor> descriptors = ResourceDescriptorHeap[DescriptorBufIdx];
    StructuredBuffer<row_major matrix> globalTransforms = ResourceDescriptorHeap[GlobalTransformBufferIdx];
    uint slot = descriptors[patchIdx].TransformSlot;
    row_major matrix entityWorld = globalTransforms[slot];

    StructuredBuffer<TerrainPatchData> terrainData = ResourceDescriptorHeap[TerrainDataIdx];
    TerrainPatchData patch = terrainData[patchIdx];

    row_major matrix world = ComputePatchWorld(patch, entityWorld);

    StructuredBuffer<float3> positions = ResourceDescriptorHeap[PosBufferIdx];
    float3 pos = positions[primitiveVertexID];      // drawn indexed, as VS()

    // Edge-stitched vertex displacement (must match VS() for consistent shadow depth)
    float4 rect = patch.rect;
    float2 rectSize = float2(rect.z - rect.x, rect.w - rect.y);
    float2 heightUV;
    uint stitchMask = (uint)patch.level.y;
    float3 stitchedPos = StitchVertex(pos, rect.xy, rectSize, stitchMask, entityWorld, heightUV);

    float4 worldPosition = mul(float4(stitchedPos, 1), world);
    output.WorldXZ = worldPosition.xz;

    // Project using cascade-specific VP from cascade buffer
    StructuredBuffer<CascadeData> cascades = ResourceDescriptorHeap[CascadeBufferSRVIdx];
    output.Position = mul(worldPosition, cascades[cascadeIdx].VP);
    output.RTIndex = cascadeIdx;
    return output;
}

RasterizerState Wireframe
{
    FillMode = WireFrame;
    CullMode = None;
};

// The terrain draws twice in the G-buffer pass (TerrainRenderer): DepthPrepass before the mesh batches (depth only:
// the terrain still hides what is behind hills from their pixel shaders), Shade after them (no depth write; the
// default GreaterEqual passes only where the terrain is still the nearest surface). Drawn in one pass before the
// batches, every terrain pixel ran the full splat shader, also those the batches then covered: a town plaza view
// paid ~1.4 ms for terrain it hardly showed (2026-10-16).
// Opaque is the single-pass draw (kept for anything that draws the terrain without the late queue).
// DepthPrepass_Holes / Shadow_Holes are what a terrain with holes (HoleStamp) draws instead of DepthPrepass / Shadow:
// a pixel shader that can discard writes its depth late, so a terrain without holes keeps the variants without one.
// The shading passes (Opaque, Shade) test for holes in either case: they write no depth (Shade) or run the full PS anyway.
technique11 GPUTerrain
{
    pass Opaque
    {
        SetRasterizerState(Wireframe);
        SetVertexShader(CompileShader(vs_6_6, VS()));
        SetPixelShader(CompileShader(ps_6_6, PS()));
    }
    pass DepthPrepass
    {
        SetRasterizerState(Wireframe);
        SetVertexShader(CompileShader(vs_6_6, VS()));
        SetPixelShader(CompileShader(ps_6_6, PS_DepthOnly()));
        SetRenderState(RenderTargets=6, Blend=NoColor)
    }
    pass Shade
    {
        SetRasterizerState(Wireframe);
        SetVertexShader(CompileShader(vs_6_6, VS()));
        SetPixelShader(CompileShader(ps_6_6, PS()));
        SetRenderState(RenderTargets=6, DepthWrite=false)
    }
    pass Shadow
    {
        SetVertexShader(CompileShader(vs_6_6, VS_Shadow()));
        SetPixelShader(CompileShader(ps_6_6, PS_Shadow()));
    }
    pass DepthPrepass_Holes
    {
        SetRasterizerState(Wireframe);
        SetVertexShader(CompileShader(vs_6_6, VS()));
        SetPixelShader(CompileShader(ps_6_6, PS_DepthOnly_Holes()));
        SetRenderState(RenderTargets=6, Blend=NoColor)
    }
    pass Shadow_Holes
    {
        SetVertexShader(CompileShader(vs_6_6, VS_Shadow()));
        SetPixelShader(CompileShader(ps_6_6, PS_Shadow_Holes()));
    }
}

// Minimal depth-only PS — required for SV_RenderTargetArrayIndex propagation
void PS_Shadow(ShadowVSOutput input) { }

// Depth prepass: no colour (Blend=NoColor masks the G-buffer targets)
void PS_DepthOnly(VertexOutput input) { }

// The same where a hole opens the terrain: no depth (the cave below shows) and no shadow (light falls into it)
void PS_Shadow_Holes(ShadowVSOutput input)
{
    if (InTerrainHole(input.WorldXZ)) discard;
}

void PS_DepthOnly_Holes(VertexOutput input)
{
    if (InTerrainHole(input.WorldPos.xz)) discard;
}
