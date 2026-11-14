cbuffer PushConstants : register(b3)
{
    // Slots 0-1: Reserved for light/composition passes
    uint _reserved0;
    uint _reserved1;
    // Slots 2-3: PER-DRAW (command signature writes these)
    uint MeshPartId;                // 2: Index into MeshRegistry
    uint InstanceBaseOffset;        // 3: Base offset for instance ID (per-command)
    // Slots 4-8: PER-BATCH (set before ExecuteIndirect)
    uint DescriptorBufIdx;          // 4: StructuredBuffer<InstanceDescriptor>
    uint SortedIndicesIdx;          // 5: StructuredBuffer<uint> - sorted draw order indices
    uint MeshRegistryIdx;           // 6: StructuredBuffer<MeshPartEntry>
    uint MaterialsIdx;              // 7: Index to materials buffer
    uint GlobalTransformBufferIdx;  // 8: Index to global TransformBuffer
    // Slots 9-15: Reserved
    uint _reserved9;
    uint _reserved10;
    uint _reserved11;
    uint _reserved12;
    uint _reserved13;
    uint _reserved14;
    uint _reserved15;
    // Slot 16: Debug
    uint DebugMode;             // 16: Debug visualization mode
    uint IndexedDraw;           // 17: 1 = an indexed draw, the vertex index is SV_VertexID (common.fx MeshVertexIndex)
    uint _reserved18;
    uint _reserved19;
    // Slots 20-21: Shadow pass
    uint ExpansionBufferIdx;    // 20: SRV: expansion buffer (cascadeIdx<<30 | instanceIdx)
    uint CascadeBufferSRVIdx;   // 21: SRV: StructuredBuffer<CascadeData>
};

#include "common.fx"
#include "far_forest.hlsli"
// @RenderState(RenderTargets=6, CullMode=None, LodFade=true)

// Crossmesh/Far-LOD Tree GBuffer shader:
// - Two-sided rendering (CullMode=None) for cross-planes
// - Alpha clip on albedo alpha
// - Baked AO texture support (softened to avoid crushing)
// - The crown's volume as normal (out of its centre, as gbuffer_foliage.fx) and depth (an ellipsoid facing the camera),
//   a normal map adding the leaf clumps on top
// - Low-frequency trunk sway (matches gbuffer_trunk.fx)
// - Vegetation flag (data.a = 0.5) for wrap lighting

inline MaterialData GetMaterial(uint id)
{
    StructuredBuffer<MaterialData> materials = ResourceDescriptorHeap[MaterialsIdx];
    return materials[id];
}
#define GET_MATERIAL(id) GetMaterial(id)

struct VSOutput
{
    float4 Position : SV_POSITION;
    float2 TexCoord : TEXCOORD0;
    float4 WorldPos : TEXCOORD1;
    nointerpolation uint MaterialID : TEXCOORD2;
    float Depth : TEXCOORD3;
    nointerpolation uint TransformSlot : TEXCOORD4;
    nointerpolation uint MeshPartIdx : TEXCOORD5;
    nointerpolation float3 TreeTint : TEXCOORD6;   // this tree's colour (common.fx TreeTint), as its nearer LODs
    nointerpolation uint LodFade : TEXCOORD7;      // DrawEntryFade (0 = no LOD crossfade)
    float3 VolumeNormal : TEXCOORD8;               // out of the crown's centre: every card's normal, a normal map on top
    nointerpolation float4 Crown : TEXCOORD9;      // the crown's centre (world) and half width: the depth every card
    nointerpolation float2 CrownSpan : TEXCOORD10; // writes; and its half height, its centre's view depth
};


// A card's vertex placed and shaded by its tree: World is the tree's (an instance's, a far forest tree's) or the merged
// far mesh's frame (its vertices name their trees, MESH_CARD_ORIGINS). Leaves the material, the ids and the fade to
// the caller.
VSOutput CardVertex(MeshPartEntry part, uint vertexID, row_major matrix World)
{
    VSOutput output;
    StructuredBuffer<float3> positions = ResourceDescriptorHeap[part.PosBufferIdx];
    StructuredBuffer<float3> normals = ResourceDescriptorHeap[part.NormBufferIdx];
    StructuredBuffer<float2> uvs = ResourceDescriptorHeap[part.UVBufferIdx];

    float3 pos = positions[vertexID];
    float3 norm = normals[vertexID];
    float2 uv = uvs[vertexID];

    float4 worldPos = mul(float4(pos, 1.0f), World);

    // The tree this card belongs to: the instance's, or in a merged far mesh (Mesh.CardOrigins) the one its vertex names
    float3 treeOrigin = (part.Flags & MESH_CARD_ORIGINS) != 0 ? mul(float4(norm, 1.0f), World).xyz
                                                              : float3(World._41, World._42, World._43);

    // Trunk sway — stronger at height, zero at base
    // The whole tree bends about its base in the scene wind (TreeSway, common.fx); foliage rides the same bend
    worldPos.xyz += TreeSway(worldPos.xyz, treeOrigin);

    output.WorldPos = worldPos;
    output.Position = mul(mul(worldPos, View), Projection);

    output.TreeTint = TreeTint(treeOrigin);

    // The crown's volume, as gbuffer_foliage.fx's fake one: out of the crown's centre (the card part's bounds, or the
    // centre a merged far mesh's tangent stream names), biased up. (A dome out of the tree's base, as cards with a
    // normal map had until 2026-11-14, faces up all over a tall crown and lights it flat.)
    float3 crownLocal = part.LocalBounds.xyz;
    if ((part.Flags & MESH_CARD_ORIGINS) != 0)
    {
        StructuredBuffer<float4> tangents = ResourceDescriptorHeap[part.TanBufferIdx];
        crownLocal = tangents[vertexID].xyz;
    }
    float3 crown = mul(float4(crownLocal, 1.0f), World).xyz;
    output.VolumeNormal = normalize(lerp(float3(0, 1, 0), normalize(worldPos.xyz - crown + float3(0, 1e-4, 0)), 0.6));

    // The crown the depth is drawn from (PS): a card's corners lie on its edges, so this corner's offset from the centre
    // is the card's half width and half height (nointerpolation: the triangle's first corner says it for all of it)
    float3 reach = mul(float4(pos, 1.0f), World).xyz - crown;
    output.Crown = float4(crown, length(reach.xz));
    output.CrownSpan = float2(abs(reach.y), mul(mul(float4(crown, 1.0), View), Projection).w);

    output.TexCoord = uv;
    output.TexCoord.y = 1 - output.TexCoord.y;
    output.Depth = output.Position.w;
    output.MaterialID = 0;
    output.TransformSlot = 0;
    output.MeshPartIdx = 0;
    output.LodFade = 0;
    return output;
}

// The far tier (common.fx FarTierShows) for a card of the tree at treeOrigin: false = collapse it, else its fade word
// (0 = keep the LOD crossfade's)
void FarTierCard(inout VSOutput output, float3 treeOrigin, bool farSide)
{
    uint fade;
    if (!FarTierShows(FarTierCellOf(treeOrigin), farSide, fade))
        output.Position = float4(0.0, 0.0, 0.0, 1.0);
    else if (fade != 0u)
        output.LodFade = fade;
}

VSOutput VS(uint primitiveVertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
    StructuredBuffer<InstanceDescriptor> descriptors = ResourceDescriptorHeap[DescriptorBufIdx];
    StructuredBuffer<row_major matrix> globalTransforms = ResourceDescriptorHeap[GlobalTransformBufferIdx];
    StructuredBuffer<uint> sortedIndices = ResourceDescriptorHeap[SortedIndicesIdx];

    StructuredBuffer<MeshPartEntry> meshRegistry = ResourceDescriptorHeap[MeshRegistryIdx];
    MeshPartEntry part = meshRegistry[MeshPartId];
    uint vertexID = MeshVertexIndex(part, primitiveVertexID, primitiveVertexID, IndexedDraw);

    uint packedIdx = sortedIndices[InstanceBaseOffset + instanceID];
    uint idx = DrawEntryIndex(packedIdx);   // and the LOD crossfade's tags (lod_fade.hlsli)
    InstanceDescriptor desc = descriptors[idx];
    row_major matrix World = globalTransforms[desc.TransformSlot];

    VSOutput output = CardVertex(part, vertexID, World);
    output.MaterialID = desc.MaterialId;
    output.TransformSlot = desc.TransformSlot;
    output.MeshPartIdx = desc.MeshPartIdx;
    output.LodFade = DrawEntryFade(packedIdx);

    // The far tier: a merged far mesh (MESH_CELL_FADE) and the instances of its near side (DRAW_FAR_TIER) dither into
    // each other by the tree's cell; where one side does not show, its triangles collapse to a point
    bool farSide = (part.Flags & MESH_CELL_FADE) != 0;
    if (farSide || (desc.MeshPartIdx & DRAW_FAR_TIER) != 0)
    {
        StructuredBuffer<float3> normals = ResourceDescriptorHeap[part.NormBufferIdx];
        float3 treeOrigin = (part.Flags & MESH_CARD_ORIGINS) != 0 ? mul(float4(normals[vertexID], 1.0f), World).xyz
                                                                  : float3(World._41, World._42, World._43);
        FarTierCard(output, treeOrigin, farSide);
    }
    return output;
}

// The far forest (Graphics/FarForest.cs, far_forest.hlsli): a tree the cull put on its kind's list. Push constants
// reused: InstanceBaseOffset = the kind's list start (per draw), DescriptorBufIdx = the trees, SortedIndicesIdx = the
// visible list, GlobalTransformBufferIdx = the kinds. Always the far side of the far tier.
VSOutput VS_FarForest(uint primitiveVertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
    StructuredBuffer<FarTree> trees = ResourceDescriptorHeap[DescriptorBufIdx];
    StructuredBuffer<uint> visible = ResourceDescriptorHeap[SortedIndicesIdx];
    StructuredBuffer<FarKind> kinds = ResourceDescriptorHeap[GlobalTransformBufferIdx];

    StructuredBuffer<MeshPartEntry> meshRegistry = ResourceDescriptorHeap[MeshRegistryIdx];
    MeshPartEntry part = meshRegistry[MeshPartId];
    uint vertexID = MeshVertexIndex(part, primitiveVertexID, primitiveVertexID, IndexedDraw);

    uint entry = visible[InstanceBaseOffset + instanceID];   // tree | growth (far_forest.hlsli thinning)
    FarTree tree = trees[entry & 0xFFFFFFu];
    row_major matrix World = FarGrow(FarTreeWorld(tree), FarGrowOf(entry >> 24, 255u));

    VSOutput output = CardVertex(part, vertexID, World);
    output.MaterialID = kinds[FarTreeKind(tree)].MaterialId;
    FarTierCard(output, tree.Position, true);
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

SamplerState Sampler : register(s0);

PSOutput PS(VSOutput input)
{
    PSOutput output;
    LodFadeClip(input.Position.xy, input.LodFade);

    MaterialData mat = GET_MATERIAL(input.MaterialID);
    Texture2D albedoTex = ResourceDescriptorHeap[NonUniformResourceIndex(mat.AlbedoIdx)];

    float4 color = albedoTex.Sample(Sampler, input.TexCoord);
    color.rgb *= mat.Tint * input.TreeTint; // Material Tint (default white), this tree's own colour
    clip(color.a - 0.2f);

    // Read AO map — encodes canopy depth:
    // bright = outer surface, dark = deep interior
    float aoValue = 1.0;
    if (mat.AOIdx != 0)
    {
        Texture2D aoTex = ResourceDescriptorHeap[NonUniformResourceIndex(mat.AOIdx)];
        aoValue = aoTex.Sample(Sampler, input.TexCoord).g;
    }

    // ── Normal: the crown's volume + the normal map's leaf clumps + leaf noise ──
    float3 domeN = normalize(input.VolumeNormal);

    // Per-pixel leaf noise: randomize normal direction so adjacent pixels
    // light differently, breaking the flat-plane uniformity
    float n1 = InterleavedGradientNoise(input.Position.xy);
    float n2 = InterleavedGradientNoise(input.Position.xy + float2(37.0, 17.0));
    float3 leafNoise = float3(n1 * 2.0 - 1.0, n2 * 2.0 - 1.0, 0) * 0.35;

    float3 N = domeN;

    if (mat.NormalIdx != 0)
    {
        // Normal map adds leaf cluster detail on top of the dome shape
        // Use cotangent frame to transform tangent-space normal map
        float3 dp1 = ddx(input.WorldPos.xyz);
        float3 dp2 = ddy(input.WorldPos.xyz);
        float2 duv1 = ddx(input.TexCoord);
        float2 duv2 = ddy(input.TexCoord);

        float3 dp2perp = cross(dp2, domeN);
        float3 dp1perp = cross(domeN, dp1);
        float3 T = dp2perp * duv1.x + dp1perp * duv2.x;
        float3 B = dp2perp * duv1.y + dp1perp * duv2.y;

        float handedness = dot(cross(T, B), domeN) < 0.0 ? -1.0 : 1.0;
        T *= handedness;

        float invmax = rsqrt(max(dot(T, T), dot(B, B)));
        float3x3 TBN = float3x3(T * invmax, B * invmax, domeN);

        Texture2D normalTex = ResourceDescriptorHeap[NonUniformResourceIndex(mat.NormalIdx)];
        float2 nXY = normalTex.Sample(Sampler, input.TexCoord).rg * 2.0 - 1.0;
        nXY.y = -nXY.y;
        float3 texNormal = float3(nXY, sqrt(max(0.001, 1.0 - dot(nXY, nXY))));

        N = normalize(mul(texNormal, TBN));
    }

    // Blend leaf noise into normal for per-pixel breakup
    N = normalize(N + leafNoise);

    // Subtle interior darkening — just enough for depth, not crushing
    float ao = lerp(1.0, aoValue, 0.5);

    // How much of the falling snow this pixel takes (common.fx FarTreeFlag, read by the decal pass): the card shows the
    // crown from the side, its outer boughs, the shell the snow lies on, and the mask's occlusion (green, 0..1: shut in
    // .. open; the generated trees' sky visibility, build_tree.py, stored in 0..0.15 until 2026-11-14) tells them from
    // the gaps into it
    float snowExposure = smoothstep(0.2, 0.8, saturate(aoValue / 0.15));

    output.Albedo = float4(color.rgb, 0);
    // The lighting's occlusion (light_directional.fx, composition_cs.hlsl read Data.b) and translucency. It saturates
    // above 0.15: only a mask's darkest tenth goes dark, crisp holes between fully lit clusters, which is most of the
    // Sweetgum's texture. Spread over the whole range (lerp(0.4, 1, g)) the Sweetgum went darker and flatter
    // (pixel-to-pixel 12.3 -> 8.6, 2026-11-14), so it stays.
    output.Normal = float4(N, saturate(0.4 + aoValue * 4)); // AO as translucency for SSS backlighting
    output.Data = float4(0.7, 0.0, saturate(0.4 + aoValue * 4), FarTreeFlag(snowExposure)); // vegetation flag
    // The depth of the crown facing the camera (screen-space shadows, SSDM read it): an ellipsoid as wide and high as the
    // card and as deep as wide, the same from every side; the mask's occlusion sets the gaps back into it. The card's own
    // plane gave it away, seen edge-on a wall that shadowed the other card; blended from the trunk to the plane by the
    // occlusion (the Sweetgum's depth until 2026-11-14) a card turned half away put its lit rim behind its dark middle,
    // and the crown looked hollow.
    float3 toCrown = input.Crown.xyz - CamPos;
    float3 view = toCrown / max(length(toCrown), 1e-3);
    float3 across = normalize(cross(float3(0, 1, 0), view) + float3(1e-5, 0, 0));
    float3 upward = cross(view, across);
    float3 offset = input.WorldPos.xyz - input.Crown.xyz;
    float halfWidth = max(input.Crown.w, 0.1);
    float2 q = float2(dot(offset, across) / halfWidth, dot(offset, upward) / max(input.CrownSpan.x, 0.1));
    float bulge = sqrt(saturate(1.0 - dot(q, q)));
    output.Depth = input.CrownSpan.y - halfWidth * (bulge - 0.6 * (1.0 - aoValue));
    output.Displacement = float2(0, 0);
    output.EntityId = (input.TransformSlot << 8u) | (input.MeshPartIdx & 0xFFu) | (2u << 30);   // class 2: vegetation (decals_cs.hlsl receivers)
    return output;
}


// Shadow pass
struct ShadowVSOutput
{
    float4 Position : SV_POSITION;
    float2 TexCoord : TEXCOORD0;            // a card laid flat (TopDown): where in its square, -1..1 both ways
    nointerpolation uint MaterialID : TEXCOORD1;
    nointerpolation uint TopDown : TEXCOORD2;
    nointerpolation uint RTIndex : SV_RenderTargetArrayIndex;
    float4 ClipDistance : SV_ClipDistance0;
};

// Seen straight down (the cover map, ShelterMap; the sky captures' top-down slice) the cards are vertical and have no
// area: a far tree kept neither the snow nor the rain off the ground under it. There each card is laid flat at the
// crown's top as a square as wide as the card, and PS_Shadow keeps a disc of it, CROSS_CROWN_REACH of its half width
// (the crown at its widest; the side view does not say more). A part's bounds are its box's centre and half diagonal
// (ModelImporter), so a corner's height off the centre is half the card's height
#define CROSS_CROWN_REACH 0.75

bool StraightDown(row_major float4x4 vp)
{
    // The world's up moves nothing across the view, only in depth
    return abs(vp._21) + abs(vp._22) < 1e-4 * (abs(vp._11) + abs(vp._12) + abs(vp._31) + abs(vp._32));
}

// A card's corner (mesh-local) laid flat: its position, and where it lies in the card's square (-1..1 both ways)
float3 LayCardFlat(float3 pos, float3 centre, out float2 square)
{
    float2 along = pos.xz - centre.xz;                       // half the card's width, along it
    float halfWidth = max(length(along), 1e-4);
    float2 dir = along / halfWidth;
    float2 axis = dir.x > 0.01 || (dir.x >= -0.01 && dir.y > 0.0) ? dir : -dir;   // the same at both ends
    float side = pos.y > centre.y ? 1.0 : -1.0;              // the top edge to one side, the foot to the other
    square = float2(dot(dir, axis), side);
    return float3(pos.x - axis.y * side * halfWidth, centre.y + abs(pos.y - centre.y), pos.z + axis.x * side * halfWidth);
}

// A card's vertex in a shadow view (a sun cascade, or a point light face when local), for the tree World places (an
// instance's, a far forest tree's) or the merged far mesh's frame. Gives the tree's origin for the far tier.
ShadowVSOutput ShadowCardVertex(MeshPartEntry part, uint vertexID, row_major matrix World, uint view, bool local, out float3 treeOrigin)
{
    ShadowVSOutput output;
    output.Position = float4(0, 0, 0, 1);
    output.TexCoord = float2(0, 0);
    output.MaterialID = 0;
    output.TopDown = 0;
    output.RTIndex = 0;

    StructuredBuffer<float3> positions = ResourceDescriptorHeap[part.PosBufferIdx];
    StructuredBuffer<float2> uvs = ResourceDescriptorHeap[part.UVBufferIdx];

    float3 pos = positions[vertexID];
    output.TexCoord = uvs[vertexID];
    output.TexCoord.y = 1 - output.TexCoord.y;

    // A merged far mesh (Mesh.CardOrigins) names each card's tree origin and centre in its normal and tangent streams
    bool merged = (part.Flags & MESH_CARD_ORIGINS) != 0;
    float3 cardCentre = part.LocalBounds.xyz;
    treeOrigin = float3(World._41, World._42, World._43);
    if (merged)
    {
        StructuredBuffer<float3> normals = ResourceDescriptorHeap[part.NormBufferIdx];
        StructuredBuffer<float4> tangents = ResourceDescriptorHeap[part.TanBufferIdx];
        treeOrigin = mul(float4(normals[vertexID], 1.0f), World).xyz;
        cardCentre = tangents[vertexID].xyz;
    }

    if (!local)
    {
        StructuredBuffer<CascadeData> cascades = ResourceDescriptorHeap[CascadeBufferSRVIdx];
        if (StraightDown(cascades[view].VP))
        {
            pos = LayCardFlat(pos, cardCentre, output.TexCoord);
            output.TopDown = 1;
        }
    }
    float4 worldPos = mul(float4(pos, 1.0f), World);

    // Match trunk sway from VS
    // The whole tree bends about its base in the scene wind (TreeSway, common.fx); foliage rides the same bend
    worldPos.xyz += TreeSway(worldPos.xyz, treeOrigin);

    output.Position = ShadowClipPosition(worldPos, view, local, CascadeBufferSRVIdx, output.RTIndex, output.ClipDistance);
    return output;
}

// Sun cascades, or point light cube faces when local
ShadowVSOutput ShadowVertex(uint primitiveVertexID, uint instanceID, bool local)
{
    StructuredBuffer<uint> expansion = ResourceDescriptorHeap[ExpansionBufferIdx];
    uint view, idx;
    DecodeShadowEntry(expansion[InstanceBaseOffset + instanceID], local, view, idx);

    StructuredBuffer<row_major matrix> globalTransforms = ResourceDescriptorHeap[GlobalTransformBufferIdx];
    StructuredBuffer<MeshPartEntry> meshRegistry = ResourceDescriptorHeap[MeshRegistryIdx];
    MeshPartEntry part = meshRegistry[MeshPartId];
    uint vertexID = MeshVertexIndex(part, primitiveVertexID, primitiveVertexID, IndexedDraw);

    StructuredBuffer<InstanceDescriptor> descriptors = ResourceDescriptorHeap[DescriptorBufIdx];
    InstanceDescriptor desc = descriptors[idx];
    row_major matrix World = globalTransforms[desc.TransformSlot];

    float3 treeOrigin;
    ShadowVSOutput output = ShadowCardVertex(part, vertexID, World, view, local, treeOrigin);
    output.MaterialID = desc.MaterialId;

    // The far tier: the far mesh or the instances cast, switching at half way
    bool farSide = (part.Flags & MESH_CELL_FADE) != 0;
    if ((farSide || (desc.MeshPartIdx & DRAW_FAR_TIER) != 0) && !FarTierCasts(FarTierCellOf(treeOrigin), farSide))
        output.Position = float4(0.0, 0.0, 0.0, 1.0);
    return output;
}

// The far forest in the sun's cascades: an entry of its kind's shadow list is (cascade, growth, tree; far_forest.hlsli), written by
// far_forest_cull.hlsl CSCullShadow for the trees that cast (the far tier's far side past half way). Push constants as
// VS_FarForest, the list in SortedIndicesIdx.
ShadowVSOutput VS_FarForestShadow(uint primitiveVertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
    StructuredBuffer<FarTree> trees = ResourceDescriptorHeap[DescriptorBufIdx];
    StructuredBuffer<uint> entries = ResourceDescriptorHeap[SortedIndicesIdx];
    StructuredBuffer<FarKind> kinds = ResourceDescriptorHeap[GlobalTransformBufferIdx];

    StructuredBuffer<MeshPartEntry> meshRegistry = ResourceDescriptorHeap[MeshRegistryIdx];
    MeshPartEntry part = meshRegistry[MeshPartId];
    uint vertexID = MeshVertexIndex(part, primitiveVertexID, primitiveVertexID, IndexedDraw);

    uint entry = entries[InstanceBaseOffset + instanceID];   // tree | growth | cascade (far_forest.hlsli thinning)
    FarTree tree = trees[entry & 0x3FFFFFu];
    row_major float4x4 world = FarGrow(FarTreeWorld(tree), FarGrowOf((entry >> 22) & 0x7Fu, 127u));

    float3 treeOrigin;
    ShadowVSOutput output = ShadowCardVertex(part, vertexID, world, entry >> 29, false, treeOrigin);
    output.MaterialID = kinds[FarTreeKind(tree)].MaterialId;
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

void PS_Shadow(ShadowVSOutput input)
{
    if (input.TopDown != 0)
    {
        if (dot(input.TexCoord, input.TexCoord) > CROSS_CROWN_REACH * CROSS_CROWN_REACH)
            discard;
        return;
    }
    MaterialData mat = GET_MATERIAL(input.MaterialID);
    Texture2D albedoTex = ResourceDescriptorHeap[NonUniformResourceIndex(mat.AlbedoIdx)];
    float alpha = albedoTex.Sample(Sampler, input.TexCoord).a;
    clip(alpha - 0.2f);
}

technique11 GBuffer
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

    // The far forest (Graphics/FarForest.cs): drawn by name only, with the G-buffer's state and the cascades'
    pass FarForest
    {
        SetVertexShader(CompileShader(vs_6_6, VS_FarForest()));
        SetPixelShader(CompileShader(ps_6_6, PS()));
    }

    pass Shadow_FarForest
    {
        SetVertexShader(CompileShader(vs_6_6, VS_FarForestShadow()));
        SetPixelShader(CompileShader(ps_6_6, PS_Shadow()));
    }
}
