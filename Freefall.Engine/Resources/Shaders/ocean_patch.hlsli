// The ocean's grid as ocean.fx and ocean_lip.fx draw it: their push constants, includes, the screen-space tessellation
// and patch culling, and the control points (VS_Control). Moved out of ocean.fx for the lip (2026-10-19).
#ifndef OCEAN_PATCH_HLSLI
#define OCEAN_PATCH_HLSLI

cbuffer PushConstants : register(b3)
{
    // Slots 0-1: set by the forward pass for every forward batch
    uint ShadowMapIdx;              // 0: Shadow cascade array SRV
    uint _reserved1;                // 1: CompositeSnapshot SRV (taken from OceanData instead)
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
    uint OceanDataIdx;              // 9: Per-instance OceanData buffer
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

#include "ocean_surf.hlsli"

// Tessellation params
static const float TessMinFactor = 1.0;
static const float TessMaxFactor = 48.0;
static const float TargetPixelsPerEdge = 10.0; // tessellate until edges are ~10 pixels

// Screen-space edge length tessellation heuristic
float ScreenSpaceEdgeFactor(float3 p0, float3 p1)
{
    // Project both endpoints to clip space
    float4 clip0 = mul(mul(float4(p0, 1.0), View), Projection);
    float4 clip1 = mul(mul(float4(p1, 1.0), View), Projection);

    // NDC [-1,1] → screen pixels
    // Use Projection[0][0] for horizontal FOV scale
    float2 screen0 = clip0.xy / clip0.w;
    float2 screen1 = clip1.xy / clip1.w;

    // Get viewport size from projection matrix
    // ViewportWidth ≈ 2 / Projection[0][0], but we just need relative pixel count
    float2 pixelScale = float2(abs(Projection[0][0]), abs(Projection[1][1])) * 512.0;
    float2 diff = (screen1 - screen0) * pixelScale;
    float edgePixels = length(diff);

    return clamp(edgePixels / TargetPixelsPerEdge, TessMinFactor, TessMaxFactor);
}

// How far the waves can move a point of the grid, in every direction (metres): patch bounds are widened by it
static const float PatchDisplacementBound = 20.0;

// Whether a patch can be left out (tessellation factors 0): its box, widened by the waves, lies outside the
// view, or behind the opaque scene in the Hi-Z pyramid. This used to test only "entirely behind the camera",
// so every patch in front of the camera plane was tessellated and displaced, on screen or not, hidden behind
// terrain or not (0.74 ms with no water in view on Eldermoor, 2026-10-15).
bool CullTriangle(float3 p0, float3 p1, float3 p2, uint hiZSrv)
{
    float3 minP = min(min(p0, p1), p2) - PatchDisplacementBound;
    float3 maxP = max(max(p0, p1), p2) + PatchDisplacementBound;

    // Clip-space corners; frustum planes as half-spaces (valid for corners behind the camera too)
    uint outside = 0x1F;
    float2 ndcMin = 1e30, ndcMax = -1e30;
    float nearestW = 1e30;
    bool reachesBehind = false;
    [unroll]
    for (uint i = 0; i < 8; i++)
    {
        float3 c = float3((i & 1) ? maxP.x : minP.x, (i & 2) ? maxP.y : minP.y, (i & 4) ? maxP.z : minP.z);
        float4 clip = mul(mul(float4(c, 1.0), View), Projection);

        uint beyond = 0;
        if (clip.x < -clip.w) beyond |= 1;
        if (clip.x >  clip.w) beyond |= 2;
        if (clip.y < -clip.w) beyond |= 4;
        if (clip.y >  clip.w) beyond |= 8;
        if (clip.w <= 0.0)    beyond |= 16;
        outside &= beyond;

        if (clip.w <= 0.01) reachesBehind = true;
        else
        {
            float2 ndc = clip.xy / clip.w;
            ndcMin = min(ndcMin, ndc);
            ndcMax = max(ndcMax, ndc);
            nearestW = min(nearestW, clip.w);
        }
    }

    if (outside != 0) return true;              // all corners beyond one plane of the view
    if (hiZSrv == 0 || reachesBehind) return false;

    // Hi-Z: linear view depth (clip w), max over each texel's footprint, sky = FLT_MAX (never occludes).
    // At the mip where the box's screen rectangle spans at most one texel, it touches at most 2x2 texels;
    // it is hidden if even its nearest point is behind the farthest opaque depth there.
    ndcMin = max(ndcMin, -1.0);
    ndcMax = min(ndcMax, 1.0);
    float2 uvMin = float2(ndcMin.x, -ndcMax.y) * 0.5 + 0.5;
    float2 uvMax = float2(ndcMax.x, -ndcMin.y) * 0.5 + 0.5;

    Texture2D<float> hiZ = ResourceDescriptorHeap[hiZSrv];
    float w, h, levels;
    hiZ.GetDimensions(0, w, h, levels);
    float2 extent = (uvMax - uvMin) * float2(w, h);
    uint mip = (uint)min(ceil(log2(max(max(extent.x, extent.y), 1.0))), levels - 1.0);

    float mw, mh, unused;
    hiZ.GetDimensions(mip, mw, mh, unused);
    int2 maxCoord = int2(mw, mh) - 1;
    int2 t0 = clamp(int2(uvMin * float2(mw, mh)), int2(0, 0), maxCoord);
    int2 t1 = clamp(int2(uvMax * float2(mw, mh)), int2(0, 0), maxCoord);

    float occluder = max(max(hiZ.Load(int3(t0.x, t0.y, mip)), hiZ.Load(int3(t1.x, t0.y, mip))),
                         max(hiZ.Load(int3(t0.x, t1.y, mip)), hiZ.Load(int3(t1.x, t1.y, mip))));
    return nearestW > occluder;
}

// ═══════════════════════════════════════════════════════════════════════
// CONTROL POINT SHADER — pass world position through
// ═══════════════════════════════════════════════════════════════════════

struct ControlPoint
{
    float3 WorldPos : WORLDPOS;
    nointerpolation uint InstanceIdx : TEXCOORD0;
};

ControlPoint VS_Control(uint primitiveVertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
    ControlPoint output;

    StructuredBuffer<InstanceDescriptor> descriptors = ResourceDescriptorHeap[DescriptorBufIdx];
    StructuredBuffer<row_major matrix> globalTransforms = ResourceDescriptorHeap[GlobalTransformBufferIdx];
    StructuredBuffer<uint> sortedIndices = ResourceDescriptorHeap[SortedIndicesIdx];

    // Look up mesh buffer indices from MeshRegistry
    StructuredBuffer<MeshPartEntry> meshRegistry = ResourceDescriptorHeap[MeshRegistryIdx];
    MeshPartEntry part = meshRegistry[MeshPartId];

    StructuredBuffer<uint> indices = ResourceDescriptorHeap[part.IndexBufferIdx];
    uint vertexID = indices[primitiveVertexID + part.BaseIndex];
    StructuredBuffer<float3> positions = ResourceDescriptorHeap[part.PosBufferIdx];

    uint dataPos = InstanceBaseOffset + instanceID;
    uint packedIdx = sortedIndices[dataPos];
    uint idx = packedIdx & 0x7FFFFFFFu;

    InstanceDescriptor desc = descriptors[idx];
    row_major matrix World = globalTransforms[desc.TransformSlot];

    output.WorldPos = mul(float4(positions[vertexID], 1.0f), World).xyz;
    output.InstanceIdx = idx;

    return output;
}

struct HullConstantOutput
{
    float EdgeTess[3] : SV_TessFactor;
    float InsideTess : SV_InsideTessFactor;
};

#endif
