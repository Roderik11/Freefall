// terrain_stamp_overlay.hlsl — Compute prepass for procedural auto-mask + non-destructive stamp overlays
// SM 6.6 bindless, push constants at b3
//
// CS_ProceduralMask: Evaluates height/slope auto-mask and writes procedural weights into packed ControlMaps.
// CS_SplatStamp:     Overlays splat weights onto packed ControlMapArray after procedural.
// CS_DecoStamp:      Modulates decoration control weights after CSBuildDecoControl.

#pragma kernel CS_ProceduralMask
#pragma kernel CS_SplatStamp
#pragma kernel CS_DecoStamp

#include "terrain_stamp_common.hlsli"

// Push constants (root parameter 0, register b3)
// Shared layout — not all kernels use all slots.
cbuffer PushConstants : register(b3)
{
    uint StampBufIdx;       // slot 0 — SRV: stamp descriptor buffer (SplatStamp/DecoStamp)
    uint OutputIdx;         // slot 1 — UAV: RWTexture2DArray<float4> (splat) or RWTexture2DArray<uint4> (deco)
    uint HeightTexIdx;      // slot 2 — SRV: heightmap (ProceduralMask)
    uint AutoMaskBufIdx;    // slot 3 — SRV: StructuredBuffer<LayerAutoMask> (ProceduralMask)
    uint HeightScaleIdx;    // slot 4 — float bits: MaxHeight / (2 * TerrainSize.x * HeightTexel)
    uint StampCount;        // slot 5 — stamp count (SplatStamp/DecoStamp) or LayerCount (ProceduralMask)
    float BrushRadius;      // slot 6 — repurposed as SplineBufSrvIdx via asuint
    uint Resolution;        // slot 7
    uint SliceCount;        // slot 8
    uint HeightTexelIdx;    // slot 9 — float bits: 1.0 / heightmapResolution
};

#define SplineBufSrvIdx asuint(BrushRadius)
#define HeightScale asfloat(HeightScaleIdx)
#define HeightTexel asfloat(HeightTexelIdx)
#define LayerCount StampCount

// ═══════════════════════════════════════
// ── CS_ProceduralMask ──
// ═══════════════════════════════════════

// Per-layer procedural auto-mask parameters (matches C# LayerAutoMask / gputerrain.fx)
struct LayerAutoMask
{
    float HeightMin, HeightMax;
    float SlopeMin, SlopeMax;
    float HeightBlend, SlopeBlend;
    float ProceduralWeight;
    float _pad;
};

SamplerState sampHeightFilter : register(s2); // ClampedBilinear2D

[numthreads(8, 8, 1)]
void CS_ProceduralMask(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x >= Resolution || dtid.y >= Resolution) return;

    Texture2D HeightTex = ResourceDescriptorHeap[HeightTexIdx];
    StructuredBuffer<LayerAutoMask> AutoMaskBuf = ResourceDescriptorHeap[AutoMaskBufIdx];
    RWTexture2DArray<float4> output = ResourceDescriptorHeap[OutputIdx];

    float2 uv = (float2(dtid.xy) + 0.5) / float2(Resolution, Resolution);

    // ControlMaps are Y-flipped relative to heightmap — flip for height sampling
    float2 heightUV = float2(uv.x, 1.0 - uv.y);

    // Sample height (normalized 0..1)
    float heightNorm = HeightTex.SampleLevel(sampHeightFilter, heightUV, 0).r;

    // Compute slope from heightmap finite differences (matches gputerrain.fx GetNormal)
    float4 h;
    h[0] = HeightTex.SampleLevel(sampHeightFilter, heightUV + float2(0, -HeightTexel), 0).r;
    h[1] = HeightTex.SampleLevel(sampHeightFilter, heightUV + float2(-HeightTexel, 0), 0).r;
    h[2] = HeightTex.SampleLevel(sampHeightFilter, heightUV + float2(HeightTexel, 0), 0).r;
    h[3] = HeightTex.SampleLevel(sampHeightFilter, heightUV + float2(0, HeightTexel), 0).r;

    // HeightScale = MaxHeight / (2 * TerrainSize.x * HeightTexel), pre-computed on CPU
    // (the central difference spans two texels)
    float heightScale = HeightScale;

    float3 normal;
    normal.x = (h[1] - h[2]) * heightScale;
    normal.z = (h[0] - h[3]) * heightScale;
    normal.y = 1.0;
    normal = normalize(normal);

    float slopeDeg = acos(saturate(normal.y)) * (180.0 / 3.14159265);

    // Read current packed weights
    uint clampedSliceCount = min(SliceCount, 8);
    float4 slices[8];
    for (uint s = 0; s < clampedSliceCount; s++)
        slices[s] = output[uint3(dtid.xy, s)];

    // Evaluate procedural auto-mask per layer
    for (uint layer = 0; layer < LayerCount; layer++)
    {
        LayerAutoMask mask = AutoMaskBuf[layer];
        if (mask.ProceduralWeight == 0) continue;

        float pmask = 1;
        pmask *= smoothstep(mask.SlopeMin - mask.SlopeBlend, mask.SlopeMin, slopeDeg);
        pmask *= smoothstep(mask.SlopeMax + mask.SlopeBlend, mask.SlopeMax, slopeDeg);
        pmask *= smoothstep(mask.HeightMin - mask.HeightBlend, mask.HeightMin, heightNorm);
        pmask *= smoothstep(mask.HeightMax + mask.HeightBlend, mask.HeightMax, heightNorm);

        uint si = layer / 4;
        uint ch = layer % 4;
        float existing = slices[si][ch];

        if (mask.ProceduralWeight > 0)
            slices[si][ch] = max(existing, pmask * mask.ProceduralWeight);
        else
            slices[si][ch] = pmask * abs(mask.ProceduralWeight) * (1.0 - existing);
    }

    // Write back
    for (uint s2 = 0; s2 < clampedSliceCount; s2++)
        output[uint3(dtid.xy, s2)] = slices[s2];
}

// ═══════════════════════════════════════
// ── CS_SplatStamp ──
// ═══════════════════════════════════════

// Per-stamp descriptor (matches C# SplatStampDescriptorGPU, 48 bytes)
struct SplatStampDescriptor
{
    float2 Center;
    float  Radius;
    float  Falloff;
    uint   SplinePointOffset;
    uint   SplinePointCount;
    float  NoiseFreq;
    float  NoiseAmp;
    uint   NoiseSeed;
    // Splat-specific
    uint   TargetLayer;      // layer index to paint
    float  Strength;         // paint strength 0..1
    uint   _pad;
};

[numthreads(8, 8, 1)]
void CS_SplatStamp(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x >= Resolution || dtid.y >= Resolution) return;

    RWTexture2DArray<float4> output = ResourceDescriptorHeap[OutputIdx];
    StructuredBuffer<SplatStampDescriptor> stamps = ResourceDescriptorHeap[StampBufIdx];
    StructuredBuffer<StampSplinePoint> splinePoints = ResourceDescriptorHeap[SplineBufSrvIdx];

    float2 uv = (float2(dtid.xy) + 0.5) / float2(Resolution, Resolution);

    // ControlMap space is Y-flipped relative to world UV — flip for stamp center comparison
    float2 stampUV = float2(uv.x, 1.0 - uv.y);

    // Read all current slice values
    float4 slices[8]; // max 32 layers (8 slices × 4 channels)
    uint clampedSliceCount = min(SliceCount, 8);
    for (uint s = 0; s < clampedSliceCount; s++)
        slices[s] = output[uint3(dtid.xy, s)];

    bool modified = false;

    for (uint i = 0; i < StampCount; i++)
    {
        SplatStampDescriptor stamp = stamps[i];

        float nearestH, nearestHalfW;
        float weight = EvaluateStampWeight(
            stampUV, stamp.Center, stamp.Radius, stamp.Falloff,
            stamp.SplinePointOffset, stamp.SplinePointCount,
            stamp.NoiseFreq, stamp.NoiseAmp, stamp.NoiseSeed,
            splinePoints, nearestH, nearestHalfW);

        if (weight <= 0) continue;

        float paintWeight = weight * stamp.Strength;

        // Target channel: layer / 4 = slice, layer % 4 = channel
        uint targetSlice = stamp.TargetLayer / 4;
        uint targetChannel = stamp.TargetLayer % 4;
        if (targetSlice >= clampedSliceCount) continue;

        // Lerp target channel toward 1.0 (adding paint)
        slices[targetSlice][targetChannel] = lerp(
            slices[targetSlice][targetChannel], 1.0, paintWeight);

        modified = true;
    }

    if (modified)
    {
        for (uint s = 0; s < clampedSliceCount; s++)
            output[uint3(dtid.xy, s)] = slices[s];
    }
}

// ═══════════════════════════════════════
// ── CS_DecoStamp ──
// ═══════════════════════════════════════

// Per-stamp descriptor (matches C# DecoStampDescriptorGPU, 48 bytes)
struct DecoStampDescriptor
{
    float2 Center;
    float  Radius;
    float  Falloff;
    uint   SplinePointOffset;
    uint   SplinePointCount;
    float  NoiseFreq;
    float  NoiseAmp;
    uint   NoiseSeed;
    // Deco-specific
    float  Density;          // 0 = suppress, 1 = no change, >1 = boost (clamped)
    uint   _pad0;
    uint   _pad1;
};

[numthreads(8, 8, 1)]
void CS_DecoStamp(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x >= Resolution || dtid.y >= Resolution) return;

    RWTexture2DArray<uint4> output = ResourceDescriptorHeap[OutputIdx];
    StructuredBuffer<DecoStampDescriptor> stamps = ResourceDescriptorHeap[StampBufIdx];
    StructuredBuffer<StampSplinePoint> splinePoints = ResourceDescriptorHeap[SplineBufSrvIdx];

    float2 uv = (float2(dtid.xy) + 0.5) / float2(Resolution, Resolution);

    // Compute combined density multiplier from all stamps
    float densityMul = 1.0;
    bool anyHit = false;

    for (uint i = 0; i < StampCount; i++)
    {
        DecoStampDescriptor stamp = stamps[i];

        float nearestH, nearestHalfW;
        float weight = EvaluateStampWeight(
            uv, stamp.Center, stamp.Radius, stamp.Falloff,
            stamp.SplinePointOffset, stamp.SplinePointCount,
            stamp.NoiseFreq, stamp.NoiseAmp, stamp.NoiseSeed,
            splinePoints, nearestH, nearestHalfW);

        if (weight <= 0) continue;

        // Lerp density multiplier toward stamp's target density
        densityMul = lerp(densityMul, stamp.Density, weight);
        anyHit = true;
    }

    if (!anyHit) return;

    // Modulate both deco control slices (8 slots total, packed as (slotIndex << 8) | weight)
    [unroll]
    for (uint slice = 0; slice < 2; slice++)
    {
        uint4 packed = output[uint3(dtid.xy, slice)];

        [unroll]
        for (uint c = 0; c < 4; c++)
        {
            uint slotIdx = (packed[c] >> 8) & 0xFF;
            uint w = packed[c] & 0xFF;

            if (slotIdx == 255 || w == 0) continue;

            // Scale weight by density multiplier
            float newW = float(w) * densityMul;
            packed[c] = (slotIdx << 8) | clamp((uint)(newW + 0.5), 0, 255);
        }

        output[uint3(dtid.xy, slice)] = packed;
    }
}
