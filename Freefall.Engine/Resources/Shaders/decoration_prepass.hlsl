// decoration_prepass.hlsl — Build decoration control texture from layer-driven + standalone decorators
// SM 6.6 bindless, push constants at b3
//
// For each texel: iterates all decoration slots, computes effective weight either from
// splatmap layer weights (layer-driven via SourceLayerMask) or from standalone density maps.
// Applies exclusion masks. Finds top 8 by weight, packs (slotIndex, weight) into RGBA16_UINT.
// Each channel: (slotIndex << 8) | weight. Unused slots: index=255, weight=0.

#pragma kernel CSBuildDecoControl

// Push constants (root parameter 0, register b3) — bindless indices only
cbuffer PushConstants : register(b3)
{
    uint SlotsIdx;          // slot 0 — SRV: DecoratorSlot structured buffer
    uint ControlUAVIdx;     // slot 1 — UAV: output RWTexture2DArray<uint4>
    uint SlotCountIdx;      // slot 2 — number of decorator slots
    uint ResolutionIdx;     // slot 3 — control texture width/height
    uint HeightTexIdx;      // slot 4 — SRV: baked heightmap for slope/height evaluation
    uint AutoMaskBufIdx;    // slot 5 — SRV: StructuredBuffer<LayerAutoMask> for layer weight computation
    uint LayerCountIdx;     // slot 6 — total number of texture layers
    uint ControlMapsIdx;    // slot 7 — SRV: packed Texture2DArray (RGBA, same as surface shader)
    uint MaxHeightIdx;      // slot 8 — MaxHeight as uint bits (asfloat on GPU)
    uint TerrainSizeXIdx;   // slot 9 — TerrainSize.x as uint bits
};

// Must match DecoratorSlot in grass.fx / grass_compute.hlsl
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
    uint DecoMapSlice;          // bindless SRV for standalone density map (0 = none)
    uint SourceLayerMask;       // which layers drive density (0 = standalone)
    uint Mode;
    uint TextureIdx;
    float3 HealthyColor;
    float3 DryColor;
    float NoiseSpread;
    uint ExclusionLayerMask;
    float ProceduralBlend;
};

// Must match LayerAutoMask in gputerrain.fx
struct LayerAutoMask
{
    float HeightMin, HeightMax;
    float SlopeMin, SlopeMax;
    float HeightBlend, SlopeBlend;
    float ProceduralWeight;
    float _pad;
};

// Sampler for density/height maps
SamplerState ClampSampler : register(s2);

[numthreads(8, 8, 1)]
void CSBuildDecoControl(uint3 dtid : SV_DispatchThreadID)
{
    RWTexture2DArray<uint4> controlTex = ResourceDescriptorHeap[ControlUAVIdx];
    StructuredBuffer<DecoratorSlot> slots = ResourceDescriptorHeap[SlotsIdx];

    uint resolution = ResolutionIdx;
    if (dtid.x >= resolution || dtid.y >= resolution) return;

    // Normalized UV for SampleLevel (texel center)
    float2 uv = (float2(dtid.xy) + 0.5) / float2(resolution, resolution);
    float2 flippedUv = float2(uv.x, 1.0 - uv.y);

    // ── Height and slope (for procedural auto-mask) ──
    float maxHeight = asfloat(MaxHeightIdx);
    float terrainSizeX = asfloat(TerrainSizeXIdx);
    Texture2D HeightTex = ResourceDescriptorHeap[HeightTexIdx];

    float heightNorm = HeightTex.SampleLevel(ClampSampler, flippedUv, 0).r;

    // Use heightmap's OWN resolution for neighbor offsets (may differ from control texture resolution)
    uint hmW, hmH;
    HeightTex.GetDimensions(hmW, hmH);
    float heightTexel = 1.0 / float(hmW);
    float4 h;
    h[0] = HeightTex.SampleLevel(ClampSampler, flippedUv + float2(0, -heightTexel), 0).r;
    h[1] = HeightTex.SampleLevel(ClampSampler, flippedUv + float2(-heightTexel, 0), 0).r;
    h[2] = HeightTex.SampleLevel(ClampSampler, flippedUv + float2(heightTexel, 0), 0).r;
    h[3] = HeightTex.SampleLevel(ClampSampler, flippedUv + float2(0, heightTexel), 0).r;
    float texelWorldSize = terrainSizeX * heightTexel;
    float heightScale = maxHeight / texelWorldSize;
    float3 n;
    n.z = (h[0] - h[3]) * heightScale;
    n.x = (h[1] - h[2]) * heightScale;
    n.y = 1.0f;
    n = normalize(n);
    float slopeDeg = acos(saturate(n.y)) * (180.0 / 3.14159265);

    // ── Sample layer weights from the SAME packed ControlMapArray the surface shader uses ──
    // This guarantees decoration weights exactly match the visual terrain splatmap result.
    StructuredBuffer<LayerAutoMask> autoMaskBuf = ResourceDescriptorHeap[AutoMaskBufIdx];
    Texture2DArray ControlMaps = ResourceDescriptorHeap[ControlMapsIdx];

    uint cmW, cmH, sliceCount;
    ControlMaps.GetDimensions(cmW, cmH, sliceCount);

    uint layerCount = LayerCountIdx;
    uint clampedLayerCount = min(layerCount, 32);

    // Compute raw per-layer weights — identical to gputerrain.fx PS
    float rawWeight[32];
    float effectiveWeight[32];

    for (uint si = 0; si < sliceCount; si++)
    {
        // Packed array was written by CS_PackChannels using dtid-based UV (no flip).
        // Prepass also uses dtid-based UV, so sample with uv (not flipped).
        float4 weights = ControlMaps.SampleLevel(ClampSampler, float3(uv, si), 0);

        for (uint sj = 0; sj < 4; sj++)
        {
            uint layerIdx = si * 4 + sj;
            if (layerIdx >= clampedLayerCount) break;

            float weight = weights[sj];

            // Procedural auto-mask: max(painted, procedural) — same as gputerrain.fx lines 276-283
            LayerAutoMask mask = autoMaskBuf[layerIdx];
            if (mask.ProceduralWeight != 0)
            {
                float pmask = 1;
                pmask *= smoothstep(mask.SlopeMin - mask.SlopeBlend, mask.SlopeMin, slopeDeg);
                pmask *= smoothstep(mask.SlopeMax + mask.SlopeBlend, mask.SlopeMax, slopeDeg);
                pmask *= smoothstep(mask.HeightMin - mask.HeightBlend, mask.HeightMin, heightNorm);
                pmask *= smoothstep(mask.HeightMax + mask.HeightBlend, mask.HeightMax, heightNorm);
                if (mask.ProceduralWeight > 0)
                    weight = max(weight, pmask * mask.ProceduralWeight);
                else
                    weight = pmask * abs(mask.ProceduralWeight) * (1.0 - weight);
            }

            rawWeight[layerIdx] = weight;
            effectiveWeight[layerIdx] = 0;
        }
    }

    // ── Simulate sequential lerp to derive effective weights ──
    // Exactly mirrors: color = lerp(color, layerColor[i], weight[i])
    // After each lerp, all prior layers' contributions scale by (1 - w).
    for (uint li = 0; li < clampedLayerCount; li++)
    {
        float w = rawWeight[li];
        if (w <= 0) continue;

        // Scale down all previous layers' contributions
        for (uint p = 0; p < li; p++)
            effectiveWeight[p] *= (1.0 - w);

        // This layer claims w of the final result
        effectiveWeight[li] = w;
    }

    // ── Collect top 8 decorator slots by weight (descending) ──
    uint topIdx[8];
    uint topWt[8];
    [unroll] for (uint k = 0; k < 8; k++)
    {
        topIdx[k] = 255;
        topWt[k] = 0;
    }
    uint topCount = 0;
    uint count = min(SlotCountIdx, 32);
    for (uint i = 0; i < count; i++)
    {
        DecoratorSlot s = slots[i];
        if (s.LODCount == 0) continue;

        // Painted density (standalone ControlMap)
        float painted = 0;
        if (s.DecoMapSlice != 0)
        {
            Texture2D<float> densityMap = ResourceDescriptorHeap[s.DecoMapSlice];
            painted = densityMap.SampleLevel(ClampSampler, uv, 0).r;
        }

        // Layer-driven procedural weight
        float procedural = 0;
        if (s.SourceLayerMask != 0)
        {
            for (uint li2 = 0; li2 < clampedLayerCount; li2++)
            {
                if (s.SourceLayerMask & (1u << li2))
                    procedural = max(procedural, effectiveWeight[li2]);
            }
        }

        // Blend: positive = max(painted, procedural * blend), negative = procedural * |blend| * (1 - painted)
        float blend = s.ProceduralBlend;
        float val;
        if (blend >= 0)
            val = max(painted, procedural * blend);
        else
            val = procedural * abs(blend) * (1.0 - painted);
        if (val <= 0) continue;

        // Apply exclusion layers (use raw weight — exclusion checks presence, not visual dominance)
        if (s.ExclusionLayerMask != 0)
        {
            float exclusion = 0;
            for (uint ei = 0; ei < clampedLayerCount; ei++)
            {
                if (s.ExclusionLayerMask & (1u << ei))
                    exclusion = max(exclusion, rawWeight[ei]);
            }
            val *= (1.0 - exclusion);
        }

        uint weight = (uint)(val * 255.0 + 0.5);
        if (weight == 0) continue;

        // Insertion sort: find position and shift down
        uint pos = topCount < 8 ? topCount : 7;
        if (topCount >= 8 && weight <= topWt[7])
            continue;  // not heavy enough to make top 8

        // Find insertion point (descending order)
        [unroll] for (uint j = 0; j < 8; j++)
        {
            if (j < topCount && weight > topWt[j])
            {
                pos = j;
                break;
            }
        }

        // Shift entries down to make room
        [unroll] for (uint j = 7; j > 0; j--)
        {
            if (j > pos)
            {
                topIdx[j] = topIdx[j - 1];
                topWt[j]  = topWt[j - 1];
            }
        }

        topIdx[pos] = i;
        topWt[pos]  = weight;
        if (topCount < 8) topCount++;
    }

    // Pack: (slotIndex << 8) | weight
    uint4 packed0, packed1;
    [unroll] for (uint c = 0; c < 4; c++)
    {
        packed0[c] = (topIdx[c] << 8) | topWt[c];
        packed1[c] = (topIdx[c + 4] << 8) | topWt[c + 4];
    }

    controlTex[uint3(dtid.xy, 0)] = packed0;
    controlTex[uint3(dtid.xy, 1)] = packed1;
}
