// decoration_prepass.hlsl — Build decoration control texture from per-slot density maps
// SM 6.6 bindless, push constants at b3
//
// Iterates all decoration slots, reads each slot's density map (via DecoMapSlice),
// finds the top 8 by weight, and packs (slotIndex, weight) into a 2-slice RGBA16_UINT texture.
// Each channel: (slotIndex << 8) | weight. Unused slots: 0xFFFF.

#pragma kernel CSBuildDecoControl

// Inlined from common.fx (runtime Shader() compilation has no include path)
struct PushConstantsData { uint4 indices[8]; };
#define GET_INDEX(i) PushConstants.indices[i/4][i%4]
ConstantBuffer<PushConstantsData> PushConstants : register(b3);

#define DecoMapsIdx     GET_INDEX(0)   // SRV: density map Texture2DArray
#define SlotsIdx        GET_INDEX(1)   // SRV: DecoratorSlot structured buffer
#define ControlUAVIdx   GET_INDEX(2)   // UAV: output RWTexture2DArray<uint4>
#define SlotCountIdx    GET_INDEX(3)   // number of decorator slots

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
    uint DecoMapSlice;
    uint _pad0;
    uint Mode;
    uint TextureIdx;
    float3 HealthyColor;
    float3 DryColor;
    float NoiseSpread;
    uint _colorPad;
};

// Sampler for density maps — block-compressed textures (BC1) don't support Load()
SamplerState ClampSampler : register(s2);

[numthreads(8, 8, 1)]
void CSBuildDecoControl(uint3 dtid : SV_DispatchThreadID)
{
    RWTexture2DArray<uint4> controlTex = ResourceDescriptorHeap[ControlUAVIdx];
    Texture2DArray decoMaps = ResourceDescriptorHeap[DecoMapsIdx];
    StructuredBuffer<DecoratorSlot> slots = ResourceDescriptorHeap[SlotsIdx];

    uint2 dims;
    uint slices;
    decoMaps.GetDimensions(dims.x, dims.y, slices);

    if (dtid.x >= dims.x || dtid.y >= dims.y) return;

    // Normalized UV for SampleLevel (texel center)
    float2 uv = (float2(dtid.xy) + 0.5) / float2(dims);

    // Collect all valid slots, then keep top 8 sorted by weight (descending)
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

        float val = 0.0;
        if (s.DecoMapSlice != 0xFFFFFFFF)
            val = decoMaps.SampleLevel(ClampSampler, float3(uv, s.DecoMapSlice), 1).r;  // mip 1: smooth density edges
        else
            continue;

        if (s.LODCount == 0) continue;

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
