// decoration_prepass.hlsl — Build decoration control texture from per-slot density maps
// SM 6.6 bindless, push constants at b3
//
// Iterates all decoration slots, reads each slot's density map (via DecoMapSlice),
// finds the top 8 by weight, and packs (slotIndex, weight) into a 2-slice RGBA16_UINT texture.
// Each channel: (slotIndex << 8) | weight. Unused slots: 0xFFFF.

// Inlined from common.fx (runtime Shader() compilation has no include path)
struct PushConstantsData { uint4 indices[8]; };
#define GET_INDEX(i) PushConstants.indices[i/4][i%4]
ConstantBuffer<PushConstantsData> PushConstants : register(b3);

#define DecoMapsIdx     GET_INDEX(0)   // SRV: density map Texture2DArray
#define SlotsIdx        GET_INDEX(1)   // SRV: DecoratorSlot structured buffer
#define ControlUAVIdx   GET_INDEX(2)   // UAV: output RWTexture2DArray<uint4>
#define SlotCount       GET_INDEX(3)   // number of decorator slots

// Must match DecoratorSlot in grass.fx
struct DecoratorSlot
{
    float Density;
    float MinH, MaxH;
    float MinW, MaxW;
    uint LODCount;
    uint LODTableOffset;
    float3x3 RootMat;
    float SlopeBias;
    uint DecoMapSlice;
    uint _pad0;
    uint Mode;
    uint TextureIdx;
};

// No sampler needed — direct Load with thread IDs as texel coordinates

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

    // Top-8 selection: find slots with highest density map values
    uint topIdx[8];
    uint topWt[8];
    [unroll] for (uint k = 0; k < 8; k++)
    {
        topIdx[k] = 255;   // sentinel: empty
        topWt[k] = 0;
    }

    uint count = min(SlotCount, 32);
    for (uint i = 0; i < count; i++)
    {
        DecoratorSlot s = slots[i];

        float val = 1.0;  // default: present everywhere (no density map = no spatial restriction)
        if (s.DecoMapSlice != 0xFFFFFFFF)
            val = decoMaps.Load(int4(dtid.xy, s.DecoMapSlice, 0)).r;

        uint weight = (uint)(val * 255.0 + 0.5);
        if (weight == 0) continue;

        // Insertion sort into top-8 (descending by weight, lower index wins ties)
        [unroll] for (uint j = 0; j < 8; j++)
        {
            if (weight > topWt[j] || (weight == topWt[j] && i < topIdx[j]))
            {
                // Shift down
                [unroll] for (uint m = 7; m > j; m--)
                {
                    topIdx[m] = topIdx[m - 1];
                    topWt[m] = topWt[m - 1];
                }
                topIdx[j] = i;
                topWt[j] = weight;
                break;
            }
        }
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
