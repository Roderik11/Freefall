// LOD crossfade (InstanceBatch; cull_instances.hlsl CSVisibility): while an instance changes LOD both parts draw for a
// moment, each on its share of the pixels (LodFadeClip in common.fx). A camera draw-list entry carries that share beside
// the instance index. Only batches of effects that declare it (LodFade=true in their render state line) get such
// entries. Never quote that line here: FXParser.ParseRenderState takes the first one in the effect with its includes.
#ifndef LOD_FADE_HLSLI
#define LOD_FADE_HLSLI

#define DRAW_FADE_BIT       0x80000000u     // the instance is in a LOD fade
#define DRAW_FADE_OUT_BIT   0x40000000u     // ... and this entry draws the part it fades from
#define DRAW_FADE_SHIFT     23              // bits 23-29: the new part's share of the pixels, 1..127 (128ths)
#define DRAW_INDEX_MASK     0x007FFFFFu     // the instance index (a batch holds fewer than 8 million)

uint DrawEntryIndex(uint entry)
{
    return entry & DRAW_INDEX_MASK;
}

// 0 = no fade; else the new part's share (1..127) | 128 when the entry draws the part faded from
uint DrawEntryFade(uint entry)
{
    return (entry & DRAW_FADE_BIT) != 0 ? (entry >> DRAW_FADE_SHIFT) & 0xFFu : 0u;
}

#endif
