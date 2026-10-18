// Sun shadow receivers: layout of the buffer shadow_receivers.hlsl builds and the cascade culling reads
// (cull_instances.hlsl CSVisibilityCascades, push constant slot 39; ShadowReceivers.cs on the CPU).
//
// One RWStructuredBuffer<uint>: a header of stats and settings, then per cascade a max pyramid of the deepest
// receiver light depth (standard Z, asuint; 0 = no receiver) over a RECEIVER_TILES² grid of its shadow map, level 0
// first, down to 1×1. A receiver is anything that reads the cascade: a pixel of this frame's linear depth that the
// directional light shades with that cascade (also the next one inside the 10 % blend band, as
// light_directional_cs.hlsl does), and the readers the depth buffer does not hold: the ocean's wave band along each
// pixel's view ray, the forward instances (glass, water bodies) seen last frame, and the sun samples around each
// visible point light (local_shadow_select.hlsl SunVisibility).

#ifndef SHADOW_RECEIVERS_HLSLI
#define SHADOW_RECEIVERS_HLSLI

#include "shadow_cascades.hlsli"

#define RECEIVER_MAX_CASCADES   SHADOW_MAX_CASCADES
#define RECEIVER_TILES          64u     // tiles per side of a cascade's level 0 (2048² map: 32 texels each)
#define RECEIVER_LEVELS         7u      // 64, 32, 16, 8, 4, 2, 1
#define RECEIVER_CASCADE_WORDS  5461u   // 4096 + 1024 + 256 + 64 + 16 + 4 + 1
#define RECEIVER_HEADER_WORDS   48u

// Header (ShadowReceivers.Header* on the CPU); the per-cascade runs hold RECEIVER_MAX_CASCADES words each
#define RECEIVER_MODE           0u      // 1 = count the entries the test would drop, 2 = drop them
#define RECEIVER_MARGIN         1u      // asuint: the shadow filter's reach in shadow-map UV (PCF disk + bilinear tap)
#define RECEIVER_INSTANCES      2u      // instances in at least one cascade
#define RECEIVER_INSTANCES_CUT  3u      // of those, in none once the test has run
#define RECEIVER_TILE_COUNT     4u      // 4..9: level-0 tiles with a receiver, per cascade
#define RECEIVER_ENTRIES        10u     // 10..15: cascade entries (instance × cascade) before the test, per cascade
#define RECEIVER_ENTRIES_CUT    16u     // 16..21: of those, shading no receiver
#define RECEIVER_TRIS           22u     // 22..27: triangles of the entries
#define RECEIVER_TRIS_CUT       28u     // 28..33: triangles of the entries shading no receiver
#define RECEIVER_WAVE_BOUND     34u     // 34..41: per ocean band, its largest |vertical displacement| (asuint, metres)
#define RECEIVER_CASCADES       42u     // the cascades this build marked (stats)

#define RECEIVER_MODE_COUNT     1u
#define RECEIVER_MODE_CULL      2u
#define RECEIVER_MAX_BANDS      8u      // ocean.fx BandAngles[8]

// First word of a cascade's pyramid level (row-major, (RECEIVER_TILES >> level)² words)
uint ReceiverLevelBase(uint cascadeIdx, uint level)
{
    // Levels before this one: 4096 + 1024 + ... = (4096 - 4096 / 4^level) * 4 / 3
    uint full = RECEIVER_TILES * RECEIVER_TILES;
    return RECEIVER_HEADER_WORDS + cascadeIdx * RECEIVER_CASCADE_WORDS + (full - (full >> (2u * level))) * 4u / 3u;
}

#endif
