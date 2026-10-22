// Decals (DECALS_PLAN.md): what the registry, the cull and the decal draw share. Graphics/DecalBuffer.cs keeps a slot per
// Decal component (its record below), Graphics/DecalPass.cs culls the registry per view (decal_cull.hlsl) into a list
// sorted by draw order, and decal.fx draws a box per decal onto the opaque G-buffer.
// The box is the unit cube (-0.5..0.5) under the decal entity's world matrix: X and Z span the decal, Y is the projection
// axis (the decal faces +Y and projects down it, so one lying on the ground needs no tilt). Its size is the scale.
#ifndef DECALS_HLSLI
#define DECALS_HLSLI

#define MAX_VISIBLE_DECALS 4096u    // DecalBuffer.MaxVisible: the sort's key keeps 12 bits of the list's index
#define MAX_DECAL_SLOTS    65536u   // DecalBuffer.MaxSlots: ... and 16 of the registry slot

#define DECAL_ENABLED       1u
#define DECAL_PRIORITY_SHIFT 8u     // Flags bits 8..11: the priority (0..15), drawn in ascending order
#define DECAL_PRIORITY_MASK 0xFu

// One per Decal (DecalBuffer.Record, 32 bytes)
struct DecalRecord
{
    uint TransformSlot;     // the entity's world matrix in the global transform buffer
    uint MaterialId;        // the decal material: its settings block (decal.fx Settings) by MaterialData
    uint Flags;
    uint Variant;           // tile of the material's atlas
    float3 Tint;            // multiplies the material's colour
    float Opacity;          // multiplies every target's opacity
};

#define DECAL_TILE 16u
#define DECAL_MASK_WORDS (MAX_VISIBLE_DECALS / 32u)     // a bit per visible decal in each tile's mask

// The view the pass runs for, and the weather it applies (DecalPass.Frame, 240 bytes)
struct DecalFrame
{
    float4 Planes[5];           // camera-relative left, right, bottom, top, near; normals point in
    float3 CameraPosition;
    float PixelsPerMetre;       // at one metre: the projection's y scale × half the view's height
    float3 CameraForward;       // linear depth = dot(camera-relative place, this)
    uint TilesX;
    row_major float4x4 ShelterVP;   // WeatherCover, field for field
    uint ShelterDepth;
    uint ShelterSlice;
    uint TerrainHeightmap;
    float TerrainMaxHeight;
    float3 TerrainOrigin;
    float Wetness;              // Weather.SurfaceWetness: how wet the open surfaces are
    float2 TerrainSize;
    float PuddleFill;           // Weather.PuddleFill: how full rain-filled puddles are
    float SnowCover;            // Weather.SnowCover: how much snow lies on the open ground
    float Rain;                 // Weather.Rain at the camera: puddles' rings
    uint TilesY;
    uint WeatherOn;             // 0: no weather to apply, only the decals
    uint _framePad;
};

// The world matrix's rows: X and Z span the decal (scaled), Y is the projection axis (scaled), T its centre
float4x4 DecalMatrix(uint transformsIdx, uint transformSlot)
{
    StructuredBuffer<row_major float4x4> transforms = ResourceDescriptorHeap[transformsIdx];
    return transforms[transformSlot];
}

#endif
