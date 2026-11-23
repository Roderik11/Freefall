// Decals (DECALS_PLAN.md): what the registry, the cull and the decal draw share. Graphics/DecalBuffer.cs keeps a slot per
// Decal component (its record below), Graphics/DecalPass.cs culls the registry per view (decal_cull.hlsl) into a list
// sorted by draw order, and decal.fx draws a box per decal onto the opaque G-buffer.
// The box is the unit cube (-0.5..0.5) under the decal entity's world matrix: X and Z span the decal, Y is the projection
// axis (the decal faces +Y and projects down it, so one lying on the ground needs no tilt). Its size is the scale.
#ifndef DECALS_HLSLI
#define DECALS_HLSLI

#include "weather_cover.hlsli"

#define MAX_VISIBLE_DECALS 4096u    // DecalBuffer.MaxVisible: the sort's key keeps 12 bits of the list's index
#define MAX_DECAL_SLOTS    65536u   // DecalBuffer.MaxSlots: ... and 16 of the registry slot

#define DECAL_ENABLED       1u
#define DECAL_SPAWNED       2u      // in the spawned ring (Decals.Spawn): fades out by its age
#define DECAL_PRIORITY_SHIFT 8u     // Flags bits 8..11: the priority (0..15), drawn in ascending order
#define DECAL_PRIORITY_MASK 0xFu

// One per Decal, and one per slot of the spawned ring (DecalBuffer.Record, 48 bytes)
struct DecalRecord
{
    uint TransformSlot;     // the world matrix in the global transform buffer (the entity's, or the ring slot's own)
    uint MaterialId;        // the decal material: its settings block (decal.fx Settings) by MaterialData
    uint Flags;
    uint Variant;           // bits 0..7 the tile of the material's atlas (DecalTile); 8..31 spawned: the transform slot
                            // + 1 of what carries it (0: nothing), its own matrix then the box in that one's space
    float3 Tint;            // multiplies the material's colour
    float Opacity;          // multiplies every target's opacity
    float SpawnTime;        // spawned: Time when it was made
    float Lifetime;         // spawned: seconds it lasts, fading out over the last FadeOut of them
    float FadeOut;
    float SpawnSnow;        // spawned: Weather.SnowFallen when it was made (a print fills with what falls after)
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
    ShelterLevel Shelter[SHELTER_LEVELS];   // the cover map (WeatherShelter, field for field)
    float ShelterMinY;
    float ShelterRangeY;
    uint TerrainPages;
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
    float Time;                 // Time.TotalTime: spawned decals' age
    float SnowFallen;           // Weather.SnowFallen: covers' worth of snow fallen so far (prints fill with it)
    float2 ShelterEye;          // world xz the cover map's levels are centred on (ShelterMap.EyeFor): their blend
};

#define DECAL_CARRIER_SHIFT 8u      // DecalBuffer.CarrierShift
#define DECAL_TILE_MASK 0xFFu

// The tile of the material's atlas it shows
uint DecalTile(DecalRecord r)
{
    return r.Variant & DECAL_TILE_MASK;
}

// A transform slot's world matrix (row vectors: rows 0-2 the axes, row 3 the origin)
float4x4 DecalMatrix(uint transformsIdx, uint transformSlot)
{
    StructuredBuffer<row_major float4x4> transforms = ResourceDescriptorHeap[transformsIdx];
    return transforms[transformSlot];
}

// A decal's world matrix: X and Z span the decal (scaled), Y is the projection axis (scaled), T its centre. A spawned
// mark that something carries (a print on a ship's deck: Decals.Spawn's carrier) keeps its box in that one's space and
// goes where it goes: its matrix under the carrier's, read from the same buffer, so nothing is written as it moves.
float4x4 DecalMatrix(uint transformsIdx, DecalRecord r)
{
    float4x4 m = DecalMatrix(transformsIdx, r.TransformSlot);
    uint carrier = r.Variant >> DECAL_CARRIER_SHIFT;
    if (carrier != 0u)
        m = mul(m, DecalMatrix(transformsIdx, carrier - 1u));
    return m;
}

#endif
