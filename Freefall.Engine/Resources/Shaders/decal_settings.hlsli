// A decal material's settings (decal.fx is the effect its .mat names; decals_cs.hlsl reads the block by the decal's
// MaterialId). Packed tightly in declaration order (Material.WriteSetting): every member carries its "//=" default, and
// the block stays within Material.SettingsStride (256 bytes). FXParser reads the struct from decal.fx's text, which
// includes this file.
#ifndef DECAL_SETTINGS_HLSLI
#define DECAL_SETTINGS_HLSLI

struct Settings
{
    TextureRef Albedo;          //= White      -- Colour (rgb) and opacity (a). With RainFilled the opacity is the puddle's depth
    TextureRef Normal;          //= FlatNormal -- Normal map in the decal's frame (its X right, its -Z up the texture)
    TextureRef Mask;            //= White      -- Roughness (r), metal (g), AO (b)
    TextureRef Height;          //= Black      -- Relief (r), 0..1 onto HeightMin..HeightMax, for the screen-space displacement
    float3 Color;               //= 1, 1, 1    -- Multiplies the albedo map
    uint AlbedoMode;            //= 0 [0, 1]   -- 0 Replace (paint over), 1 Darken (multiplies the surface by the colour: grime, soot, damp)
    float AlbedoOpacity;        //= 1 [0, 1]   -- How much of the colour (0 = the surface keeps its own)
    uint NormalMode;            //= 1 [0, 1]   -- 0 Replace (the decal's normal), 1 Detail (adds the decal's tilt: the surface's own detail stays)
    float NormalOpacity;        //= 1 [0, 1]   -- How much of the normal
    float NormalStrength;       //= 1 [0, 4]   -- Steepens or flattens the normal map
    float MaskOpacity;          //= 0 [0, 1]   -- How much of roughness, metal and AO (0 = the surface keeps its own)
    float Roughness;            //= 1 [0, 1]   -- Multiplies the mask's roughness
    float Metal;                //= 1 [0, 1]   -- Multiplies the mask's metal
    uint HeightMode;            //= 0 [0, 1]   -- 0 Add (on top of the surface's relief: a crack), 1 Replace (the surface's relief goes: a puddle flattens the cobbles)
    float HeightOpacity;        //= 0 [0, 1]   -- How much of the relief (0 = none)
    float HeightMin;            //= 0          -- Metres at the height map's 0 (negative: a dent)
    float HeightMax;            //= 0.02       -- Metres at its 1
    float2 AngleFade;           //= 50, 75     -- Degrees between the surface and the projection axis: full up to x, gone at y
    float EdgeFade;             //= 0.15 [0, 0.5] -- Share of the box's height over which it fades out at the top and bottom
    uint AtlasColumns;          //= 1          -- Tiles across the textures (Decal.Variant picks one)
    uint AtlasRows;             //= 1          -- Tiles down
    uint Receivers;             //= 3 [0, 15]  -- What it lands on, a bit each: 1 static meshes, 2 terrain, 4 vegetation, 8 characters
    bool RainFilled;            //= false      -- Puddles: the opacity is the hollow's depth (the middle fills first). Damp first, water once the rain's puddles are a quarter full, its surface level with the shoreline (the height settings and map are not used); drying, the water goes and a damp stain shrinks to the middle
    float Ripples;              //= 0 [0, 2]   -- Puddles: rings where the drops strike, as many as the rain at the camera
    float DentDepth;            //= 0 [0, 0.08] -- Puddles: metres the ground sinks at the deepest, wet or dry (the screen-space displacement, sloped normals): the hollow the rain fills and the snow drifts into
    float3 SiltColor;           //= 0.6, 0.53, 0.44 -- Puddles: the silt and grit the water leaves in the hollow, multiplying the ground (in its joints most, a dusting on the stones)
    float SiltOpacity;          //= 0 [0, 1]   -- Puddles: how much silt covers the hollow's floor (0 = none)
    float SiltRoughness;        //= 0.85 [0, 1] -- Puddles: the dry silt's roughness
    float SiltFlatten;          //= 0.5 [0, 1] -- Puddles: how much the silt evens out the stones in the hollow (fills their joints)
};

Settings LoadDecalSettings(uint materialsIdx, uint materialId)
{
    StructuredBuffer<MaterialData> materials = ResourceDescriptorHeap[materialsIdx];
    ByteAddressBuffer all = ResourceDescriptorHeap[NonUniformResourceIndex(materials[materialId].SettingsBufferIdx)];
    return all.Load<Settings>(materialId * 256u);
}

#endif
