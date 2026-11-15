// The far forest (Graphics/FarForest.cs): every owner's trees drawn past their hand-over as the grass is, from one
// buffer of 16 bytes a tree, picked each frame on the GPU (far_forest_cull.hlsl) and drawn by gbuffer_crossmesh.fx's
// FarForest passes. Shared by both.

#ifndef FAR_FOREST_HLSLI
#define FAR_FOREST_HLSLI

// One tree: its world position, and packed: yaw (10 bits, of a turn), scale across (7 bits) and up (7 bits, in
// FAR_TREE_SCALE_STEP), its kind (8 bits)
struct FarTree
{
    float3 Position;
    uint Packed;
};

// A kind of tree: the part it draws (the coarsest LOD's cards) and its indices (an indexed draw: a card's corners are
// shaded once), where its lists start in the visible buffer and the shadow buffer, its material, the part's bounding
// sphere (centre height, radius) and its crown's half width (the part's box across) in the tree's own frame
struct FarKind
{
    uint MeshPartId;
    uint IndexCount;
    uint VisibleBase;
    uint ShadowBase;
    uint ShadowCapacity;
    uint MaterialId;
    float CentreY;
    float Radius;
    uint IndexAddressLo;      // the mesh's index buffer (Mesh.IndexBufferView)
    uint IndexAddressHi;
    uint IndexBytes;
    uint BaseIndex;           // the part's first index
    float HalfWidth;          // what thinning measures on screen
    uint Profile;             // the far tier profile it hands over at (common.fx FarTierProfile; 0 = FarTierFrom)
    uint2 _pad;
};

// Thinning (as the grass's plants, stochastic simplification): a tree whose crown is narrower on screen than ThinPixels
// is kept with the share (width / ThinPixels)^2, so the trees per bit of screen stay the same however far, and a kept
// tree grows by 1 / sqrt(share), up to MaxGrow, so they cover what they covered. Its growth rides in its list entry:
//   the camera's: tree (24 bits) | growth (8 bits << 24)
//   the shadows': tree (22 bits) | growth (7 bits << 22) | cascade (3 bits << 29)
// growth over 1 .. 1 + FAR_GROW_RANGE
#define FAR_GROW_RANGE 3.0

uint FarGrowBits(float grow, uint levels)
{
    return (uint)(saturate((grow - 1.0) / FAR_GROW_RANGE) * (float)levels + 0.5);
}

float FarGrowOf(uint bits, uint levels)
{
    return 1.0 + (float)bits / (float)levels * FAR_GROW_RANGE;
}

// The tree's world matrix grown about its foot
row_major float4x4 FarGrow(row_major float4x4 world, float grow)
{
    world[0].xyz *= grow;
    world[1].xyz *= grow;
    world[2].xyz *= grow;
    return world;
}

// A draw command (InstanceBatch's IndexedDrawCommand, 12 uints, GraphicsDevice.IndexedCommandSignature): MeshPartId,
// InstanceBaseOffset, the index buffer view (address lo, hi, bytes, format), then DrawIndexed (index count, instance
// count, start index, base vertex, start instance) and a pad
#define FAR_COMMAND_UINTS 12u
#define FAR_COMMAND_INSTANCES 28u   // the instance count's byte offset in a command

// A cell of the far tier (Graphics.FarTier.CellSize) with one owner's trees, which the buffer holds together from First
// on: the box round their bounding spheres, the cell's centre the far tier fades it by (every tree of it alike), and the
// nearest hand-over of its trees' kinds (a far tier profile): the cell cull goes by it, each tree then by its own kind's
struct FarCell
{
    float3 Min;
    uint First;
    float3 Max;
    uint Count;
    float2 Centre;
    uint Profile;
    uint _pad;
};

#define FAR_TREE_SCALE_STEP 0.025

uint FarTreeKind(FarTree t)
{
    return t.Packed >> 24;
}

// The tree's world matrix: scale, then a turn about up, then its position (as the placement's)
row_major float4x4 FarTreeWorld(FarTree t)
{
    float yaw = (float)(t.Packed & 0x3FFu) * (6.28318530718 / 1024.0);
    float across = (float)((t.Packed >> 10) & 0x7Fu) * FAR_TREE_SCALE_STEP;
    float up = (float)((t.Packed >> 17) & 0x7Fu) * FAR_TREE_SCALE_STEP;
    float s, c;
    sincos(yaw, s, c);
    return float4x4(c * across, 0, -s * across, 0,
                    0, up, 0, 0,
                    s * across, 0, c * across, 0,
                    t.Position.x, t.Position.y, t.Position.z, 1);
}

// The larger of the two scales: what the bounding sphere grows by
float FarTreeScale(FarTree t)
{
    return max((float)((t.Packed >> 10) & 0x7Fu), (float)((t.Packed >> 17) & 0x7Fu)) * FAR_TREE_SCALE_STEP;
}

#endif
