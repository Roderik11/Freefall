// The sea's quadtree leaves as the walk writes them (ocean_quadtree.hlsl) and the ocean's vertex shader reads them
// (ocean_patch.hlsli VS_Leaf). Graphics/OceanQuadtree.cs keeps LeafCells equal to OCEAN_LEAF_CELLS.
#ifndef OCEAN_LEAF_HLSLI
#define OCEAN_LEAF_HLSLI

// Cells a side of every leaf, each two of the tessellated triangle patches
#define OCEAN_LEAF_CELLS 8
// Control points a leaf is drawn with (non-indexed: two triangles a cell)
#define OCEAN_LEAF_VERTICES (OCEAN_LEAF_CELLS * OCEAN_LEAF_CELLS * 6)
// The leaves' buffer (Graphics.OceanQuadtree.MaxLeaves)
#define OCEAN_MAX_LEAVES 4096

// A node of the tree in one uint: its depth (0 = the root), the edges where the neighbour is a level coarser (bit 0 −x,
// 1 +x, 2 −z, 3 +z: the leaf's control points on that edge are snapped to every second one) and its place among the
// 2^depth × 2^depth nodes of its level. 12 bits a coordinate: depth 12 at most
uint PackOceanNode(uint depth, uint stitch, uint ix, uint iz)
{
    return (depth << 28) | (stitch << 24) | (ix << 12) | iz;
}

void UnpackOceanNode(uint node, out uint depth, out uint stitch, out uint ix, out uint iz)
{
    depth = node >> 28;
    stitch = (node >> 24) & 0xF;
    ix = (node >> 12) & 0xFFF;
    iz = node & 0xFFF;
}

#endif
