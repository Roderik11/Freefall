using System;
using PhysX;

namespace Freefall.Base
{
    public static class PhysicsHelper
    {
        // The height field's samples come from the CPU store: TerrainPages.ToSamples (until 2026-11-07 ToSamples here, on
        // a float[,] of the whole map), the hole of each cell from TerrainPages.IsHoleCell.

        /// <summary>PxHeightFieldMaterial::eHOLE: a height-field triangle with this material does not collide.</summary>
        public const byte HoleMaterial = 127;

        /// <summary>
        /// A sample as HeightField.ModifySamples takes it, one int the way PxHeightFieldSample lies in memory: the height,
        /// then the material of each of the cell's two triangles, the first with the tessellation flag as in ToSamples.
        /// </summary>
        public static int PackedSample(short height, byte material) => (ushort)height | (material | 0x80) << 16 | material << 24;
    }
}
