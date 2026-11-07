using System;
using PhysX;

namespace Freefall.Base
{
    public static class PhysicsHelper
    {
        /// <summary>
        /// Convert a float[,] heightmap (values 0..1) to PhysX HeightFieldSample array.
        /// Ported from Apex Physics.cs.
        /// </summary>
        public static HeightFieldSample[] ToSamples(this float[,] heightMap, Freefall.Assets.Terrain holes = null) =>
            heightMap.ToSamples(holes?.Holes, holes?.HoleResolution ?? 0);

        /// <summary>
        /// The same from a hole mask as Terrain.Holes holds it (null: no holes), so a worker can convert a snapshot. Rows
        /// in parallel: 4097² samples took ~1.2 s on one thread.
        /// </summary>
        public static HeightFieldSample[] ToSamples(this float[,] heightMap, byte[] holes, int holeResolution)
        {
            if (heightMap == null)
                throw new ArgumentNullException(nameof(heightMap));

            int rows = heightMap.GetLength(0);
            int cols = heightMap.GetLength(1);

            var samples = new HeightFieldSample[rows * cols];

            bool cutHoles = holes != null && holeResolution > 1;

            System.Threading.Tasks.Parallel.For(0, rows, i =>
            {
                for (int j = 0; j < cols; j++)
                {
                    int index = i * cols + j;
                    short storedHeight = (short)(heightMap[i, j] * short.MaxValue);

                    // A sample's two materials are those of the cell it is the low corner of (row i = x, column j = z);
                    // material 127 (PxHeightFieldMaterial::eHOLE) leaves the triangle out
                    byte material = cutHoles && IsHoleCell(holes, holeResolution, rows, cols, i, j) ? HoleMaterial : (byte)0;

                    samples[index] = new HeightFieldSample
                    {
                        MaterialIndex0 = new BitAndByte(material, true),
                        MaterialIndex1 = new BitAndByte(material, false),
                        Height = storedHeight,
                    };
                }
            });

            return samples;
        }

        /// <summary>
        /// Whether the height-field cell with sample (x, z) as its low corner is open: the hole mask at the cell's
        /// middle, read as the terrain shader reads it (Terrain.HoleWeight). The height field's samples sit on the
        /// vertices, sample i at i · size / (N − 1) (TerrainCollider, terrain_grid.hlsli), so the cell's middle is
        /// (i + 0.5) · size / (N − 1); the mask is looked up by place, not by index, so it may have a size of its own.
        /// As a share of the terrain the size drops out.
        /// </summary>
        public static bool IsHoleCell(byte[] holes, int holeResolution, int rows, int cols, int x, int z)
        {
            if (x < 0 || z < 0 || x >= rows - 1 || z >= cols - 1) return false;
            return Freefall.Assets.Terrain.HoleWeight(holes, holeResolution, System.Numerics.Vector2.One,
                                                      (x + 0.5f) / (rows - 1), (z + 0.5f) / (cols - 1)) >= 0.5f;
        }

        /// <summary>PxHeightFieldMaterial::eHOLE: a height-field triangle with this material does not collide.</summary>
        public const byte HoleMaterial = 127;

        /// <summary>
        /// A sample as HeightField.ModifySamples takes it, one int the way PxHeightFieldSample lies in memory: the height,
        /// then the material of each of the cell's two triangles, the first with the tessellation flag as in ToSamples.
        /// </summary>
        public static int PackedSample(short height, byte material) => (ushort)height | (material | 0x80) << 16 | material << 24;
    }
}
