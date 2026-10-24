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
        /// middle, read as the terrain shader reads it (Terrain.HoleWeight). The height field's samples sit at the height
        /// map's texel centres, sample i at (i + 0.5) · size / N (TerrainCollider), so the cell's middle is
        /// (i + 1) · size / N; the mask has a grid of its own (sample k at k · size / (w − 1)), so it is looked up by
        /// place, not by index. As a share of the terrain the size drops out.
        /// </summary>
        public static bool IsHoleCell(byte[] holes, int holeResolution, int rows, int cols, int x, int z)
        {
            if (x < 0 || z < 0 || x >= rows - 1 || z >= cols - 1) return false;
            return Freefall.Assets.Terrain.HoleWeight(holes, holeResolution, System.Numerics.Vector2.One,
                                                      (x + 1f) / rows, (z + 1f) / cols) >= 0.5f;
        }

        /// <summary>PxHeightFieldMaterial::eHOLE: a height-field triangle with this material does not collide.</summary>
        public const byte HoleMaterial = 127;
    }
}
