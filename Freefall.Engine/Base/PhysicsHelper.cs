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

            // Holes (HoleStamp) only when the mask was baked for this height field's resolution
            bool cutHoles = holes != null && holeResolution == rows && rows == cols;

            System.Threading.Tasks.Parallel.For(0, rows, i =>
            {
                for (int j = 0; j < cols; j++)
                {
                    int index = i * cols + j;
                    short storedHeight = (short)(heightMap[i, j] * short.MaxValue);

                    // A sample's two materials are those of the cell it is the low corner of (row i = x, column j = z);
                    // material 127 (PxHeightFieldMaterial::eHOLE) leaves the triangle out
                    byte material = cutHoles && IsHoleCell(holes, holeResolution, i, j) ? HoleMaterial : (byte)0;

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

        // Terrain.IsHoleCell on a mask
        private static bool IsHoleCell(byte[] holes, int resolution, int x, int z)
        {
            if (x < 0 || z < 0 || x >= resolution - 1 || z >= resolution - 1) return false;
            int i = z * resolution + x;
            return holes[i] + holes[i + 1] + holes[i + resolution] + holes[i + resolution + 1] >= 2 * 255;
        }

        /// <summary>PxHeightFieldMaterial::eHOLE: a height-field triangle with this material does not collide.</summary>
        public const byte HoleMaterial = 127;
    }
}
