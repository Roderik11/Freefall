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
        public static HeightFieldSample[] ToSamples(this float[,] heightMap, Freefall.Assets.Terrain holes = null)
        {
            if (heightMap == null)
                throw new ArgumentNullException(nameof(heightMap));

            int rows = heightMap.GetLength(0);
            int cols = heightMap.GetLength(1);

            var samples = new HeightFieldSample[rows * cols];

            // Holes (HoleStamp) only when the mask was baked for this height field's resolution
            bool cutHoles = holes is { HasHoles: true } && holes.HoleResolution == rows && rows == cols;

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    int index = i * cols + j;
                    short storedHeight = (short)(heightMap[i, j] * short.MaxValue);

                    // A sample's two materials are those of the cell it is the low corner of (row i = x, column j = z);
                    // material 127 (PxHeightFieldMaterial::eHOLE) leaves the triangle out
                    byte material = cutHoles && holes.IsHoleCell(i, j) ? HoleMaterial : (byte)0;

                    samples[index] = new HeightFieldSample
                    {
                        MaterialIndex0 = new BitAndByte(material, true),
                        MaterialIndex1 = new BitAndByte(material, false),
                        Height = storedHeight,
                    };
                }
            }

            return samples;
        }

        /// <summary>PxHeightFieldMaterial::eHOLE: a height-field triangle with this material does not collide.</summary>
        public const byte HoleMaterial = 127;
    }
}
