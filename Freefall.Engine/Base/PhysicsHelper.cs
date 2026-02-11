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
        public static HeightFieldSample[] ToSamples(this float[,] heightMap)
        {
            if (heightMap == null)
                throw new ArgumentNullException(nameof(heightMap));

            int rows = heightMap.GetLength(0);
            int cols = heightMap.GetLength(1);

            var samples = new HeightFieldSample[rows * cols];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    int index = i * cols + j;
                    short storedHeight = (short)(heightMap[i, j] * short.MaxValue);

                    samples[index] = new HeightFieldSample
                    {
                        MaterialIndex0 = new BitAndByte(0, true),
                        MaterialIndex1 = new BitAndByte(0, false),
                        Height = storedHeight,
                    };
                }
            }

            return samples;
        }
    }
}
