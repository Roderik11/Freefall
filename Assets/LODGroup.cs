using System;
using System.Collections.Generic;
using System.Linq;

namespace Freefall.Assets
{
    public class LODGroup : Asset
    {
        public List<float> Ranges = new List<float>();

        public LODGroup() { }
        public LODGroup(string name, params float[] ranges)
        {
            Name = name;
            Ranges = ranges.ToList();
        }
    }

    public static class LODGroups
    {
        public static LODGroup Trees;
        public static LODGroup LargeProps;
        public static LODGroup SmallProps;

        static LODGroups()
        {
            // LOD ranges in world units (distance from camera)
            Trees = new LODGroup("Trees", 50f, 100f, 200f, 400f, 800f);
            LargeProps = new LODGroup("LargeProps", 32f, 64f, 128f, 256f);
            SmallProps = new LODGroup("SmallProps", 32, 64, 128, 256, 512, 1024);
        }
    }
}
