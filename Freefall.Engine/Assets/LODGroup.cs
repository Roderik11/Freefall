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
            Trees = new LODGroup("Trees", .5f, .4f, .3f, .2f, .1f, .0f);
            LargeProps = new LODGroup("LargeProps", .6f, .3f, .1f, .05f);
            SmallProps = new LODGroup("SmallProps", .8f, .6f, .4f, .2f, .1f, .05f);
        }
    }
}
