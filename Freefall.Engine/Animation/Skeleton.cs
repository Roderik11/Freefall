using System.Numerics;
using Freefall.Assets;

namespace Freefall.Animation
{
    /// <summary>
    /// A shared skeleton definition (like Unity's Avatar).
    /// Contains the bone hierarchy, bind poses, and offset matrices.
    /// Referenced by both meshes (for skinning) and animations (for playback/retargeting).
    /// </summary>
    public class Skeleton : Asset
    {
        /// <summary>Ordered bone array — index = bone ID used everywhere.</summary>
        public Bone[] Bones { get; set; } = [];

        /// <summary>Bone names in order, for fast lookup.</summary>
        public string[] BoneNames { get; set; } = [];

        /// <summary>Find a bone index by name. Returns -1 if not found.</summary>
        public int FindBone(string name)
        {
            for (int i = 0; i < BoneNames.Length; i++)
            {
                if (BoneNames[i] == name) return i;
            }
            return -1;
        }

        /// <summary>Find a bone index by name hash. Returns -1 if not found.</summary>
        public int FindBone(int nameHash)
        {
            for (int i = 0; i < Bones.Length; i++)
            {
                if (Bones[i].Name.GetHashCode() == nameHash) return i;
            }
            return -1;
        }

        /// <summary>Number of bones in the skeleton.</summary>
        public int BoneCount => Bones.Length;
    }
}
