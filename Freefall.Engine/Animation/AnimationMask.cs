using System;
using System.Collections.Generic;

namespace Freefall.Animation
{
    /// <summary>
    /// Defines which bones are affected by an animation layer.
    /// Used for partial body animations (e.g., upper body aiming).
    /// </summary>
    public class AnimationMask : Assets.Asset
    {
        public List<string> Bones = new List<string>();

        public bool Contains(string name)
        {
            return Bones.Contains(name);
        }
    }
}
