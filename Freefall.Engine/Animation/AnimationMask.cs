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

        // Runtime lookup cache
        private HashSet<string> _boneSet;

        private void EnsureSet()
        {
            if (_boneSet != null && _boneSet.Count == Bones.Count) return;
            _boneSet = new HashSet<string>(Bones);
        }

        public bool Contains(string name)
        {
            EnsureSet();
            return _boneSet.Contains(name);
        }

        /// <summary>
        /// Invalidates the lookup cache. Call after modifying the Bones list at runtime.
        /// </summary>
        public void InvalidateCache() => _boneSet = null;
    }
}
