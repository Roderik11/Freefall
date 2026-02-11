using System;
using System.Collections.Generic;

namespace Freefall.Animation
{
    /// <summary>
    /// Root animation controller containing layers and parameters.
    /// Parameters are used by blend trees and transitions.
    /// </summary>
    public class Animation : Assets.Asset
    {
        public Dictionary<string, float> Parameters = new Dictionary<string, float>();
        public List<AnimationLayer> Layers = new List<AnimationLayer>();

        public float GetParam(string name)
        {
            Parameters.TryGetValue(name, out var result);
            return result;
        }

        public void SetParam(string name, float value)
        {
            if (!Parameters.ContainsKey(name)) return;
            Parameters[name] = value;
        }
    }
}
