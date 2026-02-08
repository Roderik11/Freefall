using System;
using System.Collections.Generic;

namespace Freefall.Animation
{
    /// <summary>
    /// A curve that can animate a parameter over the duration of an animation state.
    /// </summary>
    public class AnimationCurve
    {
        public string TargetParameter;
        public List<FloatKey> Keys = new List<FloatKey>();

        public void AddKeyFrame(float time, float value)
        {
            Keys.Add(new FloatKey { Time = time, Value = value });
        }

        public float Evaluate(float normalizedTime)
        {
            if (Keys.Count == 0) return 0;
            if (Keys.Count == 1) return Keys[0].Value;

            // Find surrounding keyframes
            for (int i = 0; i < Keys.Count - 1; i++)
            {
                if (normalizedTime >= Keys[i].Time && normalizedTime <= Keys[i + 1].Time)
                {
                    float t = (normalizedTime - Keys[i].Time) / (Keys[i + 1].Time - Keys[i].Time);
                    return Keys[i].Value + (Keys[i + 1].Value - Keys[i].Value) * t;
                }
            }

            // Return last value if past end
            return Keys[Keys.Count - 1].Value;
        }
    }

    public struct FloatKey
    {
        public float Time;
        public float Value;
    }
}
