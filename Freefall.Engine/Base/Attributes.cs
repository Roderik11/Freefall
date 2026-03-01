using System;

namespace Freefall
{
    /// <summary>
    /// Constrains a numeric field to a [min, max] range in the inspector.
    /// Ported from Apex/Spark.
    /// </summary>
    public class ValueRangeAttribute : Attribute
    {
        public float Min;
        public float Max;
        public float Step = 1;

        public ValueRangeAttribute(float min, float max)
        {
            Min = min;
            Max = max;
        }

        public ValueRangeAttribute(float min, float max, float step)
        {
            Min = min;
            Max = max;
            Step = step;
        }
    }
}
