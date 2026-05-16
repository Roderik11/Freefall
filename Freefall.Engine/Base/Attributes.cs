using System;

namespace Freefall
{
    [AttributeUsage(AttributeTargets.Class)]
    public sealed class UpdateInEditorAttribute : Attribute { }

    /// <summary>
    /// Constrains a numeric field to a [min, max] range in the inspector.
    /// Ported from Apex/Spark.
    /// </summary>
    [AttributeUsage(AttributeTargets.Field | AttributeTargets.Property)]
    public sealed class ValueRangeAttribute : Attribute
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

    [AttributeUsage(AttributeTargets.Field | AttributeTargets.Property)]
    public sealed class FilePathAttribute(string filter, string title) : Attribute
    {
        public string Filter = filter;
        public string Title = title;
    }

    [AttributeUsage(AttributeTargets.All)]
    public sealed class IconAttribute(string name) : Attribute
    {
        public string Name = name;
    }

    [AttributeUsage(AttributeTargets.Class)]
    public sealed class CreateAssetAttribute(string caption) : Attribute
    {
        public string Caption = caption;
    }
}
