using System;

namespace Freefall
{
    [AttributeUsage(AttributeTargets.Class)]
    public sealed class UpdateInEditorAttribute : Attribute { }

    /// <summary>
    /// The <see cref="Base.SystemGroup"/> a system is updated by. Without it a system lands in
    /// <see cref="Base.UpdateGroup"/>.
    /// </summary>
    [AttributeUsage(AttributeTargets.Class)]
    public sealed class UpdateInGroupAttribute(Type group) : Attribute
    {
        public Type Group = group;
    }

    /// <summary>Update this system before another system of the same group.</summary>
    [AttributeUsage(AttributeTargets.Class, AllowMultiple = true)]
    public sealed class UpdateBeforeAttribute(Type system) : Attribute
    {
        public Type SystemType = system;
    }

    /// <summary>Update this system after another system of the same group.</summary>
    [AttributeUsage(AttributeTargets.Class, AllowMultiple = true)]
    public sealed class UpdateAfterAttribute(Type system) : Attribute
    {
        public Type SystemType = system;
    }

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
