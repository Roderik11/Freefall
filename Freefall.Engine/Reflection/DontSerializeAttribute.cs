using System;

namespace Freefall.Reflection
{
    /// <summary>
    /// Mark a field or property to be skipped during serialization.
    /// Works with both FieldInfo and PropertyInfo, unlike [NonSerialized].
    /// </summary>
    [AttributeUsage(AttributeTargets.Field | AttributeTargets.Property)]
    public class DontSerializeAttribute : Attribute { }
}
