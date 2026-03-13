using System;

namespace Freefall.Base
{
    /// <summary>
    /// Marks a class or field as having been previously serialized under a different name.
    /// Enables forward-compatible deserialization after type or field renames.
    /// </summary>
    [AttributeUsage(AttributeTargets.Class | AttributeTargets.Field | AttributeTargets.Property, AllowMultiple = true)]
    public class FormerlySerializedAsAttribute : Attribute
    {
        public string Name { get; }

        public FormerlySerializedAsAttribute(string name)
        {
            Name = name;
        }
    }
}
