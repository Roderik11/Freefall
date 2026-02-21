using System;

namespace Freefall.Base
{
    /// <summary>
    /// Marks a class as having been previously serialized under a different name.
    /// Enables forward-compatible deserialization after type renames.
    /// </summary>
    [AttributeUsage(AttributeTargets.Class)]
    public class FormerlySerializedAsAttribute : Attribute
    {
        public string Name { get; }

        public FormerlySerializedAsAttribute(string name)
        {
            Name = name;
        }
    }
}
