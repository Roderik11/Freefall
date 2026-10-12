using System;

namespace Freefall.Reflection
{
    /// <summary>
    /// A member of a component whose value belongs to the one instance of a prefab and not to the
    /// prefab: who a rolled character turned out to be. Applying an instance to its prefab leaves
    /// the prefab's own value for it (Prefab.UpdateFromInstance), and bringing the instances in
    /// line with the prefab leaves each its own (Prefab.UpdateInstance).
    ///
    /// An instance is saved as its prefab and a transform, so after a load the value is the
    /// prefab's again. Creating a prefab from an entity writes it like any other member.
    /// </summary>
    [AttributeUsage(AttributeTargets.Field | AttributeTargets.Property)]
    public class PerInstanceAttribute : Attribute { }
}
