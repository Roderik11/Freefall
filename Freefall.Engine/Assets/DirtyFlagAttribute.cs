using System;

namespace Freefall.Assets
{
    /// <summary>
    /// Declares which TerrainDirtyFlags should be raised when this member is
    /// modified by the inspector.  If absent, the inspector falls back to a
    /// conservative default for the owning asset type.
    /// </summary>
    [AttributeUsage(AttributeTargets.Field | AttributeTargets.Property)]
    public class DirtyFlagAttribute : Attribute
    {
        public TerrainDirtyFlags Flags { get; }
        public DirtyFlagAttribute(TerrainDirtyFlags flags) => Flags = flags;
    }
}
