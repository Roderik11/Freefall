namespace Freefall.Graphics
{
    /// <summary>
    /// Describes a resource binding parsed from shader source.
    /// Maps a semantic name (e.g. "Albedo") to a push constant slot.
    /// </summary>
    public class ShaderResourceBinding
    {
        /// <summary>
        /// Semantic name used in C# (e.g. "Albedo", "LightBuffer", "Transforms")
        /// Derived from shader define by stripping "Idx" suffix
        /// </summary>
        public string Name { get; set; } = string.Empty;
        
        /// <summary>
        /// Push constant slot index (0-11)
        /// </summary>
        public int Slot { get; set; }
        
        /// <summary>
        /// Original define name from shader (e.g. "AlbedoIdx")
        /// </summary>
        public string DefineName { get; set; } = string.Empty;
        
        /// <summary>
        /// Optional comment describing the resource type
        /// </summary>
        public string? Comment { get; set; }
        
        public override string ToString() => $"{Name} -> Slot {Slot}";
    }
}
