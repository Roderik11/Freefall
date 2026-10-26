namespace Freefall.Base
{
    /// <summary>
    /// Engine-side message constants for MessageDispatcher.
    /// </summary>
    public static class EngineMsg
    {
        public const string SplineChanged = "SplineChanged";
        public const string GraphChanged = "GraphChanged";
        public const string PCGExecuted = "PCGExecuted";
        public const string StampChanged = "StampChanged";
        /// <summary>CPU-side terrain HeightField was replaced (GPU bake readback). Data = TerrainHeightsChange.</summary>
        public const string TerrainHeightsChanged = "TerrainHeightsChanged";
        /// <summary>A local SplatStamp changed: the painted ground is new in a region, the heights are not. Data =
        /// TerrainHeightsChange (the region). Sent when the stamp changes, not after a bake: nothing on the CPU reads
        /// the bake's result for it (PCG's ExcludeStamps tests the stamps' shapes).</summary>
        public const string TerrainPaintChanged = "TerrainPaintChanged";
    }
}
