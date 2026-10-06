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
    }
}
