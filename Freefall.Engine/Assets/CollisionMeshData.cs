namespace Freefall.Assets
{
    /// <summary>
    /// Container for pre-cooked PhysX collision mesh bytes (TriangleMesh or HeightField).
    /// Stored as a hidden subasset during import, loaded at runtime by asset loaders.
    /// </summary>
    public class CollisionMeshData
    {
        public byte[] CookedBytes { get; set; }
    }
}
