namespace Freefall.Assets
{
    /// <summary>
    /// Implemented by Asset types that need to rebuild runtime state
    /// after deserialization (e.g. resolving ID-based references,
    /// building lookup caches, recreating ports).
    /// Called automatically by NativeAssetLoader after YAML load.
    /// </summary>
    public interface IRebuildAfterLoad
    {
        void RebuildAfterLoad();
    }
}
