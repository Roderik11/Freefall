namespace Freefall.Assets
{
    /// <summary>
    /// CPU-only prefab data — raw YAML bytes.
    /// The packer stores this in cache; PrefabLoader reads it back.
    /// </summary>
    public class PrefabData
    {
        public byte[] Yaml;
    }
}
