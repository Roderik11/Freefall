namespace Freefall.Assets
{
    /// <summary>
    /// Scene asset — a .scene YAML file that describes an entity hierarchy.
    /// Registered in the asset database like any other asset, so scripts can
    /// reference scenes by GUID (e.g. for scene loading, additive scenes).
    /// Named SceneAsset to avoid collision with Assimp.Scene.
    /// </summary>
    public class SceneAsset : Asset
    {
        /// <summary>
        /// Raw YAML text of the scene. Populated by the scene loader.
        /// </summary>
        public string YamlContent { get; set; }
    }
}
