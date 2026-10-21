using System.ComponentModel;

namespace Freefall.Assets
{
    /// <summary>
    /// Scene asset — a .scene YAML file that describes an entity hierarchy. Imported like any other asset
    /// (SceneImporter → SceneData in the Cache → SceneLoader), so a scene has a GUID: build lists and script
    /// fields reference scenes the way they reference everything else, and a game loads a scene from its
    /// artifact, never from a path. Load one with <see cref="Freefall.Base.SceneManager"/>.
    /// Named SceneAsset to avoid collision with Assimp.Scene.
    /// </summary>
    [AssetTypeAlias("SceneData")]
    public class SceneAsset : Asset
    {
        /// <summary>The scene's YAML, as the .scene file holds it.</summary>
        [Browsable(false)]
        public byte[] Yaml { get; set; }
    }

    /// <summary>
    /// CPU-only scene data — the .scene file's YAML bytes.
    /// The packer stores this in cache; SceneLoader reads it back.
    /// </summary>
    public class SceneData
    {
        public byte[] Yaml;
    }
}
