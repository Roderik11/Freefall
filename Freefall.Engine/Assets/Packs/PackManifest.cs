using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;

namespace Freefall.Assets.Packs
{
    /// <summary>
    /// What a published game knows of its assets, next to its packs (Data/game.manifest): the scenes in the build's order
    /// (the game starts in the first), the pack files in mount order, and the metas of every packed asset cut down to
    /// their packed artifacts. AssetDatabase.InitializeFromManifest registers those metas instead of scanning a
    /// project, so the lookups the loaders make (types, names, siblings, importer settings) work as in the editor.
    /// </summary>
    public sealed class PackManifest
    {
        public const string FileName = "game.manifest";
        public const int CurrentVersion = 1;

        public int Version { get; set; } = CurrentVersion;
        public string Product { get; set; }
        public DateTime Created { get; set; }
        public List<string> Scenes { get; set; } = new();
        public List<string> Packs { get; set; } = new();
        /// <summary>The engine's own files (shaders, default textures: EngineFiles), mounted before the engine starts; null = from the disk.</summary>
        public string EnginePack { get; set; }
        public List<MetaFile> Metas { get; set; } = new();

        public void Save(string path) =>
            File.WriteAllText(path, JsonSerializer.Serialize(this, AssetDatabase.MetaJsonOptions));

        public static PackManifest Load(string path)
        {
            var manifest = JsonSerializer.Deserialize<PackManifest>(File.ReadAllText(path), AssetDatabase.MetaJsonOptions)
                ?? throw new InvalidDataException($"{path} is empty");
            if (manifest.Version != CurrentVersion)
                throw new InvalidDataException($"{path}: manifest version {manifest.Version}, this engine reads {CurrentVersion}");
            return manifest;
        }
    }
}
