using System.IO;
using System.Text;
using Freefall.Assets.Packers;
using Freefall.Graph;
using Freefall.Serialization;

namespace Freefall.Assets.Loaders
{
    /// <summary>
    /// Loads NodeGraph-derived assets (PCGGraph, etc.) from cache.
    /// Unpacks AssetDefinitionData (YAML) → NodeGraph, then rebuilds
    /// runtime state (ports, connection refs).
    /// </summary>
    [AssetLoader(typeof(NodeGraph), ".asset")]
    public class NodeGraphLoader : IAssetLoader
    {
        private readonly AssetDefinitionPacker _packer = new();

        public Asset Load(string name, AssetManager manager)
        {
            var cachePath = AssetDatabase.ResolveCachePath(name, "AssetDefinitionData");
            if (cachePath == null || !File.Exists(cachePath))
                throw new FileNotFoundException($"Cache file not found for graph '{name}'");

            return LoadFromCache(cachePath, name, manager);
        }

        public Asset LoadFromCache(string cachePath, string name, AssetManager manager, string sourceGuid = null)
        {
            AssetDefinitionData defData;
            using (var stream = File.OpenRead(cachePath))
                defData = _packer.Read(stream);

            var yaml = Encoding.UTF8.GetString(defData.YamlBytes);
            var graph = NativeImporter.LoadFromString(yaml, manager) as NodeGraph;

            if (graph == null)
                throw new InvalidDataException($"Failed to deserialize NodeGraph from cache: {name}");

            graph.Name = name;
            graph.RebuildAfterLoad();
            graph.MarkReady();

            return graph;
        }

        public void Save(Asset asset, string savePath)
        {
            var yaml = NativeImporter.SaveToString(asset);
            File.WriteAllText(savePath, yaml, Encoding.UTF8);
            Debug.Log($"[NodeGraphLoader] Saved: {savePath}");
        }
    }
}
