using System;
using System.IO;
using System.Linq;
using System.Text;
using Freefall.Assets.Packers;

namespace Freefall.Assets.Importers
{
    /// <summary>
    /// Imports .navmesh files (YAML-serialized NavMesh definitions).
    /// Produces two artifacts:
    /// 1. The NavMesh definition (AssetDefinitionData) — bake parameters in YAML
    /// 2. A hidden NavMeshData subasset containing the baked DotRecast binary blob
    ///    (preserved across reimports by finding existing cached data via meta subassets).
    /// Mirrors the TerrainImporter pattern.
    /// </summary>
    [AssetImporter(".navmesh", ImportPriority = 4)]
    public class NavMeshImporter : IImporter
    {
        public Type AssetType => typeof(NavMesh);

        public ImportResult Import(string filepath)
        {
            var result = new ImportResult();
            var name = Path.GetFileNameWithoutExtension(filepath);
            var yaml = File.ReadAllText(filepath, Encoding.UTF8);

            // ── 1. Main artifact: NavMesh definition (YAML) ──
            result.Artifacts.Add(new ImportArtifact
            {
                Name = name,
                Type = "NavMesh",
                Data = new AssetDefinitionData
                {
                    TypeName = "NavMesh",
                    YamlBytes = Encoding.UTF8.GetBytes(yaml)
                }
            });

            // ── 2. Preserve existing baked data across reimports ──
            // Find the source GUID from the file path, then look up the
            // existing NavMeshData subasset in the meta (still populated
            // at this point — AssetDatabase clears it AFTER Import returns).
            try
            {
                var bakedBytes = LoadExistingBakedData(filepath);
                if (bakedBytes != null)
                {
                    result.Artifacts.Add(new ImportArtifact
                    {
                        Name = name,
                        Type = nameof(NavMeshData),
                        Data = new NavMeshData { BakedBytes = bakedBytes },
                        Hidden = true
                    });
                }
            }
            catch (Exception ex)
            {
                Debug.LogWarning("NavMeshImporter", $"Failed to load baked data for '{name}': {ex.Message}");
            }

            return result;
        }

        /// <summary>
        /// Find existing baked navmesh bytes from the cache via the meta's subasset list.
        /// </summary>
        private static byte[] LoadExistingBakedData(string filepath)
        {
            // Resolve source GUID from file path
            var assetsDir = AssetDatabase.Project?.AssetsDirectory;
            if (assetsDir == null) return null;

            var relativePath = Path.GetRelativePath(assetsDir, filepath);
            var sourceGuid = AssetDatabase.PathToGuid(relativePath);
            if (sourceGuid == null) return null;

            var meta = AssetDatabase.GetMeta(sourceGuid);
            if (meta == null) return null;

            // Find the existing NavMeshData subasset
            var navSub = meta.SubAssets.FirstOrDefault(
                s => s.Type == nameof(NavMeshData));
            if (navSub == null) return null;

            // Load from cache
            var cachePath = AssetDatabase.ResolveCachePathByGuid(navSub.Guid);
            if (cachePath == null || !File.Exists(cachePath)) return null;

            var packer = new NavMeshDataPacker();
            using var stream = File.OpenRead(cachePath);
            var data = packer.Read(stream);
            return data?.BakedBytes;
        }
    }
}
