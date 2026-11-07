using System;
using System.IO;
using System.Linq;
using System.Text;
using Freefall.Assets.Packers;
using Freefall.Base;
using Freefall.Components;
using Freefall.Serialization;

namespace Freefall.Assets.Loaders
{
    /// <summary>
    /// Loads and saves Terrain assets.
    /// Load: unpacks AssetDefinitionData (YAML) from cache, deserializes Terrain, resolves GUID references, then reads
    ///       the heights and holes from its pages file (<see cref="TerrainPagesFile"/>) into the CPU store.
    /// Save: writes the CPU store's heights and holes into the pages file, then the YAML. Since 4c (2026-11-07); before, a
    ///       save read the whole map back from the GPU into a flat R16 DDS, wrote the hole mask as a flat R8 DDS and cooked
    ///       one PhysX height field of the whole terrain beside them.
    /// </summary>
    [AssetLoader(typeof(Terrain), ".terrain")]
    public class TerrainLoader : IAssetLoader
    {
        private readonly AssetDefinitionPacker _packer = new();

        public Asset Load(string name, AssetManager manager)
        {
            var cachePath = AssetDatabase.ResolveCachePath(name, "AssetDefinitionData");
            if (cachePath == null || !Artifacts.Exists(cachePath))
                throw new FileNotFoundException($"Cache file not found for terrain '{name}'");

            return LoadFromCache(cachePath, name, manager);
        }

        public Asset LoadFromCache(string cachePath, string name, AssetManager manager, string sourceGuid = null)
        {
            try
            {
                Debug.Log($"[TerrainLoader] Loading '{name}' from {cachePath}");

                AssetDefinitionData defData;
                using (var stream = Artifacts.Open(cachePath))
                    defData = _packer.Read(stream);

                // Deserialize YAML and auto-resolve all asset GUID references
                var yaml = Encoding.UTF8.GetString(defData.YamlBytes);
                Terrain terrain;
                try
                {
                    terrain = NativeImporter.LoadFromString(yaml, manager) as Terrain;
                }
                catch (Exception ex)
                {
                    throw new InvalidDataException(
                        $"Failed to deserialize Terrain from cache: {name} - {ex.GetType().Name}: {ex.Message}", ex);
                }

                if (terrain == null)
                    throw new InvalidDataException($"Failed to deserialize Terrain from cache: {name}");

                terrain.Name = name;

                // The heights and holes as saved, so GetHeight, the collider's cook and a server have them before (or
                // without) a render
                LoadPages(terrain);

                // What the bakes are compared with: a bake that comes out different puts the terrain on Save Changed Assets
                terrain.MarkMapsSaved();

                Debug.Log($"[TerrainLoader] '{name}' loaded: heights={terrain.Pages is { HasHeights: true }}, holes={terrain.HasHoles}");

                MessageDispatcher.Send("TerrainLoaded", terrain);

                return terrain;
            }
            catch (Exception ex)
            {
                Debug.LogWarning("TerrainLoader", $"FAILED to load '{name}': {ex}");
                return null;
            }
        }

        /// <summary>
        /// Reads the pages file into the CPU store (Terrain.Pages) and leaves its heights for the renderer to upload
        /// (Terrain.PendingPagesUpload). A terrain never saved since 4c has none: its stamps bake it, and the first bake
        /// puts it on Save Changed Assets. Pages of another density than the terrain's (changed and not saved) are left
        /// for the stamps too.
        /// </summary>
        private static void LoadPages(Terrain terrain)
        {
            terrain.SetPages(null);
            terrain.PendingPagesUpload = false;
            if (string.IsNullOrEmpty(terrain.PagesGuid))
                return;

            var path = TerrainPagesFile.PathOf(terrain.PagesGuid);
            if (!Artifacts.Exists(path))
            {
                Debug.LogWarning("TerrainLoader", $"'{terrain.Name}': its pages file {terrain.PagesGuid} is missing; the stamps bake the heights");
                return;
            }

            try
            {
                long started = System.Diagnostics.Stopwatch.GetTimestamp();
                TerrainPages pages;
                using (var stream = Artifacts.Open(path))
                    pages = TerrainPagesFile.Read(stream);

                if (pages.Resolution != terrain.EffectiveHeightmapResolution)
                {
                    Debug.Log($"[TerrainLoader] '{terrain.Name}': pages saved for {pages.Resolution}² samples, the terrain has {terrain.EffectiveHeightmapResolution}²; the stamps bake it");
                    return;
                }

                terrain.SetPages(pages);
                terrain.PendingPagesUpload = pages.HasHeights;
                terrain.PagesStampsHash = terrain.BakedStampsHash;   // the saved heights, from the saved stamps
                Debug.LogAlways($"[TerrainLoader] '{terrain.Name}': {pages.PagesAcross}² pages read ({pages.HolePages} with holes) " +
                                $"in {System.Diagnostics.Stopwatch.GetElapsedTime(started).TotalMilliseconds:0} ms");
            }
            catch (Exception ex)
            {
                Debug.LogWarning("TerrainLoader", $"'{terrain.Name}': reading its pages file failed ({ex.Message}); the stamps bake the heights");
                terrain.SetPages(null);
                terrain.PendingPagesUpload = false;
            }
        }

        // ── Save ──

        /// <summary>
        /// Save the heights and holes into the pages file, then the terrain YAML. A failure is thrown on, so the asset
        /// stays changed (AssetManager.SaveAsset clears it only after a save that returned) and the caller reports it:
        /// swallowed, the editor took a terrain that had not been written off its changed list (16 km at 1 m per
        /// vertex, 2026-11-07).
        /// </summary>
        public void Save(Asset asset, string savePath)
        {
            if (asset is not Terrain terrain) return;

            try
            {
                SavePages(terrain);
                terrain.MarkMapsSaved();

                // The YAML names the pages file (PagesGuid) and the stamps its heights came from
                NativeImporter.Save(savePath, terrain);
                Debug.Log($"[TerrainLoader] YAML saved: {savePath}");
            }
            catch (Exception ex)
            {
                Debug.LogWarning("TerrainLoader", $"Failed to save terrain: {ex.Message}");
                throw;
            }
        }

        /// <summary>
        /// Writes the CPU store into the pages file: the heights the GPU baked (read back once its bakes have settled; a
        /// readback that is due runs now) and the holes. Nothing to write when nothing changed since the file was read or
        /// written, or when there are no heights (nothing baked or loaded).
        /// </summary>
        private static void SavePages(Terrain terrain)
        {
            var renderer = ComponentCache<TerrainRenderer>.All.FirstOrDefault(r => r.Terrain == terrain);
            renderer?.FinishHeightsForSave();

            var pages = terrain.Pages;
            if (pages is not { HasHeights: true })
                return;

            if (terrain.PagesStampsHash != 0)
                terrain.BakedStampsHash = terrain.PagesStampsHash;

            string path = string.IsNullOrEmpty(terrain.PagesGuid) ? null : TerrainPagesFile.PathOf(terrain.PagesGuid);
            if (path != null && terrain.MapsAreSaved && File.Exists(path))
                return;

            if (path == null)
            {
                terrain.PagesGuid = Guid.NewGuid().ToString("N");
                path = TerrainPagesFile.PathOf(terrain.PagesGuid);
            }

            long started = System.Diagnostics.Stopwatch.GetTimestamp();
            long bytes = TerrainPagesFile.Write(path, pages);
            Debug.LogAlways($"[TerrainLoader] '{terrain.Name}': {pages.PagesAcross}² pages saved ({pages.HolePages} with holes), " +
                            $"{bytes / 1048576.0:0.0} MB in {System.Diagnostics.Stopwatch.GetElapsedTime(started).TotalMilliseconds:0} ms");
        }

        private static readonly System.Text.RegularExpressions.Regex PagesRef = new(
            @"^\s*PagesGuid:\s*""?([0-9a-fA-F]{32})""?\s*$",
            System.Text.RegularExpressions.RegexOptions.Multiline);

        /// <summary>
        /// The terrain's pages file (PagesGuid), read from its YAML: no meta lists it, so the hook gives its path in the
        /// Cache root, where LoadPages finds it.
        /// </summary>
        public IEnumerable<ImplicitArtifact> GetImplicitDependencies(string guid)
        {
            var cachePath = AssetDatabase.ResolveCachePathByGuid(guid);
            if (cachePath == null || !Artifacts.Exists(cachePath)) yield break;

            AssetDefinitionData defData;
            using (var stream = Artifacts.Open(cachePath))
                defData = _packer.Read(stream);

            foreach (System.Text.RegularExpressions.Match m in PagesRef.Matches(Encoding.UTF8.GetString(defData.YamlBytes)))
            {
                var pagesGuid = m.Groups[1].Value;
                yield return new ImplicitArtifact(pagesGuid, TerrainPagesFile.PathOf(pagesGuid), "TerrainPages");
            }
        }
    }
}
