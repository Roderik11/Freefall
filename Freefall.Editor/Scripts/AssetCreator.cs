using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using Freefall.Assets;
using Freefall.Base;
using Freefall.Reflection;
using Freefall.Serialization;

namespace Freefall.Editor
{
    /// <summary>
    /// Shared helper for creating and saving assets from the editor.
    /// Used by both UI menus and the CreateAssetCommand API endpoint.
    /// Also tracks dirty assets via MessageDispatcher for save/warn workflows.
    /// </summary>
    public static class AssetCreator
    {
        private static readonly HashSet<Asset> _dirtyAssets = new();

        /// <summary>
        /// True if any loaded asset has unsaved changes.
        /// </summary>
        public static bool HasChangedAssets => _dirtyAssets.Count > 0;

        /// <summary>
        /// Number of assets with unsaved changes.
        /// </summary>
        public static int DirtyAssetCount => _dirtyAssets.Count;

        /// <summary>
        /// Subscribe to AssetDirty messages. Call once at editor startup.
        /// </summary>
        public static void Initialize()
        {
            MessageDispatcher.AddListener(Msg.AssetDirty, (msg) =>
            {
                if (msg.Data is Asset asset)
                    _dirtyAssets.Add(asset);
            });
        }

        /// <summary>
        /// Save all dirty assets to disk and clear their dirty flags.
        /// Evicts stale entries from AssetManager cache and triggers reimport.
        /// Returns the number of assets saved.
        /// </summary>
        public static int SaveChangedAssets()
        {
            int saved = 0;

            foreach (var asset in _dirtyAssets)
            {
                string savePath = null;

                if (!string.IsNullOrEmpty(asset.Guid))
                    savePath = AssetDatabase.GuidToPath(asset.Guid);

                if (!string.IsNullOrEmpty(savePath) && Engine.Project != null)
                    savePath = Path.Combine(Engine.Project.AssetsDirectory, savePath);

                if (string.IsNullOrEmpty(savePath))
                {
                    Debug.Log($"[AssetCreator] Cannot save dirty asset '{asset.Name}': no source path");
                    continue;
                }

                try
                {
                    Engine.Assets.SaveAsset(asset, savePath);
                    saved++;

                    // Reimport source → cache so the packed binary matches
                    if (!string.IsNullOrEmpty(asset.Guid))
                    {
                        var relativePath = AssetDatabase.GuidToPath(asset.Guid);
                        if (relativePath != null)
                            AssetDatabase.ImportAssetByPath(relativePath);
                    }

                    Debug.Log($"[AssetCreator] Saved: {asset.Name} → {savePath}");
                }
                catch (Exception ex)
                {
                    Debug.Log($"[AssetCreator] Failed to save '{asset.Name}': {ex.Message}");
                }
            }

            _dirtyAssets.Clear();

            if (saved > 0)
                MessageDispatcher.Send(Msg.RefreshAssets);

            return saved;
        }

        /// <summary>Unsaved state of an asset whose class is defined in project scripts, on its way across a script reload.</summary>
        internal readonly record struct UnsavedScriptAsset(string Guid, string TypeName, string Name, string Yaml);

        /// <summary>
        /// Script hot reload, while the old script assemblies are still loaded: take the assets whose class they
        /// define out of the dirty list (the instances would keep the old assembly from unloading) and return
        /// the state of those that have unsaved changes, for <see cref="RestoreScriptAssets"/>.
        /// </summary>
        internal static List<UnsavedScriptAsset> DetachScriptAssets(ICollection<Assembly> assemblies)
        {
            var unsaved = new List<UnsavedScriptAsset>();

            foreach (var asset in _dirtyAssets.Where(a => Reflector.ReferencesAssembly(a.GetType(), assemblies)).ToList())
            {
                _dirtyAssets.Remove(asset);

                // Saved since it was edited (saving one asset clears its flag, not this list), or never had a file
                if (!asset.IsDirty || string.IsNullOrEmpty(asset.Guid)) continue;

                try
                {
                    unsaved.Add(new UnsavedScriptAsset(asset.Guid, asset.GetType().FullName, asset.Name, NativeImporter.SaveToString(asset)));
                }
                catch (Exception ex)
                {
                    Debug.LogWarning("ScriptReload", $"Unsaved changes to '{asset.Name}' could not be kept: {ex.Message}");
                }
            }

            return unsaved;
        }

        /// <summary>
        /// After the reload: load each asset again as its new class, put the unsaved state back on it and mark
        /// it dirty.
        /// </summary>
        internal static void RestoreScriptAssets(List<UnsavedScriptAsset> unsaved)
        {
            foreach (var entry in unsaved)
            {
                try
                {
                    var type = Reflector.GetType(entry.TypeName);
                    var asset = type != null && typeof(Asset).IsAssignableFrom(type) ? Engine.Assets.LoadByGuid(entry.Guid, type) : null;
                    if (asset != null && NativeImporter.LoadInto(asset, entry.Yaml, Engine.Assets))
                    {
                        asset.MarkDirty();
                        continue;
                    }
                }
                catch (Exception ex)
                {
                    Debug.LogWarning("ScriptReload", $"'{entry.Name}': {ex.Message}");
                }

                Debug.LogWarning("ScriptReload", $"Unsaved changes to '{entry.Name}' ({entry.TypeName}) were lost: the asset could not be loaded as that type after the reload");
            }
        }

        /// <summary>Why the last CreateAsset call returned null (for callers that report errors).</summary>
        public static string LastError { get; private set; }

        /// <summary>
        /// Create a new asset of the given type, save to disk, refresh AssetDatabase,
        /// import it, and return the GUID. Returns null on failure (see LastError).
        /// </summary>
        public static string CreateAsset(Type assetType, string folderPath, string name = null)
        {
            LastError = null;
            name ??= "New " + assetType.Name;

            string ext = AssetManager.GetFileExtension(assetType);
            string savePath = Path.Combine(folderPath, name + ext);

            // Avoid overwriting existing files
            savePath = GetUniquePath(savePath);
            name = Path.GetFileNameWithoutExtension(savePath);

            try
            {
                var asset = (Asset)Activator.CreateInstance(assetType);
                asset.Name = name;

                Engine.Assets.SaveAsset(asset, savePath);

                // Register in AssetDatabase
                AssetDatabase.Refresh();
                string relativePath = GetRelativePath(savePath);
                var guid = AssetDatabase.PathToGuid(relativePath);

                if (guid == null)
                    throw new InvalidOperationException($"'{relativePath}' was saved but the AssetDatabase did not register it");
                AssetDatabase.ImportAssetByPath(relativePath);

                Debug.Log($"[AssetCreator] Created {assetType.Name}: {savePath}");
                return guid;
            }
            catch (Exception ex)
            {
                LastError = ex.Message;
                Debug.LogWarning("AssetCreator", $"Failed to create {assetType.Name}: {ex.Message}");
                // Don't leave a half-created file behind (it would block the name next time)
                try { if (File.Exists(savePath)) File.Delete(savePath); } catch { }
                return null;
            }
        }


        /// <summary>
        /// Get a unique file path by appending (1), (2), etc. if the file already exists.
        /// </summary>
        private static string GetUniquePath(string path)
        {
            if (!File.Exists(path)) return path;

            var dir = Path.GetDirectoryName(path);
            var name = Path.GetFileNameWithoutExtension(path);
            var ext = Path.GetExtension(path);
            int counter = 1;

            while (File.Exists(path))
            {
                path = Path.Combine(dir, $"{name} ({counter}){ext}");
                counter++;
            }

            return path;
        }

        public static string GetRelativePath(string fullPath)
        {
            var assetsDir = Engine.Project?.AssetsDirectory;
            if (string.IsNullOrEmpty(assetsDir)) return fullPath;

            return Path.GetRelativePath(assetsDir, fullPath).Replace('\\', '/');
        }
    }
}

