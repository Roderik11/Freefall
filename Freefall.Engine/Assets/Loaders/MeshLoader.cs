using System.IO;
using System.Linq;
using System.Collections.Generic;
using Freefall.Animation;
using Freefall.Assets;
using Freefall.Assets.Importers;
using Freefall.Graphics;

namespace Freefall.Assets.Loaders
{
    /// <summary>
    /// Loads Mesh assets from cache (.mesh files).
    /// Unpacks MeshData via MeshPacker, creates GPU buffers via Mesh.CreateAsync.
    /// Resolves sibling Skeleton sub-asset from the same source file.
    /// </summary>
    [AssetLoader(typeof(Mesh))]
    public class MeshLoader : IAssetLoader
    {
        private readonly MeshPacker _packer = new();

        public Asset Load(string name, AssetManager manager)
        {
            var cachePath = AssetDatabase.ResolveCachePath(name, "MeshData");
            if (cachePath == null || !File.Exists(cachePath))
                throw new FileNotFoundException($"Cache file not found for mesh '{name}'");

            // Extract GUID from cache path (format: .../XX/{guid}.mesh)
            var guid = Path.GetFileNameWithoutExtension(cachePath);

            return LoadFromCache(cachePath, name, manager, guid);
        }

        public Asset LoadFromCache(string cachePath, string name, AssetManager manager, string sourceGuid = null)
        {
            var mesh = Mesh.CreateAsync(Engine.Device, ReadMeshData(cachePath));
            mesh.Name = name;

            // Resolve sibling Skeleton BEFORE registering mesh parts,
            // so MeshRegistry.NumBones is correct for GPU bone indexing.
            ResolveSkeleton(mesh, sourceGuid, manager);
            ApplyMeshConfig(mesh, sourceGuid);

            mesh.RegisterMeshParts();

            return mesh;
        }

        public bool Reload(Asset existing, string cachePath, string name, AssetManager manager, string guid)
        {
            if (existing is not Mesh mesh || existing.GetType() != typeof(Mesh))
                return false;

            // Build the new buffers on a throwaway mesh (never registered), wait for the copy queue so
            // no frame can read half-uploaded buffers, then move them into the live instance.
            var fresh = Mesh.CreateAsync(Engine.Device, ReadMeshData(cachePath));
            ResolveSkeleton(fresh, guid, manager);
            StreamingManager.Instance?.Flush();

            mesh.ReplaceGeometry(fresh);
            ApplyMeshConfig(mesh, guid);
            return true;
        }

        /// <summary>
        /// Apply per-mesh settings stored on the source file's ModelImporter (ModelImporter.Meshes).
        /// These are not baked into the cache, so editing them needs no reimport.
        /// </summary>
        private static void ApplyMeshConfig(Mesh mesh, string meshGuid)
        {
            if (string.IsNullOrEmpty(meshGuid)) return;

            // Skip the importer deserialization unless a non-default bias could be stored
            var settings = AssetDatabase.GetMeta(meshGuid)?.ImporterSettings;
            if (settings == null || !settings.Contains("\"LODBias\"")) return;

            if (AssetDatabase.GetImporter(meshGuid) is not ModelImporter importer) return;

            var config = importer.FindMeshConfig(AssetDatabase.ResolveFriendlyName(meshGuid));
            if (config != null)
                mesh.LODBias = config.LODBias;
        }

        private MeshData ReadMeshData(string cachePath)
        {
            MeshData meshData;
            using (var stream = File.OpenRead(cachePath))
                meshData = _packer.Read(stream);

            // Match Apex MeshReader: sort parts alphabetically by name
            // Build remap table so LOD indices stay correct after reorder
            var originalOrder = new List<MeshPart>(meshData.Parts);
            meshData.Parts.Sort((a, b) => a.Name.CompareTo(b.Name));

            if (meshData.LODs.Count > 0)
            {
                // Build old→new index map
                var remap = new int[originalOrder.Count];
                for (int i = 0; i < originalOrder.Count; i++)
                    remap[i] = meshData.Parts.IndexOf(originalOrder[i]);

                foreach (var lod in meshData.LODs)
                {
                    for (int i = 0; i < lod.MeshPartIndices.Length; i++)
                        lod.MeshPartIndices[i] = remap[lod.MeshPartIndices[i]];
                }
            }

            return meshData;
        }

        private static void ResolveSkeleton(Mesh mesh, string meshGuid, AssetManager manager)
        {
            if (string.IsNullOrEmpty(meshGuid)) return;

            var skelEntry = AssetDatabase.FindSiblingSubAsset(meshGuid, nameof(Skeleton));
            if (skelEntry == null) return;

            var skeleton = manager.LoadByGuid<Skeleton>(skelEntry.Guid);
            if (skeleton != null)
                mesh.Skeleton = skeleton;
        }
    }
}
