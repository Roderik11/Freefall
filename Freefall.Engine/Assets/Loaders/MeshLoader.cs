using System.IO;
using System.Linq;
using System.Collections.Generic;
using Freefall.Animation;
using Freefall.Assets;
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

            var mesh = Mesh.CreateAsync(Engine.Device, meshData);
            mesh.Name = name;

            // Resolve sibling Skeleton BEFORE registering mesh parts,
            // so MeshRegistry.NumBones is correct for GPU bone indexing.
            ResolveSkeleton(mesh, sourceGuid, manager);

            mesh.RegisterMeshParts();

            return mesh;
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
