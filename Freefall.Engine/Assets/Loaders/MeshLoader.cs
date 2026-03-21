using System.Collections.Generic;
using System.IO;
using Freefall.Assets;
using Freefall.Graphics;

namespace Freefall.Assets.Loaders
{
    /// <summary>
    /// Loads Mesh assets from cache (.mesh files).
    /// Unpacks MeshData via MeshPacker, creates GPU buffers via Mesh.CreateAsync.
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

            return LoadFromCache(cachePath, name, manager);
        }

        public Asset LoadFromCache(string cachePath, string name, AssetManager manager, string sourceGuid = null)
        {
            MeshData meshData;
            using (var stream = File.OpenRead(cachePath))
                meshData = _packer.Read(stream);

            if (meshData.Positions?.Length > 0)
                Debug.Log($"[MeshLoader] '{name}' Vert[0]={meshData.Positions[0]} Vert[1]={meshData.Positions[1]} BBox={meshData.BoundingBox.Min}..{meshData.BoundingBox.Max}");
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
            mesh.RegisterMeshParts();
            return mesh;
        }
    }
}
