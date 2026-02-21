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
            var cachePath = AssetDatabase.ResolveCachePath(name);
            if (cachePath == null || !File.Exists(cachePath))
                throw new FileNotFoundException($"Cache file not found for mesh '{name}'");

            return LoadFromCache(cachePath, name, manager);
        }

        public Asset LoadFromCache(string cachePath, string name, AssetManager manager)
        {
            MeshData meshData;
            using (var stream = File.OpenRead(cachePath))
                meshData = _packer.Read(stream);

            // Match Apex MeshReader: sort parts alphabetically by name
            meshData.Parts.Sort((a, b) => a.Name.CompareTo(b.Name));

            var mesh = Mesh.CreateAsync(Engine.Device, meshData);
            mesh.Name = name;
            mesh.RegisterMeshParts();
            return mesh;
        }
    }
}
