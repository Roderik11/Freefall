using System.IO;
using Freefall.Assets;
using Freefall.Assets.Packers;
using Freefall.Graphics;

namespace Freefall.Assets.Loaders
{
    /// <summary>
    /// Loads Texture assets from cache (.dds files).
    /// Unpacks DDS bytes via DdsTexturePacker, parses to CpuTextureData,
    /// and creates GPU texture via Texture.CreateAsync.
    /// </summary>
    [AssetLoader(typeof(Texture))]
    public class TextureLoader : IAssetLoader
    {
        private readonly DdsTexturePacker _packer = new();

        public Asset Load(string name, AssetManager manager)
        {
            var cachePath = AssetDatabase.ResolveCachePath(name, "DdsTextureData");
            if (cachePath == null || !File.Exists(cachePath))
                throw new FileNotFoundException($"Cache file not found for texture '{name}'");

            return LoadFromCache(cachePath, name, manager);
        }

        public Asset LoadFromCache(string cachePath, string name, AssetManager manager, string sourceGuid = null)
        {
            DdsTextureData ddsData;
            using (var stream = File.OpenRead(cachePath))
                ddsData = _packer.Read(stream);

            var cpuData = Texture.ParseDDSFromBytes(ddsData.Bytes);
            cpuData.Path = name;
            var texture = Texture.CreateAsync(Engine.Device, cpuData);
            texture.Name = name;
            return texture;
        }

        public bool Reload(Asset existing, string cachePath, string name, AssetManager manager, string guid)
        {
            // Only plain cache-loaded textures; render targets and arrays own their resources differently.
            if (existing is not Texture texture || existing.GetType() != typeof(Texture) || texture.BindlessIndex == 0)
                return false;

            var fresh = (Texture)LoadFromCache(cachePath, name, manager, guid);
            StreamingManager.Instance?.Flush();

            texture.ReplaceResource(fresh);
            return true;
        }
    }
}
