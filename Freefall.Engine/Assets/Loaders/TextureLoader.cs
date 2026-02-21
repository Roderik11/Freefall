using System;
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
            var cachePath = AssetDatabase.ResolveCachePath(name);
            if (cachePath == null || !File.Exists(cachePath))
                throw new FileNotFoundException($"Cache file not found for texture '{name}'");

            return LoadFromCache(cachePath, name, manager);
        }

        public Asset LoadFromCache(string cachePath, string name, AssetManager manager)
        {
            DdsTextureData ddsData;
            using (var stream = File.OpenRead(cachePath))
                ddsData = _packer.Read(stream);

            // ParseDDS expects a file path — write bytes to temp for now.
            // TODO: Add Texture.ParseDDSFromBytes(byte[]) to avoid temp file.
            var tempPath = Path.Combine(Path.GetTempPath(), $"freefall_tex_{Guid.NewGuid():N}.dds");
            try
            {
                File.WriteAllBytes(tempPath, ddsData.Bytes);
                var cpuData = Texture.ParseFromFile(Engine.Device, tempPath);
                var texture = Texture.CreateAsync(Engine.Device, cpuData);
                texture.Name = name;
                return texture;
            }
            finally
            {
                try { File.Delete(tempPath); } catch { }
            }
        }
    }
}
