using System.IO;
using Freefall.Assets;
using Freefall.Assets.Packers;
using Freefall.Graphics;

namespace Freefall.Assets.Loaders
{
    /// <summary>
    /// Loads Texture assets from cache (.dds files).
    /// Unpacks DDS bytes via DdsTexturePacker, parses to CpuTextureData,
    /// and creates GPU texture via Texture.CreateAsync. A texture imported with LoadOnDemand loads without its GPU copy
    /// (<see cref="Texture.OnDemand"/>): nothing is read until somebody acquires it.
    /// </summary>
    [AssetLoader(typeof(Texture))]
    public class TextureLoader : IAssetLoader
    {
        public Asset Load(string name, AssetManager manager)
        {
            var cachePath = AssetDatabase.ResolveCachePath(name, "DdsTextureData");
            if (cachePath == null || !Artifacts.Exists(cachePath))
                throw new FileNotFoundException($"Cache file not found for texture '{name}'");

            return LoadFromCache(cachePath, name, manager);
        }

        public Asset LoadFromCache(string cachePath, string name, AssetManager manager, string sourceGuid = null)
        {
            if (IsOnDemand(sourceGuid))
                return Texture.OnDemandFrom(cachePath, name);

            var texture = Texture.CreateAsync(Engine.Device, ReadCpuData(cachePath, name));
            texture.Name = name;
            return texture;
        }

        /// <summary>The texture's pixels from its cache file, parsed. Any thread.</summary>
        public static CpuTextureData ReadCpuData(string cachePath, string name)
        {
            DdsTextureData ddsData;
            using (var stream = Artifacts.Open(cachePath))
                ddsData = new DdsTexturePacker().Read(stream);

            var cpuData = Texture.ParseDDSFromBytes(ddsData.Bytes);
            cpuData.Path = name;
            return cpuData;
        }

        /// <summary>
        /// Imported with TextureImporter.LoadOnDemand. The settings are read from the meta as JSON written without
        /// indentation (AssetDatabase), so a look for the field set to true answers without parsing them for every texture
        /// loaded. A published game's manifest keeps importer settings, so it loads them on demand too.
        /// </summary>
        private static bool IsOnDemand(string guid)
        {
            if (string.IsNullOrEmpty(guid)) return false;
            var settings = AssetDatabase.GetMeta(guid)?.ImporterSettings;
            return settings != null && settings.Contains("\"" + nameof(Importers.TextureImporter.LoadOnDemand) + "\":true");
        }

        public bool Reload(Asset existing, string cachePath, string name, AssetManager manager, string guid)
        {
            // On demand: read from the new file the next time somebody holds it (a held copy is loaded again now). Whether
            // a texture is on demand changes with its next load, not here: the scene and the asset cache hold this instance,
            // and a material may hold its bindless index.
            if (existing is Texture { OnDemand: true } onDemand)
            {
                onDemand.ReloadOnDemand(cachePath);
                Freefall.Base.MessageDispatcher.Send("AssetReloaded", existing);
                return true;
            }

            // Only plain cache-loaded textures; render targets and arrays own their resources differently.
            if (existing is not Texture texture || existing.GetType() != typeof(Texture) || texture.BindlessIndex == 0)
                return false;

            var fresh = Texture.CreateAsync(Engine.Device, ReadCpuData(cachePath, name));
            fresh.Name = name;
            StreamingManager.Instance?.Flush();

            texture.ReplaceResource(fresh);

            // Whoever copied the old pixels (the terrain's layer arrays) rebuilds from the new ones
            Freefall.Base.MessageDispatcher.Send("AssetReloaded", existing);
            return true;
        }
    }
}
