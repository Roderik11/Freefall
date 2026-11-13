using System;
using System.Buffers.Binary;
using System.IO;
using System.Runtime.InteropServices;
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

            // A material's texture of some size loads cold: its copy is made while something draws with the material
            // (Material.OnDemand.cs). Nothing is read now; any other load of it later makes it resident (AssetManager, Pin)
            if (t_forMaterial && OnDemandTextures.MaterialTexturesOnDemand
                && Artifacts.Length(cachePath) >= OnDemandTextures.ColdMinBytes)
                return Texture.ColdFrom(cachePath, name);

            var texture = Texture.CreateAsync(Engine.Device, ReadCpuData(cachePath, name));
            texture.Name = name;
            return texture;
        }

        [System.ThreadStatic] private static bool t_forMaterial;

        /// <summary>Whether the calling thread is resolving a material's texture (<see cref="LoadForMaterial"/>).</summary>
        internal static bool ForMaterial => t_forMaterial;

        /// <summary>
        /// A material's texture by GUID (MaterialLoader): loaded cold when it is not loaded yet and its cache file is at
        /// least <see cref="OnDemandTextures.ColdMinBytes"/>; the instance already loaded otherwise, cold or not.
        /// </summary>
        public static Texture LoadForMaterial(AssetManager manager, string guid)
        {
            bool was = t_forMaterial;
            t_forMaterial = true;
            try { return manager.LoadByGuid<Texture>(guid); }
            finally { t_forMaterial = was; }
        }

        private static readonly int PackerVersion = new DdsTexturePacker().Version;

        /// <summary>
        /// The texture's pixels from its cache file, parsed. The DDS file goes straight from the stream into native memory
        /// (<see cref="CpuTextureData.Native"/>, freed by the upload): no array on the managed heap, where every texture of
        /// 1024² and up was a large object that was garbage once uploaded. Any thread.
        /// </summary>
        public static unsafe CpuTextureData ReadCpuData(string cachePath, string name)
        {
            IntPtr block;
            int length;
            using (var stream = Artifacts.Open(cachePath))
            {
                // DdsTexturePacker's layout: the packer's version, the DDS file's length, the DDS file
                Span<byte> head = stackalloc byte[8];
                stream.ReadExactly(head);
                int version = BinaryPrimitives.ReadInt32LittleEndian(head);
                length = BinaryPrimitives.ReadInt32LittleEndian(head[4..]);
                if (version != PackerVersion || length <= 0)
                    throw new InvalidDataException($"Texture cache file '{cachePath}': packer version {version}, length {length}");

                block = (IntPtr)NativeMemory.Alloc((nuint)length);
                try { stream.ReadExactly(new Span<byte>((void*)block, length)); }
                catch { NativeMemory.Free((void*)block); throw; }
            }

            var cpuData = Texture.ParseDDSNative(block, length);
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
