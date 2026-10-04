using System;
using System.IO;
using System.Text;
using Freefall.Assets.Packers;
using Freefall.Assets.Serializers;
using Freefall.Graphics;
using Freefall.Serialization;

namespace Freefall.Assets.Loaders
{
    /// <summary>
    /// Loads Material assets from cache (.asset files).
    /// Unpacks AssetDefinitionData (YAML) â†’ MaterialDefinition,
    /// then resolves Effect + Texture GUIDs into a live GPU Material.
    /// </summary>
    [AssetLoader(typeof(Material), ".mat")]
    public class MaterialLoader : IAssetLoader
    {
        private readonly AssetDefinitionPacker _packer = new();

        public Asset Load(string name, AssetManager manager)
        {
            var cachePath = AssetDatabase.ResolveCachePath(name, "AssetDefinitionData");
            if (cachePath == null || !File.Exists(cachePath))
                throw new FileNotFoundException($"Cache file not found for material '{name}'");

            return LoadFromCache(cachePath, name, manager);
        }

        public Asset LoadFromCache(string cachePath, string name, AssetManager manager, string sourceGuid = null)
        {
            AssetDefinitionData defData;
            using (var stream = File.OpenRead(cachePath))
                defData = _packer.Read(stream);

            // Deserialize YAML â†’ MaterialDefinition
            var yaml = Encoding.UTF8.GetString(defData.YamlBytes);
            var def = NativeImporter.LoadFromString(yaml) as MaterialDefinition;

            if (def == null)
                throw new InvalidDataException($"Failed to deserialize Material from cache: {name}");

            // Create live Material
            var material = new Material(ResolveEffect(def, manager));
            material.Name = name;

            ApplyDefinition(material, def, manager);

            return material;
        }

        /// <summary>
        /// Hot reload: apply the reimported definition to the live Material, keeping its MaterialID
        /// so every renderer referencing it updates.
        /// </summary>
        public bool Reload(Asset existing, string cachePath, string name, AssetManager manager, string guid)
        {
            if (existing is not Material material)
                return false;

            MaterialDefinition def;
            using (var stream = File.OpenRead(cachePath))
                def = NativeImporter.LoadFromString(Encoding.UTF8.GetString(_packer.Read(stream).YamlBytes)) as MaterialDefinition;
            if (def == null)
                throw new InvalidDataException($"Failed to deserialize Material from cache: {name}");

            // Effects by name are shared instances with fresh wrappers, so compare names:
            // SetEffect rebuilds every PSO and should only run when the shader actually changed.
            var effect = ResolveEffect(def, manager);
            if (material.Effect == null || material.Effect.Name != effect.Name)
                material.SetEffect(effect);

            foreach (var slot in material.Textures.Keys.ToList())
            {
                if (!def.TextureRefs.ContainsKey(slot))
                    material.ClearTexture(slot);
            }

            ApplyDefinition(material, def, manager, resetMissing: true);
            return true;
        }

        private static Effect ResolveEffect(MaterialDefinition def, AssetManager manager)
        {
            Effect effect = null;
            if (!string.IsNullOrEmpty(def.EffectRef))
            {
                // Try as GUID first, then as name
                effect = manager.LoadByGuid<Effect>(def.EffectRef);
                if (effect == null)
                {
                    // Fall back to loading by name (e.g. "gbuffer")
                    effect = new Effect(def.EffectRef);
                }
            }

            return effect ?? InternalAssets.DefaultEffect;
        }

        private static void ApplyDefinition(Material material, MaterialDefinition def, AssetManager manager, bool resetMissing = false)
        {
            // Resolve and bind textures
            foreach (var (slot, texRef) in def.TextureRefs)
            {
                var texture = manager.LoadByGuid<Texture>(texRef);
                if (texture != null)
                    material.SetTexture(slot, texture);
                else
                    Debug.LogWarning("MaterialLoader", $"Could not resolve texture '{texRef}' for slot '{slot}'");
            }

            // Load material properties — data-driven via MaterialProperty.TryParseAndSet
            foreach (var prop in material.MaterialProperties)
            {
                if (def.Parameters.TryGetValue(prop.Name, out var val))
                    prop.TryParseAndSet(val);
                else if (resetMissing)
                    prop.ResetToDefault();
            }
        }

        /// <summary>
        /// Save a Material to its source .mat file as YAML, then re-import
        /// so the binary cache (.asset) stays in sync.
        /// </summary>
        public void Save(Asset asset, string savePath)
        {
            var serializer = new MaterialSerializer();
            var yaml = serializer.Serialize(asset);
            File.WriteAllText(savePath, yaml, Encoding.UTF8);
            Debug.Log($"[MaterialLoader] Saved: {savePath}");

            // Re-import to update the binary cache. A newly created file isn't tracked yet
            // (ImportAssetByPath would throw); the caller's Refresh registers and imports it.
            var assetsDir = Engine.Project?.AssetsDirectory;
            if (!string.IsNullOrEmpty(assetsDir) && savePath.StartsWith(assetsDir))
            {
                var relativePath = Path.GetRelativePath(assetsDir, savePath).Replace('\\', '/');
                if (AssetDatabase.PathToGuid(relativePath) != null)
                    AssetDatabase.ImportAssetByPath(relativePath);
            }
        }
    }
}
