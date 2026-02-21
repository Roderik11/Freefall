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
    /// Unpacks AssetDefinitionData (YAML) → MaterialDefinition,
    /// then resolves Effect + Texture GUIDs into a live GPU Material.
    /// </summary>
    [AssetLoader(typeof(Material))]
    public class MaterialLoader : IAssetLoader
    {
        private readonly AssetDefinitionPacker _packer = new();

        public Asset Load(string name, AssetManager manager)
        {
            var cachePath = AssetDatabase.ResolveCachePath(name);
            if (cachePath == null || !File.Exists(cachePath))
                throw new FileNotFoundException($"Cache file not found for material '{name}'");

            return LoadFromCache(cachePath, name, manager);
        }

        public Asset LoadFromCache(string cachePath, string name, AssetManager manager)
        {
            AssetDefinitionData defData;
            using (var stream = File.OpenRead(cachePath))
                defData = _packer.Read(stream);

            // Deserialize YAML → MaterialDefinition
            var yaml = Encoding.UTF8.GetString(defData.YamlBytes);
            var def = NativeImporter.LoadFromString(yaml) as MaterialDefinition;

            if (def == null)
                throw new InvalidDataException($"Failed to deserialize Material from cache: {name}");

            // Resolve Effect
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

            if (effect == null)
                effect = InternalAssets.DefaultEffect;

            // Create live Material
            var material = new Material(effect);
            material.Name = name;

            // Resolve and bind textures
            foreach (var (slot, texRef) in def.TextureRefs)
            {
                var texture = manager.LoadByGuid<Texture>(texRef);
                if (texture != null)
                    material.SetTexture(slot, texture);
                else
                    Debug.LogWarning("MaterialLoader", $"Could not resolve texture '{texRef}' for slot '{slot}'");
            }

            return material;
        }
    }
}
