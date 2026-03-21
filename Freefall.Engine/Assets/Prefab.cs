using System;
using System.Collections.Generic;
using System.Text;
using Freefall.Base;
using Freefall.Serialization;

namespace Freefall.Assets
{
    /// <summary>
    /// A Prefab is a reusable entity template — a hierarchy of entities
    /// serialized as a flat list, identical to the .scene YAML format.
    /// Instantiation clones via serialize/deserialize with fresh UIDs.
    /// </summary>
    public class Prefab : Asset
    {
        /// <summary>
        /// Raw YAML source for the prefab. Stored so we can re-parse 
        /// on each Instantiate() call to get fresh instances with new UIDs.
        /// </summary>
        internal byte[] SourceYaml { get; set; }

        /// <summary>
        /// Name of the root entity in this prefab (for display/debugging).
        /// </summary>
        public string RootEntityName { get; set; }

        /// <summary>
        /// Number of entities in this prefab (for display/debugging).
        /// </summary>
        public int EntityCount { get; set; }

        /// <summary>
        /// Instantiate this prefab: parse the YAML, assign fresh UIDs,
        /// resolve asset references, and register entities in EntityManager.
        /// Returns the root entity (first in the list).
        /// </summary>
        public Entity Instantiate()
        {
            if (SourceYaml == null || SourceYaml.Length == 0)
            {
                Debug.LogWarning("Prefab", $"Cannot instantiate '{Name}': no source YAML");
                return null;
            }

            // SceneLoader handles everything:
            //   - Deserializes entities (Entity constructor auto-generates fresh UIDs + registers)
            //   - Resolves UID cross-references within the prefab
            //   - Resolves asset references via LoadByGuid
            var loader = new SceneLoader();
            var entities = loader.LoadFromBytes(SourceYaml);

            if (entities.Count == 0)
            {
                Debug.LogWarning("Prefab", $"Cannot instantiate '{Name}': YAML produced 0 entities");
                return null;
            }

            Debug.Log($"[Prefab] Instantiated '{Name}': {entities.Count} entities");
            return entities[0]; // root entity
        }

        /// <summary>
        /// Create a Prefab asset from a list of entities (e.g., from scene selection).
        /// Serializes the entities to YAML using the same format as .scene files.
        /// </summary>
        public static Prefab CreateFromEntities(List<Entity> entities, string name = null)
        {
            if (entities == null || entities.Count == 0)
                return null;

            var serializer = new YAMLSerializer();
            var emitter = serializer.Begin();

            foreach (var entity in entities)
            {
                serializer.Serialize(entity, ref emitter);
                foreach (var component in entity.Components)
                    serializer.Serialize(component, ref emitter);
            }

            var yaml = serializer.ToString(in emitter);

            var prefab = new Prefab
            {
                SourceYaml = Encoding.UTF8.GetBytes(yaml),
                RootEntityName = entities[0].Name ?? "Unnamed",
                EntityCount = entities.Count,
                Name = name ?? entities[0].Name ?? "Prefab",
            };

            return prefab;
        }

        /// <summary>
        /// Get the YAML source as a string (for saving to .prefab file).
        /// </summary>
        public string ToYaml()
        {
            if (SourceYaml == null) return string.Empty;
            return Encoding.UTF8.GetString(SourceYaml);
        }
    }
}
