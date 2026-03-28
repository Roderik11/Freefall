using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Freefall.Base;
using Freefall.Components;
using Freefall.Reflection;
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
        public byte[] SourceYaml { get; set; }

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
            var serializer = new EntitySerializer();
            var entities = serializer.LoadFromBytes(SourceYaml);

            if (entities.Count == 0)
            {
                Debug.LogWarning("Prefab", $"Cannot instantiate '{Name}': YAML produced 0 entities");
                return null;
            }

            // Tag all entities as instances of this prefab
            foreach (var e in entities)
                e.Prefab = this;

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

            var serializer = new EntitySerializer();
            var yaml = serializer.SaveToString(entities);

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

        /// <summary>
        /// Apply this prefab's components to an existing entity.
        /// Used during scene loading: the entity already has UID + Transform from the scene file,
        /// so we only add the components defined in the prefab (MeshRenderer, RigidBody, etc.).
        /// For multi-entity prefabs, child entities are re-parented to the target.
        /// </summary>
        public void ApplyTo(Entity target)
        {
            if (SourceYaml == null || SourceYaml.Length == 0) return;

            var serializer = new EntitySerializer();
            var prefabEntities = serializer.LoadFromBytes(SourceYaml, skipPrefabHydration: true);

            if (prefabEntities.Count == 0) return;

            // Copy non-Transform components from the prefab's root entity to the target
            var root = prefabEntities[0];
            foreach (var component in root.Components)
            {
                if (component is Transform) continue;
                target.AddComponent(component);
            }

            // Re-parent child entities: any entity whose Transform.Parent was the root
            // gets re-parented to the target's Transform
            for (int i = 1; i < prefabEntities.Count; i++)
            {
                var child = prefabEntities[i];
                if (child.Transform.Parent == root.Transform)
                    child.Transform.Parent = target.Transform;

                child.Prefab = this;
            }

            // Tag it
            target.Prefab = this;

            // Only remove the temporary root entity (children stay alive and registered)
            EntityManager.RemoveEntity(root);
        }

        /// <summary>
        /// Capture an instance's current state back into this prefab's source YAML,
        /// then propagate the changes to all other instances.
        /// Flow: instance → prefab asset → all other instances.
        /// Transform is reset to identity in the saved prefab.
        /// Returns the number of other instances updated.
        /// </summary>
        public int UpdateFromInstance(Entity source)
        {
            // Temporarily zero out transform for serialization
            var savedPos = source.Transform.Position;
            var savedRot = source.Transform.Rotation;
            var savedScale = source.Transform.Scale;
            var savedParent = source.Transform.Parent;

            source.Transform.Position = System.Numerics.Vector3.Zero;
            source.Transform.Rotation = System.Numerics.Quaternion.Identity;
            source.Transform.Scale = System.Numerics.Vector3.One;
            source.Transform.Parent = null;

            // Serialize to YAML
            var serializer = new EntitySerializer();
            var yaml = serializer.SaveToString(new List<Entity> { source });

            // Restore transform
            source.Transform.Position = savedPos;
            source.Transform.Rotation = savedRot;
            source.Transform.Scale = savedScale;
            source.Transform.Parent = savedParent;

            // Update the prefab asset
            SourceYaml = Encoding.UTF8.GetBytes(yaml);
            RootEntityName = source.Name;
            MarkDirty();

            Debug.Log($"[Prefab] Updated prefab '{Name}' from instance '{source.Name}'");

            // Propagate to all other instances
            return UpdateAllInstances();
        }

        /// <summary>
        /// Delta-update a single live entity from prefab source.
        /// Uses a pre-deserialized template to avoid redundant YAML parsing.
        /// Matches components by type: copies fields for existing, adds new, removes orphaned.
        /// Preserves Transform (position, rotation, scale).
        /// </summary>
        public void UpdateInstance(Entity target, Entity template)
        {
            // Build type → component map from template (skip Transform)
            var templateComponents = new Dictionary<Type, Component>();
            foreach (var comp in template.Components)
            {
                if (comp is Transform) continue;
                templateComponents[comp.GetType()] = comp;
            }

            // Walk target's non-Transform components
            var toRemove = new List<Component>();
            var matchedTypes = new HashSet<Type>();

            foreach (var comp in target.Components)
            {
                if (comp is Transform) continue;
                var type = comp.GetType();

                if (templateComponents.TryGetValue(type, out var templateComp))
                {
                    // Type exists in both — delta copy fields
                    CopyFields(templateComp, comp);
                    matchedTypes.Add(type);
                }
                else
                {
                    // Orphaned: exists in target but not in template
                    toRemove.Add(comp);
                }
            }

            // Remove orphaned components
            foreach (var comp in toRemove)
                target.RemoveComponent(comp);

            // Add new components (in template but not in target)
            foreach (var (type, templateComp) in templateComponents)
            {
                if (matchedTypes.Contains(type)) continue;
                target.AddComponent(templateComp);
            }

            // Ensure prefab link
            target.Prefab = this;
        }

        /// <summary>
        /// Update all live instances of this prefab in the scene.
        /// Deserializes YAML once and reuses the template for all instances.
        /// Returns the number of entities updated.
        /// </summary>
        public int UpdateAllInstances()
        {
            if (SourceYaml == null || SourceYaml.Length == 0) return 0;

            // Deserialize once
            var serializer = new EntitySerializer();
            var templateEntities = serializer.LoadFromBytes(SourceYaml, skipPrefabHydration: true);
            if (templateEntities.Count == 0) return 0;

            var template = templateEntities[0];

            // Find all instances
            var instances = EntityManager.Entities
                .Where(e => e.Prefab == this)
                .ToList();

            foreach (var instance in instances)
                UpdateInstance(instance, template);

            // Clean up temporary template entities
            foreach (var te in templateEntities)
                EntityManager.RemoveEntity(te);

            Debug.Log($"[Prefab] Updated {instances.Count} instances of '{Name}'");
            return instances.Count;
        }

        /// <summary>
        /// Copy serializable fields from source component to target component (same type).
        /// Uses Reflector.GetMapping for field discovery, skips DontSerialize/Ignored fields.
        /// </summary>
        private static void CopyFields(Component source, Component target)
        {
            var mapping = Reflector.GetMapping(source.GetType());

            foreach (var field in mapping)
            {
                if (field.Ignored) continue;
                if (!field.CanWrite) continue;
                if (field.Name == "Entity") continue; // don't overwrite entity backref

                try
                {
                    var value = field.GetValue(source);
                    field.SetValue(target, value);
                }
                catch { /* skip fields that fail to copy */ }
            }
        }
    }
}
