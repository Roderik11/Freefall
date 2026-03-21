using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Freefall.Assets;
using Freefall.Base;
using Freefall.Components;
using Freefall.Reflection;

namespace Freefall.Serialization
{
    /// <summary>
    /// Loads .scene YAML files in apex-style multi-document format.
    /// Three-phase loading:
    ///   1. Parse YAML with deferred asset stubs (no LoadByGuid during parse)
    ///   2. Build UID lookup tables and resolve entity/component references
    ///   3. Resolve asset stubs via LoadByGuid
    /// </summary>
    public class SceneLoader
    {
        private readonly YAMLSerializer _serializer = new();

        public List<Entity> Load(string path)
        {
            var bytes = File.ReadAllBytes(path);
            return LoadFromBytes(bytes);
        }

        public List<Entity> LoadFromString(string yaml)
        {
            return LoadFromBytes(Encoding.UTF8.GetBytes(yaml));
        }

        public List<Entity> LoadFromBytes(byte[] bytes)
        {
            // Strip BOM if present
            if (bytes.Length >= 3 && bytes[0] == 0xEF && bytes[1] == 0xBB && bytes[2] == 0xBF)
            {
                var trimmed = new byte[bytes.Length - 3];
                Array.Copy(bytes, 3, trimmed, 0, trimmed.Length);
                bytes = trimmed;
            }

            // ── Phase 1: Parse with deferred asset loading ──
            // Asset fields get stub objects (Guid set, nothing else).
            // Entity/Component ref fields collect UIDs into DeferredUniqueIdRefs.
            // No LoadByGuid calls during parse = no parser corruption.
            _serializer.DeferAssetLoading = true;
            _serializer.DeferredUniqueIdRefs.Clear();

            List<object> objects;
            try
            {
                objects = _serializer.DeserializeAll(bytes);
            }
            finally
            {
                _serializer.DeferAssetLoading = false;
            }

            var entities = new List<Entity>();
            Entity current = null;

            foreach (var obj in objects)
            {
                if (obj is Entity entity)
                {
                    current = entity;
                    entities.Add(entity);
                }
                else if (obj is Component component && current != null)
                {
                    if (component is Transform t)
                    {
                        CopyFields(t, current.Transform);
                    }
                    else
                    {
                        current.AddComponent(component);
                    }
                }
            }

            // ── Phase 2: Build UID table and resolve IUniqueId references ──
            var uidLookup = new Dictionary<ulong, IUniqueId>();

            foreach (var entity in entities)
            {
                if (entity.UID != 0)
                    uidLookup.TryAdd(entity.UID, entity);

                foreach (var component in entity.Components)
                {
                    if (component.UID != 0)
                        uidLookup.TryAdd(component.UID, component);
                }
            }

            // Resolve deferred IUniqueId refs (entities + components)
            foreach (var deferred in _serializer.DeferredUniqueIdRefs)
            {
                if (uidLookup.TryGetValue(deferred.UID, out var resolved))
                {
                    try
                    {
                        deferred.Field.SetValue(deferred.Parent, resolved);
                    }
                    catch (Exception ex)
                    {
                        Debug.LogWarning("SceneLoader",
                            $"UID ref resolve failed: UID {deferred.UID} on " +
                            $"{deferred.Parent.GetType().Name}.{deferred.Field.Name}: {ex.Message}");
                    }
                }
                else
                {
                    Debug.LogWarning("SceneLoader",
                        $"UID ref not found: UID {deferred.UID} for " +
                        $"{deferred.Parent.GetType().Name}.{deferred.Field.Name}");
                }
            }

            _serializer.DeferredUniqueIdRefs.Clear();

            // ── Phase 3: Resolve deferred Asset stubs ──
            // Now that parsing is done, resolve every Asset stub via LoadByGuid.
            if (Engine.Assets != null)
                ResolveAssetStubs(entities);

            // ── Phase 4: Hydrate prefab instances ──
            // Entities saved as prefab references only have Entity + Transform in the scene file.
            // Now that Prefab asset stubs are resolved, apply the prefab's components.
            int prefabCount = 0;
            foreach (var entity in entities)
            {
                if (entity.Prefab != null)
                {
                    entity.Prefab.ApplyTo(entity);
                    prefabCount++;
                }
            }

            Debug.Log($"[SceneLoader] Loaded {entities.Count} entities " +
                      $"({prefabCount} prefab instances, {uidLookup.Count} UIDs resolved)");

            return entities;
        }

        /// <summary>
        /// Walk all components on all entities. For each Asset-typed field
        /// that holds a stub (has Guid but not loaded), resolve via LoadByGuid.
        /// Also recurses into List fields of [Serializable] elements.
        /// </summary>
        private static void ResolveAssetStubs(List<Entity> entities)
        {
            foreach (var entity in entities)
            {
                foreach (var component in entity.Components)
                    ResolveObjectStubs(component, component.GetType().Name);
            }
        }

        private static void ResolveObjectStubs(object obj, string context)
        {
            var mapping = Reflector.GetMapping(obj.GetType());
            foreach (var field in mapping)
            {
                // Direct Asset field
                if (typeof(Asset).IsAssignableFrom(field.Type))
                {
                    var stub = field.GetValue(obj) as Asset;
                    if (stub == null || string.IsNullOrEmpty(stub.Guid))
                        continue;

                    try
                    {
                        var loaded = Engine.Assets.LoadByGuid(stub.Guid, field.Type);
                        if (loaded != null)
                            field.SetValue(obj, loaded);
                    }
                    catch (Exception ex)
                    {
                        Debug.LogWarning("SceneLoader",
                            $"Stub resolve failed: {field.Type.Name} " +
                            $"'{stub.Guid}' on {context}: {ex.Message}");
                    }
                    continue;
                }

                // List of [Serializable] objects — recurse into each element
                if (typeof(IList).IsAssignableFrom(field.Type))
                {
                    var list = field.GetValue(obj) as IList;
                    if (list == null || list.Count == 0) continue;

                    var elType = field.Type.IsArray
                        ? field.Type.GetElementType()
                        : field.Type.IsGenericType
                            ? field.Type.GetGenericArguments()[0]
                            : null;

                    if (elType != null && Reflector.GetAttribute<SerializableAttribute>(elType) != null)
                    {
                        for (int i = 0; i < list.Count; i++)
                        {
                            if (list[i] != null)
                                ResolveObjectStubs(list[i], $"{context}.{field.Name}[{i}]");
                        }
                    }
                }
            }
        }

        private static void CopyFields(object source, object target)
        {
            var mapping = Reflector.GetMapping(source.GetType());
            foreach (var field in mapping)
            {
                if (!field.CanWrite) continue;
                var value = field.GetValue(source);
                if (value != null)
                    field.SetValue(target, value);
            }
        }
    }
}
