using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Freefall.Assets;
using Freefall.Base;
using Freefall.Components;
using Freefall.Reflection;

namespace Freefall.Serialization
{
    /// <summary>
    /// Loads .scene YAML files in apex-style multi-document format.
    /// Two-phase loading: parse with deferred asset stubs, then resolve afterward.
    /// This prevents nested LoadByGuid calls from corrupting the YAML parser state.
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
            // No LoadByGuid calls during parse = no parser corruption.
            _serializer.DeferAssetLoading = true;
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

            // ── Phase 2: Resolve deferred Asset stubs ──
            // Now that parsing is done, resolve every Asset stub via LoadByGuid.
            if (Engine.Assets != null)
                ResolveAssetStubs(entities);

            Debug.Log($"[SceneLoader] Loaded {entities.Count} entities");

            return entities;
        }

        /// <summary>
        /// Walk all components on all entities. For each Asset-typed field
        /// that holds a stub (has Guid but not loaded), resolve via LoadByGuid.
        /// </summary>
        private static void ResolveAssetStubs(List<Entity> entities)
        {
            foreach (var entity in entities)
            {
                foreach (var component in entity.Components)
                {
                    var mapping = Reflector.GetMapping(component.GetType());
                    foreach (var field in mapping)
                    {
                        if (!typeof(Asset).IsAssignableFrom(field.Type))
                            continue;

                        var stub = field.GetValue(component) as Asset;
                        if (stub == null || string.IsNullOrEmpty(stub.Guid))
                            continue;

                        try
                        {
                            var loaded = Engine.Assets.LoadByGuid(stub.Guid, field.Type);
                            if (loaded != null)
                                field.SetValue(component, loaded);
                        }
                        catch (Exception ex)
                        {
                            Debug.LogWarning("SceneLoader",
                                $"Stub resolve failed: {field.Type.Name} " +
                                $"'{stub.Guid}' on {component.GetType().Name}: {ex.Message}");
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
