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
    /// Uses YAMLSerializer.DeserializeAll for parsing.
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

            var objects = _serializer.DeserializeAll(bytes);
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
                        // Transform already exists on entity — copy field values
                        CopyFields(t, current.Transform);
                    }
                    else
                    {
                        current.AddComponent(component);
                    }
                }
            }

            Debug.Log($"[SceneLoader] Loaded {entities.Count} entities");

            return entities;
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
