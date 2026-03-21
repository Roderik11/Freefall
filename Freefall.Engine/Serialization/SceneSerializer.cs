using System.Collections.Generic;
using System.IO;
using System.Text;
using Freefall.Base;

namespace Freefall.Serialization
{
    /// <summary>
    /// Serializes entities and their components to .scene YAML files.
    /// Prefab instances emit only Entity (with Prefab GUID) + Transform;
    /// non-prefab entities emit full inline component data.
    /// </summary>
    public class SceneSerializer
    {
        private readonly YAMLSerializer _yaml = new();

        public void Save(string path, IEnumerable<Entity> entities)
        {
            File.WriteAllText(path, SaveToString(entities), Encoding.UTF8);
        }

        public string SaveToString(IEnumerable<Entity> entities)
        {
            var emitter = _yaml.Begin();

            foreach (var entity in entities)
            {
                if (entity.HideAndDontSave) continue;

                // Skip child entities of prefab instances — they're reconstructed on load
                if (entity.IsPrefabInstance && entity.Transform.Parent != null)
                    continue;

                emitter.SetTag("---");
                _yaml.Serialize(entity, ref emitter);

                if (entity.IsPrefabInstance)
                {
                    // Prefab instance: only emit Transform (position/rotation/scale).
                    // All other components come from the prefab on load.
                    emitter.SetTag("---");
                    _yaml.Serialize(entity.Transform, ref emitter);
                }
                else
                {
                    // Inline entity: emit all components
                    foreach (var component in entity.Components)
                    {
                        emitter.SetTag("---");
                        _yaml.Serialize(component, ref emitter);
                    }
                }
            }

            return _yaml.ToString(emitter);
        }
    }
}
