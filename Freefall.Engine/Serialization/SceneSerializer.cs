using System.Collections.Generic;
using System.IO;
using System.Text;
using Freefall.Base;

namespace Freefall.Serialization
{
    /// <summary>
    /// Serializes entities and their components to .scene YAML files.
    /// Thin wrapper — delegates all field serialization to YAMLSerializer.
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

                emitter.SetTag("---");
                _yaml.Serialize(entity, ref emitter);

                foreach (var component in entity.Components)
                {
                    emitter.SetTag("---");
                    _yaml.Serialize(component, ref emitter);
                }
            }

            return _yaml.ToString(emitter);
        }
    }
}
