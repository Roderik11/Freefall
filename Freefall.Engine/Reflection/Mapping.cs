using System.Collections;
using System.Collections.Generic;

namespace Freefall.Reflection
{
    /// <summary>
    /// Ordered collection of Field descriptors with O(1) name lookup.
    /// Used by Reflector for type mapping and by YAMLSerializer for field iteration.
    /// </summary>
    public class Mapping : IEnumerable<Field>
    {
        private readonly List<Field> _fields = new();
        private readonly Dictionary<string, Field> _cache = new();

        public int Count => _fields.Count;

        public void SetValue(string name, object instance, object value)
        {
            if (_cache.TryGetValue(name, out var field))
                field.SetValue(instance, value);
        }

        public bool TryGetValue(string name, out Field field)
            => _cache.TryGetValue(name, out field);

        public void Add(Field field)
        {
            if (_cache.TryAdd(field.Name, field))
                _fields.Add(field);
        }

        /// <summary>
        /// Register an alias name that maps to an existing field.
        /// Used for backward-compatible deserialization when a field has been renamed.
        /// </summary>
        public void AddAlias(string oldName, Field field)
        {
            _cache.TryAdd(oldName, field);
        }

        public void AddRange(IEnumerable<Field> collection)
        {
            foreach (var field in collection)
                Add(field);
        }

        public void Sort(Comparison<Field> comparison)
            => _fields.Sort(comparison);

        public IEnumerator<Field> GetEnumerator()
            => _fields.GetEnumerator();

        IEnumerator IEnumerable.GetEnumerator()
            => GetEnumerator();
    }
}
