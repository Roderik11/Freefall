using System;
using System.Collections.Generic;
using System.Reflection;
using System.Linq;
using Freefall.Base;

namespace Freefall.Reflection
{
    /// <summary>
    /// Cached reflection utilities for type mapping, field discovery, and type resolution.
    /// Central to serialization (YAMLSerializer) and future inspector (GUIObject).
    /// </summary>
    public static class Reflector
    {
        // Assembly registry for cross-assembly type resolution
        private static readonly Dictionary<string, Assembly> _assemblies = new();

        // Type → ordered field mapping (base-class fields first)
        private static readonly Dictionary<Type, Mapping> _mappingCache = new();

        // Type name → Type (for deserialization)
        private static readonly Dictionary<string, Type> _typeCache = new();

        // FormerlySerializedAs lookups (old name → current type)
        private static readonly Dictionary<string, Type> _formerNames = new();

        // Subtype cache (base → all derived types across registered assemblies)
        private static readonly Dictionary<Type, List<Type>> _typesByBase = new();

        /// <summary>
        /// Initialize with the engine assembly. Call once at startup.
        /// Editor/game assemblies should be registered via RegisterAssemblies().
        /// </summary>
        static Reflector()
        {
            var engineAssembly = Assembly.GetAssembly(typeof(Component));
            if (engineAssembly != null)
            {
                RegisterAssemblies(engineAssembly);
            }
        }

        /// <summary>
        /// Register additional assemblies for type resolution (e.g. editor, game).
        /// Also scans for [FormerlySerializedAs] attributes.
        /// </summary>
        public static void RegisterAssemblies(params Assembly[] assemblies)
        {
            foreach (var assembly in assemblies)
            {
                if (!_assemblies.TryAdd(assembly.FullName!, assembly))
                    continue;

                // Scan for renamed types
                foreach (var type in assembly.GetTypes())
                {
                    var att = type.GetCustomAttribute<FormerlySerializedAsAttribute>();
                    if (att != null)
                        _formerNames.TryAdd(att.Name, type);
                }
            }
        }

        /// <summary>
        /// Get the ordered field mapping for a type. Fields are ordered base-class first.
        /// Results are cached.
        /// </summary>
        public static Mapping GetMapping(Type type)
        {
            if (_mappingCache.TryGetValue(type, out var cached))
                return cached;

            var fields = new List<Field>();

            foreach (var member in type.GetMembers())
            {
                var field = new Field(member);
                if (field.Type == null) continue;
                fields.Add(field);
            }

            // Build depth map: current type = 0, base = 1, base.base = 2, ...
            var depth = new Dictionary<Type, int>();
            int level = 0;
            depth[type] = level++;
            var parent = type.BaseType;
            while (parent != null)
            {
                depth[parent] = level++;
                parent = parent.BaseType;
            }

            // Order: base-class fields first (highest depth first)
            fields = fields
                .Where(f => depth.ContainsKey(f.DeclaringType))
                .OrderByDescending(f => depth[f.DeclaringType])
                .ToList();

            var mapping = new Mapping();
            mapping.AddRange(fields);
            _mappingCache[type] = mapping;

            return mapping;
        }

        /// <summary>
        /// Resolve a type name (e.g. "Freefall.Components.Transform") to its Type.
        /// Searches registered assemblies and handles FormerlySerializedAs renames.
        /// </summary>
        public static Type GetType(string fullName)
        {
            if (_typeCache.TryGetValue(fullName, out var cached))
                return cached;

            // Check renamed types
            if (_formerNames.TryGetValue(fullName, out var renamed))
            {
                _typeCache[fullName] = renamed;
                return renamed;
            }

            // Search all registered assemblies
            foreach (var assembly in _assemblies.Values)
            {
                var result = assembly.GetType(fullName);
                if (result != null)
                {
                    _typeCache[fullName] = result;
                    return result;
                }
            }

            return null;
        }

        /// <summary>
        /// Find a type by its simple (unqualified) name across all registered assemblies.
        /// E.g. "StaticMesh" → Type for Freefall.Assets.StaticMesh.
        /// Results are cached.
        /// </summary>
        public static Type FindTypeBySimpleName(string simpleName)
        {
            if (_typeCache.TryGetValue(simpleName, out var cached))
                return cached;

            foreach (var assembly in _assemblies.Values)
            {
                foreach (var type in assembly.GetTypes())
                {
                    if (type.Name == simpleName)
                    {
                        _typeCache[simpleName] = type;
                        return type;
                    }
                }
            }

            return null;
        }

        /// <summary>
        /// Get a specific field descriptor from a type by name.
        /// </summary>
        public static Field GetField(Type type, string name)
        {
            var mem = type.GetMember(name);
            if (mem.Length > 0)
                return new Field(mem[0]);
            return null;
        }

        /// <summary>
        /// Get all types that derive from or implement the given base type,
        /// across all registered assemblies.
        /// </summary>
        public static List<Type> GetTypes(Type baseType)
        {
            if (_typesByBase.TryGetValue(baseType, out var cached))
                return cached;

            var result = new List<Type>();

            foreach (var assembly in _assemblies.Values)
            {
                foreach (var type in assembly.GetTypes())
                {
                    if (baseType.IsInterface && baseType.IsAssignableFrom(type))
                        result.Add(type);
                    else if (type.IsSubclassOf(baseType))
                        result.Add(type);
                }
            }

            _typesByBase[baseType] = result;
            return result;
        }

        /// <summary>
        /// Get all types that derive from or implement T.
        /// </summary>
        public static List<Type> GetTypes<T>() => GetTypes(typeof(T));

        /// <summary>
        /// Check if a type has a specific attribute.
        /// </summary>
        public static T GetAttribute<T>(Type type, bool inherit = false) where T : Attribute
        {
            var atts = type.GetCustomAttributes(typeof(T), inherit);
            if (atts.Length > 0)
                return (T)atts[0];
            return null;
        }

        /// <summary>
        /// Extension: check if a type has an attribute.
        /// </summary>
        public static bool HasAttribute<T>(this Type type, bool inherit = false) where T : Attribute
        {
            return type.GetCustomAttributes(typeof(T), inherit).Length > 0;
        }
    }
}
