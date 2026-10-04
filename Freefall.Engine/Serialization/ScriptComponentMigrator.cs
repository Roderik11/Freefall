using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Freefall.Assets;
using Freefall.Base;
using Freefall.Reflection;

namespace Freefall.Serialization
{
    /// <summary>
    /// Moves live components across a script hot-reload. Without it, components already in the scene keep their
    /// types from the unloaded script assembly: new members are unknown, new code never runs and the old
    /// AssemblyLoadContext can't be collected.
    ///
    /// Sequence (main thread, between frames):
    ///   1. <see cref="Capture"/>  — while the old assembly is loaded: record every component whose type comes from
    ///      it (entity, index in Components, UID, scene-save YAML, live values of members whose type the reload
    ///      doesn't touch).
    ///   2. <see cref="Detach"/>   — remove the old instances (Destroy() + ComponentCache removal), then
    ///      <see cref="ForgetTypes"/> drops engine caches keyed by the old types.
    ///   3. caller unloads the old assembly and registers the new one with Reflector.
    ///   4. <see cref="Restore"/>  — create the same-named type from the new assembly at the old index, restore
    ///      matching members, resolve Entity/Component/asset references, remap references to the old instances.
    ///      New components wake up (Awake) on the next frame like freshly loaded ones.
    ///
    /// Member values: script-defined data (enums, [Serializable] script classes, lists of them, references to
    /// script components) round-trips through the same YAML the scene save writes; members whose type isn't
    /// defined by the script assemblies (floats, vectors, lists of primitives, assets, Entity refs) are copied
    /// directly, so runtime-only assets survive too. Members that no longer exist are dropped with a warning.
    ///
    /// A component whose type disappeared is removed (warning) and its YAML kept as an orphan for the session:
    /// if a later reload brings the type back while the entity is still alive, it is restored.
    /// </summary>
    public sealed class ScriptComponentMigrator
    {
        private sealed class Entry
        {
            public Entity Entity;
            public Component Old;           // null for an orphan retried from an earlier reload
            public string TypeName;         // full name of the old type
            public int Index;
            public ulong UID;
            public string Yaml;
            public readonly Dictionary<string, (Type Type, object Value)> Direct = new();
            public readonly HashSet<string> Applied = new();  // Direct members actually set on New
            public Component New;
        }

        private sealed class Orphan
        {
            public WeakReference<Entity> Entity;
            public string TypeName;
            public int Index;
            public ulong UID;
            public string Yaml;
        }

        private static readonly List<Orphan> _orphans = new();

        private readonly HashSet<Assembly> _assemblies;
        private readonly List<Entry> _entries = new();
        private readonly Dictionary<string, HashSet<string>> _oldMembers = new();
        private readonly YAMLSerializer _yaml = new();

        public int Count => _entries.Count;

        private ScriptComponentMigrator(IEnumerable<Assembly> assemblies)
        {
            _assemblies = new HashSet<Assembly>(assemblies);
        }

        // ── 1. Capture ────────────────────────────────────────────

        /// <summary>
        /// Snapshot every live component whose type is defined in one of the script assemblies.
        /// Call before anything is unregistered — serialization needs the old types' mappings.
        /// </summary>
        public static ScriptComponentMigrator Capture(IEnumerable<Assembly> scriptAssemblies)
        {
            var migrator = new ScriptComponentMigrator(scriptAssemblies);
            if (migrator._assemblies.Count == 0) return migrator;

            foreach (var entity in EntityManager.Entities.ToList())
            {
                var components = entity.Components;
                for (int i = 0; i < components.Count; i++)
                {
                    if (migrator._assemblies.Contains(components[i].GetType().Assembly))
                        migrator.CaptureComponent(entity, components[i], i);
                }
            }

            return migrator;
        }

        private void CaptureComponent(Entity entity, Component component, int index)
        {
            var type = component.GetType();
            var entry = new Entry
            {
                Entity = entity,
                Old = component,
                TypeName = type.FullName,
                Index = index,
                UID = component.UID,
            };

            try
            {
                entry.Yaml = _yaml.Serialize(component);
            }
            catch (Exception ex)
            {
                Debug.LogWarning("ScriptReload", $"{type.Name} on '{entity.Name}': serialize failed ({ex.Message}); only engine-typed members are carried over");
            }

            if (!_oldMembers.TryGetValue(type.FullName, out var members))
                _oldMembers[type.FullName] = members = new HashSet<string>();

            foreach (var field in Reflector.GetMapping(type))
            {
                if (!field.CanWrite || field.Ignored) continue;
                members.Add(field.Name);

                // Script-defined member types can't be carried as-is: those go through the YAML
                if (Reflector.ReferencesAssembly(field.Type, _assemblies)) continue;

                try { entry.Direct[field.Name] = (field.Type, field.GetValue(component)); }
                catch { /* throwing getter: the YAML (if any) still has it */ }
            }

            _entries.Add(entry);
        }

        // ── 2. Detach ─────────────────────────────────────────────

        /// <summary>
        /// Remove the captured components from their entities: Destroy() runs with the old code (unhooking
        /// listeners, clearing generated output) and each leaves its ComponentCache.
        /// </summary>
        public void Detach()
        {
            foreach (var entry in _entries)
            {
                try
                {
                    entry.Entity.RemoveComponent(entry.Old);
                }
                catch (Exception ex)
                {
                    Debug.LogWarning("ScriptReload", $"{entry.Old.GetType().Name}.Destroy() on '{entry.Entity.Name}' threw: {ex.Message}");
                }
            }
        }

        /// <summary>
        /// Drop engine-side caches keyed by types from these assemblies (ScriptExecution dispatch list,
        /// Entity's ComponentCache&lt;T&gt; type map). Call after <see cref="Detach"/>; Reflector has its own
        /// UnregisterAssembly.
        /// </summary>
        public static void ForgetTypes(ICollection<Assembly> assemblies)
        {
            if (assemblies.Count == 0) return;
            bool Stale(Type t) => Reflector.ReferencesAssembly(t, assemblies);
            ScriptExecution.RemoveCaches(Stale);
            Entity.ForgetCacheTypes(Stale);
        }

        // ── 4. Restore ────────────────────────────────────────────

        /// <summary>
        /// Recreate the captured components from the currently registered (new) script types.
        /// Returns the number of components migrated.
        /// </summary>
        public int Restore()
        {
            RetryOrphans();

            _yaml.DeferredRefs.Clear();
            _yaml.DeferredUniqueIdRefs.Clear();

            var oldToNew = new Dictionary<object, object>(ReferenceEqualityComparer.Instance);
            var newToEntry = new Dictionary<object, Entry>(ReferenceEqualityComparer.Instance);
            var missing = new Dictionary<string, List<string>>();   // type → entity names
            var dropped = new Dictionary<string, HashSet<string>>(); // type → member names
            int restored = 0;

            // Per entity in ascending index: all script components were removed, so inserting in order puts
            // each back exactly where it was (a component whose type vanished just closes the gap).
            foreach (var entry in _entries.OrderBy(e => e.Entity.Id).ThenBy(e => e.Index))
            {
                if (!IsAlive(entry.Entity)) continue; // e.g. generated output destroyed by its owner's Destroy()

                var type = ResolveType(entry.TypeName);
                if (type == null)
                {
                    if (entry.Yaml != null)
                    {
                        _orphans.Add(new Orphan
                        {
                            Entity = new WeakReference<Entity>(entry.Entity),
                            TypeName = entry.TypeName,
                            Index = entry.Index,
                            UID = entry.UID,
                            Yaml = entry.Yaml,
                        });
                    }
                    if (!missing.TryGetValue(entry.TypeName, out var names))
                        missing[entry.TypeName] = names = new List<string>();
                    names.Add(entry.Entity.Name);
                    continue;
                }

                try
                {
                    var component = (Component)Activator.CreateInstance(type);

                    if (entry.Yaml != null)
                    {
                        try { _yaml.Populate(entry.Yaml, component); }
                        catch (Exception ex)
                        {
                            Debug.LogWarning("ScriptReload", $"{type.Name} on '{entry.Entity.Name}': reading saved members failed ({ex.Message})");
                        }
                    }

                    var mapping = Reflector.GetMapping(type);
                    foreach (var (name, (oldType, value)) in entry.Direct)
                    {
                        // Same engine-side type on both sides: carry the live value. A changed type was converted
                        // through the YAML above; a member that's gone is reported below.
                        if (!mapping.TryGetValue(name, out var field) || !field.CanWrite || field.Ignored || field.Type != oldType)
                            continue;
                        try { field.SetValue(component, value); entry.Applied.Add(name); }
                        catch (Exception ex)
                        {
                            Debug.LogWarning("ScriptReload", $"{type.Name}.{name} on '{entry.Entity.Name}': {ex.Message}");
                        }
                    }

                    CollectDropped(entry.TypeName, mapping, dropped);

                    component.UID = entry.UID;
                    entry.Entity.InsertComponent(entry.Index, component);
                    entry.New = component;
                    newToEntry[component] = entry;
                    if (entry.Old != null)
                        oldToNew[entry.Old] = component;
                    restored++;
                }
                catch (Exception ex)
                {
                    Debug.LogWarning("ScriptReload", $"Could not recreate {type.Name} on '{entry.Entity.Name}': {ex.Message}");
                }
            }

            ResolveUniqueIdRefs(newToEntry);
            ResolveAssetStubs(newToEntry);
            if (oldToNew.Count > 0)
                RemapReferences(oldToNew);

            foreach (var (typeName, members) in dropped)
            {
                if (members.Count == 0) continue;
                Debug.LogWarning("ScriptReload",
                    $"{ShortName(typeName)}: member(s) {string.Join(", ", members)} no longer exist — their values were dropped");
            }

            foreach (var (typeName, entities) in missing)
            {
                Debug.LogWarning("ScriptReload",
                    $"Script type '{typeName}' no longer exists — removed from {entities.Count} entit{(entities.Count == 1 ? "y" : "ies")} " +
                    $"({string.Join(", ", entities.Distinct().Take(5))}{(entities.Count > 5 ? ", ..." : "")}). " +
                    "Its data is kept for this session and restored if the type comes back; reload the scene without saving to recover it");
            }

            return restored;
        }

        /// <summary>
        /// Components orphaned by an earlier reload whose type resolves again join this restore.
        /// Orphans whose entity is gone (scene reloaded, entity deleted) are forgotten.
        /// </summary>
        private void RetryOrphans()
        {
            for (int i = _orphans.Count - 1; i >= 0; i--)
            {
                var orphan = _orphans[i];
                if (!orphan.Entity.TryGetTarget(out var entity) || !IsAlive(entity))
                {
                    _orphans.RemoveAt(i);
                    continue;
                }
                if (ResolveType(orphan.TypeName) == null) continue;

                _orphans.RemoveAt(i);
                _entries.Add(new Entry
                {
                    Entity = entity,
                    TypeName = orphan.TypeName,
                    Index = orphan.Index,
                    UID = orphan.UID,
                    Yaml = orphan.Yaml,
                });
                Debug.LogAlways($"[ScriptReload] Restored {ShortName(orphan.TypeName)} on '{entity.Name}' (type is back)");
            }
        }

        private Type ResolveType(string fullName)
        {
            // Reflector.GetType also follows [FormerlySerializedAs] on a renamed class
            var type = Reflector.GetType(fullName);
            if (type == null || type.IsAbstract || !typeof(Component).IsAssignableFrom(type)) return null;
            if (_assemblies.Contains(type.Assembly)) return null; // still the old type: assembly wasn't unregistered
            return type;
        }

        private static bool IsAlive(Entity entity) => ReferenceEquals(EntityManager.GetEntity(entity.Id), entity);

        private void CollectDropped(string typeName, Mapping newMapping, Dictionary<string, HashSet<string>> dropped)
        {
            if (dropped.ContainsKey(typeName) || !_oldMembers.TryGetValue(typeName, out var oldMembers)) return;
            var set = new HashSet<string>();
            foreach (var name in oldMembers)
            {
                if (!newMapping.TryGetValue(name, out var field) || !field.CanWrite || field.Ignored)
                    set.Add(name);
            }
            dropped[typeName] = set;
        }

        /// <summary>
        /// Entity/Component references read from the YAML → live objects by UID. Members carried over directly
        /// already hold the live object and are skipped.
        /// </summary>
        private void ResolveUniqueIdRefs(Dictionary<object, Entry> newToEntry)
        {
            _yaml.DeferredUniqueIdRefs.RemoveAll(d => CarriedDirectly(newToEntry, d.Parent, d.Field));
            if (_yaml.DeferredUniqueIdRefs.Count == 0) return;

            var lookup = new Dictionary<ulong, IUniqueId>();
            foreach (var entity in EntityManager.Entities.ToList())
            {
                if (entity.UID != 0) lookup.TryAdd(entity.UID, entity);
                foreach (var component in entity.Components)
                {
                    if (component.UID != 0) lookup.TryAdd(component.UID, component);
                }
            }

            foreach (var deferred in _yaml.DeferredUniqueIdRefs)
            {
                if (!lookup.TryGetValue(deferred.UID, out var target))
                {
                    Debug.LogWarning("ScriptReload", $"{deferred.Parent.GetType().Name}.{deferred.Field.Name}: referenced UID {deferred.UID} not found");
                    continue;
                }
                try { deferred.Field.SetValue(deferred.Parent, target); }
                catch (Exception ex)
                {
                    Debug.LogWarning("ScriptReload", $"{deferred.Parent.GetType().Name}.{deferred.Field.Name}: {ex.Message}");
                }
            }
            _yaml.DeferredUniqueIdRefs.Clear();
        }

        /// <summary>Asset GUID stubs read from the YAML → loaded assets (members carried directly are skipped).</summary>
        private void ResolveAssetStubs(Dictionary<object, Entry> newToEntry)
        {
            foreach (var r in _yaml.DeferredRefs)
            {
                if (CarriedDirectly(newToEntry, r.Parent, r.Field))
                    continue;
                if (Engine.Assets == null) break;

                try
                {
                    var loaded = Engine.Assets.LoadByGuid(r.Guid, r.AssetType);
                    if (loaded == null) continue;

                    if (r.ListIndex >= 0 && r.Parent is IList list)
                        list[r.ListIndex] = loaded;
                    else if (r.Field != null)
                        r.Field.SetValue(r.Parent, loaded);
                }
                catch (Exception ex)
                {
                    Debug.LogWarning("ScriptReload", $"Failed to resolve {r.AssetType.Name} '{r.Guid}': {ex.Message}");
                }
            }
            _yaml.DeferredRefs.Clear();
        }

        /// <summary>
        /// Point public members that still hold an old instance (an engine component's Component/interface-typed
        /// field, a carried-over List&lt;Component&gt;) at its replacement.
        /// </summary>
        private static void RemapReferences(Dictionary<object, object> oldToNew)
        {
            foreach (var entity in EntityManager.Entities.ToList())
            {
                foreach (var component in entity.Components)
                {
                    foreach (var field in Reflector.GetMapping(component.GetType()))
                    {
                        if (!field.CanWrite) continue;
                        try
                        {
                            if (CanHoldComponent(field.Type))
                            {
                                var value = field.GetValue(component);
                                if (value != null && oldToNew.TryGetValue(value, out var replacement))
                                    field.SetValue(component, replacement);
                            }
                            else if (typeof(IList).IsAssignableFrom(field.Type) && CanHoldComponent(ElementType(field.Type)))
                            {
                                if (field.GetValue(component) is not IList list) continue;
                                for (int i = 0; i < list.Count; i++)
                                {
                                    if (list[i] != null && oldToNew.TryGetValue(list[i], out var replacement))
                                        list[i] = replacement;
                                }
                            }
                        }
                        catch { /* getter/setter with side effects that throws: leave it */ }
                    }
                }
            }
        }

        private static bool CarriedDirectly(Dictionary<object, Entry> newToEntry, object parent, Field field)
            => field != null && newToEntry.TryGetValue(parent, out var entry) && entry.Applied.Contains(field.Name);

        private static bool CanHoldComponent(Type type)
            => type != null && !type.IsValueType && !type.IsSealed
               && (type.IsInterface || type == typeof(object) || typeof(Component).IsAssignableFrom(type));

        private static Type ElementType(Type listType)
        {
            if (listType.HasElementType) return listType.GetElementType();
            if (listType.IsGenericType) return listType.GetGenericArguments()[0];
            return null;
        }

        private static string ShortName(string fullName)
        {
            int dot = fullName.LastIndexOf('.');
            return dot >= 0 ? fullName[(dot + 1)..] : fullName;
        }
    }
}
