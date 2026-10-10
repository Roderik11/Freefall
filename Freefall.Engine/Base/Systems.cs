using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using System.Text;
using Freefall.Reflection;

namespace Freefall.Base
{
    /// <summary>
    /// Owns every <see cref="EntitySystem"/>: one instance per concrete type found in the assemblies
    /// registered with <see cref="Reflector"/>. Systems of a script assembly are dropped when it unloads.
    /// </summary>
    public static class Systems
    {
        public static readonly UpdateGroup UpdateGroup = new() { RunsInEditor = true };
        public static readonly RenderGroup RenderGroup = new() { RunsInEditor = true };

        private static readonly Dictionary<Type, EntitySystem> _systems = new()
        {
            [typeof(UpdateGroup)] = UpdateGroup,
            [typeof(RenderGroup)] = RenderGroup,
        };

        private static readonly HashSet<Assembly> _assemblies = [];
        private static ConcurrentQueue<Assembly> _pending = new();

        static Systems()
        {
            _pending.Enqueue(typeof(Systems).Assembly);
        }

        public static T? Get<T>() where T : EntitySystem
            => _systems.TryGetValue(typeof(T), out var system) ? (T)system : null;

        /// <summary>Queue an assembly; its systems are created on the main thread before the next group update.</summary>
        public static void RegisterAssembly(Assembly assembly) => _pending.Enqueue(assembly);

        /// <summary>
        /// Main thread: destroy and forget the systems whose type comes from this assembly, so nothing here
        /// keeps its load context alive.
        /// </summary>
        public static void UnregisterAssembly(Assembly assembly)
        {
            if (!_pending.IsEmpty)
                _pending = new ConcurrentQueue<Assembly>(_pending.Where(a => a != assembly));

            if (!_assemblies.Remove(assembly)) return;

            var set = new[] { assembly };
            var stale = _systems.Values.Where(s => Reflector.ReferencesAssembly(s.GetType(), set)).ToList();
            var orphans = new List<EntitySystem>();

            foreach (var system in stale)
            {
                system.Group?.Remove(system);
                _systems.Remove(system.GetType());

                if (system is SystemGroup group)
                    orphans.AddRange(group.Systems);

                try { system.Destroy(); }
                catch (Exception ex)
                {
                    Debug.LogWarning("Systems", $"{system.GetType().Name}.Destroy() threw: {ex.Message}");
                }
            }

            // Members of a removed group that are staying (defined elsewhere) need a new home
            foreach (var system in orphans)
            {
                if (!_systems.ContainsKey(system.GetType())) continue;
                system.Group = null;
                Attach(system);
            }
        }

        /// <summary>Once per frame, from the engine tick.</summary>
        public static void RunUpdate()
        {
            ProcessPending();
            Run(UpdateGroup);
        }

        /// <summary>Once per rendered view, from the renderer.</summary>
        public static void RunDraw()
        {
            ProcessPending();
            Run(RenderGroup);
        }

        private static void Run(SystemGroup root)
        {
            long start = Stopwatch.GetTimestamp();
            root.Update();
            root.ElapsedMs = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
        }

        private static void ProcessPending()
        {
            if (_pending.IsEmpty) return;

            var created = new List<EntitySystem>();

            while (_pending.TryDequeue(out var assembly))
            {
                if (!_assemblies.Add(assembly)) continue;

                Type[] types;
                try { types = assembly.GetTypes(); }
                catch (ReflectionTypeLoadException ex) { types = ex.Types.Where(t => t != null).ToArray()!; }

                foreach (var type in types)
                {
                    if (type.IsAbstract || !type.IsSubclassOf(typeof(EntitySystem)) || _systems.ContainsKey(type))
                        continue;

                    if (type.GetConstructor(Type.EmptyTypes) == null)
                    {
                        Debug.LogWarning("Systems", $"{type.Name} has no public parameterless constructor and was not created");
                        continue;
                    }

                    try
                    {
                        var system = (EntitySystem)Activator.CreateInstance(type)!;
                        system.RunsInEditor = system is SystemGroup || type.IsDefined(typeof(UpdateInEditorAttribute), false);
                        _systems[type] = system;
                        created.Add(system);
                    }
                    catch (Exception ex)
                    {
                        Debug.LogError("Systems", $"Could not create {type.Name}: {(ex.InnerException ?? ex).Message}");
                    }
                }
            }

            // Attach once all of them exist, so a system can name a group from the same batch
            foreach (var system in created)
                Attach(system);

            foreach (var system in created)
            {
                try { system.Initialize(); }
                catch (Exception ex)
                {
                    Debug.LogError("Systems", $"{system.GetType().Name}.Initialize() threw: {ex.Message}");
                }
            }
        }

        private static void Attach(EntitySystem system)
        {
            var type = system.GetType();
            var groupType = type.GetCustomAttribute<UpdateInGroupAttribute>(false)?.Group ?? typeof(UpdateGroup);

            if (!_systems.TryGetValue(groupType, out var target) || target is not SystemGroup group)
            {
                Debug.LogWarning("Systems", $"{type.Name}: group {groupType?.Name} does not exist, using UpdateGroup");
                group = UpdateGroup;
            }

            // A group placed inside itself (directly or through its own members) would never be updated
            for (SystemGroup? ancestor = group; ancestor != null; ancestor = ancestor.Group)
            {
                if (ancestor != system) continue;
                Debug.LogWarning("Systems", $"{type.Name}: [UpdateInGroup({groupType.Name})] puts the group inside itself, using UpdateGroup");
                group = UpdateGroup;
                break;
            }

            group.Add(system);
        }

        /// <summary>The system tree in update order with the last update's timings, for diagnostics.</summary>
        public static string Describe()
        {
            var sb = new StringBuilder();
            Describe(sb, UpdateGroup, 0);
            Describe(sb, RenderGroup, 0);
            return sb.ToString();
        }

        private static void Describe(StringBuilder sb, EntitySystem system, int depth)
        {
            sb.Append(' ', depth * 2).Append(system.GetType().Name)
              .Append("  ").Append(system.ElapsedMs.ToString("F3")).Append(" ms");
            if (!system.Enabled) sb.Append("  (disabled)");
            else if (Engine.IsEditor && !system.RunsInEditor) sb.Append("  (play mode only)");
            sb.Append('\n');

            if (system is not SystemGroup group) return;
            foreach (var member in group.Systems)
                Describe(sb, member, depth + 1);
        }
    }
}
