using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Reflection;

namespace Freefall.Base
{
    /// <summary>
    /// Per-frame logic that runs once for all components it cares about, instead of once per component
    /// (<see cref="IUpdate"/>). Systems are discovered by reflection (see <see cref="Systems"/>), need a
    /// parameterless constructor, and are placed with <see cref="UpdateInGroupAttribute"/>,
    /// <see cref="UpdateBeforeAttribute"/> and <see cref="UpdateAfterAttribute"/>.
    ///
    /// Systems run one after another on the main thread; a system may fan its own loop out over its
    /// components. Without <see cref="UpdateInEditorAttribute"/> a system only runs in play mode.
    /// </summary>
    public abstract class EntitySystem
    {
        public bool Enabled = true;

        /// <summary>Duration of the last <see cref="Update"/> in milliseconds; 0 when it was skipped.</summary>
        public double ElapsedMs { get; internal set; }

        /// <summary>The group that updates this system; null for the root groups.</summary>
        public SystemGroup? Group { get; internal set; }

        internal bool RunsInEditor;

        /// <summary>Called once on the main thread after every system of the same batch exists.</summary>
        protected internal virtual void Initialize() { }

        public abstract void Update();

        /// <summary>
        /// Called when the system is removed (its script assembly is unloading). Undo here whatever
        /// <see cref="Initialize"/> registered, or the old assembly stays referenced.
        /// </summary>
        protected internal virtual void Destroy() { }
    }

    /// <summary>
    /// A system that updates a sorted list of systems. Groups are plain containers: they always run and
    /// carry no editor flag of their own, the check happens per member.
    /// </summary>
    public abstract class SystemGroup : EntitySystem
    {
        private readonly List<EntitySystem> _systems = [];
        private bool _sortDirty;

        public IReadOnlyList<EntitySystem> Systems
        {
            get
            {
                if (_sortDirty) Sort();
                return _systems;
            }
        }

        /// <summary>
        /// A member every other member runs after unless its attributes place it (directly or through
        /// other systems) before that member. Null: unconstrained members sort by type name only.
        /// </summary>
        protected virtual Type? DefaultAfter => null;

        internal void Add(EntitySystem system)
        {
            system.Group = this;
            _systems.Add(system);
            _sortDirty = true;
        }

        internal void Remove(EntitySystem system)
        {
            if (!_systems.Remove(system)) return;
            system.Group = null;
            _sortDirty = true;
        }

        public override void Update()
        {
            if (_sortDirty) Sort();

            bool editor = Engine.IsEditor;
            for (int i = 0; i < _systems.Count; i++)
            {
                var system = _systems[i];
                if (!system.Enabled || (editor && !system.RunsInEditor))
                {
                    system.ElapsedMs = 0;
                    continue;
                }

                long start = Stopwatch.GetTimestamp();
                system.Update();
                system.ElapsedMs = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
            }
        }

        /// <summary>
        /// Topological sort by the members' UpdateBefore/UpdateAfter attributes. Ties are broken by type
        /// name so the order never depends on discovery order. Runs only when the member list changed.
        /// </summary>
        private void Sort()
        {
            _sortDirty = false;
            int count = _systems.Count;
            if (count < 2) return;

            _systems.Sort((a, b) => string.CompareOrdinal(a.GetType().FullName, b.GetType().FullName));

            var index = new Dictionary<Type, int>(count);
            for (int i = 0; i < count; i++)
                index[_systems[i].GetType()] = i;

            var successors = new List<int>?[count];
            var predecessors = new List<int>?[count];
            var pending = new int[count];

            void AddEdge(int first, int second)
            {
                (successors[first] ??= []).Add(second);
                (predecessors[second] ??= []).Add(first);
                pending[second]++;
            }

            int Resolve(Type owner, Type other, string attribute)
            {
                if (other != null && other != owner && index.TryGetValue(other, out int result))
                    return result;
                Debug.LogWarning("Systems", $"{owner.Name}: [{attribute}({other?.Name})] ignored, it is not a member of {GetType().Name}");
                return -1;
            }

            for (int i = 0; i < count; i++)
            {
                var type = _systems[i].GetType();

                foreach (var before in type.GetCustomAttributes<UpdateBeforeAttribute>(false))
                {
                    int other = Resolve(type, before.SystemType, "UpdateBefore");
                    if (other >= 0) AddEdge(i, other);
                }

                foreach (var after in type.GetCustomAttributes<UpdateAfterAttribute>(false))
                {
                    int other = Resolve(type, after.SystemType, "UpdateAfter");
                    if (other >= 0) AddEdge(other, i);
                }
            }

            // Everything not placed before the anchor runs after it
            if (DefaultAfter != null && index.TryGetValue(DefaultAfter, out int anchor))
            {
                var beforeAnchor = new bool[count];
                var stack = new Stack<int>();
                stack.Push(anchor);
                while (stack.Count > 0)
                {
                    var list = predecessors[stack.Pop()];
                    if (list == null) continue;
                    foreach (int p in list)
                    {
                        if (beforeAnchor[p]) continue;
                        beforeAnchor[p] = true;
                        stack.Push(p);
                    }
                }

                for (int i = 0; i < count; i++)
                {
                    if (i != anchor && !beforeAnchor[i])
                        AddEdge(anchor, i);
                }
            }

            var sorted = new List<EntitySystem>(count);
            var done = new bool[count];

            while (sorted.Count < count)
            {
                // Lowest index = first by type name among the systems whose predecessors all ran
                int next = -1;
                for (int i = 0; i < count; i++)
                {
                    if (!done[i] && pending[i] == 0) { next = i; break; }
                }

                if (next < 0)
                {
                    var names = new List<string>();
                    for (int i = 0; i < count; i++)
                    {
                        if (done[i]) continue;
                        names.Add(_systems[i].GetType().Name);
                        sorted.Add(_systems[i]);
                    }
                    Debug.LogError("Systems", $"{GetType().Name}: UpdateBefore/UpdateAfter form a cycle between {string.Join(", ", names)}; these run in name order");
                    break;
                }

                done[next] = true;
                sorted.Add(_systems[next]);

                var list = successors[next];
                if (list == null) continue;
                foreach (int s in list)
                    pending[s]--;
            }

            _systems.Clear();
            _systems.AddRange(sorted);
        }
    }

    /// <summary>
    /// Root group, updated once per frame from the engine tick after components woke up. The default
    /// group. Members that state no relation to <see cref="ScriptGroup"/> run after it.
    /// </summary>
    public sealed class UpdateGroup : SystemGroup
    {
        protected override Type? DefaultAfter => typeof(ScriptGroup);
    }

    /// <summary>
    /// Root group, updated inside <see cref="Graphics.DeferredRenderer"/>'s render (once per rendered
    /// view) at the point where draws are enqueued; DeferredRenderer.Current is valid.
    /// </summary>
    public sealed class RenderGroup : SystemGroup { }

    /// <summary>
    /// The anchor for "before scripts" / "after scripts" inside <see cref="UpdateGroup"/>. Holds the
    /// system that dispatches <see cref="IUpdate"/>.
    /// </summary>
    public sealed class ScriptGroup : SystemGroup { }

    /// <summary>Calls Update() on every <see cref="IUpdate"/> component.</summary>
    [UpdateInEditor]
    [UpdateInGroup(typeof(ScriptGroup))]
    public sealed class ScriptUpdateSystem : EntitySystem
    {
        public override void Update() => ScriptExecution.UpdateComponents();
    }

    /// <summary>Calls Draw() on every <see cref="IDraw"/> component.</summary>
    [UpdateInEditor]
    [UpdateInGroup(typeof(RenderGroup))]
    public sealed class ScriptDrawSystem : EntitySystem
    {
        public override void Update() => ScriptExecution.Draw();
    }
}
