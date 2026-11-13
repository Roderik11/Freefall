using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Threading.Tasks;
using System.Xml.Serialization;
using Freefall.Components;

namespace Freefall.Base
{
    public class Entity : IUniqueId, IIndex
    {
        private readonly List<Component> _components = new List<Component>();
        public IReadOnlyList<Component> Components => _components;

        // Adding a component of a type known only at run time to its ComponentCache<T> and taking it out: delegates made
        // once per component type, in place of GetMethod + Invoke with an argument array per call (558 ns per component
        // of every prefab copy, and as much again when it was destroyed: ram.md "What a spawn takes"). The map is made
        // anew when a script assembly comes or goes (Reflector.Generation), so it never keeps an unloaded type alive.
        private sealed record CacheOps(Action<Entity, Component> Add, Func<Entity, bool> Remove);
        private static Dictionary<Type, CacheOps> _cacheOps = new();
        private static int _cacheOpsGeneration = -1;

        private static class CacheOf<T> where T : Component
        {
            public static void Add(Entity entity, Component component) => ComponentCache<T>.Add(entity, (T)component);
            public static bool Remove(Entity entity) => ComponentCache<T>.Remove(entity);
        }

        private static CacheOps OpsFor(Type componentType)
        {
            int generation = Reflection.Reflector.Generation;
            if (_cacheOpsGeneration != generation)
            {
                _cacheOps = new Dictionary<Type, CacheOps>();
                _cacheOpsGeneration = generation;
            }
            if (!_cacheOps.TryGetValue(componentType, out var ops))
            {
                var of = typeof(CacheOf<>).MakeGenericType(componentType);
                ops = new CacheOps(
                    of.GetMethod(nameof(CacheOf<Component>.Add))!.CreateDelegate<Action<Entity, Component>>(),
                    of.GetMethod(nameof(CacheOf<Component>.Remove))!.CreateDelegate<Func<Entity, bool>>());
                _cacheOps[componentType] = ops;
            }
            return ops;
        }

        public int Id { get; } = IDGenerator.GetId();

        public ulong UID { get; set; } = IDGenerator.GetUID();
        public string Name { get; set; } = "Entity";
        public Transform Transform { get; private set; }

        /// <summary>
        /// Source prefab this entity was instantiated from. 
        /// null if the entity was created directly (not from a prefab).
        /// Used by EntitySerializer to emit compact PrefabInstance documents.
        /// </summary>
        public Assets.Prefab Prefab { get; set; }

        public bool IsPrefabInstance => Prefab != null;

        /// <summary>
        /// True if any ancestor of this entity is a prefab instance root.
        /// Used to skip prefab children during scene saves — they're reconstructed on load.
        /// </summary>
        public bool IsChildOfPrefabInstance
        {
            get
            {
                var parent = Transform.Parent;
                while (parent != null)
                {
                    if (parent.Entity.IsPrefabInstance)
                        return true;
                    parent = parent.Parent;
                }
                return false;
            }
        }
       
        [Reflection.DontSerialize]
        public bool Expanded { get; set; }

        [Reflection.DontSerialize]
        public EntityFlags Flags { get; set; }

        public bool DontDestroy => (Flags & EntityFlags.DontDestroy) != 0;

        public bool DontSave
        {
            get
            {
                if ((Flags & EntityFlags.DontSave) != 0)
                    return true;

                return Transform.Parent?.Entity?.DontSave ?? false;
            }
        }
            
        [Reflection.DontSerialize]
        public bool HideInHierarchy
        {
            get => (Flags & EntityFlags.HideInHierarchy) != 0;
            set
            {
                if (value)
                    Flags |= EntityFlags.HideInHierarchy;
                else
                    Flags &= ~EntityFlags.HideInHierarchy;
            }
        }

        public Entity() : this("Entity") { }

        public Entity(string name) : this(name, register: true) { }

        /// <summary>
        /// Internal constructor. When register is false, the entity is NOT added
        /// to EntityManager — used by importers to build temp entities for
        /// serialization without polluting the live scene.
        /// </summary>
        public Entity(string name, bool register)
        {
            Name = name;
            Transform = AddComponent<Transform>();
            if (register)
                EntityManager.AddEntity(this);
        }

        public T AddComponent<T>() where T : Component, new()
        {
            var component = new T();
            return AddComponent(component);
        }

        public T AddComponent<T>(T component) where T : Component
        {
            ParallelWork.AssertSerial("Entity.AddComponent");   // the component caches other threads iterate
            _components.Add(component);
            component.Entity = this;
            
            ComponentCache<T>.Add(this, component);
            
            if (component is Transform t)
            {
                Transform = t;
            }

            component.Attached();

            return component;
        }

        public T? GetComponent<T>() where T : Component
        {
            return ComponentCache<T>.Get(this);
        }

        public T? GetComponentInChildren<T>() where T : Component
        {
            var component = GetComponent<T>();
            if (component != null)
                return component;
            foreach (Transform child in Transform)
            {
                var childComponent = child.Entity?.GetComponentInChildren<T>();
                if (childComponent != null)
                    return childComponent;
            }
            return null;
        }

        public T? GetComponentInParents<T>() where T : Component
        {
            var current = this;
            while (current != null)
            {
                var component = current.GetComponent<T>();
                if (component != null)
                    return component;
                current = current.Transform.Parent?.Entity;
            }
            return null;
        }

        public List<T> GetComponents<T>() where T : Component
        {
            // foreach subclass of T, get components of that type and add to list
            var list = new List<T>();
            foreach(var component in  _components)
            {
                if (component is T t)
                    list.Add(t);
            }
            return list;
        }

        public List<T> GetComponentsInChildren<T>() where T : Component
        {
            // foreach subclass of T, get components of that type and add to list
            var list = GetComponents<T>();
            foreach (Transform child in Transform)
                child.Entity.GetComponentsInChildren(list);
            return list;
        }

        private void GetComponentsInChildren<T>(List<T> list) where T : Component
        {
            foreach (var component in _components)
            {
                if (component is T t)
                    list.Add(t);
            }

            foreach (Transform child in Transform)
                child.Entity.GetComponentsInChildren(list);
        }

        /// <summary>
        /// Non-generic AddComponent for runtime deserialization.
        /// Uses reflection to call ComponentCache&lt;T&gt;.Add with the actual component type.
        /// </summary>
        public Component AddComponent(Component component) => InsertComponent(_components.Count, component);

        /// <summary>
        /// Non-generic add at a given position in <see cref="Components"/> (clamped to the list).
        /// Used by script hot-reload to put a migrated component back where the old one was.
        /// </summary>
        public Component InsertComponent(int index, Component component)
        {
            _components.Insert(Math.Clamp(index, 0, _components.Count), component);
            component.Entity = this;

            if (component is Transform t)
            {
                Transform = t;
            }

            OpsFor(component.GetType()).Add(this, component);

            component.Attached();

            return component;
        }

        public Component AddComponent(Type type)
        {
            MethodInfo info1 = typeof(Entity).GetMethod("AddComponent", new Type[] { });
            MethodInfo info2 = info1.MakeGenericMethod(type);
            return info2.Invoke(this, null) as Component;
        }

        /// <summary>
        /// Remove a specific component from this entity.
        /// Calls Destroy() on the component and unregisters from ComponentCache.
        /// Cannot remove Transform.
        /// </summary>
        public void RemoveComponent(Component component)
        {
            ParallelWork.AssertSerial("Entity.RemoveComponent");
            if (component is Transform) return; // never remove Transform
            if (!_components.Remove(component)) return;

            // A throwing Destroy() is reported and the component still leaves its cache
            component.IsDestroyed = true;
            component.Destroyed();

            OpsFor(component.GetType()).Remove(this);
        }

        /// <summary>
        /// Take a component off this entity WITHOUT destroying it, so it can be added to another entity
        /// (prefab hydration moves components off a temporary root). Unregisters it from ComponentCache
        /// under this entity; adding it elsewhere registers it again.
        /// </summary>
        internal void DetachComponent(Component component)
        {
            if (component is Transform) return;
            if (!_components.Remove(component)) return;

            OpsFor(component.GetType()).Remove(this);
            component.Entity = null;
        }

        /// <summary>
        /// Forget the ComponentCache&lt;T&gt; delegates made for component types matching the predicate
        /// (script types whose assembly is being unloaded), so this static map doesn't pin their ALC.
        /// </summary>
        internal static void ForgetCacheTypes(Func<Type, bool> predicate)
        {
            foreach (var type in _cacheOps.Keys.Where(predicate).ToList())
                _cacheOps.Remove(type);
        }

        public void RemoveComponent<T>() where T : Component
        {
            var component = GetComponent<T>();
            if (component != null)
                RemoveComponent(component);
        }

        /// <summary>
        /// Set when <see cref="Destroy"/> starts, before its children go: a component on a child can tell that what it
        /// belongs to goes with it (a collider whose shape is on a parent's RigidBody leaves the shape to the body's actor).
        /// </summary>
        [System.ComponentModel.Browsable(false)]
        [Reflection.DontSerialize]
        public bool IsBeingDestroyed { get; private set; }

        /// <summary>
        /// Destroy this entity: call Destroy() on all components,
        /// unregister from ComponentCaches, remove from EntityManager.
        /// </summary>
        public void Destroy()
        {
            ParallelWork.AssertSerial("Entity.Destroy");
            IsBeingDestroyed = true;
            var childCount = Transform.Count;

            for (int i = childCount - 1; i >= 0; i--)
            {
                var child = Transform.GetChild(i);
                child?.Entity?.Destroy();
            }

            foreach (var component in _components)
            {
                component.IsDestroyed = true;
                component.Destroyed();
            }

            // Unregister each component from its ComponentCache<T>
            foreach (var component in _components)
                OpsFor(component.GetType()).Remove(this);

            _components.Clear();
            Transform.Parent = null;

            EntityManager.RemoveEntity(this);
        }
    }
}
