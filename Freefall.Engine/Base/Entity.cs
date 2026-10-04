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
        private static readonly Dictionary<Type, Type> _cacheTypes = new();
        private readonly List<Component> _components = new List<Component>();
        public IReadOnlyList<Component> Components => _components;

        private static Type GetCacheType(Type componentType)
        {
            if (!_cacheTypes.TryGetValue(componentType, out var cacheType))
            {
                cacheType = typeof(ComponentCache<>).MakeGenericType(componentType);
                _cacheTypes[componentType] = cacheType;
            }
            return cacheType;
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
            _components.Add(component);
            component.Entity = this;
            
            ComponentCache<T>.Add(this, component);
            
            if (component is Transform t)
            {
                Transform = t;
            }

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

            // Invoke ComponentCache<T>.Add(this, component) via reflection
            var cacheType = GetCacheType(component.GetType());
            var addMethod = cacheType.GetMethod("Add", BindingFlags.Public | BindingFlags.Static);
            addMethod?.Invoke(null, [this, component]);

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
            if (component is Transform) return; // never remove Transform
            if (!_components.Remove(component)) return;

            try
            {
                component.Destroy();
            }
            finally
            {
                // A throwing Destroy() must not leave the component registered in its cache
                var cacheType = GetCacheType(component.GetType());
                var removeMethod = cacheType.GetMethod("Remove", BindingFlags.Public | BindingFlags.Static, [typeof(Entity)]);
                removeMethod?.Invoke(null, [this]);
            }
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

            var cacheType = GetCacheType(component.GetType());
            var removeMethod = cacheType.GetMethod("Remove", BindingFlags.Public | BindingFlags.Static, [typeof(Entity)]);
            removeMethod?.Invoke(null, [this]);
            component.Entity = null;
        }

        /// <summary>
        /// Forget the ComponentCache&lt;T&gt; types built for component types matching the predicate
        /// (script types whose assembly is being unloaded), so this static map doesn't pin their ALC.
        /// </summary>
        internal static void ForgetCacheTypes(Func<Type, bool> predicate)
        {
            foreach (var type in _cacheTypes.Keys.Where(predicate).ToList())
                _cacheTypes.Remove(type);
        }

        public void RemoveComponent<T>() where T : Component
        {
            var component = GetComponent<T>();
            if (component != null)
                RemoveComponent(component);
        }

        /// <summary>
        /// Destroy this entity: call Destroy() on all components,
        /// unregister from ComponentCaches, remove from EntityManager.
        /// </summary>
        public void Destroy()
        {
            var childCount = Transform.Count;

            for (int i = childCount - 1; i >= 0; i--)
            {
                var child = Transform.GetChild(i);
                child?.Entity?.Destroy();
            }

            foreach (var component in _components)
                component.Destroy();

            // Unregister each component from its ComponentCache<T>
            foreach (var component in _components)
            {
                var cacheType = GetCacheType(component.GetType());
                var removeMethod = cacheType.GetMethod("Remove", BindingFlags.Public | BindingFlags.Static, [typeof(Entity)]);
                removeMethod?.Invoke(null, [this]);
            }

            _components.Clear();
            Transform.Parent = null;

            EntityManager.RemoveEntity(this);
        }
    }
}
