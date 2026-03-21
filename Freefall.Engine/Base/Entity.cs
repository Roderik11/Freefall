using System;
using System.Collections.Generic;
using System.Reflection;
using System.Threading.Tasks;
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
        /// Used by SceneSerializer to emit compact PrefabInstance documents.
        /// </summary>
        public Assets.Prefab Prefab { get; set; }

        public bool IsPrefabInstance => Prefab != null;

        [Reflection.DontSerialize]
        public bool Hidden { get; set; }
        [Reflection.DontSerialize]
        public bool Expanded { get; set; }
        [Reflection.DontSerialize]
        public EntityFlags Flags { get; set; }

        public bool DontDestroy => (Flags & EntityFlags.DontDestroy) != 0;
        public bool HideAndDontSave => (Flags & EntityFlags.HideAndDontSave) != 0;

        public Entity() : this("Entity") { }

        public Entity(string name)
        {
            Name = name;
            Transform = AddComponent<Transform>();
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

        /// <summary>
        /// Non-generic AddComponent for runtime deserialization.
        /// Uses reflection to call ComponentCache&lt;T&gt;.Add with the actual component type.
        /// </summary>
        public Component AddComponent(Component component)
        {
            _components.Add(component);
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
        /// Destroy this entity: call Destroy() on all components,
        /// unregister from ComponentCaches, remove from EntityManager.
        /// </summary>
        public void Destroy()
        {
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

            EntityManager.RemoveEntity(this);
        }
    }
}
