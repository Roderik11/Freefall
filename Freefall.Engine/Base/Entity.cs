using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Freefall.Components;

namespace Freefall.Base
{
    public class Entity
    {
        private readonly List<Component> _components = new List<Component>();
        internal IReadOnlyList<Component> Components => _components;

        public string Name { get; set; } = "Entity";
        public Transform Transform { get; private set; }

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
            var componentType = component.GetType();
            var cacheType = typeof(ComponentCache<>).MakeGenericType(componentType);
            var addMethod = cacheType.GetMethod("Add", System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static);
            addMethod?.Invoke(null, [this, component]);

            return component;
        }
    }
}
