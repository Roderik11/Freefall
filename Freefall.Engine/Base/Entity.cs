using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Freefall.Components;

namespace Freefall.Base
{
    public static class ScriptExecution
    {
        internal static List<IComponentCache> list = new List<IComponentCache>();

        internal static void Add(IComponentCache cache)
        {
            list.Add(cache);
        }

        public static void Update()
        {
            for (int i = 0; i < list.Count; i++)
            {
                list[i].Update();
            }
        }

        public static void Draw()
        {
            for (int i = 0; i < list.Count; i++)
            {
                list[i].Draw();
            }
        }
    }

    public interface IComponentCache
    {
        void Update();
        void Draw();
    }

    public class ComponentCache<T> : IComponentCache where T : Component
    {
        public static readonly Type Type = typeof(T);
        internal static readonly List<T> List = new List<T>();
        internal static readonly ComponentSet<IUpdate> UpdateList = new ComponentSet<IUpdate>();
        internal static readonly ComponentSet<IDraw> DrawList = new ComponentSet<IDraw>();

        private static readonly Dictionary<int, int> Indices = new Dictionary<int, int>();
        private static readonly object _lock = new object();

        private bool IsParallel;
        private bool HasUpdate;
        private bool HasDraw;

        static ComponentCache()
        {
            bool hasUpdate = typeof(IUpdate).IsAssignableFrom(Type);
            bool hasDraw = typeof(IDraw).IsAssignableFrom(Type);
            bool isParallel = typeof(IParallel).IsAssignableFrom(Type);

            // Only register once, even if component implements both IUpdate and IDraw
            if (hasUpdate || hasDraw)
            {
                ScriptExecution.Add(new ComponentCache<T>
                {
                    IsParallel = isParallel,
                    HasUpdate = hasUpdate,
                    HasDraw = hasDraw
                });
            }
        }

        static void FastRemove(int index)
        {
            int last = List.Count - 1;
            if (index > last) return;

            var a = List[index];
            var b = List[last];

            Indices[b.Entity.GetHashCode()] = index;
            Indices.Remove(a.Entity.GetHashCode());

            List[index] = b;
            List[last] = a;

            List.RemoveAt(last);

            if (a is IUpdate update) UpdateList.Remove(update);
            if (a is IDraw draw) DrawList.Remove(draw);
        }

        public static T Get(Entity entity)
        {
            lock (_lock)
            {
                if (Indices.TryGetValue(entity.GetHashCode(), out int result))
                    return List[result];

                return null;
            }
        }

        public static bool Remove(Entity entity)
        {
            lock (_lock)
            {
                if (!Indices.TryGetValue(entity.GetHashCode(), out int result))
                    return false;

                FastRemove(result);
                return true;
            }
        }

        public static T Add(Entity entity, T component)
        {
            lock (_lock)
            {
                int hash = entity.GetHashCode();
                if (Indices.TryGetValue(hash, out int result))
                    return List[result];

                if (component is IUpdate update)
                    UpdateList.Add(update);

                if (component is IDraw draw)
                    DrawList.Add(draw);

                Indices.Add(hash, List.Count);
                component.Entity = entity;
                List.Add(component);

                return component;
            }
        }

        public void Update()
        {
            if(!HasUpdate) return;
            
            // Take snapshot for thread safety
            IUpdate[] snapshot;
            lock (_lock) { snapshot = UpdateList.Collection.ToArray(); }
            
            if (HasUpdate)
            {
                if (IsParallel)
                    Parallel.ForEach(snapshot, comp => comp.Update());
                else
                    foreach (var comp in snapshot) comp.Update();
            }
        }

        public void Draw()
        {
            if (!HasDraw) return;
            
            if (IsParallel)
                Parallel.ForEach(DrawList.Collection, comp => comp.Draw());
            else
                foreach (var comp in DrawList.Collection) comp.Draw();
        }

        public override string ToString()
        {
            return typeof(T).Name;
        }
    }

    public class Entity
    {
        private readonly List<Component> _components = new List<Component>();
        internal IReadOnlyList<Component> Components => _components;

        public string Name { get; set; } = "Entity";
        public Transform Transform { get; private set; }

        public Entity(string name = "Entity")
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

        public void Update()
        {
            // Deprecated - Update logic moved to ScriptExecution
        }
    }
}
