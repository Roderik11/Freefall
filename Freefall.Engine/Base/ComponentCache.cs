using System;
using System.Collections.Generic;
using System.Text;
using Freefall.Components;

namespace Freefall.Base
{
    public interface IComponentCache
    {
        void Update();
        void Draw();
    }

    public class ComponentCache<T> : IComponentCache where T : Component
    {
        public static readonly Type Type = typeof(T);
        internal static readonly List<T> List = [];
        internal static readonly ComponentSet<IUpdate> UpdateList = new();
        internal static readonly ComponentSet<IDraw> DrawList = new();

        private static readonly Dictionary<int, int> Indices = [];
        private static readonly Lock _lock = new ();

        private bool IsParallel;
        private bool HasUpdate;
        private bool HasDraw;

        static ComponentCache()
        {
            bool hasUpdate = typeof(IUpdate).IsAssignableFrom(Type);
            bool hasDraw = typeof(IDraw).IsAssignableFrom(Type);
            bool isParallel = typeof(IParallel).IsAssignableFrom(Type);

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

        public static T? Get(Entity entity)
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
            if (!HasUpdate) return;

            if (HasUpdate)
            {
                if (IsParallel)
                    Parallel.ForEach(UpdateList.Collection, comp => comp.Update());
                else
                    foreach (var comp in UpdateList.Collection) comp.Update();
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

}
