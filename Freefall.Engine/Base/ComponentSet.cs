using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;

namespace Freefall.Base
{
    public interface IInstanceId
    {
        int GetInstanceId();
    }

    public class ComponentSet<T> where T : IInstanceId
    {
        private readonly List<T> list = [];
        private readonly Dictionary<int, int> indices = [];

        public int Count => list.Count;

        public ReadOnlyCollection<T> Collection => list.AsReadOnly();

        public IEnumerator<T> GetEnumerator() => list.GetEnumerator();

        public T this[int index] => list.Count > index ? list[index] : default;

        public bool TryGetValue(int itemId, out T result)
        {
            if (indices.TryGetValue(itemId, out int index))
            {
                result = list[index];
                return true;
            }

            result = default;
            return false;
        }

        public bool Add(T item)
        {
            if (item == null) return false;

            var index = item.GetInstanceId();
            if (indices.TryGetValue(index, out _))
                return false;

            indices.Add(index, list.Count);
            list.Add(item);
            return true;
        }

        public void Add(IEnumerable<T> collection)
        {
            foreach (var ent in collection)
                Add(ent);
        }

        public bool Remove(T item)
        {
            if (item == null) return false;

            var index = item.GetInstanceId();

            if (indices.TryGetValue(index, out int id))
            {
                FastRemove(id);
                return true;
            }

            return false;
        }

        public void Clear()
        {
            if (Count < 1) return;

            list.Clear();
            indices.Clear();
        }

        public bool Contains(T item)
        {
            if (item == null) return false;

            return indices.ContainsKey(item.GetInstanceId());
        }

        void FastRemove(int index)
        {
            int last = list.Count - 1;
            if (index > last) return;

            var a = list[index];
            var b = list[last];

            indices[b.GetInstanceId()] = index;
            indices.Remove(a.GetInstanceId());

            list[index] = b;
            list[last] = a;

            list.RemoveAt(last);
        }
    }
}