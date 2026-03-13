using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;

namespace Freefall.Base
{
    public interface IIndex
    {
        int Id { get; }
    }

    /// <summary>
    /// QuickSet provides fast insertion, removal and uniqueness
    /// </summary>
    public class QuickSet<T> where T : class, IIndex
    {
        private readonly List<T> list = [];
        private readonly Dictionary<int, int> indices = [];

        public int Count => list.Count;

        public ReadOnlyCollection<T> Collection => list.AsReadOnly();

        public IEnumerator<T> GetEnumerator() => list.GetEnumerator();

        public T this[int index] => list.Count > index ? list[index] : null;

        public bool TryGetValue(int itemId, out T result)
        {
            if (indices.TryGetValue(itemId, out int index))
            {
                result = list[index];
                return true;
            }

            result = null;
            return false;
        }

        public bool Add(T item)
        {
            if (item == null) return false;

            var index = item.Id;
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

            var index = item.Id;

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

            return indices.ContainsKey(item.Id);
        }

        void FastRemove(int index)
        {
            int last = list.Count - 1;
            if (index > last) return;

            var a = list[index];
            var b = list[last];

            indices[b.Id] = index;
            indices.Remove(a.Id);

            list[index] = b;
            list[last] = a;

            list.RemoveAt(last);
        }
    }

    public class EntitySet : QuickSet<Entity> { }

}
