using System;
using System.Collections.Generic;
using Freefall.Assets;
using Freefall.Graphics;

namespace Freefall.Base
{
    public static class EntityManager
    {
        private static readonly EntitySet _entities = new();

        private static readonly Lock _lock = new();

        public static IReadOnlyList<Entity> Entities => _entities.Collection;

        public static void AddEntity(Entity entity)
        {
            lock (_lock)
            {
                _entities.Add(entity);
            }
        }

        public static void RemoveEntity(Entity entity)
        {
            lock (_lock)
            {
                _entities.Remove(entity);
            }
        }

        /// <summary>
        /// O(1) entity lookup by Id.
        /// </summary>
        public static Entity GetEntity(int id)
        {
            lock (_lock)
            {
                _entities.TryGetValue(id, out var entity);
                return entity;
            }
        }

        /// <summary>
        /// Remove all entities except those flagged DontDestroyOnLoad.
        /// Call before loading a new scene.
        /// </summary>
        public static void ClearScene() => new ClearJob().Step(long.MaxValue);

        /// <summary>
        /// <see cref="ClearScene"/> in slices (a scene load clears ~77,000 entities in ~0.4 s): the same entities,
        /// destroyed in the same order, as many as fit before a deadline. Call <see cref="Step"/> until it returns true.
        /// </summary>
        internal sealed class ClearJob
        {
            private readonly List<Entity> _list;
            private int _next;

            public ClearJob()
            {
                lock (_lock)
                {
                    CommandBuffer.Clear();
                    _list = new List<Entity>(_entities.Collection);
                }
                _next = _list.Count - 1;
            }

            /// <summary>0..1</summary>
            public float Progress => _list.Count == 0 ? 1f : 1f - (float)(_next + 1) / _list.Count;

            public int Total => _list.Count;

            /// <summary>Destroy until done or <paramref name="deadline"/> (a Stopwatch timestamp) has passed. True when done.</summary>
            public bool Step(long deadline)
            {
                lock (_lock)
                {
                    while (_next >= 0)
                    {
                        var entity = _list[_next--];
                        if (entity != null && !entity.DontDestroy)
                            entity.Destroy();
                        if (deadline != long.MaxValue && System.Diagnostics.Stopwatch.GetTimestamp() >= deadline)
                            break;
                    }
                }
                if (_next >= 0) return false;

                Graphics.AutoExposure.Snap();     // the next scene starts at its own exposure, not a fade from this one
                Graphics.SkyCapture.Invalidate(); // and its sky captures are drawn anew
                return true;
            }
        }

        public static void Update()
        {
            ScriptExecution.WakeUp();
            Systems.RunUpdate();
        }

        public static T? FindComponent<T>() where T : Component
        {
            lock (_lock)
            {
                foreach (var entity in _entities)
                {
                    var component = entity.GetComponent<T>();
                    if (component != null)
                        return component;
                }
            }
            return null;
        }

        public static IEnumerable<T> FindComponents<T>() where T : Component
        {
            // Return snapshot-free results — callers should not add/remove entities during enumeration
            lock (_lock)
            {
                var results = new List<T>();
                foreach (var entity in _entities)
                {
                    var component = entity.GetComponent<T>();
                    if (component != null)
                        results.Add(component);
                }
                return results;
            }
        }
    }
}
