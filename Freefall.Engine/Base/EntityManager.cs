using System;
using System.Collections.Generic;
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
        /// Remove all entities except those flagged DontDestroyOnLoad.
        /// Call before loading a new scene.
        /// </summary>
        public static void ClearScene()
        {
            lock (_lock)
            {
                for (int i = _entities.Count - 1; i >= 0; i--)
                {
                    if (!_entities[i].DontDestroy)
                        _entities[i].Destroy();
                }
            }
        }

        public static void Update()
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            ScriptExecution.Update();
            sw.Stop();
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
